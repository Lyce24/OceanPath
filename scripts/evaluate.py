"""
Stage 6: Comprehensive model evaluation.

Reads Stage 5 training outputs and produces:

  Part 1 — CV analysis (from existing predictions):
    - OOF metrics (slide + patient level, comprehensive)
    - Per-fold test metrics
    - Aggregated CV summary with all metrics

  Part 2 — Final model evaluation (best_fold, ensemble, refit):
    - Full performance with bootstrap CIs (slide + patient level)
    - Calibration analysis (ECE, Brier, reliability diagram)
    - Operating points (Youden's J, high-sensitivity, high-specificity)
    - Threshold stability (±0.01/±0.05 perturbation impact)
    - All plots (ROC, PR, calibration, confusion, stability)

  Part 3 — Model comparison and selection:
    - Head-to-head comparison on composite score
    - Recommended model with rationale

Config design:
    evaluate.yaml composes the SAME config groups as train.yaml
    (platform, data, encoder, splits, model, training) so all paths
    resolve identically. No need to load train_dir/config.yaml.

Usage:
    # Evaluate the training run matching these config groups
    python scripts/evaluate.py platform=local data=gej encoder=univ1 \\
           splits=kfold5 model=abmil training=gej

    # Override bootstrap settings
    python scripts/evaluate.py ... eval.n_bootstrap=5000

    # Skip inference (use existing predictions only)
    python scripts/evaluate.py ... eval.skip_inference=true

    # Point to a specific run (override derived train_dir)
    python scripts/evaluate.py ... train_dir=/explicit/path

    # Custom class names for confusion matrix labels
    python scripts/evaluate.py ... eval.class_names=[benign,low_grade,high_grade]
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Part 1: CV Analysis (from existing predictions)
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_oof(
    train_dir: Path,
    eval_dir: Path,
    patient_column: str = "patient_id",
    csv_path: Optional[str] = None,
    filename_column: str = "filename",
) -> dict:
    """
    Evaluate OOF (out-of-fold) predictions from cross-validation.

    Reads oof_predictions.parquet (concatenated val preds across all folds)
    and computes comprehensive metrics at slide and patient level.
    """
    from oceanpath.eval.core import (
        compute_metrics,
        aggregate_to_patient_level,
        extract_probs_and_labels,
    )

    oof_path = train_dir / "oof_predictions.parquet"
    if not oof_path.is_file():
        logger.warning(f"No OOF predictions found at {oof_path}")
        return {"error": "oof_predictions.parquet not found"}

    oof_df = pd.read_parquet(oof_path)
    logger.info(f"OOF predictions: {len(oof_df)} slides")

    # Attach patient_id if CSV is available
    oof_df = _attach_patient_id(oof_df, csv_path, filename_column, patient_column)

    # ── Slide-level metrics ──────────────────────────────────────────────
    labels, probs = extract_probs_and_labels(oof_df)
    slide_metrics = compute_metrics(labels, probs, level="slide")

    # ── Patient-level metrics ────────────────────────────────────────────
    patient_df = aggregate_to_patient_level(oof_df, patient_column=patient_column)
    p_labels, p_probs = extract_probs_and_labels(patient_df)
    patient_metrics = compute_metrics(p_labels, p_probs, level="patient")

    result = {
        "n_slides": len(oof_df),
        "n_patients": len(patient_df),
        "n_folds": int(oof_df["fold"].nunique()) if "fold" in oof_df.columns else None,
        "slide_level": slide_metrics.to_dict(),
        "patient_level": patient_metrics.to_dict(),
    }

    # Save
    out_path = eval_dir / "oof_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=_json_default))
    logger.info(f"OOF evaluation → {out_path}")

    _print_metrics_report("OOF (slide)", slide_metrics)
    _print_metrics_report("OOF (patient)", patient_metrics)

    return result


def evaluate_per_fold_test(
    train_dir: Path,
    eval_dir: Path,
    n_folds: int,
    patient_column: str = "patient_id",
    csv_path: Optional[str] = None,
    filename_column: str = "filename",
) -> dict:
    """
    Evaluate per-fold test predictions.

    Reads each fold's preds_test.parquet and computes metrics.
    Also computes aggregated test metrics (concatenated across folds).
    """
    from oceanpath.eval.core import (
        compute_metrics,
        aggregate_to_patient_level,
        extract_probs_and_labels,
    )

    fold_results = []
    all_test_dfs = []

    for fold_idx in range(n_folds):
        test_path = train_dir / f"fold_{fold_idx}" / "preds_test.parquet"
        if not test_path.is_file():
            fold_results.append({"fold": fold_idx, "error": "preds_test.parquet not found"})
            continue

        test_df = pd.read_parquet(test_path)
        test_df["fold"] = fold_idx
        test_df = _attach_patient_id(test_df, csv_path, filename_column, patient_column)

        labels, probs = extract_probs_and_labels(test_df)
        slide_metrics = compute_metrics(labels, probs, level="slide")

        patient_df = aggregate_to_patient_level(test_df, patient_column=patient_column)
        p_labels, p_probs = extract_probs_and_labels(patient_df)
        patient_metrics = compute_metrics(p_labels, p_probs, level="patient")

        fold_results.append({
            "fold": fold_idx,
            "n_slides": len(test_df),
            "n_patients": len(patient_df),
            "slide_level": slide_metrics.to_dict(),
            "patient_level": patient_metrics.to_dict(),
        })
        all_test_dfs.append(test_df)

    # Aggregated test (all folds concatenated)
    aggregated = None
    if all_test_dfs:
        concat_df = pd.concat(all_test_dfs, ignore_index=True)
        labels, probs = extract_probs_and_labels(concat_df)
        slide_agg = compute_metrics(labels, probs, level="slide")

        patient_df = aggregate_to_patient_level(concat_df, patient_column=patient_column)
        p_labels, p_probs = extract_probs_and_labels(patient_df)
        patient_agg = compute_metrics(p_labels, p_probs, level="patient")

        aggregated = {
            "n_slides": len(concat_df),
            "n_patients": len(patient_df),
            "slide_level": slide_agg.to_dict(),
            "patient_level": patient_agg.to_dict(),
        }

        _print_metrics_report("Test aggregated (slide)", slide_agg)
        _print_metrics_report("Test aggregated (patient)", patient_agg)

    # Mean ± std across folds for key metrics
    fold_summary = _summarize_fold_metrics(fold_results)

    result = {
        "per_fold": fold_results,
        "aggregated": aggregated,
        "fold_summary": fold_summary,
    }

    out_path = eval_dir / "per_fold_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=_json_default))
    logger.info(f"Per-fold test evaluation → {out_path}")

    return result


def _summarize_fold_metrics(fold_results: list) -> dict:
    """Compute mean ± std for key metrics across folds."""
    summary = {}
    metric_keys = [
        "auroc_macro", "balanced_accuracy", "accuracy",
        "f1_macro", "f1_weighted", "kappa", "precision_macro", "recall_macro",
    ]

    for level in ["slide_level", "patient_level"]:
        summary[level] = {}
        for key in metric_keys:
            values = []
            for fr in fold_results:
                if "error" in fr:
                    continue
                val = fr.get(level, {}).get(key)
                if val is not None and np.isfinite(val):
                    values.append(val)
            if values:
                summary[level][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_folds": len(values),
                }

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Part 2: Final Model Evaluation
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_final_model(
    model_name: str,
    preds_df: pd.DataFrame,
    eval_dir: Path,
    patient_column: str = "patient_id",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    class_names: Optional[list] = None,
    perturbations: Optional[tuple] = None,
) -> dict:
    """
    Full evaluation of a single final model.

    Produces:
      - performance.json (metrics + bootstrap CIs)
      - calibration.json
      - operating_points.json
      - threshold_stability.json
      - plots/*.png
    """
    from oceanpath.eval.core import (
        compute_metrics_with_ci,
        compute_calibration,
        compute_operating_points,
        compute_threshold_stability,
        compute_pr_curve,
        extract_probs_and_labels,
    )
    from oceanpath.eval.plots import (
        plot_roc_curve,
        plot_pr_curve,
        plot_calibration_curve,
        plot_confusion_matrix,
        plot_threshold_stability,
    )

    if perturbations is None:
        perturbations = (0.01, 0.02, 0.05, 0.10)

    model_dir = eval_dir / "final_models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if preds_df.empty:
        logger.warning(f"{model_name}: no predictions — skipping")
        return {"error": "no predictions"}

    logger.info(f"Evaluating {model_name}: {len(preds_df)} slides")

    # Save predictions
    preds_df.to_parquet(str(model_dir / "predictions.parquet"), index=False, engine="pyarrow")

    labels, probs = extract_probs_and_labels(preds_df)

    # ── 1. Performance with bootstrap CIs ────────────────────────────────
    performance = compute_metrics_with_ci(
        preds_df,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
        patient_column=patient_column,
    )
    _save_json(performance, model_dir / "performance.json")

    # Print report
    from oceanpath.eval.core import compute_metrics, aggregate_to_patient_level
    slide_metrics = compute_metrics(labels, probs, level="slide")
    pat_df = aggregate_to_patient_level(preds_df, patient_column=patient_column)
    p_labels, p_probs = extract_probs_and_labels(pat_df)
    patient_metrics = compute_metrics(p_labels, p_probs, level="patient")

    _print_metrics_report(f"{model_name} (slide)", slide_metrics)
    _print_metrics_report(f"{model_name} (patient)", patient_metrics)

    # ── 2. Calibration ───────────────────────────────────────────────────
    calibration = compute_calibration(labels, probs)
    _save_json(calibration, model_dir / "calibration.json")

    plot_calibration_curve(
        calibration, plots_dir / "calibration_curve.png",
        title=f"{model_name} — Calibration",
    )

    # ── 3. Operating points ──────────────────────────────────────────────
    operating_points = compute_operating_points(labels, probs)
    _save_json(operating_points, model_dir / "operating_points.json")

    # ROC curve
    auroc = slide_metrics.auroc_macro
    plot_roc_curve(
        operating_points, auroc if np.isfinite(auroc) else 0.0,
        plots_dir / "roc_curve.png",
        title=f"{model_name} — ROC",
    )

    # PR curve
    pr_data = compute_pr_curve(labels, probs)
    _save_json(pr_data, model_dir / "pr_curve.json")
    plot_pr_curve(
        pr_data, plots_dir / "pr_curve.png",
        title=f"{model_name} — Precision-Recall",
    )

    # Confusion matrices (slide + patient level)
    cm = slide_metrics.confusion_matrix
    if cm:
        plot_confusion_matrix(
            cm, plots_dir / "confusion_matrix.png",
            class_names=class_names,
            title=f"{model_name} — Confusion Matrix (slide-level)",
        )
    cm_patient = patient_metrics.confusion_matrix
    if cm_patient:
        plot_confusion_matrix(
            cm_patient, plots_dir / "confusion_matrix_patient.png",
            class_names=class_names,
            title=f"{model_name} — Confusion Matrix (patient-level)",
        )

    # ── 4. Threshold stability ───────────────────────────────────────────
    youden_threshold = _get_youden_threshold(operating_points)
    stability = compute_threshold_stability(
        labels, probs, youden_threshold,
        perturbations=tuple(perturbations),
    )
    _save_json(stability, model_dir / "threshold_stability.json")

    plot_threshold_stability(
        stability, plots_dir / "threshold_stability.png",
        title=f"{model_name} — Threshold Stability",
    )

    return {
        "performance": performance,
        "calibration": calibration,
        "operating_points": operating_points,
        "threshold_stability": stability,
    }


def _get_youden_threshold(operating_points: dict) -> float:
    """Extract Youden's J threshold from operating_points dict."""
    if "youdens_j" in operating_points:
        t = operating_points["youdens_j"].get("threshold")
        if t is not None:
            return t
    if "per_class" in operating_points:
        for c, data in operating_points["per_class"].items():
            yj = data.get("youdens_j", {})
            if yj.get("threshold") is not None:
                return yj["threshold"]
    return 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Part 2b: Inference + Evaluation for Final Models
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_final_models(
    train_dir: Path,
    eval_dir: Path,
    cfg: DictConfig,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    skip_inference: bool = False,
    patient_column: str = "patient_id",
    class_names: Optional[list] = None,
    perturbations: Optional[tuple] = None,
) -> dict:
    """
    Run inference and evaluate all final models (best_fold, ensemble, refit).

    If skip_inference=True, looks for existing predictions.parquet in each
    final model's eval directory.
    """
    from oceanpath.eval.inference import run_inference, get_test_slide_ids

    final_dir = train_dir / "final"
    if not final_dir.is_dir():
        logger.warning(f"No final/ directory found in {train_dir}")
        return {}

    # Load test slide IDs
    splits_dir = str(
        Path(cfg.platform.splits_root) / cfg.data.name / cfg.splits.name
    )
    test_ids = get_test_slide_ids(
        splits_dir, scheme=cfg.splits.scheme, fold=0,
    )
    logger.info(f"Test set: {len(test_ids)} slides")

    fallback_df = None
    if not test_ids:
        logger.warning("No test slides found — using OOF for final model evaluation")
        oof_path = train_dir / "oof_predictions.parquet"
        if oof_path.is_file():
            fallback_df = pd.read_parquet(oof_path)
            fallback_df = _attach_patient_id(
                fallback_df, cfg.data.csv_path,
                cfg.data.filename_column, patient_column,
            )
            test_ids = None
        else:
            return {"error": "No test set and no OOF predictions"}

    model_results = {}
    strategies = ["best_fold", "ensemble", "refit"]

    for strategy in strategies:
        model_dir = final_dir / strategy
        info_path = model_dir / "info.json"

        if not info_path.is_file():
            logger.info(f"{strategy}: no info.json found — skipping")
            continue

        info = json.loads(info_path.read_text())
        if "error" in info:
            logger.warning(f"{strategy}: {info['error']} — skipping")
            continue

        logger.info(f"{'=' * 50}")
        logger.info(f"  Evaluating: {strategy}")
        logger.info(f"{'=' * 50}")

        # Get predictions
        if skip_inference:
            existing = eval_dir / "final_models" / strategy / "predictions.parquet"
            if existing.is_file():
                preds_df = pd.read_parquet(existing)
            else:
                logger.warning(f"{strategy}: no existing predictions, skipping")
                continue
        elif test_ids is None:
            preds_df = fallback_df.copy()
        else:
            try:
                preds_df = run_inference(
                    model_dir=model_dir,
                    mmap_dir=cfg.data.mmap_dir,
                    splits_dir=splits_dir,
                    csv_path=cfg.data.csv_path,
                    label_column=cfg.data.label_columns[0],
                    filename_column=cfg.data.filename_column,
                    scheme=cfg.splits.scheme,
                    test_slide_ids=test_ids,
                    batch_size=cfg.training.batch_size,
                    max_instances=cfg.training.max_instances,
                    num_workers=cfg.platform.num_workers,
                    accelerator=cfg.platform.accelerator,
                    devices=cfg.platform.devices,
                    precision=cfg.platform.precision,
                )
            except Exception as e:
                logger.error(f"{strategy} inference failed: {e}", exc_info=True)
                model_results[strategy] = {"error": str(e)}
                continue

        if preds_df.empty:
            logger.warning(f"{strategy}: empty predictions")
            model_results[strategy] = {"error": "empty predictions"}
            continue

        # Attach patient IDs
        preds_df = _attach_patient_id(
            preds_df, cfg.data.csv_path,
            cfg.data.filename_column, patient_column,
        )

        # Full evaluation
        model_results[strategy] = evaluate_final_model(
            model_name=strategy,
            preds_df=preds_df,
            eval_dir=eval_dir,
            patient_column=patient_column,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
            class_names=class_names,
            perturbations=perturbations,
        )

    return model_results


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _attach_patient_id(
    df: pd.DataFrame,
    csv_path: Optional[str],
    filename_column: str,
    patient_column: str,
) -> pd.DataFrame:
    """
    Attach patient_id from the labels CSV to predictions DataFrame.

    If patient_column not in CSV or csv_path is None, uses slide_id as patient_id.
    """
    if patient_column in df.columns:
        return df

    if csv_path is None or not Path(csv_path).is_file():
        df = df.copy()
        df[patient_column] = df["slide_id"]
        return df

    try:
        csv_df = pd.read_csv(csv_path)
        if patient_column in csv_df.columns and filename_column in csv_df.columns:
            csv_df["_slide_id"] = csv_df[filename_column].apply(
                lambda x: Path(str(x)).stem
            )
            mapping = csv_df.set_index("_slide_id")[patient_column].to_dict()
            df = df.copy()
            df[patient_column] = df["slide_id"].map(mapping)

            mask = df[patient_column].isna()
            if mask.any():
                df.loc[mask, patient_column] = df.loc[mask, "slide_id"]
                logger.warning(
                    f"{mask.sum()}/{len(df)} slides have no patient mapping — "
                    f"using slide_id"
                )
        else:
            df = df.copy()
            df[patient_column] = df["slide_id"]
            logger.info(
                f"'{patient_column}' not in CSV columns "
                f"{list(csv_df.columns)} — using slide_id"
            )
    except Exception as e:
        logger.warning(f"Failed to attach patient_id: {e} — using slide_id")
        df = df.copy()
        df[patient_column] = df["slide_id"]

    return df


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _print_metrics_report(label: str, metrics) -> None:
    """Print a concise metrics report."""
    d = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
    print(f"\n{'─' * 55}")
    print(f"  {label} (n={d.get('n_samples', '?')})")
    print(f"{'─' * 55}")

    for key in [
        "auroc_macro", "balanced_accuracy", "accuracy",
        "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted",
        "f1_macro", "f1_weighted",
        "kappa", "mcc", "log_loss_val",
    ]:
        val = d.get(key)
        if val is not None:
            print(f"  {key:25s}: {val:.4f}")

    for key in ["precision_per_class", "recall_per_class", "f1_per_class",
                "specificity_per_class"]:
        pcd = d.get(key, {})
        if pcd:
            parts = [f"c{k}={v:.3f}" for k, v in pcd.items() if v is not None]
            if parts:
                print(f"  {key:25s}: {', '.join(parts)}")

    print(f"{'─' * 55}")


def _get_n_folds(cfg: DictConfig) -> int:
    scheme = cfg.splits.scheme
    if scheme in ("holdout", "custom_holdout"):
        return 1
    elif scheme in ("kfold", "custom_kfold"):
        return cfg.splits.n_folds
    elif scheme == "monte_carlo":
        return cfg.splits.n_repeats
    elif scheme == "nested_cv":
        return cfg.splits.n_folds
    else:
        return 1


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Stage 6 entry point.

    Config composes the same groups as train.yaml (platform, data, encoder,
    splits, model, training), so all paths resolve identically — no need
    to load train_dir/config.yaml. Eval-specific settings live under cfg.eval.
    """
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        force=True,
    )

    logger.info("=" * 60)
    logger.info("  Stage 6: Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    start_time = time.monotonic()

    # ── Resolve paths ─────────────────────────────────────────────────────
    # train_dir and eval_dir are derived from the same exp_name formula
    # as train.py, so they point to the same directories automatically.
    train_dir = Path(cfg.train_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    eval_dir = Path(cfg.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Read eval settings ────────────────────────────────────────────────
    # All under cfg.eval.* — typed and documented in evaluate.yaml
    e = cfg.eval
    patient_column = e.patient_column        # from ${data.patient_id_column}
    n_bootstrap = e.n_bootstrap
    alpha = e.alpha
    seed = e.seed
    skip_inference = e.skip_inference
    perturbations = tuple(e.perturbations)   # (0.01, 0.02, 0.05, 0.10)

    class_names = OmegaConf.select(cfg, "eval.class_names", default=None)
    if class_names is not None:
        class_names = list(class_names)

    comparison_weights = OmegaConf.to_container(
        e.comparison_weights, resolve=True,
    )

    n_folds = _get_n_folds(cfg)

    # ── Save eval config (fully resolved) ────────────────────────────────
    (eval_dir / "eval_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Part 1: OOF evaluation ───────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("  Part 1: OOF Evaluation")
    logger.info("=" * 50)

    oof_result = evaluate_oof(
        train_dir, eval_dir,
        patient_column=patient_column,
        csv_path=cfg.data.csv_path,
        filename_column=cfg.data.filename_column,
    )

    # ── Part 1b: Per-fold test ───────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("  Part 1b: Per-Fold Test Evaluation")
    logger.info("=" * 50)

    fold_test_result = evaluate_per_fold_test(
        train_dir, eval_dir, n_folds,
        patient_column=patient_column,
        csv_path=cfg.data.csv_path,
        filename_column=cfg.data.filename_column,
    )

    # ── Part 2: Final model evaluation ───────────────────────────────────
    logger.info("=" * 50)
    logger.info("  Part 2: Final Model Evaluation")
    logger.info("=" * 50)

    model_results = evaluate_final_models(
        train_dir, eval_dir, cfg,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
        skip_inference=skip_inference,
        patient_column=patient_column,
        class_names=class_names,
        perturbations=perturbations,
    )

    # ── Part 3: Model comparison ─────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("  Part 3: Model Comparison")
    logger.info("=" * 50)

    valid_results = {
        k: v for k, v in model_results.items()
        if isinstance(v, dict) and "error" not in v
    }

    if valid_results:
        from oceanpath.eval.compare import compare_models, save_comparison, save_recommendation
        from oceanpath.eval.plots import plot_model_comparison

        comparison = compare_models(
            valid_results,
            primary_metric=e.primary_metric,
            primary_level=e.primary_level,
            weights=comparison_weights,
        )
        save_comparison(comparison, eval_dir / "model_comparison.json")
        save_recommendation(comparison, eval_dir / "recommended_model.json")

        plot_model_comparison(
            comparison, eval_dir / "final_models" / "model_comparison.png",
        )

        print(f"\n{'=' * 60}")
        print("  MODEL RECOMMENDATION")
        print(f"{'=' * 60}")
        print(comparison.get("rationale", "No recommendation available"))
        print(f"{'=' * 60}\n")
    else:
        logger.warning("No valid final models to compare")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time

    print(f"\n{'=' * 60}")
    print(f"  Stage 6 Complete")
    print(f"{'=' * 60}")
    print(f"  Output:       {eval_dir}")
    print(f"  Time:         {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  OOF slides:   {oof_result.get('n_slides', 'N/A')}")

    if valid_results:
        for name in valid_results:
            perf = valid_results[name].get("performance", {})
            auroc = (
                perf.get("patient_level", {})
                .get("auroc", {})
                .get("point", "N/A")
            )
            auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) else str(auroc)
            print(f"  {name:12s} patient AUROC: {auroc_str}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()