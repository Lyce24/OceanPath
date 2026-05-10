#!/usr/bin/env python3
"""
Lightweight supervised MIL training with patient-level evaluation.

Designed for the proposal's supervised baselines and fine-tuning experiments.
Output format matches linear_probing_sklearn.py for direct comparison.

Supports:
  1. ABMIL supervised from scratch
  2. Random-init Mamba2MIL supervised
  3. SSL-pretrained Mamba2MIL fine-tuning

Evaluation contract (apples-to-apples vs the linear-probing pipeline):
  - Predictions are ALWAYS aggregated to patient level before scoring (mean
    of slide-level prob_*). The LP pipeline exposes an `aggregate_to_patient`
    toggle for slide-level scoring; supervised does not, by design — every
    supervised baseline in the proposal scores at the patient level.
  - Multiclass tasks report `auroc_macro_ovr` matching the LP convention.
  - Ordinal tasks (e.g. GEJ Barrett's) use plain CE loss with QWK as a
    metric only — there is no differentiable QWK proxy / ordinal head. This
    is intentional: it matches the linear-probe protocol (which also uses
    softmax probabilities and computes QWK on argmax predictions).

What this script does NOT do (by design):
  - No W&B logging (file logs only)
  - No Phase 2 finalize (no ensemble/refit)
  - No fold-level resume logic
  - No embedding collection
  - No automatic final-report refit or post-hoc ensembling

Output:
  output_dir/
  ├── fold_0/
  │   ├── checkpoints/best.ckpt
  │   ├── patient_predictions_val.csv
  │   └── fold_metrics.json
  ├── ...
  ├── fold_metrics.csv
  ├── patient_predictions_oof.csv
  └── summary.json

Usage:
  python scripts/train_supervised.py data=tcga_nsclc model=abmil
  python scripts/train_supervised.py data=panda model=mamba2mil
  python scripts/train_supervised.py data=tcga_nsclc model=mamba2mil \\
      training.aggregator_weights_path=path/to/ssl.ckpt \\
      training.aggregator_lr=1e-5 training.head_lr=1e-4
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from oceanpath.data.datamodule import (
    MILCollator,
    MILDataModule,
    SimpleMILCollator,
    _normalize_slide_id,
)
from oceanpath.data.dataset import MmapDataset
from oceanpath.modules.train_module import MILTrainModule

# Platform / DDP / hyperparam resolvers are shared with any future training
# entry-point — see oceanpath.training._helpers for the single source of truth.
from oceanpath.training import barrier as _barrier
from oceanpath.training import global_rank as _global_rank
from oceanpath.training import is_rank_zero as _is_rank_zero
from oceanpath.training import json_default as _json_default
from oceanpath.training import resolve_accelerator as _resolve_accelerator
from oceanpath.training import resolve_devices as _resolve_devices
from oceanpath.training import resolve_force_float32 as _resolve_force_float32
from oceanpath.training import resolve_num_nodes as _resolve_num_nodes
from oceanpath.training import resolve_num_workers as _resolve_num_workers
from oceanpath.training import resolve_persistent_workers as _resolve_persistent_workers
from oceanpath.training import resolve_pin_memory as _resolve_pin_memory
from oceanpath.training import resolve_precision as _resolve_precision
from oceanpath.training import resolve_prefetch_factor as _resolve_prefetch_factor
from oceanpath.training import resolve_strategy as _resolve_strategy

logger = logging.getLogger(__name__)


# ── Patient-level metrics ────────────────────────────────────────────────────


def _patient_metrics(
    slide_preds: pd.DataFrame,
    manifest: pd.DataFrame,
    task_type: str,
) -> dict:
    # CIs are computed once on the OOF concat in `run()` (matching
    # linear_probing_sklearn.py:1253-1265), not per-fold here.
    """Compute patient-level metrics from slide predictions.

    Joins slide predictions with patient_id from manifest, averages
    probabilities per patient, and computes standard metrics.
    """
    # Join to get patient_id
    df = slide_preds.merge(
        manifest[["slide_id", "patient_id"]],
        on="slide_id",
        how="left",
    )
    if df["patient_id"].isna().any():
        n_miss = df["patient_id"].isna().sum()
        logger.warning("%d slides have no patient_id — dropping.", n_miss)
        df = df.dropna(subset=["patient_id"])

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        prob_arr = df[prob_cols].to_numpy(dtype=np.float64, copy=True)
        finite_mask = np.isfinite(prob_arr).all(axis=1)
        if not finite_mask.all():
            n_bad = int((~finite_mask).sum())
            logger.warning(
                "%d slide predictions contain non-finite probabilities — dropping before patient aggregation.",
                n_bad,
            )
            df = df.loc[finite_mask].copy()

    if df.empty:
        return {"n_patients": 0, "error": "no_finite_predictions"}, pd.DataFrame()

    # Aggregate to patient level
    agg = {"label": "first"}
    for c in prob_cols:
        agg[c] = "mean"
    patient = df.groupby("patient_id", sort=False).agg(agg).reset_index()

    y_true = patient["label"].to_numpy(dtype=int)
    if len(y_true) == 0:
        return {"n_patients": 0, "error": "no_patients_after_aggregation"}, pd.DataFrame()
    out: dict = {"n_patients": len(y_true)}

    if task_type == "binary":
        y_prob = patient["prob_1"].to_numpy(dtype=np.float64)
        y_pred = (y_prob >= 0.5).astype(int)
        out["auroc"] = _safe(roc_auc_score, y_true, y_prob)
        out["auprc"] = _safe(average_precision_score, y_true, y_prob)
        out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    else:
        y_prob = patient[prob_cols].to_numpy(dtype=np.float64)
        y_pred = y_prob.argmax(axis=1)
        out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
        out["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        # Match the LP pipeline (linear_probing_sklearn.py) so multiclass
        # supervised metrics are apples-to-apples with linear probing.
        out["auroc_macro_ovr"] = _safe(
            roc_auc_score, y_true, y_prob, multi_class="ovr", average="macro"
        )

    # Build patient prediction DataFrame for saving
    patient_df = patient[["patient_id", "label", *prob_cols]].copy()
    patient_df["pred"] = (
        y_pred if task_type != "binary" else (patient["prob_1"].to_numpy() >= 0.5).astype(int)
    )
    return out, patient_df


def _safe(fn, y_true, y_prob, **kwargs):
    try:
        return float(fn(y_true, y_prob, **kwargs))
    except ValueError:
        return None


def _bootstrap_ci(y_true, y_prob, fn, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _get_checkpoint_epoch(ckpt_path: str) -> int | None:
    """Read the saved epoch from a Lightning checkpoint."""
    if not ckpt_path:
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt.get("epoch")
    return int(epoch) if epoch is not None else None


def _infer_task_type(y) -> str:
    return "binary" if len(np.unique(y)) == 2 else "multiclass"


# ── External test evaluation ─────────────────────────────────────────────────


def _eval_external_test(
    cfg: DictConfig,
    best_ckpt: str,
    fold_dir: Path,
) -> tuple[dict, pd.DataFrame] | None:
    """Evaluate best checkpoint on an external test cohort (e.g. CPTAC).

    Returns (metrics_dict, patient_predictions_df) or None if not configured.
    """
    test_cfg = OmegaConf.select(cfg, "test_data", default=None)
    if test_cfg is None:
        return None
    test_mmap = OmegaConf.select(test_cfg, "mmap_dir", default=None)
    test_csv = OmegaConf.select(test_cfg, "manifest_csv", default=None)
    if not test_mmap or not test_csv:
        return None

    t = cfg.training

    # Load test manifest
    label_col = (
        OmegaConf.select(test_cfg, "label_column", default=None) or cfg.data.label_columns[0]
    )
    patient_col = (
        OmegaConf.select(test_cfg, "patient_column", default=None) or cfg.data.patient_id_column
    )
    filename_col = (
        OmegaConf.select(test_cfg, "filename_column", default=None) or cfg.data.filename_column
    )

    test_manifest = pd.read_csv(test_csv)
    test_manifest["slide_id"] = test_manifest[filename_col].astype(str).map(_normalize_slide_id)
    test_manifest["patient_id"] = test_manifest[patient_col].astype(str)
    test_manifest["label"] = test_manifest[label_col].astype(int)

    labels = dict(zip(test_manifest["slide_id"], test_manifest["label"]))
    slide_ids = test_manifest["slide_id"].tolist()

    # Build dataset
    use_spatial = (
        OmegaConf.select(cfg, "training.cap_strategy", default="random") == "spatial_stratified"
    )
    test_max = (
        OmegaConf.select(cfg, "training.dataset_max_instances", default=None)
        if use_spatial
        else None
    )

    ds = MmapDataset(
        mmap_dir=test_mmap,
        slide_ids=slide_ids,
        labels=labels,
        max_instances=test_max,
        is_train=False,
        cap_strategy=OmegaConf.select(cfg, "training.cap_strategy", default="random"),
        cap_grid_size=OmegaConf.select(cfg, "training.cap_grid_size", default=32),
        return_coords=t.return_coords,
        force_float32=_resolve_force_float32(cfg),
        eval_crop_seed=0,
    )

    from torch.utils.data import DataLoader

    num_workers = _resolve_num_workers(cfg)
    pin_memory = _resolve_pin_memory(cfg)
    persistent_workers = _resolve_persistent_workers(cfg, num_workers)
    collate_fn = (
        MILCollator(
            max_instances=t.max_instances,
            feat_dim=cfg.encoder.feature_dim,
            batch_size=t.batch_size,
            pin_memory=pin_memory,
        )
        if t.use_preallocated_collator and t.max_instances is not None
        else SimpleMILCollator(max_instances=t.max_instances)
    )

    loader_kwargs = {
        "batch_size": t.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = _resolve_prefetch_factor(cfg)

    loader = DataLoader(ds, **loader_kwargs)

    # Inference
    module = MILTrainModule.load_from_checkpoint(best_ckpt)
    module.eval()

    # Always single-device: this function is only called on rank 0 (other
    # ranks are blocked at the post-fold barrier). Forcing devices=1 keeps
    # the eval Trainer from trying to attach to the existing NCCL group.
    trainer = L.Trainer(
        accelerator=_resolve_accelerator(cfg),
        devices=1,
        precision=_resolve_precision(cfg),
        logger=False,
        enable_progress_bar=True,
    )
    trainer.test(module, dataloaders=loader)

    slide_preds = pd.DataFrame(module._test_predictions)
    if slide_preds.empty:
        logger.warning("External test: no predictions collected.")
        return None

    task_type = _infer_task_type(slide_preds["label"].to_numpy())
    metrics, patient_df = _patient_metrics(slide_preds, test_manifest, task_type)

    patient_df.to_csv(fold_dir / "patient_predictions_test_external.csv", index=False)
    logger.info("External test | %s", _fmt_metrics(metrics, task_type))

    del module
    return metrics, patient_df


# ── Fold runner ──────────────────────────────────────────────────────────────


def _run_fold(
    cfg: DictConfig,
    fold_idx: int,
    output_dir: Path,
    manifest: pd.DataFrame,
) -> dict:
    """Train one fold, return patient-level metrics dict."""
    import shutil

    t = cfg.training
    fold_dir = output_dir / f"fold_{fold_idx}"

    # Clean stale outputs from previous runs to prevent ModelCheckpoint
    # from picking up a corrupt old checkpoint as the "best". Rank 0 only,
    # then barrier so other ranks see a consistent fold directory.
    if _is_rank_zero():
        ckpt_dir = fold_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info("Cleaned stale checkpoint dir: %s", ckpt_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    L.seed_everything(t.seed + fold_idx, workers=True)

    splits_dir = str(Path(cfg.platform.splits_root) / cfg.data.name / cfg.splits.name)

    # ── Data ─────────────────────────────────────────────────────────────
    dm = MILDataModule(
        mmap_dir=cfg.data.mmap_dir,
        splits_dir=splits_dir,
        csv_path=cfg.data.csv_path,
        label_column=cfg.data.label_columns[0],
        filename_column=cfg.data.filename_column,
        scheme=cfg.splits.scheme,
        fold=fold_idx,
        batch_size=t.batch_size,
        max_instances=t.max_instances,
        dataset_max_instances=OmegaConf.select(cfg, "training.dataset_max_instances", default=None),
        num_workers=_resolve_num_workers(cfg),
        prefetch_factor=_resolve_prefetch_factor(cfg),
        pin_memory=OmegaConf.select(cfg, "training.pin_memory", default=None),
        persistent_workers=OmegaConf.select(cfg, "training.persistent_workers", default=None),
        class_weighted_sampling=t.class_weighted_sampling,
        instance_dropout=t.instance_dropout,
        feature_noise_std=t.feature_noise_std,
        cache_size_mb=t.cache_size_mb,
        return_coords=t.return_coords,
        verify_splits=t.verify_splits,
        use_preallocated_collator=t.use_preallocated_collator,
        force_float32=_resolve_force_float32(cfg),
        cap_strategy=OmegaConf.select(cfg, "training.cap_strategy", default="random"),
        cap_grid_size=OmegaConf.select(cfg, "training.cap_grid_size", default=32),
        eval_n_crops=OmegaConf.select(cfg, "training.eval_n_crops", default=1),
        length_bucket=OmegaConf.select(cfg, "training.length_bucket", default=False),
        length_bucket_size=OmegaConf.select(cfg, "training.length_bucket_size", default=64),
    )
    dm.setup(stage="fit")

    # ── Model ────────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    module = MILTrainModule(
        arch=cfg.model.arch,
        in_dim=cfg.encoder.feature_dim,
        num_classes=dm.num_classes,
        model_cfg=model_cfg,
        lr=t.lr,
        weight_decay=t.weight_decay,
        lr_scheduler=t.lr_scheduler,
        warmup_epochs=t.warmup_epochs,
        max_epochs=t.max_epochs,
        loss_type=t.loss_type,
        class_weights=OmegaConf.select(cfg, "training.class_weights", default=None),
        focal_gamma=t.focal_gamma,
        monitor_metric=t.monitor_metric,
        monitor_mode=t.monitor_mode,
        canary_interval=t.canary_interval,
        freeze_aggregator=t.freeze_aggregator,
        collect_embeddings=False,
        aggregator_weights_path=OmegaConf.select(
            cfg, "training.aggregator_weights_path", default=None
        ),
        aggregator_lr=OmegaConf.select(cfg, "training.aggregator_lr", default=None),
        head_lr=OmegaConf.select(cfg, "training.head_lr", default=None),
    )

    # ── Callbacks (minimal — no LRMonitor since logger=False) ──────────
    callbacks = [
        ModelCheckpoint(
            dirpath=str(fold_dir / "checkpoints"),
            filename="best",
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            save_top_k=t.get("save_top_k", 1),
            save_last=t.get("save_last", True),
        ),
        EarlyStopping(
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            patience=t.early_stopping_patience,
            min_delta=t.get("early_stopping_min_delta", 0.001),
        ),
    ]

    # ── Train ────────────────────────────────────────────────────────────
    # Use bf16-mixed (not fp16-mixed) to avoid overflow at large batch x N.
    # bf16 has the same exponent range as fp32 so dot products over D=1536
    # elements cannot overflow, unlike fp16 which caps at ~65504.
    precision = _resolve_precision(cfg)
    trainer = L.Trainer(
        max_epochs=t.max_epochs,
        accelerator=_resolve_accelerator(cfg),
        devices=_resolve_devices(cfg),
        strategy=_resolve_strategy(cfg),
        num_nodes=_resolve_num_nodes(cfg),
        precision=precision,
        callbacks=callbacks,
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        deterministic=t.deterministic,
        enable_progress_bar=_is_rank_zero(),
        logger=False,  # no W&B — file logs only
        default_root_dir=str(fold_dir),
    )

    trainer.fit(module, datamodule=dm)

    # ── Evaluate best checkpoint ─────────────────────────────────────────
    # Resolve best checkpoint path on all ranks (paths are deterministic).
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt:
        fallback_ckpt = trainer.checkpoint_callback.last_model_path
        if fallback_ckpt:
            if _is_rank_zero():
                logger.warning(
                    "Fold %d: no best checkpoint was selected for monitor='%s'; falling back to last checkpoint.",
                    fold_idx,
                    t.monitor_metric,
                )
            best_ckpt = fallback_ckpt
    if not best_ckpt:
        if _is_rank_zero():
            logger.error("Fold %d: no checkpoint saved.", fold_idx)
        _barrier()
        return {"fold": fold_idx, "error": "no checkpoint"}

    best_module = MILTrainModule.load_from_checkpoint(best_ckpt)
    best_module.eval()

    # Run on all ranks under DDP — gathering happens in
    # MILTrainModule.on_validation_epoch_end so rank 0 ends up with the full
    # union of slide predictions.
    trainer.validate(best_module, datamodule=dm)

    # From here on only rank 0 does work (metrics, external test, file IO).
    # Other ranks free GPU memory and wait at the post-fold barrier so they
    # don't proceed to fold N+1 before rank 0's writes are durable.
    if not _is_rank_zero():
        del best_module, module, trainer, dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _barrier()
        return {"fold": fold_idx, "rank": _global_rank(), "skipped_on_non_rank0": True}

    slide_preds = pd.DataFrame(best_module._val_predictions)

    if slide_preds.empty:
        logger.error("Fold %d: no predictions collected.", fold_idx)
        del best_module, module, trainer, dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _barrier()
        return {"fold": fold_idx, "error": "no predictions"}

    # ── Patient-level metrics ────────────────────────────────────────────
    task_type = _infer_task_type(slide_preds["label"].to_numpy())
    metrics, patient_df = _patient_metrics(slide_preds, manifest, task_type)
    metrics["fold"] = fold_idx
    metrics["task_type"] = task_type
    metrics["best_epoch"] = _get_checkpoint_epoch(best_ckpt)
    if trainer.checkpoint_callback.best_model_score is not None:
        metrics["best_score"] = float(trainer.checkpoint_callback.best_model_score)

    # ── Optional external test (rank 0 only, single-device) ─────────────
    ext_result = _eval_external_test(cfg, best_ckpt, fold_dir)
    if ext_result is not None:
        ext_metrics, _ = ext_result
        metrics["external_test"] = ext_metrics

    # Save
    patient_df.to_csv(fold_dir / "patient_predictions_val.csv", index=False)
    with open(fold_dir / "fold_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    logger.info("Fold %d | val: %s", fold_idx, _fmt_metrics(metrics, task_type))
    if ext_result is not None:
        logger.info("Fold %d | ext: %s", fold_idx, _fmt_metrics(ext_result[0], task_type))

    # Cleanup
    del best_module, module, trainer, dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Release the other ranks waiting on the post-fold barrier.
    _barrier()
    return metrics


# ── Aggregation ──────────────────────────────────────────────────────────────


def _summarize_folds(fold_metrics: list[dict]) -> dict:
    """Compute mean +/- std across folds for numeric metrics."""
    summary: dict = {"n_folds": len(fold_metrics)}
    numeric_keys = {
        k
        for fm in fold_metrics
        for k, v in fm.items()
        if isinstance(v, (int, float, np.floating)) and k not in {"fold", "n_patients"}
    }
    for k in sorted(numeric_keys):
        vals = [float(fm[k]) for fm in fold_metrics if fm.get(k) is not None]
        if not vals:
            continue
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary


# ── Utilities ────────────────────────────────────────────────────────────────


def _load_manifest(cfg: DictConfig) -> pd.DataFrame:
    """Load manifest and normalize slide IDs to match mmap convention."""
    from oceanpath.data.datamodule import _normalize_slide_id

    df = pd.read_csv(cfg.data.csv_path)
    df["slide_id"] = df[cfg.data.filename_column].astype(str).map(_normalize_slide_id)
    df["patient_id"] = df[cfg.data.patient_id_column].astype(str)
    df["label"] = df[cfg.data.label_columns[0]].astype(int)
    return df


def _fmt_metrics(m: dict, task_type: str) -> str:
    if task_type == "binary":
        parts = [
            f"auroc={m.get('auroc', '?'):.4f}" if isinstance(m.get("auroc"), float) else "auroc=?"
        ]
        if m.get("auprc") is not None:
            parts.append(f"auprc={m['auprc']:.4f}")
        parts.append(f"bal_acc={m.get('balanced_acc', 0):.4f}")
    else:
        parts = [
            f"qwk={m.get('qwk', 0):.4f}",
            f"f1={m.get('macro_f1', 0):.4f}",
            f"bal_acc={m.get('balanced_acc', 0):.4f}",
        ]
        if isinstance(m.get("auroc_macro_ovr"), float):
            parts.append(f"auroc_ovr={m['auroc_macro_ovr']:.4f}")
    return ", ".join(parts)


def _get_n_folds(cfg: DictConfig) -> int:
    scheme = cfg.splits.scheme
    if scheme in ("kfold", "custom_kfold"):
        return cfg.splits.n_folds
    if scheme == "holdout":
        return 1
    if scheme == "monte_carlo":
        return cfg.splits.n_repeats
    if scheme == "nested_cv":
        return cfg.splits.n_inner_folds
    return cfg.splits.get("n_folds", 5)


# ── Main ─────────────────────────────────────────────────────────────────────


@hydra.main(config_path="../configs", config_name="train_supervised", version_base=None)
def main(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        force=True,
    )

    # Use TF32 for matmuls on Ampere+ GPUs (faster, negligible precision loss)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(cfg.platform.output_root) / "supervised" / cfg.exp_name
    if _is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save resolved config
        (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    manifest = _load_manifest(cfg)
    n_folds = _get_n_folds(cfg)

    if _is_rank_zero():
        logger.info("Experiment: %s | %d folds | output: %s", cfg.exp_name, n_folds, output_dir)

    # ── Fold loop ────────────────────────────────────────────────────────
    fold_metrics: list[dict] = []
    all_patient_preds: list[pd.DataFrame] = []

    for fold_idx in range(n_folds):
        if _is_rank_zero():
            logger.info("=" * 60)
            logger.info("  Fold %d / %d", fold_idx, n_folds - 1)
            logger.info("=" * 60)

        fm = _run_fold(cfg, fold_idx, output_dir, manifest)

        # Aggregation runs on rank 0 only (file writes + summary).
        if not _is_rank_zero():
            continue

        fold_metrics.append(fm)
        # Collect OOF patient predictions
        pred_path = output_dir / f"fold_{fold_idx}" / "patient_predictions_val.csv"
        if pred_path.exists():
            pdf = pd.read_csv(pred_path)
            pdf["fold"] = fold_idx
            all_patient_preds.append(pdf)

    # Non-rank-0 ranks have nothing left to do; the rank-0 path below writes
    # the summary and prints the final report.
    if not _is_rank_zero():
        return

    # ── Aggregate ────────────────────────────────────────────────────────
    successful_folds = [fm for fm in fold_metrics if "error" not in fm]
    if not successful_folds:
        raise RuntimeError("All folds failed. See fold_metrics.csv / logs for details.")

    summary = _summarize_folds(successful_folds)
    task_type = successful_folds[0].get("task_type", "binary")
    summary["task_type"] = task_type
    summary["exp_name"] = cfg.exp_name
    summary["fold_metrics"] = fold_metrics
    summary["n_requested_folds"] = len(fold_metrics)
    summary["n_successful_folds"] = len(successful_folds)

    # Aggregate external test metrics across folds (if present)
    ext_folds = [fm["external_test"] for fm in successful_folds if "external_test" in fm]
    if ext_folds:
        summary["external_test"] = _summarize_folds(ext_folds)

    # Save fold metrics CSV
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)

    # Save OOF patient predictions
    oof: pd.DataFrame | None = None
    if all_patient_preds:
        oof = pd.concat(all_patient_preds, ignore_index=True)
        oof.to_csv(output_dir / "patient_predictions_oof.csv", index=False)

    # OOF-concat bootstrap CIs — matches linear_probing_sklearn.py:1253-1265
    # so supervised vs LP CI numbers are statistically comparable. Per-fold
    # bootstraps were removed from `_patient_metrics`; the OOF concat below
    # is the single source of truth for confidence intervals.
    n_boot = int(cfg.get("n_boot", 0))
    if n_boot > 0 and task_type == "binary" and oof is not None and "prob_1" in oof.columns:
        oof_y = oof["label"].to_numpy(dtype=int)
        oof_prob = oof["prob_1"].to_numpy(dtype=np.float64)
        for name, fn in [("auroc", roc_auc_score), ("auprc", average_precision_score)]:
            lo, hi = _bootstrap_ci(oof_y, oof_prob, fn, n_boot)
            summary[f"oof_{name}_ci_low"] = lo
            summary[f"oof_{name}_ci_high"] = hi

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    # ── Print ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {cfg.exp_name} — complete ({n_folds} folds)")
    print(f"{'=' * 60}")

    if task_type == "binary":
        for k in ["auroc", "auprc", "balanced_acc", "macro_f1"]:
            mk = f"mean_{k}"
            sk = f"std_{k}"
            if mk in summary:
                print(f"  {k:15s}  {summary[mk]:.4f} +/- {summary.get(sk, 0):.4f}")
    else:
        for k in ["qwk", "macro_f1", "balanced_acc"]:
            mk = f"mean_{k}"
            sk = f"std_{k}"
            if mk in summary:
                print(f"  {k:15s}  {summary[mk]:.4f} +/- {summary.get(sk, 0):.4f}")

    if "external_test" in summary:
        ext = summary["external_test"]
        print(f"\n  External test ({ext.get('n_folds', '?')} folds):")
        if task_type == "binary":
            for k in ["auroc", "auprc", "balanced_acc"]:
                mk, sk = f"mean_{k}", f"std_{k}"
                if mk in ext:
                    print(f"  {k:15s}  {ext[mk]:.4f} +/- {ext.get(sk, 0):.4f}")
        else:
            for k in ["qwk", "macro_f1", "balanced_acc"]:
                mk, sk = f"mean_{k}", f"std_{k}"
                if mk in ext:
                    print(f"  {k:15s}  {ext[mk]:.4f} +/- {ext.get(sk, 0):.4f}")

    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
