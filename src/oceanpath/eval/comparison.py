"""
Cross-experiment comparison with statistical rigor.

Compares multiple experiments (different architectures, encoders, hyperparams)
on the same cohort using proper paired statistical tests.

Why this exists:
════════════════
Eyeballing W&B dashboards gives you "0.87 vs 0.85 AUROC" but NOT:
  - Is the difference statistically significant?
  - What's the confidence interval on the difference?
  - Could the improvement come from a specific subgroup?

This module provides three levels of comparison:

  1. DeLong test for AUC — the gold standard for comparing ROC curves
     on the same test set. Non-parametric, handles correlated predictions.

  2. McNemar's test — for comparing accuracy on paired observations.
     "Did the two models disagree on the same slides, and if so, which
     model was right more often?"

  3. Bootstrap paired differences — for ANY metric. Resamples the paired
     predictions and computes the distribution of (metric_A - metric_B).
     A 95% CI that excludes 0 means the difference is significant.

All tests require PAIRED predictions: same slides, same labels.
Run evaluate.py on the same cohort for each experiment first.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DeLong test for paired AUC comparison
# ═════════════════════════════════════════════════════════════════════════════
#
# Reference: DeLong et al. (1988) "Comparing the areas under two or more
# correlated receiver operating characteristic curves: a nonparametric
# approach." Biometrics 44(3):837-845.
#
# The implementation uses the fast O(N log N) algorithm from Sun & Xu (2014).


def delong_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> dict:
    """
    DeLong's test for comparing two correlated AUCs.

    Tests H0: AUC_A = AUC_B on the SAME set of samples.

    Parameters
    ----------
    labels : (N,) binary labels {0, 1}.
    probs_a : (N,) predicted probabilities from model A.
    probs_b : (N,) predicted probabilities from model B.

    Returns
    -------
    dict with: auc_a, auc_b, auc_diff, z_stat, p_value, significant_at_05.
    """
    labels = np.asarray(labels, dtype=int)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)

    if len(np.unique(labels)) < 2:
        return {"error": "Need both positive and negative samples"}

    # Separate positive and negative
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_a = probs_a[pos_mask]
    neg_a = probs_a[neg_mask]
    pos_b = probs_b[pos_mask]
    neg_b = probs_b[neg_mask]

    m = pos_mask.sum()  # number of positives
    n = neg_mask.sum()  # number of negatives

    # Structural components (placement values)
    V_a10 = _compute_placements(pos_a, neg_a) / n  # [m]
    V_a01 = _compute_placements(neg_a, pos_a) / m  # [n]
    V_b10 = _compute_placements(pos_b, neg_b) / n
    V_b01 = _compute_placements(neg_b, pos_b) / m

    auc_a = V_a10.mean()
    auc_b = V_b10.mean()

    # Covariance matrix of (AUC_A, AUC_B)
    S10 = np.cov(np.column_stack([V_a10, V_b10]).T)  # [2, 2]
    S01 = np.cov(np.column_stack([V_a01, V_b01]).T)

    S = S10 / m + S01 / n

    # Z-statistic for the difference
    diff = auc_a - auc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return {
            "auc_a": float(auc_a), "auc_b": float(auc_b),
            "auc_diff": float(diff),
            "error": "Variance of difference is non-positive",
        }

    z = diff / np.sqrt(var_diff)
    p = 2.0 * stats.norm.sf(abs(z))  # two-sided

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(diff),
        "z_stat": float(z),
        "p_value": float(p),
        "significant_at_05": bool(p < 0.05),
        "significant_at_01": bool(p < 0.01),
    }


def _compute_placements(
    positives: np.ndarray, negatives: np.ndarray,
) -> np.ndarray:
    """
    Compute placement values for the DeLong test.

    For each positive, count how many negatives it exceeds (ties count 0.5).
    Uses sorting for O(N log N) instead of O(N^2) pairwise comparison.
    """
    placements = np.zeros(len(positives))
    for i, p in enumerate(positives):
        placements[i] = np.sum(p > negatives) + 0.5 * np.sum(p == negatives)
    return placements


# ═════════════════════════════════════════════════════════════════════════════
# McNemar's test for paired accuracy comparison
# ═════════════════════════════════════════════════════════════════════════════


def mcnemar_test(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """
    McNemar's test for comparing two classifiers on paired data.

    Focuses on the discordant pairs: slides where A and B disagree.
    Tests whether A is right-when-B-is-wrong more often than vice versa.

    Parameters
    ----------
    labels : (N,) true labels.
    preds_a : (N,) predicted classes from model A.
    preds_b : (N,) predicted classes from model B.

    Returns
    -------
    dict with: n_concordant, n_a_right_b_wrong, n_a_wrong_b_right,
               chi2, p_value, significant_at_05.
    """
    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # 2x2 contingency table of (A correct?, B correct?)
    both_right = (correct_a & correct_b).sum()
    both_wrong = (~correct_a & ~correct_b).sum()
    a_right_b_wrong = (correct_a & ~correct_b).sum()  # b
    a_wrong_b_right = (~correct_a & correct_b).sum()   # c

    n_discordant = a_right_b_wrong + a_wrong_b_right

    if n_discordant == 0:
        return {
            "both_right": int(both_right),
            "both_wrong": int(both_wrong),
            "a_right_b_wrong": int(a_right_b_wrong),
            "a_wrong_b_right": int(a_wrong_b_right),
            "p_value": 1.0,
            "significant_at_05": False,
            "note": "No discordant pairs — models agree on every sample",
        }

    # Use exact binomial test for small counts, chi-squared otherwise
    if n_discordant < 25:
        # Exact binomial: under H0, P(A right | discordant) = 0.5
        p_val = float(stats.binomtest(a_right_b_wrong, n_discordant, 0.5).pvalue)
        test_type = "exact_binomial"
        chi2 = None
    else:
        # McNemar with continuity correction
        chi2 = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n_discordant
        p_val = float(stats.chi2.sf(chi2, df=1))
        chi2 = float(chi2)
        test_type = "chi_squared_corrected"

    return {
        "both_right": int(both_right),
        "both_wrong": int(both_wrong),
        "a_right_b_wrong": int(a_right_b_wrong),
        "a_wrong_b_right": int(a_wrong_b_right),
        "n_discordant": int(n_discordant),
        "test_type": test_type,
        "chi2": chi2,
        "p_value": p_val,
        "significant_at_05": bool(p_val < 0.05),
        "significant_at_01": bool(p_val < 0.01),
        "advantage": (
            "A" if a_right_b_wrong > a_wrong_b_right
            else "B" if a_wrong_b_right > a_right_b_wrong
            else "tied"
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Bootstrap paired difference test
# ═════════════════════════════════════════════════════════════════════════════


def bootstrap_paired_difference(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for the difference in any metric.

    Resamples the PAIRED predictions and computes:
      delta_i = metric(A, resample_i) - metric(B, resample_i)

    A 95% CI for delta that excludes 0 means significant difference.

    Parameters
    ----------
    labels : (N,) true labels.
    probs_a, probs_b : (N,) or (N,C) predicted probabilities.
    metric_fn : callable(labels, probs) → float.
    n_bootstrap : number of iterations.
    alpha : significance level.
    seed : random seed.

    Returns
    -------
    dict with: point_a, point_b, diff, ci_lower, ci_upper,
               p_value, significant_at_05.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)

    # Point estimates
    point_a = float(metric_fn(labels, probs_a))
    point_b = float(metric_fn(labels, probs_b))
    point_diff = point_a - point_b

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels[idx]

        if len(np.unique(boot_labels)) < 2:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sa = metric_fn(boot_labels, probs_a[idx])
                sb = metric_fn(boot_labels, probs_b[idx])
            if np.isfinite(sa) and np.isfinite(sb):
                diffs.append(sa - sb)
        except Exception:
            continue

    if len(diffs) < 50:
        return {
            "point_a": point_a, "point_b": point_b,
            "diff": point_diff,
            "error": f"Only {len(diffs)} valid bootstrap samples",
        }

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # Two-sided p-value: proportion of bootstrap diffs on wrong side of 0
    if point_diff >= 0:
        p = 2 * (diffs < 0).mean()
    else:
        p = 2 * (diffs > 0).mean()
    p = min(float(p), 1.0)

    return {
        "point_a": point_a,
        "point_b": point_b,
        "diff": point_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(diffs)),
        "p_value": p,
        "significant_at_05": bool(ci_lower > 0 or ci_upper < 0),
        "n_valid_bootstrap": len(diffs),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Multi-experiment loader
# ═════════════════════════════════════════════════════════════════════════════


def load_experiment_predictions(
    exp_dir: str,
    prediction_file: str = "oof_predictions.parquet",
) -> Optional[pd.DataFrame]:
    """
    Load predictions from an experiment directory.

    Searches for predictions in order:
      1. {exp_dir}/{prediction_file}       (OOF predictions)
      2. {exp_dir}/eval/best_fold/predictions.parquet
      3. {exp_dir}/eval/ensemble/predictions.parquet

    Returns DataFrame with: slide_id, label, prob_* columns.
    """
    exp_dir = Path(exp_dir)

    candidates = [
        exp_dir / prediction_file,
        exp_dir / "eval" / "best_fold" / "predictions.parquet",
        exp_dir / "eval" / "ensemble" / "predictions.parquet",
    ]

    for path in candidates:
        if path.is_file():
            df = pd.read_parquet(str(path))
            logger.info(f"Loaded {len(df)} predictions from {path}")
            return df

    logger.warning(f"No predictions found in {exp_dir}")
    return None


def load_experiment_metrics(exp_dir: str) -> Optional[dict]:
    """
    Load metrics summary from an experiment's evaluation output.

    Searches for:
      1. {exp_dir}/eval/oof_evaluation.json
      2. {exp_dir}/eval/evaluation_report.json
    """
    exp_dir = Path(exp_dir)
    candidates = [
        exp_dir / "eval" / "oof_evaluation.json",
        exp_dir / "eval" / "evaluation_report.json",
    ]

    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.warning(f"Cannot parse {path}: {e}")

    return None


def load_experiment_config(exp_dir: str) -> Optional[dict]:
    """Load the resolved Hydra config from a training run."""
    exp_dir = Path(exp_dir)
    candidates = [
        exp_dir / "resolved_config.yaml",
        exp_dir / "eval" / "eval_config.yaml",
        exp_dir / ".hydra" / "config.yaml",
    ]

    for path in candidates:
        if path.is_file():
            try:
                from omegaconf import OmegaConf
                cfg = OmegaConf.load(str(path))
                return OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                pass

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Full pairwise comparison
# ═════════════════════════════════════════════════════════════════════════════


def compare_experiments(
    experiment_dirs: dict[str, str],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Run full pairwise comparison across multiple experiments.

    Parameters
    ----------
    experiment_dirs : {name: path} mapping of experiment names to directories.
    n_bootstrap : bootstrap iterations for paired tests.
    alpha : significance level.
    seed : random seed.

    Returns
    -------
    dict with:
      experiments: {name: {metrics_summary, config_summary}}
      pairwise: [{pair, delong, mcnemar, bootstrap_auroc, bootstrap_bacc}]
      ranking: sorted by AUROC
      summary_table: markdown-formatted comparison table
    """
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    # ── Load all experiments ──────────────────────────────────────────────
    experiments = {}
    for name, exp_dir in experiment_dirs.items():
        preds = load_experiment_predictions(exp_dir)
        metrics = load_experiment_metrics(exp_dir)
        config = load_experiment_config(exp_dir)

        if preds is None:
            logger.warning(f"Skipping {name}: no predictions found")
            continue

        experiments[name] = {
            "dir": exp_dir,
            "predictions": preds,
            "metrics": metrics,
            "config": config,
        }

    if len(experiments) < 2:
        return {"error": f"Need >= 2 experiments, found {len(experiments)}"}

    # ── Build metrics summary ─────────────────────────────────────────────
    summary = {}
    for name, exp in experiments.items():
        preds = exp["predictions"]
        prob_cols = sorted([c for c in preds.columns if c.startswith("prob_")])
        labels = preds["label"].values

        entry = {
            "n_slides": len(preds),
            "n_classes": len(prob_cols),
            "dir": exp["dir"],
        }

        if prob_cols:
            if len(prob_cols) == 1:
                probs = preds[prob_cols[0]].values
            else:
                probs = preds[prob_cols].values

            try:
                if len(prob_cols) == 1:
                    entry["auroc"] = float(roc_auc_score(labels, probs))
                else:
                    entry["auroc"] = float(roc_auc_score(
                        labels, probs, multi_class="ovr", average="macro",
                    ))
            except Exception:
                entry["auroc"] = None

            pred_classes = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(axis=1)
            try:
                entry["balanced_accuracy"] = float(balanced_accuracy_score(labels, pred_classes))
            except Exception:
                entry["balanced_accuracy"] = None

        # Config highlights
        cfg = exp.get("config") or {}
        entry["arch"] = _nested_get(cfg, "model.arch", "?")
        entry["encoder"] = _nested_get(cfg, "encoder.name", "?")
        entry["lr"] = _nested_get(cfg, "training.lr", "?")

        summary[name] = entry

    # ── Pairwise comparisons ──────────────────────────────────────────────
    names = sorted(experiments.keys())
    pairwise = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            result = _compare_pair(
                name_a, experiments[name_a],
                name_b, experiments[name_b],
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                seed=seed,
            )
            pairwise.append(result)

    # ── Ranking ───────────────────────────────────────────────────────────
    ranking = sorted(
        summary.items(),
        key=lambda x: x[1].get("auroc") or 0,
        reverse=True,
    )

    # ── Summary table ─────────────────────────────────────────────────────
    table = _build_comparison_table(summary, ranking, pairwise)

    return {
        "experiments": summary,
        "pairwise": pairwise,
        "ranking": [{"name": n, **s} for n, s in ranking],
        "summary_table": table,
    }


def _compare_pair(
    name_a: str, exp_a: dict,
    name_b: str, exp_b: dict,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run full statistical comparison between two experiments."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    preds_a = exp_a["predictions"]
    preds_b = exp_b["predictions"]

    # Align on shared slides
    shared = set(preds_a["slide_id"]) & set(preds_b["slide_id"])
    if len(shared) < 10:
        return {
            "pair": f"{name_a} vs {name_b}",
            "error": f"Only {len(shared)} shared slides (need >= 10)",
        }

    df_a = preds_a[preds_a["slide_id"].isin(shared)].sort_values("slide_id")
    df_b = preds_b[preds_b["slide_id"].isin(shared)].sort_values("slide_id")

    labels = df_a["label"].values
    prob_cols = sorted([c for c in df_a.columns if c.startswith("prob_")])

    # Extract probabilities
    if len(prob_cols) == 1:
        probs_a = df_a[prob_cols[0]].values
        probs_b = df_b[prob_cols[0]].values
    else:
        probs_a = df_a[prob_cols].values
        probs_b = df_b[prob_cols].values

    # Predicted classes
    if probs_a.ndim == 1:
        preds_cls_a = (probs_a >= 0.5).astype(int)
        preds_cls_b = (probs_b >= 0.5).astype(int)
    else:
        preds_cls_a = probs_a.argmax(axis=1)
        preds_cls_b = probs_b.argmax(axis=1)

    result = {
        "pair": f"{name_a} vs {name_b}",
        "n_shared_slides": len(shared),
        "n_total_a": len(preds_a),
        "n_total_b": len(preds_b),
    }

    # ── DeLong test (binary only) ─────────────────────────────────────────
    if probs_a.ndim == 1 and len(np.unique(labels)) == 2:
        result["delong"] = delong_test(labels, probs_a, probs_b)
    else:
        result["delong"] = {"skipped": "multiclass — DeLong requires binary"}

    # ── McNemar test ──────────────────────────────────────────────────────
    result["mcnemar"] = mcnemar_test(labels, preds_cls_a, preds_cls_b)

    # ── Bootstrap AUROC ───────────────────────────────────────────────────
    def _auroc(lab, prob):
        if prob.ndim == 1:
            return roc_auc_score(lab, prob)
        return roc_auc_score(lab, prob, multi_class="ovr", average="macro")

    try:
        result["bootstrap_auroc"] = bootstrap_paired_difference(
            labels, probs_a, probs_b,
            metric_fn=_auroc,
            n_bootstrap=n_bootstrap, alpha=alpha, seed=seed,
        )
    except Exception as e:
        result["bootstrap_auroc"] = {"error": str(e)}

    # ── Bootstrap balanced accuracy ───────────────────────────────────────
    def _bacc(lab, prob):
        pred = (prob >= 0.5).astype(int) if prob.ndim == 1 else prob.argmax(axis=1)
        return balanced_accuracy_score(lab, pred)

    try:
        result["bootstrap_bacc"] = bootstrap_paired_difference(
            labels, probs_a, probs_b,
            metric_fn=_bacc,
            n_bootstrap=n_bootstrap, alpha=alpha, seed=seed,
        )
    except Exception as e:
        result["bootstrap_bacc"] = {"error": str(e)}

    return result


def _build_comparison_table(
    summary: dict, ranking: list, pairwise: list,
) -> str:
    """Build markdown comparison table."""
    lines = ["| Experiment | Arch | Encoder | AUROC | Bal. Acc | N |",
             "|------------|------|---------|-------|---------|---|"]

    for name, s in ranking:
        auroc = f"{s['auroc']:.4f}" if s.get('auroc') is not None else "—"
        bacc = f"{s['balanced_accuracy']:.4f}" if s.get('balanced_accuracy') is not None else "—"
        lines.append(
            f"| {name} | {s.get('arch', '?')} | {s.get('encoder', '?')} "
            f"| {auroc} | {bacc} | {s.get('n_slides', '?')} |"
        )

    if pairwise:
        lines.append("")
        lines.append("**Pairwise significance:**")
        for pw in pairwise:
            if "error" in pw:
                lines.append(f"- {pw['pair']}: {pw['error']}")
                continue

            dl = pw.get("delong", {})
            mc = pw.get("mcnemar", {})

            dl_sig = "**sig**" if dl.get("significant_at_05") else "n.s."
            mc_sig = "**sig**" if mc.get("significant_at_05") else "n.s."

            dl_p = f"p={dl.get('p_value', '?'):.4f}" if isinstance(dl.get("p_value"), float) else ""
            mc_p = f"p={mc.get('p_value', '?'):.4f}" if isinstance(mc.get("p_value"), float) else ""

            lines.append(
                f"- {pw['pair']}: DeLong {dl_sig} ({dl_p}), "
                f"McNemar {mc_sig} ({mc_p})"
            )

    return "\n".join(lines)


def _nested_get(d: dict, dotted_key: str, default=None):
    """Get a value from a nested dict using dot notation."""
    keys = dotted_key.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current