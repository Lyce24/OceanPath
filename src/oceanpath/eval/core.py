"""
Core evaluation engine for Stage 6.

Provides comprehensive metric computation for MIL classification:
  - Slide-level and patient-level metrics
  - Bootstrap confidence intervals with correct resampling units
  - Calibration analysis (ECE, Brier, reliability diagrams)
  - Operating point selection (Youden's J, sensitivity/specificity targets)
  - Threshold stability analysis
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MetricsSuite:
    """Complete metrics for one evaluation (slide or patient level)."""

    level: str  # "slide" or "patient"
    n_samples: int = 0
    n_classes: int = 2

    # Classification metrics
    auroc_macro: float = float("nan")
    auroc_per_class: dict = field(default_factory=dict)
    accuracy: float = float("nan")
    balanced_accuracy: float = float("nan")
    precision_macro: float = float("nan")
    precision_weighted: float = float("nan")
    recall_macro: float = float("nan")
    recall_weighted: float = float("nan")
    f1_macro: float = float("nan")
    f1_weighted: float = float("nan")
    kappa: float = float("nan")
    mcc: float = float("nan")
    log_loss_val: float = float("nan")

    # Per-class
    precision_per_class: dict = field(default_factory=dict)
    recall_per_class: dict = field(default_factory=dict)
    f1_per_class: dict = field(default_factory=dict)
    specificity_per_class: dict = field(default_factory=dict)

    confusion_matrix: list | None = None

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d


# ═════════════════════════════════════════════════════════════════════════════
# Probability renormalization (fixes float16 AMP precision drift)
# ═════════════════════════════════════════════════════════════════════════════


def _renormalize_probs(probs: np.ndarray) -> np.ndarray:
    """
    Renormalize multiclass probabilities so rows sum to exactly 1.0.

    AMP float16 softmax can produce rows summing to 0.99998 or 1.00003.
    sklearn's roc_auc_score warns if row sums deviate from 1.0.
    """
    if probs.ndim == 1:
        return np.clip(probs, 0.0, 1.0)

    row_sums = probs.sum(axis=1, keepdims=True)
    # Guard against all-zero rows (shouldn't happen, but be safe)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return probs / row_sums


# ═════════════════════════════════════════════════════════════════════════════
# Core metric computation
# ═════════════════════════════════════════════════════════════════════════════


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    level: str = "slide",
) -> MetricsSuite:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    labels : (N,) int array
    probs : (N,) for binary or (N, C) for multiclass probability array
    level : "slide" or "patient"

    Returns
    -------
    MetricsSuite with all metrics populated.
    """
    n = len(labels)
    if n == 0:
        return MetricsSuite(level=level)

    # Determine task
    if probs.ndim == 1:
        n_classes = 2
        preds = (probs >= 0.5).astype(int)
        probs_matrix = np.column_stack([1 - probs, probs])
    else:
        n_classes = probs.shape[1]
        probs_matrix = _renormalize_probs(probs)
        preds = probs_matrix.argmax(axis=1)

    unique_labels = np.unique(labels)

    m = MetricsSuite(level=level, n_samples=n, n_classes=n_classes)

    # ── Global metrics ───────────────────────────────────────────────────
    m.accuracy = float(accuracy_score(labels, preds))
    m.balanced_accuracy = float(balanced_accuracy_score(labels, preds))
    m.kappa = float(cohen_kappa_score(labels, preds))

    try:
        m.mcc = float(matthews_corrcoef(labels, preds))
    except Exception:
        m.mcc = float("nan")

    # Precision / Recall / F1
    avg_kwargs = {"zero_division": 0}
    m.precision_macro = float(precision_score(labels, preds, average="macro", **avg_kwargs))
    m.precision_weighted = float(precision_score(labels, preds, average="weighted", **avg_kwargs))
    m.recall_macro = float(recall_score(labels, preds, average="macro", **avg_kwargs))
    m.recall_weighted = float(recall_score(labels, preds, average="weighted", **avg_kwargs))
    m.f1_macro = float(f1_score(labels, preds, average="macro", **avg_kwargs))
    m.f1_weighted = float(f1_score(labels, preds, average="weighted", **avg_kwargs))

    # AUROC
    try:
        if n_classes == 2 and len(unique_labels) >= 2:
            m.auroc_macro = float(roc_auc_score(labels, probs_matrix[:, 1]))
        elif n_classes > 2 and len(unique_labels) >= 2:
            m.auroc_macro = float(
                roc_auc_score(
                    labels,
                    probs_matrix,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        m.auroc_macro = float("nan")

    # Per-class AUROC (OVR)
    for c in range(n_classes):
        if c in unique_labels and len(unique_labels) >= 2:
            try:
                binary_labels = (labels == c).astype(int)
                m.auroc_per_class[str(c)] = float(roc_auc_score(binary_labels, probs_matrix[:, c]))
            except ValueError:
                m.auroc_per_class[str(c)] = None
        else:
            m.auroc_per_class[str(c)] = None

    # Log loss
    try:
        m.log_loss_val = float(log_loss(labels, probs_matrix, labels=list(range(n_classes))))
    except Exception:
        m.log_loss_val = float("nan")

    # ── Per-class metrics ────────────────────────────────────────────────
    prec_per = precision_score(labels, preds, average=None, **avg_kwargs)
    rec_per = recall_score(labels, preds, average=None, **avg_kwargs)
    f1_per = f1_score(labels, preds, average=None, **avg_kwargs)

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    m.confusion_matrix = cm.tolist()

    for c in range(n_classes):
        if c < len(prec_per):
            m.precision_per_class[str(c)] = float(prec_per[c])
            m.recall_per_class[str(c)] = float(rec_per[c])
            m.f1_per_class[str(c)] = float(f1_per[c])

        # Specificity = TN / (TN + FP)
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        m.specificity_per_class[str(c)] = float(spec)

    return m


# ═════════════════════════════════════════════════════════════════════════════
# Patient-level aggregation
# ═════════════════════════════════════════════════════════════════════════════


def aggregate_to_patient_level(
    df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Aggregate slide-level predictions to patient-level.

    For patients with multiple slides: average probabilities, keep majority label.
    """
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("No probability columns (prob_*) found in DataFrame")

    if patient_column not in df.columns:
        logger.warning(
            f"Column '{patient_column}' not found. "
            f"Falling back to slide_id as patient_id (1:1 mapping)."
        )
        df = df.copy()
        df[patient_column] = df["slide_id"]

    agg_dict = dict.fromkeys(prob_cols, "mean")
    agg_dict["label"] = lambda x: x.mode().iloc[0]  # majority vote
    agg_dict["slide_id"] = "count"  # n_slides per patient

    patient_df = df.groupby(patient_column).agg(agg_dict).reset_index()
    patient_df = patient_df.rename(columns={"slide_id": "n_slides"})

    logger.info(
        f"Patient aggregation: {len(df)} slides → {len(patient_df)} patients "
        f"(max {patient_df['n_slides'].max()} slides/patient)"
    )
    return patient_df


def extract_probs_and_labels(df: pd.DataFrame):
    """Extract (labels, probs) arrays from a predictions DataFrame.

    Automatically renormalizes multiclass probabilities to sum to 1.0
    (fixes float16 AMP precision drift from training).
    """
    prob_cols = sorted([c for c in df.columns if c.startswith("prob_")])
    labels = df["label"].values.astype(int)

    if len(prob_cols) == 1 and prob_cols[0] == "prob_1":
        # Binary: single probability column
        probs = df["prob_1"].values.astype(np.float64)
        probs = np.clip(probs, 0.0, 1.0)
    else:
        probs = df[prob_cols].values.astype(np.float64)
        probs = _renormalize_probs(probs)

    return labels, probs


# ═════════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═════════════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    resample_column: str = "patient_id",
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI with correct resampling unit.

    Parameters
    ----------
    df : DataFrame with predictions (slide_id, label, prob_*, patient_id)
    metric_fn : callable(labels, probs) -> float
    n_bootstrap : number of bootstrap iterations
    alpha : significance level (0.05 → 95% CI)
    resample_column : column to resample over (patient_id for patient-level,
                      slide_id for slide-level)
    seed : random seed

    Returns
    -------
    dict with keys: point, ci_lower, ci_upper, se, n_bootstrap
    """
    rng = np.random.RandomState(seed)
    labels, probs = extract_probs_and_labels(df)

    # Point estimate
    point = float(metric_fn(labels, probs))

    # Handle missing resample column
    if resample_column not in df.columns:
        resample_column = "slide_id"

    units = df[resample_column].unique()
    n_units = len(units)
    scores = []

    for _ in range(n_bootstrap):
        sampled_units = rng.choice(units, size=n_units, replace=True)

        # Expand: for each sampled unit, get ALL rows (handles multi-slide patients)
        # Use value_counts for efficient resampling with replacement
        unit_counts = pd.Series(sampled_units).value_counts()
        sampled_rows = []
        for unit_val, count in unit_counts.items():
            unit_rows = df[df[resample_column] == unit_val]
            for _ in range(count):
                sampled_rows.append(unit_rows)

        if not sampled_rows:
            continue
        boot_df = pd.concat(sampled_rows, ignore_index=True)

        try:
            boot_labels, boot_probs = extract_probs_and_labels(boot_df)
            if len(np.unique(boot_labels)) < 2:
                continue  # skip degenerate samples
            # Suppress expected warnings from bootstrap resampling:
            # - "y_pred contains classes not in y_true" (class missing from sample)
            # - "y_prob values do not sum to one" (float16 precision)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                s = metric_fn(boot_labels, boot_probs)
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            continue

    if len(scores) < 10:
        logger.warning(
            f"Only {len(scores)} valid bootstrap samples "
            f"(target: {n_bootstrap}). CI may be unreliable."
        )
        return {
            "point": point,
            "ci_lower": None,
            "ci_upper": None,
            "se": None,
            "n_valid_bootstrap": len(scores),
            "n_bootstrap": n_bootstrap,
        }

    scores = np.array(scores)
    ci_lower = float(np.percentile(scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    return {
        "point": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(scores)),
        "n_valid_bootstrap": len(scores),
        "n_bootstrap": n_bootstrap,
    }


def compute_metrics_with_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    patient_column: str = "patient_id",
) -> dict:
    """
    Compute full metrics with bootstrap CIs at both slide and patient level.

    Returns dict with keys:
      slide_level: {metric_name: {point, ci_lower, ci_upper, se}}
      patient_level: {metric_name: {point, ci_lower, ci_upper, se}}
      slide_metrics: MetricsSuite (full detail)
      patient_metrics: MetricsSuite (full detail)
    """
    labels, probs = extract_probs_and_labels(df)
    # is_binary = probs.ndim == 1 or probs.shape[1] == 2

    # ── Point estimates ──────────────────────────────────────────────────
    slide_metrics = compute_metrics(labels, probs, level="slide")

    # Patient-level
    patient_df = aggregate_to_patient_level(df, patient_column=patient_column)
    p_labels, p_probs = extract_probs_and_labels(patient_df)
    patient_metrics = compute_metrics(p_labels, p_probs, level="patient")

    # ── Define metric functions for bootstrap ────────────────────────────
    def _auroc(labels, probs):
        if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 2):
            p = probs if probs.ndim == 1 else probs[:, 1]
            return roc_auc_score(labels, p)
        return roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    def _bacc(labels, probs):
        preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
        return balanced_accuracy_score(labels, preds)

    def _acc(labels, probs):
        preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
        return accuracy_score(labels, preds)

    def _f1_macro(labels, probs):
        preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
        return f1_score(labels, preds, average="macro", zero_division=0)

    def _f1_weighted(labels, probs):
        preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
        return f1_score(labels, preds, average="weighted", zero_division=0)

    def _kappa(labels, probs):
        preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
        return cohen_kappa_score(labels, preds)

    def _precision_macro(labels, probs):
        preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
        return precision_score(labels, preds, average="macro", zero_division=0)

    def _recall_macro(labels, probs):
        preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
        return recall_score(labels, preds, average="macro", zero_division=0)

    metric_fns = {
        "auroc": _auroc,
        "balanced_accuracy": _bacc,
        "accuracy": _acc,
        "f1_macro": _f1_macro,
        "f1_weighted": _f1_weighted,
        "kappa": _kappa,
        "precision_macro": _precision_macro,
        "recall_macro": _recall_macro,
    }

    # ── Bootstrap at slide level ─────────────────────────────────────────
    slide_ci = {}
    for name, fn in metric_fns.items():
        slide_ci[name] = bootstrap_ci(
            df,
            fn,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            resample_column="slide_id",
            seed=seed,
        )

    # ── Bootstrap at patient level ───────────────────────────────────────
    # Resample patients, then expand to slides for metric computation
    patient_ci = {}
    if patient_column in df.columns:
        # For patient-level CI, we resample patients, aggregate, then compute
        def _patient_metric_wrapper(metric_fn, patient_col):
            def _fn(labels, probs):
                # This is called on the patient-aggregated data
                return metric_fn(labels, probs)

            return _fn

        for name, fn in metric_fns.items():
            patient_ci[name] = bootstrap_ci(
                patient_df,
                fn,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                resample_column=patient_column,
                seed=seed,
            )
    else:
        patient_ci = slide_ci  # fallback

    return {
        "slide_level": slide_ci,
        "patient_level": patient_ci,
        "slide_metrics": slide_metrics.to_dict(),
        "patient_metrics": patient_metrics.to_dict(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Calibration analysis
# ═════════════════════════════════════════════════════════════════════════════


def compute_calibration(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics and reliability diagram data.

    Returns dict with:
      ece: Expected Calibration Error
      brier: Brier score (binary) or multi-class Brier
      bins: list of {bin_lower, bin_upper, bin_center, avg_confidence,
                     avg_accuracy, count, gap}
      per_class_brier: {class_id: brier_score}
    """
    if probs.ndim == 1:
        n_classes = 2
        probs = np.clip(probs, 0.0, 1.0)
        probs_matrix = np.column_stack([1 - probs, probs])
        # Binary Brier score
        brier = float(brier_score_loss(labels, probs))
    else:
        n_classes = probs.shape[1]
        probs_matrix = _renormalize_probs(probs)
        # Multi-class Brier: mean of per-class Brier scores
        one_hot = np.zeros_like(probs_matrix)
        for i, label in enumerate(labels):
            one_hot[i, label] = 1.0
        brier = float(np.mean(np.sum((probs_matrix - one_hot) ** 2, axis=1)))

    # Per-class Brier
    per_class_brier = {}
    for c in range(n_classes):
        binary_labels = (labels == c).astype(float)
        per_class_brier[str(c)] = float(brier_score_loss(binary_labels, probs_matrix[:, c]))

    # ECE and reliability diagram — use predicted class confidence
    preds = probs_matrix.argmax(axis=1)
    confidences = probs_matrix[np.arange(len(preds)), preds]
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins_data = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(mask.sum())
        if count == 0:
            bins_data.append(
                {
                    "bin_lower": float(lo),
                    "bin_upper": float(hi),
                    "bin_center": float((lo + hi) / 2),
                    "avg_confidence": None,
                    "avg_accuracy": None,
                    "count": 0,
                    "gap": None,
                }
            )
            continue

        avg_conf = float(confidences[mask].mean())
        avg_acc = float(correct[mask].mean())
        gap = abs(avg_conf - avg_acc)
        ece += gap * (count / len(labels))

        bins_data.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "bin_center": float((lo + hi) / 2),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "count": count,
                "gap": float(gap),
            }
        )

    return {
        "ece": float(ece),
        "brier": brier,
        "per_class_brier": per_class_brier,
        "n_bins": n_bins,
        "bins": bins_data,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Operating points & threshold analysis
# ═════════════════════════════════════════════════════════════════════════════


def compute_operating_points(
    labels: np.ndarray,
    probs: np.ndarray,
) -> dict:
    """
    Compute optimal operating points for binary classification.

    For multiclass, computes one-vs-rest operating points for each class.

    Returns dict with:
      youdens_j: {threshold, sensitivity, specificity, j_statistic}
      high_sensitivity: threshold for ≥95% sensitivity
      high_specificity: threshold for ≥95% specificity
      balanced: threshold for |sens - spec| minimized
      confusion_at_youden: confusion matrix at Youden's J threshold
      per_class: (for multiclass) per-class operating points
    """
    if probs.ndim == 1:
        return _binary_operating_points(labels, probs)
    if probs.shape[1] == 2:
        return _binary_operating_points(labels, probs[:, 1])
    # Multiclass: OVR operating points for each class
    result = {"per_class": {}}
    for c in range(probs.shape[1]):
        binary_labels = (labels == c).astype(int)
        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            result["per_class"][str(c)] = {"error": "single-class — cannot compute"}
            continue
        result["per_class"][str(c)] = _binary_operating_points(
            binary_labels,
            probs[:, c],
        )
    return result


def _binary_operating_points(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compute operating points for binary (or OVR) classification."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    sensitivity = tpr
    specificity = 1 - fpr

    # Youden's J
    j_scores = sensitivity + specificity - 1
    best_j_idx = np.argmax(j_scores)
    youden = {
        "threshold": float(thresholds[best_j_idx]),
        "sensitivity": float(sensitivity[best_j_idx]),
        "specificity": float(specificity[best_j_idx]),
        "j_statistic": float(j_scores[best_j_idx]),
    }

    # High sensitivity (≥95%)
    high_sens = _find_threshold_for_target(
        thresholds,
        sensitivity,
        specificity,
        target_metric="sensitivity",
        target_value=0.95,
    )

    # High specificity (≥95%)
    high_spec = _find_threshold_for_target(
        thresholds,
        sensitivity,
        specificity,
        target_metric="specificity",
        target_value=0.95,
    )

    # Balanced: minimize |sensitivity - specificity|
    balance_diff = np.abs(sensitivity - specificity)
    balanced_idx = np.argmin(balance_diff)
    balanced = {
        "threshold": float(thresholds[balanced_idx]),
        "sensitivity": float(sensitivity[balanced_idx]),
        "specificity": float(specificity[balanced_idx]),
    }

    # Confusion matrix at Youden's threshold
    youden_preds = (probs >= youden["threshold"]).astype(int)
    cm = confusion_matrix(labels, youden_preds).tolist()

    return {
        "youdens_j": youden,
        "high_sensitivity_95": high_sens,
        "high_specificity_95": high_spec,
        "balanced": balanced,
        "confusion_at_youden": cm,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def _find_threshold_for_target(
    thresholds,
    sensitivity,
    specificity,
    target_metric: str,
    target_value: float,
) -> dict:
    """Find threshold achieving target sensitivity or specificity."""
    if target_metric == "sensitivity":
        valid = sensitivity >= target_value
        # metric_vals = sensitivity
        other_vals = specificity
    else:
        valid = specificity >= target_value
        # metric_vals = specificity
        other_vals = sensitivity

    if not valid.any():
        return {
            "threshold": None,
            "sensitivity": None,
            "specificity": None,
            "note": f"Cannot achieve {target_metric} >= {target_value}",
        }

    # Among valid, pick the one with best "other" metric
    valid_indices = np.where(valid)[0]
    best_idx = valid_indices[np.argmax(other_vals[valid_indices])]

    return {
        "threshold": float(thresholds[best_idx]),
        "sensitivity": float(sensitivity[best_idx]),
        "specificity": float(specificity[best_idx]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Threshold stability analysis
# ═════════════════════════════════════════════════════════════════════════════


def compute_threshold_stability(
    labels: np.ndarray,
    probs: np.ndarray,
    base_threshold: float,
    perturbations: tuple = (0.01, 0.02, 0.05, 0.10),
) -> dict:
    """
    Analyze how metrics change with threshold perturbations.

    Reports sensitivity, specificity, PPV, NPV at base ± each perturbation.
    A stable model shows small changes; a fragile one cliff-dives.
    """
    if probs.ndim == 2:
        if probs.shape[1] == 2:
            probs = probs[:, 1]
        else:
            return {"note": "Threshold stability is only for binary classification"}

    def _metrics_at_threshold(t):
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "threshold": float(t),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "accuracy": float(acc),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    results = {"base": _metrics_at_threshold(base_threshold), "perturbations": []}

    for delta in perturbations:
        for sign, _label in [(-1, "minus"), (1, "plus")]:
            t = base_threshold + sign * delta
            t = max(0.0, min(1.0, t))
            entry = _metrics_at_threshold(t)
            entry["delta"] = float(sign * delta)
            entry["delta_label"] = f"{'+' if sign > 0 else '-'}{delta}"

            # Compute change from base
            base = results["base"]
            entry["sens_change"] = entry["sensitivity"] - base["sensitivity"]
            entry["spec_change"] = entry["specificity"] - base["specificity"]
            entry["ppv_change"] = entry["ppv"] - base["ppv"]
            entry["npv_change"] = entry["npv"] - base["npv"]

            results["perturbations"].append(entry)

    # Stability score: max absolute change in sensitivity across perturbations
    # sens_changes = [abs(p["sens_change"]) for p in results["perturbations"]]
    # spec_changes = [abs(p["spec_change"]) for p in results["perturbations"]]
    results["max_sens_change_01"] = max(
        [abs(p["sens_change"]) for p in results["perturbations"] if abs(p["delta"]) <= 0.011],
        default=0.0,
    )
    results["max_sens_change_05"] = max(
        [abs(p["sens_change"]) for p in results["perturbations"] if abs(p["delta"]) <= 0.051],
        default=0.0,
    )
    results["stability_grade"] = _stability_grade(results["max_sens_change_01"])

    return results


def _stability_grade(max_change_01: float) -> str:
    """Grade threshold stability based on ±0.01 sensitivity change."""
    if max_change_01 <= 0.02:
        return "A"  # Excellent: <2% change at ±0.01
    if max_change_01 <= 0.05:
        return "B"  # Good: 2-5% change
    if max_change_01 <= 0.10:
        return "C"  # Fair: 5-10% change
    return "D"  # Poor: >10% change — clinically unstable


# ═════════════════════════════════════════════════════════════════════════════
# PR curve data
# ═════════════════════════════════════════════════════════════════════════════


def compute_pr_curve(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compute precision-recall curve data for binary (or per-class OVR)."""
    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 2):
        p = probs if probs.ndim == 1 else probs[:, 1]
        precision, recall, thresholds = precision_recall_curve(labels, p)
        # Average precision
        from sklearn.metrics import average_precision_score

        ap = float(average_precision_score(labels, p))
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": ap,
        }
    # Multiclass: per-class PR
    per_class = {}
    from sklearn.metrics import average_precision_score

    for c in range(probs.shape[1]):
        binary_labels = (labels == c).astype(int)
        if binary_labels.sum() == 0:
            continue
        precision, recall, thresholds = precision_recall_curve(binary_labels, probs[:, c])
        ap = float(average_precision_score(binary_labels, probs[:, c]))
        per_class[str(c)] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": ap,
        }
    return {"per_class": per_class}
