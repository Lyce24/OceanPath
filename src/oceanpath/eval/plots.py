"""
Stage 6 visualizations.

All plot functions follow the same contract:
  - Take data (numpy arrays / dicts from core.py)
  - Save to a Path
  - Return the Path for logging
  - Use matplotlib with a consistent clinical-report style
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Style defaults (imported once) ────────────────────────────────────────────

_STYLE_APPLIED = False


def _apply_style():
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (6, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    _STYLE_APPLIED = True


COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "tertiary": "#059669",
    "diagonal": "#9CA3AF",
    "fill": "#DBEAFE",
    "bar_gap": "#FCA5A5",
    "bar_acc": "#93C5FD",
}


# ═════════════════════════════════════════════════════════════════════════════
# ROC Curve
# ═════════════════════════════════════════════════════════════════════════════


def plot_roc_curve(
    operating_points: dict,
    auroc: float,
    save_path: Path,
    title: str = "ROC Curve",
) -> Path:
    """Plot ROC curve with operating points annotated."""
    _apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # Handle binary and multiclass
    if "roc_curve" in operating_points:
        _plot_single_roc(ax, operating_points, auroc, title)
    elif "per_class" in operating_points:
        # Multiclass: plot each class
        for c, data in operating_points["per_class"].items():
            if "error" in data:
                continue
            roc = data.get("roc_curve", {})
            if roc:
                ax.plot(roc["fpr"], roc["tpr"], label=f"Class {c}", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "--", color=COLORS["diagonal"], linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title} (macro AUROC={auroc:.3f})")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "No ROC data available", ha="center", va="center")
        ax.set_title(title)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    logger.info(f"ROC curve → {save_path}")
    return save_path


def _plot_single_roc(ax, op: dict, auroc: float, title: str):
    import matplotlib.pyplot as plt

    roc = op["roc_curve"]
    ax.plot(roc["fpr"], roc["tpr"], color=COLORS["primary"], linewidth=2,
            label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["diagonal"], linewidth=1)

    # Mark operating points
    markers = [
        ("youdens_j", "★", COLORS["secondary"], "Youden's J"),
        ("high_sensitivity_95", "▲", COLORS["tertiary"], "Sens ≥ 0.95"),
        ("balanced", "●", "#7C3AED", "Balanced"),
    ]
    for key, marker, color, label in markers:
        pt = op.get(key, {})
        if pt and pt.get("threshold") is not None:
            sens = pt["sensitivity"]
            spec = pt["specificity"]
            fpr_pt = 1 - spec
            ax.plot(fpr_pt, sens, marker, color=color, markersize=10,
                    label=f"{label} (t={pt['threshold']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"{title} (AUROC={auroc:.3f})")
    ax.legend(loc="lower right")


# ═════════════════════════════════════════════════════════════════════════════
# PR Curve
# ═════════════════════════════════════════════════════════════════════════════


def plot_pr_curve(
    pr_data: dict,
    save_path: Path,
    title: str = "Precision-Recall Curve",
) -> Path:
    """Plot precision-recall curve."""
    _apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if "precision" in pr_data:
        # Binary
        ap = pr_data.get("average_precision", 0)
        ax.plot(pr_data["recall"], pr_data["precision"],
                color=COLORS["primary"], linewidth=2,
                label=f"AP = {ap:.3f}")
    elif "per_class" in pr_data:
        for c, data in pr_data["per_class"].items():
            ap = data.get("average_precision", 0)
            ax.plot(data["recall"], data["precision"],
                    linewidth=1.5, label=f"Class {c} (AP={ap:.3f})")
    else:
        ax.text(0.5, 0.5, "No PR data", ha="center", va="center")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    logger.info(f"PR curve → {save_path}")
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Calibration Plot
# ═════════════════════════════════════════════════════════════════════════════


def plot_calibration_curve(
    calibration: dict,
    save_path: Path,
    title: str = "Calibration (Reliability Diagram)",
) -> Path:
    """Plot calibration reliability diagram with gap bars."""
    _apply_style()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[3, 1],
                                     gridspec_kw={"hspace": 0.3})

    bins = calibration["bins"]
    ece = calibration["ece"]
    brier = calibration["brier"]

    centers = []
    accuracies = []
    confidences = []
    counts = []
    gaps = []

    for b in bins:
        if b["count"] > 0:
            centers.append(b["bin_center"])
            accuracies.append(b["avg_accuracy"])
            confidences.append(b["avg_confidence"])
            counts.append(b["count"])
            gaps.append(b["gap"])

    if centers:
        width = 0.8 / calibration["n_bins"]

        # Top: reliability diagram
        ax1.bar(centers, accuracies, width=width, color=COLORS["bar_acc"],
                edgecolor=COLORS["primary"], linewidth=0.8, label="Accuracy", zorder=2)
        ax1.bar(centers, gaps, bottom=accuracies, width=width,
                color=COLORS["bar_gap"], edgecolor=COLORS["secondary"],
                linewidth=0.5, alpha=0.6, label="Gap", zorder=2)
        ax1.plot([0, 1], [0, 1], "--", color=COLORS["diagonal"],
                 linewidth=1.5, label="Perfect calibration")

    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"{title}\nECE={ece:.4f}, Brier={brier:.4f}")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper left")

    # Bottom: sample count histogram
    if centers:
        ax2.bar(centers, counts, width=width, color=COLORS["primary"], alpha=0.7)
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Samples per Bin")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    logger.info(f"Calibration plot → {save_path}")
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Confusion Matrix
# ═════════════════════════════════════════════════════════════════════════════


def plot_confusion_matrix(
    cm: list,
    save_path: Path,
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> Path:
    """Plot confusion matrix heatmap."""
    _apply_style()
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)
    n_classes = cm_arr.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(5, n_classes + 2), max(4, n_classes + 1)))

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div by 0
        cm_norm = cm_arr / row_sums
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    else:
        im = ax.imshow(cm_arr, cmap="Blues")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells with both count and percentage
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm_arr[i, j]
            if normalize:
                pct = cm_norm[i, j]
                text = f"{count}\n({pct:.1%})"
            else:
                text = str(count)

            # White text on dark cells, black on light
            threshold = 0.5 if normalize else cm_arr.max() / 2
            val = cm_norm[i, j] if normalize else cm_arr[i, j]
            color = "white" if val > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    logger.info(f"Confusion matrix → {save_path}")
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Threshold Stability
# ═════════════════════════════════════════════════════════════════════════════


def plot_threshold_stability(
    stability: dict,
    save_path: Path,
    title: str = "Threshold Stability",
) -> Path:
    """Plot how sensitivity/specificity change with threshold perturbations."""
    _apply_style()
    import matplotlib.pyplot as plt

    if "note" in stability:
        # Multiclass — skip
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, stability["note"], ha="center", va="center")
        ax.set_title(title)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path))
        plt.close(fig)
        return save_path

    perturbations = stability.get("perturbations", [])
    base = stability.get("base", {})

    if not perturbations or not base:
        return save_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sort perturbations by threshold
    perturbations = sorted(perturbations, key=lambda p: p["threshold"])
    thresholds = [p["threshold"] for p in perturbations]
    sens = [p["sensitivity"] for p in perturbations]
    spec = [p["specificity"] for p in perturbations]
    ppv = [p["ppv"] for p in perturbations]
    npv = [p["npv"] for p in perturbations]

    base_t = base["threshold"]

    # Left: Sensitivity & Specificity vs threshold
    ax1.plot(thresholds, sens, "o-", color=COLORS["primary"], linewidth=1.5,
             markersize=4, label="Sensitivity")
    ax1.plot(thresholds, spec, "s-", color=COLORS["secondary"], linewidth=1.5,
             markersize=4, label="Specificity")
    ax1.axvline(base_t, color=COLORS["diagonal"], linestyle="--",
                linewidth=1, label=f"Base t={base_t:.3f}")
    ax1.axhline(base["sensitivity"], color=COLORS["primary"], linestyle=":",
                alpha=0.4, linewidth=0.8)
    ax1.axhline(base["specificity"], color=COLORS["secondary"], linestyle=":",
                alpha=0.4, linewidth=0.8)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Sens/Spec vs Threshold")
    ax1.legend(loc="best")
    ax1.set_ylim(-0.02, 1.05)

    # Right: PPV & NPV vs threshold
    ax2.plot(thresholds, ppv, "o-", color=COLORS["tertiary"], linewidth=1.5,
             markersize=4, label="PPV")
    ax2.plot(thresholds, npv, "s-", color="#7C3AED", linewidth=1.5,
             markersize=4, label="NPV")
    ax2.axvline(base_t, color=COLORS["diagonal"], linestyle="--",
                linewidth=1, label=f"Base t={base_t:.3f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("PPV/NPV vs Threshold")
    ax2.legend(loc="best")
    ax2.set_ylim(-0.02, 1.05)

    grade = stability.get("stability_grade", "?")
    fig.suptitle(f"{title} (Stability Grade: {grade})", fontsize=13, y=1.02)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    logger.info(f"Threshold stability → {save_path}")
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# Model Comparison
# ═════════════════════════════════════════════════════════════════════════════


def plot_model_comparison(
    comparison: dict,
    save_path: Path,
    title: str = "Final Model Comparison",
) -> Path:
    """Bar chart comparing key metrics across final models."""
    _apply_style()
    import matplotlib.pyplot as plt

    models = list(comparison.get("models", {}).keys())
    if not models:
        return save_path

    metrics_to_plot = [
        "auroc", "balanced_accuracy", "f1_macro", "kappa", "precision_macro", "recall_macro",
    ]

    # Collect data
    data = {}
    ci_lo = {}
    ci_hi = {}
    for metric in metrics_to_plot:
        data[metric] = []
        ci_lo[metric] = []
        ci_hi[metric] = []
        for model in models:
            m_data = comparison["models"][model].get("patient_level", {}).get(metric, {})
            point = m_data.get("point", 0) or 0
            lo = m_data.get("ci_lower") or point
            hi = m_data.get("ci_upper") or point
            data[metric].append(point)
            ci_lo[metric].append(point - lo)
            ci_hi[metric].append(hi - point)

    n_metrics = len(metrics_to_plot)
    n_models = len(models)
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))

    for i, model in enumerate(models):
        vals = [data[m][i] for m in metrics_to_plot]
        errs_lo = [ci_lo[m][i] for m in metrics_to_plot]
        errs_hi = [ci_hi[m][i] for m in metrics_to_plot]
        offset = x + (i - n_models / 2 + 0.5) * width

        ax.bar(offset, vals, width * 0.9,
               yerr=[errs_lo, errs_hi], capsize=3,
               color=colors[i % len(colors)], alpha=0.85,
               label=model, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    logger.info(f"Model comparison → {save_path}")
    return save_path