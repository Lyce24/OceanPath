"""
Model comparison and selection logic for Stage 6.

Compares best_fold, ensemble, and refit on multiple axes:
  - Patient-level AUROC (primary)
  - Calibration (ECE)
  - Threshold stability
  - Bootstrap CI width (tighter = more reliable)

Produces a recommended_model with rationale.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default weights if none provided from config
DEFAULT_WEIGHTS = {
    "primary_metric": 0.50,
    "calibration": 0.15,
    "stability": 0.15,
    "ci_width": 0.10,
    "secondary": 0.10,
}


def compare_models(
    model_results: dict[str, dict],
    primary_metric: str = "auroc",
    primary_level: str = "patient_level",
    weights: dict | None = None,
) -> dict:
    """
    Compare final models and select the best one.

    Parameters
    ----------
    model_results : {model_name: {performance, calibration, threshold_stability, ...}}
    primary_metric : metric to rank by (default: auroc)
    primary_level : "patient_level" or "slide_level"
    weights : composite score weights from config (see DEFAULT_WEIGHTS)

    Returns
    -------
    dict with:
      models: {name: {summary metrics}}
      ranking: ordered list of (name, score)
      recommended: name of best model
      rationale: explanation string
    """
    if not model_results:
        return {"error": "No models to compare"}

    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    comparison = {"models": {}, "ranking": [], "scores_detail": {}, "weights_used": w}

    for name, result in model_results.items():
        perf = result.get("performance", {})
        cal = result.get("calibration", {})
        stab = result.get("threshold_stability", {})

        # Extract key metrics
        model_summary = {}

        # Performance at both levels
        for level in ["slide_level", "patient_level"]:
            level_data = perf.get(level, {})
            model_summary[level] = {}
            for metric_name in [
                "auroc",
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "kappa",
                "precision_macro",
                "recall_macro",
            ]:
                m = level_data.get(metric_name, {})
                model_summary[level][metric_name] = {
                    "point": m.get("point"),
                    "ci_lower": m.get("ci_lower"),
                    "ci_upper": m.get("ci_upper"),
                    "ci_width": (
                        (m.get("ci_upper", 0) or 0) - (m.get("ci_lower", 0) or 0)
                        if m.get("ci_lower") is not None
                        else None
                    ),
                }

        # Calibration
        model_summary["ece"] = cal.get("ece")
        model_summary["brier"] = cal.get("brier")

        # Stability
        model_summary["stability_grade"] = stab.get("stability_grade")
        model_summary["max_sens_change_01"] = stab.get("max_sens_change_01")
        model_summary["max_sens_change_05"] = stab.get("max_sens_change_05")

        comparison["models"][name] = model_summary

    # ── Scoring ──────────────────────────────────────────────────────────
    scores = {}
    for name, summary in comparison["models"].items():
        s = _compute_composite_score(summary, primary_metric, primary_level, w)
        scores[name] = s
        comparison["scores_detail"][name] = s

    # Rank by composite score (higher = better)
    ranking = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)
    comparison["ranking"] = [
        {
            "model": name,
            "composite_score": s["composite"],
            f"{primary_metric}": s.get(primary_metric, 0),
        }
        for name, s in ranking
    ]

    # Recommend best
    best_name = ranking[0][0]
    best_scores = ranking[0][1]

    comparison["recommended"] = best_name
    comparison["rationale"] = _build_rationale(
        best_name,
        best_scores,
        comparison["models"][best_name],
        primary_metric,
        primary_level,
    )

    return comparison


def _compute_composite_score(
    summary: dict,
    primary_metric: str,
    primary_level: str,
    weights: dict,
) -> dict:
    """
    Compute composite score for model ranking.

    Weights come from config (eval.comparison_weights).
    """
    scores = {}

    # Primary metric
    primary_data = summary.get(primary_level, {}).get(primary_metric, {})
    primary_val = primary_data.get("point") or 0.0
    scores[primary_metric] = primary_val

    # Calibration score (1 - ECE, clamped)
    ece = summary.get("ece")
    cal_score = max(0.0, 1.0 - (ece or 1.0))
    scores["calibration"] = cal_score

    # Stability score
    grade = summary.get("stability_grade", "D")
    grade_map = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}
    stab_score = grade_map.get(grade, 0.25)
    scores["stability"] = stab_score

    # CI width score (narrower = better, normalize to 0-1)
    ci_width = primary_data.get("ci_width")
    if ci_width is not None and ci_width > 0:
        ci_score = max(0.0, 1.0 - ci_width)  # width of 0.1 → score 0.9
    else:
        ci_score = 0.5  # neutral if unknown
    scores["ci_width"] = ci_score

    # Secondary metrics
    kappa = summary.get(primary_level, {}).get("kappa", {}).get("point") or 0.0
    bacc = summary.get(primary_level, {}).get("balanced_accuracy", {}).get("point") or 0.0
    secondary = (kappa + bacc) / 2
    scores["secondary"] = secondary

    # Composite
    w = weights
    scores["composite"] = (
        w.get("primary_metric", 0.50) * primary_val
        + w.get("calibration", 0.15) * cal_score
        + w.get("stability", 0.15) * stab_score
        + w.get("ci_width", 0.10) * ci_score
        + w.get("secondary", 0.10) * secondary
    )

    return scores


def _build_rationale(
    name: str,
    scores: dict,
    summary: dict,
    primary_metric: str,
    primary_level: str,
) -> str:
    """Build human-readable rationale for model selection."""
    primary_val = scores.get(primary_metric, 0)
    primary_data = summary.get(primary_level, {}).get(primary_metric, {})
    ci_lo = primary_data.get("ci_lower")
    ci_hi = primary_data.get("ci_upper")
    ci_str = f" (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])" if ci_lo is not None else ""

    grade = summary.get("stability_grade", "?")
    ece = summary.get("ece", "?")
    ece_str = f"{ece:.4f}" if isinstance(ece, float) else str(ece)

    parts = [
        f"Recommended model: {name}",
        f"  {primary_metric} ({primary_level}): {primary_val:.4f}{ci_str}",
        f"  Calibration (ECE): {ece_str}",
        f"  Threshold stability grade: {grade}",
        f"  Composite score: {scores['composite']:.4f}",
    ]
    return "\n".join(parts)


def save_comparison(comparison: dict, save_path: Path) -> None:
    """Save comparison results to JSON."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(comparison, indent=2, default=_json_default))
    logger.info(f"Model comparison → {save_path}")


def save_recommendation(comparison: dict, save_path: Path) -> None:
    """Save recommended model selection."""
    rec = {
        "recommended_model": comparison.get("recommended"),
        "rationale": comparison.get("rationale"),
        "ranking": comparison.get("ranking"),
        "composite_scores": comparison.get("scores_detail"),
        "weights_used": comparison.get("weights_used"),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(rec, indent=2, default=_json_default))
    logger.info(f"Recommendation → {save_path}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
