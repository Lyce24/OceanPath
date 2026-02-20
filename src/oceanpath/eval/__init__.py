"""
OceanPath evaluation module (Stage 6).

Comprehensive evaluation of MIL classification models:
  - OOF and per-fold metrics from CV
  - Final model evaluation with bootstrap CIs
  - Calibration, operating points, threshold stability
  - Model comparison and selection
"""

from oceanpath.eval.compare import compare_models
from oceanpath.eval.core import (
    MetricsSuite,
    aggregate_to_patient_level,
    bootstrap_ci,
    compute_calibration,
    compute_metrics,
    compute_metrics_with_ci,
    compute_operating_points,
    compute_pr_curve,
    compute_threshold_stability,
    extract_probs_and_labels,
)

__all__ = [
    "MetricsSuite",
    "aggregate_to_patient_level",
    "bootstrap_ci",
    "compare_models",
    "compute_calibration",
    "compute_metrics",
    "compute_metrics_with_ci",
    "compute_operating_points",
    "compute_pr_curve",
    "compute_threshold_stability",
    "extract_probs_and_labels",
]
