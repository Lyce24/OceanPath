"""
OceanPath evaluation module (Stage 6).

Comprehensive evaluation of MIL classification models:
  - OOF and per-fold metrics from CV
  - Final model evaluation with bootstrap CIs
  - Calibration, operating points, threshold stability
  - Model comparison and selection
"""

from oceanpath.eval.core import (
    MetricsSuite,
    compute_metrics,
    compute_metrics_with_ci,
    compute_calibration,
    compute_operating_points,
    compute_threshold_stability,
    compute_pr_curve,
    aggregate_to_patient_level,
    extract_probs_and_labels,
    bootstrap_ci,
)
from oceanpath.eval.compare import compare_models