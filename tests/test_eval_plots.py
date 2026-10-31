"""
Tests for oceanpath.eval.plots — the Stage 6 figures.

Each figure is drawn from what oceanpath.eval.core actually returns for a
binary and a three-class prediction set, as scripts/evaluate.py does, and must
land as a valid, non-blank PNG at the requested path (parent directories are
created) with that path returned for logging.

Covers: ROC (binary, one-vs-rest, no data), PR, calibration, confusion matrix
(normalised and raw counts, a class with no slides), threshold stability
(binary and the multiclass note), and the final-model comparison bars.
"""

import matplotlib
import numpy as np
import pytest
from PIL import Image

from oceanpath.eval.compare import compare_models
from oceanpath.eval.core import (
    compute_calibration,
    compute_metrics,
    compute_operating_points,
    compute_pr_curve,
    compute_threshold_stability,
)
from oceanpath.eval.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_stability,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _low_dpi(monkeypatch):
    """Render small images; resolution is not what these tests check."""
    monkeypatch.setitem(matplotlib.rcParams, "savefig.dpi", 50)


def assert_png(path):
    """A readable PNG with something drawn on it (not one flat colour)."""
    assert path.is_file()
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        assert img.width > 50 and img.height > 50
        extrema = img.convert("L").getextrema()
        assert extrema[0] < extrema[1], "blank image"


def _case(n_classes: int) -> dict:
    """Core outputs for one synthetic prediction set."""
    rng = np.random.default_rng(n_classes)
    n = 60
    labels = np.arange(n) % n_classes
    if n_classes == 2:
        probs = np.clip(0.3 + 0.4 * labels + rng.normal(0, 0.15, n), 0, 1)
    else:
        probs = rng.dirichlet(np.ones(n_classes), n)
        probs[np.arange(n), labels] += 0.5
        probs /= probs.sum(1, keepdims=True)
    metrics = compute_metrics(labels, probs)
    return {
        "metrics": metrics,
        "operating_points": compute_operating_points(labels, probs),
        "pr": compute_pr_curve(labels, probs),
        "calibration": compute_calibration(labels, probs),
        "stability": compute_threshold_stability(labels, probs, 0.5),
    }


@pytest.fixture(scope="module", params=[2, 3], ids=["binary", "multiclass"])
def case(request):
    return _case(request.param)


# ═════════════════════════════════════════════════════════════════════════════
# Per-model figures
# ═════════════════════════════════════════════════════════════════════════════


class TestPerModelFigures:
    def test_roc_curve(self, case, tmp_path):
        path = tmp_path / "plots" / "roc_curve.png"
        out = plot_roc_curve(case["operating_points"], case["metrics"].auroc_macro, path)
        assert out == path
        assert_png(path)

    def test_pr_curve(self, case, tmp_path):
        path = tmp_path / "plots" / "pr_curve.png"
        assert plot_pr_curve(case["pr"], path) == path
        assert_png(path)

    def test_calibration_curve(self, case, tmp_path):
        path = tmp_path / "plots" / "calibration_curve.png"
        assert plot_calibration_curve(case["calibration"], path) == path
        assert_png(path)

    def test_confusion_matrix(self, case, tmp_path):
        cm = case["metrics"].confusion_matrix
        path = tmp_path / "plots" / "confusion_matrix.png"
        names = [f"class {c}" for c in range(len(cm))]
        assert plot_confusion_matrix(cm, path, class_names=names) == path
        assert_png(path)

    def test_threshold_stability(self, case, tmp_path):
        """Binary: sens/spec and PPV/NPV panels; multiclass: a figure with the
        'binary only' note, so the plots directory has the same files."""
        path = tmp_path / "plots" / "threshold_stability.png"
        assert ("note" in case["stability"]) == (case["metrics"].n_classes > 2)
        assert plot_threshold_stability(case["stability"], path) == path
        assert_png(path)
        with Image.open(path) as img:
            if case["metrics"].n_classes == 2:
                assert img.width > 1.8 * img.height  # two panels side by side


class TestEdgeInputs:
    def test_roc_without_curve_data_draws_a_placeholder(self, tmp_path):
        path = tmp_path / "roc.png"
        plot_roc_curve({}, 0.0, path)
        assert_png(path)

    def test_roc_skips_a_class_that_has_no_slides(self, tmp_path):
        labels = np.array([0, 1, 0, 1, 0, 1])
        probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]] * 3)
        op = compute_operating_points(labels, probs)
        assert "error" in op["per_class"]["2"]
        path = tmp_path / "roc.png"
        plot_roc_curve(op, 1.0, path)
        assert_png(path)

    def test_raw_count_confusion_matrix_with_an_empty_row(self, tmp_path):
        """A class with no slides (all-zero row) must not divide by zero."""
        path = tmp_path / "cm.png"
        plot_confusion_matrix([[5, 1, 0], [0, 0, 0], [2, 0, 7]], path, normalize=False)
        assert_png(path)
        plot_confusion_matrix([[5, 1, 0], [0, 0, 0], [2, 0, 7]], path, normalize=True)
        assert_png(path)


# ═════════════════════════════════════════════════════════════════════════════
# Model comparison
# ═════════════════════════════════════════════════════════════════════════════


def test_model_comparison_bars(tmp_path):
    """Four models (more than the three-colour palette), with and without CIs,
    and with metrics missing from the result."""

    def result(auroc, half_width):
        level = {}
        for i, name in enumerate(["auroc", "balanced_accuracy", "f1_macro", "kappa"]):
            point = auroc - 0.1 * i
            ci = (point - half_width, point + half_width) if half_width else (None, None)
            level[name] = {"point": point, "ci_lower": ci[0], "ci_upper": ci[1]}
        return {"performance": {"patient_level": level, "slide_level": level}}

    comparison = compare_models(
        {
            "best_fold": result(0.9, 0.05),
            "ensemble": result(0.85, None),
            "refit": result(0.8, 0.1),
            "extra": result(0.7, 0.02),
        }
    )
    path = tmp_path / "final_models" / "model_comparison.png"
    assert plot_model_comparison(comparison, path) == path
    assert_png(path)
