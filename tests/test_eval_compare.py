"""
Tests for oceanpath.eval.compare — choosing among the final models
(best_fold, ensemble, refit) of one experiment.

Model results have the shape scripts/evaluate.py builds: compute_metrics_with_ci
output under "performance", compute_calibration under "calibration",
compute_threshold_stability under "threshold_stability".

Covers:
  - the composite score against hand-computed values (default weights)
  - ranking and recommendation, custom weights, slide-level primary metric
  - conservative scores for missing fields; full credit for an ECE of exactly 0
  - the rationale text and the two JSON files downstream stages read
    (model_comparison.json, recommended_model.json), including numpy scalars
  - end to end on predictions scored by oceanpath.eval.core
"""

import json

import numpy as np
import pandas as pd
import pytest

from oceanpath.eval.compare import compare_models, save_comparison, save_recommendation
from oceanpath.eval.core import (
    compute_calibration,
    compute_metrics_with_ci,
    compute_threshold_stability,
    extract_probs_and_labels,
)

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _result(
    auroc=None,
    ci=(None, None),
    ece=None,
    grade=None,
    kappa=None,
    bacc=None,
    slide_auroc=None,
) -> dict:
    """One model's evaluation result with the fields compare_models reads."""

    def level(point):
        return {
            "auroc": {"point": point, "ci_lower": ci[0], "ci_upper": ci[1]},
            "kappa": {"point": kappa},
            "balanced_accuracy": {"point": bacc},
        }

    return {
        "performance": {"slide_level": level(slide_auroc), "patient_level": level(auroc)},
        "calibration": {"ece": ece, "brier": None},
        "threshold_stability": {"stability_grade": grade},
    }


# Hand-computed composites with the default weights
# (.50 AUROC, .15 calibration, .15 stability, .10 CI width, .10 secondary):
#   best_fold: .5*.90 + .15*(1-.05) + .15*1.0 + .10*(1-.10) + .10*(.6+.8)/2  = .9025
#   ensemble:  .5*.92 + .15*(1-.20) + .15*0.5 + .10*(1-.19) + .10*(.5+.75)/2 = .7985
#   refit:     .5*.85 + .15*(1-.02) + .15*1.0 + .10*(1-.06) + .10*(.55+.78)/2 = .8825
MODELS = {
    "best_fold": _result(0.90, (0.85, 0.95), ece=0.05, grade="A", kappa=0.6, bacc=0.8),
    "ensemble": _result(0.92, (0.80, 0.99), ece=0.20, grade="C", kappa=0.5, bacc=0.75),
    "refit": _result(0.85, (0.82, 0.88), ece=0.02, grade="A", kappa=0.55, bacc=0.78),
}
EXPECTED_COMPOSITE = {"best_fold": 0.9025, "ensemble": 0.7985, "refit": 0.8825}


# ═════════════════════════════════════════════════════════════════════════════
# Scoring and ranking
# ═════════════════════════════════════════════════════════════════════════════


class TestCompositeScore:
    def test_component_scores_of_one_model(self):
        detail = compare_models(MODELS)["scores_detail"]["best_fold"]
        assert detail == {
            "auroc": pytest.approx(0.90),
            "calibration": pytest.approx(0.95),
            "stability": pytest.approx(1.0),
            "ci_width": pytest.approx(0.90),
            "secondary": pytest.approx(0.70),
            "composite": pytest.approx(0.9025),
        }

    def test_ranking_follows_the_composite_not_the_auroc(self):
        """The ensemble has the best AUROC but the worst calibration, stability
        and interval width, so it ranks last."""
        comparison = compare_models(MODELS)
        assert [r["model"] for r in comparison["ranking"]] == ["best_fold", "refit", "ensemble"]
        auroc = {"best_fold": 0.90, "ensemble": 0.92, "refit": 0.85}
        for row in comparison["ranking"]:
            assert row["composite_score"] == pytest.approx(EXPECTED_COMPOSITE[row["model"]])
            assert row["auroc"] == auroc[row["model"]]
        assert comparison["recommended"] == "best_fold"

    def test_weights_decide_the_winner(self):
        auroc_only = {"primary_metric": 1, "calibration": 0, "stability": 0, "ci_width": 0}
        comparison = compare_models(MODELS, weights={**auroc_only, "secondary": 0})
        assert comparison["recommended"] == "ensemble"
        assert comparison["ranking"][0]["composite_score"] == pytest.approx(0.92)

    def test_partial_weights_keep_the_other_defaults(self):
        comparison = compare_models(MODELS, weights={"calibration": 0.5})
        assert comparison["weights_used"] == {
            "primary_metric": 0.50,
            "calibration": 0.5,
            "stability": 0.15,
            "ci_width": 0.10,
            "secondary": 0.10,
        }
        # best_fold: .9025 - .15*.95 + .5*.95
        assert comparison["scores_detail"]["best_fold"]["composite"] == pytest.approx(1.235)

    def test_slide_level_primary_metric(self):
        models = {
            "a": _result(auroc=0.95, slide_auroc=0.70, grade="A"),
            "b": _result(auroc=0.75, slide_auroc=0.90, grade="A"),
        }
        assert compare_models(models)["recommended"] == "a"
        by_slide = compare_models(models, primary_level="slide_level")
        assert by_slide["recommended"] == "b"
        assert by_slide["scores_detail"]["b"]["auroc"] == 0.90

    def test_perfect_calibration_earns_full_calibration_credit(self):
        """ECE 0.0 is the best possible calibration, not a missing value."""
        models = {
            "perfect": _result(0.9, ece=0.0, grade="A"),
            "slightly_off": _result(0.9, ece=0.01, grade="A"),
        }
        comparison = compare_models(models)
        assert comparison["scores_detail"]["perfect"]["calibration"] == 1.0
        assert comparison["recommended"] == "perfect"

    def test_missing_fields_get_conservative_scores(self):
        """No ECE -> worst calibration, no grade -> grade D, no interval ->
        neutral 0.5, no secondary metrics -> 0."""
        detail = compare_models({"bare": _result(auroc=0.8)})["scores_detail"]["bare"]
        assert detail["calibration"] == 0.0
        assert detail["stability"] == 0.25
        assert detail["ci_width"] == 0.5
        assert detail["secondary"] == 0.0
        assert detail["composite"] == pytest.approx(0.5 * 0.8 + 0.15 * 0.25 + 0.10 * 0.5)

    def test_summary_carries_interval_width_and_calibration(self):
        summary = compare_models(MODELS)["models"]["refit"]
        assert summary["patient_level"]["auroc"]["ci_width"] == pytest.approx(0.06)
        assert summary["patient_level"]["accuracy"] == {
            "point": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_width": None,
        }
        assert (summary["ece"], summary["stability_grade"]) == (0.02, "A")

    def test_no_models_is_an_error(self):
        assert "error" in compare_models({})

    def test_rationale_names_the_winner_and_its_numbers(self):
        rationale = compare_models(MODELS)["rationale"].splitlines()
        assert rationale[0] == "Recommended model: best_fold"
        assert "auroc (patient_level): 0.9000 (95% CI: [0.850, 0.950])" in rationale[1]
        assert rationale[2].endswith("Calibration (ECE): 0.0500")
        assert rationale[3].endswith("Threshold stability grade: A")
        assert rationale[4].endswith("Composite score: 0.9025")


# ═════════════════════════════════════════════════════════════════════════════
# Saved files
# ═════════════════════════════════════════════════════════════════════════════


class TestSavedFiles:
    def test_comparison_and_recommendation_round_trip(self, tmp_path):
        """numpy scalars (as the metric code produces them) serialise, and
        recommended_model.json holds what export_model.py / analyze.py read."""
        models = {
            "best_fold": _result(
                np.float32(0.9), (np.float32(0.85), np.float32(0.95)), ece=np.float32(0.05)
            ),
            "refit": _result(0.8, ece=0.1, grade="B"),
        }
        comparison = compare_models(models)
        comp_path = tmp_path / "eval" / "model_comparison.json"
        rec_path = tmp_path / "eval" / "nested" / "recommended_model.json"
        save_comparison(comparison, comp_path)
        save_recommendation(comparison, rec_path)

        saved = json.loads(comp_path.read_text())
        assert saved["recommended"] == "best_fold"
        assert saved["models"]["best_fold"]["patient_level"]["auroc"]["point"] == pytest.approx(0.9)
        assert [r["model"] for r in saved["ranking"]] == ["best_fold", "refit"]

        rec = json.loads(rec_path.read_text())
        assert rec["recommended_model"] == "best_fold"
        assert rec["rationale"] == comparison["rationale"]
        assert rec["weights_used"] == comparison["weights_used"]
        assert rec["composite_scores"]["refit"]["composite"] == pytest.approx(
            comparison["scores_detail"]["refit"]["composite"]
        )


# ═════════════════════════════════════════════════════════════════════════════
# End to end on scored predictions
# ═════════════════════════════════════════════════════════════════════════════


def _evaluate(preds: pd.DataFrame) -> dict:
    """What scripts/evaluate.py stores per final model (without plots)."""
    labels, probs = extract_probs_and_labels(preds)
    return {
        "performance": compute_metrics_with_ci(preds, n_bootstrap=10, seed=0),
        "calibration": compute_calibration(labels, probs),
        "threshold_stability": compute_threshold_stability(labels, probs, 0.5),
    }


def test_informative_model_beats_a_noisy_one():
    rng = np.random.default_rng(0)
    n_patients = 20
    labels = np.repeat(np.arange(n_patients) % 2, 2)
    base = pd.DataFrame(
        {
            "slide_id": [f"s{i}" for i in range(2 * n_patients)],
            "patient_id": np.repeat([f"p{i}" for i in range(n_patients)], 2),
            "label": labels,
        }
    )
    informative = base.assign(prob_1=np.clip(0.2 + 0.6 * labels + rng.normal(0, 0.1, 40), 0, 1))
    noisy = base.assign(prob_1=rng.uniform(0.3, 0.7, 40))

    comparison = compare_models({"noisy": _evaluate(noisy), "informative": _evaluate(informative)})
    assert comparison["recommended"] == "informative"
    scores = comparison["scores_detail"]
    assert scores["informative"]["auroc"] > 0.95 > 0.75 > scores["noisy"]["auroc"]
    assert scores["informative"]["composite"] > scores["noisy"]["composite"]
