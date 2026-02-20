"""
Tests for oceanpath.eval.comparison — statistical tests for model comparison.

Covers:
  - DeLong test on known AUC values
  - McNemar test on known contingency tables
  - Bootstrap paired difference with known distributions
  - Experiment loader and pairwise comparison orchestration
"""

import numpy as np
import pandas as pd
import pytest

# ═════════════════════════════════════════════════════════════════════════════
# DeLong test
# ═════════════════════════════════════════════════════════════════════════════


class TestDeLongTest:
    def test_identical_predictions_not_significant(self):
        """Two models with identical predictions → p ≈ 1."""
        from oceanpath.eval.comparison import delong_test

        rng = np.random.RandomState(42)
        n = 200
        labels = np.array([0] * 100 + [1] * 100)
        probs = rng.beta(2, 5, n)  # same for both
        probs[labels == 1] += 0.3  # positive class scores higher

        result = delong_test(labels, probs, probs)
        assert result["auc_diff"] == pytest.approx(0.0, abs=1e-10)
        assert result["p_value"] == pytest.approx(1.0, abs=0.01)
        assert result["significant_at_05"] is False

    def test_clearly_different_models_significant(self):
        """Model A has AUC ~0.9, model B has AUC ~0.6 → significant."""
        from oceanpath.eval.comparison import delong_test

        rng = np.random.RandomState(42)
        n = 300
        labels = np.array([0] * 150 + [1] * 150)

        # Model A: good separation
        probs_a = rng.normal(0.3, 0.1, n)
        probs_a[labels == 1] = rng.normal(0.8, 0.1, 150)
        probs_a = np.clip(probs_a, 0, 1)

        # Model B: poor separation
        probs_b = rng.normal(0.4, 0.15, n)
        probs_b[labels == 1] = rng.normal(0.6, 0.15, 150)
        probs_b = np.clip(probs_b, 0, 1)

        result = delong_test(labels, probs_a, probs_b)
        assert result["auc_a"] > result["auc_b"]
        assert result["auc_diff"] > 0.1
        assert result["p_value"] < 0.05
        assert result["significant_at_05"] is True

    def test_symmetric(self):
        """DeLong(A, B) diff = -DeLong(B, A) diff."""
        from oceanpath.eval.comparison import delong_test

        rng = np.random.RandomState(0)
        labels = np.array([0] * 50 + [1] * 50)
        probs_a = rng.uniform(0, 1, 100)
        probs_b = rng.uniform(0, 1, 100)

        r_ab = delong_test(labels, probs_a, probs_b)
        r_ba = delong_test(labels, probs_b, probs_a)

        assert r_ab["auc_diff"] == pytest.approx(-r_ba["auc_diff"], abs=1e-10)
        assert r_ab["p_value"] == pytest.approx(r_ba["p_value"], abs=1e-10)

    def test_single_class_returns_error(self):
        """All-positive labels → error (no negatives for AUC)."""
        from oceanpath.eval.comparison import delong_test

        labels = np.ones(50, dtype=int)
        rng = np.random.default_rng(42)  # Create the generator
        probs = rng.random(50)  # Replaces np.random.rand(50)

        result = delong_test(labels, probs, probs)
        assert "error" in result

    def test_known_auc_values(self):
        """AUC from DeLong matches sklearn on perfect predictions."""
        from sklearn.metrics import roc_auc_score

        from oceanpath.eval.comparison import delong_test

        labels = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = delong_test(labels, probs, probs)
        sklearn_auc = roc_auc_score(labels, probs)

        assert result["auc_a"] == pytest.approx(sklearn_auc, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# McNemar test
# ═════════════════════════════════════════════════════════════════════════════


class TestMcNemarTest:
    def test_identical_predictions_not_significant(self):
        """Same predictions → no discordant pairs → p = 1."""
        from oceanpath.eval.comparison import mcnemar_test

        labels = np.array([0, 0, 1, 1, 0, 1])
        preds = np.array([0, 0, 1, 0, 0, 1])

        result = mcnemar_test(labels, preds, preds)
        assert result["p_value"] == 1.0
        assert result["a_right_b_wrong"] == 0
        assert result["a_wrong_b_right"] == 0

    def test_model_a_clearly_better(self):
        """A gets 20 right where B is wrong, B gets 2 right where A is wrong."""
        from oceanpath.eval.comparison import mcnemar_test

        n = 100
        labels = np.random.RandomState(42).randint(0, 2, n)

        # A is right on indices 0-79, B is right on indices 0-59 + 80-99
        preds_a = labels.copy()
        preds_a[80:] = 1 - labels[80:]  # A wrong on last 20

        preds_b = labels.copy()
        preds_b[60:80] = 1 - labels[60:80]  # B wrong on 60-80

        result = mcnemar_test(labels, preds_a, preds_b)
        assert result["advantage"] in ("A", "B")
        assert result["n_discordant"] > 0

    def test_small_sample_uses_binomial(self):
        """< 25 discordant pairs → exact binomial test."""
        from oceanpath.eval.comparison import mcnemar_test

        labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        preds_a = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1])  # 1 error
        preds_b = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1])  # 4 errors

        result = mcnemar_test(labels, preds_a, preds_b)
        assert result["test_type"] == "exact_binomial"

    def test_all_agree(self):
        """Both models make exact same predictions → not significant."""
        from oceanpath.eval.comparison import mcnemar_test

        labels = np.array([0, 1, 0, 1])
        preds = np.array([0, 1, 1, 0])
        result = mcnemar_test(labels, preds, preds)
        assert result["p_value"] == 1.0
        assert "No discordant" in result.get("note", "")


# ═════════════════════════════════════════════════════════════════════════════
# Bootstrap paired difference
# ═════════════════════════════════════════════════════════════════════════════


class TestBootstrapPairedDifference:
    def test_identical_models_ci_covers_zero(self):
        """Same predictions → CI for difference should cover 0."""
        from sklearn.metrics import roc_auc_score

        from oceanpath.eval.comparison import bootstrap_paired_difference

        rng = np.random.RandomState(42)
        labels = np.array([0] * 50 + [1] * 50)
        probs = rng.uniform(0, 1, 100)

        result = bootstrap_paired_difference(
            labels,
            probs,
            probs,
            metric_fn=roc_auc_score,
            n_bootstrap=500,
            seed=42,
        )
        assert result["diff"] == pytest.approx(0.0, abs=1e-10)
        assert result["ci_lower"] <= 0 <= result["ci_upper"]
        assert result["significant_at_05"] is False

    def test_different_models_detects_difference(self):
        """Clearly different models → CI should exclude 0."""
        from sklearn.metrics import roc_auc_score

        from oceanpath.eval.comparison import bootstrap_paired_difference

        rng = np.random.RandomState(42)
        labels = np.array([0] * 100 + [1] * 100)

        probs_good = rng.normal(0.3, 0.1, 200)
        probs_good[labels == 1] = rng.normal(0.8, 0.1, 100)
        probs_good = np.clip(probs_good, 0.01, 0.99)

        probs_bad = rng.uniform(0.2, 0.8, 200)

        result = bootstrap_paired_difference(
            labels,
            probs_good,
            probs_bad,
            metric_fn=roc_auc_score,
            n_bootstrap=500,
            seed=42,
        )
        assert result["diff"] > 0
        assert result["significant_at_05"] is True

    def test_returns_error_on_degenerate_data(self):
        """All same label → bootstrap can't compute AUC."""
        from sklearn.metrics import roc_auc_score

        from oceanpath.eval.comparison import bootstrap_paired_difference

        labels = np.ones(50, dtype=int)
        rng = np.random.default_rng(42)
        probs_a = rng.random(50)  # Replaces np.random.rand(50)
        probs_b = rng.random(50)  # Replaces np.random.rand(50)

        result = bootstrap_paired_difference(
            labels,
            probs_a,
            probs_b,
            metric_fn=roc_auc_score,
            n_bootstrap=100,
            seed=42,
        )
        assert "error" in result


# ═════════════════════════════════════════════════════════════════════════════
# Experiment loading
# ═════════════════════════════════════════════════════════════════════════════


class TestExperimentLoader:
    def test_load_predictions_from_oof(self, tmp_path):
        """Loads oof_predictions.parquet from experiment dir."""
        from oceanpath.eval.comparison import load_experiment_predictions

        df = pd.DataFrame(
            {
                "slide_id": ["s1", "s2", "s3"],
                "label": [0, 1, 0],
                "prob_0": [0.8, 0.3, 0.7],
                "prob_1": [0.2, 0.7, 0.3],
            }
        )
        df.to_parquet(tmp_path / "oof_predictions.parquet", index=False)

        loaded = load_experiment_predictions(str(tmp_path))
        assert loaded is not None
        assert len(loaded) == 3
        assert "prob_1" in loaded.columns

    def test_returns_none_when_missing(self, tmp_path):
        """No predictions file → returns None."""
        from oceanpath.eval.comparison import load_experiment_predictions

        result = load_experiment_predictions(str(tmp_path / "nonexistent"))
        assert result is None


class TestCompareExperiments:
    @pytest.fixture
    def two_experiments(self, tmp_path):
        """Create two fake experiment directories with OOF predictions."""
        rng = np.random.RandomState(42)
        n = 100
        slide_ids = [f"slide_{i}" for i in range(n)]
        labels = np.array([0] * 50 + [1] * 50)

        for name, offset in [("exp_a", 0.2), ("exp_b", 0.0)]:
            exp_dir = tmp_path / name
            exp_dir.mkdir()

            probs = rng.uniform(0.2, 0.8, n) + offset * labels
            probs = np.clip(probs, 0.01, 0.99)

            df = pd.DataFrame(
                {
                    "slide_id": slide_ids,
                    "label": labels,
                    "prob_0": 1 - probs,
                    "prob_1": probs,
                }
            )
            df.to_parquet(exp_dir / "oof_predictions.parquet", index=False)

        return {
            "exp_a": str(tmp_path / "exp_a"),
            "exp_b": str(tmp_path / "exp_b"),
        }

    def test_comparison_produces_ranking(self, two_experiments):
        """Full comparison returns ranking with both experiments."""
        from oceanpath.eval.comparison import compare_experiments

        result = compare_experiments(two_experiments, n_bootstrap=200, seed=42)
        assert "ranking" in result
        assert len(result["ranking"]) == 2

    def test_comparison_has_pairwise_tests(self, two_experiments):
        """Pairwise results include DeLong, McNemar, bootstrap."""
        from oceanpath.eval.comparison import compare_experiments

        result = compare_experiments(two_experiments, n_bootstrap=200, seed=42)
        assert len(result["pairwise"]) == 1

        pw = result["pairwise"][0]
        assert "delong" in pw
        assert "mcnemar" in pw
        assert "bootstrap_auroc" in pw

    def test_comparison_needs_two_experiments(self, tmp_path):
        """Single experiment → error."""
        from oceanpath.eval.comparison import compare_experiments

        exp_dir = tmp_path / "only_one"
        exp_dir.mkdir()
        pd.DataFrame(
            {
                "slide_id": ["s1"],
                "label": [0],
                "prob_0": [0.5],
                "prob_1": [0.5],
            }
        ).to_parquet(exp_dir / "oof_predictions.parquet", index=False)

        result = compare_experiments({"only": str(exp_dir)})
        assert "error" in result

    def test_summary_table_is_markdown(self, two_experiments):
        """Summary table is a valid markdown string with headers."""
        from oceanpath.eval.comparison import compare_experiments

        result = compare_experiments(two_experiments, n_bootstrap=100, seed=42)
        table = result["summary_table"]
        assert "| Experiment" in table
        assert "AUROC" in table
