"""
Tests for oceanpath.eval.core — the Stage 6 metric engine.

Expected values come from hand-worked small cases (the arithmetic is in the
comments) or from sklearn / brute force on larger random cases, never from the
implementation's own formulas.

Covers:
  - compute_metrics: binary and multiclass hand cases, per-class specificity,
    renormalisation of rows that do not sum to 1, one class present, empty input
  - aggregate_to_patient_level / extract_probs_and_labels: mean probabilities,
    majority label, slide-count column, slide fallback, class-index column order
  - bootstrap_ci: interval brackets the point estimate, seed reproducibility,
    nested intervals across alpha, patients resampled as whole units, degenerate data
  - compute_metrics_with_ci: slide and patient points against hand values
  - compute_calibration: zero ECE when calibrated, the gap when over-confident,
    binary and multiclass Brier, bin bookkeeping
  - compute_operating_points: hand-worked ROC, targets met when applied, OVR multiclass
  - compute_threshold_stability: hand-worked perturbations, clamping, grades
  - compute_pr_curve: endpoints, average precision, multiclass per class
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from oceanpath.eval.core import (
    _stability_grade,
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

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _accuracy(labels: np.ndarray, probs: np.ndarray) -> float:
    """Binary accuracy at 0.5: a cheap metric_fn for bootstrap tests."""
    return float(np.mean((probs >= 0.5).astype(int) == labels))


def _rank_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    """AUROC as the fraction of positive/negative pairs ranked correctly (ties 1/2)."""
    pos, neg = probs[labels == 1][:, None], probs[labels == 0][None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def _signal_data(n: int = 200, seed: int = 0):
    """Binary labels with scores that separate them imperfectly."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    probs = 1 / (1 + np.exp(-(2.0 * labels - 1.0 + rng.normal(0, 1.2, n))))
    return labels, probs


def _brute_force_rates(labels: np.ndarray, probs: np.ndarray, threshold: float):
    pred = probs >= threshold
    sens = pred[labels == 1].mean()
    spec = (~pred[labels == 0]).mean()
    return sens, spec


# ═════════════════════════════════════════════════════════════════════════════
# compute_metrics
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeMetrics:
    def test_binary_hand_case(self):
        """labels [0,0,1,1], P(1) [.1,.6,.4,.9] -> predictions [0,1,0,1].

        Every class has one hit and one miss; 3 of the 4 positive/negative
        pairs are ordered correctly (only .4 < .6 is not), so AUROC = 0.75.
        """
        m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]))
        assert (m.n_samples, m.n_classes) == (4, 2)
        assert m.accuracy == pytest.approx(0.5)
        assert m.balanced_accuracy == pytest.approx(0.5)
        assert m.auroc_macro == pytest.approx(0.75)
        assert m.auroc_per_class == {"0": pytest.approx(0.75), "1": pytest.approx(0.75)}
        assert m.confusion_matrix == [[1, 1], [1, 1]]
        assert m.specificity_per_class == {"0": 0.5, "1": 0.5}
        assert m.kappa == pytest.approx(0.0)
        # Log loss: the true class gets .9, .4, .4, .9
        assert m.log_loss_val == pytest.approx(-(np.log(0.9) + np.log(0.4)) / 2)

    def test_multiclass_hand_case(self):
        """Four slides, three classes; predictions (argmax) are [0, 1, 2, 2]."""
        labels = np.array([0, 1, 1, 2])
        probs = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]],
        )
        m = compute_metrics(labels, probs)
        assert m.n_classes == 3
        assert m.accuracy == pytest.approx(0.75)
        assert m.confusion_matrix == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
        # Recall per class: 1/1, 1/2, 1/1 -> balanced accuracy 5/6
        assert m.recall_per_class == {"0": 1.0, "1": 0.5, "2": 1.0}
        assert m.balanced_accuracy == pytest.approx(5 / 6)
        # Specificity of class 2: of the 3 non-class-2 slides one is called 2
        assert m.specificity_per_class == {
            "0": pytest.approx(1.0),
            "1": pytest.approx(1.0),
            "2": pytest.approx(2 / 3),
        }
        # One-vs-rest AUROC. Class 1: positives {.8, .3} vs negatives {.2, .3}
        # -> 3 wins and one tie = 3.5 / 4. Class 2: .4 beats .1 and .1, loses to .5.
        assert m.auroc_per_class == {
            "0": pytest.approx(1.0),
            "1": pytest.approx(0.875),
            "2": pytest.approx(2 / 3),
        }
        assert m.auroc_macro == pytest.approx((1.0 + 0.875 + 2 / 3) / 3)

    def test_multiclass_agrees_with_sklearn_on_random_data(self):
        from sklearn.metrics import (
            balanced_accuracy_score,
            cohen_kappa_score,
            f1_score,
            log_loss,
        )

        rng = np.random.default_rng(1)
        labels = rng.integers(0, 4, 80)
        probs = rng.dirichlet(np.ones(4), 80)
        probs[np.arange(80), labels] += 0.3  # some signal
        probs /= probs.sum(1, keepdims=True)

        m = compute_metrics(labels, probs)
        preds = probs.argmax(1)
        assert m.auroc_macro == pytest.approx(roc_auc_score(labels, probs, multi_class="ovr"))
        assert m.balanced_accuracy == pytest.approx(balanced_accuracy_score(labels, preds))
        assert m.f1_macro == pytest.approx(f1_score(labels, preds, average="macro"))
        assert m.kappa == pytest.approx(cohen_kappa_score(labels, preds))
        assert m.log_loss_val == pytest.approx(log_loss(labels, probs))

    def test_rows_not_summing_to_one_are_renormalised(self):
        """AMP float16 softmax drift (rows summing to 0.99998 / 1.00003) does
        not change any metric."""
        labels = np.array([0, 1, 2, 1, 0, 2])
        probs = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.2, 0.7],
                [0.4, 0.35, 0.25],
                [0.5, 0.25, 0.25],
                [0.3, 0.3, 0.4],
            ]
        )
        drift = np.array([0.99998, 1.00003, 0.99999, 1.00002, 1.00001, 0.99997])[:, None]
        clean = compute_metrics(labels, probs).to_dict()
        drifted = compute_metrics(labels, probs * drift).to_dict()
        for key in ("auroc_macro", "accuracy", "f1_macro", "log_loss_val"):
            assert drifted[key] == pytest.approx(clean[key], abs=1e-9)
        assert drifted["confusion_matrix"] == clean["confusion_matrix"]

    def test_one_class_present_leaves_auroc_undefined(self):
        m = compute_metrics(np.array([0, 0, 0]), np.array([0.1, 0.6, 0.4]))
        assert np.isnan(m.auroc_macro)
        assert m.auroc_per_class == {"0": None, "1": None}
        assert m.accuracy == pytest.approx(2 / 3)
        d = m.to_dict()
        assert d["auroc_macro"] is None
        json.dumps(d, allow_nan=False)  # strict JSON: NaN became null

    def test_multiclass_with_an_absent_class(self):
        """No slide of class 2: its AUROC and the macro AUROC are undefined,
        the present classes still get theirs."""
        labels = np.array([0, 1, 1, 1])
        probs = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]],
        )
        m = compute_metrics(labels, probs)
        assert np.isnan(m.auroc_macro)
        assert m.auroc_per_class["2"] is None
        assert m.auroc_per_class["0"] == pytest.approx(1.0)
        assert m.auroc_per_class["1"] == pytest.approx(1.0)

    def test_empty_input_gives_an_empty_suite(self):
        m = compute_metrics(np.array([], dtype=int), np.array([]), level="patient")
        d = m.to_dict()
        assert d["level"] == "patient"
        assert d["n_samples"] == 0
        assert d["accuracy"] is None and d["auroc_macro"] is None


# ═════════════════════════════════════════════════════════════════════════════
# Patient aggregation and probability extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestPatientAggregation:
    def test_mean_probability_majority_label_and_slide_count(self):
        df = pd.DataFrame(
            {
                "slide_id": ["a1", "a2", "b1", "c1", "c2", "c3"],
                "patient_id": ["A", "A", "B", "C", "C", "C"],
                "label": [1, 1, 0, 0, 0, 1],
                "prob_1": [0.2, 0.4, 0.9, 0.1, 0.3, 0.8],
            }
        )
        out = aggregate_to_patient_level(df).set_index("patient_id")
        assert sorted(out.index) == ["A", "B", "C"]
        assert out.loc["A", "prob_1"] == pytest.approx(0.3)
        assert out.loc["B", "prob_1"] == pytest.approx(0.9)
        assert out.loc["C", "prob_1"] == pytest.approx(0.4)
        assert out["label"].to_dict() == {"A": 1, "B": 0, "C": 0}  # C: two of three are 0
        assert out["n_slides"].to_dict() == {"A": 2, "B": 1, "C": 3}

    def test_every_probability_column_is_averaged(self):
        df = pd.DataFrame(
            {
                "slide_id": ["s1", "s2", "s3"],
                "patient_id": ["P", "P", "Q"],
                "label": [2, 2, 0],
                "prob_0": [0.2, 0.0, 0.7],
                "prob_1": [0.2, 0.4, 0.2],
                "prob_2": [0.6, 0.6, 0.1],
            }
        )
        out = aggregate_to_patient_level(df).set_index("patient_id")
        np.testing.assert_allclose(out.loc["P", ["prob_0", "prob_1", "prob_2"]], [0.1, 0.3, 0.6])
        labels, probs = extract_probs_and_labels(out.reset_index())
        np.testing.assert_allclose(probs.sum(1), 1.0)

    def test_missing_patient_column_falls_back_to_one_patient_per_slide(self):
        df = pd.DataFrame({"slide_id": ["s1", "s2"], "label": [0, 1], "prob_1": [0.3, 0.6]})
        out = aggregate_to_patient_level(df)
        assert sorted(out["patient_id"]) == ["s1", "s2"]
        assert out["n_slides"].tolist() == [1, 1]
        assert "patient_id" not in df.columns  # caller's frame untouched

    def test_no_probability_columns_is_an_error(self):
        df = pd.DataFrame({"slide_id": ["s1"], "patient_id": ["P"], "label": [0]})
        with pytest.raises(ValueError, match="prob_"):
            aggregate_to_patient_level(df)


class TestExtractProbsAndLabels:
    def test_binary_single_column_is_one_dimensional_and_clipped(self):
        df = pd.DataFrame({"label": [0, 1, 1], "prob_1": [-1e-7, 0.5, 1.0000001]})
        labels, probs = extract_probs_and_labels(df)
        assert labels.tolist() == [0, 1, 1]
        assert probs.ndim == 1
        np.testing.assert_array_equal(probs, [0.0, 0.5, 1.0])

    def test_two_columns_stay_a_matrix_with_unit_rows(self):
        df = pd.DataFrame({"label": [0, 1], "prob_0": [0.70002, 0.2], "prob_1": [0.3, 0.79997]})
        _, probs = extract_probs_and_labels(df)
        assert probs.shape == (2, 2)
        np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-12)

    def test_columns_follow_class_index_beyond_ten_classes(self):
        """Column j of the matrix is prob_j, also for prob_10 and prob_11
        (a plain string sort would put prob_10 right after prob_1)."""
        n_classes = 12
        labels = np.arange(n_classes)
        probs = np.full((n_classes, n_classes), 0.02)
        probs[labels, labels] = 1 - 0.02 * (n_classes - 1)  # true class is the argmax
        df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in range(n_classes)])
        df["label"] = labels

        _, extracted = extract_probs_and_labels(df)
        np.testing.assert_array_equal(extracted.argmax(1), labels)
        assert compute_metrics(labels, extracted).accuracy == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═════════════════════════════════════════════════════════════════════════════


def _slides_df(labels, probs, patients=None) -> pd.DataFrame:
    df = pd.DataFrame(
        {"slide_id": [f"s{i}" for i in range(len(labels))], "label": labels, "prob_1": probs}
    )
    if patients is not None:
        df["patient_id"] = patients
    return df


class TestBootstrapCI:
    def test_interval_brackets_the_point_and_is_reproducible_for_a_seed(self):
        labels, probs = _signal_data(n=60, seed=3)
        df = _slides_df(labels, probs)
        first = bootstrap_ci(df, _rank_auroc, n_bootstrap=200, resample_column="slide_id")
        again = bootstrap_ci(df, _rank_auroc, n_bootstrap=200, resample_column="slide_id")
        other = bootstrap_ci(df, _rank_auroc, n_bootstrap=200, resample_column="slide_id", seed=43)

        assert first["point"] == pytest.approx(roc_auc_score(labels, probs))
        assert first["ci_lower"] < first["point"] < first["ci_upper"]
        assert first["n_valid_bootstrap"] == first["n_bootstrap"] == 200
        assert first["se"] > 0
        assert again == first
        assert (other["ci_lower"], other["ci_upper"]) != (first["ci_lower"], first["ci_upper"])
        assert other["point"] == first["point"]

    def test_wider_confidence_level_gives_an_enclosing_interval(self):
        labels, probs = _signal_data(n=60, seed=4)
        df = _slides_df(labels, probs)
        ci90 = bootstrap_ci(df, _rank_auroc, n_bootstrap=200, alpha=0.10)
        ci99 = bootstrap_ci(df, _rank_auroc, n_bootstrap=200, alpha=0.01)
        assert ci99["ci_lower"] < ci90["ci_lower"] < ci90["ci_upper"] < ci99["ci_upper"]

    def test_patients_are_resampled_as_whole_units(self):
        """Every bootstrap sample contains each drawn patient's slides together:
        the slides of one patient always appear the same number of times."""
        patients = ["A", "A", "B", "B", "B", "C", "D", "D", "E", "E", "F", "F"]
        labels = [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        probs = np.linspace(0.05, 0.95, len(patients))  # unique: identifies the slide
        df = _slides_df(labels, probs, patients)
        slide_of = {round(p, 9): i for i, p in enumerate(probs)}

        def per_slide_counts(run_column):
            samples = []

            def record(_, boot_probs):
                samples.append([slide_of[round(p, 9)] for p in boot_probs])
                return 0.5

            bootstrap_ci(df, record, n_bootstrap=50, resample_column=run_column)
            return [np.bincount(s, minlength=len(probs)) for s in samples[1:]]  # [0] = point

        for counts in per_slide_counts("patient_id"):
            drawn_patients = 0
            for patient in set(patients):
                rows = [i for i, p in enumerate(patients) if p == patient]
                assert len(set(counts[rows])) == 1, f"patient {patient} split: {counts[rows]}"
                drawn_patients += counts[rows[0]]
            assert drawn_patients == len(set(patients))  # n_units draws per sample

        # Control: resampling slides separates slides of the same patient.
        split_up = [
            c for c in per_slide_counts("slide_id") if c[0] != c[1] or c[2] != c[3] or c[3] != c[4]
        ]
        assert split_up

    def test_missing_resample_column_falls_back_to_slides(self):
        labels, probs = _signal_data(n=40, seed=5)
        df = _slides_df(labels, probs)  # no patient_id column
        by_patient = bootstrap_ci(df, _accuracy, n_bootstrap=100, resample_column="patient_id")
        by_slide = bootstrap_ci(df, _accuracy, n_bootstrap=100, resample_column="slide_id")
        assert by_patient == by_slide

    def test_single_class_data_yields_no_interval(self):
        df = _slides_df([1] * 12, np.linspace(0.2, 0.9, 12))
        result = bootstrap_ci(df, _accuracy, n_bootstrap=100, resample_column="slide_id")
        assert result["point"] == pytest.approx(_accuracy(np.ones(12), np.linspace(0.2, 0.9, 12)))
        assert result["ci_lower"] is None and result["ci_upper"] is None and result["se"] is None
        assert result["n_valid_bootstrap"] == 0

    def test_samples_where_the_metric_fails_are_skipped(self):
        labels, probs = _signal_data(n=40, seed=6)
        df = _slides_df(labels, probs)
        calls = {"n": 0}

        def flaky(y, p):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise ValueError("metric undefined on this sample")
            return _accuracy(y, p)

        result = bootstrap_ci(df, flaky, n_bootstrap=100, resample_column="slide_id")
        assert 10 <= result["n_valid_bootstrap"] < 100
        assert result["ci_lower"] <= result["point"] <= result["ci_upper"]


class TestMetricsWithCI:
    """Eight patients with two slides each; slide probabilities are the patient
    mean +/- 0.05, so patient-level values can be worked out by hand."""

    PATIENT_MEANS = [0.9, 0.8, 0.7, 0.35, 0.6, 0.3, 0.2, 0.1]
    PATIENT_LABELS = [1, 1, 1, 1, 0, 0, 0, 0]

    @pytest.fixture(scope="class")
    def predictions(self):
        rows = []
        for i, (mean, label) in enumerate(
            zip(self.PATIENT_MEANS, self.PATIENT_LABELS, strict=True)
        ):
            for j, offset in enumerate((0.05, -0.05)):
                rows.append(
                    {
                        "slide_id": f"p{i}_s{j}",
                        "patient_id": f"p{i}",
                        "label": label,
                        "prob_1": mean + offset,
                    }
                )
        return pd.DataFrame(rows)

    @pytest.fixture(scope="class")
    def result(self, predictions):
        return compute_metrics_with_ci(predictions, n_bootstrap=15, seed=0)

    def test_point_estimates_at_both_levels(self, result, predictions):
        slide = result["slide_level"]
        patient = result["patient_level"]
        assert slide["auroc"]["point"] == pytest.approx(
            roc_auc_score(predictions["label"], predictions["prob_1"])
        )
        # Patients: .9 .8 .7 beat all four negatives, .35 beats three -> 15/16
        assert patient["auroc"]["point"] == pytest.approx(15 / 16)
        # Called positive at >= .5: .9 .8 .7 (right), .6 (wrong); .35 missed -> 6/8
        assert patient["accuracy"]["point"] == pytest.approx(0.75)
        assert result["slide_metrics"]["n_samples"] == 16
        assert result["patient_metrics"]["n_samples"] == 8
        assert result["patient_metrics"]["auroc_macro"] == pytest.approx(15 / 16)

    def test_every_metric_has_an_interval_around_its_point(self, result):
        names = {
            "auroc",
            "balanced_accuracy",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "kappa",
            "precision_macro",
            "recall_macro",
        }
        for level in ("slide_level", "patient_level"):
            assert set(result[level]) == names
            for name, ci in result[level].items():
                assert ci["n_bootstrap"] == 15
                assert ci["ci_lower"] <= ci["point"] <= ci["ci_upper"], (level, name, ci)

    def test_result_is_strict_json(self, result):
        json.dumps(result, allow_nan=False)

    def test_multiclass_points_at_both_levels(self):
        """Nine patients with two slides each and three classes: the slide level
        is sklearn's one-vs-rest macro AUROC, the patient level the same on
        each patient's mean probabilities."""
        rng = np.random.default_rng(9)
        labels = np.repeat(np.arange(9) % 3, 2)
        probs = rng.dirichlet(np.ones(3), 18)
        probs[np.arange(18), labels] += 0.4
        probs /= probs.sum(1, keepdims=True)
        df = pd.DataFrame(probs, columns=["prob_0", "prob_1", "prob_2"])
        df["slide_id"] = [f"s{i}" for i in range(18)]
        df["patient_id"] = np.repeat([f"p{i}" for i in range(9)], 2)
        df["label"] = labels

        result = compute_metrics_with_ci(df, n_bootstrap=10, seed=0)
        assert result["slide_level"]["auroc"]["point"] == pytest.approx(
            roc_auc_score(labels, probs, multi_class="ovr")
        )
        patient_probs = probs.reshape(9, 2, 3).mean(1)
        assert result["patient_level"]["auroc"]["point"] == pytest.approx(
            roc_auc_score(labels[::2], patient_probs, multi_class="ovr")
        )
        assert result["patient_level"]["accuracy"]["point"] == pytest.approx(
            np.mean(patient_probs.argmax(1) == labels[::2])
        )

    def test_without_patient_ids_patient_level_reuses_slide_level(self, predictions):
        result = compute_metrics_with_ci(
            predictions.drop(columns="patient_id"), n_bootstrap=10, seed=0
        )
        assert result["patient_level"] is result["slide_level"]
        assert result["patient_metrics"]["n_samples"] == 16


# ═════════════════════════════════════════════════════════════════════════════
# Calibration
# ═════════════════════════════════════════════════════════════════════════════


class TestCalibration:
    def test_calibrated_probabilities_have_zero_ece(self):
        """P(1)=.75 on four slides of which three are positive (and the mirror
        case at .25), plus two certain, correct slides: in every occupied bin
        the confidence equals the accuracy."""
        probs = np.array([0.75] * 4 + [0.25] * 4 + [1.0, 0.0])
        labels = np.array([1, 1, 1, 0] + [0, 0, 0, 1] + [1, 0])
        cal = compute_calibration(labels, probs)
        assert cal["ece"] == pytest.approx(0.0, abs=1e-12)
        # Brier: eight slides at (.25)^2 or (.75)^2 -> (6 * .0625 + 2 * .5625) / 10
        assert cal["brier"] == pytest.approx(0.15)
        assert cal["per_class_brier"] == {"0": pytest.approx(0.15), "1": pytest.approx(0.15)}

        bins = cal["bins"]
        assert len(bins) == cal["n_bins"] == 10
        assert [b["count"] for b in bins] == [0] * 7 + [8, 0, 2]
        assert bins[7]["avg_confidence"] == pytest.approx(0.75)
        assert bins[7]["avg_accuracy"] == pytest.approx(0.75)
        assert bins[9]["avg_confidence"] == 1.0  # the top edge belongs to the last bin
        assert bins[0]["avg_confidence"] is None and bins[0]["gap"] is None

    def test_overconfident_probabilities_show_the_gap(self):
        """P(1)=.95 on ten slides of which six are positive: confidence .95,
        accuracy .6 -> ECE .35."""
        probs = np.full(10, 0.95)
        labels = np.array([1] * 6 + [0] * 4)
        cal = compute_calibration(labels, probs)
        assert cal["ece"] == pytest.approx(0.35)
        assert cal["brier"] == pytest.approx((6 * 0.05**2 + 4 * 0.95**2) / 10)
        (occupied,) = [b for b in cal["bins"] if b["count"]]
        assert occupied["gap"] == pytest.approx(0.35)

    def test_multiclass_brier_and_ece(self):
        """Slide 1: true class 0 with p=.7 (right). Slide 2: true class 2 but
        argmax is class 1 at .5 (wrong)."""
        probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
        labels = np.array([0, 2])
        cal = compute_calibration(labels, probs, n_bins=5)
        # Squared errors: .3²+.2²+.1² = .14 and .2²+.5²+.7² = .78
        assert cal["brier"] == pytest.approx((0.14 + 0.78) / 2)
        # Class 0 one-vs-rest: (.7 - 1)² and (.2 - 0)²
        assert cal["per_class_brier"]["0"] == pytest.approx((0.09 + 0.04) / 2)
        # Bins: .7 -> gap .3, .5 -> gap .5; each holds half the slides
        assert cal["ece"] == pytest.approx(0.5 * 0.3 + 0.5 * 0.5)
        assert len(cal["bins"]) == 5
        assert sum(b["count"] for b in cal["bins"]) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Operating points
# ═════════════════════════════════════════════════════════════════════════════


class TestOperatingPoints:
    LABELS = np.array([0, 0, 1, 1])
    PROBS = np.array([0.1, 0.4, 0.35, 0.8])

    def test_hand_worked_roc(self):
        """Thresholds .8/.4/.35/.1 give (sens, spec) = (.5, 1), (.5, .5), (1, .5),
        (1, 0)."""
        op = compute_operating_points(self.LABELS, self.PROBS)
        assert op["youdens_j"] == {
            "threshold": 0.8,
            "sensitivity": 0.5,
            "specificity": 1.0,
            "j_statistic": 0.5,
        }
        assert op["high_sensitivity_95"] == {
            "threshold": 0.35,
            "sensitivity": 1.0,
            "specificity": 0.5,
        }
        assert op["high_specificity_95"]["threshold"] == 0.8
        assert op["balanced"] == {"threshold": 0.4, "sensitivity": 0.5, "specificity": 0.5}
        # At t=.8 only the .8 slide is called positive
        assert op["confusion_at_youden"] == [[2, 0], [1, 1]]
        roc = op["roc_curve"]
        assert roc["fpr"][0] == 0.0 and roc["tpr"][0] == 0.0
        assert roc["fpr"][-1] == 1.0 and roc["tpr"][-1] == 1.0

    def test_thresholds_meet_their_targets_and_are_optimal(self):
        """Applying each reported threshold reproduces its sensitivity and
        specificity, and a brute-force search over every observed score finds
        no better threshold under the same constraint."""
        labels, probs = _signal_data(n=200, seed=0)
        op = compute_operating_points(labels, probs)
        candidates = np.unique(probs)
        rates = np.array([_brute_force_rates(labels, probs, t) for t in candidates])

        for key, target_col in (("high_sensitivity_95", 0), ("high_specificity_95", 1)):
            point = op[key]
            sens, spec = _brute_force_rates(labels, probs, point["threshold"])
            assert (sens, spec) == pytest.approx((point["sensitivity"], point["specificity"]))
            assert (sens, spec)[target_col] >= 0.95
            feasible = rates[rates[:, target_col] >= 0.95]
            assert (sens, spec)[1 - target_col] == pytest.approx(feasible[:, 1 - target_col].max())

        youden = op["youdens_j"]
        sens, spec = _brute_force_rates(labels, probs, youden["threshold"])
        assert sens + spec - 1 == pytest.approx(youden["j_statistic"])
        assert youden["j_statistic"] == pytest.approx((rates[:, 0] + rates[:, 1] - 1).max())

    def test_two_column_probabilities_use_the_positive_column(self):
        two_col = np.column_stack([1 - self.PROBS, self.PROBS])
        assert compute_operating_points(self.LABELS, two_col) == compute_operating_points(
            self.LABELS, self.PROBS
        )

    def test_multiclass_is_one_vs_rest_per_class(self):
        rng = np.random.default_rng(2)
        labels = np.array([0, 1, 2] * 10)
        probs = rng.dirichlet(np.ones(3), 30)
        probs[np.arange(30), labels] += 0.4
        probs /= probs.sum(1, keepdims=True)

        op = compute_operating_points(labels, probs)
        assert set(op) == {"per_class"}
        for c in range(3):
            assert op["per_class"][str(c)] == compute_operating_points(
                (labels == c).astype(int), probs[:, c]
            )

    def test_multiclass_class_without_slides_is_flagged(self):
        labels = np.array([0, 1, 0, 1])
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1], [0.3, 0.6, 0.1]])
        op = compute_operating_points(labels, probs)
        assert "error" in op["per_class"]["2"]
        assert op["per_class"]["0"]["youdens_j"]["j_statistic"] == pytest.approx(1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Threshold stability
# ═════════════════════════════════════════════════════════════════════════════


class TestThresholdStability:
    LABELS = np.array([0, 0, 0, 1, 1, 1])
    PROBS = np.array([0.1, 0.42, 0.53, 0.47, 0.57, 0.9])

    def test_hand_worked_perturbations(self):
        """At t=.5: TP {.57, .9}, FN {.47}, FP {.53}, TN {.1, .42}.
        No score lies within .01 of .5, so +/-.01 changes nothing (grade A);
        t=.45 also catches .47 (sens 1), t=.55 drops .53 (spec 1)."""
        stab = compute_threshold_stability(
            self.LABELS, self.PROBS, base_threshold=0.5, perturbations=(0.01, 0.05)
        )
        base = stab["base"]
        assert (base["tp"], base["fn"], base["fp"], base["tn"]) == (2, 1, 1, 2)
        for key in ("sensitivity", "specificity", "ppv", "npv"):
            assert base[key] == pytest.approx(2 / 3)
        assert base["accuracy"] == pytest.approx(4 / 6)

        by_delta = {p["delta"]: p for p in stab["perturbations"]}
        assert sorted(by_delta) == [-0.05, -0.01, 0.01, 0.05]
        assert by_delta[-0.01]["sens_change"] == by_delta[0.01]["sens_change"] == 0.0
        assert by_delta[-0.05]["sensitivity"] == pytest.approx(1.0)
        assert by_delta[-0.05]["sens_change"] == pytest.approx(1 / 3)
        assert by_delta[0.05]["specificity"] == pytest.approx(1.0)
        assert by_delta[0.05]["spec_change"] == pytest.approx(1 / 3)
        assert by_delta[0.05]["delta_label"] == "+0.05"
        assert stab["max_sens_change_01"] == 0.0
        assert stab["max_sens_change_05"] == pytest.approx(1 / 3)
        assert stab["stability_grade"] == "A"

    def test_scores_piled_at_the_threshold_grade_d(self):
        """Every positive sits at .505: moving the threshold to .51 loses all of them."""
        labels = np.array([1] * 10 + [0] * 10)
        probs = np.array([0.505] * 10 + [0.2] * 10)
        stab = compute_threshold_stability(labels, probs, base_threshold=0.5)
        assert stab["base"]["sensitivity"] == 1.0
        assert stab["max_sens_change_01"] == pytest.approx(1.0)
        assert stab["stability_grade"] == "D"

    def test_perturbed_thresholds_are_clamped_to_unit_interval(self):
        stab = compute_threshold_stability(
            self.LABELS, self.PROBS, base_threshold=0.97, perturbations=(0.05,)
        )
        assert [p["threshold"] for p in stab["perturbations"]] == [pytest.approx(0.92), 1.0]

    def test_two_column_input_uses_the_positive_column(self):
        two_col = np.column_stack([1 - self.PROBS, self.PROBS])
        assert compute_threshold_stability(self.LABELS, two_col, 0.5) == (
            compute_threshold_stability(self.LABELS, self.PROBS, 0.5)
        )

    def test_multiclass_is_not_supported(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        assert "note" in compute_threshold_stability(np.array([0, 2]), probs, 0.5)

    @pytest.mark.parametrize(
        ("max_change", "grade"),
        [(0.0, "A"), (0.02, "A"), (0.021, "B"), (0.05, "B"), (0.1, "C"), (0.11, "D")],
    )
    def test_grade_boundaries(self, max_change, grade):
        assert _stability_grade(max_change) == grade


# ═════════════════════════════════════════════════════════════════════════════
# PR curve
# ═════════════════════════════════════════════════════════════════════════════


class TestPRCurve:
    def test_binary_endpoints_and_average_precision(self):
        """Scores .8(+) .4(-) .35(+) .1(-): AP = .5 * 1 + .5 * 2/3."""
        labels = np.array([0, 0, 1, 1])
        pr = compute_pr_curve(labels, np.array([0.1, 0.4, 0.35, 0.8]))
        assert pr["average_precision"] == pytest.approx(0.5 + 0.5 * 2 / 3)
        # First point: everything called positive -> recall 1, precision = prevalence
        assert (pr["recall"][0], pr["precision"][0]) == (1.0, 0.5)
        # Last point: the (recall 0, precision 1) end with no threshold of its own
        assert (pr["recall"][-1], pr["precision"][-1]) == (0.0, 1.0)
        assert len(pr["thresholds"]) == len(pr["precision"]) - 1
        assert np.all(np.diff(pr["recall"]) <= 0)

    def test_two_column_input_matches_the_positive_column(self):
        labels, probs = _signal_data(n=50, seed=8)
        two_col = np.column_stack([1 - probs, probs])
        assert compute_pr_curve(labels, two_col) == compute_pr_curve(labels, probs)

    def test_multiclass_has_a_curve_per_present_class(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1]])
        pr = compute_pr_curve(labels, probs)
        assert set(pr["per_class"]) == {"0", "1"}  # class 2 has no slides
        for curve in pr["per_class"].values():
            assert curve["average_precision"] == pytest.approx(1.0)  # perfectly ranked
            assert curve["recall"][-1] == 0.0 and curve["precision"][-1] == 1.0
