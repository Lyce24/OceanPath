"""
Tests for the sklearn linear probe (``oceanpath.modules.linear_probing_sklearn``).

Builds on tests/test_linear_probe_label_efficiency.py (subsampling edge cases, inner-CV
fallbacks) without repeating it.

Covers:
  - Inputs: manifest loading, embedding-mmap loading (non-singleton policies, non-finite rows),
    manifest/embedding merge with slides missing on either side
  - Patient-grouped CV: folds never split a patient, every patient is scored once,
    near-perfect metrics on separable embeddings, saved fold models reproduce the OOF
    predictions, multiclass metrics, OOF bootstrap CIs bracket the pooled AUROC
  - External test: the probe is fit on the source cohort only and scores every target
    patient; predefined train/test split from a manifest column
  - Label fractions: per-class patient quotas, determinism per seed, held-out folds untouched
  - Patient aggregation: a patient's probability is the mean of its slides' probabilities
  - Metrics and statistics on hand-computed cases; C selection ties and fallbacks
  - CLI main() through sys.argv for all three protocols, writing the summary keys that
    scripts/collect_lp_results.py and collect_label_efficiency.py read
"""

import argparse
import json
import logging
import math
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from oceanpath.modules import linear_probing_sklearn as lp
from oceanpath.modules.extract_slide_embeddings import write_embedding_mmap

DIM = 4


@pytest.fixture(autouse=True)
def _detach_probe_log_handlers():
    """setup_logging() attaches a file handler to the 'linear_probe' logger; close it."""
    yield
    log = logging.getLogger("linear_probe")
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)


def _cohort(
    n_patients: int,
    *,
    n_classes: int = 2,
    shift: float = 4.0,
    noise: float = 0.5,
    seed: int = 0,
    prefix: str = "p",
    max_slides: int = 3,
) -> pd.DataFrame:
    """Patients with 1..max_slides slides; class k adds ``shift`` to embedding dim k."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        label = p % n_classes
        for s in range(1 + p % max_slides):
            emb = rng.normal(0.0, noise, DIM).astype(np.float32)
            emb[label] += shift
            rows.append(
                {
                    "slide_id": f"{prefix}{p:03d}_s{s}",
                    "patient_id": f"{prefix}{p:03d}",
                    "label": label,
                    "embedding": emb,
                }
            )
    return pd.DataFrame(rows)


def _args(output_dir, **overrides) -> argparse.Namespace:
    params = {
        "output_dir": str(output_dir),
        "n_splits": 3,
        "inner_splits": 0,
        "c_grid": [1.0],
        "max_iter": 500,
        "seed": 0,
    }
    params.update(overrides)
    return lp.make_probe_args(**params)


def _matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["embedding"].to_list())


def _write_store(path, df: pd.DataFrame) -> None:
    store_args = argparse.Namespace(
        mode="mean_pool",
        ckpt=None,
        mmap_dir="patch_store",
        feat_precision=32,
        coord_dtype="int32",
        max_chunk_gb=1.0,
    )
    write_embedding_mmap(path, df["slide_id"].tolist(), _matrix(df), store_args)


def _write_manifest(path, df: pd.DataFrame, **renames) -> None:
    out = pd.DataFrame(
        {
            "filename": df["slide_id"] + ".svs",
            "label": df["label"],
            "patient": df["patient_id"],
        }
    )
    if "split" in df.columns:
        out["split"] = df["split"]
    out.rename(columns=renames).to_csv(path, index=False)


def _patients_per_class(df: pd.DataFrame) -> dict[int, set[str]]:
    by_patient = df.groupby("patient_id")["label"].first()
    return {int(c): set(by_patient[by_patient == c].index) for c in sorted(by_patient.unique())}


# ═════════════════════════════════════════════════════════════════════════════
# Inputs: manifest, embedding store, merge
# ═════════════════════════════════════════════════════════════════════════════


class TestManifest:
    def test_columns_are_normalised(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame({"file": ["a.svs", "dir\\b.h5"], "y": [0.0, 1.0], "case": [7, 8]}).to_csv(
            csv, index=False
        )

        df = lp.load_manifest(csv, None, "file", "y", "case")

        assert df["slide_id"].tolist() == ["a", "dir/b"]
        assert df["label"].tolist() == [0, 1]
        assert df["label"].dtype.kind == "i"
        assert df["patient_id"].tolist() == ["7", "8"]

    def test_explicit_slide_id_column_wins(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame(
            {"filename": ["x.svs", "y.svs"], "sid": ["s1.h5", "s2"], "y": [0, 1], "p": [1, 2]}
        ).to_csv(csv, index=False)
        assert lp.load_manifest(csv, "sid", "filename", "y", "p")["slide_id"].tolist() == [
            "s1",
            "s2",
        ]
        with pytest.raises(ValueError, match="slide_id_column='slide' not found"):
            lp.load_manifest(csv, "slide", "filename", "y", "p")
        with pytest.raises(ValueError, match="Neither slide_id_column nor filename_column"):
            lp.load_manifest(csv, None, "path", "y", "p")

    def test_missing_required_columns_rejected(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame({"filename": ["a.svs"], "label": [0]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match=r"missing required columns: \['patient_id'\]"):
            lp.load_manifest(csv, None, "filename", "label", "patient_id")

    def test_missing_split_column_rejected(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame({"filename": ["a.svs"], "label": [0], "patient_id": ["p"]}).to_csv(
            csv, index=False
        )
        with pytest.raises(ValueError, match="split_column='fold'"):
            lp.load_manifest(csv, None, "filename", "label", "patient_id", split_column="fold")

    def test_duplicate_slides_rejected(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame(
            {"filename": ["a.svs", "a.ndpi"], "label": [0, 1], "patient_id": ["p", "q"]}
        ).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="duplicate slide_id"):
            lp.load_manifest(csv, None, "filename", "label", "patient_id")


class TestLoadEmbeddingMmap:
    @pytest.fixture(scope="class")
    def patch_store(self, build_feature_store):
        """A patch-level store: several rows per slide, i.e. not one embedding per slide."""
        return build_feature_store([3, 4, 1], feat_dim=DIM, seed=1)

    def test_multi_row_slides_need_an_explicit_policy(self, patch_store):
        with pytest.raises(ValueError, match="length=3"):
            lp.load_embedding_mmap(patch_store.mmap_dir)

    def test_mean_and_first_policies(self, patch_store):
        mean = lp.load_embedding_mmap(patch_store.mmap_dir, non_singleton_policy="mean")
        first = lp.load_embedding_mmap(patch_store.mmap_dir, non_singleton_policy="first")

        assert mean.slide_ids == first.slide_ids == patch_store.slide_ids
        for i, sid in enumerate(patch_store.slide_ids):
            feats = patch_store.features[sid]
            np.testing.assert_allclose(mean.embeddings[i], feats.mean(axis=0), rtol=1e-6)
            np.testing.assert_array_equal(first.embeddings[i], feats[0])

    def test_unknown_policy_rejected(self, patch_store):
        with pytest.raises(ValueError, match="Unknown non_singleton_policy"):
            lp.load_embedding_mmap(patch_store.mmap_dir, non_singleton_policy="median")

    def test_duplicate_slide_ids_rejected(self, tmp_path):
        df = _cohort(3)
        df.loc[2, "slide_id"] = df.loc[0, "slide_id"]
        _write_store(tmp_path / "emb", df)
        with pytest.raises(ValueError, match="contains duplicate slide_ids"):
            lp.load_embedding_mmap(tmp_path / "emb")

    def test_non_finite_embedding_rejected(self, tmp_path):
        df = _cohort(3)
        df.at[1, "embedding"] = np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float32)
        _write_store(tmp_path / "emb", df)
        with pytest.raises(ValueError, match=f"non-finite values for slide '{df.slide_id[1]}'"):
            lp.load_embedding_mmap(tmp_path / "emb")


class TestMerge:
    @pytest.fixture
    def store(self):
        df = _cohort(4)
        return lp.EmbeddingStore(slide_ids=df["slide_id"].tolist(), embeddings=_matrix(df)), df

    def test_each_slide_gets_its_own_embedding(self, store):
        emb_store, df = store
        manifest = df.drop(columns="embedding").iloc[::-1].reset_index(drop=True)

        merged = lp.merge_manifest_and_embeddings(manifest, emb_store)

        assert merged["slide_id"].tolist() == manifest["slide_id"].tolist()
        by_id = dict(zip(df["slide_id"], df["embedding"], strict=True))
        for sid, emb in zip(merged["slide_id"], merged["embedding"], strict=True):
            assert emb.dtype == np.float32
            np.testing.assert_array_equal(emb, by_id[sid])

    def test_manifest_slide_missing_from_store(self, store):
        emb_store, df = store
        extra = pd.DataFrame([{"slide_id": "ghost", "patient_id": "pX", "label": 0}])
        manifest = pd.concat([df.drop(columns="embedding"), extra], ignore_index=True)

        with pytest.raises(ValueError, match="1 manifest slide_ids were not found"):
            lp.merge_manifest_and_embeddings(manifest, emb_store)
        merged = lp.merge_manifest_and_embeddings(manifest, emb_store, allow_missing_mmap=True)
        assert "ghost" not in set(merged["slide_id"])
        assert len(merged) == len(df)

    def test_store_slide_missing_from_manifest(self, store):
        emb_store, df = store
        manifest = df.drop(columns="embedding").iloc[1:]

        with pytest.raises(ValueError, match="1 mmap slide_ids are missing from the manifest"):
            lp.merge_manifest_and_embeddings(manifest, emb_store)
        merged = lp.merge_manifest_and_embeddings(manifest, emb_store, allow_missing_manifest=True)
        assert merged["slide_id"].tolist() == manifest["slide_id"].tolist()

    def test_no_overlap_rejected(self, store):
        emb_store, _ = store
        manifest = pd.DataFrame([{"slide_id": "other", "patient_id": "q", "label": 1}])
        with pytest.raises(ValueError, match="No rows remain"):
            lp.merge_manifest_and_embeddings(
                manifest, emb_store, allow_missing_mmap=True, allow_missing_manifest=True
            )


# ═════════════════════════════════════════════════════════════════════════════
# Patient-grouped cross-validation
# ═════════════════════════════════════════════════════════════════════════════


class TestPickGroupCV:
    @pytest.mark.parametrize("n_rare", [6, 2], ids=["stratified", "rare_class_fallback"])
    def test_validation_folds_partition_the_patients(self, n_rare):
        df = _cohort(24, max_slides=1)  # the fallback rule counts slides
        rare_patients = sorted(_patients_per_class(df)[1])[:n_rare]
        df = df[(df["label"] == 0) | df["patient_id"].isin(rare_patients)]
        y, groups = df["label"].to_numpy(), df["patient_id"].to_numpy()

        splitter = lp.pick_group_cv(y, groups, n_splits=3, seed=0)

        expected_cls = "StratifiedGroupKFold" if n_rare >= 3 else "GroupKFold"
        assert type(splitter).__name__ == expected_cls
        seen: list[str] = []
        for tr, va in splitter.split(np.zeros(len(y)), y, groups):
            assert not set(groups[tr]) & set(groups[va])
            seen.extend(set(groups[va]))
        assert sorted(seen) == sorted(set(groups))

    def test_fewer_patients_than_folds_rejected(self):
        with pytest.raises(ValueError, match="Need at least 5 unique groups"):
            lp.pick_group_cv(np.array([0, 1, 0, 1]), np.array(["a", "a", "b", "c"]), 5)


class TestGroupedCV:
    C_GRID = [0.1, 10.0]

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("grouped_cv")
        df = _cohort(30)
        args = _args(out, inner_splits=2, c_grid=self.C_GRID, save_model=True)
        return lp.run_grouped_cv(args, df), df, out

    def test_every_patient_is_scored_exactly_once(self, run):
        _, df, out = run
        oof = pd.read_csv(out / "patient_predictions_oof.csv", dtype={"patient_id": str})
        assert oof["patient_id"].is_unique
        assert set(oof["patient_id"]) == set(df["patient_id"])
        assert set(oof["outer_fold"]) == {0, 1, 2}

    def test_separable_embeddings_score_near_perfectly(self, run):
        summary, _, _ = run
        assert summary["protocol"] == "grouped_cv"
        assert summary["task_type"] == "binary"
        assert summary["primary_metric"] == "auroc"
        assert summary["n_folds"] == 3
        assert summary["mean_auroc"] >= 0.99
        assert summary["mean_auprc"] >= 0.99
        assert summary["mean_balanced_acc"] >= 0.95
        assert summary["label_fraction"] == 1.0

    def test_c_is_chosen_by_the_inner_search(self, run):
        summary, _, out = run
        search = pd.read_csv(out / "inner_cv_search.csv")
        # One score per (C, inner fold) in every outer fold.
        assert len(search) == len(self.C_GRID) * 2 * 3
        assert set(search["C"]) == set(self.C_GRID)
        for fm in summary["fold_metrics"]:
            assert fm["best_C"] in self.C_GRID
        folds = pd.read_csv(out / "fold_metrics.csv")
        assert folds["fold"].tolist() == [0, 1, 2]

    def test_saved_fold_models_reproduce_the_oof_probabilities(self, run):
        """Each fold's saved probe, averaged over a patient's slides, gives the reported prob."""
        _, df, out = run
        oof = pd.read_csv(out / "patient_predictions_oof.csv", dtype={"patient_id": str})
        for fold, fold_rows in oof.groupby("outer_fold"):
            model = joblib.load(out / f"linear_probe_fold{fold}.joblib")
            for pid, prob in zip(fold_rows["patient_id"], fold_rows["prob_1"], strict=True):
                slides = df[df["patient_id"] == pid]
                expected = model.predict_proba(_matrix(slides))[:, 1].mean()
                assert prob == pytest.approx(expected, abs=1e-6)

    def test_slide_level_folds_never_split_a_patient(self, tmp_path):
        df = _cohort(30, seed=1)
        lp.run_grouped_cv(_args(tmp_path, n_splits=4, aggregate_to_patient=False), df)

        oof = pd.read_csv(tmp_path / "patient_predictions_oof.csv")
        # Slide-level evaluation: the id column holds slide IDs, each scored once.
        assert sorted(oof["patient_id"]) == sorted(df["slide_id"])
        fold_of = dict(zip(oof["patient_id"], oof["outer_fold"], strict=True))
        folds_per_patient = df.groupby("patient_id")["slide_id"].agg(
            lambda s: len({fold_of[x] for x in s})
        )
        assert (folds_per_patient == 1).all()

    def test_bootstrap_ci_brackets_the_pooled_oof_auroc(self, tmp_path):
        df = _cohort(40, shift=1.2, noise=1.0, seed=2)
        summary = lp.run_grouped_cv(_args(tmp_path, n_boot=60), df)

        oof = pd.read_csv(tmp_path / "patient_predictions_oof.csv")
        auroc = roc_auc_score(oof["label"], oof["prob_1"])
        auprc = average_precision_score(oof["label"], oof["prob_1"])
        assert 0.6 < auroc < 0.99  # noisy enough for a non-degenerate interval
        assert summary["oof_auroc_ci_low"] <= auroc <= summary["oof_auroc_ci_high"]
        assert summary["oof_auroc_ci_low"] < summary["oof_auroc_ci_high"]
        assert summary["oof_auprc_ci_low"] <= auprc <= summary["oof_auprc_ci_high"]

    def test_three_classes_report_qwk_and_macro_auroc(self, tmp_path):
        df = _cohort(30, n_classes=3)
        summary = lp.run_grouped_cv(_args(tmp_path), df)

        assert summary["task_type"] == "multiclass"
        assert summary["primary_metric"] == "macro_f1"
        assert summary["mean_qwk"] >= 0.95
        assert summary["mean_auroc_macro_ovr"] >= 0.99
        oof = pd.read_csv(tmp_path / "patient_predictions_oof.csv")
        probs = oof[["prob_0", "prob_1", "prob_2"]].to_numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
        assert (probs.argmax(axis=1) == oof["pred"]).all()

    def test_binary_labels_must_be_zero_and_one(self, tmp_path):
        df = _cohort(12)
        df["label"] += 1
        with pytest.raises(ValueError, match=r"encoded as \{0, 1\}"):
            lp.run_grouped_cv(_args(tmp_path), df)


# ═════════════════════════════════════════════════════════════════════════════
# External test and predefined split
# ═════════════════════════════════════════════════════════════════════════════


class TestExternalTest:
    def test_fits_on_the_source_and_scores_every_target_patient(self, tmp_path):
        source, target = _cohort(24, prefix="src"), _cohort(15, prefix="tgt", seed=5)
        summary = lp.run_external_test(_args(tmp_path, save_model=True), source, target)

        assert summary["protocol"] == "external_test"
        assert summary["n_train_patients"] == summary["n_train_patients_full"] == 24
        assert summary["n_train_slides"] == len(source)
        assert summary["n_test_patients"] == summary["n_patients"] == 15
        assert summary["auroc"] >= 0.99
        assert summary["balanced_acc"] >= 0.95

        preds = pd.read_csv(tmp_path / "patient_predictions_test.csv")
        assert sorted(preds["patient_id"]) == sorted(target["patient_id"].unique())
        model = joblib.load(tmp_path / "linear_probe_model.joblib")
        for pid, prob in zip(preds["patient_id"], preds["prob_1"], strict=True):
            slides = target[target["patient_id"] == pid]
            assert prob == pytest.approx(model.predict_proba(_matrix(slides))[:, 1].mean())

    def test_target_labels_never_reach_the_probe(self, tmp_path):
        """With the target's labels inverted, a source-only probe ranks every pair wrong."""
        source, target = _cohort(24, prefix="src"), _cohort(15, prefix="tgt", seed=5)
        target["label"] = 1 - target["label"]

        summary = lp.run_external_test(_args(tmp_path), source, target)

        assert summary["auroc"] <= 0.01

    def test_bootstrap_ci_contains_the_point_estimate(self, tmp_path):
        source = _cohort(40, prefix="src", shift=1.2, noise=1.0, seed=3)
        target = _cohort(40, prefix="tgt", shift=1.2, noise=1.0, seed=4)

        summary = lp.run_external_test(_args(tmp_path, n_boot=60), source, target)

        assert 0.6 < summary["auroc"] < 0.99
        assert summary["auroc_ci_low"] <= summary["auroc"] <= summary["auroc_ci_high"]
        assert summary["auroc_ci_low"] < summary["auroc_ci_high"]
        assert summary["auprc_ci_low"] <= summary["auprc"] <= summary["auprc_ci_high"]

    def test_multiclass_target_gets_one_probability_per_class(self, tmp_path):
        source = _cohort(18, n_classes=3, prefix="src")
        target = _cohort(9, n_classes=3, prefix="tgt", seed=5)

        summary = lp.run_external_test(_args(tmp_path), source, target)

        assert summary["task_type"] == "multiclass"
        assert summary["qwk"] >= 0.95
        preds = pd.read_csv(tmp_path / "patient_predictions_test.csv")
        probs = preds[["prob_0", "prob_1", "prob_2"]].to_numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
        assert (probs.argmax(axis=1) == preds["pred"]).all()

    def test_single_class_training_cohort_rejected(self, tmp_path):
        source = _cohort(12, prefix="src")
        source = source[source["label"] == 1]
        with pytest.raises(ValueError, match="contains only one class"):
            lp.run_external_test(_args(tmp_path), source, _cohort(6, prefix="tgt"))

    def test_target_label_unseen_in_training_rejected(self, tmp_path):
        source, target = _cohort(12, prefix="src"), _cohort(9, n_classes=3, prefix="tgt")
        with pytest.raises(ValueError, match=r"not present in the training split: \[2\]"):
            lp.run_external_test(_args(tmp_path), source, target)


class TestPredefinedSplit:
    @pytest.fixture
    def cohort(self):
        df = _cohort(30)
        patient_rank = df["patient_id"].str[1:].astype(int)
        df["split"] = np.select(
            [patient_rank < 18, patient_rank < 24], ["Train", "val"], default="TEST"
        )
        return df

    def test_train_rows_fit_and_test_rows_score(self, tmp_path, cohort):
        summary = lp.run_predefined_split(_args(tmp_path, protocol="predefined_split"), cohort)

        train, test = cohort[cohort["split"] == "Train"], cohort[cohort["split"] == "TEST"]
        assert summary["n_train_slides"] == len(train)
        assert summary["n_test_patients"] == test["patient_id"].nunique()
        preds = pd.read_csv(tmp_path / "patient_predictions_test.csv")
        assert set(preds["patient_id"]) == set(test["patient_id"])  # "val" rows unused
        assert summary["auroc"] >= 0.99

    def test_split_column_required(self, tmp_path, cohort):
        with pytest.raises(ValueError, match="requires --split_column"):
            lp.run_predefined_split(_args(tmp_path), cohort.drop(columns="split"))

    def test_both_partitions_required(self, tmp_path, cohort):
        cohort["split"] = cohort["split"].replace("TEST", "holdout")
        with pytest.raises(ValueError, match=r"Found split values: \['holdout', 'train', 'val'\]"):
            lp.run_predefined_split(_args(tmp_path), cohort)


# ═════════════════════════════════════════════════════════════════════════════
# Label fractions
# ═════════════════════════════════════════════════════════════════════════════


class TestLabelFractions:
    @pytest.mark.parametrize("fraction", [0.05, 0.25, 0.5])
    def test_each_class_keeps_ceil_fraction_of_its_patients(self, fraction):
        df = _cohort(40)  # 20 patients per class, 1-3 slides each

        sampled, info = lp.subsample_labeled_training_data(df, fraction, seed=3, context="t")

        quota = math.ceil(20 * fraction)
        for cls, patients in _patients_per_class(sampled).items():
            assert len(patients) == quota, cls
        # A selected patient keeps all of its slides.
        kept = df[df["patient_id"].isin(sampled["patient_id"])]
        assert sorted(sampled["slide_id"]) == sorted(kept["slide_id"])
        assert info["train_labeled_patients"] == 2 * quota
        assert info["train_labeled_patient_fraction"] == pytest.approx(2 * quota / 40)
        assert info["train_labeled_slides"] == len(sampled)

    def test_subsample_is_deterministic_per_seed(self):
        df = _cohort(40)

        a, _ = lp.subsample_labeled_training_data(df, 0.25, seed=7, context="t")
        b, _ = lp.subsample_labeled_training_data(df, 0.25, seed=7, context="t")
        c, _ = lp.subsample_labeled_training_data(df, 0.25, seed=8, context="t")

        pd.testing.assert_frame_equal(a, b)
        assert set(a["patient_id"]) != set(c["patient_id"])

    def test_full_fraction_keeps_every_row(self):
        df = _cohort(10)
        sampled, info = lp.subsample_labeled_training_data(df, 1.0, seed=0, context="t")
        assert sampled["slide_id"].tolist() == df["slide_id"].tolist()
        assert info["train_labeled_patient_fraction"] == 1.0

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_fraction_outside_the_unit_interval_rejected(self, fraction):
        with pytest.raises(ValueError, match="label_fraction must be in"):
            lp.subsample_labeled_training_data(_cohort(4), fraction, seed=0, context="t")

    def test_grouped_cv_subsamples_training_folds_only_and_reproducibly(self, tmp_path):
        df = _cohort(36)
        args = _args(tmp_path, label_fraction=0.25, label_fraction_seed=11)

        first = lp.run_grouped_cv(args, df)
        oof_first = pd.read_csv(tmp_path / "patient_predictions_oof.csv")
        second = lp.run_grouped_cv(args, df)
        oof_second = pd.read_csv(tmp_path / "patient_predictions_oof.csv")

        pd.testing.assert_frame_equal(oof_first, oof_second)
        assert first["fold_metrics"] == second["fold_metrics"]
        assert first["label_fraction_seed"] == 11
        # Validation folds stay full-size: every patient is still scored once.
        assert sorted(oof_first["patient_id"]) == sorted(df["patient_id"].unique())
        patients = _patients_per_class(df)
        for fm in first["fold_metrics"]:
            held_out = set(oof_first.loc[oof_first["outer_fold"] == fm["fold"], "patient_id"])
            quota = sum(math.ceil(len(p - held_out) * 0.25) for p in patients.values())
            assert fm["train_labeled_patients"] == quota
            assert fm["train_full_patients"] == 36 - len(held_out)
            assert fm["label_fraction_seed"] == 11 + fm["fold"]


# ═════════════════════════════════════════════════════════════════════════════
# Patient aggregation
# ═════════════════════════════════════════════════════════════════════════════


class TestPatientAggregation:
    SLIDES = pd.DataFrame(
        {
            "slide_id": ["a1", "a2", "b1", "c1", "c2", "c3"],
            "patient_id": ["A", "A", "B", "C", "C", "C"],
            "label": [1, 1, 1, 0, 0, 0],
        }
    )

    def test_patient_probability_is_the_mean_of_its_slides(self):
        prob = np.array([0.2, 0.6, 0.9, 0.1, 0.3, 0.8])

        pp = lp.aggregate_to_patient_level(self.SLIDES, prob, "binary")

        assert pp.patient_ids.tolist() == ["A", "B", "C"]
        np.testing.assert_allclose(pp.prob, [0.4, 0.9, 0.4])
        assert pp.y_true.tolist() == [1, 1, 0]
        assert pp.y_pred.tolist() == [0, 1, 0]

    def test_multiclass_probabilities_average_per_class(self):
        prob = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.3, 0.5],
                [0.1, 0.1, 0.8],
                [1.0, 0.0, 0.0],
                [0.4, 0.4, 0.2],
                [0.1, 0.8, 0.1],
            ]
        )

        pp = lp.aggregate_to_patient_level(self.SLIDES, prob, "multiclass")

        np.testing.assert_allclose(pp.prob[0], [0.4, 0.3, 0.3])
        np.testing.assert_allclose(pp.prob[2], [0.5, 0.4, 0.1])
        assert pp.y_pred.tolist() == [0, 2, 0]

    def test_conflicting_slide_labels_rejected(self):
        slides = self.SLIDES.assign(label=[1, 0, 1, 0, 0, 0])
        with pytest.raises(ValueError, match="conflicting slide labels"):
            lp.aggregate_to_patient_level(slides, np.full(6, 0.5), "binary")

    def test_slide_level_reduction_keeps_every_slide(self):
        prob = np.array([0.2, 0.6, 0.9, 0.1, 0.3, 0.8])
        pp = lp.reduce_predictions(self.SLIDES, prob, "binary", aggregate_to_patient=False)
        assert pp.patient_ids.tolist() == self.SLIDES["slide_id"].tolist()
        assert pp.y_pred.tolist() == [0, 1, 1, 0, 0, 1]

        multi = np.eye(3)[[2, 0, 1, 1, 0, 2]] * 0.6 + 0.1
        pp = lp.reduce_predictions(self.SLIDES, multi, "multiclass", aggregate_to_patient=False)
        assert pp.y_pred.tolist() == [2, 0, 1, 1, 0, 2]


# ═════════════════════════════════════════════════════════════════════════════
# Metrics, fold summaries and bootstrap intervals
# ═════════════════════════════════════════════════════════════════════════════


class TestMetrics:
    def test_binary_metrics_on_a_hand_computed_case(self):
        pp = lp.PatientPredictions(
            patient_ids=np.array(["a", "b", "c", "d"]),
            y_true=np.array([0, 0, 1, 1]),
            prob=np.array([0.1, 0.6, 0.4, 0.9]),
            y_pred=np.array([0, 1, 0, 1]),
        )

        m = lp.compute_metrics_from_patient_preds(pp, "binary")

        assert m["auroc"] == pytest.approx(0.75)  # 3 of 4 (neg, pos) pairs ranked right
        assert m["auprc"] == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
        assert m["balanced_acc"] == pytest.approx(0.5)
        assert m["macro_f1"] == pytest.approx(0.5)
        assert m["confusion_matrix"] == [[1, 1], [1, 1]]
        assert m["n_patients"] == 4

    def test_undefined_scores_rank_below_every_c(self):
        pp = lp.PatientPredictions(
            patient_ids=np.array(["a", "b"]),
            y_true=np.array([1, 1]),
            prob=np.array([0.3, 0.7]),
            y_pred=np.array([0, 1]),
        )
        m = lp.compute_metrics_from_patient_preds(pp, "binary")
        assert m["auroc"] is None or math.isnan(m["auroc"])  # sklearn >= 1.6 returns NaN
        assert lp.get_primary_metric_value(m, "auroc") == float("-inf")
        assert lp.get_primary_metric_value({"qwk": float("nan")}, "qwk") == float("-inf")
        assert lp.get_primary_metric_value({"qwk": 0.25}, "qwk") == 0.25

    def test_multiclass_metrics_for_perfect_predictions(self):
        y = np.array([0, 1, 2, 2, 1, 0])
        pp = lp.PatientPredictions(patient_ids=np.arange(6), y_true=y, prob=np.eye(3)[y], y_pred=y)
        m = lp.compute_metrics_from_patient_preds(pp, "multiclass")
        assert m["qwk"] == pytest.approx(1.0)
        assert m["auroc_macro_ovr"] == pytest.approx(1.0)
        assert "auroc" not in m

    def test_fold_summary_skips_non_finite_values(self):
        nan = float("nan")
        folds = [
            {"fold": 0, "best_C": 1.0, "n_patients": 10, "auroc": 0.8, "qwk": nan, "k": nan},
            {"fold": 1, "best_C": 10.0, "n_patients": 12, "auroc": 0.6, "qwk": 0.5, "k": None},
            {"fold": 2, "best_C": 1.0, "n_patients": 11, "auroc": None, "qwk": 0.7, "k": nan},
        ]

        s = lp.summarize_fold_metrics(folds)

        assert s["n_folds"] == 3
        assert s["mean_auroc"] == pytest.approx(0.7)
        assert s["std_auroc"] == pytest.approx(np.std([0.8, 0.6], ddof=1))
        assert s["mean_qwk"] == pytest.approx(0.6)
        assert s["n_valid_auroc"] == s["n_valid_qwk"] == 2
        assert not {"mean_k", "std_k", "n_valid_k"} & set(s)  # never defined
        assert not {"mean_fold", "mean_best_C", "mean_n_patients"} & set(s)
        single = lp.summarize_fold_metrics([{"auroc": 0.9}])
        assert single["std_auroc"] == 0.0
        assert lp.summarize_fold_metrics([]) == {}

    def test_bootstrap_ci_is_seeded_and_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1], 40)
        prob = np.clip(rng.normal(0.4 + 0.2 * y, 0.2), 0, 1)
        auroc = roc_auc_score(y, prob)

        ci = lp.bootstrap_metric_ci(y, prob, roc_auc_score, n_boot=60, seed=1)

        assert ci == lp.bootstrap_metric_ci(y, prob, roc_auc_score, n_boot=60, seed=1)
        assert ci != lp.bootstrap_metric_ci(y, prob, roc_auc_score, n_boot=60, seed=2)
        assert ci[0] < auroc < ci[1]

    def test_bootstrap_ci_skips_one_class_resamples(self):
        """A small cohort with 3 positives draws some resamples without any positive;
        those have no AUROC and must not turn the interval into NaN."""
        rng = np.random.default_rng(0)
        y = np.array([1] * 3 + [0] * 17)
        prob = np.clip(rng.normal(0.4 + 0.15 * y, 0.15), 0, 1)
        one_class_draws = []

        def auroc(y_true, y_prob):
            one_class_draws.append(len(np.unique(y_true)) < 2)
            return roc_auc_score(y_true, y_prob)

        low, high = lp.bootstrap_metric_ci(y, prob, auroc, n_boot=100, seed=5)

        assert any(one_class_draws)
        assert np.isfinite([low, high]).all()
        assert low <= roc_auc_score(y, prob) <= high

    def test_bootstrap_ci_is_none_when_every_resample_fails(self):
        y = np.zeros(10, dtype=int)
        assert lp.bootstrap_metric_ci(y, np.linspace(0, 1, 10), roc_auc_score, 50) == (
            None,
            None,
        )

    def test_save_summary_writes_numpy_values_as_json(self, tmp_path):
        summary = {"a": np.int64(3), "b": np.float32(0.5), "c": np.arange(3), "d": [1.5]}
        lp.save_summary(tmp_path, summary)
        assert json.loads((tmp_path / "summary.json").read_text()) == {
            "a": 3,
            "b": 0.5,
            "c": [0, 1, 2],
            "d": [1.5],
        }


class TestSelectBestC:
    GRID = [0.01, 0.1, 1.0, 10.0]

    def _select(self, df, **kwargs):
        params = {
            "train_df": df,
            "task_type": "binary",
            "c_grid": self.GRID,
            "inner_splits": 3,
            "primary_metric": "auroc",
            "class_weight": "balanced",
            "max_iter": 200,
            "multi_class_mode": "auto",
            "seed": 0,
        }
        params.update(kwargs)
        return lp.select_best_c(**params)

    def test_ties_go_to_the_most_regularised_c(self):
        """Every C ranks separable patients perfectly, so the smallest C wins the tie."""
        chosen, search = self._select(_cohort(30), inner_splits=2)
        assert len(search) == len(self.GRID) * 2
        assert (search["score"] == 1.0).all()
        assert chosen == min(self.GRID)

    def test_inner_cv_disabled_or_single_c_uses_the_first_c(self):
        assert self._select(_cohort(30), inner_splits=0)[0] == self.GRID[0]
        assert self._select(_cohort(30), c_grid=[3.0])[0] == 3.0

    def test_no_finite_inner_score_falls_back_to_the_grid_midpoint(self, monkeypatch):
        """An inner validation fold with one class leaves AUROC undefined for every C;
        selection must then use the grid midpoint, not the smallest C."""
        df = _cohort(30, max_slides=1)
        pos = np.flatnonzero(df["label"].to_numpy() == 1)
        neg = np.flatnonzero(df["label"].to_numpy() == 0)

        class _OneFoldWithoutPositives:
            def split(self, X, y, groups):
                folds = [
                    np.concatenate([pos[:8], neg[:5]]),
                    np.concatenate([pos[8:], neg[5:10]]),
                    neg[10:],
                ]
                for i, va in enumerate(folds):
                    yield np.concatenate([f for j, f in enumerate(folds) if j != i]), va

        monkeypatch.setattr(lp, "pick_group_cv", lambda *a, **k: _OneFoldWithoutPositives())
        chosen, search = self._select(df)

        assert (search.groupby("C")["score"].min() == float("-inf")).all()
        assert chosen == 1.0  # sorted grid [0.01, 0.1, 1.0, 10.0] -> index 2


# ═════════════════════════════════════════════════════════════════════════════
# CLI: main() through sys.argv
# ═════════════════════════════════════════════════════════════════════════════


class TestMain:
    @pytest.fixture
    def cohort_files(self, tmp_path):
        df = _cohort(24)
        _write_store(tmp_path / "emb", df)
        _write_manifest(tmp_path / "manifest.csv", df)
        return df, tmp_path

    @staticmethod
    def _main(monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["linear_probing_sklearn", *argv])
        lp.main()

    def test_grouped_cv_writes_the_summary_downstream_reads(self, cohort_files, monkeypatch):
        df, root = cohort_files
        out = root / "out"
        self._main(
            monkeypatch,
            *("--mmap_dir", str(root / "emb"), "--manifest_csv", str(root / "manifest.csv")),
            *("--output_dir", str(out), "--label_column", "label", "--patient_column", "patient"),
            *("--n_splits", "3", "--inner_splits", "0", "--c_grid", "0.1", "1.0"),
            *("--n_boot", "20", "--save_model", "--seed", "3"),
        )

        summary = json.loads((out / "summary.json").read_text())
        assert summary["protocol"] == "grouped_cv"
        assert summary["task_type"] == "binary"
        assert summary["label_fraction"] == 1.0
        assert summary["label_fraction_seed"] == 3
        assert summary["n_folds"] == 3
        for key in ("auroc", "auprc", "balanced_acc", "macro_f1"):
            assert f"mean_{key}" in summary and f"std_{key}" in summary
        for key in ("oof_auroc_ci_low", "oof_auroc_ci_high", "oof_auprc_ci_low"):
            assert summary[key] is not None
        assert [fm["best_C"] for fm in summary["fold_metrics"]] == [0.1] * 3  # inner CV off
        assert summary["mean_auroc"] >= 0.99

        oof = pd.read_csv(out / "patient_predictions_oof.csv")
        assert {"outer_fold", "patient_id", "label", "pred", "prob_1"} <= set(oof.columns)
        assert len(oof) == df["patient_id"].nunique()
        for name in ("fold_metrics.csv", "inner_cv_search.csv", "linear_probe.log"):
            assert (out / name).is_file()
        assert sorted(p.name for p in out.glob("*.joblib")) == [
            f"linear_probe_fold{k}.joblib" for k in range(3)
        ]

    def test_external_test_uses_the_test_cohort_columns(self, cohort_files, monkeypatch):
        _, root = cohort_files
        target = _cohort(12, prefix="tgt", seed=9)
        _write_store(root / "tgt_emb", target)
        _write_manifest(root / "tgt.csv", target, filename="path", label="y", patient="case_id")
        out = root / "out"
        self._main(
            monkeypatch,
            *("--protocol", "external_test", "--mmap_dir", str(root / "emb")),
            *("--manifest_csv", str(root / "manifest.csv"), "--output_dir", str(out)),
            *("--label_column", "label", "--patient_column", "patient"),
            *("--test_mmap_dir", str(root / "tgt_emb")),
            *("--test_manifest_csv", str(root / "tgt.csv")),
            *("--test_filename_column", "path", "--test_label_column", "y"),
            *("--test_patient_column", "case_id", "--inner_splits", "0", "--n_boot", "20"),
        )

        summary = json.loads((out / "summary.json").read_text())
        assert summary["protocol"] == "external_test"
        assert summary["n_train_patients"] == 24
        assert summary["n_test_patients"] == 12
        for key in ("auroc", "auprc", "balanced_acc", "macro_f1", "best_C"):
            assert summary[key] is not None
        assert summary["auroc_ci_low"] <= summary["auroc"] <= summary["auroc_ci_high"]
        preds = pd.read_csv(out / "patient_predictions_test.csv")
        assert sorted(preds["patient_id"]) == sorted(target["patient_id"].unique())

    def test_predefined_split_reads_the_split_column(self, tmp_path, monkeypatch):
        df = _cohort(24)
        df["split"] = np.where(df["patient_id"].str[1:].astype(int) < 16, "train", "test")
        _write_store(tmp_path / "emb", df)
        _write_manifest(tmp_path / "manifest.csv", df)
        out = tmp_path / "out"
        self._main(
            monkeypatch,
            *("--protocol", "predefined_split", "--split_column", "split"),
            *("--mmap_dir", str(tmp_path / "emb")),
            *("--manifest_csv", str(tmp_path / "manifest.csv")),
            *("--output_dir", str(out), "--label_column", "label", "--patient_column", "patient"),
            *("--inner_splits", "0"),
        )

        summary = json.loads((out / "summary.json").read_text())
        assert summary["protocol"] == "external_test"  # predefined split runs the same path
        assert summary["n_train_slides"] == int((df["split"] == "train").sum())
        assert summary["n_test_patients"] == df.loc[df["split"] == "test", "patient_id"].nunique()

    def test_external_test_requires_the_test_cohort(self, cohort_files, monkeypatch):
        _, root = cohort_files
        with pytest.raises(ValueError, match="requires both --test_mmap_dir"):
            self._main(
                monkeypatch,
                *("--protocol", "external_test", "--mmap_dir", str(root / "emb")),
                *("--manifest_csv", str(root / "manifest.csv"), "--output_dir", str(root / "o")),
                *("--label_column", "label", "--patient_column", "patient"),
            )
