"""
Tests for split generation and loading (``oceanpath.data.splits``).

The synthetic manifest has 40 patients (labels alternate, 20 per class) holding
1-3 slides each (79 slides), two sites, and a per-slide ``grade`` that is 2 on
the first slide of every fifth patient, so a filter on it matches only part of
some patients.

Covers:
  * manifest loading: slide-id normalisation, required columns, duplicates and
    missing labels, missing group column, unknown scheme
  * every scheme (holdout, kfold with and without a held-out test set, custom
    holdout / k-fold, Monte Carlo, nested CV): no patient in two partitions,
    every slide assigned, label balance, determinism per seed
  * get_slide_ids_for_fold for each scheme
  * outputs: splits.parquet, summary.json, the integrity hash, reuse unless
    forced; verify_split_integrity and load_splits on tampered files
  * site-balance warning and feature verification
"""

import hashlib
import json
import logging

import pandas as pd
import pytest

from oceanpath.data.splits import (
    SplitConfig,
    SplitIntegrityError,
    _check_site_balance,
    _normalize_slide_id,
    generate_splits,
    get_slide_ids_for_fold,
    load_splits,
    verify_split_integrity,
)

N_PATIENTS = 40
N_SLIDES = 79  # patient p holds 1 + p % 3 slides


def _manifest_rows(n_patients: int = N_PATIENTS) -> list[dict]:
    rows = []
    for p in range(n_patients):
        for s in range(1 + p % 3):
            rows.append(
                {
                    "filename": f"P{p:02d}_S{s}.svs",
                    "patient_id": f"P{p:02d}",
                    "label": p % 2,
                    "site": "north" if (p // 2) % 2 == 0 else "south",
                    "grade": 2 if p % 5 == 0 and s == 0 else 1,
                }
            )
    return rows


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(_manifest_rows()).to_csv(path, index=False)
    return path


def _config(manifest, out, **kwargs) -> SplitConfig:
    kwargs.setdefault("scheme", "kfold")
    kwargs.setdefault("group_column", "patient_id")
    return SplitConfig(name="test", csv_path=str(manifest), output_dir=str(out), **kwargs)


def _generate(manifest, out, **kwargs):
    result = generate_splits(_config(manifest, out, **kwargs))
    return result, pd.read_parquet(result.parquet_path)


def _assert_patients_disjoint(df: pd.DataFrame, column: str) -> None:
    """Every patient's slides share one value of ``column``."""
    spread = df.groupby("patient_id")[column].nunique()
    assert (spread == 1).all(), spread[spread > 1].to_dict()


def _patient_labels(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Patients per (partition, label)."""
    patients = df.drop_duplicates("patient_id")
    return pd.crosstab(patients[column], patients["label"])


def _assert_partition(ids: dict[str, list[str]], all_ids) -> None:
    train, val, test = (set(ids[k]) for k in ("train", "val", "test"))
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(all_ids)


# ═════════════════════════════════════════════════════════════════════════════
# Manifest loading
# ═════════════════════════════════════════════════════════════════════════════


class TestManifest:
    @pytest.mark.parametrize(
        ("raw", "slide_id"),
        [
            ("a.svs", "a"),
            ("dir\\b.H5", "dir/b"),
            (" c.sdpc ", "c"),
            ("TCGA-AA-1234-01Z-00-DX1.8923A151-A690.svs", "TCGA-AA-1234-01Z-00-DX1.8923A151-A690"),
            ("TCGA-AA-1234-01Z-00-DX1.8923A151-A690", "TCGA-AA-1234-01Z-00-DX1.8923A151-A690"),
            ("cohort/slide.v2.tiff", "cohort/slide.v2"),
        ],
    )
    def test_slide_ids_drop_known_extensions_only(self, raw, slide_id):
        assert _normalize_slide_id(raw) == slide_id

    def test_missing_manifest_or_columns_raise(self, tmp_path, manifest):
        with pytest.raises(FileNotFoundError):
            generate_splits(_config(tmp_path / "nowhere.csv", tmp_path / "out"))
        with pytest.raises(ValueError, match="Filename column 'path'"):
            generate_splits(_config(manifest, tmp_path / "out", filename_column="path"))
        with pytest.raises(ValueError, match="Label column 'grade_x'"):
            generate_splits(_config(manifest, tmp_path / "out", label_column="grade_x"))

    def test_unknown_scheme_raises(self, tmp_path, manifest):
        with pytest.raises(ValueError, match="Unknown scheme"):
            generate_splits(_config(manifest, tmp_path / "out", scheme="bootstrap"))

    def test_duplicate_slides_and_missing_labels_are_dropped(self, tmp_path):
        rows = _manifest_rows()
        rows.append({**rows[0], "filename": "P00_S0.tiff"})  # same slide, other format
        rows.append({**rows[1], "filename": "P01_S9.svs", "label": None})
        path = tmp_path / "manifest.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        result, df = _generate(path, tmp_path / "out", test_ratio=0.0)
        assert result.n_slides == len(df) == N_SLIDES
        assert df["slide_id"].is_unique
        assert "P01_S9" not in set(df["slide_id"])

    def test_missing_group_column_falls_back_to_one_group_per_slide(self, tmp_path, manifest):
        result, df = _generate(manifest, tmp_path / "out", group_column="case", test_ratio=0.0)
        assert result.n_groups == N_SLIDES
        assert (df["group_id"] == df["slide_id"]).all()


# ═════════════════════════════════════════════════════════════════════════════
# Holdout
# ═════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    RATIOS = {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25}

    def test_ratios_must_sum_to_one(self, tmp_path, manifest):
        with pytest.raises(ValueError, match="sum to 1.0"):
            _generate(manifest, tmp_path / "out", scheme="holdout", test_ratio=0.3)

    def test_patients_never_span_two_partitions(self, tmp_path, manifest):
        _, df = _generate(manifest, tmp_path / "out", scheme="holdout", **self.RATIOS)
        assert len(df) == N_SLIDES
        assert set(df["split"]) == {"train", "val", "test"}
        _assert_patients_disjoint(df, "split")

    def test_partition_sizes_and_labels_follow_the_ratios(self, tmp_path, manifest):
        _, df = _generate(manifest, tmp_path / "out", scheme="holdout", **self.RATIOS)
        per_label = _patient_labels(df, "split")
        for split, share in [("train", 0.5), ("val", 0.25), ("test", 0.25)]:
            assert abs(per_label.loc[split].sum() - share * N_PATIENTS) <= 1
            assert abs(per_label.loc[split, "0"] - per_label.loc[split, "1"]) <= 1

    def test_deterministic_per_seed(self, tmp_path, manifest):
        _, first = _generate(manifest, tmp_path / "a", scheme="holdout", seed=7, **self.RATIOS)
        _, again = _generate(manifest, tmp_path / "b", scheme="holdout", seed=7, **self.RATIOS)
        _, other = _generate(manifest, tmp_path / "c", scheme="holdout", seed=8, **self.RATIOS)
        pd.testing.assert_frame_equal(first, again)
        assert not first["split"].equals(other["split"])


# ═════════════════════════════════════════════════════════════════════════════
# K-fold
# ═════════════════════════════════════════════════════════════════════════════


class TestKFold:
    def test_patient_never_spans_two_folds(self, tmp_path, manifest):
        result, df = _generate(manifest, tmp_path / "out", n_folds=5, test_ratio=0.0)
        assert len(df) == N_SLIDES
        assert set(df["fold"]) == set(range(5))
        _assert_patients_disjoint(df, "fold")
        assert result.n_groups == N_PATIENTS
        # Folds are balanced in patients (slide counts vary with slides per patient).
        assert df.drop_duplicates("patient_id")["fold"].value_counts().tolist() == [8] * 5

    @pytest.mark.xfail(
        strict=True,
        reason="StratifiedGroupKFold(shuffle=True) shuffles the per-group label counts "
        "apart from the group ids (scikit-learn 1.7), so patient-grouped folds are not "
        "label-stratified",
    )
    def test_patient_grouped_folds_are_label_stratified(self, tmp_path, manifest):
        _, df = _generate(manifest, tmp_path / "out", n_folds=5, test_ratio=0.0)
        per_fold = _patient_labels(df, "fold")
        assert (abs(per_fold["0"] - per_fold["1"]) <= 1).all(), per_fold

    def test_without_patients_slides_are_stratified_individually(self, tmp_path, manifest):
        _, df = _generate(manifest, tmp_path / "out", group_column=None, test_ratio=0.0)
        assert (df["group_id"] == df["slide_id"]).all()
        per_fold = pd.crosstab(df["fold"], df["label"])
        for label in per_fold.columns:
            assert per_fold[label].max() - per_fold[label].min() <= 1

    def test_test_ratio_holds_out_whole_patients_before_folding(self, tmp_path, manifest):
        _, df = _generate(manifest, tmp_path / "out", n_folds=4, test_ratio=0.25)
        _assert_patients_disjoint(df, "fold")
        per_label = _patient_labels(df, "fold")
        assert per_label.loc[-1].tolist() == [5, 5]  # 10 of 40 patients, stratified
        assert set(df["fold"]) == {-1, 0, 1, 2, 3}

    def test_fold_ids_split_into_train_val_and_held_out_test(self, tmp_path, manifest):
        _, df = _generate(manifest, tmp_path / "out", n_folds=4, test_ratio=0.25)
        held_out = set(df.loc[df["fold"] == -1, "slide_id"])
        for fold in range(4):
            ids = get_slide_ids_for_fold(df, fold, scheme="kfold")
            _assert_partition(ids, df["slide_id"])
            assert set(ids["val"]) == set(df.loc[df["fold"] == fold, "slide_id"])
            assert set(ids["test"]) == held_out

    def test_deterministic_per_seed(self, tmp_path, manifest):
        _, first = _generate(manifest, tmp_path / "a", seed=3, test_ratio=0.0)
        _, again = _generate(manifest, tmp_path / "b", seed=3, test_ratio=0.0)
        _, other = _generate(manifest, tmp_path / "c", seed=4, test_ratio=0.0)
        pd.testing.assert_frame_equal(first, again)
        assert not first["fold"].equals(other["fold"])


# ═════════════════════════════════════════════════════════════════════════════
# Custom test sets
# ═════════════════════════════════════════════════════════════════════════════

# Patients whose first slide is grade 2: every fifth patient, 16 slides in all.
GRADE2_PATIENTS = {f"P{p:02d}" for p in range(0, N_PATIENTS, 5)}


class TestCustomSchemes:
    @pytest.mark.parametrize("scheme", ["custom_holdout", "custom_kfold"])
    def test_test_filter_is_required(self, tmp_path, manifest, scheme):
        with pytest.raises(ValueError, match="test_filter"):
            _generate(manifest, tmp_path / "out", scheme=scheme)

    def test_custom_holdout_expands_the_filter_to_whole_patients(self, tmp_path, manifest):
        _, df = _generate(
            manifest,
            tmp_path / "out",
            scheme="custom_holdout",
            test_filter="grade == 2",
            train_ratio=0.75,
            val_ratio=0.25,
        )
        test = df[df["split"] == "test"]
        assert set(test["patient_id"]) == GRADE2_PATIENTS
        assert len(test) == 16  # all slides of those patients, not just the grade-2 ones
        _assert_patients_disjoint(df, "split")
        val_patients = df.loc[df["split"] == "val", "patient_id"].nunique()
        assert abs(val_patients - 0.25 * (N_PATIENTS - len(GRADE2_PATIENTS))) <= 1

    def test_custom_holdout_val_filter(self, tmp_path, manifest):
        _, df = _generate(
            manifest,
            tmp_path / "out",
            scheme="custom_holdout",
            test_filter="grade == 2",
            val_filter="site == 'north'",
            site_column="site",
        )
        assert set(df.loc[df["split"] == "test", "patient_id"]) == GRADE2_PATIENTS
        val = df[df["split"] == "val"]
        assert not val.empty and (val["site"] == "north").all()
        train_north = df[(df["split"] == "train") & (df["site"] == "north")]
        assert train_north.empty
        ids = get_slide_ids_for_fold(df, 0, scheme="custom_holdout")
        _assert_partition(ids, df["slide_id"])
        assert set(ids["val"]) == set(val["slide_id"])

    def test_custom_kfold_keeps_a_fixed_test_set_for_every_fold(self, tmp_path, manifest):
        _, df = _generate(
            manifest, tmp_path / "out", scheme="custom_kfold", test_filter="grade == 2", n_folds=4
        )
        test = df[df["fold"] == -1]
        assert set(test["patient_id"]) == GRADE2_PATIENTS
        assert (test["split"] == "test").all()
        pool = df[df["fold"] != -1]
        assert set(pool["fold"]) == {0, 1, 2, 3}
        assert (pool["split"] == "train_val_pool").all()
        _assert_patients_disjoint(df, "fold")
        for fold in range(4):
            ids = get_slide_ids_for_fold(df, fold, scheme="custom_kfold")
            _assert_partition(ids, df["slide_id"])
            assert set(ids["test"]) == set(test["slide_id"])

    def test_custom_kfold_rejects_a_bad_filter_or_too_few_slides(self, tmp_path, manifest):
        with pytest.raises(ValueError, match="Failed to evaluate"):
            _generate(manifest, tmp_path / "a", scheme="custom_kfold", test_filter="grade ===")
        with pytest.raises(ValueError, match="Not enough data"):
            _generate(manifest, tmp_path / "b", scheme="custom_kfold", test_filter="grade > 0")


# ═════════════════════════════════════════════════════════════════════════════
# Monte Carlo and nested CV
# ═════════════════════════════════════════════════════════════════════════════


class TestMonteCarlo:
    def test_each_repeat_is_a_patient_safe_holdout(self, tmp_path, manifest):
        _, df = _generate(
            manifest,
            tmp_path / "out",
            scheme="monte_carlo",
            n_repeats=3,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )
        assert len(df) == 3 * N_SLIDES
        test_sets = []
        for repeat in range(3):
            rows = df[df["repeat"] == repeat]
            assert rows["slide_id"].is_unique and len(rows) == N_SLIDES
            _assert_patients_disjoint(rows, "split")
            ids = get_slide_ids_for_fold(df, repeat, scheme="monte_carlo")
            _assert_partition(ids, rows["slide_id"])
            assert set(ids["test"]) == set(rows.loc[rows["split"] == "test", "slide_id"])
            test_sets.append(frozenset(ids["test"]))
        assert len(set(test_sets)) == 3  # each repeat draws its own split


class TestNestedCV:
    @pytest.fixture(scope="class")
    def nested(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("nested")
        pd.DataFrame(_manifest_rows()).to_csv(root / "manifest.csv", index=False)
        _, df = _generate(
            root / "manifest.csv", root / "out", scheme="nested_cv", n_folds=4, n_inner_folds=3
        )
        return df

    def test_every_slide_is_outer_test_exactly_once(self, nested):
        assert len(nested) == 4 * N_SLIDES
        outer_test = nested[nested["inner_fold"] == -1]
        assert outer_test["slide_id"].is_unique
        assert len(outer_test) == N_SLIDES

    def test_inner_folds_partition_each_outer_training_set_by_patient(self, nested):
        for outer in range(4):
            rows = nested[nested["outer_fold"] == outer]
            assert rows["slide_id"].is_unique and len(rows) == N_SLIDES
            assert set(rows["inner_fold"]) == {-1, 0, 1, 2}
            _assert_patients_disjoint(rows, "inner_fold")  # incl. outer test vs inner folds

    def test_fold_ids_need_an_outer_fold(self, nested):
        with pytest.raises(ValueError, match="outer_fold"):
            get_slide_ids_for_fold(nested, 0, scheme="nested_cv")
        for outer in range(4):
            rows = nested[nested["outer_fold"] == outer]
            for inner in range(3):
                ids = get_slide_ids_for_fold(nested, inner, scheme="nested_cv", outer_fold=outer)
                _assert_partition(ids, rows["slide_id"])
                assert set(ids["test"]) == set(rows.loc[rows["inner_fold"] == -1, "slide_id"])
                assert set(ids["val"]) == set(rows.loc[rows["inner_fold"] == inner, "slide_id"])


# ═════════════════════════════════════════════════════════════════════════════
# Outputs, integrity and loading
# ═════════════════════════════════════════════════════════════════════════════


class TestOutputsAndIntegrity:
    def test_outputs_and_integrity_hash(self, tmp_path, manifest):
        result, df = _generate(manifest, tmp_path / "out", n_folds=4, test_ratio=0.0)
        out = tmp_path / "out"
        expected = hashlib.sha256(manifest.read_bytes() + (out / "splits.parquet").read_bytes())
        stored = json.loads((out / ".integrity_hash").read_text())
        assert stored == {"hash": expected.hexdigest(), "csv_path": str(manifest)}
        assert result.integrity_hash == expected.hexdigest()

        assert (result.n_slides, result.n_groups) == (N_SLIDES, N_PATIENTS)
        assert {str(k): int(v) for k, v in result.label_distribution.items()} == {
            "0": int((df["label"] == "0").sum()),
            "1": int((df["label"] == "1").sum()),
        }
        assert sum(result.fold_distribution.values()) == N_SLIDES

        summary = json.loads((out / "summary.json").read_text())
        assert summary["scheme"] == "kfold"
        assert (summary["n_slides"], summary["n_groups"]) == (N_SLIDES, N_PATIENTS)
        assert sum(summary[f"fold_{k}_count"] for k in range(4)) == N_SLIDES

    def test_existing_splits_are_reused_unless_forced(self, tmp_path, manifest):
        cfg = _config(manifest, tmp_path / "out", test_ratio=0.0)
        first = generate_splits(cfg)
        parquet = (tmp_path / "out" / "splits.parquet").read_bytes()
        pd.DataFrame(_manifest_rows(30)).to_csv(manifest, index=False)  # manifest shrinks

        reused = generate_splits(cfg)
        assert reused.n_slides == first.n_slides == N_SLIDES
        assert reused.integrity_hash == first.integrity_hash
        assert (tmp_path / "out" / "splits.parquet").read_bytes() == parquet

        regenerated = generate_splits(cfg, force=True)
        assert regenerated.n_slides == len(pd.read_parquet(regenerated.parquet_path)) < N_SLIDES

    def test_integrity_check_catches_a_tampered_split_or_manifest(self, tmp_path, manifest):
        out = tmp_path / "out"
        _, df = _generate(manifest, out, n_folds=4, test_ratio=0.0)
        assert verify_split_integrity(out, manifest) is True
        original = (out / "splits.parquet").read_bytes()

        moved = df.copy()
        moved.loc[0, "fold"] = (moved.loc[0, "fold"] + 1) % 4  # one slide changes fold
        moved.to_parquet(out / "splits.parquet", index=False)
        with pytest.raises(SplitIntegrityError):
            verify_split_integrity(out, manifest)
        with pytest.raises(SplitIntegrityError):
            load_splits(out, csv_path=manifest)
        assert load_splits(out, csv_path=manifest, verify=False).loc[0, "fold"] != df.loc[0, "fold"]

        (out / "splits.parquet").write_bytes(original)  # restore the split ...
        assert verify_split_integrity(out, manifest) is True
        with manifest.open("a") as f:  # ... then change the manifest
            f.write("P99_S0.svs,P99,1,north,1\n")
        with pytest.raises(SplitIntegrityError):
            verify_split_integrity(out, manifest)

    def test_integrity_check_without_a_hash_or_a_split_file(self, tmp_path, manifest):
        out = tmp_path / "out"
        _, df = _generate(manifest, out, test_ratio=0.0)
        (out / ".integrity_hash").unlink()
        assert verify_split_integrity(out, manifest) is True  # nothing to compare against
        pd.testing.assert_frame_equal(load_splits(out, csv_path=manifest), df)
        (out / "splits.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            load_splits(out)
        (out / ".integrity_hash").write_text(json.dumps({"hash": "0" * 64}))
        with pytest.raises(FileNotFoundError):
            verify_split_integrity(out, manifest)

    def test_site_column_is_kept_and_checked(self, tmp_path, manifest):
        _, df = _generate(manifest, tmp_path / "out", site_column="site", test_ratio=0.0)
        assert df.groupby("slide_id")["site"].first().to_dict() == {
            f"P{p:02d}_S{s}": "north" if (p // 2) % 2 == 0 else "south"
            for p in range(N_PATIENTS)
            for s in range(1 + p % 3)
        }

    def test_site_confined_to_one_fold_is_flagged(self, caplog):
        cfg = SplitConfig(scheme="kfold", name="t", csv_path="", output_dir="", site_column="site")
        balanced = pd.DataFrame({"fold": [0, 0, 1, 1], "site": ["a", "b", "a", "b"]})
        confined = pd.DataFrame({"fold": [0, 0, 1, 1], "site": ["a", "a", "b", "b"]})
        with caplog.at_level(logging.INFO, logger="oceanpath.data.splits"):
            _check_site_balance(balanced, cfg)
            assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
            _check_site_balance(confined, cfg)
            assert [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_feature_verification_does_not_filter_slides(self, tmp_path, manifest, caplog):
        h5_dir = tmp_path / "h5"
        h5_dir.mkdir()
        (h5_dir / "P00_S0.h5").touch()
        with caplog.at_level(logging.WARNING, logger="oceanpath.data.splits"):
            result, _ = _generate(manifest, tmp_path / "out", feature_h5_dir=str(h5_dir))
        assert result.n_slides == N_SLIDES
        assert [r for r in caplog.records if r.levelno == logging.WARNING]
