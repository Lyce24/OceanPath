import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _make_probe_df(n_patients: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n_patients // 2), dtype=int)
    signal = labels.astype(np.float32) * 2.0 + rng.normal(0, 0.1, size=n_patients)
    noise = rng.normal(0, 1.0, size=n_patients)
    return pd.DataFrame(
        {
            "slide_id": [f"s{i:03d}" for i in range(n_patients)],
            "patient_id": [f"p{i:03d}" for i in range(n_patients)],
            "label": labels,
            "embedding": [
                np.asarray([signal[i], noise[i]], dtype=np.float32) for i in range(n_patients)
            ],
        }
    )


def _probe_args(output_dir, protocol: str = "grouped_cv") -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(output_dir),
        protocol=protocol,
        n_splits=2,
        inner_splits=0,
        primary_metric="auroc",
        c_grid=[1.0],
        class_weight="balanced",
        max_iter=1000,
        multi_class_mode="auto",
        aggregate_to_patient=True,
        n_boot=0,
        seed=7,
        label_fraction=0.25,
        label_fraction_seed=123,
        save_model=False,
    )


def test_subsample_labeled_training_data_keeps_classes_and_groups():
    from oceanpath.modules.linear_probing_sklearn import subsample_labeled_training_data

    df = _make_probe_df(n_patients=20)
    extra = df.iloc[[0]].copy()
    extra["slide_id"] = "s000_section2"
    df = pd.concat([df, extra], ignore_index=True)

    sampled, info = subsample_labeled_training_data(
        train_df=df,
        label_fraction=0.2,
        seed=11,
        context="unit-test",
    )

    assert set(sampled["label"].unique()) == {0, 1}
    assert info["train_labeled_patients"] == 4
    assert info["train_full_patients"] == 20

    # Patient-level sampling keeps all slides from a chosen patient together.
    selected_counts = sampled.groupby("patient_id")["slide_id"].count()
    if "p000" in selected_counts:
        assert selected_counts.loc["p000"] == 2


def test_subsample_labeled_training_data_backfills_labels_from_mixed_groups():
    from oceanpath.modules.linear_probing_sklearn import subsample_labeled_training_data

    df = _make_probe_df(n_patients=20)
    df = df[df["label"] == 0].copy()
    df.loc[df.index[0], "label"] = 1
    df.loc[df.index[0], "patient_id"] = "mixed"
    df.loc[df.index[1], "patient_id"] = "mixed"

    sampled, info = subsample_labeled_training_data(
        train_df=df,
        label_fraction=0.1,
        seed=13,
        context="mixed-label-test",
    )

    assert set(sampled["label"].unique()) == {0, 1}
    assert info["train_labeled_classes"] == 2


def test_grouped_cv_label_fraction_keeps_full_validation_sets(tmp_path):
    from oceanpath.modules.linear_probing_sklearn import run_grouped_cv

    df = _make_probe_df(n_patients=40)
    out = tmp_path / "grouped"
    out.mkdir()

    summary = run_grouped_cv(_probe_args(out), df)
    oof = pd.read_csv(out / "patient_predictions_oof.csv")

    assert summary["label_fraction"] == 0.25
    assert summary["mean_train_labeled_patient_fraction"] < 1.0
    assert len(oof) == len(df)


def test_external_test_label_fraction_subsamples_train_only(tmp_path):
    from oceanpath.modules.linear_probing_sklearn import run_external_test

    train_df = _make_probe_df(n_patients=40, seed=1)
    test_df = _make_probe_df(n_patients=20, seed=2)
    out = tmp_path / "external"
    out.mkdir()

    summary = run_external_test(_probe_args(out, protocol="external_test"), train_df, test_df)
    preds = pd.read_csv(out / "patient_predictions_test.csv")

    assert summary["label_fraction"] == 0.25
    assert summary["n_train_patients"] < summary["n_train_patients_full"]
    assert summary["n_test_patients"] == 20
    assert len(preds) == len(test_df)


def test_subsample_labels_continuous_falls_back_to_unstratified():
    from oceanpath.modules.linear_probing_sklearn import subsample_labeled_training_data

    df = _make_probe_df(n_patients=30)
    # Inject continuous regression-style targets.
    rng = np.random.default_rng(0)
    df = df.copy()
    df["label"] = rng.uniform(-1.0, 1.0, size=len(df)).astype(np.float64)

    sampled, info = subsample_labeled_training_data(
        train_df=df,
        label_fraction=0.4,
        seed=5,
        context="continuous-test",
    )

    assert info["label_fraction_stratified"] is False
    # No class structure — should still produce a non-empty patient subset.
    assert info["train_labeled_patients"] >= 1
    assert info["train_labeled_patients"] <= info["train_full_patients"]
    # All slides for selected patients are kept together.
    selected = set(sampled["patient_id"].astype(str).unique())
    for pid in selected:
        assert (sampled["patient_id"].astype(str) == pid).sum() == (
            df["patient_id"].astype(str) == pid
        ).sum()


def test_subsample_records_categorical_flag():
    from oceanpath.modules.linear_probing_sklearn import subsample_labeled_training_data

    df = _make_probe_df(n_patients=20)
    _, info = subsample_labeled_training_data(
        train_df=df,
        label_fraction=0.5,
        seed=1,
        context="cat-flag-test",
    )
    assert info["label_fraction_stratified"] is True


def test_select_best_c_falls_back_when_too_few_groups(tmp_path):
    from oceanpath.modules.linear_probing_sklearn import select_best_c

    df = _make_probe_df(n_patients=4)
    # 4 patients, 3-fold inner CV would normally pass the n_groups check
    # but min_class_count=2 still triggers warning paths cleanly. Here we
    # force a very thin split to trigger the fallback explicitly.
    df = df.iloc[:2].reset_index(drop=True)
    chosen, search_df = select_best_c(
        train_df=df,
        task_type="binary",
        c_grid=[0.01, 0.1, 1.0, 10.0],
        inner_splits=3,
        primary_metric="auroc",
        class_weight="balanced",
        max_iter=200,
        multi_class_mode="auto",
        seed=0,
    )

    # Grid midpoint of [0.01, 0.1, 1.0, 10.0] (sorted) = grid[2] = 1.0.
    assert chosen == 1.0
    assert "fallback" in search_df.columns
    assert bool(search_df["fallback"].iloc[0]) is True


def test_select_best_c_falls_back_when_class_count_below_inner_splits(tmp_path):
    from oceanpath.modules.linear_probing_sklearn import select_best_c

    df = _make_probe_df(n_patients=4)
    chosen, search_df = select_best_c(
        train_df=df,
        task_type="binary",
        c_grid=[0.01, 0.1, 1.0, 10.0],
        inner_splits=3,
        primary_metric="auroc",
        class_weight="balanced",
        max_iter=200,
        multi_class_mode="auto",
        seed=0,
    )

    assert chosen == 1.0
    assert "fallback" in search_df.columns
    assert bool(search_df["fallback"].iloc[0]) is True
    assert int(search_df["min_class_count"].iloc[0]) == 2


def test_select_best_c_runs_inner_cv_when_data_sufficient(tmp_path):
    from oceanpath.modules.linear_probing_sklearn import select_best_c

    df = _make_probe_df(n_patients=60)
    chosen, search_df = select_best_c(
        train_df=df,
        task_type="binary",
        c_grid=[0.1, 1.0, 10.0],
        inner_splits=3,
        primary_metric="auroc",
        class_weight="balanced",
        max_iter=500,
        multi_class_mode="auto",
        seed=0,
    )

    assert chosen in {0.1, 1.0, 10.0}
    # Real inner-CV results have one row per (C, inner_fold), no fallback flag.
    assert "fallback" not in search_df.columns
    assert "score" in search_df.columns
    assert "inner_fold" in search_df.columns


def test_collect_lp_results_aggregates_seeded_label_efficiency_layout(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "collect_lp_results.py"
    spec = importlib.util.spec_from_file_location("collect_lp_results", module_path)
    assert spec is not None
    collector = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(collector)

    root = tmp_path / "sweep"
    for seed, auroc in [(42, 0.7), (43, 0.9)]:
        out_dir = root / "frac_1pct" / "nsclc_cv__mean_pool" / f"seed_{seed}" / "mean_pool_cap2048"
        out_dir.mkdir(parents=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "protocol": "grouped_cv",
                    "task_type": "binary",
                    "label_fraction": 0.01,
                    "mean_auroc": auroc,
                    "std_auroc": 0.0,
                    "mean_auprc": 0.5,
                    "mean_balanced_acc": 0.6,
                    "mean_macro_f1": 0.55,
                },
                f,
            )

    rows = collector.collect(root, by_seed=False)

    assert len(rows) == 1
    row = rows[0]
    assert row["Task"] == "NSCLC subtyping"
    assert row["Fraction"] == "1%"
    assert row["Mode"] == "Mean Pool"
    assert row["Seeds"] == 2
    assert row["AUROC"].startswith("0.800 +/- 0.141")
