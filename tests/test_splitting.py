"""Focused tests for split-profile and training-dispatch contracts."""

from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir


def test_kfold_profile_preserves_unused_null_ratios():
    from oceanpath.workflows.splitting import build_split_config

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="make_splits", overrides=["splits=kfold5"])

    spec = build_split_config(cfg)

    assert spec.train_ratio is None
    assert spec.val_ratio is None
    assert spec.test_ratio is None


def test_kfold_without_test_ratio_uses_every_group_for_cross_validation(tmp_path):
    from oceanpath.splitting import SplitConfig, generate_splits

    manifest = tmp_path / "manifest.csv"
    rows = [
        {"filename": f"slide_{index:02d}.svs", "patient_id": f"p{index:02d}", "label": index % 2}
        for index in range(20)
    ]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    spec = SplitConfig(
        scheme="kfold",
        name="test_kfold",
        csv_path=str(manifest),
        output_dir=str(tmp_path / "splits"),
        filename_column="filename",
        label_column="label",
        group_column="patient_id",
        n_folds=5,
        test_ratio=None,
    )

    generated = generate_splits(spec)
    splits = pd.read_parquet(generated.parquet_path)

    assert len(splits) == len(rows)
    assert set(splits["fold"]) == set(range(5))
    assert (splits["fold"] >= 0).all()


def test_monte_carlo_fold_dispatch_selects_requested_repeat():
    from oceanpath.splitting import get_slide_ids_for_fold

    splits = pd.DataFrame(
        {
            "slide_id": ["r0_train", "r0_val", "r0_test", "r1_train", "r1_val", "r1_test"],
            "repeat": [0, 0, 0, 1, 1, 1],
            "split": ["train", "val", "test", "train", "val", "test"],
        }
    )

    selected = get_slide_ids_for_fold(splits, fold=1, scheme="monte_carlo")

    assert selected == {
        "train": ["r1_train"],
        "val": ["r1_val"],
        "test": ["r1_test"],
    }


def test_nested_cv_is_not_an_advertised_foundation_profile():
    from oceanpath.splitting import SUPPORTED_SPLIT_SCHEMES

    split_profiles = Path(__file__).resolve().parents[1] / "configs" / "splits"

    assert "nested_cv" not in SUPPORTED_SPLIT_SCHEMES
    assert not (split_profiles / "nested_5x3.yaml").exists()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    for profile_path in split_profiles.glob("*.yaml"):
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="make_splits", overrides=[f"splits={profile_path.stem}"])
        assert cfg.splits.scheme in SUPPORTED_SPLIT_SCHEMES
