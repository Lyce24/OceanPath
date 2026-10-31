"""
Integration test of scripts/train.py at toy scale: two k-fold folds through
``run_fold``, then ``aggregate_cv_results`` and Phase-2 ``finalize_models``,
with W&B offline and the training settings of configs/training/default.yaml.

Marked slow: two folds plus a refit cost several seconds on CPU, mostly
Lightning and progress-bar overhead. The pieces are covered quickly in
tests/test_train_module.py and tests/test_finalize.py; this pins the glue.

Covers:
  * run_fold writes the per-fold files that evaluation and aggregation read,
    and a fold_metrics.json whose best_epoch comes from the best checkpoint
  * aggregate_cv_results: every non-test slide once in oof_predictions.parquet
    with its fold, one test row per fold and test slide, fold mean / std
  * finalize_models on those folds: best fold = lowest val/loss, refit epochs
    from the folds' best epochs
"""

import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from oceanpath.modules.finalize import finalize_models

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[1]
IN_DIM = 16
# (n_patches, label, fold); fold -1 is the held-out test set
COHORT = [
    (12, 0, 0),
    (9, 1, 0),
    (15, 0, 0),
    (11, 1, 0),
    (14, 0, 1),
    (8, 1, 1),
    (10, 0, 1),
    (13, 1, 1),
    (16, 0, -1),
    (7, 1, -1),
]
FOLD_FILES = [
    "config.yaml",
    "metadata.json",
    "batch_indices.npy",
    "checkpoints/last.ckpt",
    "preds_val.parquet",
    "embeddings_val.npz",
    "preds_test.parquet",
    "embeddings_test.npz",
    "fold_metrics.json",
]


def load_train_script():
    spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv_run(build_feature_store, tmp_path_factory):
    store = build_feature_store([n for n, _, _ in COHORT], feat_dim=IN_DIM, seed=9)
    root = tmp_path_factory.mktemp("train_script")
    folds = {sid: fold for sid, (*_, fold) in zip(store.slide_ids, COHORT, strict=True)}
    pd.DataFrame(
        {"filename": [f"{s}.h5" for s in store.slide_ids], "label": [lab for _, lab, _ in COHORT]}
    ).to_csv(root / "manifest.csv", index=False)
    splits_dir = root / "splits" / "toy" / "kfold2"
    splits_dir.mkdir(parents=True)
    pd.DataFrame({"slide_id": list(folds), "fold": list(folds.values())}).to_parquet(
        splits_dir / "splits.parquet", index=False
    )

    training = OmegaConf.load(REPO / "configs" / "training" / "default.yaml")
    training.merge_with(
        {
            "max_epochs": 2,
            "warmup_epochs": 0,
            "batch_size": 2,
            "max_instances": 64,
            "verify_splits": False,
        }
    )
    cfg = OmegaConf.create(
        {
            "exp_name": "toy_abmil",
            "wandb": {"offline": True, "project": "unused", "group": "toy", "tags": []},
            "platform": {
                "splits_root": str(root / "splits"),
                "output_root": str(root / "outputs"),
                "accelerator": "cpu",
                "devices": 1,
                "strategy": "auto",
                "precision": "32-true",
                "num_workers": 0,
            },
            "data": {
                "name": "toy",
                "mmap_dir": str(store.mmap_dir),
                "csv_path": str(root / "manifest.csv"),
                "label_columns": ["label"],
                "filename_column": "filename",
            },
            "splits": {"name": "kfold2", "scheme": "kfold", "n_folds": 2},
            "encoder": {"name": "synthetic", "feature_dim": IN_DIM},
            "model": {
                "name": "abmil",
                "arch": "abmil",
                "embed_dim": 8,
                "attn_dim": 8,
                "dropout": 0.0,
            },
            "training": training,
        }
    )

    script = load_train_script()
    output_dir = script._get_output_dir(cfg)
    n_folds = script._get_n_folds(cfg)
    fold_metrics = [script.run_fold(cfg, fold_idx=k, output_dir=output_dir) for k in range(n_folds)]
    summary = script.aggregate_cv_results(output_dir, n_folds)
    finalized = finalize_models(cfg, output_dir, n_folds=n_folds, all_fold_metrics=fold_metrics)
    return SimpleNamespace(
        out=output_dir,
        folds=folds,
        test_ids=sorted(s for s, f in folds.items() if f == -1),
        fold_metrics=fold_metrics,
        summary=summary,
        finalized=finalized,
    )


def test_each_fold_writes_the_files_downstream_reads(cv_run):
    assert cv_run.out == cv_run.out.parent / "toy_abmil"
    for k in range(2):
        for name in FOLD_FILES:
            assert (cv_run.out / f"fold_{k}" / name).is_file(), f"fold_{k}/{name}"


def test_fold_metrics_record_val_test_and_the_best_epoch_of_the_checkpoint(cv_run):
    for k, returned in enumerate(cv_run.fold_metrics):
        saved = json.loads((cv_run.out / f"fold_{k}" / "fold_metrics.json").read_text())
        assert saved == json.loads(json.dumps(returned, default=str))
        assert {"val/loss", "val/auroc", "test/loss", "test/auroc"} <= saved.keys()
        epoch = int(re.search(r"best-epoch=(\d+)-val", saved["best_checkpoint"]).group(1))
        assert saved["best_epoch"] == epoch + 1
        assert saved["val/loss"] == pytest.approx(saved["best_model_score"], abs=1e-5)


def test_oof_predictions_hold_each_non_test_slide_once_with_its_fold(cv_run):
    oof = pd.read_parquet(cv_run.out / "oof_predictions.parquet")
    assert list(oof.columns) == ["slide_id", "label", "prob_1", "fold"]
    assert sorted(oof["slide_id"]) == sorted(s for s, f in cv_run.folds.items() if f >= 0)
    assert all(cv_run.folds[s] == f for s, f in zip(oof["slide_id"], oof["fold"], strict=True))
    test = pd.read_parquet(cv_run.out / "test_predictions.parquet")
    assert sorted(test["slide_id"]) == sorted(cv_run.test_ids * 2)  # one row per fold
    assert cv_run.summary["n_oof_slides"] == len(oof)


def test_cv_summary_is_the_mean_and_std_over_folds(cv_run):
    losses = [fm["val/loss"] for fm in cv_run.fold_metrics]
    saved = json.loads((cv_run.out / "cv_summary.json").read_text())
    assert saved["val/loss_mean"] == pytest.approx(np.mean(losses))
    assert saved["val/loss_std"] == pytest.approx(np.std(losses))


def test_finalize_picks_the_lowest_val_loss_fold_and_refits_for_its_epochs(cv_run):
    losses = [fm["val/loss"] for fm in cv_run.fold_metrics]
    best = cv_run.finalized["best_fold"]
    assert best["source_fold"] == int(np.argmin(losses))
    epochs = [fm["best_epoch"] for fm in cv_run.fold_metrics]
    assert cv_run.finalized["refit"]["refit_epochs"] == math.ceil(np.percentile(epochs, 75))
    assert cv_run.finalized["ensemble"]["n_checkpoints"] == 2
