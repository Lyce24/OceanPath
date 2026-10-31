"""
Tests for model finalization after cross-validation (``oceanpath.modules.finalize``).

``finalize`` runs once per module on two fold checkpoints saved where
scripts/train.py's ModelCheckpoint puts them (``best-epoch=E-val/loss=X.ckpt``,
which the "/" turns into a sub-directory), over a synthetic memmap store, with
the refit trained for real on CPU. The outputs are then read back the way
evaluation reads them: ``EnsembleWrapper`` for every strategy and
``run_inference`` for the ensemble.

Covers:
  * finalize_models: the exact file layout under final/, the best-fold copy,
    the ensemble directory and its info.json, the refit (epoch rule, training
    set = every non-test slide, plateau -> cosine), finalize_summary.json, and
    one failing strategy not stopping the others
  * best-fold selection in min / max mode, folds missing the metric or the
    checkpoint
  * the ensemble skipping a fold whose checkpoint is missing
  * refit epoch rules (p75 / median / max / mean), invalid entries, fallback
  * parse_best_epoch_from_checkpoint on Lightning checkpoint names
"""

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from oceanpath.eval.ensemble import EnsembleWrapper
from oceanpath.eval.inference import run_inference
from oceanpath.modules.finalize import (
    _compute_refit_epochs,
    _save_best_fold,
    _save_ensemble,
    finalize_models,
    parse_best_epoch_from_checkpoint,
)
from oceanpath.modules.train_module import MILTrainModule

IN_DIM = 16
MODEL_CFG = {"name": "abmil", "arch": "abmil", "embed_dim": 8, "attn_dim": 8, "dropout": 0.0}
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
# fold k's best checkpoint: (0-based epoch in its name, val/loss, val/auroc)
FOLD_RESULTS = [(0, 0.4, 0.9), (1, 0.3, 0.7)]
FINAL_FILES = {
    "best_fold/model.ckpt",
    "best_fold/info.json",
    "ensemble/fold_0.ckpt",
    "ensemble/fold_1.ckpt",
    "ensemble/info.json",
    "refit/model.ckpt",
    "refit/info.json",
    "finalize_summary.json",
}


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═════════════════════════════════════════════════════════════════════════════


def save_module_checkpoint(path: Path, seed: int) -> Path:
    """Save an untrained MILTrainModule the way training does (Trainer.save_checkpoint)."""
    torch.manual_seed(seed)
    module = MILTrainModule(arch="abmil", in_dim=IN_DIM, num_classes=2, model_cfg=dict(MODEL_CFG))
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return path


def make_cfg(root: Path, store, **training) -> DictConfig:
    t = {
        "monitor_metric": "val/loss",
        "monitor_mode": "min",
        "refit_epoch_rule": "p75",
        "max_epochs": 30,
        "batch_size": 2,
        "max_instances": 64,
        "class_weighted_sampling": True,
        "instance_dropout": 0.0,
        "feature_noise_std": 0.0,
        "return_coords": False,
        "use_preallocated_collator": True,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "lr_scheduler": "plateau",
        "warmup_epochs": 0,
        "loss_type": "ce",
        "focal_gamma": 2.0,
        "canary_interval": 0,
        "compile_model": False,
        "freeze_aggregator": False,
        "seed": 0,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "deterministic": False,
    }
    t.update(training)
    return OmegaConf.create(
        {
            "platform": {
                "splits_root": str(root / "splits"),
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
            "splits": {"name": "kfold2", "scheme": "kfold"},
            "encoder": {"feature_dim": IN_DIM},
            "model": dict(MODEL_CFG),
            "training": t,
        }
    )


@pytest.fixture(scope="module")
def env(build_feature_store, tmp_path_factory):
    """Store, manifest, splits and two fold checkpoints laid out as scripts/train.py leaves them."""
    store = build_feature_store([n for n, _, _ in COHORT], feat_dim=IN_DIM, seed=5)
    root = tmp_path_factory.mktemp("finalize")
    labels = [label for _, label, _ in COHORT]
    pd.DataFrame({"filename": [f"{s}.h5" for s in store.slide_ids], "label": labels}).to_csv(
        root / "manifest.csv", index=False
    )
    splits_dir = root / "splits" / "toy" / "kfold2"
    splits_dir.mkdir(parents=True)
    pd.DataFrame({"slide_id": store.slide_ids, "fold": [f for *_, f in COHORT]}).to_parquet(
        splits_dir / "splits.parquet", index=False
    )
    fold_metrics = []
    for fold, (epoch, loss, auroc) in enumerate(FOLD_RESULTS):
        ckpt = (
            root
            / "train"
            / f"fold_{fold}"
            / "checkpoints"
            / f"best-epoch={epoch}-val/loss={loss:.4f}.ckpt"
        )
        save_module_checkpoint(ckpt, seed=fold)
        fold_metrics.append(
            {
                "val/loss": loss,
                "val/auroc": auroc,
                "fold": fold,
                "best_checkpoint": str(ckpt),
                "best_epoch": parse_best_epoch_from_checkpoint(str(ckpt)),
            }
        )
    return SimpleNamespace(
        store=store,
        root=root,
        labels=dict(zip(store.slide_ids, labels, strict=True)),
        test_ids=[s for s, (*_, f) in zip(store.slide_ids, COHORT, strict=True) if f == -1],
        fold_metrics=fold_metrics,
        cfg=make_cfg(root, store),
    )


@pytest.fixture(scope="module")
def finalized(env):
    output_dir = env.root / "train"
    results = finalize_models(env.cfg, output_dir, n_folds=2, all_fold_metrics=env.fold_metrics)
    return SimpleNamespace(results=results, final=output_dir / "final")


def read_info(path: Path) -> dict:
    return json.loads((path / "info.json").read_text())


def model_state(ckpt_path) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    return {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}


# ═════════════════════════════════════════════════════════════════════════════
# finalize_models
# ═════════════════════════════════════════════════════════════════════════════


class TestFinalizeModels:
    def test_writes_exactly_the_files_downstream_reads(self, finalized):
        files = {
            p.relative_to(finalized.final).as_posix()
            for p in finalized.final.rglob("*")
            if p.is_file()
        }
        assert files == FINAL_FILES

    def test_fold_best_epochs_come_from_the_checkpoint_paths(self, env):
        assert [fm["best_epoch"] for fm in env.fold_metrics] == [1, 2]

    def test_best_fold_is_a_copy_of_the_lowest_val_loss_checkpoint(self, env, finalized):
        dest = finalized.final / "best_fold"
        source = env.fold_metrics[1]["best_checkpoint"]
        assert (dest / "model.ckpt").read_bytes() == Path(source).read_bytes()
        info = read_info(dest)
        assert info == finalized.results["best_fold"]
        assert info["strategy"] == "best_fold"
        assert (info["source_fold"], info["best_score"], info["best_epoch"]) == (1, 0.3, 2)
        assert info["model_path"] == str(dest / "model.ckpt")
        assert info["source_checkpoint"] == source

    def test_ensemble_holds_every_fold_checkpoint_and_the_architecture(self, env, finalized):
        dest = finalized.final / "ensemble"
        for fold, fm in enumerate(env.fold_metrics):
            assert (dest / f"fold_{fold}.ckpt").read_bytes() == Path(
                fm["best_checkpoint"]
            ).read_bytes()
        info = read_info(dest)
        assert info["strategy"] == "ensemble"
        assert (info["n_folds"], info["n_checkpoints"], info["ensemble_method"]) == (
            2,
            2,
            "mean_prob",
        )
        assert (info["model_arch"], info["in_dim"], info["model_cfg"]) == (
            "abmil",
            IN_DIM,
            MODEL_CFG,
        )
        assert [f["fold"] for f in info["folds"]] == [0, 1]
        assert [f["monitor_score"] for f in info["folds"]] == [0.4, 0.3]
        assert [f["model_path"] for f in info["folds"]] == [
            str(dest / f"fold_{i}.ckpt") for i in range(2)
        ]

    def test_refit_trains_on_every_non_test_slide_for_the_p75_epochs(self, env, finalized):
        dest = finalized.final / "refit"
        info = read_info(dest)
        n_train = len(env.labels) - len(env.test_ids)
        assert info["refit_epochs"] == math.ceil(np.percentile([1, 2], 75)) == 2
        assert (info["refit_epoch_rule"], info["fold_best_epochs"]) == ("p75", [1, 2])
        assert info["n_train_slides"] == n_train
        assert info["label_counts"] == {"0": 4, "1": 4}
        assert info["lr_scheduler"] == "cosine"  # plateau has no val loss to watch in a refit
        ckpt = torch.load(dest / "model.ckpt", map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 2
        assert ckpt["global_step"] == 2 * math.ceil(n_train / 2)
        hp = ckpt["hyper_parameters"]
        assert (hp["max_epochs"], hp["lr_scheduler"], hp["collect_embeddings"]) == (
            2,
            "cosine",
            False,
        )

    def test_refit_is_a_freshly_trained_model(self, env, finalized):
        refit = model_state(finalized.final / "refit" / "model.ckpt")
        for fm in env.fold_metrics:
            fold = model_state(fm["best_checkpoint"])
            assert any(not torch.equal(refit[k], fold[k]) for k in fold)

    def test_summary_records_every_strategy(self, finalized):
        summary = json.loads((finalized.final / "finalize_summary.json").read_text())
        assert summary == json.loads(json.dumps(finalized.results))
        assert {k: v["strategy"] for k, v in summary.items()} == {
            "best_fold": "best_fold",
            "ensemble": "ensemble",
            "refit": "refit",
        }

    def test_evaluation_loads_every_final_model(self, env, finalized):
        counts = {
            s: EnsembleWrapper(finalized.final / s, device="cpu").n_models
            for s in ("best_fold", "ensemble", "refit")
        }
        assert counts == {"best_fold": 1, "ensemble": 2, "refit": 1}

    def test_ensemble_inference_averages_the_fold_probabilities(self, env, finalized):
        """run_inference reads ensemble/info.json and fold_<k>.ckpt; check it by hand."""
        preds = run_inference(
            model_dir=finalized.final / "ensemble",
            mmap_dir=str(env.store.mmap_dir),
            splits_dir=str(env.root / "splits" / "toy" / "kfold2"),
            csv_path=str(env.root / "manifest.csv"),
            label_column="label",
            filename_column="filename",
            scheme="kfold",
            accelerator="cpu",
            precision="32-true",
        ).set_index("slide_id")
        assert sorted(preds.index) == env.test_ids
        folds = [
            MILTrainModule.load_from_checkpoint(fm["best_checkpoint"], weights_only=False).eval()
            for fm in env.fold_metrics
        ]
        with torch.no_grad():
            for sid in env.test_ids:
                bag = torch.from_numpy(env.store.features[sid])
                expected = np.mean([torch.softmax(m(bag).logits, -1)[0, 1].item() for m in folds])
                assert preds.loc[sid, "prob_1"] == pytest.approx(expected, rel=1e-5)
                assert preds.loc[sid, "label"] == env.labels[sid]

    def test_a_failing_strategy_does_not_stop_the_others(self, env, tmp_path):
        """No fold carries the monitored metric and the refit's store is missing."""
        cfg = make_cfg(env.root, env.store, monitor_metric="val/auprc", monitor_mode="max")
        cfg.data.mmap_dir = str(tmp_path / "no_store")
        results = finalize_models(cfg, tmp_path, n_folds=2, all_fold_metrics=env.fold_metrics)
        assert "error" in results["best_fold"] and "error" in results["refit"]
        assert results["ensemble"]["n_checkpoints"] == 2
        final = tmp_path / "final"
        assert (final / "ensemble" / "info.json").is_file()
        # evaluation skips a strategy without info.json
        assert not (final / "best_fold" / "info.json").exists()
        assert not (final / "refit" / "info.json").exists()
        summary = json.loads((final / "finalize_summary.json").read_text())
        assert summary["best_fold"]["error"] == results["best_fold"]["error"]


# ═════════════════════════════════════════════════════════════════════════════
# Best fold and ensemble selection
# ═════════════════════════════════════════════════════════════════════════════


class TestBestFoldSelection:
    def test_max_mode_picks_the_highest_score(self, env, tmp_path):
        cfg = make_cfg(env.root, env.store, monitor_metric="val/auroc", monitor_mode="max")
        info = _save_best_fold(cfg, tmp_path, env.fold_metrics)
        assert (info["source_fold"], info["best_score"]) == (0, 0.9)

    def test_folds_without_the_metric_are_passed_over(self, env, tmp_path):
        metrics = [
            dict(env.fold_metrics[0]),
            {k: v for k, v in env.fold_metrics[1].items() if k != "val/loss"},
        ]
        assert _save_best_fold(env.cfg, tmp_path, metrics)["source_fold"] == 0

    def test_no_fold_with_the_metric_is_an_error(self, env, tmp_path):
        cfg = make_cfg(env.root, env.store, monitor_metric="val/auprc")
        with pytest.raises(ValueError, match="val/auprc"):
            _save_best_fold(cfg, tmp_path, env.fold_metrics)

    def test_missing_best_checkpoint_is_an_error(self, env, tmp_path):
        metrics = [dict(fm, best_checkpoint=str(tmp_path / "gone.ckpt")) for fm in env.fold_metrics]
        with pytest.raises(FileNotFoundError, match="gone.ckpt"):
            _save_best_fold(env.cfg, tmp_path, metrics)

    def test_ensemble_skips_a_fold_whose_checkpoint_is_missing(self, env, tmp_path):
        moved = tmp_path / "elsewhere.ckpt"
        shutil.copy2(env.fold_metrics[1]["best_checkpoint"], moved)
        metrics = [
            dict(env.fold_metrics[0], best_checkpoint=""),
            dict(env.fold_metrics[1], best_checkpoint=str(moved)),
        ]
        info = _save_ensemble(env.cfg, tmp_path, 2, metrics)
        assert info["n_checkpoints"] == 1
        assert info["folds"][0] == {"fold": 0, "error": "checkpoint not found"}
        assert sorted(p.name for p in (tmp_path / "ensemble").glob("fold_*.ckpt")) == [
            "fold_1.ckpt"
        ]

    def test_ensemble_without_any_checkpoint_is_an_error(self, env, tmp_path):
        metrics = [dict(fm, best_checkpoint="") for fm in env.fold_metrics]
        with pytest.raises(FileNotFoundError):
            _save_ensemble(env.cfg, tmp_path, 2, metrics)


# ═════════════════════════════════════════════════════════════════════════════
# Refit epochs and checkpoint epochs
# ═════════════════════════════════════════════════════════════════════════════


class TestRefitEpochs:
    EPOCHS = [{"best_epoch": e} for e in (3, 5, 8, 10)]

    @pytest.mark.parametrize(
        "rule, expected",
        [
            ("p75", 9),  # 75th percentile of 3, 5, 8, 10 = 8.5 -> 9
            ("median", 7),  # 6.5 -> 7
            ("max", 10),
            ("mean", 7),  # 6.5 -> 7
        ],
    )
    def test_rules(self, rule, expected):
        assert _compute_refit_epochs(self.EPOCHS, rule=rule) == expected

    def test_entries_without_a_positive_epoch_are_ignored(self):
        metrics = [
            {"best_epoch": None},
            {"best_epoch": 0},
            {"best_epoch": -2},
            {"best_epoch": "7"},
            {},
            {"best_epoch": 4},
        ]
        assert _compute_refit_epochs(metrics, rule="max", fallback_epochs=30) == 4

    def test_no_usable_epoch_falls_back(self):
        assert _compute_refit_epochs([{"best_epoch": None}, {}], fallback_epochs=17) == 17

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="refit_epoch_rule"):
            _compute_refit_epochs(self.EPOCHS, rule="p90")


class TestParseBestEpoch:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("fold_0/checkpoints/best-epoch=5-val/loss=0.1234.ckpt", 6),
            ("fold_0/checkpoints/best-epoch=0-val/auroc=0.9100.ckpt", 1),
            ("epoch=12-step=500.ckpt", 13),
            ("runs/epoch=99_sweep/fold_1/checkpoints/best-epoch=3-val/loss=0.5.ckpt", 4),
            ("fold_0/checkpoints/last.ckpt", None),
            ("", None),
            (None, None),
        ],
    )
    def test_epoch_count_from_checkpoint_path(self, path, expected):
        assert parse_best_epoch_from_checkpoint(path) == expected
