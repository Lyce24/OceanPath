"""
Tests for the supervised-training callbacks (``oceanpath.modules.callbacks``).

tests/test_train_module.py runs every callback inside a real fit; here each one
is driven directly through its hooks with a stand-in trainer that carries only
the attributes the callback reads, so schedules and edge cases are pinned
exactly.

Covers:
  * BagCurriculumCallback: the training bag cap grows linearly over the warmup
    epochs, then holds
  * BatchIndexLogger: slide ids written in training order; no file without
    batches
  * FoldTimingCallback: fit wall time logged to the trainer's logger
  * WandbFoldSummary: best value and epoch in min and max mode, the sanity-check
    pass ignored, epochs without the metric skipped, loggers without a W&B run
  * MetadataWriter: config and parameter counts (frozen aggregator), output dir
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oceanpath.modules import callbacks as callbacks_module
from oceanpath.modules.callbacks import (
    BagCurriculumCallback,
    BatchIndexLogger,
    FoldTimingCallback,
    MetadataWriter,
    WandbFoldSummary,
)
from oceanpath.modules.train_module import MILTrainModule


def make_trainer(epoch=0, metrics=None, logger=None, sanity_checking=False, train_dataset=None):
    """The trainer attributes the callbacks read, and nothing else."""
    return SimpleNamespace(
        current_epoch=epoch,
        callback_metrics=metrics or {},
        logger=logger,
        sanity_checking=sanity_checking,
        datamodule=SimpleNamespace(train_dataset=train_dataset),
    )


# ═════════════════════════════════════════════════════════════════════════════
# BagCurriculumCallback
# ═════════════════════════════════════════════════════════════════════════════


class TestBagCurriculum:
    def _caps(self, callback, n_epochs):
        dataset = SimpleNamespace(max_instances=None)
        caps = []
        for epoch in range(n_epochs):
            callback.on_train_epoch_start(make_trainer(epoch, train_dataset=dataset), None)
            caps.append(dataset.max_instances)
        return caps

    def test_cap_grows_linearly_over_the_warmup_then_holds(self):
        callback = BagCurriculumCallback(start_instances=100, end_instances=500, warmup_epochs=4)
        assert self._caps(callback, 7) == [100, 200, 300, 400, 500, 500, 500]

    def test_default_config_schedule(self):
        """configs/training/default.yaml: 512 -> 8000 over 5 epochs (caps truncated to int)."""
        callback = BagCurriculumCallback(start_instances=512, end_instances=8000, warmup_epochs=5)
        assert self._caps(callback, 7) == [512, 2009, 3507, 5004, 6502, 8000, 8000]

    def test_without_a_datamodule_nothing_is_touched(self):
        """The refit fits from a bare DataLoader: trainer.datamodule is None."""
        trainer = SimpleNamespace(current_epoch=1, datamodule=None)
        BagCurriculumCallback(4, 12, 2).on_train_epoch_start(trainer, None)
        assert trainer.datamodule is None


# ═════════════════════════════════════════════════════════════════════════════
# BatchIndexLogger
# ═════════════════════════════════════════════════════════════════════════════


class TestBatchIndexLogger:
    def test_writes_the_slide_ids_in_training_order(self, tmp_path):
        out = tmp_path / "fold_0"
        callback = BatchIndexLogger(output_dir=str(out))
        for i, ids in enumerate([["s3", "s1"], ["s2"], ["s1", "s3"]]):
            callback.on_train_batch_start(None, None, {"slide_ids": ids}, i)
        callback.on_train_batch_start(None, None, {"features": torch.zeros(1)}, 3)  # no ids
        callback.on_train_end(None, None)
        saved = np.load(out / "batch_indices.npy", allow_pickle=True)
        assert saved.tolist() == ["s3", "s1", "s2", "s1", "s3"]

    def test_no_batches_no_file(self, tmp_path):
        BatchIndexLogger(output_dir=str(tmp_path / "fold_0")).on_train_end(None, None)
        assert not (tmp_path / "fold_0").exists()


# ═════════════════════════════════════════════════════════════════════════════
# FoldTimingCallback
# ═════════════════════════════════════════════════════════════════════════════


class TestFoldTiming:
    def test_logs_the_fit_wall_time(self, monkeypatch):
        readings = iter([100.0, 130.5])
        monkeypatch.setattr(callbacks_module.time, "monotonic", lambda: next(readings))
        logged = []
        trainer = make_trainer(logger=SimpleNamespace(log_metrics=logged.append))
        callback = FoldTimingCallback()
        callback.on_fit_start(trainer, None)
        callback.on_fit_end(trainer, None)
        assert logged == [{"fold/wall_time_s": 30.5}]


# ═════════════════════════════════════════════════════════════════════════════
# WandbFoldSummary
# ═════════════════════════════════════════════════════════════════════════════


class TestWandbFoldSummary:
    def _validate(self, callback, values, key="val/loss", run=None, start_epoch=0):
        logger = SimpleNamespace(experiment=run if run is not None else SimpleNamespace(summary={}))
        for epoch, value in enumerate(values, start=start_epoch):
            metrics = {key: torch.tensor(value)}
            callback.on_validation_epoch_end(make_trainer(epoch, metrics, logger), None)
        return logger.experiment

    def test_min_mode_keeps_the_lowest_value_and_its_epoch(self):
        callback = WandbFoldSummary(fold=2, monitor_metric="val/loss", monitor_mode="min")
        run = self._validate(callback, [0.7, 0.5, 0.6])
        assert run.summary["fold_2/best_val/loss"] == pytest.approx(0.5)
        assert run.summary["fold_2/best_epoch"] == 1

    def test_max_mode_keeps_the_highest_value_and_its_epoch(self):
        callback = WandbFoldSummary(fold=0, monitor_metric="val/auroc", monitor_mode="max")
        run = self._validate(callback, [0.6, 0.8, 0.7], key="val/auroc")
        assert run.summary["fold_0/best_val/auroc"] == pytest.approx(0.8)
        assert run.summary["fold_0/best_epoch"] == 1

    def test_the_sanity_check_pass_is_not_a_candidate(self):
        """Lightning validates an untrained model on two batches before epoch 0."""
        callback = WandbFoldSummary(fold=0, monitor_metric="val/loss", monitor_mode="min")
        run = SimpleNamespace(summary={})
        sanity = make_trainer(
            0, {"val/loss": torch.tensor(0.1)}, SimpleNamespace(experiment=run), True
        )
        callback.on_validation_epoch_end(sanity, None)
        self._validate(callback, [0.7, 0.5, 0.6], run=run)
        assert (callback.best_value, callback.best_epoch) == (pytest.approx(0.5), 1)
        assert run.summary["fold_0/best_val/loss"] == pytest.approx(0.5)

    def test_epochs_without_the_monitored_metric_are_skipped(self):
        callback = WandbFoldSummary(fold=0, monitor_metric="val/auroc", monitor_mode="max")
        run = self._validate(callback, [0.3], key="val/loss")
        assert callback.best_value == float("-inf")
        assert run.summary == {}

    def test_tracking_survives_a_logger_without_a_wandb_run(self):
        """A CSV logger's experiment has no summary; the best value is still tracked."""
        callback = WandbFoldSummary(fold=0)
        self._validate(callback, [0.7, 0.4, 0.6], run=object())
        assert (callback.best_value, callback.best_epoch) == (pytest.approx(0.4), 1)


# ═════════════════════════════════════════════════════════════════════════════
# MetadataWriter
# ═════════════════════════════════════════════════════════════════════════════


class TestMetadataWriter:
    def test_records_config_and_counts_frozen_parameters_as_untrainable(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        module = MILTrainModule(
            arch="abmil",
            in_dim=16,
            num_classes=2,
            model_cfg={"embed_dim": 8, "attn_dim": 8, "dropout": 0.0},
            freeze_aggregator=True,
        )
        out = tmp_path / "fold_3"  # created by the callback
        config = {"exp_name": "toy", "training": {"lr": 1e-4, "max_epochs": 2}}
        MetadataWriter(output_dir=str(out), config=config).on_fit_start(None, module)
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["config"] == config
        # ABMIL(16 -> 8, attn 8) = 289 parameters, frozen; the Linear(8, 2) head = 18
        assert (meta["num_parameters"], meta["num_trainable"]) == (307, 18)
        assert meta["cuda_available"] is False
        assert "gpu_name" not in meta
