"""
Custom Lightning callbacks for MIL training.

BagCurriculumCallback:  Anneal max_instances from small → large over warmup epochs.
BatchIndexLogger:       Record exact batch indices for deterministic replay.
FoldTimingCallback:     Log per-fold wall time.
WandbFoldSummary:       Log fold-level metrics to W&B summary at fold end.
"""

import json
import logging
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch

logger = logging.getLogger(__name__)


class BagCurriculumCallback(L.Callback):
    """
    Dynamic bag curriculum: start with smaller bags, anneal to full.

    Gradually increases dataset.max_instances from start_instances
    to end_instances over warmup_epochs. After warmup, keeps end_instances.

    Parameters
    ----------
    start_instances : int
        Bag size at epoch 0.
    end_instances : int
        Bag size after warmup.
    warmup_epochs : int
        Number of epochs to anneal over.
    """

    def __init__(
        self,
        start_instances: int = 512,
        end_instances: int = 8000,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.start = start_instances
        self.end = end_instances
        self.warmup = warmup_epochs

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.datamodule is None:
            return

        progress = min(trainer.current_epoch / max(1, self.warmup), 1.0)
        current = int(self.start + (self.end - self.start) * progress)

        # Update dataset max_instances
        if (
            hasattr(trainer.datamodule, "train_dataset")
            and trainer.datamodule.train_dataset is not None
        ):
            trainer.datamodule.train_dataset.max_instances = current

        if trainer.current_epoch <= self.warmup:
            logger.info(f"BagCurriculum: epoch {trainer.current_epoch}, max_instances={current}")


class BatchIndexLogger(L.Callback):
    """
    Record the exact sequence of batch indices for deterministic replay.

    At training end, writes batch_indices.npy — an array of all
    sample indices seen during training, in order. Combined with
    the dataset, this enables replaying the exact same training run
    for debugging.
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self._indices: list[list[str]] = []

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: dict,
        batch_idx: int,
    ) -> None:
        if "slide_ids" in batch:
            self._indices.append(batch["slide_ids"])

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._indices:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Flatten to a single list of slide_ids
        flat = [sid for batch_ids in self._indices for sid in batch_ids]
        path = self.output_dir / "batch_indices.npy"
        np.save(str(path), np.array(flat, dtype=object))
        logger.info(f"Saved {len(flat)} batch indices → {path}")


class FoldTimingCallback(L.Callback):
    """Log wall-clock time for the current fold."""

    def __init__(self):
        super().__init__()
        self._start_time: float | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start_time = time.monotonic()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            logger.info(f"Fold training completed in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
            if trainer.logger:
                trainer.logger.log_metrics({"fold/wall_time_s": elapsed})


class WandbFoldSummary(L.Callback):
    """
    Log fold-level summary metrics to W&B at the end of validation.

    Tracks the configured monitor metric and logs best values as W&B summary
    (visible in the runs table). Also logs all val/* metrics for completeness.
    """

    def __init__(self, fold: int, monitor_metric: str = "val/loss", monitor_mode: str = "min"):
        super().__init__()
        self.fold = fold
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.best_value = float("inf") if monitor_mode == "min" else float("-inf")
        self.best_epoch = 0

    def _is_improvement(self, value: float) -> bool:
        if self.monitor_mode == "min":
            return value < self.best_value
        return value > self.best_value

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics

        # Extract the monitored metric
        current = metrics.get(self.monitor_metric)
        if current is None:
            return
        if isinstance(current, torch.Tensor):
            current = current.item()

        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = trainer.current_epoch

        # Log to W&B summary
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            try:
                wandb_run = trainer.logger.experiment
                wandb_run.summary[f"fold_{self.fold}/best_{self.monitor_metric}"] = self.best_value
                wandb_run.summary[f"fold_{self.fold}/best_epoch"] = self.best_epoch

                # Also log other key metrics at best epoch
                for key in ["val/loss", "val/auroc", "val/acc", "val/f1", "val/balanced_acc"]:
                    val = metrics.get(key)
                    if val is not None:
                        v = val.item() if isinstance(val, torch.Tensor) else val
                        wandb_run.summary[f"fold_{self.fold}/{key}"] = v
            except Exception:
                pass


class MetadataWriter(L.Callback):
    """
    Write metadata.json at training start with config, system info, git hash.
    """

    def __init__(self, output_dir: str, config: dict | None = None):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.config = config or {}

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "config": self.config,
            "num_parameters": sum(p.numel() for p in pl_module.parameters()),
            "num_trainable": sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            metadata["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )

        # Git hash
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                metadata["git_hash"] = result.stdout.strip()
        except Exception:
            pass

        path = self.output_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info(f"Metadata written → {path}")
