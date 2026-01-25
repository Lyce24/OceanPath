
from typing import Any, Dict, Literal, Optional
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

# =============================================================================
# Callbacks
# =============================================================================
class BestStateTracker(Callback):
    """Track best model state in RAM based on monitored metric (with min_delta support)."""

    def __init__(
        self,
        monitor: str,
        mode: Literal["max", "min"] = "max",
        min_delta: float = 0.0,
    ):
        super().__init__()
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)

        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def reset(self) -> None:
        self.best_score = None
        self.best_epoch = None
        self.best_state = None

    def _to_float(self, x: Any) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                return None
            return float(x.detach().cpu().item())
        try:
            return float(x)
        except Exception:
            return None

    def _is_better(self, new: float, best: float) -> bool:
        if self.mode == "max":
            return new > (best + self.min_delta)
        return new < (best - self.min_delta)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        score_raw = trainer.callback_metrics.get(self.monitor)
        score_val = self._to_float(score_raw)
        if score_val is None:
            return

        if self.best_score is None or self._is_better(score_val, self.best_score):
            self.best_score = score_val
            self.best_epoch = trainer.current_epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}


class FoldMetricLogger(Callback):
    """Log fold metrics to wandb with fold prefix; avoids noisy duplicates."""

    def __init__(self, fold: int, wandb_run: Optional[Any] = None, log_lr: bool = True):
        super().__init__()
        self.fold = int(fold)
        self.wandb_run = wandb_run
        self.log_lr = bool(log_lr)

    def _scalarize(self, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                return None
            return float(v.detach().cpu().item())
        try:
            return float(v)
        except Exception:
            return None

    def _add_lrs(self, trainer: pl.Trainer, payload: Dict[str, Any]) -> None:
        if not self.log_lr or not trainer.optimizers:
            return
        opt = trainer.optimizers[0]
        for i, pg in enumerate(opt.param_groups):
            if "lr" in pg:
                payload[f"fold_{self.fold}/lr_group{i}"] = float(pg["lr"])

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.wandb_run is None:
            return
        payload: Dict[str, Any] = {}
        for k, v in trainer.callback_metrics.items():
            if "train/" not in k:
                continue
            sv = self._scalarize(v)
            if sv is None:
                continue
            payload[f"fold_{self.fold}/{k}"] = sv
        self._add_lrs(trainer, payload)
        payload["epoch"] = int(trainer.current_epoch)
        payload["global_step"] = int(trainer.global_step)
        if payload:
            self.wandb_run.log(payload)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.wandb_run is None:
            return
        payload: Dict[str, Any] = {}
        for k, v in trainer.callback_metrics.items():
            if "val/" not in k:
                continue
            sv = self._scalarize(v)
            if sv is None:
                continue
            payload[f"fold_{self.fold}/{k}"] = sv
        self._add_lrs(trainer, payload)
        payload["epoch"] = int(trainer.current_epoch)
        payload["global_step"] = int(trainer.global_step)
        if payload:
            self.wandb_run.log(payload)
