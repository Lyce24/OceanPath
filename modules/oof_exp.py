"""
Self-contained WSI MIL Classification with K-Fold Cross-Validation.

Final version with:
- Fine-tune (ft) and Linear Probe (lp) modes
- Separate LR/WD for encoder vs head
- Robust error handling
- Consistent typing
- Stratified bootstrap
- Proper device handling
- Memory management
- Clean logging
"""

from __future__ import annotations

import copy
import gc
import math
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassAUROC,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from data.data_module import WSIMILDataModule
from models.MIL.wsi_model import WSIModel

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ExperimentConfig:
    """Experiment configuration with validation."""
    
    # Data paths
    csv_path: str
    feature_dir: str
    id_col: str
    target_col: str
    
    # Model architecture
    mil: str
    mil_attrs: Dict[str, Any] = field(default_factory=dict)
    head_dim: int = 512
    head_dropout: float = 0.25
    num_fc_layers: int = 1
    hidden_dim: int = 128
    ds_dropout: float = 0.3
    simple_mlp: bool = False
    encoder_weights_path: Optional[str] = None
    
    # Training mode: "ft" (fine-tune) or "lp" (linear probe)
    mode: Literal["ft", "lp"] = "ft"
    
    # Learning rates
    lr: float = 3e-4              # Base LR (used for head in ft mode, all params if not specified)
    encoder_lr: Optional[float] = None  # Encoder LR for ft mode (default: lr * 0.1)
    head_lr: Optional[float] = None     # Head LR (default: lr)
    
    # Weight decay
    l2_reg: float = 1e-3          # Base weight decay
    encoder_wd: Optional[float] = None  # Encoder WD for ft mode (default: l2_reg)
    head_wd: Optional[float] = None     # Head WD (default: l2_reg)
    
    # Training
    max_epochs: int = 60
    warmup_epochs: int = 0
    scheduler: Literal["cosine", "none"] = "cosine"
    precision: Literal["16-mixed", "bf16-mixed", "32"] = "16-mixed"
    early_stopping_patience: int = 10
    
    # Gradient settings
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = 1.0
    
    # Data loading
    batch_size: int = 1
    bag_size: Optional[int] = None
    replacement: bool = False
    num_workers: Optional[int] = None
    
    # Experiment settings
    final_strategy: Literal["refit_full", "best_fold"] = "refit_full"
    use_optimal_threshold: bool = False
    use_class_weights: bool = True
    n_bootstraps: int = 1000
    seed: int = 42
    
    def __post_init__(self):
        # Num workers
        if self.num_workers is None:
            try:
                self.num_workers = min(len(os.sched_getaffinity(0)), 8)
            except AttributeError:
                self.num_workers = 4
        
        # Path validation
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not os.path.isdir(self.feature_dir):
            raise NotADirectoryError(f"Feature dir not found: {self.feature_dir}")
        
        # Batch size validation
        if self.batch_size > 1:
            if not (self.bag_size and self.bag_size > 0):
                raise ValueError("bag_size must be positive when batch_size > 1")
            if not self.replacement:
                raise ValueError("replacement must be True when batch_size > 1")
        
        # Basic validation
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")
        
        # Mode-specific validation and defaults
        if self.mode not in ("ft", "lp"):
            raise ValueError(f"mode must be 'ft' or 'lp', got '{self.mode}'")
        
        if self.mode == "lp":
            # Linear probe requires pretrained encoder
            if self.encoder_weights_path is None:
                raise ValueError(
                    "Linear probe mode (mode='lp') requires encoder_weights_path. "
                    "Please provide path to pretrained encoder weights."
                )
            if not os.path.exists(self.encoder_weights_path):
                raise FileNotFoundError(
                    f"Encoder weights not found: {self.encoder_weights_path}"
                )
        
        # Set default LR values
        if self.encoder_lr is None:
            # For fine-tune: use lower LR for encoder (10x smaller)
            # For linear probe: doesn't matter (encoder is frozen)
            self.encoder_lr = self.lr * 0.1 if self.mode == "ft" else 0.0
        
        if self.head_lr is None:
            self.head_lr = self.lr
        
        # Set default WD values
        if self.encoder_wd is None:
            self.encoder_wd = self.l2_reg
        
        if self.head_wd is None:
            self.head_wd = self.l2_reg
        
        # Validate LR values
        if self.mode == "ft" and self.encoder_lr <= 0:
            raise ValueError("encoder_lr must be positive for fine-tune mode")
        if self.head_lr <= 0:
            raise ValueError("head_lr must be positive")

    @property
    def freeze_encoder(self) -> bool:
        """Encoder is frozen in linear probe mode."""
        return self.mode == "lp"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulate_grad_batches
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for logging."""
        return {
            # Mode
            "mode": self.mode,
            "freeze_encoder": self.freeze_encoder,
            # Architecture
            "mil": self.mil,
            "mil_attrs": self.mil_attrs,
            "head_dim": self.head_dim,
            "head_dropout": self.head_dropout,
            "encoder_weights_path": self.encoder_weights_path,
            # Learning rates
            "lr": self.lr,
            "encoder_lr": self.encoder_lr,
            "head_lr": self.head_lr,
            # Weight decay
            "l2_reg": self.l2_reg,
            "encoder_wd": self.encoder_wd,
            "head_wd": self.head_wd,
            # Training
            "max_epochs": self.max_epochs,
            "scheduler": self.scheduler,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "effective_batch_size": self.effective_batch_size,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "gradient_clip_val": self.gradient_clip_val,
            "early_stopping_patience": self.early_stopping_patience,
            "use_class_weights": self.use_class_weights,
            "final_strategy": self.final_strategy,
            "seed": self.seed,
        }


# =============================================================================
# Evaluation Functions
# =============================================================================
def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    """
    Convert logits to probability matrix.
    
    Args:
        logits: Shape [N], [N, 1], or [N, C]
        
    Returns:
        Probabilities [N, C] where C >= 2
    """
    logits = logits.detach().cpu().float()
    
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    
    if logits.size(-1) == 1:
        p = torch.sigmoid(logits.squeeze(-1))
        probs = torch.stack([1 - p, p], dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
    
    return probs.numpy()


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute classification metrics with proper error handling."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)
    
    if y_true.size == 0 or y_prob.size == 0:
        return {}
    
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2D, got shape {y_prob.shape}")
    
    N, C = y_prob.shape
    if N != len(y_true):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_prob={N}")
    
    # Numerical stability
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    y_prob = y_prob / np.where(row_sums > 0, row_sums, 1.0)
    
    # Predictions
    if threshold is not None and C == 2:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)
    
    avg = "binary" if C == 2 else "macro"
    
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    
    # ROC-AUC with proper exception handling
    try:
        if len(np.unique(y_true)) < 2:
            metrics["roc_auc"] = float("nan")
        elif C == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            ))
    except ValueError as e:
        warnings.warn(f"ROC-AUC computation failed: {e}")
        metrics["roc_auc"] = float("nan")
    
    return metrics


def tune_threshold_youden(y_true: np.ndarray, p_pos: np.ndarray) -> Tuple[float, float]:
    """Find optimal threshold using Youden's J statistic."""
    y_true = np.asarray(y_true).ravel()
    p_pos = np.asarray(p_pos).ravel()
    
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    
    fpr, tpr, thresholds = roc_curve(y_true, p_pos)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    
    # Handle edge cases where threshold is 0 or > 1
    threshold = float(np.clip(thresholds[best_idx], 0.01, 0.99))
    return threshold, float(j_scores[best_idx])


def operating_point_metrics(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary operating point metrics."""
    y_true = np.asarray(y_true).ravel()
    p_pos = np.asarray(p_pos).ravel()
    y_pred = (p_pos >= threshold).astype(int)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(classes) > 2:
        warnings.warn("operating_point_metrics is for binary classification only")
        return {"threshold": float(threshold)}
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return {
            "threshold": float(threshold),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
        }
    
    tn, fp, fn, tp = cm.ravel()
    eps = 1e-10
    
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn + eps)),
        "specificity": float(tn / (tn + fp + eps)),
        "ppv": float(tp / (tp + fp + eps)),
        "npv": float(tn / (tn + fn + eps)),
    }


def _stratified_bootstrap_indices(
    y_true: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stratified bootstrap indices to maintain class balance."""
    classes, counts = np.unique(y_true, return_counts=True)
    
    if len(classes) <= 1:
        return rng.choice(len(y_true), size=n_samples, replace=True)
    
    # Target samples per class
    props = counts / len(y_true)
    target_sizes = np.round(props * n_samples).astype(int)
    
    # Adjust to ensure exact n_samples
    diff = n_samples - target_sizes.sum()
    if diff != 0:
        idx = rng.choice(len(classes))
        target_sizes[idx] += diff
    
    # Sample from each class
    indices = []
    for cls, size in zip(classes, target_sizes):
        cls_indices = np.where(y_true == cls)[0]
        if size > 0 and len(cls_indices) > 0:
            sampled = rng.choice(cls_indices, size=size, replace=True)
            indices.append(sampled)
    
    return np.concatenate(indices) if indices else np.array([], dtype=int)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    threshold: Optional[float] = None,
    seed: int = 42,
    stratified: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals with stratified sampling."""
    y_true = np.asarray(y_true).ravel()
    rng = np.random.default_rng(seed)
    N = len(y_true)
    
    metric_samples: Dict[str, List[float]] = defaultdict(list)
    
    for _ in range(n_bootstraps):
        if stratified:
            idx = _stratified_bootstrap_indices(y_true, N, rng)
        else:
            idx = rng.choice(N, size=N, replace=True)
        
        if len(idx) == 0:
            continue
            
        m = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for k, v in m.items():
            if np.isfinite(v):
                metric_samples[k].append(v)
    
    alpha = (1 - ci) / 2
    results: Dict[str, Dict[str, float]] = {}
    
    for name, vals in metric_samples.items():
        if vals:
            arr = np.array(vals)
            results[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "lower": float(np.percentile(arr, 100 * alpha)),
                "upper": float(np.percentile(arr, 100 * (1 - alpha))),
            }
        else:
            results[name] = {"mean": np.nan, "std": np.nan, "lower": np.nan, "upper": np.nan}
    
    return results


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def cleanup_memory() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV, Excel, or Parquet."""
    path_lower = path.lower()
    if path_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    elif path_lower.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_y(y: torch.Tensor) -> torch.Tensor:
    """Normalize labels to 1D long tensor."""
    y = torch.as_tensor(y)
    if y.ndim == 2:
        if y.size(-1) > 1:      # one-hot / probs
            y = y.argmax(dim=-1)
        else:                   # [B,1]
            y = y.squeeze(-1)
    return y.long().view(-1)


@torch.inference_mode()
def collect_predictions(model, loader, device, use_amp=True):
    """Collect predictions from a model on a dataloader."""
    model.eval()
    all_logits, all_labels = [], []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out

        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        all_logits.append(logits.detach().float().cpu())
        all_labels.append(_normalize_y(y).cpu())

    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def _is_head_param(name: str) -> bool:
    """
    Check if a parameter belongs to the classification head.
    
    Head parameters typically contain these keywords in their names.
    """
    head_keywords = [
        'head', 'heads', 'classifier', 'fc', 'final', 
        'linear_out', 'output_layer', 'mlp_head'
    ]
    name_lower = name.lower()
    return any(kw in name_lower for kw in head_keywords)


def _is_encoder_param(name: str) -> bool:
    """
    Check if a parameter belongs to the encoder.
    
    Encoder parameters typically contain these keywords.
    """
    encoder_keywords = [
        'encoder', 'feature_encoder', 'aggregator', 'attention',
        'transformer', 'embed', 'projection', 'norm'
    ]
    name_lower = name.lower()
    # It's encoder if it matches encoder keywords OR doesn't match head keywords
    return any(kw in name_lower for kw in encoder_keywords) or not _is_head_param(name)


# =============================================================================
# Callbacks
# =============================================================================
class BestStateTracker(Callback):
    """Track best model state in RAM based on monitored metric."""
    
    def __init__(self, monitor: str, mode: Literal["max", "min"] = "max"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        
        score_val = float(score)
        is_better = (
            self.best_score is None
            or (self.mode == "max" and score_val > self.best_score)
            or (self.mode == "min" and score_val < self.best_score)
        )
        
        if is_better:
            self.best_score = score_val
            self.best_epoch = trainer.current_epoch
            # Clone to CPU to save GPU memory
            self.best_state = {
                k: v.detach().cpu().clone() 
                for k, v in pl_module.state_dict().items()
            }

    def reset(self) -> None:
        """Reset state for reuse."""
        self.best_score = None
        self.best_epoch = None
        self.best_state = None


class FoldMetricLogger(Callback):
    """Log metrics with fold prefix to wandb."""
    
    def __init__(self, fold: int, wandb_run: Optional[Any] = None):
        super().__init__()
        self.fold = fold
        self.wandb_run = wandb_run
        self.epoch_metrics: List[Dict[str, float]] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.wandb_run is None:
            return
        
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics[f"fold_{self.fold}/{key}"] = value
        
        # Add global step for x-axis alignment
        metrics["epoch"] = trainer.current_epoch
        metrics["global_step"] = trainer.global_step
        
        self.wandb_run.log(metrics)
        self.epoch_metrics.append(metrics)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.wandb_run is None:
            return
        
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            # Only log validation metrics here (avoid duplicates)
            if "val" in key:
                metrics[f"fold_{self.fold}/{key}"] = value
        
        if metrics:
            metrics["epoch"] = trainer.current_epoch
            self.wandb_run.log(metrics)


# =============================================================================
# Lightning Module
# =============================================================================
class WSIClassificationModule(pl.LightningModule):
    """
    WSI MIL classification with fine-tune/linear-probe modes.
    
    Supports:
    - Fine-tune (ft): Train encoder + head with different LR/WD
    - Linear probe (lp): Freeze encoder, train only head
    """

    def __init__(
        self,
        mil: str,
        n_classes: int,
        feature_dim: int,
        mil_attrs: Optional[Dict[str, Any]] = None,
        head_dim: int = 512,
        head_dropout: float = 0.25,
        num_fc_layers: int = 1,
        hidden_dim: int = 128,
        ds_dropout: float = 0.3,
        simple_mlp: bool = False,
        # Mode settings
        mode: Literal["ft", "lp"] = "ft",
        encoder_weights_path: Optional[str] = None,
        # Learning rates
        encoder_lr: float = 3e-5,
        head_lr: float = 3e-4,
        # Weight decay
        encoder_wd: float = 1e-3,
        head_wd: float = 1e-3,
        # Scheduler
        scheduler: str = "cosine",
        warmup_epochs: int = 0,
        max_epochs: int = 60,
        # Class weights
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        # Validate mode
        if mode not in ("ft", "lp"):
            raise ValueError(f"mode must be 'ft' or 'lp', got '{mode}'")
        
        if mode == "lp" and encoder_weights_path is None:
            raise ValueError("Linear probe mode requires encoder_weights_path")
        
        # Determine freeze_encoder from mode
        freeze_encoder = (mode == "lp")
        
        # Save all hyperparameters (except class_weights which is a tensor)
        self.save_hyperparameters(ignore=["class_weights"])
        
        self.n_classes = n_classes
        self.is_binary = n_classes <= 2

        # Build model
        self.model = WSIModel(
            input_feature_dim=feature_dim,
            n_classes=n_classes,
            encoder_type=mil,
            head_dropout=head_dropout,
            head_dim=head_dim,
            num_fc_layers=num_fc_layers,
            hidden_dim=hidden_dim,
            ds_dropout=ds_dropout,
            simple_mlp=simple_mlp,
            freeze_encoder=freeze_encoder,
            encoder_attrs=mil_attrs or {},
            encoder_weights_path=encoder_weights_path,
        )
        
        if hasattr(self.model, "initialize_weights"):
            self.model.initialize_weights()

        # Class weights as buffer
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Cached loss function
        self._loss_fn: Optional[nn.Module] = None
        
        # Validation metrics
        if self.is_binary:
            self.val_auroc = BinaryAUROC()
            self.val_acc = BinaryAccuracy()
        else:
            self.val_auroc = MulticlassAUROC(num_classes=n_classes, average="macro")
            self.val_acc = MulticlassAccuracy(num_classes=n_classes, average="macro")
        
        # Log mode info
        self._log_param_counts()

    def _log_param_counts(self) -> None:
        """Log parameter counts for debugging."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"[WSIClassificationModule] Mode: {self.hparams.mode.upper()}")
        print(f"[WSIClassificationModule] Total params: {total_params:,}")
        print(f"[WSIClassificationModule] Trainable: {trainable_params:,}")
        print(f"[WSIClassificationModule] Frozen: {frozen_params:,}")
        
        if self.hparams.mode == "ft":
            print(f"[WSIClassificationModule] Encoder LR: {self.hparams.encoder_lr}, WD: {self.hparams.encoder_wd}")
            print(f"[WSIClassificationModule] Head LR: {self.hparams.head_lr}, WD: {self.hparams.head_wd}")
        else:
            print(f"[WSIClassificationModule] Head LR: {self.hparams.head_lr}, WD: {self.hparams.head_wd}")

    def _get_loss_fn(self) -> nn.Module:
        """Get or create loss function with proper device handling."""
        if self._loss_fn is not None:
            return self._loss_fn
        
        if self.n_classes == 1:
            pos_weight = None
            if self.class_weights is not None and len(self.class_weights) == 2:
                ratio = self.class_weights[1] / (self.class_weights[0] + 1e-6)
                pos_weight = torch.tensor([max(float(ratio), 1.0)])
            self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self._loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        
        return self._loss_fn

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(x, labels=labels)

    def _normalize_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize labels to 1D long tensor."""
        if y.ndim == 2:
            y = y.argmax(dim=-1) if y.size(-1) > 1 else y.squeeze(-1)
        return y.long()

    def _compute_loss(
        self, 
        logits: torch.Tensor, 
        y: torch.Tensor,
        log_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss and convert outputs to probs/labels for metrics.
        
        Returns:
            (loss, probs_for_metrics, labels_for_metrics)
        """
        loss_fn = self._get_loss_fn()
        
        # Move loss function params to correct device if needed
        if hasattr(loss_fn, 'pos_weight') and loss_fn.pos_weight is not None:
            if loss_fn.pos_weight.device != logits.device:
                loss_fn.pos_weight = loss_fn.pos_weight.to(logits.device)
        if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
            if loss_fn.weight.device != logits.device:
                loss_fn.weight = loss_fn.weight.to(logits.device)
        
        if self.n_classes == 1:
            # BCE
            logits_flat = logits.view(-1)
            y_flat = y.view(-1).float()
            loss = loss_fn(logits_flat, y_flat)
            probs = torch.sigmoid(logits_flat)
            labels = y_flat.long()
        else:
            # CE
            y_norm = self._normalize_labels(y)
            loss = loss_fn(logits, y_norm)
            probs = torch.softmax(logits, dim=-1)
            if self.n_classes == 2:
                probs = probs[:, 1]  # Binary: use positive class prob
            labels = y_norm
        
        # Add instance loss for CLAM/DSMIL if available
        if log_dict and isinstance(log_dict, dict):
            instance_loss = log_dict.get("instance_loss")
            if instance_loss is not None and instance_loss != -1:
                if isinstance(instance_loss, torch.Tensor):
                    encoder_type = getattr(self.model, "encoder_type", "").upper()
                    if encoder_type == "CLAM":
                        loss = 0.7 * loss + 0.3 * instance_loss
                    elif encoder_type == "DSMIL":
                        loss = 0.5 * loss + 0.5 * instance_loss
        
        return loss, probs, labels

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits, log_dict = self(x, labels=y)
        loss, probs, labels = self._compute_loss(logits, y, log_dict)
        
        # Update metrics (validation only)
        if stage == "val":
            self.val_auroc.update(probs.detach(), labels.int())
            if self.is_binary:
                preds = (probs >= 0.5).long()
            else:
                preds = probs.argmax(dim=-1)
            self.val_acc.update(preds.detach(), labels.int())
        
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        auroc = self.val_auroc.compute()
        acc = self.val_acc.compute()
        
        auroc_key = "val/roc_auc" if self.is_binary else "val/roc_auc_macro"
        acc_key = "val/accuracy" if self.is_binary else "val/balanced_accuracy"
        
        self.log(auroc_key, auroc, prog_bar=True, sync_dist=True)
        self.log(acc_key, acc, prog_bar=True, sync_dist=True)
        
        self.val_auroc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """
        Configure optimizer with mode-specific parameter groups.
        
        - Linear Probe (lp): Only trainable params (head) with head_lr/head_wd
        - Fine-Tune (ft): Encoder params with encoder_lr/encoder_wd + Head params with head_lr/head_wd
        """
        mode = self.hparams.mode
        
        if mode == "lp":
            # Linear probe: only optimize trainable parameters (head)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            if not trainable_params:
                raise RuntimeError("No trainable parameters found in linear probe mode!")
            
            optimizer = AdamW(
                trainable_params,
                lr=self.hparams.head_lr,
                weight_decay=self.hparams.head_wd,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            
            n_params = sum(p.numel() for p in trainable_params)
            print(f"[Optimizer] LP mode: {n_params:,} params, lr={self.hparams.head_lr}, wd={self.hparams.head_wd}")
            
        else:
            # Fine-tune: separate param groups for encoder and head
            encoder_params = []
            head_params = []
            encoder_param_names = []
            head_param_names = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if _is_head_param(name):
                    head_params.append(param)
                    head_param_names.append(name)
                else:
                    encoder_params.append(param)
                    encoder_param_names.append(name)
            
            param_groups = []
            
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': self.hparams.encoder_lr,
                    'weight_decay': self.hparams.encoder_wd,
                    'name': 'encoder',
                })
                n_enc = sum(p.numel() for p in encoder_params)
                print(f"[Optimizer] Encoder: {n_enc:,} params, lr={self.hparams.encoder_lr}, wd={self.hparams.encoder_wd}")
            
            if head_params:
                param_groups.append({
                    'params': head_params,
                    'lr': self.hparams.head_lr,
                    'weight_decay': self.hparams.head_wd,
                    'name': 'head',
                })
                n_head = sum(p.numel() for p in head_params)
                print(f"[Optimizer] Head: {n_head:,} params, lr={self.hparams.head_lr}, wd={self.hparams.head_wd}")
            
            if not param_groups:
                raise RuntimeError("No trainable parameters found!")
            
            optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        
        # Scheduler
        if self.hparams.scheduler != "cosine":
            return {"optimizer": optimizer}

        max_epochs = self.hparams.max_epochs
        warmup_epochs = self.hparams.warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# =============================================================================
# Data Module Variants
# =============================================================================
class FullTrainDataModule(WSIMILDataModule):
    """All data for training, no validation."""
    def _build_splits(self):
        self.train_ids = self.data_df[self.id_col].astype(str).tolist()
        self.val_ids = []
        self.test_ids = []


class TestOnlyDataModule(WSIMILDataModule):
    """All data for testing only."""
    def _build_splits(self):
        self.train_ids = []
        self.val_ids = []
        self.test_ids = self.data_df[self.id_col].astype(str).tolist()


# =============================================================================
# Main Experiment Class
# =============================================================================
class ExperimentLIT:
    """
    WSI MIL experiment with K-Fold CV.
    
    Supports two modes:
    - Fine-tune (ft): Train encoder + head with different LR/WD
    - Linear probe (lp): Freeze encoder, train only head
    
    Pipeline:
        1. K-fold CV → OOF predictions
        2. Threshold tuning (Youden's J)
        3. Final model (refit or best fold)
        4. Test evaluation with bootstrap CIs
    
    Usage:
        config = ExperimentConfig(...)
        exp = ExperimentLIT(config)
        results = exp.run(save_path="model.pt")
    """

    def __init__(
        self, 
        config: ExperimentConfig, 
        device: Optional[torch.device] = None,
        wandb_run: Optional[Any] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run = wandb_run
        
        # State (reset on each run)
        self._fold_results: List[Dict[str, Any]] = []
        self._oof_logits: Optional[torch.Tensor] = None
        self._oof_labels: Optional[torch.Tensor] = None
        
        # Log mode
        mode_str = "FINE-TUNE" if config.mode == "ft" else "LINEAR PROBE"
        print(f"\n[ExperimentLIT] Mode: {mode_str}")
        if config.mode == "ft":
            print(f"[ExperimentLIT] Encoder LR: {config.encoder_lr}, WD: {config.encoder_wd}")
        print(f"[ExperimentLIT] Head LR: {config.head_lr}, WD: {config.head_wd}")

    def _log_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb if available."""
        if self.wandb_run is not None:
            if step is not None:
                self.wandb_run.log(metrics, step=step)
            else:
                self.wandb_run.log(metrics)

    def _log_summary(self, key: str, value: Any) -> None:
        """Log to wandb summary."""
        if self.wandb_run is not None:
            self.wandb_run.summary[key] = value

    def _log_table(self, name: str, columns: List[str], data: List[List[Any]]) -> None:
        """Log a table to wandb."""
        if self.wandb_run is not None and WANDB_AVAILABLE:
            table = wandb.Table(columns=columns, data=data)
            self.wandb_run.log({name: table})

    def _reset_state(self) -> None:
        """Reset experiment state for fresh run."""
        self._fold_results = []
        self._oof_logits = None
        self._oof_labels = None
        cleanup_memory()

    def _create_model(
        self,
        n_classes: int,
        feature_dim: int,
        class_weights: Optional[torch.Tensor],
        max_epochs: Optional[int] = None,
    ) -> WSIClassificationModule:
        """Factory for creating classification module."""
        return WSIClassificationModule(
            mil=self.config.mil,
            n_classes=n_classes,
            feature_dim=feature_dim,
            mil_attrs=self.config.mil_attrs,
            head_dim=self.config.head_dim,
            head_dropout=self.config.head_dropout,
            num_fc_layers=self.config.num_fc_layers,
            hidden_dim=self.config.hidden_dim,
            ds_dropout=self.config.ds_dropout,
            simple_mlp=self.config.simple_mlp,
            # Mode settings
            mode=self.config.mode,
            encoder_weights_path=self.config.encoder_weights_path,
            # Learning rates
            encoder_lr=self.config.encoder_lr,
            head_lr=self.config.head_lr,
            # Weight decay
            encoder_wd=self.config.encoder_wd,
            head_wd=self.config.head_wd,
            # Scheduler
            scheduler=self.config.scheduler,
            warmup_epochs=self.config.warmup_epochs,
            max_epochs=max_epochs or self.config.max_epochs,
            # Class weights
            class_weights=class_weights if self.config.use_class_weights else None,
        )

    def _create_trainer(
        self, 
        max_epochs: int, 
        callbacks: List[Callback],
        fold: Optional[int] = None,
    ) -> pl.Trainer:
        # Add fold metric logger if wandb is available
        if self.wandb_run is not None and fold is not None:
            fold_logger = FoldMetricLogger(fold=fold, wandb_run=self.wandb_run)
            callbacks = callbacks + [fold_logger]
        
        return pl.Trainer(
            max_epochs=max_epochs,
            precision=self.config.precision,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            gradient_clip_val=self.config.gradient_clip_val,
            callbacks=callbacks,
            logger=False,  # We handle logging ourselves
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=False,
        )

    def _train_fold(self, fold: int) -> Dict[str, Any]:
        """Train a single fold and return results."""
        print(f"\n{'='*20} FOLD {fold} {'='*20}")
        set_seed(self.config.seed + fold)

        # Data
        dm = WSIMILDataModule(
            csv_path=self.config.csv_path,
            feature_dir=self.config.feature_dir,
            id_col=self.config.id_col,
            target_col=self.config.target_col,
            current_fold=fold,
            batch_size=self.config.batch_size,
            bag_size=self.config.bag_size,
            replacement=self.config.replacement,
            num_workers=self.config.num_workers,
            precision=16,
            return_key=False,
            verbose=False,
        )
        dm.setup("fit")

        # Model
        model = self._create_model(
            dm.num_classes, 
            dm.dim_features, 
            dm.class_weight_tensor,
        )
        
        # Callbacks
        monitor = "val/roc_auc" if dm.num_classes <= 2 else "val/roc_auc_macro"
        tracker = BestStateTracker(monitor=monitor, mode="max")
        early_stop = EarlyStopping(
            monitor=monitor, 
            mode="max", 
            patience=self.config.early_stopping_patience,
            verbose=False,
        )

        trainer = self._create_trainer(
            self.config.max_epochs, 
            [tracker, early_stop],
            fold=fold,
        )
        trainer.fit(model, datamodule=dm)

        # Restore best state
        if tracker.best_state is not None:
            model.load_state_dict(tracker.best_state)
        else:
            warnings.warn(f"Fold {fold}: No best state tracked")

        # Collect OOF predictions
        model.to(self.device)
        logits, labels = collect_predictions(model.model, dm.val_dataloader(), self.device)

        result = {
            "fold": fold,
            "val_ids": list(dm.val_ids),
            "logits": logits,
            "labels": labels,
            "best_score": tracker.best_score,
            "best_epoch": tracker.best_epoch,
            "state_dict": copy.deepcopy(tracker.best_state) if tracker.best_state else copy.deepcopy(model.state_dict()),
            "n_classes": dm.num_classes,
            "feature_dim": dm.dim_features,
        }

        score_str = f"{tracker.best_score:.4f}" if tracker.best_score is not None else "N/A"
        epoch_str = f"{tracker.best_epoch + 1}" if tracker.best_epoch is not None else "N/A"
        print(f"[Fold {fold}] score={score_str}, epoch={epoch_str}")
        
        # Log fold summary to wandb
        if tracker.best_score is not None:
            self._log_summary(f"fold_{fold}/best_score", tracker.best_score)
            self._log_summary(f"fold_{fold}/best_epoch", tracker.best_epoch + 1 if tracker.best_epoch else None)
        
        del model, trainer, dm
        cleanup_memory()
        
        return result

    def run_cv(self) -> Dict[str, Any]:
        """Run K-fold cross-validation."""
        df = load_data(self.config.csv_path)
        
        # Get valid folds
        if "k_fold" not in df.columns:
            raise ValueError("CSV must have 'k_fold' column")
        
        folds = sorted([int(f) for f in df["k_fold"].unique() if f >= 0])
        if not folds:
            raise ValueError("No valid folds found (k_fold >= 0)")
        
        print(f"\n{'='*50}")
        print(f"Running {len(folds)}-fold CV")
        print(f"Mode: {self.config.mode.upper()}")
        print(f"Effective batch size: {self.config.effective_batch_size}")
        print(f"{'='*50}")
        
        # Log config to wandb
        if self.wandb_run is not None:
            self.wandb_run.config.update(self.config.to_dict())
        
        # Train folds
        self._fold_results = []
        for k in folds:
            try:
                result = self._train_fold(k)
                self._fold_results.append(result)
            except Exception as e:
                warnings.warn(f"Fold {k} failed: {e}")
                cleanup_memory()
                raise

        if not self._fold_results:
            raise RuntimeError("All folds failed")

        # Build OOF arrays
        df_cv = df[df["k_fold"] >= 0]
        id_to_idx = {str(sid): i for i, sid in enumerate(df_cv[self.config.id_col])}
        
        n_samples = len(id_to_idx)
        n_out = self._fold_results[0]["logits"].shape[-1]
        
        self._oof_logits = torch.zeros(n_samples, n_out)
        self._oof_labels = torch.zeros(n_samples)
        
        for r in self._fold_results:
            idx = torch.tensor([id_to_idx[str(sid)] for sid in r["val_ids"]], dtype=torch.long)
            self._oof_logits[idx] = r["logits"].float()
            self._oof_labels[idx] = r["labels"].float()

        # Compute OOF metrics
        y_true = self._oof_labels.numpy()
        y_prob = logits_to_probs(self._oof_logits)
        p_pos = y_prob[:, 1] if y_prob.shape[1] == 2 else None
        
        # Threshold tuning (binary only)
        if p_pos is not None:
            best_thr, j_score = tune_threshold_youden(y_true, p_pos)
            op_metrics = operating_point_metrics(y_true, p_pos, best_thr)
        else:
            best_thr = 0.5
            op_metrics = {}
        
        oof_metrics = compute_metrics(y_true, y_prob, threshold=best_thr)

        # Print results
        print(f"\n{'='*20} OOF Results {'='*20}")
        print(f"Mode: {self.config.mode.upper()}")
        print(f"Threshold (Youden): {best_thr:.4f}")
        for k, v in oof_metrics.items():
            print(f"  {k}: {v:.4f}")
            self._log_summary(f"oof/{k}", v)
        self._log_summary("oof/threshold", best_thr)
        self._log_summary("mode", self.config.mode)
        
        for k, v in op_metrics.items():
            self._log_summary(f"oof/{k}", v)

        # Log fold scores as table
        if self._fold_results:
            fold_data = [
                [r["fold"], r["best_score"] or 0.0, (r["best_epoch"] or 0) + 1]
                for r in self._fold_results
            ]
            self._log_table(
                "fold_results",
                columns=["fold", "best_score", "best_epoch"],
                data=fold_data
            )
        
        return {
            "oof_logits": self._oof_logits,
            "oof_labels": self._oof_labels,
            "oof_metrics": oof_metrics,
            "op_metrics": op_metrics,
            "threshold": best_thr,
            "fold_results": self._fold_results,
        }

    def train_final(self, df_train: pd.DataFrame, epochs: int) -> WSIClassificationModule:
        """Train final model on all training data."""
        print(f"\n{'='*20} Final Training ({epochs} epochs) {'='*20}")
        set_seed(self.config.seed)

        dm = FullTrainDataModule(
            combined_data=df_train,
            feature_dir=self.config.feature_dir,
            id_col=self.config.id_col,
            target_col=self.config.target_col,
            batch_size=self.config.batch_size,
            bag_size=self.config.bag_size,
            replacement=self.config.replacement,
            num_workers=self.config.num_workers,
            precision=16,
            current_fold=0,
            return_key=False,
        )
        dm.setup("fit")

        model = self._create_model(
            dm.num_classes, 
            dm.dim_features, 
            dm.class_weight_tensor,
            max_epochs=epochs,
        )
        
        trainer = self._create_trainer(epochs, callbacks=[])
        trainer.fit(model, datamodule=dm)
        
        del dm
        cleanup_memory()
        
        return model

    def evaluate_test(
        self,
        model: WSIClassificationModule,
        df_test: pd.DataFrame,
        threshold: float,
    ) -> Dict[str, Any]:
        """Evaluate on test set with bootstrap CIs."""
        print(f"\n{'='*20} Test Evaluation {'='*20}")
        
        if len(df_test) == 0:
            warnings.warn("Empty test set, skipping evaluation")
            return {"metrics": {}, "ci": {}, "op_metrics": {}}

        dm = TestOnlyDataModule(
            combined_data=df_test,
            feature_dir=self.config.feature_dir,
            id_col=self.config.id_col,
            target_col=self.config.target_col,
            batch_size=1,
            num_workers=self.config.num_workers,
            precision=16,
        )
        dm.setup("test")

        model.to(self.device)
        logits, labels = collect_predictions(model.model, dm.test_dataloader(), self.device)
        
        y_true = labels.numpy()
        y_prob = logits_to_probs(logits)
        p_pos = y_prob[:, 1] if y_prob.shape[1] == 2 else None

        # Metrics
        metrics = compute_metrics(y_true, y_prob, threshold)
        ci = bootstrap_ci(
            y_true, y_prob, 
            n_bootstraps=self.config.n_bootstraps,
            ci=0.95, 
            threshold=threshold, 
            seed=self.config.seed,
            stratified=True,
        )
        op = operating_point_metrics(y_true, p_pos, threshold) if p_pos is not None else {}

        # Print and log results
        print(f"\nMode: {self.config.mode.upper()}")
        print("Point Estimates:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            self._log_summary(f"test/{k}", v)
        
        print("\n95% Bootstrap CIs:")
        ci_table_data = []
        for k, v in ci.items():
            print(f"  {k}: {v['mean']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")
            self._log_summary(f"test/{k}_mean", v["mean"])
            self._log_summary(f"test/{k}_lower", v["lower"])
            self._log_summary(f"test/{k}_upper", v["upper"])
            ci_table_data.append([k, v["mean"], v["std"], v["lower"], v["upper"]])
        
        # Log CI as table
        self._log_table(
            "test_bootstrap_ci",
            columns=["metric", "mean", "std", "lower", "upper"],
            data=ci_table_data
        )
        
        # Log operating point metrics
        for k, v in op.items():
            self._log_summary(f"test/{k}", v)

        del dm
        cleanup_memory()
        
        return {"metrics": metrics, "ci": ci, "op_metrics": op}

    def run(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete experiment pipeline.
        
        Args:
            save_path: Optional path to save final model weights
            
        Returns:
            Dict with cv_results, threshold, test_results, final_model
        """
        # Reset state
        self._reset_state()
        
        df = load_data(self.config.csv_path)
        
        # 1. Cross-validation
        cv_results = self.run_cv()
        
        # 2. Determine threshold
        threshold = cv_results["threshold"] if self.config.use_optimal_threshold else 0.5
        print(f"\nUsing threshold: {threshold:.4f}")
        
        # 3. Split data
        df_train = df[df["k_fold"] >= 0].copy()
        df_test = df[df["k_fold"] == -1].copy()
        print(f"Train: {len(df_train)} | Test: {len(df_test)}")
        
        # Log data stats
        self._log_summary("data/n_train", len(df_train))
        self._log_summary("data/n_test", len(df_test))
        self._log_summary("data/n_folds", len(self._fold_results))
        
        # 4. Final model
        if self.config.final_strategy == "refit_full":
            # Use 75th percentile of best epochs
            epochs = [r["best_epoch"] + 1 for r in self._fold_results if r["best_epoch"] is not None]
            final_epochs = int(np.ceil(np.percentile(epochs, 75))) if epochs else self.config.max_epochs
            self._log_summary("final/epochs", final_epochs)
            final_model = self.train_final(df_train, final_epochs)
        else:
            # Use best fold
            scores = [r["best_score"] if r["best_score"] is not None else -np.inf for r in self._fold_results]
            best_idx = int(np.argmax(scores))
            best = self._fold_results[best_idx]
            print(f"Using best fold {best['fold']} (score={best['best_score']:.4f})")
            self._log_summary("final/best_fold", best["fold"])
            final_model = self._create_model(best["n_classes"], best["feature_dim"], None)
            final_model.load_state_dict(best["state_dict"])
        
        # 5. Test evaluation
        test_results = self.evaluate_test(final_model, df_test, threshold)
        
        # 6. Save model
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(final_model.model.state_dict(), save_path)
            print(f"\nModel saved to {save_path}")
            
            # Log model artifact if wandb available
            if self.wandb_run is not None and WANDB_AVAILABLE:
                artifact = wandb.Artifact(
                    name=f"model-{self.wandb_run.id}",
                    type="model",
                    description=f"Final trained model (mode={self.config.mode})"
                )
                artifact.add_file(save_path)
                self.wandb_run.log_artifact(artifact)

        return {
            "cv_results": cv_results,
            "threshold": threshold,
            "test_results": test_results,
            "final_model": final_model,
        }


# =============================================================================
# Exports & Example
# =============================================================================
__all__ = [
    "ExperimentConfig",
    "ExperimentLIT",
    "WSIClassificationModule",
    "compute_metrics",
    "bootstrap_ci",
    "tune_threshold_youden",
    "operating_point_metrics",
]


if __name__ == "__main__":
    # Example: Fine-tune from pretrained encoder
    print("=" * 60)
    print("Example: Fine-Tune Mode")
    print("=" * 60)
    
    config_ft = ExperimentConfig(
        csv_path="data/colon.csv",
        feature_dir="features/uni/",
        id_col="slide_id",
        target_col="grade",
        mil="ABMIL",
        mil_attrs={"L": 512, "D": 128, "K": 1},
        # Mode
        mode="ft",
        encoder_weights_path="checkpoints/ssl_encoder.pt",  # Optional for ft
        # Learning rates (encoder uses 10x lower LR)
        lr=3e-4,           # Base LR
        encoder_lr=3e-5,   # Encoder LR (default: lr * 0.1)
        head_lr=3e-4,      # Head LR (default: lr)
        # Weight decay
        l2_reg=1e-3,
        encoder_wd=1e-3,
        head_wd=1e-3,
        # Training
        max_epochs=60,
        accumulate_grad_batches=8,
        seed=42,
    )
    
    print(f"Config: {config_ft.to_dict()}")
    
    # Example: Linear probe from pretrained encoder
    print("\n" + "=" * 60)
    print("Example: Linear Probe Mode")
    print("=" * 60)
    
    config_lp = ExperimentConfig(
        csv_path="data/colon.csv",
        feature_dir="features/uni/",
        id_col="slide_id",
        target_col="grade",
        mil="ABMIL",
        mil_attrs={"L": 512, "D": 128, "K": 1},
        # Mode
        mode="lp",
        encoder_weights_path="checkpoints/ssl_encoder.pt",  # Required for lp
        # Learning rates (only head is trained)
        lr=3e-4,
        head_lr=3e-4,
        # Weight decay
        l2_reg=1e-3,
        head_wd=1e-3,
        # Training
        max_epochs=30,  # Usually fewer epochs for linear probe
        early_stopping_patience=5,
        seed=42,
    )
    
    print(f"Config: {config_lp.to_dict()}")
    print(f"freeze_encoder: {config_lp.freeze_encoder}")
    
    # Uncomment to run:
    # exp = ExperimentLIT(config_ft)
    # results = exp.run(save_path="checkpoints/model_ft.pt")