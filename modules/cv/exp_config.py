from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

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
    mil_attrs: Dict[str, Any] = field(default_factory=dict) # embed_dim, num_fc_layers, dropout, attn_dim, gate, num_attention_layers, num_heads
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
    min_lr: float = 1e-6
    precision: Literal["16-mixed", "bf16-mixed", "32"] = "16-mixed"
    early_stopping_patience: int = 10
    min_delta: float = 0.002 # Minimum change to qualify as an improvement
    
    # Gradient settings
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = 1.0
    
    # Data loading
    batch_size: int = 1
    bag_size: Optional[int] = None
    replacement: bool = False
    num_workers: Optional[int] = None
    
    # Experiment settings
    cv_type: Literal["k-fold", "oof", "mccv"] = "k-fold"
    final_strategy: Literal["refit_full", "best_fold", "ensemble"] = "ensemble"
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    n_bootstraps: int = 1000
    seed: int = 42
    
    # Logging
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 10
    
    
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
        if self.min_lr <= 0 or self.min_lr >= self.lr:
            raise ValueError(f"min_lr must be positive and < lr, got min_lr={self.min_lr}, lr={self.lr}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.check_val_every_n_epoch < 1:
            raise ValueError("check_val_every_n_epoch must be >= 1")
        if self.log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be >= 1")
        
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
            "cv_type": self.cv_type,
            "max_epochs": self.max_epochs,
            "scheduler": self.scheduler,
            "min_lr": self.min_lr,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "effective_batch_size": self.effective_batch_size,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "gradient_clip_val": self.gradient_clip_val,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
            "use_class_weights": self.use_class_weights,
            "label_smoothing": self.label_smoothing,
            "final_strategy": self.final_strategy,
            "seed": self.seed,
            # Logging
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
            "log_every_n_steps": self.log_every_n_steps,
        }