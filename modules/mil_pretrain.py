"""
Bag-Level Self-Supervised Learning for WSI MIL.

Uses existing ABMIL and TransMIL aggregators from models.MIL.
Supports: SimCLR (InfoNCE), BYOL, Barlow Twins, VICReg

Usage:
    python ssl_pretrain.py --csv_path /path/to/data.csv --aggregator abmil --ssl_method vicreg
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# Import data module
from data.mmap_datamodule import (
    BagAugmentationConfig,
)

# Import existing MIL aggregators
from models.MIL.ABMIL import ABMIL
from models.MIL.TransMIL import TransMIL


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class SSLConfig:
    """Complete configuration for bag-level SSL training."""
    
    # === Data ===
    csv_path: str = ""
    path_col: str = "path"
    id_col: str = "slide_id"
    
    # === Augmentation ===
    augmentation_strength: str = "medium"
    subsample_ratio_min: float = 0.5
    subsample_ratio_max: float = 0.85
    instance_dropout_prob: float = 0.1
    noise_std: float = 0.08
    feature_dropout_prob: float = 0.05
    min_instances: int = 128
    max_instances: int = 8000
    
    # === Model Architecture ===
    input_dim: int = 1536           # UNIv2=1536, UNI=1024, CONCH=512
    embed_dim: int = 512            # Aggregator embedding dimension
    proj_dim: int = 256             # Projection head output dimension
    aggregator: str = "abmil"       # "abmil" or "transmil"
    
    # ABMIL-specific
    abmil_num_fc_layers: int = 1
    abmil_dropout: float = 0.25
    abmil_attn_dim: int = 384
    abmil_gate: bool = True
    
    # TransMIL-specific
    transmil_num_fc_layers: int = 1
    transmil_dropout: float = 0.1
    transmil_num_attention_layers: int = 2
    transmil_num_heads: int = 8
    
    # === SSL Method ===
    ssl_method: str = "vicreg"      # "simclr", "byol", "barlow", "vicreg"
    
    # SimCLR
    temperature: float = 0.1
    
    # BYOL
    ema_decay: float = 0.996
    ema_decay_end: float = 1.0
    
    # Barlow Twins
    barlow_lambda: float = 0.005
    
    # VICReg
    vicreg_sim_weight: float = 25.0
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 1.0
    
    # === Training ===
    max_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # === Data Loading ===
    num_workers: int = 8
    precision: int = 16
    pin_memory: bool = True
    val_fraction: float = 0.05
    
    # === Checkpointing ===
    checkpoint_dir: str = "checkpoints/ssl"
    save_top_k: int = 3
    early_stopping_patience: int = 15
    
    # === Logging ===
    use_wandb: bool = True
    wandb_project: str = "wsi-ssl"
    wandb_name: Optional[str] = None
    log_every_n_steps: int = 10
    
    # === Misc ===
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_augmentation_config(self) -> BagAugmentationConfig:
        return BagAugmentationConfig(
            subsample_ratio=(self.subsample_ratio_min, self.subsample_ratio_max),
            instance_dropout_prob=self.instance_dropout_prob,
            noise_std=self.noise_std,
            feature_dropout_prob=self.feature_dropout_prob,
            min_instances=self.min_instances,
        )


# =============================================================================
# Projection Heads
# =============================================================================
class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PredictorHead(nn.Module):
    """Predictor head for BYOL-style learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# =============================================================================
# Contrastive Losses
# =============================================================================
class InfoNCELoss(nn.Module):
    """InfoNCE loss (SimCLR / NT-Xent)."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))
        
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        
        return F.cross_entropy(sim, labels)


class BYOLLoss(nn.Module):
    """BYOL loss (negative cosine similarity)."""
    
    def forward(self, online_pred: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
        online_pred = F.normalize(online_pred, dim=-1)
        target_proj = F.normalize(target_proj.detach(), dim=-1)
        return -2 * (online_pred * target_proj).sum(dim=-1).mean()


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss."""
    
    def __init__(self, lambda_coeff: float = 0.005):
        super().__init__()
        self.lambda_coeff = lambda_coeff
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B, D = z1.shape
        
        z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
        z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
        
        c = torch.mm(z1.t(), z2) / B
        
        diag = torch.diagonal(c)
        off_diag = c.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        
        return (diag - 1).pow(2).sum() + self.lambda_coeff * off_diag.pow(2).sum()


class VICRegLoss(nn.Module):
    """VICReg loss."""
    
    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, D = z1.shape
        
        # Invariance
        sim_loss = F.mse_loss(z1, z2)
        
        # Variance
        std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))
        
        # Covariance
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        cov1 = (z1_c.t() @ z1_c) / (B - 1)
        cov2 = (z2_c.t() @ z2_c) / (B - 1)
        cov1.fill_diagonal_(0)
        cov2.fill_diagonal_(0)
        cov_loss = (cov1.pow(2).sum() + cov2.pow(2).sum()) / D
        
        loss = self.sim_weight * sim_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        
        breakdown = {
            "sim_loss": sim_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }
        
        return loss, breakdown


# =============================================================================
# Aggregator Factory
# =============================================================================
def create_aggregator(config: SSLConfig) -> nn.Module:
    """
    Factory function to create ABMIL or TransMIL aggregator.
    
    Args:
        config: SSLConfig with aggregator parameters
        
    Returns:
        Aggregator module (ABMIL or TransMIL)
    """
    aggregator_type = config.aggregator.lower()
    
    if aggregator_type == "abmil":
        return ABMIL(
            in_dim=config.input_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.abmil_num_fc_layers,
            dropout=config.abmil_dropout,
            attn_dim=config.abmil_attn_dim,
            gate=config.abmil_gate,
            num_classes=1,  # Dummy, classifier head not used
        )
    elif aggregator_type == "transmil":
        return TransMIL(
            in_dim=config.input_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.transmil_num_fc_layers,
            dropout=config.transmil_dropout,
            num_attention_layers=config.transmil_num_attention_layers,
            num_classes=1,  # Dummy, classifier head not used
            num_heads=config.transmil_num_heads,
        )
    else:
        raise ValueError(
            f"Unknown aggregator: '{config.aggregator}'. "
            f"Choose 'abmil' or 'transmil'."
        )


# =============================================================================
# SSL Encoder
# =============================================================================
class BagSSLEncoder(nn.Module):
    """
    SSL Encoder: Aggregator (ABMIL/TransMIL) + Projection Head.
    
    Wraps ABMIL/TransMIL with consistent interface for SSL training.
    Only uses forward_features() - ignores classification head.
    """
    
    def __init__(self, config: SSLConfig):
        super().__init__()
        
        self.config = config
        self.aggregator_type = config.aggregator.lower()
        
        # Create aggregator
        self.aggregator = create_aggregator(config)
        
        # Projection head: embed_dim -> proj_dim
        self.projector = ProjectionHead(
            input_dim=config.embed_dim,
            hidden_dim=config.embed_dim,
            output_dim=config.proj_dim,
            n_layers=2,
        )
        
        # Store dimensions for external access
        self.output_dim = config.embed_dim
        self.proj_dim = config.proj_dim
    
    def _forward_aggregator(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward through aggregator with consistent interface.
        
        Args:
            x: [B, N, D] bag features
            mask: [B, N] boolean mask (True = valid) - only ABMIL supports this
            return_attention: Whether to return attention weights
            
        Returns:
            features: [B, embed_dim] bag-level features
            attention: [B, N] attention weights (or None)
        """
        if self.aggregator_type == "abmil":
            # ABMIL expects attn_mask with 1=valid, 0=masked (float)
            attn_mask = None
            if mask is not None:
                attn_mask = mask.float()
            
            features, log_dict = self.aggregator.forward_features(
                x, attn_mask=attn_mask, return_attention=return_attention
            )
            attention = log_dict.get('attention') if return_attention else None
            
        elif self.aggregator_type == "transmil":
            # TransMIL doesn't support masking
            if mask is not None:
                # Log warning once
                if not hasattr(self, '_mask_warning_logged'):
                    print("⚠️  TransMIL does not support masking. Mask will be ignored.")
                    self._mask_warning_logged = True
            
            features, log_dict = self.aggregator.forward_features(
                x, return_attention=return_attention
            )
            attention = log_dict.get('attention') if return_attention else None
        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")
        
        return features, attention
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Full forward pass: Aggregator + Projection.
        
        Args:
            x: [B, N, D] or [N, D] bag features
            mask: [B, N] boolean mask (True = valid)
            return_attention: Whether to return attention weights
            
        Returns:
            If return_attention=False: projection [B, proj_dim]
            If return_attention=True: (projection, representation, attention)
        """
        # Ensure 3D input
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        # Forward through aggregator
        repr_feat, attention = self._forward_aggregator(x, mask, return_attention)
        
        # Project
        proj = self.projector(repr_feat)
        
        # Squeeze if input was 2D
        if squeeze:
            proj = proj.squeeze(0)
            repr_feat = repr_feat.squeeze(0)
            if attention is not None and attention.ndim > 1:
                attention = attention.squeeze(0)
        
        if return_attention:
            return proj, repr_feat, attention
        return proj
    
    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get bag representation (before projection).
        
        Args:
            x: [B, N, D] or [N, D] bag features
            mask: [B, N] boolean mask
            
        Returns:
            representation: [B, embed_dim] or [embed_dim]
        """
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        repr_feat, _ = self._forward_aggregator(x, mask, return_attention=False)
        
        if squeeze:
            repr_feat = repr_feat.squeeze(0)
        
        return repr_feat


# =============================================================================
# Lightning Module
# =============================================================================
class BagSSLModule(pl.LightningModule):
    """
    PyTorch Lightning module for bag-level SSL.
    
    Supports: SimCLR, BYOL, Barlow Twins, VICReg
    Aggregators: ABMIL, TransMIL
    """
    
    def __init__(self, config: SSLConfig):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        
        # Online encoder
        self.encoder = BagSSLEncoder(config)
        
        # SSL method setup
        self.ssl_method = config.ssl_method.lower()
        
        if self.ssl_method == "byol":
            # Target encoder (EMA updated)
            self.target_encoder = BagSSLEncoder(config)
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            
            # Freeze target encoder
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            
            # Predictor (online only)
            self.predictor = PredictorHead(
                input_dim=config.proj_dim,
                hidden_dim=config.embed_dim,
                output_dim=config.proj_dim,
            )
            
            self.loss_fn = BYOLLoss()
            self.ema_decay = config.ema_decay
            
        elif self.ssl_method == "simclr":
            self.loss_fn = InfoNCELoss(temperature=config.temperature)
            
        elif self.ssl_method == "barlow":
            self.loss_fn = BarlowTwinsLoss(lambda_coeff=config.barlow_lambda)
            
        elif self.ssl_method == "vicreg":
            self.loss_fn = VICRegLoss(
                sim_weight=config.vicreg_sim_weight,
                var_weight=config.vicreg_var_weight,
                cov_weight=config.vicreg_cov_weight,
            )
        else:
            raise ValueError(
                f"Unknown SSL method: '{config.ssl_method}'. "
                f"Choose 'simclr', 'byol', 'barlow', or 'vicreg'."
            )
        
        # Tracking
        self.train_loss_accum = []
    
    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """EMA update of target encoder (BYOL only)."""
        if self.ssl_method != "byol":
            return
        
        # Cosine schedule for EMA decay
        if self.config.ema_decay_end != self.config.ema_decay:
            progress = self.current_epoch / max(1, self.config.max_epochs - 1)
            decay = self.config.ema_decay + (self.config.ema_decay_end - self.config.ema_decay) * (
                1 - math.cos(math.pi * progress)
            ) / 2
        else:
            decay = self.ema_decay
        
        for online_p, target_p in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_p.data = decay * target_p.data + (1 - decay) * online_p.data
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode bags to representations (for downstream use).
        
        Args:
            x: [B, N, D] or [N, D] bag features
            mask: [B, N] boolean mask
            
        Returns:
            representation: [B, embed_dim] bag-level features
        """
        return self.encoder.encode(x, mask)
    
    def _process_batch(
        self,
        views1: List[torch.Tensor],
        views2: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process a batch of variable-size bags.
        
        Args:
            views1, views2: Lists of [N_i, D] tensors
            
        Returns:
            loss: Scalar loss
            log_dict: Metrics to log
        """
        device = self.device
        B = len(views1)
        
        # Collect projections
        z1_list = []
        z2_list = []
        repr_list = []
        
        for v1, v2 in zip(views1, views2):
            # Move to device and add batch dim
            v1 = v1.unsqueeze(0).to(device, non_blocking=True)
            v2 = v2.unsqueeze(0).to(device, non_blocking=True)
            
            if self.ssl_method == "byol":
                # Online forward
                p1 = self.encoder(v1)
                p2 = self.encoder(v2)
                q1 = self.predictor(p1)
                q2 = self.predictor(p2)
                
                # Target forward (no grad)
                with torch.no_grad():
                    t1 = self.target_encoder(v1)
                    t2 = self.target_encoder(v2)
                
                z1_list.append((q1.squeeze(0), t2.squeeze(0)))
                z2_list.append((q2.squeeze(0), t1.squeeze(0)))
            else:
                # SimCLR / Barlow / VICReg
                proj1, repr1, _ = self.encoder(v1, return_attention=True)
                proj2 = self.encoder(v2)
                
                z1_list.append(proj1.squeeze(0))
                z2_list.append(proj2.squeeze(0))
                repr_list.append(repr1.squeeze(0))
        
        # Compute loss
        log_dict = {}
        
        if self.ssl_method == "byol":
            # Symmetric BYOL loss
            loss = torch.tensor(0.0, device=device)
            for (q1, t2), (q2, t1) in zip(z1_list, z2_list):
                loss = loss + (self.loss_fn(q1, t2) + self.loss_fn(q2, t1)) / 2
            loss = loss / B
        else:
            # Stack for batch loss computation
            z1 = torch.stack(z1_list, dim=0)
            z2 = torch.stack(z2_list, dim=0)
            
            if self.ssl_method == "vicreg":
                loss, breakdown = self.loss_fn(z1, z2)
                log_dict.update({f"ssl/{k}": v for k, v in breakdown.items()})
            else:
                loss = self.loss_fn(z1, z2)
            
            # Track representation statistics
            if repr_list:
                repr_stack = torch.stack(repr_list, dim=0)
                log_dict["ssl/repr_std"] = repr_stack.std(dim=0).mean()
                log_dict["ssl/repr_mean"] = repr_stack.mean()
        
        log_dict["ssl/loss"] = loss.detach()
        
        return loss, log_dict
    
    def training_step(
        self,
        batch: Tuple[List[torch.Tensor], List[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        views1, views2 = batch
        bs = len(views1)
        
        loss, log_dict = self._process_batch(views1, views2)
        
        # Update target encoder (BYOL)
        if self.ssl_method == "byol":
            self._update_target_encoder()
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        for k, v in log_dict.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=bs)
        
        self.train_loss_accum.append(loss.detach())
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        if self.train_loss_accum:
            avg_loss = torch.stack(self.train_loss_accum).mean()
            self.log("train/epoch_loss", avg_loss, batch_size=1)
            self.train_loss_accum.clear()
    
    def validation_step(
        self,
        batch: Tuple[List[torch.Tensor], List[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        views1, views2 = batch
        bs = len(views1)
        
        loss, log_dict = self._process_batch(views1, views2)
        
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        for k, v in log_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, batch_size=bs)
        
        return loss
    
    def configure_optimizers(self):
        # Parameters to optimize
        if self.ssl_method == "byol":
            params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        else:
            params = self.encoder.parameters()
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        
        # Cosine schedule with warmup
        def lr_lambda(epoch):
            if epoch < self.config.warmup_epochs:
                return (epoch + 1) / self.config.warmup_epochs
            progress = (epoch - self.config.warmup_epochs) / max(
                1, self.config.max_epochs - self.config.warmup_epochs
            )
            min_lr_ratio = self.config.min_lr / self.config.lr
            return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }