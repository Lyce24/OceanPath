from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import lightning.pytorch as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassAUROC,
)

from modules.cv.utils.train_utils import _is_head_param
    
from models.MIL.wsi_model import WSIModel

# =============================================================================
# Lightning Module
# =============================================================================
class WSIClassificationModule(pl.LightningModule):
    """
    WSI MIL classification with fine-tune/linear-probe modes.

    """

    def __init__(
        self,
        mil: str,
        n_classes: int,
        feature_dim: int,
        mil_attrs: Optional[Dict[str, Any]] = None,
        # Mode
        mode: Literal["ft", "lp"] = "ft",
        encoder_weights_path: Optional[str] = None,
        # LR
        encoder_lr: float = 3e-5,
        head_lr: float = 3e-4,
        # WD
        encoder_wd: float = 1e-3,
        head_wd: float = 1e-3,
        # Scheduler
        scheduler: Literal["cosine", "none"] = "cosine",
        min_lr: float = 1e-6,
        warmup_epochs: int = 0,
        max_epochs: int = 60,
        # Loss
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        if mode not in ("ft", "lp"):
            raise ValueError(f"mode must be 'ft' or 'lp', got '{mode}'")
        if mode == "lp" and encoder_weights_path is None:
            raise ValueError("Linear probe mode requires encoder_weights_path")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

        freeze_encoder = (mode == "lp")
        self.save_hyperparameters(ignore=["class_weights"])

        self.n_classes = int(n_classes)
        self.is_binary = self.n_classes == 2

        self.model = WSIModel(
            input_feature_dim=feature_dim,
            n_classes=n_classes,
            encoder_type=mil,
            encoder_attrs=mil_attrs or {},
            encoder_weights_path=encoder_weights_path,
            freeze_encoder=freeze_encoder,
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self._loss_fn: Optional[nn.Module] = None

        # Online val metrics
        if self.is_binary:
            self.val_auroc = BinaryAUROC()
            self.val_acc = BinaryAccuracy()
        else:
            self.val_auroc = MulticlassAUROC(num_classes=self.n_classes, average="macro")
            # macro average accuracy ~= balanced accuracy
            self.val_acc = MulticlassAccuracy(num_classes=self.n_classes, average="macro")

        self._log_param_counts()

    def _log_param_counts(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"[WSIClassificationModule] Mode: {self.hparams.mode.upper()}")
        print(f"[WSIClassificationModule] Total params: {total_params:,}")
        print(f"[WSIClassificationModule] Trainable: {trainable_params:,}")
        print(f"[WSIClassificationModule] Frozen: {frozen_params:,}")
        if self.hparams.mode == "ft":
            print(f"[WSIClassificationModule] Encoder LR/WD: {self.hparams.encoder_lr} / {self.hparams.encoder_wd}")
        print(f"[WSIClassificationModule] Head   LR/WD: {self.hparams.head_lr} / {self.hparams.head_wd}")
        print(f"[WSIClassificationModule] Scheduler: {self.hparams.scheduler}")

    def _get_loss_fn(self) -> nn.Module:
        if self._loss_fn is not None:
            return self._loss_fn
        self._loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=float(self.hparams.label_smoothing),
        )
        return self._loss_fn

    def forward(self, x: torch.Tensor, return_raw_attention: bool = False, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logits, log_dict = self.model.forward(x, return_raw_attention=return_raw_attention, labels=labels)
        return logits, log_dict
    
    def forward_features(self, x, return_raw_attention=False):
        features, log_dict = self.model.forward_features(x, return_attention=return_raw_attention) # [B, D] or [B, N, D]
        return features, log_dict
    
    def _normalize_labels(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2:
            y = y.argmax(dim=-1) if y.size(-1) > 1 else y.squeeze(-1)
        return y.long()

    def _compute_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        log_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_fn = self._get_loss_fn()
        y_norm = self._normalize_labels(y)
        loss = loss_fn(logits, y_norm)

        probs_full = torch.softmax(logits, dim=-1)
        if self.n_classes == 2:
            probs_for_metrics = probs_full[:, 1]  # [B]
        else:
            probs_for_metrics = probs_full  # [B,C]
            
        return loss, probs_for_metrics, y_norm

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits, log_dict = self(x, labels=y)
        loss, probs, labels = self._compute_loss(logits, y, log_dict)

        if stage == "val":
            if self.is_binary:
                self.val_auroc.update(probs.detach(), labels.int())
                preds = (probs >= 0.5).long()
                self.val_acc.update(preds.detach(), labels.int())
            else:
                self.val_auroc.update(probs.detach(), labels.int())
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

        if self.is_binary:
            self.log("val/roc_auc", auroc, prog_bar=True, sync_dist=True)
            self.log("val/accuracy", acc, prog_bar=True, sync_dist=True)
        else:
            self.log("val/roc_auc_macro", auroc, prog_bar=True, sync_dist=True)
            # macro accuracy ~= balanced accuracy
            self.log("val/balanced_accuracy", acc, prog_bar=True, sync_dist=True)

        self.val_auroc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        mode = self.hparams.mode

        if mode == "lp":
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters found in linear probe mode!")
            optimizer = AdamW(
                trainable_params,
                lr=float(self.hparams.head_lr),
                weight_decay=float(self.hparams.head_wd),
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            encoder_params: List[torch.nn.Parameter] = []
            head_params: List[torch.nn.Parameter] = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if _is_head_param(name):
                    head_params.append(param)
                else:
                    encoder_params.append(param)

            param_groups = []
            if encoder_params:
                param_groups.append(
                    {
                        "params": encoder_params,
                        "lr": float(self.hparams.encoder_lr),
                        "weight_decay": float(self.hparams.encoder_wd),
                        "name": "encoder",
                    }
                )
            if head_params:
                param_groups.append(
                    {
                        "params": head_params,
                        "lr": float(self.hparams.head_lr),
                        "weight_decay": float(self.hparams.head_wd),
                        "name": "head",
                    }
                )
            if not param_groups:
                raise RuntimeError("No trainable parameters found!")

            optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

        # Scheduler
        if self.hparams.scheduler == "none":
            return {"optimizer": optimizer}

        if self.hparams.scheduler != "cosine":
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

        min_lr = float(self.hparams.min_lr)
        base_lr = float(self.hparams.head_lr)  # floor uses head LR reference
        lr_floor_ratio = min_lr / base_lr

        max_epochs = int(self.hparams.max_epochs)
        warmup_epochs = int(self.hparams.warmup_epochs)

        def lr_lambda_epoch(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return max(cosine_decay, lr_floor_ratio)

        scheduler = LambdaLR(optimizer, lr_lambda_epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
