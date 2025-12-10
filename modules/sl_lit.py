from __future__ import annotations

import math
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import lightning.pytorch as pl

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    MulticlassAUROC,
    MulticlassAccuracy,
)

from utils.train_utils import adamw_param_groups
from models.wsi_model import WSIModel


class WSIClassificationModule(pl.LightningModule):
    """
    Lightning wrapper around WSIModel for classification.

    - Supports ABMIL / TransMIL / CLAM / DSMIL / etc. via `mil` + `mil_attrs`
    - Handles binary vs multiclass metrics
    - Optional class weights
    - Optional pretrained_state
    """

    def __init__(
        self,
        # MIL / model config
        mil: str,
        n_classes: int,
        feature_dim: int,
        mil_attrs: Optional[Dict[str, Any]] = None,
        freeze_encoder: bool = False,
        head_dim: int = 512,
        head_dropout: float = 0.25,
        num_fc_layers: int = 1,
        hidden_dim: int = 128,
        ds_dropout: float = 0.3,
        simple_mlp: bool = False,

        # optimization
        lr: float = 3e-4,
        l2_reg: float = 1e-3,
        scheduler: str = "cosine",   # or "none"
        warmup_epochs: int = 0,
        max_epochs: int = 60,
        step_on_epochs: bool = True,
        
        # data-driven
        class_weights: Optional[torch.Tensor] = None,
        pretrained_state: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["pretrained_state", "class_weights"])

        self.n_classes = int(n_classes)
        self.is_binary_like = (self.n_classes == 1 or self.n_classes == 2)

        # -----------------
        # Build WSI model
        # -----------------
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
        )

        if pretrained_state is not None:
            self.model.load_state_dict(pretrained_state)
            print("[WSIClassificationModule] Loaded pretrained state_dict.")
        else:
            if hasattr(self.model, "initialize_weights"):
                self.model.initialize_weights()

        # -----------------
        # Loss weights
        # -----------------
        if class_weights is not None:
            # Will be moved to correct device automatically
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # -----------------
        # Metrics
        # -----------------
        if self.is_binary_like:
            # We will map logits -> probs for positive class
            self.val_auroc = BinaryAUROC()
            self.val_acc = BinaryAccuracy()
            self.test_auroc = BinaryAUROC()
            self.test_acc = BinaryAccuracy()
        else:
            self.val_auroc_mc = MulticlassAUROC(
                num_classes=self.n_classes, average="macro"
            )
            self.val_acc_mc = MulticlassAccuracy(
                num_classes=self.n_classes, average="macro"
            )
            self.test_auroc_mc = MulticlassAUROC(
                num_classes=self.n_classes, average="macro"
            )
            self.test_acc_mc = MulticlassAccuracy(
                num_classes=self.n_classes, average="macro"
            )

    # -----------------
    # Forward
    # -----------------
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_raw_attention: bool = False,
    ):
        """
        Forward through WSIModel.

        WSIModel.forward returns (logits, log_dict).
        We expose that so we can use instance losses / attention if needed.
        """
        logits, log_dict = self.model(
            x,
            return_raw_attention=return_raw_attention,
            labels=labels,
        )
        return logits, log_dict

    # -----------------
    # Shared step
    # -----------------
    def _shared_step(self, batch, stage: str):
        # Expected batch: (bags, labels) from WSIBagDataset
        x, y = batch

        logits, log_dict = self(x, labels=y)

        # ------------- Loss -------------
        if self.n_classes == 1:
            # BCE-with-logits, y in {0,1}
            logits_flat = logits.view(-1)
            y_flat = y.view(-1).float()

            if self.class_weights is not None and len(self.class_weights) == 2:
                # class_weights from prepare_labels are inverse-freq style.
                # Derive a pos_weight that roughly matches class imbalance.
                # pos_weight > 1 => penalize FN more.
                w0, w1 = self.class_weights[0].item(), self.class_weights[1].item()
                # guard: if w0==0, fallback to 1.0
                pos_weight = torch.tensor(
                    max(w1 / (w0 + 1e-6), 1.0),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()

            loss = criterion(logits_flat, y_flat)
            probs = torch.sigmoid(logits_flat)
            preds = (probs >= 0.5).long()
            y_for_metrics = y_flat.long()

        elif self.n_classes == 2:
            # 2-class CE, but binary metrics on prob of class 1
            # logits: [B,2], y: [B]
            if y.ndim == 2 and y.size(-1) == 1:
                y = y.view(-1)
            elif y.ndim == 2 and y.size(-1) == 2:
                y = y.argmax(dim=-1)
            y = y.long()

            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights if self.class_weights is not None else None
            )
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=-1)[:, 1]  # prob of positive class
            preds = (probs >= 0.5).long()
            y_for_metrics = y

        else:
            # Multiclass CE, macro metrics
            if y.ndim == 2:
                if y.size(-1) == 1:
                    y = y.view(-1)
                else:
                    y = y.argmax(dim=-1)
            y = y.long()

            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights if self.class_weights is not None else None
            )
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            y_for_metrics = y

        # ------------- Optional instance loss (CLAM) -------------
        # DSMIL stored instance_loss as float in log_dict['instance_loss'],
        # CLAM likely returns a tensor. We only LOG it, not backprop it
        # (your original train loop is not shown, so we keep this conservative).
        instance_loss_val = None
        if isinstance(log_dict, dict) and "instance_loss" in log_dict:
            il = log_dict["instance_loss"]
            # Make it a scalar tensor for logging
            if isinstance(il, torch.Tensor):
                instance_loss_val = il.detach()
            elif isinstance(il, (float, int)):
                instance_loss_val = torch.tensor(il, device=loss.device, dtype=loss.dtype)

        # ------------- Metrics update -------------
        if stage == "val":
            if self.is_binary_like:
                self.val_auroc.update(probs.detach(), y_for_metrics.int())
                self.val_acc.update(preds.detach(), y_for_metrics.int())
            else:
                self.val_auroc_mc.update(probs.detach(), y_for_metrics)
                self.val_acc_mc.update(preds.detach(), y_for_metrics)
        elif stage == "test":
            if self.is_binary_like:
                self.test_auroc.update(probs.detach(), y_for_metrics.int())
                self.test_acc.update(preds.detach(), y_for_metrics.int())
            else:
                self.test_auroc_mc.update(probs.detach(), y_for_metrics)
                self.test_acc_mc.update(preds.detach(), y_for_metrics)

        return loss, instance_loss_val

    # -----------------
    # Training
    # -----------------
    def training_step(self, batch, batch_idx):
        loss, instance_loss_val = self._shared_step(batch, stage="train")
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if instance_loss_val is not None:
            self.log(
                "train/instance_loss",
                instance_loss_val,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    # -----------------
    # Validation
    # -----------------
    def validation_step(self, batch, batch_idx):
        loss, instance_loss_val = self._shared_step(batch, stage="val")
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        if instance_loss_val is not None:
            self.log(
                "val/instance_loss",
                instance_loss_val,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def on_validation_epoch_end(self):
        if self.is_binary_like:
            auroc = self.val_auroc.compute()
            acc = self.val_acc.compute()
            self.log("val/roc_auc", auroc, prog_bar=True, sync_dist=True)
            self.log("val/accuracy", acc, prog_bar=True, sync_dist=True)
            self.val_auroc.reset()
            self.val_acc.reset()
        else:
            auroc = self.val_auroc_mc.compute()
            acc = self.val_acc_mc.compute()
            self.log("val/roc_auc_macro", auroc, prog_bar=True, sync_dist=True)
            self.log("val/balanced_accuracy", acc, prog_bar=True, sync_dist=True)
            self.val_auroc_mc.reset()
            self.val_acc_mc.reset()

    # -----------------
    # Test
    # -----------------
    def test_step(self, batch, batch_idx):
        loss, instance_loss_val = self._shared_step(batch, stage="test")
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        if instance_loss_val is not None:
            self.log(
                "test/instance_loss",
                instance_loss_val,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def on_test_epoch_end(self):
        if self.is_binary_like:
            auroc = self.test_auroc.compute()
            acc = self.test_acc.compute()
            self.log("test/roc_auc", auroc, sync_dist=True)
            self.log("test/accuracy", acc, sync_dist=True)
            self.test_auroc.reset()
            self.test_acc.reset()
        else:
            auroc = self.test_auroc_mc.compute()
            acc = self.test_acc_mc.compute()
            self.log("test/roc_auc_macro", auroc, sync_dist=True)
            self.log("test/balanced_accuracy", acc, sync_dist=True)
            self.test_auroc_mc.reset()
            self.test_acc_mc.reset()

    # -----------------
    # Optimizer / Scheduler
    # -----------------
    def configure_optimizers(self):
        optim_groups = adamw_param_groups(self.model, self.hparams.l2_reg)
        optimizer = AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if self.hparams.scheduler.lower() != "cosine":
            return {"optimizer": optimizer}

        max_epochs = self.hparams.max_epochs
        warmup_epochs = self.hparams.warmup_epochs
        step_on_epochs = self.hparams.step_on_epochs

        def lr_lambda(current_epoch: int):
            # Warmup
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))

            total = max(max_epochs - warmup_epochs, 1)
            progress = (current_epoch - warmup_epochs) / total
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if step_on_epochs else "step",
            },
        }
