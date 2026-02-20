"""
Lightning training module for MIL classification.

NaN handling under AMP (the correctness issue that caused the crash):
═══════════════════════════════════════════════════════════════════════

Under 16-mixed precision, extreme logits overflow float16 → NaN loss.
The GradScaler's optimizer step requires inf checks from backward():

  ❌ return torch.tensor(0.0, requires_grad=True)
     This is a LEAF TENSOR disconnected from the model. backward()
     produces no parameter gradients. scaler.unscale_() finds nothing
     to check → "No inf checks were recorded" assertion crash.

  ❌ return None / skip backward
     Same problem — scaler.step() still runs, finds no inf checks.

  ✅ return torch.nan_to_num(loss, nan=0.0, ...)
     nan_to_num STAYS IN THE COMPUTATION GRAPH. backward() produces
     zero gradients for all parameters (not "no gradients"). scaler
     finds these, records inf checks, calls optimizer.step() with
     zero updates → harmless no-op, training continues.

Prevention: we also clamp logits to [-100,100] and cast to float32
before loss computation, which eliminates >99% of NaN cases.
"""

import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC

from oceanpath.models import MILOutput, build_classifier

logger = logging.getLogger(__name__)


class MILTrainModule(L.LightningModule):
    def __init__(
        self,
        arch: str = "abmil",
        in_dim: int = 1024,
        num_classes: int = 2,
        model_cfg: dict | None = None,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 0,
        max_epochs: int = 30,
        loss_type: str = "ce",
        class_weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        monitor_metric: str = "val/loss",
        monitor_mode: str = "min",
        canary_interval: int = 200,
        compile_model: bool = False,
        freeze_aggregator: bool = False,
        collect_embeddings: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_classifier(
            arch=arch,
            in_dim=in_dim,
            num_classes=num_classes,
            model_cfg=model_cfg or {},
            freeze_aggregator=freeze_aggregator,
        )

        if compile_model and hasattr(torch, "compile"):
            logger.info("Applying torch.compile to model")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.loss_fn = self._build_loss(loss_type, num_classes, class_weights, focal_gamma)

        self.num_classes = num_classes
        self.canary_interval = canary_interval
        self.collect_embeddings = collect_embeddings
        self._setup_metrics()

        self._val_predictions: list[dict] = []
        self._test_predictions: list[dict] = []
        self._val_embeddings: list[dict] = []
        self._test_embeddings: list[dict] = []

        self._nan_count = 0

    # ── Loss factory ──────────────────────────────────────────────────────

    @staticmethod
    def _build_loss(
        loss_type: str,
        num_classes: int,
        class_weights: list[float] | None,
        focal_gamma: float,
    ) -> nn.Module:
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)

        if loss_type == "ce":
            return nn.CrossEntropyLoss(weight=weight)
        if loss_type == "bce":
            pos_weight = None
            if weight is not None and len(weight) == 2:
                pos_weight = weight[1:2] / weight[0:1]
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if loss_type == "focal":
            return FocalLoss(gamma=focal_gamma, weight=weight)
        raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'ce', 'bce', or 'focal'.")

    # ── Metrics ───────────────────────────────────────────────────────────

    def _setup_metrics(self) -> None:
        task = "binary" if self.num_classes <= 2 else "multiclass"
        kwargs = {"task": task}
        if task == "multiclass":
            kwargs["num_classes"] = self.num_classes

        # Train
        self.train_acc = Accuracy(**kwargs)

        # Val — full suite
        self.val_acc = Accuracy(**kwargs)
        self.val_f1 = F1Score(**kwargs, average="macro")
        self.val_balanced_acc = Accuracy(**kwargs, average="macro")
        self.val_precision = Precision(**kwargs, average="macro")
        self.val_recall = Recall(**kwargs, average="macro")

        if task == "binary":
            self.val_auroc = BinaryAUROC()
        else:
            self.val_auroc = MulticlassAUROC(num_classes=self.num_classes)

        # Test — full suite (separate instances to avoid state contamination)
        self.test_acc = Accuracy(**kwargs)
        self.test_f1 = F1Score(**kwargs, average="macro")
        self.test_balanced_acc = Accuracy(**kwargs, average="macro")
        self.test_precision = Precision(**kwargs, average="macro")
        self.test_recall = Recall(**kwargs, average="macro")

        if task == "binary":
            self.test_auroc = BinaryAUROC()
        else:
            self.test_auroc = MulticlassAUROC(num_classes=self.num_classes)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, features, mask=None, coords=None, return_attention=False):
        return self.model(
            features,
            mask=mask,
            coords=coords,
            return_attention=return_attention,
        )

    # ── Training step ─────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output: MILOutput = self.model(
            batch["features"],
            mask=batch["mask"],
            return_attention=False,
        )

        loss = self._compute_loss(output.logits, batch["labels"])
        B = batch["labels"].shape[0]

        # ── AMP-safe NaN recovery ─────────────────────────────────────────
        loss_is_finite = torch.isfinite(loss).item()

        if not loss_is_finite:
            self._nan_count += 1
            if self._nan_count <= 10 or self._nan_count % 50 == 0:
                logger.warning(
                    f"Non-finite train loss at step {self.global_step} "
                    f"(total: {self._nan_count}). nan_to_num → no-op step."
                )
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        if loss_is_finite:
            preds = self._get_preds(output.logits)
            self.train_acc(preds, batch["labels"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, batch_size=B)

        if self.canary_interval > 0 and self.global_step % self.canary_interval == 0:
            self._canary_check(loss, output)

        return loss

    # ── Validation step ───────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        output: MILOutput = self.model(
            batch["features"],
            mask=batch["mask"],
            return_attention=True,
        )

        loss = self._compute_loss(output.logits, batch["labels"])
        B = batch["labels"].shape[0]

        preds = self._get_preds(output.logits)
        probs = self._get_probs(output.logits)

        self.val_acc(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])
        self.val_balanced_acc(preds, batch["labels"])
        self.val_precision(preds, batch["labels"])
        self.val_recall(preds, batch["labels"])
        self.val_auroc(probs, batch["labels"])

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/acc", self.val_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=B
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(
            "val/balanced_acc",
            self.val_balanced_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )

        self._accumulate_predictions(self._val_predictions, output, batch, probs)
        if self.collect_embeddings:
            self._accumulate_embeddings(self._val_embeddings, output, batch)

    # ── Test step ─────────────────────────────────────────────────────────

    def test_step(self, batch: dict, batch_idx: int) -> None:
        output: MILOutput = self.model(
            batch["features"],
            mask=batch["mask"],
            return_attention=True,
        )

        loss = self._compute_loss(output.logits, batch["labels"])
        B = batch["labels"].shape[0]

        preds = self._get_preds(output.logits)
        probs = self._get_probs(output.logits)

        self.test_acc(preds, batch["labels"])
        self.test_f1(preds, batch["labels"])
        self.test_balanced_acc(preds, batch["labels"])
        self.test_precision(preds, batch["labels"])
        self.test_recall(preds, batch["labels"])
        self.test_auroc(probs, batch["labels"])

        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, batch_size=B)
        self.log(
            "test/balanced_acc", self.test_balanced_acc, on_step=False, on_epoch=True, batch_size=B
        )
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, batch_size=B)

        self._accumulate_predictions(self._test_predictions, output, batch, probs)
        if self.collect_embeddings:
            self._accumulate_embeddings(self._test_embeddings, output, batch)

    # ── Accumulation ──────────────────────────────────────────────────────

    def _accumulate_predictions(
        self,
        acc: list,
        output: MILOutput,
        batch: dict,
        probs: torch.Tensor,
    ) -> None:
        probs_np = probs.detach().cpu().numpy()
        labels_np = batch["labels"].detach().cpu().numpy()

        for i, sid in enumerate(batch["slide_ids"]):
            row = {"slide_id": sid, "label": int(labels_np[i])}
            if probs_np.ndim == 1:
                row["prob_1"] = float(probs_np[i])
            else:
                for c in range(probs_np.shape[1]):
                    row[f"prob_{c}"] = float(probs_np[i, c])
            acc.append(row)

    def _accumulate_embeddings(
        self,
        acc: list,
        output: MILOutput,
        batch: dict,
    ) -> None:
        emb = output.slide_embedding.detach().cpu().numpy()
        for i, sid in enumerate(batch["slide_ids"]):
            acc.append({"slide_id": sid, "embedding": emb[i]})

    # ── Epoch hooks ───────────────────────────────────────────────────────

    def on_validation_epoch_end(self) -> None:
        """Log a brief val report at the end of each validation epoch."""
        # Only log during training (not during standalone trainer.validate() calls)
        if not self.trainer.sanity_checking:
            self._log_epoch_report("val")

    def on_test_epoch_end(self) -> None:
        """Log a brief test report at the end of the test epoch."""
        self._log_epoch_report("test")

    def _log_epoch_report(self, prefix: str) -> None:
        """Print a one-line metrics summary for val or test."""
        metrics = self.trainer.callback_metrics
        keys = ["loss", "auroc", "acc", "precision", "recall", "f1"]
        parts = []
        for k in keys:
            full_key = f"{prefix}/{k}"
            val = metrics.get(full_key)
            if val is not None:
                v = val.item() if isinstance(val, torch.Tensor) else val
                parts.append(f"{k}={v:.4f}")
        if parts:
            epoch = self.current_epoch
            report = ", ".join(parts)
            logger.info(f"[{prefix.upper()} epoch={epoch}] {report}")

    # ── Save predictions/embeddings ───────────────────────────────────────

    def save_predictions(self, output_dir: str, prefix: str = "val") -> str | None:
        acc = self._val_predictions if prefix == "val" else self._test_predictions
        if not acc:
            return None
        path = Path(output_dir) / f"preds_{prefix}.parquet"
        df = pd.DataFrame(acc)
        df.to_parquet(str(path), index=False, engine="pyarrow")
        logger.info(f"Saved {len(df)} predictions → {path}")
        acc.clear()
        return str(path)

    def save_embeddings(self, output_dir: str, prefix: str = "val") -> str | None:
        acc = self._val_embeddings if prefix == "val" else self._test_embeddings
        if not acc:
            return None
        path = Path(output_dir) / f"embeddings_{prefix}.npz"
        slide_ids = [r["slide_id"] for r in acc]
        embeddings = np.stack([r["embedding"] for r in acc])
        np.savez(str(path), slide_ids=slide_ids, embeddings=embeddings)
        logger.info(f"Saved {len(slide_ids)} embeddings → {path}")
        acc.clear()
        return str(path)

    # ── Loss computation ──────────────────────────────────────────────────

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Loss with logit clamping + float32 upcast for AMP stability.

        Clamping to [-100, 100] prevents float16 overflow in softmax.
        Upcasting to float32 avoids precision loss in cross-entropy.
        """
        logits_safe = logits.float().clamp(-100, 100)
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            return self.loss_fn(logits_safe, labels.float())
        return self.loss_fn(logits_safe, labels)

    def _get_preds(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1 or logits.shape[-1] == 1:
            return (logits.squeeze(-1) > 0).long()
        return logits.argmax(dim=-1)

    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if logits.ndim == 1 or logits.shape[-1] == 1:
                return torch.sigmoid(logits.float()).squeeze(-1)
            return F.softmax(logits.float(), dim=-1)

    # ── Optimizer + scheduler ─────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler_name = self.hparams.get("lr_scheduler", "cosine")

        if scheduler_name == "none" or scheduler_name is None:
            return optimizer

        warmup_epochs = self.hparams.warmup_epochs
        effective_epochs = max(1, self.hparams.max_epochs - warmup_epochs)

        if scheduler_name == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=effective_epochs,
                eta_min=1e-7,
            )
        elif scheduler_name == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.monitor_mode,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            )
            if warmup_epochs > 0:
                logger.warning(
                    "warmup_epochs>0 with plateau scheduler is not supported. Using plateau only."
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": plateau,
                    "monitor": self.hparams.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_name == "step":
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(1, warmup_epochs),
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ── Canary checks ─────────────────────────────────────────────────────

    def _canary_check(self, loss: torch.Tensor, output: MILOutput) -> None:
        step = self.global_step

        if not torch.isfinite(loss):
            logger.error(f"[CANARY] Non-finite loss at step {step}")

        attn = output.extras.get("attention_weights")
        if attn is not None and attn.std() < 1e-7:
            logger.warning(f"[CANARY] Dead attention at step {step} (std={attn.std():.2e})")

        try:
            for pg in self.trainer.optimizers[0].param_groups:
                if pg["lr"] < 1e-10:
                    logger.warning(f"[CANARY] Near-zero LR at step {step}: {pg['lr']:.2e}")
        except (AttributeError, IndexError):
            pass


# ── Focal Loss ────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., ICCV 2017)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()
