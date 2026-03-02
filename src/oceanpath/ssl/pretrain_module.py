"""
Lightning module for SSL pretraining.

Supports five methods via config: vicreg, simclr, byol, dino, jepa.

Architecture (all methods):
  ┌────────────────────────────────────────────────────────────────┐
  │  view1 [B, M₁, D] ─→ aggregator ─→ embed₁ [B, E]            │
  │  view2 [B, M₂, D] ─→ aggregator ─→ embed₂ [B, E]            │
  │                                                                │
  │  Method-specific heads:                                        │
  │    VICReg:  projector(embed) → z,  loss(z₁, z₂)              │
  │    SimCLR:  projector(embed) → z,  NT-Xent(z₁, z₂)          │
  │    BYOL:    projector+predictor(embed) → p,  EMA target       │
  │    DINO:    dino_head(embed) → logits,  EMA teacher            │
  │    JEPA:    predictor(embed₁) → pred, EMA target(embed₂)     │
  └────────────────────────────────────────────────────────────────┘

The aggregator is any registered BaseMIL model (ABMIL, TransMIL, etc.).
After pretraining, the aggregator weights are saved for fine-tuning in
Stage 5 via: train.aggregator_weights_path = pretrain_checkpoint.

Batch format (from DualViewCollator):
  view1    [B, M₁, D]     mask1   [B, M₁]
  view2    [B, M₂, D]     mask2   [B, M₂]
  coords1  [B, M₁, 2]     coords2 [B, M₂, 2]   (optional)
"""

import logging

import lightning as L
import torch
import torch.nn as nn

from oceanpath.models import build_aggregator
from oceanpath.ssl.heads import (
    DINOHead,
    EMANetwork,
    Predictor,
    Projector,
)
from oceanpath.ssl.losses import (
    BYOLLoss,
    DINOLoss,
    JEPALoss,
    SimCLRLoss,
    VICRegLoss,
)

logger = logging.getLogger(__name__)


class _EMATarget(nn.Module):
    """Named wrapper for EMA target components (aggregator + head/projector).

    Replaces nn.Sequential to avoid fragile index-based access like
    ``ema_model[0]``, ``ema_model[1]``. Instead provides named attributes
    ``aggregator`` and ``head``.
    """

    def __init__(self, aggregator: nn.Module, head: nn.Module):
        super().__init__()
        self.aggregator = aggregator
        self.head = head

    def forward(self, x, **kwargs):
        out = self.aggregator(x, **kwargs)
        return self.head(out.slide_embedding)


class SSLPretrainModule(L.LightningModule):
    """
    Unified SSL pretraining module.

    Parameters
    ----------
    ssl_method : str
        One of: 'vicreg', 'simclr', 'byol', 'dino', 'jepa'.
    arch : str
        Aggregator architecture ('abmil', 'transmil', 'meanpool').
    in_dim : int
        Patch feature dimension.
    model_cfg : dict
        Aggregator hyperparameters (embed_dim, dropout, etc.).
    ssl_cfg : dict
        SSL-specific hyperparameters (projector dims, loss weights, etc.).
    lr : float
        Base learning rate.
    weight_decay : float
        AdamW weight_decay.
    warmup_epochs : int
        Linear warmup epochs.
    max_epochs : int
        Total training epochs (for cosine schedule).
    lr_scheduler : str
        'cosine' or 'none'.
    """

    def __init__(
        self,
        ssl_method: str = "vicreg",
        arch: str = "abmil",
        in_dim: int = 1024,
        model_cfg: dict | None = None,
        ssl_cfg: dict | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        lr_scheduler: str = "cosine",
    ):
        super().__init__()
        self.save_hyperparameters()

        model_cfg = model_cfg or {}
        ssl_cfg = ssl_cfg or {}

        self.ssl_method = ssl_method
        self.embed_dim = model_cfg.get("embed_dim", 512)

        # ── Build aggregator (online/student) ─────────────────────────────
        self.aggregator = build_aggregator(arch=arch, in_dim=in_dim, model_cfg=model_cfg)

        # ── Build SSL heads + losses (method-specific) ────────────────────
        proj_hidden = ssl_cfg.get("proj_hidden_dim", 2048)
        proj_out = ssl_cfg.get("proj_out_dim", 256)
        proj_layers = ssl_cfg.get("proj_num_layers", 3)

        if ssl_method == "vicreg":
            self._build_vicreg(ssl_cfg, proj_hidden, proj_out, proj_layers)
        elif ssl_method == "simclr":
            self._build_simclr(ssl_cfg, proj_hidden, proj_out, proj_layers)
        elif ssl_method == "byol":
            self._build_byol(ssl_cfg, proj_hidden, proj_out, proj_layers)
        elif ssl_method == "dino":
            self._build_dino(ssl_cfg, proj_hidden, proj_out)
        elif ssl_method == "jepa":
            self._build_jepa(ssl_cfg, proj_hidden, proj_out, proj_layers)
        else:
            raise ValueError(
                f"Unknown ssl_method '{ssl_method}'. Choose from: vicreg, simclr, byol, dino, jepa"
            )

    # ── Method-specific builders ──────────────────────────────────────────

    def _build_vicreg(self, cfg, hidden, out, layers):
        # VICReg: no BN in projector (BN interferes with variance term)
        self.projector = Projector(self.embed_dim, hidden, out, num_layers=layers, use_bn=False)
        self.loss_fn = VICRegLoss(
            inv_weight=cfg.get("inv_weight", 25.0),
            var_weight=cfg.get("var_weight", 25.0),
            cov_weight=cfg.get("cov_weight", 1.0),
        )

    def _build_simclr(self, cfg, hidden, out, layers):
        self.projector = Projector(self.embed_dim, hidden, out, num_layers=layers, use_bn=True)
        self.loss_fn = SimCLRLoss(
            temperature=cfg.get("temperature", 0.1),
        )

    def _build_byol(self, cfg, hidden, out, layers):
        pred_hidden = cfg.get("pred_hidden_dim", 1024)
        ema_momentum = cfg.get("ema_momentum", 0.996)
        ema_final = cfg.get("ema_final_momentum", 1.0)

        # Online branch: projector + predictor
        self.projector = Projector(self.embed_dim, hidden, out, num_layers=layers, use_bn=True)
        self.predictor = Predictor(out, pred_hidden, out, use_bn=True)

        # Target branch: EMA of (aggregator + projector)
        self._online_for_ema = _EMATarget(self.aggregator, self.projector)
        self.target_network = EMANetwork(
            self._online_for_ema,
            initial_momentum=ema_momentum,
            final_momentum=ema_final,
        )
        self.loss_fn = BYOLLoss()

    def _build_dino(self, cfg, hidden, out):
        n_prototypes = cfg.get("n_prototypes", 4096)
        bottleneck = cfg.get("bottleneck_dim", 256)
        teacher_temp = cfg.get("teacher_temp", 0.04)
        student_temp = cfg.get("student_temp", 0.1)
        center_momentum = cfg.get("center_momentum", 0.9)
        ema_momentum = cfg.get("ema_momentum", 0.996)
        ema_final = cfg.get("ema_final_momentum", 1.0)

        # Student head
        self.student_head = DINOHead(self.embed_dim, hidden, bottleneck, n_prototypes, use_bn=True)

        # Teacher: EMA of (aggregator + student_head)
        self._student_for_ema = _EMATarget(self.aggregator, self.student_head)
        self.teacher_network = EMANetwork(
            self._student_for_ema,
            initial_momentum=ema_momentum,
            final_momentum=ema_final,
        )

        self.loss_fn = DINOLoss(
            out_dim=n_prototypes,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

    def _build_jepa(self, cfg, hidden, out, layers):
        pred_hidden = cfg.get("pred_hidden_dim", 1024)
        ema_momentum = cfg.get("ema_momentum", 0.996)
        ema_final = cfg.get("ema_final_momentum", 1.0)
        loss_type = cfg.get("loss_type", "smooth_l1")

        # Context branch: projector
        self.projector = Projector(self.embed_dim, hidden, out, num_layers=layers, use_bn=True)
        # Predictor: context proj → target proj space
        self.predictor = Predictor(out, pred_hidden, out, use_bn=True)

        # Target: EMA of (aggregator + projector)
        self._online_for_ema = _EMATarget(self.aggregator, self.projector)
        self.target_network = EMANetwork(
            self._online_for_ema,
            initial_momentum=ema_momentum,
            final_momentum=ema_final,
        )

        self.loss_fn = JEPALoss(loss_type=loss_type)

    # ── Forward helpers ───────────────────────────────────────────────────

    def _encode_view(
        self,
        view: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run aggregator on one view → slide embedding [B, E]."""
        output = self.aggregator(view, mask=mask, coords=coords)
        return output.slide_embedding

    @torch.no_grad()
    def _encode_view_target(
        self,
        view: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run target/teacher network on one view → projected [B, proj_out]."""
        if self.ssl_method in ("byol", "jepa"):
            ema_target: _EMATarget = self.target_network.ema_model
            output = ema_target.aggregator(view, mask=mask, coords=coords)
            return ema_target.head(output.slide_embedding)
        if self.ssl_method == "dino":
            ema_target: _EMATarget = self.teacher_network.ema_model
            output = ema_target.aggregator(view, mask=mask, coords=coords)
            return ema_target.head(output.slide_embedding)
        return None

    # ── Training step ─────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        view1, mask1 = batch["view1"], batch["mask1"]
        view2, mask2 = batch["view2"], batch["mask2"]
        coords1 = batch.get("coords1")
        coords2 = batch.get("coords2")

        if self.ssl_method == "vicreg":
            return self._step_vicreg(view1, mask1, coords1, view2, mask2, coords2)
        if self.ssl_method == "simclr":
            return self._step_simclr(view1, mask1, coords1, view2, mask2, coords2)
        if self.ssl_method == "byol":
            return self._step_byol(view1, mask1, coords1, view2, mask2, coords2)
        if self.ssl_method == "dino":
            return self._step_dino(view1, mask1, coords1, view2, mask2, coords2)
        if self.ssl_method == "jepa":
            return self._step_jepa(view1, mask1, coords1, view2, mask2, coords2)
        return None

    def _step_vicreg(self, v1, m1, c1, v2, m2, c2):
        e1 = self._encode_view(v1, m1, c1)
        e2 = self._encode_view(v2, m2, c2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)

        loss_dict = self.loss_fn(z1, z2)
        self._log_losses(loss_dict, prefix="train")
        return loss_dict["loss"]

    def _step_simclr(self, v1, m1, c1, v2, m2, c2):
        e1 = self._encode_view(v1, m1, c1)
        e2 = self._encode_view(v2, m2, c2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)

        loss_dict = self.loss_fn(z1, z2)
        self._log_losses(loss_dict, prefix="train")
        return loss_dict["loss"]

    def _step_byol(self, v1, m1, c1, v2, m2, c2):
        # Online branch: encode → project → predict
        e1 = self._encode_view(v1, m1, c1)
        e2 = self._encode_view(v2, m2, c2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target branch: EMA encode → project (no predictor, stop-grad)
        with torch.no_grad():
            zt1 = self._encode_view_target(v1, m1, c1)
            zt2 = self._encode_view_target(v2, m2, c2)

        loss_dict = self.loss_fn(p1, p2, zt1, zt2)
        self._log_losses(loss_dict, prefix="train")

        # Log EMA momentum
        self.log("train/ema_momentum", self.target_network.current_momentum)

        return loss_dict["loss"]

    def _step_dino(self, v1, m1, c1, v2, m2, c2):
        # Student: encode → head
        e1 = self._encode_view(v1, m1, c1)
        e2 = self._encode_view(v2, m2, c2)
        s1 = self.student_head(e1)
        s2 = self.student_head(e2)

        # Teacher: EMA encode → head (no grad)
        with torch.no_grad():
            t1 = self._encode_view_target(v1, m1, c1)
            t2 = self._encode_view_target(v2, m2, c2)

        # Cross-distillation: student-view1 learns from teacher-view2 and vice versa
        loss1 = self.loss_fn(s1, t2)
        loss2 = self.loss_fn(s2, t1)
        loss = (loss1["loss"] + loss2["loss"]) * 0.5

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ema_momentum", self.teacher_network.current_momentum)

        return loss

    def _step_jepa(self, v1, m1, c1, v2, m2, c2):
        # Context (view1): encode → project → predict
        e1 = self._encode_view(v1, m1, c1)
        z1 = self.projector(e1)
        pred = self.predictor(z1)

        # Target (view2): EMA encode → project (stop-grad)
        with torch.no_grad():
            target = self._encode_view_target(v2, m2, c2)

        loss_dict = self.loss_fn(pred, target)
        self._log_losses(loss_dict, prefix="train")
        self.log("train/ema_momentum", self.target_network.current_momentum)

        return loss_dict["loss"]

    # ── EMA update (after optimizer step) ─────────────────────────────────

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA target network after each training step."""
        if self.ssl_method in ("byol", "jepa"):
            self.target_network.update(self._online_for_ema)
        elif self.ssl_method == "dino":
            self.teacher_network.update(self._student_for_ema)

    # ── Validation step ───────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Compute val loss + collect embeddings for RankMe/alpha-ReQ.

        Returns dict with 'loss' and 'embeddings' for callbacks.
        """
        view1, mask1 = batch["view1"], batch["mask1"]
        view2, mask2 = batch["view2"], batch["mask2"]
        coords1 = batch.get("coords1")
        coords2 = batch.get("coords2")

        # Encode both views
        e1 = self._encode_view(view1, mask1, coords1)
        e2 = self._encode_view(view2, mask2, coords2)

        # Compute loss (method-specific, same as training)
        if self.ssl_method == "vicreg" or self.ssl_method == "simclr":
            z1, z2 = self.projector(e1), self.projector(e2)
            loss_dict = self.loss_fn(z1, z2)
        elif self.ssl_method == "byol":
            z1, z2 = self.projector(e1), self.projector(e2)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            with torch.no_grad():
                zt1 = self._encode_view_target(view1, mask1, coords1)
                zt2 = self._encode_view_target(view2, mask2, coords2)
            loss_dict = self.loss_fn(p1, p2, zt1, zt2)
        elif self.ssl_method == "dino":
            s1 = self.student_head(e1)
            s2 = self.student_head(e2)
            with torch.no_grad():
                t1 = self._encode_view_target(view1, mask1, coords1)
                t2 = self._encode_view_target(view2, mask2, coords2)
            l1 = self.loss_fn(s1, t2)
            l2 = self.loss_fn(s2, t1)
            loss_dict = {"loss": (l1["loss"] + l2["loss"]) * 0.5}
        elif self.ssl_method == "jepa":
            z1 = self.projector(e1)
            pred = self.predictor(z1)
            with torch.no_grad():
                target = self._encode_view_target(view2, mask2, coords2)
            loss_dict = self.loss_fn(pred, target)

        val_loss = loss_dict["loss"]
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)

        # Return embeddings for RankMe/alpha-ReQ callbacks
        # Use view1 embeddings (arbitrary choice, both valid)
        return {"loss": val_loss, "embeddings": e1.detach()}

    # ── Optimizer ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Exclude EMA parameters (they have requires_grad=False)
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == "none":
            return optimizer

        warmup_epochs = self.hparams.warmup_epochs
        effective_epochs = max(1, self.hparams.max_epochs - warmup_epochs)

        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=effective_epochs, eta_min=1e-6
        )

        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(1, warmup_epochs),
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ── Logging helpers ───────────────────────────────────────────────────

    def _log_losses(self, loss_dict: dict, prefix: str = "train") -> None:
        for k, v in loss_dict.items():
            prog_bar = k == "loss"
            self.log(f"{prefix}/{k}", v, prog_bar=prog_bar, sync_dist=True)

    # ── Checkpoint: save only aggregator weights ──────────────────────────

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Also save bare aggregator state_dict for easy downstream loading.

        The full checkpoint has the complete module (projector, predictor, etc.)
        but downstream fine-tuning only needs the aggregator.
        """
        checkpoint["aggregator_state_dict"] = self.aggregator.state_dict()
        checkpoint["ssl_method"] = self.ssl_method
        checkpoint["embed_dim"] = self.embed_dim
