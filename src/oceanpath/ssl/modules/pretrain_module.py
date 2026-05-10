"""
Lightning module for SSL pretraining.

Supports four methods via config: vicreg, jepa, lejepa, lejepa_mc.

Architecture (dual-view methods):
  ┌────────────────────────────────────────────────────────────────┐
  │  view1 [B, M₁, D] ─→ aggregator ─→ embed₁ [B, E]            │
  │  view2 [B, M₂, D] ─→ aggregator ─→ embed₂ [B, E]            │
  │                                                                │
  │  Method-specific heads:                                        │
  │    VICReg:    projector(embed) → z,  loss(z₁, z₂)            │
  │    JEPA:      projector+predictor(embed) → p, EMA target;    │
  │               symmetric: each view predicts the other view's │
  │               EMA target.                                     │
  │    LeJEPA:    projector(embed) → z, centroid + SIGReg        │
  └────────────────────────────────────────────────────────────────┘

Architecture (multi-crop, lejepa_mc):
  ┌────────────────────────────────────────────────────────────────┐
  │  V_g global views + V_l local views                            │
  │  each [B, N, D] ─→ aggregator ─→ embed [B, E] ─→ projector(z) │
  │                                                                │
  │  Loss: centroid over GLOBAL views' z, every view predicts μ;   │
  │        SIGReg on every view; no EMA, no predictor.             │
  └────────────────────────────────────────────────────────────────┘

The aggregator is any registered BaseMIL model (ABMIL, TransMIL, Mamba2MIL).
After pretraining, the aggregator weights are saved for fine-tuning in
Stage 5 via: training.aggregator_weights_path = pretrain_checkpoint.

Batch format
────────────
* Dual-view collator output (stack_collate):
    view1, view2     [B, fixed_n, D]
    mask1, mask2     [B, fixed_n]
    coords1, coords2 [B, fixed_n, 2]   (optional)
* Multi-crop collator output (multicrop_stack_collate):
    global_views    [V_g, B, N_g, D]   local_views    [V_l, B, N_l, D]
    global_masks    [V_g, B, N_g]      local_masks    [V_l, B, N_l]
    global_coords   [V_g, B, N_g, 2]   local_coords   [V_l, B, N_l, 2]   (optional)
    num_global, num_local, slide_ids
"""

import logging

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from oceanpath.models import build_aggregator
from oceanpath.ssl.modules.heads import (
    EMANetwork,
    Predictor,
    Projector,
)
from oceanpath.ssl.modules.losses import (
    JEPALoss,
    LeJEPALoss,
    LeJEPAMCLoss,
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
    """Unified SSL pretraining module.

    Parameters
    ----------
    ssl_method : str
        One of: 'vicreg', 'jepa', 'lejepa', 'lejepa_mc'.
    arch : str
        Aggregator architecture ('abmil', 'transmil', 'meanpool', 'mamba2mil', ...).
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
    adam_betas : tuple[float, float]
        AdamW (beta1, beta2). Defaults to (0.9, 0.95) — the ViT / I-JEPA /
        LeJEPA convention. Override via `pretrain_training.adam_betas`.
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
        adam_betas: tuple[float, float] = (0.9, 0.95),
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        lr_scheduler: str = "cosine",
    ):
        super().__init__()
        self.save_hyperparameters()

        model_cfg = model_cfg or {}
        ssl_cfg = ssl_cfg or {}

        method = ssl_method.lower()
        self.requested_ssl_method = method

        # Normalise legacy / synonym names. 'jepa' is the canonical
        # slide-level JEPA name; the older 'pooled_jepa' / 'token_jepa'
        # tags collapse onto it.
        if method in {"pooled_jepa", "token_jepa"}:
            method = "jepa"

        self.ssl_method = method
        self.embed_dim = model_cfg.get("embed_dim", 512)

        self.aggregator = build_aggregator(
            arch=arch,
            in_dim=in_dim,
            model_cfg=model_cfg,
        )

        actual_embed_dim = getattr(self.aggregator, "embed_dim", self.embed_dim)
        if actual_embed_dim != self.embed_dim:
            raise ValueError(
                f"SSL head dim mismatch: model_cfg embed_dim={self.embed_dim}, "
                f"but aggregator reports embed_dim={actual_embed_dim}"
            )

        proj_hidden = ssl_cfg.get("proj_hidden_dim", 2048)
        proj_out = ssl_cfg.get("proj_out_dim", 256)
        proj_layers = ssl_cfg.get("proj_num_layers", 3)

        if method == "vicreg":
            self._build_vicreg(ssl_cfg, proj_hidden, proj_out, proj_layers)
        elif method == "jepa":
            self._build_jepa(ssl_cfg, proj_hidden, proj_out, proj_layers)
        elif method == "lejepa":
            self._build_lejepa(ssl_cfg, proj_hidden, proj_out, proj_layers)
        elif method == "lejepa_mc":
            self._build_lejepa_mc(ssl_cfg, proj_hidden, proj_out, proj_layers)
        else:
            raise ValueError(
                f"Unknown ssl_method '{ssl_method}'. "
                "Choose from: vicreg, jepa, lejepa, lejepa_mc "
                "(legacy aliases: pooled_jepa, token_jepa → jepa)."
            )

    # ── Method-specific builders ──────────────────────────────────────────

    def _build_vicreg(self, cfg, hidden, out, layers):
        # VICReg: no BN in projector (BN interferes with variance term).
        # Forward the full set of VICRegLoss kwargs from ssl_cfg so the YAML
        # remains the single source of truth — the previous version silently
        # dropped `var_target` / `var_eps`.
        self.projector = Projector(self.embed_dim, hidden, out, num_layers=layers, use_bn=False)
        self.loss_fn = VICRegLoss(
            inv_weight=cfg.get("inv_weight", 25.0),
            var_weight=cfg.get("var_weight", 25.0),
            cov_weight=cfg.get("cov_weight", 4.0),
            var_target=cfg.get("var_target", 1.0),
            eps=cfg.get("var_eps", 1e-4),
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

        self.loss_fn = JEPALoss(
            loss_type=loss_type,
            beta=cfg.get("beta", 1.0),
        )

    def _build_lejepa(self, cfg, hidden, out, layers):
        """LeJEPA-2C: same projector geometry as VICReg/JEPA, no predictor,
        no EMA target. Loss: predictive (centroid match) + SIGReg.

        Per the LeJEPA paper, BatchNorm in the projector is fine (unlike VICReg)
        because SIGReg replaces the variance-collapse mechanism. The proposal
        config uses ``proj_use_bn=true, proj_last_bn=false`` — read these from
        ssl_cfg with sensible defaults.
        """
        use_bn = bool(cfg.get("proj_use_bn", True))
        last_bn = bool(cfg.get("proj_last_bn", False))

        self.projector = Projector(
            self.embed_dim,
            hidden,
            out,
            num_layers=layers,
            use_bn=use_bn,
            last_bn=last_bn,
        )
        self.loss_fn = LeJEPALoss(
            sigreg_weight=cfg.get("sigreg_weight", 0.05),
            num_slices=cfg.get("num_slices", 1024),
            integration_points=cfg.get("integration_points", 17),
            integration_range=cfg.get("integration_range", 5.0),
        )

    def _build_lejepa_mc(self, cfg, hidden, out, layers):
        """LeJEPA-MC: same head as LeJEPA-2C; loss handles V_g + V_l views.

        The number of global / local views is set on the augmentor via the
        DataModule (``num_global_views``, ``num_local_views`` in
        augmentation_cfg). We do NOT also read them here — the per-batch
        ``num_global`` / ``num_local`` fields produced by the collator are
        the single source of truth at step time, so any mismatch between
        ``augmentation_cfg`` and ``ssl_cfg`` would be a duplicated knob
        rather than a useful sanity check.
        """
        use_bn = bool(cfg.get("proj_use_bn", True))
        last_bn = bool(cfg.get("proj_last_bn", False))

        self.projector = Projector(
            self.embed_dim,
            hidden,
            out,
            num_layers=layers,
            use_bn=use_bn,
            last_bn=last_bn,
        )
        self.loss_fn = LeJEPAMCLoss(
            sigreg_weight=cfg.get("sigreg_weight", 0.05),
            num_slices=cfg.get("num_slices", 1024),
            integration_points=cfg.get("integration_points", 17),
            integration_range=cfg.get("integration_range", 5.0),
        )

    # ── Forward helpers ───────────────────────────────────────────────────

    def _encode_view(
        self,
        view: torch.Tensor | list[torch.Tensor],
        mask: torch.Tensor | None,
        coords: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run aggregator on one view → slide embeddings [B, E].

        Handles both padded tensors [B, N, D] and lists of variable-length
        tensors [Ni, D]. The list path loops over bags individually —
        optimal for ABMIL/DSMIL where there's no inter-sample attention.
        """
        if isinstance(view, list):
            embeddings = []
            for i, bag in enumerate(view):
                bag = bag.unsqueeze(0).to(self.device)  # [1, Ni, D]
                c = None
                if coords is not None and coords[i] is not None:
                    c = coords[i].unsqueeze(0).to(self.device)
                out = self.aggregator(bag, mask=None, coords=c)
                embeddings.append(out.slide_embedding.squeeze(0))
            return torch.stack(embeddings)  # [B, E]

        # Padded tensor path — unchanged
        output = self.aggregator(view, mask=mask, coords=coords)
        return output.slide_embedding

    @torch.no_grad()
    def _encode_view_target(
        self,
        view: torch.Tensor | list[torch.Tensor],
        mask: torch.Tensor | None,
        coords: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run target/teacher network on one view → projected [B, proj_out]."""
        if self.ssl_method != "jepa":
            return None
        ema_target: _EMATarget = self.target_network.ema_model

        if isinstance(view, list):
            projections = []
            for i, bag in enumerate(view):
                bag = bag.unsqueeze(0).to(self.device)
                c = None
                if coords is not None and coords[i] is not None:
                    c = coords[i].unsqueeze(0).to(self.device)
                out = ema_target.aggregator(bag, mask=None, coords=c)
                proj = ema_target.head(out.slide_embedding)
                projections.append(proj.squeeze(0))
            return torch.stack(projections)

        output = ema_target.aggregator(view, mask=mask, coords=coords)
        return ema_target.head(output.slide_embedding)

    # ── Training step ─────────────────────────────────────────────────────

    def _batch_size(self, view) -> int:
        """Infer batch size from either list or tensor view."""
        return len(view) if isinstance(view, list) else view.shape[0]

    def on_fit_start(self):
        total_steps = int(self.trainer.estimated_stepping_batches)

        if self.ssl_method == "jepa":
            self.target_network.total_steps = max(1, total_steps)

            # The EMA update fires in ``on_train_batch_end``, which runs once
            # per dataloader batch — NOT once per optimizer step. With
            # gradient accumulation > 1 the target would drift faster than
            # the optimizer sees updates, silently breaking training. Refuse
            # to start rather than producing a hard-to-debug bad run.
            accum = int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1)
            if accum != 1:
                raise ValueError(
                    f"JEPA EMA update is dataloader-batch-aligned but "
                    f"accumulate_grad_batches={accum}. Use "
                    "accumulate_grad_batches=1 for JEPA, or move the EMA "
                    "update to an optimizer-step-aligned hook (e.g. "
                    "on_before_optimizer_step)."
                )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Multi-crop has a different batch contract — handle it before the
        # dual-view fields are unpacked.
        if self.ssl_method == "lejepa_mc":
            return self._step_lejepa_mc(batch)

        view1 = batch["view1"]
        view2 = batch["view2"]
        mask1 = batch.get("mask1")
        mask2 = batch.get("mask2")
        coords1 = batch.get("coords1")
        coords2 = batch.get("coords2")

        if self.ssl_method == "vicreg":
            return self._step_vicreg(view1, mask1, coords1, view2, mask2, coords2, batch)
        if self.ssl_method == "jepa":
            return self._step_jepa(view1, mask1, coords1, view2, mask2, coords2, batch)
        if self.ssl_method == "lejepa":
            return self._step_lejepa(view1, mask1, coords1, view2, mask2, coords2, batch)
        # Should be unreachable: __init__ only accepts the four methods above
        # (plus 'lejepa_mc' handled above).
        raise RuntimeError(f"training_step: unsupported ssl_method '{self.ssl_method}'.")

    def _step_vicreg(self, v1, m1, c1, v2, m2, c2, batch):
        e1 = self._encode_view(v1, m1, c1)
        e2 = self._encode_view(v2, m2, c2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)

        B = self._batch_size(v1)
        self._log_z_dim_std([z1, z2], batch_size=B)
        self._log_z_centroid_metrics(z1, z2, batch_size=B)
        self._log_aug_metrics(v1, v2, batch, batch_size=B)
        if self._should_log_token_overlap():
            self._log_aug_token_overlap(c1, c2, batch_size=B)

        loss_dict = self.loss_fn(z1, z2)
        self._log_losses(loss_dict, prefix="train", batch_size=B)
        return loss_dict["loss"]

    def _step_jepa(self, v1, m1, c1, v2, m2, c2, batch):
        """Symmetric slide-level JEPA step.

        Both views go through online (encode → project → predict). Both
        views also go through the EMA target (encode → project, no
        predictor, stop-grad). Loss is the average of the two cross
        predictions: pred(view1) → target(view2) and pred(view2) →
        target(view1). This matches the symmetric two-view convention used
        by VICReg / LeJEPA / SimSiam, so the four-method comparison in the
        proposal is fair.
        """
        # Online branch.
        e1 = self._encode_view(v1, m1, c1)
        e2 = self._encode_view(v2, m2, c2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # EMA target branch (stop-grad).
        with torch.no_grad():
            t1 = self._encode_view_target(v1, m1, c1)
            t2 = self._encode_view_target(v2, m2, c2)

        loss12 = self.loss_fn(p1, t2)["loss"]
        loss21 = self.loss_fn(p2, t1)["loss"]
        loss = 0.5 * (loss12 + loss21)

        loss_dict = {
            "loss": loss,
            "loss12": loss12.detach(),
            "loss21": loss21.detach(),
        }
        B = self._batch_size(v1)
        self._log_z_dim_std([z1, z2], batch_size=B)
        self._log_z_centroid_metrics(z1, z2, batch_size=B)
        self._log_aug_metrics(v1, v2, batch, batch_size=B)
        if self._should_log_token_overlap():
            self._log_aug_token_overlap(c1, c2, batch_size=B)
        self._log_losses(loss_dict, prefix="train", batch_size=B)
        self.log(
            "train/ema_momentum",
            self.target_network.current_momentum,
            batch_size=B,
        )
        return loss

    def _step_lejepa(self, v1, m1, c1, v2, m2, c2, batch):
        """LeJEPA-2C step. Symmetric, no EMA, no predictor."""
        e1 = self._encode_view(v1, m1, c1)
        e2 = self._encode_view(v2, m2, c2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)

        B = self._batch_size(v1)
        self._log_z_dim_std([z1, z2], batch_size=B)
        self._log_z_centroid_metrics(z1, z2, batch_size=B)
        self._log_aug_metrics(v1, v2, batch, batch_size=B)
        if self._should_log_token_overlap():
            self._log_aug_token_overlap(c1, c2, batch_size=B)

        loss_dict = self.loss_fn(z1, z2, global_step=int(self.trainer.global_step))
        self._log_losses(loss_dict, prefix="train", batch_size=B)
        return loss_dict["loss"]

    def _step_lejepa_mc(self, batch: dict) -> torch.Tensor:
        """LeJEPA-MC step.

        Encoding strategy: process each view's full batch through the
        aggregator in one forward pass (one .forward per view). Total
        forwards = V_g + V_l. We avoid stacking views into [V*B, N, D]
        because global and local views have different N — there is no
        homogeneous way to flatten the view axis into the batch axis.
        """
        global_views = batch["global_views"]  # [V_g, B, N_g, D]
        local_views = batch["local_views"]  # [V_l, B, N_l, D]
        global_masks = batch.get("global_masks")
        local_masks = batch.get("local_masks")
        global_coords = batch.get("global_coords")
        local_coords = batch.get("local_coords")

        V_g = int(batch["num_global"])
        V_l = int(batch["num_local"])

        # Trust the batch's num_global / num_local (set by the augmentor at
        # collation time). Duplicating the values under ssl_cfg would just
        # be a second source of truth that can drift. The loss itself
        # validates that 1 <= num_global <= V.
        if V_g < 1:
            raise RuntimeError("LeJEPA-MC requires at least one global view.")

        z_views: list[torch.Tensor] = []

        # Encode global views.
        for v in range(V_g):
            view = global_views[v]
            mask = global_masks[v] if global_masks is not None else None
            coords = global_coords[v] if global_coords is not None else None
            e = self._encode_view(view, mask, coords)
            z_views.append(self.projector(e))

        # Encode local views.
        for v in range(V_l):
            view = local_views[v]
            mask = local_masks[v] if local_masks is not None else None
            coords = local_coords[v] if local_coords is not None else None
            e = self._encode_view(view, mask, coords)
            z_views.append(self.projector(e))

        loss_dict = self.loss_fn(
            z_views,
            num_global=V_g,
            global_step=int(self.trainer.global_step),
        )

        # Use B (per-rank batch size on view axis) as the batch_size hint
        # so Lightning's logger aggregation is correct.
        B = int(global_views.shape[1])
        self._log_z_dim_std(z_views, batch_size=B)
        self._log_losses(loss_dict, prefix="train", batch_size=B)

        return loss_dict["loss"]

    # ── EMA update (after optimizer step) ─────────────────────────────────

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA target network after each training step.

        NOTE: this hook fires once per dataloader batch, NOT once per
        optimizer step. With ``accumulate_grad_batches=1`` (the default for
        these experiments) the two are equivalent; with gradient
        accumulation > 1, the EMA would drift faster than the optimizer
        sees updates, which is not what we want. Keep
        ``accumulate_grad_batches: 1`` for JEPA runs.
        """
        if self.ssl_method == "jepa":
            self.target_network.update(self._online_for_ema)

    # ── Validation step ───────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Compute val loss + collect embeddings for RankMe / alpha-ReQ.

        Contract (do not break — callbacks rely on this dict):
          * 'loss'        : scalar val loss, used by ModelCheckpoint.
          * 'embeddings'  : aggregator-output embeddings (NOT projector
                            output), shape [B, embed_dim], detached.
                            For multi-crop, uses the FIRST GLOBAL VIEW's
                            embeddings to keep the metric definition stable
                            across methods (always one slide embedding per
                            sample).
        """
        if self.ssl_method == "lejepa_mc":
            return self._validation_lejepa_mc(batch)

        view1 = batch["view1"]
        view2 = batch["view2"]
        mask1 = batch.get("mask1")
        mask2 = batch.get("mask2")
        coords1 = batch.get("coords1")
        coords2 = batch.get("coords2")

        e1 = self._encode_view(view1, mask1, coords1)
        e2 = self._encode_view(view2, mask2, coords2)

        if self.ssl_method == "vicreg":
            z1, z2 = self.projector(e1), self.projector(e2)
            loss_dict = self.loss_fn(z1, z2)
        elif self.ssl_method == "jepa":
            # Match the symmetric training step: each view predicts the
            # other view's EMA target.
            z1 = self.projector(e1)
            z2 = self.projector(e2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            with torch.no_grad():
                t1 = self._encode_view_target(view1, mask1, coords1)
                t2 = self._encode_view_target(view2, mask2, coords2)
            loss12 = self.loss_fn(p1, t2)["loss"]
            loss21 = self.loss_fn(p2, t1)["loss"]
            loss_dict = {"loss": 0.5 * (loss12 + loss21)}
        elif self.ssl_method == "lejepa":
            z1, z2 = self.projector(e1), self.projector(e2)
            loss_dict = self.loss_fn(
                z1,
                z2,
                global_step=int(self.trainer.global_step),
            )
        else:
            raise RuntimeError(f"validation_step: unsupported method {self.ssl_method}")

        val_loss = loss_dict["loss"]
        B = self._batch_size(view1)
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True, batch_size=B)
        return {"loss": val_loss, "embeddings": e1.detach()}

    def _validation_lejepa_mc(self, batch: dict) -> dict:
        global_views = batch["global_views"]
        local_views = batch["local_views"]
        global_masks = batch.get("global_masks")
        local_masks = batch.get("local_masks")
        global_coords = batch.get("global_coords")
        local_coords = batch.get("local_coords")

        V_g = int(batch["num_global"])
        V_l = int(batch["num_local"])

        z_views: list[torch.Tensor] = []
        first_global_embedding: torch.Tensor | None = None

        for v in range(V_g):
            view = global_views[v]
            mask = global_masks[v] if global_masks is not None else None
            coords = global_coords[v] if global_coords is not None else None
            e = self._encode_view(view, mask, coords)
            if v == 0:
                first_global_embedding = e
            z_views.append(self.projector(e))

        for v in range(V_l):
            view = local_views[v]
            mask = local_masks[v] if local_masks is not None else None
            coords = local_coords[v] if local_coords is not None else None
            e = self._encode_view(view, mask, coords)
            z_views.append(self.projector(e))

        loss_dict = self.loss_fn(
            z_views,
            num_global=V_g,
            global_step=int(self.trainer.global_step),
        )

        val_loss = loss_dict["loss"]
        B = int(global_views.shape[1])
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True, batch_size=B)

        # Embeddings for RankMe / alpha-ReQ: the first global view's
        # aggregator output. This keeps the metric definition stable across
        # methods (always the slide embedding from one global view).
        return {"loss": val_loss, "embeddings": first_global_embedding.detach()}

    # ── Optimizer ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Exclude EMA parameters (they have requires_grad=False).
        # Separate params that should not get weight decay (e.g. cls_token).
        no_wd = set()
        if hasattr(self.aggregator, "no_weight_decay"):
            no_wd = {f"aggregator.{n}" for n in self.aggregator.no_weight_decay()}
        decay_params = []
        no_decay_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name in no_wd or p.ndim <= 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=tuple(self.hparams.adam_betas),
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

    def _log_losses(
        self, loss_dict: dict, prefix: str = "train", batch_size: int | None = None
    ) -> None:
        # LeJEPA{,-MC}Loss attaches an `_diagnostics` sub-dict for ssl/* logging.
        # Pop it before iterating so it doesn't get logged as `train/_diagnostics`.
        diagnostics = loss_dict.pop("_diagnostics", None)
        for k, v in loss_dict.items():
            # Only torch tensors / numbers go to the logger; ints like
            # "num_global" produced by LeJEPAMCLoss are skipped.
            if not isinstance(v, (torch.Tensor, int, float)):
                continue
            if isinstance(v, int):
                # Skip pure-int diagnostics that aren't useful as time series.
                continue
            prog_bar = k == "loss"
            self.log(f"{prefix}/{k}", v, prog_bar=prog_bar, sync_dist=True, batch_size=batch_size)
        if diagnostics:
            for k, v in diagnostics.items():
                if not isinstance(v, torch.Tensor):
                    continue
                self.log(f"ssl/{k}", v, sync_dist=True, batch_size=batch_size)

    def _log_z_dim_std(
        self,
        z_list,
        batch_size: int | None = None,
    ) -> None:
        """Log per-dimension std stats of the projected embeddings.

        Computed across all provided ``z`` tensors concatenated on the batch
        axis. This is a shared SSL-quality signal — for VICReg, JEPA, LeJEPA-2C,
        and LeJEPA-MC alike — measuring how much of the projector output's
        capacity is in use at each step. ``ssl/z_dim_std_min`` collapsing
        toward zero is a leading indicator of dimensional collapse.
        """
        # Concatenate detached, fp32 copies. Computing in fp32 keeps the std
        # numerically stable when the projector runs in bf16/fp16.
        zs = [z.detach() for z in z_list]
        if not zs:
            return
        z = torch.cat([zi.float() for zi in zs], dim=0)
        if z.shape[0] < 2:
            return
        dim_std = z.std(dim=0, correction=0)
        self.log("ssl/z_dim_std_mean", dim_std.mean(), sync_dist=True, batch_size=batch_size)
        self.log("ssl/z_dim_std_min", dim_std.min(), sync_dist=True, batch_size=batch_size)
        self.log("ssl/z_dim_std_max", dim_std.max(), sync_dist=True, batch_size=batch_size)

    # Per-view aug stats forwarded by stack_collate. "Attempted" = what the
    # augmentor sampled; "realized" = what the final fixed-N view became
    # after refill. Tracking both answers the question "are the two views
    # meaningfully different after split/crop/mask/refill?".
    _AUG_PER_VIEW_KEYS = (
        "crop_area_frac",  # attempted: sampled crop area
        "mask_ratio",  # attempted: sampled mask ratio
        "mask_fraction",  # realized: tokens kept from post-mask pool
        "crop_or_better_fraction",  # realized: (mask + crop) / fixed_n
        "view_or_better_fraction",  # realized: (mask + crop + split) / fixed_n
        "full_refill_fraction",  # realized: tokens pulled from full slide
        "replacement_fraction",  # realized: oversampled with replacement
    )

    # Token overlap uses a per-sample Python loop (torch.unique + torch.isin)
    # so it's the only non-vectorized aug metric. Gate it to every N steps —
    # 50 is cheap enough for short diagnostic runs and adds negligible
    # overhead at full-run scale.
    _AUG_TOKEN_OVERLAP_LOG_INTERVAL: int = 50

    def _log_aug_metrics(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        batch: dict,
        batch_size: int | None = None,
    ) -> None:
        """Log per-step dual-view augmentation diagnostics under ``aug/*``.

        Three cheap groups (all vectorized; safe to log every step):

        1. Per-view aug stats batch-means — what was attempted vs. what the
           final fixed-N view actually became after refill.
        2. No-op indicators — fraction of samples in the batch where crop
           was a no-op or full-refill was triggered. Means alone can hide
           failures (e.g. crop_area_frac mean = 0.7 looks healthy but might
           be 50% of samples no-op'd at 1.0 mixed with the rest in a tight
           range). These give an interpretable failure signal.
        3. Input-space centroid similarity (cosine, l2, mse) — for LeJEPA-2C
           in particular, if the mean UNI2-h feature of the two views is
           already nearly identical the centroid task is saturated before
           learning starts.

        Token overlap (the one expensive metric, batch-level Python loop)
        is split out into ``_log_aug_token_overlap`` so the training step
        can rate-limit it independently.

        Inputs are detached; no gradient flows through diagnostics.
        """
        # 1. Per-view aug stats: just batch-mean the [B] tensors collated
        #    from the augmentor. Quietly skip when stats weren't surfaced.
        for key in self._AUG_PER_VIEW_KEYS:
            v1_t = batch.get(f"{key}1")
            v2_t = batch.get(f"{key}2")
            if v1_t is None or v2_t is None:
                continue
            self.log(f"aug/{key}1_mean", v1_t.float().mean(), sync_dist=True, batch_size=batch_size)
            self.log(f"aug/{key}2_mean", v2_t.float().mean(), sync_dist=True, batch_size=batch_size)

        # 2. No-op indicators. Use the same surfaced [B] tensors; threshold
        #    >=0.999 (not ==1.0) to absorb floating-point round-trips
        #    through the dataset/collator boundary.
        caf1 = batch.get("crop_area_frac1")
        caf2 = batch.get("crop_area_frac2")
        if caf1 is not None and caf2 is not None:
            self.log(
                "aug/crop_noop1_frac",
                (caf1 >= 0.999).float().mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "aug/crop_noop2_frac",
                (caf2 >= 0.999).float().mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
        frf1 = batch.get("full_refill_fraction1")
        frf2 = batch.get("full_refill_fraction2")
        if frf1 is not None and frf2 is not None:
            self.log(
                "aug/full_refill_any1_frac",
                (frf1 > 0).float().mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "aug/full_refill_any2_frac",
                (frf2 > 0).float().mean(),
                sync_dist=True,
                batch_size=batch_size,
            )

        # 3. Input-space centroid similarity (per-sample → batch-mean).
        v1d = v1.detach().float()
        v2d = v2.detach().float()
        mu1 = v1d.mean(dim=1)  # [B, D]
        mu2 = v2d.mean(dim=1)  # [B, D]
        diff = mu1 - mu2  # [B, D]
        cos = F.cosine_similarity(mu1, mu2, dim=1).mean()
        l2 = diff.norm(p=2, dim=1).mean()
        mse = diff.pow(2).mean()
        self.log("aug/input_centroid_cosine", cos, sync_dist=True, batch_size=batch_size)
        self.log("aug/input_centroid_l2", l2, sync_dist=True, batch_size=batch_size)
        self.log("aug/input_centroid_mse", mse, sync_dist=True, batch_size=batch_size)

    def _log_aug_token_overlap(
        self,
        c1: torch.Tensor | None,
        c2: torch.Tensor | None,
        batch_size: int | None = None,
    ) -> None:
        """Log ``aug/token_overlap_frac`` and ``aug/token_jaccard``.

        Coords are token identities under the shared D4 transform, so equal
        coord pairs identify the same source token. We hash to int64 and
        intersect uniques per row. Per-row work is vectorized but the outer
        Python loop is over B (typically <= 64) — call this on a coarse
        cadence (see ``_AUG_TOKEN_OVERLAP_LOG_INTERVAL``).

        Both fractions use unique-token sets so replacement-refill duplicates
        don't inflate counts. ``overlap_frac`` is asymmetric (|A∩B|/|A|);
        ``jaccard`` is symmetric (|A∩B|/|AUB|).
        """
        if c1 is None or c2 is None:
            return
        c1l = c1.detach().long()
        c2l = c2.detach().long()
        K = int(max(c1l.max().item(), c2l.max().item())) + 1
        h1 = c1l[..., 0] * K + c1l[..., 1]  # [B, N]
        h2 = c2l[..., 0] * K + c2l[..., 1]  # [B, N]
        B = h1.shape[0]
        overlap = h1.new_zeros(B, dtype=torch.float32)
        jaccard = h1.new_zeros(B, dtype=torch.float32)
        for b in range(B):
            u1 = torch.unique(h1[b])
            u2 = torch.unique(h2[b])
            inter = torch.isin(u1, u2).sum().float()
            union = float(u1.numel()) + float(u2.numel()) - float(inter)
            overlap[b] = inter / float(u1.numel())
            jaccard[b] = inter / max(union, 1.0)
        self.log("aug/token_overlap_frac", overlap.mean(), sync_dist=True, batch_size=batch_size)
        self.log("aug/token_jaccard", jaccard.mean(), sync_dist=True, batch_size=batch_size)

    def _should_log_token_overlap(self) -> bool:
        """True on every Nth global step. Defensive against being called
        outside a Lightning trainer (e.g. unit tests) — falls back to True."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return True
        step = int(getattr(trainer, "global_step", 0))
        return step % self._AUG_TOKEN_OVERLAP_LOG_INTERVAL == 0

    def _log_z_centroid_metrics(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: int | None = None,
    ) -> None:
        """Log projected-space centroid similarity under ``ssl/z_centroid_*``.

        Same three measures as the input-side centroid (cosine, l2, mse) but
        on the projector output. Useful as a "post-projector" mirror of
        ``aug/input_centroid_*``: if input centroids are already similar AND
        z centroids are similar, the projector isn't doing useful work to
        differentiate the views.
        """
        z1d = z1.detach().float()
        z2d = z2.detach().float()
        diff = z1d - z2d
        cos = F.cosine_similarity(z1d, z2d, dim=1).mean()
        l2 = diff.norm(p=2, dim=1).mean()
        mse = diff.pow(2).mean()
        self.log("ssl/z_centroid_cosine", cos, sync_dist=True, batch_size=batch_size)
        self.log("ssl/z_centroid_l2", l2, sync_dist=True, batch_size=batch_size)
        self.log("ssl/z_centroid_mse", mse, sync_dist=True, batch_size=batch_size)

    # ── Checkpoint: save only aggregator weights ──────────────────────────

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Also save bare aggregator state_dict for easy downstream loading.

        The full checkpoint has the complete module (projector, predictor, etc.)
        but downstream fine-tuning only needs the aggregator.
        """
        checkpoint["aggregator_state_dict"] = self.aggregator.state_dict()
        checkpoint["ssl_method"] = self.ssl_method
        checkpoint["embed_dim"] = self.embed_dim
