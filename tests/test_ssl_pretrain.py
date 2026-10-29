"""
Tests for SSL pretraining modules (``oceanpath.ssl.modules``).

The pretraining module supports four methods: vicreg, jepa, lejepa (2-crop
LeJEPA) and lejepa_mc (multi-crop LeJEPA). SimCLR / BYOL / DINO were retired
together with their losses and module branches, so method-parametrized tests
run over the four current methods only.

Coverage:
  1. Losses       VICRegLoss, JEPALoss, sigreg, LeJEPALoss, LeJEPAMCLoss
  2. Heads        Projector, Predictor, DINOHead (still exported), EMANetwork
  3. Module       construction per method and from the shipped configs,
                  optimizer parameter groups, LR schedule
  4. Training     every step runs inside a real ``lightning.Trainer`` (CPU, tiny
                  dims): fit per method, logged metrics, gradient flow, EMA
                  target isolation, train/eval modes, validation output
                  contract, edge-case batches, numerical stability, checkpoint
                  round trip, and PretrainDataModule -> SSLPretrainModule end to
                  end on a memmap store built with the repo's builder
  5. Callbacks    RankMe, alpha-ReQ, SSLQualityCallback (unit and in a Trainer)
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import lightning as L
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from oceanpath.models import build_aggregator
from oceanpath.models.base import BaseMIL
from oceanpath.ssl.data.datamodule import (
    PretrainDataModule,
    multicrop_stack_collate,
    stack_collate,
)
from oceanpath.ssl.modules.callbacks import (
    AlphaReQCallback,
    RankMeCallback,
    SSLQualityCallback,
)
from oceanpath.ssl.modules.heads import (
    DINOHead,
    EMANetwork,
    Predictor,
    Projector,
)
from oceanpath.ssl.modules.losses import (
    JEPALoss,
    LeJEPALoss,
    LeJEPAMCLoss,
    VICRegLoss,
    sigreg,
)
from oceanpath.ssl.modules.pretrain_module import SSLPretrainModule

# ═══════════════════════════════════════════════════════════════════
# Constants and fixtures
# ═══════════════════════════════════════════════════════════════════

B = 8  # batch size
D_PROJ = 64  # projection dim
D_EMBED = 32  # aggregator embed dim
D_IN = 16  # patch feature dim
N_PATCHES = 20  # patches per view
N_LOCAL = 10  # patches per local view (lejepa_mc)
D_PROTO = 128  # DINOHead prototype dim

METHODS = ["vicreg", "jepa", "lejepa", "lejepa_mc"]
DUAL_VIEW_METHODS = ["vicreg", "jepa", "lejepa"]
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "pretrain_training"
MODEL_CFG = {
    "embed_dim": D_EMBED,
    "dropout": 0.0,
    "attn_dim": 16,
    "gate": True,
    "num_fc_layers": 1,
}


@pytest.fixture
def z_pair():
    """Random z1, z2 pair with gradient tracking."""
    torch.manual_seed(42)
    z1 = torch.randn(B, D_PROJ, requires_grad=True)
    z2 = torch.randn(B, D_PROJ, requires_grad=True)
    return z1, z2


@pytest.fixture
def z_pair_identical():
    """Identical z1, z2 pair for testing zero-loss conditions."""
    torch.manual_seed(42)
    z = torch.randn(B, D_PROJ)
    return z.clone().requires_grad_(True), z.clone().requires_grad_(True)


def _make_module(method, ssl_cfg=None, **overrides) -> SSLPretrainModule:
    """SSLPretrainModule with small dimensions; ``ssl_cfg`` entries override the defaults."""
    ssl = {
        "proj_hidden_dim": 64,
        "proj_out_dim": D_PROJ,
        "proj_num_layers": 2,
        "pred_hidden_dim": 32,
        "ema_momentum": 0.99,
        "ema_final_momentum": 1.0,
        "num_slices": 64,
    }
    ssl.update(ssl_cfg or {})
    params = {
        "ssl_method": method,
        "arch": "abmil",
        "in_dim": D_IN,
        "model_cfg": dict(MODEL_CFG),
        "ssl_cfg": ssl,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "warmup_epochs": 0,
        "max_epochs": 10,
        "lr_scheduler": "none",
    }
    params.update(overrides)
    return SSLPretrainModule(**params)


_AUG_STATS = {
    "crop_area_frac": 0.6,
    "mask_ratio": 0.2,
    "mask_fraction": 1.0,
    "crop_or_better_fraction": 1.0,
    "view_or_better_fraction": 1.0,
    "full_refill_fraction": 0.0,
    "replacement_fraction": 0.0,
}


def _dual_batch(b=B, n=N_PATCHES, *, seed=0, scale=1.0) -> dict:
    """Dual-view batch built with the real collator (coords + augmentation stats)."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(b):
        sample = {"slide_id": f"s{seed}_{i}"}
        for v in ("1", "2"):
            sample[f"view{v}_features"] = torch.randn(n, D_IN, generator=g) * scale
            sample[f"view{v}_coords"] = (
                torch.randint(0, 40, (n, 2), generator=g, dtype=torch.int32) * 256
            )
            sample[f"view{v}_length"] = n
            for key, value in _AUG_STATS.items():
                sample[f"view{v}_{key}"] = value
        samples.append(sample)
    return stack_collate(samples)


def _mc_batch(b=B, *, seed=0, scale=1.0, v_g=2, v_l=2) -> dict:
    """Multi-crop batch built with the real collator."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(b):
        samples.append(
            {
                "global_features": [
                    torch.randn(N_PATCHES, D_IN, generator=g) * scale for _ in range(v_g)
                ],
                "local_features": [
                    torch.randn(N_LOCAL, D_IN, generator=g) * scale for _ in range(v_l)
                ],
                "global_coords": [
                    torch.randint(0, 40, (N_PATCHES, 2), generator=g, dtype=torch.int32) * 256
                    for _ in range(v_g)
                ],
                "local_coords": [
                    torch.randint(0, 40, (N_LOCAL, 2), generator=g, dtype=torch.int32) * 256
                    for _ in range(v_l)
                ],
                "num_global": v_g,
                "num_local": v_l,
                "slide_id": f"s{seed}_{i}",
            }
        )
    return multicrop_stack_collate(samples)


def _batches(method, n_batches=2, *, first_seed=0, **kwargs) -> list[dict]:
    make = _mc_batch if method == "lejepa_mc" else _dual_batch
    return [make(seed=first_seed + i, **kwargs) for i in range(n_batches)]


def _trainer(**kwargs) -> L.Trainer:
    params = {
        "accelerator": "cpu",
        "devices": 1,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    params.update(kwargs)
    return L.Trainer(**params)


def _fit(module, train_batches, val_batches=None, *, callbacks=(), **trainer_kwargs):
    """One optimizer step per training batch; validation at the end of the epoch.

    The loaders yield the pre-collated batches as they are (batch_size=None).
    """
    trainer_kwargs.setdefault("max_steps", len(train_batches))
    trainer = _trainer(
        limit_val_batches=len(val_batches) if val_batches else 0,
        callbacks=list(callbacks),
        **trainer_kwargs,
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(train_batches, batch_size=None),
        val_dataloaders=DataLoader(val_batches, batch_size=None) if val_batches else None,
    )
    return trainer


def _validate(module, batches, *, callbacks=()):
    trainer = _trainer(callbacks=list(callbacks))
    trainer.validate(module, dataloaders=DataLoader(batches, batch_size=None), verbose=False)
    return trainer


class _Recorder(L.Callback):
    """Collects what the hooks see: module modes, step outputs, gradients."""

    def __init__(self):
        self.train_modes: list[bool] = []
        self.val_modes: list[bool] = []
        self.target_modes: list[bool] = []
        self.losses: list[float] = []
        self.val_outputs: list[dict] = []
        self.missing_grads: list[list[str]] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_modes.append(pl_module.training)
        if hasattr(pl_module, "target_network"):
            self.target_modes.append(pl_module.target_network.ema_model.training)

    def on_after_backward(self, trainer, pl_module):
        self.missing_grads.append(
            [
                name
                for name, p in pl_module.named_parameters()
                if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
            ]
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(float(outputs["loss"]))

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.val_modes.append(pl_module.training)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.val_outputs.append(outputs)


# ═══════════════════════════════════════════════════════════════════
# 1. VICReg loss
# ═══════════════════════════════════════════════════════════════════


class TestVICRegLoss:
    def test_output_keys(self, z_pair):
        out = VICRegLoss()(*z_pair)
        assert set(out.keys()) == {"loss", "inv_loss", "var_loss", "cov_loss"}

    def test_loss_is_scalar(self, z_pair):
        assert VICRegLoss()(*z_pair)["loss"].ndim == 0

    def test_loss_is_finite(self, z_pair):
        for v in VICRegLoss()(*z_pair).values():
            assert torch.isfinite(v)

    def test_invariance_zero_for_identical(self, z_pair_identical):
        out = VICRegLoss()(*z_pair_identical)
        assert out["inv_loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_invariance_positive_for_different(self, z_pair):
        out = VICRegLoss()(*z_pair)
        assert out["inv_loss"].item() > 0

    def test_variance_fires_for_collapsed_data(self):
        """All-constant embeddings should have std≈0, near-max hinge penalty.
        With eps=1e-4, std = sqrt(0 + eps) = 0.01, so hinge = relu(1-0.01) = 0.99 per view."""
        z = torch.ones(B, D_PROJ)
        out = VICRegLoss(var_target=1.0)(z, z.clone())
        assert out["var_loss"].item() == pytest.approx(2.0, abs=0.05)

    def test_variance_zero_when_spread_enough(self):
        """Embeddings with std >= var_target should have no variance penalty."""
        torch.manual_seed(42)
        z = torch.randn(B, D_PROJ) * 2.0  # std approx 2.0 >> 1.0
        out = VICRegLoss(var_target=1.0)(z, z.clone())
        assert out["var_loss"].item() == pytest.approx(0.0, abs=0.3)

    def test_covariance_low_for_diagonal(self):
        """Identity-like matrix has near-zero off-diagonal covariance.
        Note: eye(8, 64) is non-square, so the 8 occupied columns share
        mean=1/8, creating residual cross-correlations of -1/56 between them.
        cov_loss ≈ 2 * 56 * (1/56)^2 / 64 ≈ 5.6e-4."""
        z = torch.eye(B, D_PROJ)
        out = VICRegLoss()(z, z.clone())
        assert out["cov_loss"].item() < 1e-3

    def test_covariance_positive_for_correlated(self):
        """Highly correlated dims should produce high covariance loss."""
        torch.manual_seed(42)
        base = torch.randn(B, 1)
        z = base.expand(B, D_PROJ) + torch.randn(B, D_PROJ) * 0.01
        out = VICRegLoss()(z, z.clone())
        assert out["cov_loss"].item() > 0.1

    def test_weight_scaling(self, z_pair):
        """Doubling inv_weight should increase loss by exactly inv_loss * delta."""
        out_1x = VICRegLoss(inv_weight=25.0)(*z_pair)
        out_2x = VICRegLoss(inv_weight=50.0)(*z_pair)
        diff = out_2x["loss"].item() - out_1x["loss"].item()
        assert diff == pytest.approx(25.0 * out_1x["inv_loss"].item(), rel=0.01)

    def test_gradient_flows(self, z_pair):
        VICRegLoss()(*z_pair)["loss"].backward()
        assert z_pair[0].grad is not None
        assert torch.isfinite(z_pair[0].grad).all()
        assert z_pair[1].grad is not None
        assert torch.isfinite(z_pair[1].grad).all()

    def test_sub_losses_detached(self, z_pair):
        out = VICRegLoss()(*z_pair)
        for k in ["inv_loss", "var_loss", "cov_loss"]:
            assert not out[k].requires_grad

    def test_batch_size_one(self):
        """B=1 should work — variance uses correction=0 (population variance)."""
        z1 = torch.randn(1, D_PROJ, requires_grad=True)
        z2 = torch.randn(1, D_PROJ, requires_grad=True)
        out = VICRegLoss()(z1, z2)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert z1.grad is not None

    def test_all_weights_zero_gives_zero_loss(self, z_pair):
        out = VICRegLoss(inv_weight=0, var_weight=0, cov_weight=0)(*z_pair)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-8)

    def test_large_dim(self):
        """Check no numerical issues with larger embedding dim."""
        z1 = torch.randn(4, 512, requires_grad=True)
        z2 = torch.randn(4, 512, requires_grad=True)
        out = VICRegLoss()(z1, z2)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()


# ═══════════════════════════════════════════════════════════════════
# 2. JEPA loss
# ═══════════════════════════════════════════════════════════════════


class TestJEPALoss:
    def test_smooth_l1(self, z_pair):
        assert torch.isfinite(JEPALoss(loss_type="smooth_l1")(*z_pair)["loss"])

    def test_mse(self, z_pair):
        assert torch.isfinite(JEPALoss(loss_type="mse")(*z_pair)["loss"])

    def test_unknown_raises(self, z_pair):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            JEPALoss(loss_type="bad")(*z_pair)

    def test_zero_for_perfect_mse(self):
        z = torch.randn(B, D_PROJ)
        assert JEPALoss(loss_type="mse")(z, z.clone())["loss"].item() == pytest.approx(
            0.0, abs=1e-6
        )

    def test_zero_for_perfect_smooth_l1(self):
        z = torch.randn(B, D_PROJ)
        assert JEPALoss(loss_type="smooth_l1")(z, z.clone())["loss"].item() == pytest.approx(
            0.0, abs=1e-6
        )

    def test_target_detached(self):
        pred = torch.randn(B, D_PROJ, requires_grad=True)
        tgt = torch.randn(B, D_PROJ, requires_grad=True)
        JEPALoss()(pred, tgt)["loss"].backward()
        assert pred.grad is not None
        assert tgt.grad is None

    def test_huber_robust_to_outliers(self):
        """Smooth L1 grows linearly for large errors, MSE grows quadratically."""
        torch.manual_seed(42)
        pred = torch.randn(B, D_PROJ)
        tgt = pred + 10.0
        l_h = JEPALoss(loss_type="smooth_l1", beta=2.0)(pred, tgt)["loss"]
        l_m = JEPALoss(loss_type="mse")(pred, tgt)["loss"]
        assert l_m.item() > l_h.item()

    def test_gradient_flows_both_types(self):
        for lt in ["smooth_l1", "mse"]:
            pred = torch.randn(B, D_PROJ, requires_grad=True)
            tgt = torch.randn(B, D_PROJ)
            JEPALoss(loss_type=lt)(pred, tgt)["loss"].backward()
            assert pred.grad is not None
            assert torch.isfinite(pred.grad).all()

    def test_mse_matches_manual(self):
        """JEPA MSE should match manual MSE computation."""
        torch.manual_seed(42)
        pred = torch.randn(B, D_PROJ)
        tgt = torch.randn(B, D_PROJ)
        loss = JEPALoss(loss_type="mse")(pred, tgt)["loss"]
        manual = ((pred - tgt) ** 2).mean()
        assert loss.item() == pytest.approx(manual.item(), abs=1e-5)


# ═══════════════════════════════════════════════════════════════════
# 3. SIGReg, LeJEPA and LeJEPA-MC losses
# ═══════════════════════════════════════════════════════════════════


class TestSIGReg:
    def test_scalar_finite_and_non_negative(self):
        torch.manual_seed(0)
        value = sigreg(torch.randn(64, 16), global_step=0, num_slices=32)
        assert value.ndim == 0
        assert torch.isfinite(value)
        assert value.item() >= 0

    def test_directions_are_seeded_by_global_step(self):
        """Same step -> same projections (shared across DDP ranks); new step -> new ones."""
        torch.manual_seed(0)
        x = torch.randn(64, 16)
        assert sigreg(x, 3, num_slices=32).item() == sigreg(x, 3, num_slices=32).item()
        assert sigreg(x, 3, num_slices=32).item() != sigreg(x, 4, num_slices=32).item()

    def test_isotropic_gaussian_scores_lowest(self):
        torch.manual_seed(0)
        gaussian = torch.randn(512, 16)
        shifted = gaussian + 3.0
        collapsed = torch.randn(512, 1).expand(512, 16).contiguous()
        scores = {
            name: sigreg(x, 0, num_slices=128).item()
            for name, x in {
                "gaussian": gaussian,
                "shifted": shifted,
                "collapsed": collapsed,
            }.items()
        }
        assert scores["gaussian"] < scores["shifted"]
        assert scores["gaussian"] < scores["collapsed"]

    def test_gradient_finite_for_extreme_inputs(self):
        for scale in (1e-6, 1e3):
            x = (torch.randn(32, 16) * scale).requires_grad_(True)
            sigreg(x, 0, num_slices=32).backward()
            assert torch.isfinite(x.grad).all()


class TestLeJEPALoss:
    def test_output_keys_in_eval(self, z_pair):
        out = LeJEPALoss(num_slices=32).eval()(*z_pair)
        assert set(out) == {"loss", "pred_loss", "sigreg_loss"}
        assert out["loss"].ndim == 0

    def test_training_with_grad_adds_diagnostics(self, z_pair):
        out = LeJEPALoss(num_slices=32).train()(*z_pair)
        assert set(out["_diagnostics"]) == {
            "z_norm_mean",
            "z_norm_std",
            "weighted_sigreg_loss",
            "loss_ratio_sigreg_to_pred",
            "grad_norm_pred",
            "grad_norm_sigreg",
            "grad_ratio_sigreg_to_pred",
        }
        assert all(torch.isfinite(v) for v in out["_diagnostics"].values())

    def test_no_diagnostics_without_grad(self):
        z1, z2 = torch.randn(B, D_PROJ), torch.randn(B, D_PROJ)
        assert "_diagnostics" not in LeJEPALoss(num_slices=32).train()(z1, z2)

    def test_pred_loss_is_the_centroid_residual(self, z_pair):
        z1, z2 = z_pair
        out = LeJEPALoss(num_slices=32).eval()(z1, z2)
        # Both views sit (z1 - z2) / 2 away from their midpoint.
        expected = ((z1 - z2) / 2).pow(2).mean()
        assert out["pred_loss"].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_pred_loss_zero_for_identical_views(self, z_pair_identical):
        out = LeJEPALoss(num_slices=32).eval()(*z_pair_identical)
        assert out["pred_loss"].item() == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize("weight", [0.0, 0.05, 1.0])
    def test_loss_mixes_the_two_terms(self, z_pair, weight):
        out = LeJEPALoss(sigreg_weight=weight, num_slices=32).eval()(*z_pair, global_step=5)
        expected = (1 - weight) * out["pred_loss"] + weight * out["sigreg_loss"]
        assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_sigreg_term_averages_both_views(self, z_pair):
        z1, z2 = z_pair
        out = LeJEPALoss(num_slices=32).eval()(z1, z2, global_step=2)
        expected = 0.5 * (sigreg(z1, 2, num_slices=32) + sigreg(z2, 2, num_slices=32))
        assert out["sigreg_loss"].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_gradient_flows_to_both_views(self, z_pair):
        LeJEPALoss(num_slices=32)(*z_pair)["loss"].backward()
        for z in z_pair:
            assert z.grad is not None
            assert torch.isfinite(z.grad).all()


class TestLeJEPAMCLoss:
    def test_two_global_views_reduce_to_lejepa(self, z_pair):
        mc = LeJEPAMCLoss(num_slices=32).eval()(list(z_pair), num_global=2, global_step=1)
        two_crop = LeJEPALoss(num_slices=32).eval()(*z_pair, global_step=1)
        for key in ("loss", "pred_loss", "sigreg_loss"):
            assert mc[key].item() == pytest.approx(two_crop[key].item(), rel=1e-5)

    def test_list_and_stacked_inputs_agree(self):
        torch.manual_seed(0)
        views = [torch.randn(B, D_PROJ) for _ in range(4)]
        loss = LeJEPAMCLoss(num_slices=32).eval()
        a = loss(views, num_global=2)
        b = loss(torch.stack(views), num_global=2)
        assert a["loss"].item() == pytest.approx(b["loss"].item(), rel=1e-6)

    def test_centroid_comes_from_global_views_only(self):
        torch.manual_seed(0)
        g1, g2, local = (torch.randn(B, D_PROJ) for _ in range(3))
        loss = LeJEPAMCLoss(num_slices=32).eval()
        a = loss([g1, g2, local], num_global=2)
        b = loss([g1, g2, local * 5.0], num_global=2)
        assert a["pred_loss_global_mean"].item() == pytest.approx(
            b["pred_loss_global_mean"].item(), rel=1e-6
        )
        assert b["pred_loss_local_mean"].item() > a["pred_loss_local_mean"].item()
        assert (a["num_global"], a["num_local"]) == (2, 1)

    def test_without_local_views_the_local_term_is_zero(self, z_pair):
        out = LeJEPAMCLoss(num_slices=32).eval()(list(z_pair), num_global=2)
        assert out["pred_loss_local_mean"].item() == 0.0
        assert out["num_local"] == 0

    def test_training_with_grad_adds_diagnostics(self, z_pair):
        out = LeJEPAMCLoss(num_slices=32).train()(list(z_pair), num_global=1)
        assert "grad_ratio_sigreg_to_pred" in out["_diagnostics"]

    @pytest.mark.parametrize("num_global", [0, 3])
    def test_num_global_must_fit_the_views(self, z_pair, num_global):
        with pytest.raises(ValueError, match="num_global"):
            LeJEPAMCLoss(num_slices=32)(list(z_pair), num_global=num_global)

    def test_bad_inputs_rejected(self):
        loss = LeJEPAMCLoss(num_slices=32)
        with pytest.raises(ValueError, match="empty"):
            loss([], num_global=1)
        with pytest.raises(ValueError, match=r"\[V, B, D\]"):
            loss(torch.randn(B, D_PROJ), num_global=1)
        with pytest.raises(ValueError, match="sigreg_weight"):
            LeJEPAMCLoss(sigreg_weight=1.5)


# ═══════════════════════════════════════════════════════════════════
# 4. Heads: Projector, Predictor, DINOHead
# ═══════════════════════════════════════════════════════════════════


class TestProjector:
    def test_shape(self):
        out = Projector(D_EMBED, 64, D_PROJ, 3)(torch.randn(B, D_EMBED))
        assert out.shape == (B, D_PROJ)

    def test_two_layer(self):
        out = Projector(D_EMBED, 64, D_PROJ, 2)(torch.randn(B, D_EMBED))
        assert out.shape == (B, D_PROJ)

    def test_bn_present(self):
        p = Projector(D_EMBED, 64, D_PROJ, use_bn=True)
        assert any(isinstance(m, nn.BatchNorm1d) for m in p.mlp.modules())

    def test_bn_absent(self):
        p = Projector(D_EMBED, 64, D_PROJ, use_bn=False)
        assert not any(isinstance(m, nn.BatchNorm1d) for m in p.mlp.modules())

    def test_last_bn(self):
        p = Projector(D_EMBED, 64, D_PROJ, use_bn=True, last_bn=True)
        bn_count = sum(1 for m in p.mlp.modules() if isinstance(m, nn.BatchNorm1d))
        p_no = Projector(D_EMBED, 64, D_PROJ, use_bn=True, last_bn=False)
        bn_no = sum(1 for m in p_no.mlp.modules() if isinstance(m, nn.BatchNorm1d))
        assert bn_count == bn_no + 1

    def test_gradient(self):
        p = Projector(D_EMBED, 64, D_PROJ)
        p.train()
        p(torch.randn(B, D_EMBED)).sum().backward()
        assert all(q.grad is not None for q in p.parameters() if q.requires_grad)

    def test_eval_mode(self):
        """Projector should work in eval mode (BN uses running stats)."""
        p = Projector(D_EMBED, 64, D_PROJ, use_bn=True)
        p.train()
        p(torch.randn(B, D_EMBED))  # populate running stats
        p.eval()
        out = p(torch.randn(B, D_EMBED))
        assert out.shape == (B, D_PROJ)

    def test_deterministic_in_eval(self):
        """Eval mode should be deterministic."""
        p = Projector(D_EMBED, 64, D_PROJ, use_bn=True)
        p.train()
        p(torch.randn(B, D_EMBED))
        p.eval()
        x = torch.randn(B, D_EMBED)
        assert torch.allclose(p(x), p(x))


class TestPredictor:
    def test_shape(self):
        assert Predictor(D_PROJ, 32, D_PROJ)(torch.randn(B, D_PROJ)).shape == (B, D_PROJ)

    def test_bottleneck_smaller(self):
        """Bottleneck predictor should have fewer params than a full-rank linear."""
        pred = Predictor(256, 64, 256)
        assert sum(p.numel() for p in pred.parameters()) < 256 * 256 + 256

    def test_bn_toggle(self):
        assert (
            sum(
                1
                for m in Predictor(D_PROJ, 32, D_PROJ, True).modules()
                if isinstance(m, nn.BatchNorm1d)
            )
            == 1
        )
        assert (
            sum(
                1
                for m in Predictor(D_PROJ, 32, D_PROJ, False).modules()
                if isinstance(m, nn.BatchNorm1d)
            )
            == 0
        )

    def test_gradient(self):
        pred = Predictor(D_PROJ, 32, D_PROJ)
        pred.train()
        pred(torch.randn(B, D_PROJ)).sum().backward()
        assert all(q.grad is not None for q in pred.parameters() if q.requires_grad)

    def test_different_in_out(self):
        """Predictor with in_dim != out_dim."""
        pred = Predictor(64, 32, 128)
        assert pred(torch.randn(B, 64)).shape == (B, 128)


class TestDINOHead:
    def test_shape(self):
        out = DINOHead(D_EMBED, 64, 32, D_PROTO)(torch.randn(B, D_EMBED))
        assert out.shape == (B, D_PROTO)

    def test_weight_norm_parametrization(self):
        """Last layer uses modern parametrizations.weight_norm (not deprecated utils.weight_norm)."""
        h = DINOHead(D_EMBED, 64, 32, D_PROTO)
        # Modern API stores parametrizations as modules
        assert hasattr(h.last_layer, "parametrizations")
        assert "weight" in h.last_layer.parametrizations
        # No legacy weight_norm artifacts
        assert not hasattr(h.last_layer, "weight_g")
        assert not hasattr(h.last_layer, "weight_v")

    def test_no_bias(self):
        assert DINOHead(D_EMBED, 64, 32, D_PROTO).last_layer.bias is None

    def test_weight_norm_decomposition(self):
        """Parametrized weight = g * (v / ||v||): direction-magnitude decoupled."""
        h = DINOHead(D_EMBED, 64, 32, D_PROTO)
        # Access the underlying original0 (g) and original1 (v)
        g = h.last_layer.parametrizations.weight.original0  # (out_dim, 1)
        v = h.last_layer.parametrizations.weight.original1  # (out_dim, in_dim)
        # g should be initialized to the original row norms
        v_norms = v.norm(dim=1, keepdim=True)
        assert torch.allclose(g, v_norms, atol=1e-5)
        # The composed weight should equal g * v/||v||
        w = h.last_layer.weight  # triggers parametrization
        expected = g * (v / v_norms)
        assert torch.allclose(w, expected, atol=1e-6)

    def test_l2_norm_intermediate(self):
        """Intermediate output before last layer should be L2-normalized."""
        h = DINOHead(D_EMBED, 64, 32, D_PROTO)
        x = torch.randn(B, D_EMBED)
        mlp_out = h.mlp(x)
        normed = nn.functional.normalize(mlp_out, dim=-1, p=2)
        assert torch.allclose(normed.norm(dim=-1), torch.ones(B), atol=1e-5)

    def test_gradient(self):
        h = DINOHead(D_EMBED, 64, 32, D_PROTO)
        h.train()
        h(torch.randn(B, D_EMBED)).sum().backward()
        # Modern parametrizations store originals; gradients flow through them
        has_grad = any(p.grad is not None for p in h.last_layer.parameters())
        assert has_grad

    def test_output_is_logits_not_probs(self):
        """Output should be raw logits, not probabilities (no softmax)."""
        h = DINOHead(D_EMBED, 64, 32, D_PROTO)
        out = h(torch.randn(B, D_EMBED))
        # Logits can be negative and don't sum to 1
        assert out.min().item() < 0 or out.sum(dim=-1).mean().item() != pytest.approx(1.0, abs=0.1)

    def test_deepcopy_safe(self):
        """DINOHead can be deepcopied (required for EMANetwork)."""
        h = DINOHead(D_EMBED, 64, 32, D_PROTO)
        h_copy = copy.deepcopy(h)
        x = torch.randn(B, D_EMBED)
        assert torch.allclose(h(x), h_copy(x))


# ═══════════════════════════════════════════════════════════════════
# 5. EMA network
# ═══════════════════════════════════════════════════════════════════


class TestEMANetwork:
    @pytest.fixture
    def source(self):
        torch.manual_seed(42)
        return nn.Sequential(nn.Linear(D_EMBED, D_PROJ), nn.ReLU(), nn.Linear(D_PROJ, D_PROJ))

    def test_initial_copy_exact(self, source):
        ema = EMANetwork(source, 0.99)
        for s, e in zip(source.parameters(), ema.ema_model.parameters(), strict=True):
            assert torch.allclose(s, e)

    def test_ema_frozen(self, source):
        for p in EMANetwork(source).ema_model.parameters():
            assert not p.requires_grad

    def test_update_changes_ema(self, source):
        ema = EMANetwork(source, initial_momentum=0.5)
        before = [p.clone() for p in ema.ema_model.parameters()]
        with torch.no_grad():
            for p in source.parameters():
                p.add_(torch.randn_like(p) * 10)
        ema.update(source)
        assert any(
            not torch.allclose(b, a)
            for b, a in zip(before, ema.ema_model.parameters(), strict=True)
        )

    def test_update_formula(self, source):
        """Verify theta_ema = m * theta_ema + (1-m) * theta_source exactly."""
        m = 0.8
        ema = EMANetwork(source, initial_momentum=m, final_momentum=m, total_steps=100)
        before = [p.clone() for p in ema.ema_model.parameters()]
        with torch.no_grad():
            for p in source.parameters():
                p.fill_(1.0)
        ema.update(source)
        for b, e, s in zip(before, ema.ema_model.parameters(), source.parameters(), strict=True):
            assert torch.allclose(e, m * b + (1 - m) * s, atol=1e-6)

    def test_momentum_starts_at_initial(self, source):
        ema = EMANetwork(source, 0.99, 1.0, 100)
        assert ema.current_momentum == pytest.approx(0.99, abs=1e-6)

    def test_momentum_increases(self, source):
        ema = EMANetwork(source, 0.99, 1.0, 10)
        ms = []
        for _ in range(10):
            ms.append(ema.current_momentum)
            ema.update(source)
        for i in range(1, len(ms)):
            assert ms[i] >= ms[i - 1] - 1e-6

    def test_momentum_reaches_final(self, source):
        ema = EMANetwork(source, 0.99, 1.0, 10)
        for _ in range(20):
            ema.update(source)
        assert ema.current_momentum == pytest.approx(1.0, abs=1e-4)

    def test_momentum_at_final_freezes_ema(self, source):
        """When momentum=1.0, EMA should stop updating."""
        ema = EMANetwork(source, 1.0, 1.0, 10)
        before = [p.clone() for p in ema.ema_model.parameters()]
        with torch.no_grad():
            for p in source.parameters():
                p.add_(torch.randn_like(p) * 100)
        ema.update(source)
        for b, a in zip(before, ema.ema_model.parameters(), strict=True):
            assert torch.allclose(b, a, atol=1e-6)

    def test_step_counter(self, source):
        ema = EMANetwork(source)
        assert ema._step == 0
        ema.update(source)
        assert ema._step == 1
        ema.update(source)
        assert ema._step == 2

    def test_deep_copy_independence(self, source):
        """Modifying source should NOT modify EMA (before update)."""
        ema = EMANetwork(source)
        before = [p.clone() for p in ema.ema_model.parameters()]
        with torch.no_grad():
            for p in source.parameters():
                p.fill_(999.0)
        for b, a in zip(before, ema.ema_model.parameters(), strict=True):
            assert torch.allclose(b, a)

    # EMANetwork.forward was removed (f5a9e0e): the JEPA module reads
    # ``target_network.ema_model.aggregator`` / ``.head`` directly.
    def test_ema_model_reproduces_the_source_until_updated(self, source):
        ema = EMANetwork(source)
        x = torch.randn(B, D_EMBED)
        torch.testing.assert_close(ema.ema_model(x), source(x))

    def test_train_mode_keeps_the_ema_model_in_eval(self, source):
        ema = EMANetwork(source)
        ema.train()
        assert ema.training
        assert not ema.ema_model.training

    def test_buffers_are_copied_not_averaged(self):
        source = nn.Sequential(nn.Linear(D_EMBED, D_PROJ), nn.BatchNorm1d(D_PROJ))
        ema = EMANetwork(source, initial_momentum=0.5, final_momentum=0.5)
        source[1].running_mean.fill_(3.0)
        ema.update(source)
        torch.testing.assert_close(ema.ema_model[1].running_mean, torch.full((D_PROJ,), 3.0))

    def test_step_counter_is_saved_with_the_state_dict(self, source):
        ema = EMANetwork(source, 0.99, 1.0, 10)
        ema.update(source)
        ema.update(source)
        state = ema.state_dict()
        assert int(state["_step"]) == 2
        fresh = EMANetwork(source, 0.99, 1.0, 10)
        fresh.load_state_dict(state)
        assert int(fresh._step) == 2
        assert fresh.current_momentum == pytest.approx(ema.current_momentum)


# ═══════════════════════════════════════════════════════════════════
# 6. SSLPretrainModule: construction, optimizer and LR schedule
# ═══════════════════════════════════════════════════════════════════


def _has_bn(module: nn.Module) -> bool:
    return any(isinstance(m, nn.BatchNorm1d) for m in module.modules())


class TestSSLPretrainModuleConstruction:
    @pytest.mark.parametrize("method", METHODS)
    def test_builds_aggregator_and_projector(self, method):
        m = _make_module(method)
        assert m.ssl_method == method
        assert isinstance(m.aggregator, BaseMIL)
        assert isinstance(m.projector, Projector)
        assert m.embed_dim == D_EMBED

    def test_vicreg_components(self):
        m = _make_module("vicreg")
        assert isinstance(m.loss_fn, VICRegLoss)
        assert not _has_bn(m.projector)  # BN would defeat the variance term
        assert not hasattr(m, "predictor")
        assert not hasattr(m, "target_network")

    def test_jepa_components(self):
        m = _make_module("jepa")
        assert isinstance(m.loss_fn, JEPALoss)
        assert isinstance(m.predictor, Predictor)
        assert isinstance(m.target_network, EMANetwork)
        assert _has_bn(m.projector)
        assert all(not p.requires_grad for p in m.target_network.parameters())
        # The target starts as an exact copy of the online aggregator + projector.
        online = list(m._online_for_ema.parameters())
        target = list(m.target_network.ema_model.parameters())
        assert len(online) == len(target)
        assert all(torch.equal(o, t) for o, t in zip(online, target, strict=True))

    @pytest.mark.parametrize(
        "method, loss_cls", [("lejepa", LeJEPALoss), ("lejepa_mc", LeJEPAMCLoss)]
    )
    def test_lejepa_components(self, method, loss_cls):
        m = _make_module(method)
        assert isinstance(m.loss_fn, loss_cls)
        assert _has_bn(m.projector)  # proj_use_bn defaults to True
        assert not isinstance(m.projector.mlp[-1], nn.BatchNorm1d)  # proj_last_bn False
        assert not hasattr(m, "predictor")
        assert not hasattr(m, "target_network")

    @pytest.mark.parametrize("alias", ["pooled_jepa", "token_jepa", "JEPA"])
    def test_legacy_jepa_aliases(self, alias):
        m = _make_module(alias)
        assert m.ssl_method == "jepa"
        assert m.requested_ssl_method == alias.lower()

    @pytest.mark.parametrize("method", ["simclr", "byol", "dino", "bad"])
    def test_retired_and_unknown_methods_rejected(self, method):
        with pytest.raises(ValueError, match="Unknown ssl_method"):
            _make_module(method)

    def test_ssl_cfg_reaches_the_losses_and_heads(self):
        vicreg = _make_module(
            "vicreg",
            ssl_cfg={"inv_weight": 10.0, "cov_weight": 2.0, "var_target": 0.5, "var_eps": 1e-3},
        )
        loss = vicreg.loss_fn
        assert (loss.inv_weight, loss.cov_weight, loss.var_target, loss.eps) == (
            10.0,
            2.0,
            0.5,
            1e-3,
        )
        jepa = _make_module("jepa", ssl_cfg={"loss_type": "mse", "ema_momentum": 0.9})
        assert jepa.loss_fn.loss_type == "mse"
        assert jepa.target_network.initial_momentum == 0.9
        lejepa = _make_module(
            "lejepa",
            ssl_cfg={"sigreg_weight": 0.2, "num_slices": 32, "proj_use_bn": False},
        )
        assert (lejepa.loss_fn.sigreg_weight, lejepa.loss_fn.num_slices) == (0.2, 32)
        assert not _has_bn(lejepa.projector)
        last_bn = _make_module("lejepa_mc", ssl_cfg={"proj_last_bn": True})
        assert isinstance(last_bn.projector.mlp[-1], nn.BatchNorm1d)

    @pytest.mark.parametrize("name", ["vicreg", "jepa", "lejepa_2c", "lejepa_mc"])
    def test_shipped_method_configs_build(self, name):
        cfg = OmegaConf.merge(
            OmegaConf.load(CONFIG_DIR / "base.yaml"), OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
        )
        ssl_cfg = OmegaConf.to_container(cfg.ssl, resolve=True)
        ssl_cfg.update(proj_hidden_dim=32, proj_out_dim=16, pred_hidden_dim=16)  # keep it small
        m = _make_module(
            cfg.ssl_method,
            ssl_cfg=ssl_cfg,
            adam_betas=tuple(cfg.adam_betas),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        assert m.ssl_method == cfg.ssl_method
        if name == "vicreg":
            assert m.loss_fn.cov_weight == cfg.ssl.cov_weight
        if name.startswith("lejepa"):
            assert m.loss_fn.sigreg_weight == cfg.ssl.sigreg_weight
            assert m.loss_fn.num_slices == cfg.ssl.num_slices

    def test_hparams_saved(self):
        m = _make_module("vicreg")
        assert m.hparams.ssl_method == "vicreg"
        assert m.hparams.lr == 1e-3


class TestOptimizerConfiguration:
    def test_adamw_with_configured_betas_and_lr(self):
        opt = _make_module("vicreg", adam_betas=(0.8, 0.9), lr=2e-3).configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults["betas"] == (0.8, 0.9)
        assert opt.defaults["lr"] == 2e-3

    @pytest.mark.parametrize("method", METHODS)
    def test_weight_decay_skips_biases_and_norms(self, method):
        m = _make_module(method, weight_decay=0.05)
        decay, no_decay = m.configure_optimizers().param_groups
        assert decay["weight_decay"] == 0.05
        assert no_decay["weight_decay"] == 0.0
        assert all(p.ndim > 1 for p in decay["params"])
        assert all(p.ndim <= 1 for p in no_decay["params"])
        n_optimized = len(decay["params"]) + len(no_decay["params"])
        assert n_optimized == sum(1 for p in m.parameters() if p.requires_grad)

    def test_ema_target_is_not_optimized(self):
        m = _make_module("jepa")
        groups = m.configure_optimizers().param_groups
        optimized = {id(p) for g in groups for p in g["params"]}
        assert not optimized & {id(p) for p in m.target_network.parameters()}


class TestLRScheduler:
    def test_warmup_ramp(self):
        m = _make_module("vicreg", lr_scheduler="cosine", warmup_epochs=5, max_epochs=20)
        config = m.configure_optimizers()
        opt = config["optimizer"]
        sched = config["lr_scheduler"]
        lrs = []
        for _ in range(5):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-8

    def test_cosine_decay_after_warmup(self):
        m = _make_module("vicreg", lr_scheduler="cosine", warmup_epochs=2, max_epochs=20)
        config = m.configure_optimizers()
        opt = config["optimizer"]
        sched = config["lr_scheduler"]
        for _ in range(3):
            sched.step()
        lr_after_warmup = opt.param_groups[0]["lr"]
        for _ in range(10):
            sched.step()
        lr_later = opt.param_groups[0]["lr"]
        assert lr_later < lr_after_warmup

    def test_no_scheduler_constant_lr(self):
        m = _make_module("vicreg", lr_scheduler="none", lr=0.001)
        opt = m.configure_optimizers()
        assert opt.param_groups[0]["lr"] == 0.001


# ═══════════════════════════════════════════════════════════════════
# 7. Training and validation inside a real Lightning Trainer
# ═══════════════════════════════════════════════════════════════════


EXPECTED_TRAIN_METRICS = {
    "vicreg": {"train/inv_loss", "train/var_loss", "train/cov_loss"},
    "jepa": {"train/loss12", "train/loss21", "train/ema_momentum"},
    "lejepa": {"train/pred_loss", "train/sigreg_loss", "ssl/grad_ratio_sigreg_to_pred"},
    "lejepa_mc": {
        "train/pred_loss",
        "train/sigreg_loss",
        "train/pred_loss_global_mean",
        "train/pred_loss_local_mean",
        "ssl/grad_norm_sigreg",
    },
}
DUAL_VIEW_DIAGNOSTICS = {
    "aug/input_centroid_cosine",
    "aug/token_overlap_frac",
    "aug/crop_area_frac1_mean",
    "aug/crop_noop1_frac",
    "ssl/z_centroid_l2",
}


class TestTrainingStep:
    @pytest.mark.parametrize("method", METHODS)
    def test_fit_updates_weights_and_logs_finite_losses(self, method):
        torch.manual_seed(0)
        m = _make_module(method)
        before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
        trainer = _fit(m, _batches(method, 2), _batches(method, 1, first_seed=100))
        assert trainer.global_step == 2
        for key in ("train/loss", "val/loss"):
            assert torch.isfinite(trainer.callback_metrics[key])
        changed = {
            n for n, p in m.named_parameters() if n in before and not torch.equal(p, before[n])
        }
        assert any(n.startswith("aggregator.") for n in changed)
        assert any(n.startswith("projector.") for n in changed)

    @pytest.mark.parametrize("method", METHODS)
    def test_method_metrics_are_logged(self, method):
        m = _make_module(method)
        metrics = set(_fit(m, _batches(method, 1)).callback_metrics)
        assert {"train/loss", "ssl/z_dim_std_min"}.issubset(metrics)
        assert EXPECTED_TRAIN_METRICS[method].issubset(metrics)
        if method in DUAL_VIEW_METHODS:
            assert DUAL_VIEW_DIAGNOSTICS.issubset(metrics)
        # Diagnostics are logged under ssl/*, integer counts are not logged.
        assert not any("_diagnostics" in k or k.endswith("num_global") for k in metrics)

    @pytest.mark.parametrize("method", METHODS)
    def test_every_trainable_parameter_gets_a_finite_gradient(self, method):
        rec = _Recorder()
        _fit(_make_module(method), _batches(method, 2), callbacks=[rec])
        assert rec.missing_grads == [[], []]

    @pytest.mark.parametrize("method", METHODS)
    def test_modes_during_fit(self, method):
        rec = _Recorder()
        _fit(
            _make_module(method),
            _batches(method, 2),
            _batches(method, 1, first_seed=100),
            callbacks=[rec],
        )
        assert rec.train_modes == [True, True]
        assert rec.val_modes == [False]
        if method == "jepa":
            assert rec.target_modes == [False, False]  # EMA target never trains

    @pytest.mark.parametrize("method", METHODS)
    def test_loss_decreases_on_a_repeated_batch(self, method):
        torch.manual_seed(0)
        rec = _Recorder()
        batch = _batches(method, 1)[0]
        _fit(_make_module(method, lr=1e-2), [batch] * 12, callbacks=[rec])
        assert len(rec.losses) == 12
        assert sum(rec.losses[-3:]) < sum(rec.losses[:3])

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("scale", [1e2, 1e-6])
    def test_extreme_feature_scales_stay_finite(self, method, scale):
        m = _make_module(method)
        trainer = _fit(m, _batches(method, 1, scale=scale))
        assert torch.isfinite(trainer.callback_metrics["train/loss"])
        assert all(torch.isfinite(p).all() for p in m.parameters())


class TestEdgeCaseBatches:
    @pytest.mark.parametrize("method", DUAL_VIEW_METHODS)
    def test_partially_masked_views(self, method):
        batch = _dual_batch()
        batch["mask1"][:, -5:] = 0
        batch["mask2"][:, :3] = 0
        trainer = _fit(_make_module(method), [batch])
        assert torch.isfinite(trainer.callback_metrics["train/loss"])

    @pytest.mark.parametrize("method", DUAL_VIEW_METHODS)
    def test_views_of_different_lengths(self, method):
        batch = _dual_batch(n=30)
        short = _dual_batch(n=15, seed=1)
        for key in ("view2", "mask2", "coords2", "lengths2"):
            batch[key] = short[key]
        trainer = _fit(_make_module(method), [batch])
        assert torch.isfinite(trainer.callback_metrics["train/loss"])

    @pytest.mark.parametrize("method", DUAL_VIEW_METHODS)
    @pytest.mark.parametrize("n_patches", [1, 200])
    def test_bag_size_extremes(self, method, n_patches):
        trainer = _fit(_make_module(method), [_dual_batch(b=2, n=n_patches)])
        assert torch.isfinite(trainer.callback_metrics["train/loss"])

    def test_multicrop_without_local_views(self):
        trainer = _fit(_make_module("lejepa_mc"), [_mc_batch(v_l=0)])
        metrics = trainer.callback_metrics
        assert torch.isfinite(metrics["train/loss"])
        assert metrics["train/pred_loss_local_mean"].item() == 0.0

    @pytest.mark.parametrize("method", METHODS)
    def test_single_slide_validation_batch(self, method):
        """BatchNorm heads must handle B=1 at validation time (running stats)."""
        make = _mc_batch if method == "lejepa_mc" else _dual_batch
        trainer = _validate(_make_module(method), [make(b=1)])
        assert torch.isfinite(trainer.callback_metrics["val/loss"])


class TestValidationStep:
    @pytest.mark.parametrize("method", METHODS)
    def test_returns_loss_and_aggregator_embeddings(self, method):
        torch.manual_seed(0)
        m = _make_module(method)
        batch = _batches(method, 1)[0]
        rec = _Recorder()
        _validate(m, [batch], callbacks=[rec])
        (out,) = rec.val_outputs
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"])
        assert out["embeddings"].shape == (B, D_EMBED)
        assert not out["embeddings"].requires_grad
        # Embeddings are the aggregator output (not the projector) of view 1,
        # or of the first global view for multi-crop.
        if method == "lejepa_mc":
            view, mask, coords = (
                batch["global_views"][0],
                batch["global_masks"][0],
                batch["global_coords"][0],
            )
        else:
            view, mask, coords = batch["view1"], batch["mask1"], batch["coords1"]
        m.eval()
        with torch.no_grad():
            expected = m.aggregator(view, mask=mask, coords=coords).slide_embedding
        torch.testing.assert_close(out["embeddings"], expected)

    @pytest.mark.parametrize("method", METHODS)
    def test_validation_is_deterministic(self, method):
        m = _make_module(method)
        batches = _batches(method, 2)
        first = _validate(m, batches).callback_metrics["val/loss"]
        second = _validate(m, batches).callback_metrics["val/loss"]
        assert first.item() == second.item()

    def test_validation_does_not_touch_weights_or_ema(self):
        m = _make_module("jepa")
        before = copy.deepcopy(m.state_dict())
        _validate(m, _batches("jepa", 2))
        after = m.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)


class TestEMATarget:
    """JEPA's EMA target moves only through the EMA update, never by gradient."""

    def test_target_follows_the_ema_rule(self):
        torch.manual_seed(0)
        m = _make_module("jepa", lr=1e-2, ssl_cfg={"ema_momentum": 0.9, "num_slices": 64})
        seen = {}

        class _Snapshot(L.Callback):
            # Callback hooks run before the module's on_train_batch_end, i.e.
            # after the optimizer step and before the EMA update.
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                target = pl_module.target_network
                seen["target"] = [p.detach().clone() for p in target.ema_model.parameters()]
                seen["online"] = [
                    p.detach().clone() for p in pl_module._online_for_ema.parameters()
                ]
                seen["momentum"] = target.current_momentum

        _fit(m, _batches("jepa", 1), callbacks=[_Snapshot()])
        mom = seen["momentum"]
        assert mom == pytest.approx(0.9)
        target_params = m.target_network.ema_model.parameters()
        for after, before, online in zip(
            target_params, seen["target"], seen["online"], strict=True
        ):
            torch.testing.assert_close(after, mom * before + (1 - mom) * online)
            assert not torch.equal(after, before)

    def test_target_gets_no_gradient_and_counts_steps(self):
        m = _make_module("jepa")
        trainer = _fit(m, _batches("jepa", 3))
        target = m.target_network
        assert all(p.grad is None and not p.requires_grad for p in target.parameters())
        assert int(target._step) == 3
        assert target.total_steps == trainer.estimated_stepping_batches == 3
        optimized = {id(p) for g in trainer.optimizers[0].param_groups for p in g["params"]}
        assert not optimized & {id(p) for p in target.parameters()}

    def test_target_and_online_diverge_under_training(self):
        m = _make_module("jepa", lr=1e-2)
        _fit(m, _batches("jepa", 3))
        online = list(m.projector.parameters())
        target = list(m.target_network.ema_model.head.parameters())
        assert any(not torch.allclose(o, t) for o, t in zip(online, target, strict=True))

    def test_gradient_accumulation_rejected(self):
        with pytest.raises(ValueError, match="accumulate_grad_batches=2"):
            _fit(_make_module("jepa"), _batches("jepa", 2), accumulate_grad_batches=2)


class TestCheckpointRoundTrip:
    @pytest.mark.parametrize("method", METHODS)
    def test_trainer_checkpoint_restores_the_module(self, method, tmp_path):
        torch.manual_seed(0)
        m = _make_module(method, lr=1e-2)
        trainer = _fit(m, _batches(method, 2))
        path = tmp_path / "pretrain.ckpt"
        trainer.save_checkpoint(path)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["ssl_method"] == method
        assert ckpt["embed_dim"] == D_EMBED
        aggregator = build_aggregator("abmil", in_dim=D_IN, model_cfg=MODEL_CFG)
        aggregator.load_state_dict(ckpt["aggregator_state_dict"])
        bag = torch.randn(2, N_PATCHES, D_IN)
        m.eval()
        aggregator.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                aggregator(bag).slide_embedding, m.aggregator(bag).slide_embedding
            )

        restored = SSLPretrainModule.load_from_checkpoint(path, map_location="cpu")
        assert restored.ssl_method == method
        expected = m.state_dict()
        assert set(restored.state_dict()) == set(expected)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[key], msg=key)


class TestPretrainPipeline:
    """PretrainDataModule -> SSLPretrainModule on a memmap store from the builder."""

    @pytest.fixture(scope="class")
    def store(self, build_feature_store):
        return build_feature_store([300, 260, 220, 400, 280, 350], feat_dim=D_IN)

    @pytest.mark.parametrize("method", METHODS)
    def test_fit_end_to_end(self, store, method):
        if method == "lejepa_mc":
            aug = {"local_fixed_n": 16, "global_min_tokens": 8, "split_overlap": 0.2}
        else:
            aug = {"min_tokens": 8, "split_overlap": 0.2}
        dm = PretrainDataModule(
            mmap_dir=str(store.mmap_dir),
            fixed_n=32,
            batch_size=2,
            ssl_method=method,
            augmentation_cfg=aug,
            dataset_max_instances=128,
            num_workers=0,
            val_frac=0.34,
            force_float32=True,  # fp32 on CPU; GPU runs use bf16-mixed on fp16 features
        )
        m = _make_module(method)
        trainer = _trainer(max_steps=2, limit_val_batches=1)
        trainer.fit(m, datamodule=dm)
        assert trainer.global_step == 2
        assert torch.isfinite(trainer.callback_metrics["train/loss"])
        assert torch.isfinite(trainer.callback_metrics["val/loss"])
        if method == "jepa":
            assert int(m.target_network._step) == 2


class TestQualityCallbacksInTrainer:
    @pytest.mark.parametrize("method", ["vicreg", "lejepa_mc"])
    def test_ssl_quality_metrics_logged(self, method):
        cb = SSLQualityCallback(compute_every_n_epochs=1)
        trainer = _validate(_make_module(method), _batches(method, 3), callbacks=[cb])
        metrics = trainer.callback_metrics
        assert 1.0 <= metrics["ssl/rankme"].item() <= D_EMBED
        assert metrics["ssl/rankme_ratio"].item() == pytest.approx(
            metrics["ssl/rankme"].item() / D_EMBED, rel=1e-5
        )
        assert torch.isfinite(metrics["ssl/alpha_req"])
        assert cb._embeddings == []


# ═══════════════════════════════════════════════════════════════════
# 8. Quality callbacks: RankMe, alpha-ReQ, combined
# ═══════════════════════════════════════════════════════════════════


class TestRankMeCallback:
    def test_full_rank_identity(self):
        r, _ = RankMeCallback()._compute_rankme(torch.eye(50))
        assert r == pytest.approx(50.0, rel=0.05)

    def test_collapsed(self):
        r, _ = RankMeCallback()._compute_rankme(torch.ones(100, 32))
        assert r < 2.0

    def test_partial_rank(self):
        torch.manual_seed(42)
        Z = torch.randn(200, 5) @ torch.randn(5, 32)
        r, _ = RankMeCallback()._compute_rankme(Z)
        assert 3.0 < r < 8.0

    def test_high_rank_random(self):
        torch.manual_seed(42)
        r, D = RankMeCallback()._compute_rankme(torch.randn(200, 32))
        assert r / D > 0.5

    def test_bounded_by_min_dim(self):
        torch.manual_seed(42)
        r, _ = RankMeCallback()._compute_rankme(torch.randn(50, 32))
        assert r <= 32 + 0.5

    def test_monotonic_with_rank(self):
        torch.manual_seed(42)
        rs = []
        for k in [2, 5, 10, 20]:
            Z = torch.randn(100, k) @ torch.randn(k, 32)
            r, _ = RankMeCallback()._compute_rankme(Z)
            rs.append(r)
        for i in range(1, len(rs)):
            assert rs[i] > rs[i - 1] - 1.0

    def test_epoch_skip(self):
        cb = RankMeCallback(compute_every_n_epochs=5)
        tr = MagicMock()
        tr.current_epoch = 3
        cb.on_validation_batch_end(
            tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, 0
        )
        assert len(cb._embeddings) == 0
        tr.current_epoch = 5
        cb.on_validation_batch_end(
            tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, 0
        )
        assert len(cb._embeddings) == 1

    def test_max_samples_cap(self):
        cb = RankMeCallback(compute_every_n_epochs=1, max_samples=16)
        tr = MagicMock()
        tr.current_epoch = 0
        for i in range(3):
            cb.on_validation_batch_end(
                tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, i
            )
        assert len(cb._embeddings) == 3
        pl = MagicMock()
        cb.on_validation_epoch_end(tr, pl)
        assert len(cb._embeddings) == 0

    def test_clears_after_epoch(self):
        cb = RankMeCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        cb.on_validation_batch_end(
            tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, 0
        )
        pl = MagicMock()
        cb.on_validation_epoch_end(tr, pl)
        assert len(cb._embeddings) == 0

    def test_multiple_batches_accumulate(self):
        cb = RankMeCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        for i in range(4):
            cb.on_validation_batch_end(
                tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, i
            )
        assert len(cb._embeddings) == 4

    def test_no_embeddings_skips(self):
        cb = RankMeCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        pl = MagicMock()
        cb.on_validation_epoch_end(tr, pl)
        pl.log.assert_not_called()

    def test_single_sample_skips(self):
        cb = RankMeCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        cb.on_validation_batch_end(
            tr, MagicMock(), {"embeddings": torch.randn(1, D_EMBED)}, None, 0
        )
        pl = MagicMock()
        cb.on_validation_epoch_end(tr, pl)
        pl.log.assert_not_called()

    def test_ignores_non_dict_outputs(self):
        cb = RankMeCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        cb.on_validation_batch_end(tr, MagicMock(), "not_a_dict", None, 0)
        assert len(cb._embeddings) == 0

    def test_ignores_dict_without_embeddings(self):
        cb = RankMeCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        cb.on_validation_batch_end(tr, MagicMock(), {"loss": 0.5}, None, 0)
        assert len(cb._embeddings) == 0


class TestAlphaReQCallback:
    def test_flat_spectrum(self):
        torch.manual_seed(42)
        assert AlphaReQCallback()._compute_alpha_req(torch.randn(500, 32)) < 1.5

    def test_steep_spectrum(self):
        torch.manual_seed(42)
        Z = torch.randn(200, 1) @ torch.randn(1, 32) + torch.randn(200, 32) * 0.01
        assert AlphaReQCallback()._compute_alpha_req(Z) > 2.0

    def test_positive(self):
        torch.manual_seed(42)
        assert AlphaReQCallback()._compute_alpha_req(torch.randn(200, 32)) > 0

    def test_n_less_than_d(self):
        torch.manual_seed(42)
        a = AlphaReQCallback()._compute_alpha_req(torch.randn(10, 64))
        assert isinstance(a, float)
        assert a >= 0

    def test_scale_invariant(self):
        torch.manual_seed(42)
        Z = torch.randn(200, 32)
        a1 = AlphaReQCallback()._compute_alpha_req(Z)
        a10 = AlphaReQCallback()._compute_alpha_req(Z * 10.0)
        assert a1 == pytest.approx(a10, abs=0.1)

    def test_epoch_lifecycle(self):
        cb = AlphaReQCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        for i in range(5):
            cb.on_validation_batch_end(
                tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, i
            )
        assert len(cb._embeddings) == 5
        pl = MagicMock()
        cb.on_validation_epoch_end(tr, pl)
        assert len(cb._embeddings) == 0
        pl.log.assert_called_once()

    def test_few_eigenvalues_returns_zero(self):
        Z = torch.zeros(10, 32)
        Z[:, 0] = torch.randn(10)
        a = AlphaReQCallback()._compute_alpha_req(Z)
        assert isinstance(a, float)


class TestSSLQualityCallback:
    def test_has_both(self):
        cb = SSLQualityCallback()
        assert isinstance(cb.rankme, RankMeCallback)
        assert isinstance(cb.alpha_req, AlphaReQCallback)

    def test_collects_into_one_shared_buffer(self):
        """The wrapper keeps a single embedding buffer; the inner callbacks' own
        buffers stay unused (their hooks are not called by the wrapper)."""
        cb = SSLQualityCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        cb.on_validation_batch_end(
            tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, 0
        )
        assert len(cb._embeddings) == 1
        assert cb.rankme._embeddings == []
        assert cb.alpha_req._embeddings == []

    def test_shared_params(self):
        cb = SSLQualityCallback(compute_every_n_epochs=5, max_samples=1024)
        assert cb.rankme.compute_every_n_epochs == 5
        assert cb.alpha_req.max_samples == 1024

    def test_full_lifecycle(self):
        cb = SSLQualityCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        for i in range(5):
            cb.on_validation_batch_end(
                tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, i
            )
        pl = MagicMock()
        cb.on_validation_epoch_end(tr, pl)
        assert pl.log.call_count >= 2


class TestMetricConsistency:
    def test_good_vs_bad_representation(self):
        torch.manual_seed(42)
        Z_good = torch.randn(200, 32)
        Z_bad = torch.randn(200, 1) @ torch.randn(1, 32)
        cb_r, cb_a = RankMeCallback(), AlphaReQCallback()
        r_good, _ = cb_r._compute_rankme(Z_good)
        r_bad, _ = cb_r._compute_rankme(Z_bad)
        a_good = cb_a._compute_alpha_req(Z_good)
        a_bad = cb_a._compute_alpha_req(Z_bad)
        assert r_good > r_bad
        assert a_good < a_bad

    def test_medium_rank_in_between(self):
        torch.manual_seed(42)
        cb_r = RankMeCallback()
        r_full, _ = cb_r._compute_rankme(torch.randn(200, 32))
        r_10, _ = cb_r._compute_rankme(torch.randn(200, 10) @ torch.randn(10, 32))
        r_1, _ = cb_r._compute_rankme(torch.randn(200, 1) @ torch.randn(1, 32))
        assert r_full > r_10 > r_1


class TestCallbackFullLifecycle:
    def _mock_trainer(self, epoch):
        tr = MagicMock()
        tr.current_epoch = epoch
        return tr

    def test_rankme_full_cycle(self):
        cb = RankMeCallback(compute_every_n_epochs=1, max_samples=2048)
        tr = self._mock_trainer(0)
        pl = MagicMock()

        torch.manual_seed(42)
        for i in range(5):
            outputs = {"embeddings": torch.randn(B, D_EMBED)}
            cb.on_validation_batch_end(tr, pl, outputs, None, i)

        assert len(cb._embeddings) == 5
        cb.on_validation_epoch_end(tr, pl)

        log_calls = [c for c in pl.log.call_args_list if "rankme" in str(c)]
        assert len(log_calls) >= 1
        assert len(cb._embeddings) == 0

    def test_alpha_req_full_cycle(self):
        cb = AlphaReQCallback(compute_every_n_epochs=1, max_samples=2048)
        tr = self._mock_trainer(0)
        pl = MagicMock()

        torch.manual_seed(42)
        for i in range(5):
            outputs = {"embeddings": torch.randn(B, D_EMBED)}
            cb.on_validation_batch_end(tr, pl, outputs, None, i)

        cb.on_validation_epoch_end(tr, pl)
        pl.log.assert_called_once()
        assert len(cb._embeddings) == 0

    def test_combined_callback_full_cycle(self):
        cb = SSLQualityCallback(compute_every_n_epochs=1)
        tr = self._mock_trainer(0)
        pl = MagicMock()

        torch.manual_seed(42)
        for i in range(5):
            outputs = {"embeddings": torch.randn(B, D_EMBED)}
            cb.on_validation_batch_end(tr, pl, outputs, None, i)

        cb.on_validation_epoch_end(tr, pl)
        assert pl.log.call_count >= 2

    def test_skip_non_compute_epoch(self):
        cb = RankMeCallback(compute_every_n_epochs=5)
        tr = self._mock_trainer(3)
        pl = MagicMock()

        outputs = {"embeddings": torch.randn(B, D_EMBED)}
        cb.on_validation_batch_end(tr, pl, outputs, None, 0)
        assert len(cb._embeddings) == 0

        cb.on_validation_epoch_end(tr, pl)
        pl.log.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# 9. Numerical stability of the losses and metrics
# ═══════════════════════════════════════════════════════════════════


class TestLossNumericalStability:
    def test_vicreg_eps_prevents_nan(self):
        z = torch.zeros(B, D_PROJ, requires_grad=True)
        out = VICRegLoss()(z, z.clone().requires_grad_(True))
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["var_loss"])

    @pytest.mark.parametrize("scale", [1e-6, 1e4])
    @pytest.mark.parametrize("loss_name", ["vicreg", "jepa", "lejepa", "lejepa_mc"])
    def test_losses_finite_for_extreme_embeddings(self, loss_name, scale):
        torch.manual_seed(0)
        z1 = (torch.randn(B, D_PROJ) * scale).requires_grad_(True)
        z2 = (torch.randn(B, D_PROJ) * scale).requires_grad_(True)
        if loss_name == "vicreg":
            loss = VICRegLoss()(z1, z2)["loss"]
        elif loss_name == "jepa":
            loss = JEPALoss()(z1, z2)["loss"]
        elif loss_name == "lejepa":
            loss = LeJEPALoss(num_slices=32)(z1, z2)["loss"]
        else:
            loss = LeJEPAMCLoss(num_slices=32)([z1, z2], num_global=1)["loss"]
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(z1.grad).all()

    def test_rankme_all_zeros(self):
        r, _D = RankMeCallback()._compute_rankme(torch.zeros(50, 32))
        assert r >= 0

    def test_alpha_req_all_zeros(self):
        a = AlphaReQCallback()._compute_alpha_req(torch.zeros(50, 32))
        assert isinstance(a, float)
        assert a >= 0
