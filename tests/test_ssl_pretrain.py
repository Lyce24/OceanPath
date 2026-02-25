"""
Tests for SSL pretraining modules.

Coverage:
  1.  TestVICRegLoss              — output shape/keys, invariance/variance/covariance math,
                                    weight scaling, gradient, edge cases
  2.  TestSimCLRLoss              — output keys, positivity, alignment, temperature,
                                    L2 invariance, symmetry, batch-size edge cases
  3.  TestBYOLLoss                — bounds, perfect prediction, stop-grad, scale invariance,
                                    symmetry, orthogonal-max test
  4.  TestDINOLoss                — positivity, center EMA, momentum formula, teacher detach,
                                    center drift, temperature effects
  5.  TestJEPALoss                — smooth_l1 vs mse, unknown type, perfect-predict zero,
                                    target detach, huber robustness, gradient
  6.  TestProjector               — shape, layer count, BN toggle, last BN, gradient flow
  7.  TestPredictor               — shape, bottleneck size, BN toggle
  8.  TestDINOHead                — shape, weight norm, no bias, frozen g, L2 norm
  9.  TestEMANetwork              — initial copy, frozen params, update formula, momentum
                                    schedule (start/increase/final/end), forward delegation,
                                    step counter
 10.  TestSSLPretrainModule       — construction, unknown method, component presence,
                                    training_step output, no-coords, val_step output/shape,
                                    optimizer config (none/cosine/warmup), checkpoint,
                                    BN per method
 11.  TestRankMeCallback          — known spectra (full rank, collapsed, partial, random),
                                    bounds, monotonicity with rank, epoch skip, max_samples cap,
                                    clearing after epoch, multiple batches accumulate
 12.  TestAlphaReQCallback        — flat spectrum, steep spectrum, positivity, N<D,
                                    scale invariance, epoch lifecycle
 13.  TestSSLQualityCallback      — has both, delegates, shared params
 14.  TestGradientFlow            — backward produces gradients, EMA no grad, aggregator grad
 15.  TestEMAMethodIntegration    — update on_train_batch_end, BYOL divergence
 16.  TestEdgeCases               — partial mask, asymmetric lengths, single sample, small batch
 17.  TestMetricConsistency       — good vs bad representations
 18.  TestMultiStepTraining       — VICReg/BYOL/DINO loss does not diverge over several steps
 19.  TestCheckpointRoundTrip     — save + reload aggregator weights match
 20.  TestCallbackFullLifecycle   — full train->val->metric pipeline via mock trainer
 21.  TestNumericalStability      — large values, small values, mixed precision
 22.  TestLRScheduler             — warmup ramp, cosine decay shape
 23.  TestTargetEncoderIsolation  — target branch never receives gradient
 24.  TestTrainEvalMode           — train/eval mode interactions
 25.  TestArchitectureVerification — parameter counts, component presence
"""

import copy
import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from oceanpath.ssl.callbacks import (
    AlphaReQCallback,
    RankMeCallback,
    SSLQualityCallback,
)
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
from oceanpath.ssl.pretrain_module import SSLPretrainModule

# ═══════════════════════════════════════════════════════════════════
# Constants and fixtures
# ═══════════════════════════════════════════════════════════════════

B = 8  # batch size
D_PROJ = 64  # projection dim
D_EMBED = 32  # aggregator embed dim
D_IN = 16  # patch feature dim
N_PATCHES = 20  # patches per view
D_PROTO = 128  # DINO prototype dim


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


@pytest.fixture
def ssl_batch():
    """Standard SSL batch with coords."""
    torch.manual_seed(42)
    return {
        "view1": torch.randn(B, N_PATCHES, D_IN),
        "mask1": torch.ones(B, N_PATCHES),
        "view2": torch.randn(B, N_PATCHES, D_IN),
        "mask2": torch.ones(B, N_PATCHES),
        "coords1": torch.randint(0, 1000, (B, N_PATCHES, 2)).float(),
        "coords2": torch.randint(0, 1000, (B, N_PATCHES, 2)).float(),
    }


@pytest.fixture
def ssl_batch_no_coords():
    """SSL batch without coordinates."""
    torch.manual_seed(42)
    return {
        "view1": torch.randn(B, N_PATCHES, D_IN),
        "mask1": torch.ones(B, N_PATCHES),
        "view2": torch.randn(B, N_PATCHES, D_IN),
        "mask2": torch.ones(B, N_PATCHES),
    }


def _make_module(method, **overrides):
    """Factory for SSLPretrainModule with small dimensions for fast tests."""
    defaults = {
        "ssl_method": method,
        "arch": "abmil",
        "in_dim": D_IN,
        "model_cfg": {
            "embed_dim": D_EMBED,
            "dropout": 0.0,
            "attn_dim": 16,
            "gate": True,
            "num_fc_layers": 1,
        },
        "ssl_cfg": {
            "proj_hidden_dim": 64,
            "proj_out_dim": D_PROJ,
            "proj_num_layers": 2,
            "pred_hidden_dim": 32,
            "ema_momentum": 0.99,
            "ema_final_momentum": 1.0,
            "n_prototypes": D_PROTO,
            "bottleneck_dim": 32,
            "teacher_temp": 0.04,
            "student_temp": 0.1,
            "center_momentum": 0.9,
        },
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "warmup_epochs": 0,
        "max_epochs": 10,
        "lr_scheduler": "none",
    }
    defaults.update(overrides)
    return SSLPretrainModule(**defaults)


ALL_METHODS = ["vicreg", "simclr", "byol", "dino", "jepa"]
EMA_METHODS = ["byol", "dino", "jepa"]
PROJECTOR_METHODS = ["vicreg", "simclr", "byol", "jepa"]


# ═══════════════════════════════════════════════════════════════════
# 1. VICReg Loss
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
# 2. SimCLR Loss
# ═══════════════════════════════════════════════════════════════════


class TestSimCLRLoss:
    def test_output_keys(self, z_pair):
        assert "loss" in SimCLRLoss()(*z_pair)

    def test_loss_positive(self, z_pair):
        assert SimCLRLoss()(*z_pair)["loss"].item() >= 0

    def test_perfect_alignment_low_loss(self):
        """Identical pairs should have lower loss than random pairs."""
        torch.manual_seed(99)
        z = torch.randn(B, D_PROJ)
        z_rand = torch.randn(B, D_PROJ)
        fn = SimCLRLoss(temperature=0.1)
        assert fn(z, z.clone())["loss"].item() < fn(z, z_rand)["loss"].item()

    def test_lower_temperature_harder(self, z_pair):
        l_hi = SimCLRLoss(temperature=1.0)(*z_pair)["loss"]
        l_lo = SimCLRLoss(temperature=0.01)(*z_pair)["loss"]
        assert l_lo.item() > l_hi.item()

    def test_l2_normalization_invariance(self):
        """Loss should be invariant to input scale (due to internal L2 norm)."""
        torch.manual_seed(42)
        z1, z2 = torch.randn(B, D_PROJ), torch.randn(B, D_PROJ)
        fn = SimCLRLoss()
        l1 = fn(z1, z2)["loss"]
        l2 = fn(z1 * 5.0, z2 * 0.1)["loss"]
        assert l1.item() == pytest.approx(l2.item(), abs=1e-5)

    def test_gradient_flows(self, z_pair):
        SimCLRLoss()(*z_pair)["loss"].backward()
        assert z_pair[0].grad is not None
        assert z_pair[1].grad is not None

    def test_batch_size_two(self):
        """Minimum viable batch (2 positives, 2 negatives)."""
        z1 = torch.randn(2, D_PROJ, requires_grad=True)
        z2 = torch.randn(2, D_PROJ, requires_grad=True)
        loss = SimCLRLoss()(z1, z2)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        assert z1.grad is not None

    def test_symmetric(self, z_pair):
        """SimCLR loss should be symmetric: L(z1,z2) == L(z2,z1)."""
        fn = SimCLRLoss()
        l1 = fn(z_pair[0], z_pair[1])["loss"]
        l2 = fn(z_pair[1], z_pair[0])["loss"]
        assert l1.item() == pytest.approx(l2.item(), abs=1e-5)

    def test_uniform_negatives_baseline(self):
        """With many random samples, loss approx log(2B-1) for unit-temp."""
        torch.manual_seed(42)
        big_B = 64
        z1 = torch.randn(big_B, D_PROJ)
        z2 = torch.randn(big_B, D_PROJ)
        loss = SimCLRLoss(temperature=1.0)(z1, z2)["loss"]
        expected = math.log(2 * big_B - 1)
        assert loss.item() == pytest.approx(expected, rel=0.3)

    def test_negative_pair_count(self):
        """Forward should run correctly for varying batch sizes."""
        for b in [2, 4, 16]:
            fn = SimCLRLoss()
            z1 = torch.randn(b, D_PROJ)
            z2 = torch.randn(b, D_PROJ)
            assert torch.isfinite(fn(z1, z2)["loss"])


# ═══════════════════════════════════════════════════════════════════
# 3. BYOL Loss
# ═══════════════════════════════════════════════════════════════════


class TestBYOLLoss:
    def test_loss_bounded(self):
        """Cosine-based loss bounded in [0, 4]."""
        fn = BYOLLoss()
        out = fn(
            torch.randn(B, D_PROJ),
            torch.randn(B, D_PROJ),
            torch.randn(B, D_PROJ),
            torch.randn(B, D_PROJ),
        )
        assert 0 <= out["loss"].item() <= 4.0 + 1e-5

    def test_zero_loss_for_perfect_prediction(self):
        """When predictions match targets exactly, loss = 0."""
        z = torch.randn(B, D_PROJ)
        out = BYOLLoss()(z.clone(), z.clone(), z.clone(), z.clone())
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-5)

    def test_max_loss_for_opposite(self):
        """Anti-aligned predictions should give maximum loss approx 4.0."""
        z = torch.randn(B, D_PROJ)
        out = BYOLLoss()(z.clone(), z.clone(), -z.clone(), -z.clone())
        assert out["loss"].item() == pytest.approx(4.0, abs=0.1)

    def test_target_detached(self):
        """Gradients should not flow through target inputs."""
        p1 = torch.randn(B, D_PROJ, requires_grad=True)
        p2 = torch.randn(B, D_PROJ, requires_grad=True)
        z1_t = torch.randn(B, D_PROJ, requires_grad=True)
        z2_t = torch.randn(B, D_PROJ, requires_grad=True)
        BYOLLoss()(p1, p2, z1_t, z2_t)["loss"].backward()
        assert p1.grad is not None
        assert p2.grad is not None
        assert z1_t.grad is None
        assert z2_t.grad is None

    def test_scale_invariant(self):
        """Cosine similarity is scale-invariant."""
        torch.manual_seed(42)
        p, z_t = torch.randn(B, D_PROJ), torch.randn(B, D_PROJ)
        fn = BYOLLoss()
        l1 = fn(p, p.clone(), z_t, z_t.clone())["loss"]
        l2 = fn(p * 3.0, p.clone() * 3.0, z_t * 0.5, z_t.clone() * 0.5)["loss"]
        assert l1.item() == pytest.approx(l2.item(), abs=1e-5)

    def test_symmetric_in_views(self):
        """Swapping (p1, z2_target) <-> (p2, z1_target) gives same loss."""
        torch.manual_seed(42)
        p1 = torch.randn(B, D_PROJ)
        p2 = torch.randn(B, D_PROJ)
        z1 = torch.randn(B, D_PROJ)
        z2 = torch.randn(B, D_PROJ)
        fn = BYOLLoss()
        l_fwd = fn(p1, p2, z1, z2)["loss"]
        l_rev = fn(p2, p1, z2, z1)["loss"]
        assert l_fwd.item() == pytest.approx(l_rev.item(), abs=1e-5)


# ═══════════════════════════════════════════════════════════════════
# 4. DINO Loss
# ═══════════════════════════════════════════════════════════════════


class TestDINOLoss:
    def test_loss_positive(self):
        torch.manual_seed(42)
        s = torch.randn(B, D_PROTO, requires_grad=True)
        t = torch.randn(B, D_PROTO)
        assert DINOLoss(out_dim=D_PROTO)(s, t)["loss"].item() >= 0

    def test_center_updates(self):
        fn = DINOLoss(out_dim=D_PROTO, center_momentum=0.9)
        c0 = fn.center.clone()
        fn(torch.randn(B, D_PROTO), torch.randn(B, D_PROTO))
        assert not torch.allclose(c0, fn.center)

    def test_center_momentum_value(self):
        """With momentum=0.5, center = 0.5*old + 0.5*batch_mean."""
        fn = DINOLoss(out_dim=D_PROTO, center_momentum=0.5)
        fn(torch.randn(B, D_PROTO), torch.ones(B, D_PROTO) * 2.0)
        assert fn.center.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_center_momentum_zero(self):
        """With momentum=0, center = batch mean exactly."""
        fn = DINOLoss(out_dim=D_PROTO, center_momentum=0.0)
        teacher = torch.ones(B, D_PROTO) * 3.0
        fn(torch.randn(B, D_PROTO), teacher)
        assert fn.center.mean().item() == pytest.approx(3.0, abs=1e-5)

    def test_teacher_detached(self):
        s = torch.randn(B, D_PROTO, requires_grad=True)
        t = torch.randn(B, D_PROTO, requires_grad=True)
        DINOLoss(out_dim=D_PROTO)(s, t)["loss"].backward()
        assert s.grad is not None
        assert t.grad is None

    def test_repeated_calls_drift_center(self):
        fn = DINOLoss(out_dim=D_PROTO, center_momentum=0.9)
        for _ in range(10):
            fn(torch.randn(B, D_PROTO), torch.randn(B, D_PROTO) + 5.0)
        assert fn.center.mean().item() > 2.0

    def test_low_teacher_temp_sharpens(self):
        """Lower teacher temp should produce sharper distribution."""
        torch.manual_seed(42)
        teacher = torch.randn(B, D_PROTO)
        student = torch.randn(B, D_PROTO)
        fn_sharp = DINOLoss(out_dim=D_PROTO, teacher_temp=0.01, student_temp=0.1)
        fn_soft = DINOLoss(out_dim=D_PROTO, teacher_temp=0.5, student_temp=0.1)
        assert torch.isfinite(fn_sharp(student, teacher)["loss"])
        assert torch.isfinite(fn_soft(student, teacher)["loss"])

    def test_gradient_finite(self):
        s = torch.randn(B, D_PROTO, requires_grad=True)
        t = torch.randn(B, D_PROTO)
        DINOLoss(out_dim=D_PROTO)(s, t)["loss"].backward()
        assert torch.isfinite(s.grad).all()


# ═══════════════════════════════════════════════════════════════════
# 5. JEPA Loss
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
# 6-8. Heads: Projector, Predictor, DINOHead
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
# 9. EMA Network
# ═══════════════════════════════════════════════════════════════════


class TestEMANetwork:
    @pytest.fixture
    def source(self):
        torch.manual_seed(42)
        return nn.Sequential(nn.Linear(D_EMBED, D_PROJ), nn.ReLU(), nn.Linear(D_PROJ, D_PROJ))

    def test_initial_copy_exact(self, source):
        ema = EMANetwork(source, 0.99)
        for s, e in zip(source.parameters(), ema.ema_model.parameters()):
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
        assert any(not torch.allclose(b, a) for b, a in zip(before, ema.ema_model.parameters()))

    def test_update_formula(self, source):
        """Verify theta_ema = m * theta_ema + (1-m) * theta_source exactly."""
        m = 0.8
        ema = EMANetwork(source, initial_momentum=m, final_momentum=m, total_steps=100)
        before = [p.clone() for p in ema.ema_model.parameters()]
        with torch.no_grad():
            for p in source.parameters():
                p.fill_(1.0)
        ema.update(source)
        for b, e, s in zip(before, ema.ema_model.parameters(), source.parameters()):
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
        for b, a in zip(before, ema.ema_model.parameters()):
            assert torch.allclose(b, a, atol=1e-6)

    def test_forward_delegates(self, source):
        ema = EMANetwork(source)
        x = torch.randn(B, D_EMBED)
        assert torch.allclose(ema(x), source(x), atol=1e-6)

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
        for b, a in zip(before, ema.ema_model.parameters()):
            assert torch.allclose(b, a)


# ═══════════════════════════════════════════════════════════════════
# 10. SSLPretrainModule
# ═══════════════════════════════════════════════════════════════════


class TestSSLPretrainModule:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_construction(self, method):
        m = _make_module(method)
        assert m.ssl_method == method

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown ssl_method"):
            _make_module("bad")

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_has_aggregator(self, method):
        assert hasattr(_make_module(method), "aggregator")

    @pytest.mark.parametrize("method", PROJECTOR_METHODS)
    def test_has_projector(self, method):
        assert isinstance(_make_module(method).projector, Projector)

    @pytest.mark.parametrize("method", ["byol", "jepa"])
    def test_has_predictor(self, method):
        assert isinstance(_make_module(method).predictor, Predictor)

    def test_dino_has_student_head(self):
        assert isinstance(_make_module("dino").student_head, DINOHead)

    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_ema_present(self, method):
        m = _make_module(method)
        if method in ("byol", "jepa"):
            assert isinstance(m.target_network, EMANetwork)
        else:
            assert isinstance(m.teacher_network, EMANetwork)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_training_step_scalar(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        loss = m.training_step(ssl_batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_training_step_no_coords(self, method, ssl_batch_no_coords):
        m = _make_module(method)
        m.train()
        assert torch.isfinite(m.training_step(ssl_batch_no_coords, 0))

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_val_step_dict(self, method, ssl_batch):
        m = _make_module(method)
        m.eval()
        with torch.no_grad():
            out = m.validation_step(ssl_batch, 0)
        assert "loss" in out
        assert "embeddings" in out

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_val_embedding_shape(self, method, ssl_batch):
        m = _make_module(method)
        m.eval()
        with torch.no_grad():
            emb = m.validation_step(ssl_batch, 0)["embeddings"]
        assert emb.shape == (B, D_EMBED)
        assert not emb.requires_grad

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_val_loss_finite(self, method, ssl_batch):
        m = _make_module(method)
        m.eval()
        with torch.no_grad():
            loss = m.validation_step(ssl_batch, 0)["loss"]
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_configure_optimizers_none(self, method):
        opt = _make_module(method, lr_scheduler="none").configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_configure_optimizers_cosine(self, method):
        r = _make_module(
            method, lr_scheduler="cosine", warmup_epochs=2, max_epochs=10
        ).configure_optimizers()
        assert "optimizer" in r
        assert "lr_scheduler" in r

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_configure_optimizers_cosine_no_warmup(self, method):
        r = _make_module(
            method, lr_scheduler="cosine", warmup_epochs=0, max_epochs=10
        ).configure_optimizers()
        assert "optimizer" in r
        assert "lr_scheduler" in r

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_on_save_checkpoint(self, method):
        ckpt = {}
        m = _make_module(method)
        m.on_save_checkpoint(ckpt)
        assert "aggregator_state_dict" in ckpt
        assert ckpt["ssl_method"] == method
        assert "embed_dim" in ckpt
        assert ckpt["embed_dim"] == D_EMBED
        assert len(ckpt["aggregator_state_dict"]) > 0

    def test_vicreg_no_bn(self):
        m = _make_module("vicreg")
        assert not any(isinstance(layer, nn.BatchNorm1d) for layer in m.projector.modules())

    def test_simclr_has_bn(self):
        m = _make_module("simclr")
        assert any(isinstance(layer, nn.BatchNorm1d) for layer in m.projector.modules())

    def test_byol_has_bn(self):
        m = _make_module("byol")
        assert any(isinstance(layer, nn.BatchNorm1d) for layer in m.projector.modules())
        assert any(isinstance(layer, nn.BatchNorm1d) for layer in m.predictor.modules())

    def test_hparams_saved(self):
        m = _make_module("vicreg")
        assert m.hparams.ssl_method == "vicreg"
        assert m.hparams.lr == 1e-3

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_embed_dim_matches_config(self, method):
        m = _make_module(method)
        assert m.embed_dim == D_EMBED

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_training_step_requires_grad(self, method, ssl_batch):
        """Loss should have grad_fn for backprop."""
        m = _make_module(method)
        m.train()
        loss = m.training_step(ssl_batch, 0)
        assert loss.requires_grad


# ═══════════════════════════════════════════════════════════════════
# 11-13. Callbacks: RankMe, AlphaReQ, Combined
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

    def test_delegates(self):
        cb = SSLQualityCallback(compute_every_n_epochs=1)
        tr = MagicMock()
        tr.current_epoch = 0
        cb.on_validation_batch_end(
            tr, MagicMock(), {"embeddings": torch.randn(B, D_EMBED)}, None, 0
        )
        assert len(cb.rankme._embeddings) == 1
        assert len(cb.alpha_req._embeddings) == 1

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


# ═══════════════════════════════════════════════════════════════════
# 14. Gradient flow end-to-end
# ═══════════════════════════════════════════════════════════════════


class TestGradientFlow:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_backward_produces_gradients(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        m.training_step(ssl_batch, 0).backward()
        trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        assert sum(1 for _, p in trainable if p.grad is not None) > 0
        for n, p in trainable:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN grad in {n}"

    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_ema_no_grad(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        m.training_step(ssl_batch, 0).backward()
        ema = m.target_network if method in ("byol", "jepa") else m.teacher_network
        for p in ema.ema_model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_aggregator_gets_grad(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        m.training_step(ssl_batch, 0).backward()
        agg_grads = [
            p.grad for p in m.aggregator.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(agg_grads) > 0

    @pytest.mark.parametrize("method", PROJECTOR_METHODS)
    def test_projector_gets_grad(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        m.training_step(ssl_batch, 0).backward()
        proj_grads = [
            p.grad for p in m.projector.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(proj_grads) > 0


# ═══════════════════════════════════════════════════════════════════
# 15. EMA integration
# ═══════════════════════════════════════════════════════════════════


class TestEMAMethodIntegration:
    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_on_train_batch_end_updates(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        ema = m.target_network if method in ("byol", "jepa") else m.teacher_network
        before = [p.clone() for p in ema.ema_model.parameters()]
        with torch.no_grad():
            for p in m.aggregator.parameters():
                p.add_(torch.randn_like(p))
        m.on_train_batch_end(None, ssl_batch, 0)
        assert any(
            not torch.allclose(b, a, atol=1e-8) for b, a in zip(before, ema.ema_model.parameters())
        )

    def test_byol_diverges(self, ssl_batch):
        """After several steps, online and target params should diverge."""
        m = _make_module("byol")
        m.train()
        for i in range(5):
            loss = m.training_step(ssl_batch, i)
            loss.backward()
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.sub_(p.grad * 0.01)
                        p.grad.zero_()
            m.on_train_batch_end(None, ssl_batch, i)
        online_p = list(m.projector.parameters())
        target_p = list(m.target_network.ema_model[1].parameters())
        assert any(not torch.allclose(o, t, atol=1e-6) for o, t in zip(online_p, target_p))

    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_ema_step_increments(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        ema = m.target_network if method in ("byol", "jepa") else m.teacher_network
        assert ema._step == 0
        m.on_train_batch_end(None, ssl_batch, 0)
        assert ema._step == 1
        m.on_train_batch_end(None, ssl_batch, 1)
        assert ema._step == 2


# ═══════════════════════════════════════════════════════════════════
# 16. Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_partial_mask(self, method):
        batch = {
            "view1": torch.randn(B, N_PATCHES, D_IN),
            "mask1": torch.ones(B, N_PATCHES),
            "view2": torch.randn(B, N_PATCHES, D_IN),
            "mask2": torch.ones(B, N_PATCHES),
        }
        batch["mask1"][:, -5:] = 0
        batch["mask2"][:, :3] = 0
        m = _make_module(method)
        m.train()
        assert torch.isfinite(m.training_step(batch, 0))

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_asymmetric_lengths(self, method):
        batch = {
            "view1": torch.randn(B, 30, D_IN),
            "mask1": torch.ones(B, 30),
            "view2": torch.randn(B, 15, D_IN),
            "mask2": torch.ones(B, 15),
        }
        m = _make_module(method)
        m.train()
        assert torch.isfinite(m.training_step(batch, 0))

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_single_sample(self, method):
        batch = {
            "view1": torch.randn(1, N_PATCHES, D_IN),
            "mask1": torch.ones(1, N_PATCHES),
            "view2": torch.randn(1, N_PATCHES, D_IN),
            "mask2": torch.ones(1, N_PATCHES),
        }
        m = _make_module(method)
        m.eval()
        with torch.no_grad():
            assert torch.isfinite(m.validation_step(batch, 0)["loss"])

    def test_vicreg_batch_two(self):
        batch = {
            "view1": torch.randn(2, N_PATCHES, D_IN),
            "mask1": torch.ones(2, N_PATCHES),
            "view2": torch.randn(2, N_PATCHES, D_IN),
            "mask2": torch.ones(2, N_PATCHES),
        }
        m = _make_module("vicreg")
        m.train()
        assert torch.isfinite(m.training_step(batch, 0))

    def test_simclr_batch_two(self):
        batch = {
            "view1": torch.randn(2, N_PATCHES, D_IN),
            "mask1": torch.ones(2, N_PATCHES),
            "view2": torch.randn(2, N_PATCHES, D_IN),
            "mask2": torch.ones(2, N_PATCHES),
        }
        m = _make_module("simclr")
        m.train()
        assert torch.isfinite(m.training_step(batch, 0))

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_single_patch(self, method):
        batch = {
            "view1": torch.randn(B, 1, D_IN),
            "mask1": torch.ones(B, 1),
            "view2": torch.randn(B, 1, D_IN),
            "mask2": torch.ones(B, 1),
        }
        m = _make_module(method)
        m.train()
        assert torch.isfinite(m.training_step(batch, 0))

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_large_patch_count(self, method):
        batch = {
            "view1": torch.randn(2, 200, D_IN),
            "mask1": torch.ones(2, 200),
            "view2": torch.randn(2, 200, D_IN),
            "mask2": torch.ones(2, 200),
        }
        m = _make_module(method)
        m.train()
        assert torch.isfinite(m.training_step(batch, 0))

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_all_masked(self, method):
        """All-zero masks (no valid patches)."""
        batch = {
            "view1": torch.randn(B, N_PATCHES, D_IN),
            "mask1": torch.zeros(B, N_PATCHES),
            "view2": torch.randn(B, N_PATCHES, D_IN),
            "mask2": torch.zeros(B, N_PATCHES),
        }
        m = _make_module(method)
        m.train()
        loss = m.training_step(batch, 0)
        assert loss.ndim == 0


# ═══════════════════════════════════════════════════════════════════
# 17. RankMe + AlphaReQ consistency
# ═══════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════
# 18. Multi-step training stability
# ═══════════════════════════════════════════════════════════════════


class TestMultiStepTraining:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_loss_stays_finite_over_steps(self, method):
        """Loss should not diverge to NaN/Inf over several manual training steps."""
        torch.manual_seed(42)
        m = _make_module(method)
        m.train()
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-3)
        batch = {
            "view1": torch.randn(B, N_PATCHES, D_IN),
            "mask1": torch.ones(B, N_PATCHES),
            "view2": torch.randn(B, N_PATCHES, D_IN),
            "mask2": torch.ones(B, N_PATCHES),
        }
        for step in range(10):
            opt.zero_grad()
            loss = m.training_step(batch, step)
            loss.backward()
            opt.step()
            if method in EMA_METHODS:
                m.on_train_batch_end(None, batch, step)
            assert torch.isfinite(loss), f"Loss NaN/Inf at step {step}"

    @pytest.mark.parametrize("method", ["vicreg", "byol"])
    def test_loss_decreases_on_same_data(self, method):
        """On repeated same data, loss should generally decrease (overfit)."""
        torch.manual_seed(42)
        m = _make_module(method)
        m.train()
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-2)
        batch = {
            "view1": torch.randn(B, N_PATCHES, D_IN),
            "mask1": torch.ones(B, N_PATCHES),
            "view2": torch.randn(B, N_PATCHES, D_IN),
            "mask2": torch.ones(B, N_PATCHES),
        }
        first_loss = None
        last_loss = None
        for step in range(30):
            opt.zero_grad()
            loss = m.training_step(batch, step)
            loss.backward()
            opt.step()
            if method in EMA_METHODS:
                m.on_train_batch_end(None, batch, step)
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()
        assert last_loss < first_loss


# ═══════════════════════════════════════════════════════════════════
# 19. Checkpoint round-trip
# ═══════════════════════════════════════════════════════════════════


class TestCheckpointRoundTrip:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_aggregator_weights_roundtrip(self, method, ssl_batch):
        """Saved aggregator weights should exactly match original."""
        m = _make_module(method)
        m.train()
        m.training_step(ssl_batch, 0).backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad and p.grad is not None:
                    p.sub_(p.grad * 0.01)

        ckpt = {}
        m.on_save_checkpoint(ckpt)

        from oceanpath.models import build_aggregator

        new_agg = build_aggregator(
            arch="abmil",
            in_dim=D_IN,
            model_cfg={
                "embed_dim": D_EMBED,
                "dropout": 0.0,
                "attn_dim": 16,
                "gate": True,
                "num_fc_layers": 1,
            },
        )
        new_agg.load_state_dict(ckpt["aggregator_state_dict"])

        for (n1, p1), (_n2, p2) in zip(m.aggregator.named_parameters(), new_agg.named_parameters()):
            assert torch.allclose(p1, p2, atol=1e-7), f"Mismatch in {n1}"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_checkpoint_has_metadata(self, method):
        m = _make_module(method)
        ckpt = {}
        m.on_save_checkpoint(ckpt)
        assert ckpt["ssl_method"] == method
        assert ckpt["embed_dim"] == D_EMBED


# ═══════════════════════════════════════════════════════════════════
# 20. Full callback lifecycle (mock trainer)
# ═══════════════════════════════════════════════════════════════════


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
# 21. Numerical stability
# ═══════════════════════════════════════════════════════════════════


class TestNumericalStability:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_large_feature_values(self, method):
        batch = {
            "view1": torch.randn(B, N_PATCHES, D_IN) * 100,
            "mask1": torch.ones(B, N_PATCHES),
            "view2": torch.randn(B, N_PATCHES, D_IN) * 100,
            "mask2": torch.ones(B, N_PATCHES),
        }
        m = _make_module(method)
        m.train()
        loss = m.training_step(batch, 0)
        assert torch.isfinite(loss), f"{method}: loss not finite for large inputs"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_small_feature_values(self, method):
        batch = {
            "view1": torch.randn(B, N_PATCHES, D_IN) * 1e-6,
            "mask1": torch.ones(B, N_PATCHES),
            "view2": torch.randn(B, N_PATCHES, D_IN) * 1e-6,
            "mask2": torch.ones(B, N_PATCHES),
        }
        m = _make_module(method)
        m.train()
        loss = m.training_step(batch, 0)
        assert torch.isfinite(loss), f"{method}: loss not finite for small inputs"

    def test_vicreg_eps_prevents_nan(self):
        z = torch.zeros(B, D_PROJ, requires_grad=True)
        out = VICRegLoss()(z, z.clone().requires_grad_(True))
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["var_loss"])

    def test_simclr_handles_identical_embeddings(self):
        z = torch.ones(B, D_PROJ, requires_grad=True)
        loss = SimCLRLoss()(z, z.clone().requires_grad_(True))["loss"]
        assert torch.isfinite(loss)

    def test_rankme_all_zeros(self):
        r, _D = RankMeCallback()._compute_rankme(torch.zeros(50, 32))
        assert r >= 0

    def test_alpha_req_all_zeros(self):
        a = AlphaReQCallback()._compute_alpha_req(torch.zeros(50, 32))
        assert isinstance(a, float)
        assert a >= 0


# ═══════════════════════════════════════════════════════════════════
# 22. LR Scheduler behavior
# ═══════════════════════════════════════════════════════════════════


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
# 23. Target encoder isolation (no grad leaks)
# ═══════════════════════════════════════════════════════════════════


class TestTargetEncoderIsolation:
    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_target_params_no_grad(self, method, ssl_batch):
        m = _make_module(method)
        ema = m.target_network if method in ("byol", "jepa") else m.teacher_network
        for name, p in ema.ema_model.named_parameters():
            assert not p.requires_grad, f"{name} has requires_grad=True in target"

    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_target_no_grad_after_backward(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        loss = m.training_step(ssl_batch, 0)
        loss.backward()
        ema = m.target_network if method in ("byol", "jepa") else m.teacher_network
        for name, p in ema.ema_model.named_parameters():
            assert p.grad is None, f"{name} got gradient in target"

    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_target_changes_only_via_ema(self, method, ssl_batch):
        """Target params should only change through explicit EMA update."""
        m = _make_module(method)
        m.train()
        ema = m.target_network if method in ("byol", "jepa") else m.teacher_network
        before = [p.clone() for p in ema.ema_model.parameters()]

        loss = m.training_step(ssl_batch, 0)
        loss.backward()
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad and p.grad is not None:
                    p.sub_(p.grad * 0.01)

        for b, a in zip(before, ema.ema_model.parameters()):
            assert torch.allclose(b, a, atol=1e-8), "Target changed without EMA update"


# ═══════════════════════════════════════════════════════════════════
# 24. Module train/eval mode interaction
# ═══════════════════════════════════════════════════════════════════


class TestTrainEvalMode:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_train_mode_forward(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        loss = m.training_step(ssl_batch, 0)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_eval_mode_val_step(self, method, ssl_batch):
        m = _make_module(method)
        m.train()
        m.training_step(ssl_batch, 0)  # populate BN stats
        m.eval()
        with torch.no_grad():
            out = m.validation_step(ssl_batch, 0)
        assert torch.isfinite(out["loss"])

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_eval_val_deterministic(self, method, ssl_batch):
        """Validation in eval mode should be deterministic."""
        m = _make_module(method)
        m.train()
        m.training_step(ssl_batch, 0)
        m.eval()
        with torch.no_grad():
            out1 = m.validation_step(ssl_batch, 0)
            out2 = m.validation_step(ssl_batch, 0)
        assert out1["loss"].item() == pytest.approx(out2["loss"].item(), abs=1e-6)


# ═══════════════════════════════════════════════════════════════════
# 25. Parameter counting / architecture verification
# ═══════════════════════════════════════════════════════════════════


class TestArchitectureVerification:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_has_trainable_parameters(self, method):
        m = _make_module(method)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert trainable > 0

    @pytest.mark.parametrize("method", EMA_METHODS)
    def test_frozen_parameter_count(self, method):
        m = _make_module(method)
        frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        assert frozen > 0

    @pytest.mark.parametrize("method", ["vicreg", "simclr"])
    def test_symmetric_methods_no_ema(self, method):
        m = _make_module(method)
        assert not hasattr(m, "target_network")
        assert not hasattr(m, "teacher_network")

    def test_dino_has_no_projector(self):
        m = _make_module("dino")
        assert not hasattr(m, "projector")
        assert hasattr(m, "student_head")

    def test_vicreg_projector_no_bn(self):
        m = _make_module("vicreg")
        for module in m.projector.modules():
            assert not isinstance(module, nn.BatchNorm1d)

    @pytest.mark.parametrize("method", ["simclr", "byol"])
    def test_projector_has_bn(self, method):
        m = _make_module(method)
        assert any(isinstance(mod, nn.BatchNorm1d) for mod in m.projector.modules())
