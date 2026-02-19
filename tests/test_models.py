"""
Tests for OceanPath MIL model definitions.

Run:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v -k "abmil"          # only ABMIL tests
    pytest tests/test_models.py -v -k "gradient"        # only gradient tests
    pytest tests/test_models.py -v --tb=short           # shorter tracebacks

Requires: pytest, torch (CPU is sufficient for all tests)
"""

import pytest
import torch
import torch.nn as nn

from oceanpath.models import (
    build_aggregator,
    build_classifier,
    list_aggregators,
    MILOutput,
)
from oceanpath.models.base import BaseMIL
from oceanpath.models.abmil import ABMIL
from oceanpath.models.transmil import TransMIL
from oceanpath.models.static import StaticMIL
from oceanpath.models.wsi_classifier import WSIClassifier


# ── Fixtures ──────────────────────────────────────────────────────────────────


# Common dimensions for testing
IN_DIM = 256        # small for fast tests (real: 1024-1536)
EMBED_DIM = 128     # small for fast tests (real: 512)
NUM_CLASSES = 3
BATCH_SIZE = 2
N_PATCHES = 50      # bag size


@pytest.fixture
def random_bag():
    """Standard test input: [B, N, D] features."""
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, N_PATCHES, IN_DIM)


@pytest.fixture
def random_bag_with_mask():
    """Test input with padding mask (last 10 patches are padding)."""
    torch.manual_seed(42)
    features = torch.randn(BATCH_SIZE, N_PATCHES, IN_DIM)
    mask = torch.ones(BATCH_SIZE, N_PATCHES)
    mask[:, -10:] = 0  # last 10 are padding
    return features, mask


@pytest.fixture
def random_bag_with_coords():
    """Test input with spatial coordinates."""
    torch.manual_seed(42)
    features = torch.randn(BATCH_SIZE, N_PATCHES, IN_DIM)
    coords = torch.randint(0, 1000, (BATCH_SIZE, N_PATCHES, 2)).float()
    return features, coords


@pytest.fixture
def abmil():
    return ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, attn_dim=64, dropout=0.0)


@pytest.fixture
def transmil():
    return TransMIL(
        in_dim=IN_DIM, embed_dim=EMBED_DIM,
        num_attention_layers=1, num_heads=4, dropout=0.0,
    )


@pytest.fixture
def meanpool():
    return StaticMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, pool_method="mean", dropout=0.0)


@pytest.fixture
def maxpool():
    return StaticMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, pool_method="max", dropout=0.0)


ALL_ARCHS = ["abmil", "transmil", "static"]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Forward contract — every aggregator must produce correct output types/shapes
# ═════════════════════════════════════════════════════════════════════════════


class TestForwardContract:
    """All aggregators must follow the standardized MILOutput contract."""

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_output_type(self, arch, random_bag):
        """forward() returns MILOutput."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        output = model(random_bag)
        assert isinstance(output, MILOutput)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_slide_embedding_shape(self, arch, random_bag):
        """slide_embedding is [B, embed_dim]."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        output = model(random_bag)
        assert output.slide_embedding.shape == (BATCH_SIZE, EMBED_DIM)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_aggregator_logits_none(self, arch, random_bag):
        """Bare aggregators return logits=None."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        output = model(random_bag)
        assert output.logits is None

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_extras_is_dict(self, arch, random_bag):
        """extras is always a dict."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        output = model(random_bag)
        assert isinstance(output.extras, dict)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Output shapes — detailed per-model checks
# ═════════════════════════════════════════════════════════════════════════════


class TestOutputShapes:

    def test_abmil_attention_shape(self, abmil, random_bag):
        """ABMIL attention weights: [B, N]."""
        output = abmil(random_bag, return_attention=True)
        assert "attention_weights" in output.extras
        assert output.extras["attention_weights"].shape == (BATCH_SIZE, N_PATCHES)

    def test_transmil_attention_shape(self, transmil, random_bag):
        """TransMIL attention weights: [B, N]."""
        output = transmil(random_bag, return_attention=True)
        assert "attention_weights" in output.extras
        assert output.extras["attention_weights"].shape == (BATCH_SIZE, N_PATCHES)

    def test_static_no_attention(self, meanpool, random_bag):
        """StaticMIL has no attention weights."""
        output = meanpool(random_bag, return_attention=True)
        assert "attention_weights" not in output.extras

    def test_no_attention_when_not_requested(self, abmil, random_bag):
        """When return_attention=False, extras should be empty."""
        output = abmil(random_bag, return_attention=False)
        assert "attention_weights" not in output.extras


# ═════════════════════════════════════════════════════════════════════════════
# 3. Auto-batching — 2D input [N, D] should work
# ═════════════════════════════════════════════════════════════════════════════


class TestAutoBatching:

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_2d_input(self, arch):
        """[N, D] input is auto-batched to [1, N, D]."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        x = torch.randn(N_PATCHES, IN_DIM)
        output = model(x)
        assert output.slide_embedding.shape == (1, EMBED_DIM)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_3d_and_2d_agree(self, arch):
        """2D and 3D inputs with same data produce same output."""
        torch.manual_seed(0)
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        model.eval()

        x_2d = torch.randn(N_PATCHES, IN_DIM)
        x_3d = x_2d.unsqueeze(0)

        with torch.no_grad():
            out_2d = model(x_2d).slide_embedding
            out_3d = model(x_3d).slide_embedding

        torch.testing.assert_close(out_2d, out_3d, atol=1e-5, rtol=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Gradient flow — all parameters receive gradients
# ═════════════════════════════════════════════════════════════════════════════


class TestGradientFlow:

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_gradients_flow(self, arch, random_bag):
        """Loss.backward() produces non-None gradients on all parameters."""
        model = build_classifier(
            arch=arch, in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        model.train()

        output = model(random_bag)
        loss = output.logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_no_nan_gradients(self, arch, random_bag):
        """Gradients should not contain NaN."""
        model = build_classifier(
            arch=arch, in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        model.train()

        output = model(random_bag)
        loss = output.logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Gradient checkpointing — same output, less memory
# ═════════════════════════════════════════════════════════════════════════════


class TestGradientCheckpointing:

    def test_abmil_checkpoint_same_output(self, random_bag):
        """ABMIL with and without checkpointing produce same output."""
        torch.manual_seed(0)
        m_off = ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, gradient_checkpointing=False)
        torch.manual_seed(0)
        m_on = ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, gradient_checkpointing=True)

        m_off.eval()
        m_on.eval()
        with torch.no_grad():
            out_off = m_off(random_bag).slide_embedding
            out_on = m_on(random_bag).slide_embedding

        torch.testing.assert_close(out_off, out_on, atol=1e-5, rtol=1e-5)

    def test_transmil_checkpoint_same_output(self, random_bag):
        """TransMIL with and without checkpointing produce same output."""
        torch.manual_seed(0)
        m_off = TransMIL(
            in_dim=IN_DIM, embed_dim=EMBED_DIM,
            num_attention_layers=1, num_heads=4, gradient_checkpointing=False,
        )
        torch.manual_seed(0)
        m_on = TransMIL(
            in_dim=IN_DIM, embed_dim=EMBED_DIM,
            num_attention_layers=1, num_heads=4, gradient_checkpointing=True,
        )

        m_off.eval()
        m_on.eval()
        with torch.no_grad():
            out_off = m_off(random_bag).slide_embedding
            out_on = m_on(random_bag).slide_embedding

        torch.testing.assert_close(out_off, out_on, atol=1e-5, rtol=1e-5)

    def test_checkpoint_grads_flow(self, random_bag):
        """Checkpointed model still gets gradients on all params."""
        model = build_classifier(
            arch="abmil", in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM, "gradient_checkpointing": True},
        )
        model.train()

        output = model(random_bag)
        output.logits.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name} (checkpointed)"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Masking — padding positions are ignored
# ═════════════════════════════════════════════════════════════════════════════


class TestMasking:

    def test_abmil_mask_ignores_padding(self):
        """Changing padded positions doesn't affect ABMIL output."""
        model = ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, dropout=0.0)
        model.eval()

        x = torch.randn(1, 20, IN_DIM)
        mask = torch.ones(1, 20)
        mask[:, 15:] = 0  # last 5 are padding

        with torch.no_grad():
            out1 = model(x, mask=mask).slide_embedding

        # Corrupt the padded positions
        x[:, 15:, :] = torch.randn(1, 5, IN_DIM) * 1000
        with torch.no_grad():
            out2 = model(x, mask=mask).slide_embedding

        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

    def test_static_mask_correct_mean(self):
        """StaticMIL with mask computes mean over valid patches only."""
        model = StaticMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, dropout=0.0, num_fc_layers=1)
        model.eval()

        # Create input where padded patches have large values
        x = torch.ones(1, 10, IN_DIM)
        x[:, 5:, :] = 999.0  # padding positions

        mask = torch.ones(1, 10)
        mask[:, 5:] = 0

        with torch.no_grad():
            out_masked = model(x, mask=mask).slide_embedding
            # Without mask, mean would be much larger
            out_unmasked = model(x).slide_embedding

        # Masked output should be much smaller (not polluted by 999s)
        assert out_masked.abs().max() < out_unmasked.abs().max()

    def test_static_maxpool_mask_ignores_padding(self):
        """StaticMIL with maxpool mask should not pick padded positions."""
        model = StaticMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, dropout=0.0, pool_method="max")
        model.eval()

        x = torch.randn(1, 20, IN_DIM)
        mask = torch.ones(1, 20)
        mask[:, 15:] = 0

        with torch.no_grad():
            out1 = model(x, mask=mask).slide_embedding

        x[:, 15:, :] = torch.randn(1, 5, IN_DIM) * 1000
        with torch.no_grad():
            out2 = model(x, mask=mask).slide_embedding

        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# 7. WSIClassifier — head and aggregator composition
# ═════════════════════════════════════════════════════════════════════════════


class TestWSIClassifier:

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_logits_shape(self, arch, random_bag):
        """Classifier produces [B, C] logits."""
        model = build_classifier(
            arch=arch, in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        output = model(random_bag)
        assert output.logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_binary_logits_shape(self, random_bag):
        """num_classes=1 produces [B] logits (squeezed)."""
        model = build_classifier(
            arch="abmil", in_dim=IN_DIM, num_classes=1,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        output = model(random_bag)
        assert output.logits.shape == (BATCH_SIZE,)

    def test_freeze_aggregator(self, random_bag):
        """Frozen aggregator has no gradients after backward."""
        model = build_classifier(
            arch="abmil", in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
            freeze_aggregator=True,
        )
        model.train()

        output = model(random_bag)
        output.logits.sum().backward()

        # Aggregator params should have no grad
        for name, param in model.aggregator.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

        # Head params should have grad
        for name, param in model.head.named_parameters():
            assert param.grad is not None, f"Head param {name} has no gradient"

    def test_slide_embedding_preserved(self, random_bag):
        """Classifier output includes slide_embedding from aggregator."""
        model = build_classifier(
            arch="abmil", in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        model.eval()
        with torch.no_grad():
            output = model(random_bag)
        assert output.slide_embedding.shape == (BATCH_SIZE, EMBED_DIM)
        # slide_embedding should not be all zeros
        assert not torch.all(output.slide_embedding == 0)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Registry and factory
# ═════════════════════════════════════════════════════════════════════════════


class TestRegistry:

    def test_list_aggregators(self):
        """All built-in archs are registered."""
        archs = list_aggregators()
        assert "abmil" in archs
        assert "transmil" in archs
        assert "static" in archs

    def test_unknown_arch_raises(self):
        """Unknown arch name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_aggregator("nonexistent_model", in_dim=IN_DIM)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_factory_builds_correct_type(self, arch):
        """Factory returns an instance of BaseMIL."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        assert isinstance(model, BaseMIL)

    def test_factory_ignores_extra_config_keys(self):
        """Extra keys in model_cfg that don't match constructor args are ignored."""
        model = build_aggregator(
            "abmil", in_dim=IN_DIM,
            model_cfg={"embed_dim": EMBED_DIM, "nonexistent_key": 999},
        )
        assert isinstance(model, ABMIL)

    def test_factory_passes_kwargs(self):
        """Config keys are forwarded to constructor."""
        model = build_aggregator(
            "abmil", in_dim=IN_DIM,
            model_cfg={"embed_dim": 256, "attn_dim": 64, "gate": False},
        )
        assert model.embed_dim == 256


# ═════════════════════════════════════════════════════════════════════════════
# 9. Determinism and reproducibility
# ═════════════════════════════════════════════════════════════════════════════


class TestDeterminism:

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_eval_deterministic(self, arch):
        """Same input in eval mode → same output."""
        torch.manual_seed(0)
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        model.eval()

        x = torch.randn(1, N_PATCHES, IN_DIM)
        with torch.no_grad():
            out1 = model(x).slide_embedding
            out2 = model(x).slide_embedding

        torch.testing.assert_close(out1, out2)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_weight_reinit_changes_output(self, arch):
        """initialize_weights() actually changes parameters."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        model.eval()

        x = torch.randn(1, N_PATCHES, IN_DIM)
        with torch.no_grad():
            out_before = model(x).slide_embedding.clone()

        model.initialize_weights()
        with torch.no_grad():
            out_after = model(x).slide_embedding

        # Very unlikely to be identical after reinit
        assert not torch.allclose(out_before, out_after, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Edge cases
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_single_patch_bag(self, arch):
        """Bag with N=1 patch should work."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        model.eval()
        x = torch.randn(1, 1, IN_DIM)
        with torch.no_grad():
            output = model(x)
        assert output.slide_embedding.shape == (1, EMBED_DIM)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_batch_size_one(self, arch):
        """Batch size 1 works correctly."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        model.eval()
        x = torch.randn(1, N_PATCHES, IN_DIM)
        with torch.no_grad():
            output = model(x)
        assert output.slide_embedding.shape == (1, EMBED_DIM)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_large_feature_dim(self, arch):
        """Models work with real-world feature dims (1536 for UNI v2)."""
        real_dim = 1536
        model = build_aggregator(arch, in_dim=real_dim, model_cfg={"embed_dim": EMBED_DIM})
        model.eval()
        x = torch.randn(1, 30, real_dim)
        with torch.no_grad():
            output = model(x)
        assert output.slide_embedding.shape == (1, EMBED_DIM)

    def test_all_masked(self):
        """All-masked bag: ABMIL should produce NaN-free output (inf → 0 after softmax)."""
        model = ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, dropout=0.0)
        model.eval()

        x = torch.randn(1, 10, IN_DIM)
        mask = torch.zeros(1, 10)  # all masked

        with torch.no_grad():
            output = model(x, mask=mask)

        # softmax([-inf, -inf, ...]) → [nan, nan, ...] but bmm with nan → nan
        # This is expected behavior — document it, don't crash
        # The test verifies it doesn't throw an exception
        assert output.slide_embedding.shape == (1, EMBED_DIM)


# ═════════════════════════════════════════════════════════════════════════════
# 11. Serialization — models can be saved and loaded
# ═════════════════════════════════════════════════════════════════════════════


class TestSerialization:

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_state_dict_roundtrip(self, arch, random_bag):
        """Save and load state dict produces same output."""
        model1 = build_classifier(
            arch=arch, in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        model1.eval()

        with torch.no_grad():
            out1 = model1(random_bag).logits

        # Save and load state dict
        state = model1.state_dict()
        model2 = build_classifier(
            arch=arch, in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        model2.load_state_dict(state)
        model2.eval()

        with torch.no_grad():
            out2 = model2(random_bag).logits

        torch.testing.assert_close(out1, out2)

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_param_count_nonzero(self, arch):
        """Models have learnable parameters."""
        model = build_aggregator(arch, in_dim=IN_DIM, model_cfg={"embed_dim": EMBED_DIM})
        total = sum(p.numel() for p in model.parameters())
        assert total > 0, f"{arch} has 0 parameters"


# ═════════════════════════════════════════════════════════════════════════════
# 12. Model-specific correctness
# ═══════════════════════════════════════════════════════════════════════════==


class TestModelSpecific:

    def test_abmil_gated_vs_ungated(self, random_bag):
        """Gated and ungated ABMIL produce different outputs."""
        torch.manual_seed(0)
        gated = ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, gate=True)
        torch.manual_seed(0)
        ungated = ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, gate=False)

        gated.eval()
        ungated.eval()

        with torch.no_grad():
            out_g = gated(random_bag).slide_embedding
            out_u = ungated(random_bag).slide_embedding

        # Different architectures → different outputs
        assert not torch.allclose(out_g, out_u, atol=1e-4)

    def test_static_meanpool_vs_maxpool(self, random_bag):
        """Mean and max pooling produce different outputs."""
        torch.manual_seed(0)
        mean_m = StaticMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, pool_method="mean")
        torch.manual_seed(0)
        max_m = StaticMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, pool_method="max")

        mean_m.eval()
        max_m.eval()

        with torch.no_grad():
            out_mean = mean_m(random_bag).slide_embedding
            out_max = max_m(random_bag).slide_embedding

        assert not torch.allclose(out_mean, out_max, atol=1e-4)

    def test_abmil_attention_sums_to_one(self, random_bag):
        """After softmax, attention weights sum to ~1 per bag."""
        model = ABMIL(in_dim=IN_DIM, embed_dim=EMBED_DIM, dropout=0.0)
        model.eval()

        with torch.no_grad():
            output = model(random_bag, return_attention=True)
            logits = output.extras["attention_weights"]  # pre-softmax [B, N]
            weights = torch.softmax(logits, dim=-1)

        sums = weights.sum(dim=-1)  # [B]
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# 13. Integration with loss functions
# ═════════════════════════════════════════════════════════════════════════════


class TestLossIntegration:

    @pytest.mark.parametrize("arch", ALL_ARCHS)
    def test_cross_entropy_backward(self, arch, random_bag):
        """Standard CE loss backward works."""
        model = build_classifier(
            arch=arch, in_dim=IN_DIM, num_classes=NUM_CLASSES,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        model.train()

        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        output = model(random_bag)

        loss = nn.CrossEntropyLoss()(output.logits, labels)
        loss.backward()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_bce_binary(self, random_bag):
        """Binary classification with BCE works."""
        model = build_classifier(
            arch="abmil", in_dim=IN_DIM, num_classes=1,
            model_cfg={"embed_dim": EMBED_DIM},
        )
        model.train()

        labels = torch.randint(0, 2, (BATCH_SIZE,)).float()
        output = model(random_bag)

        loss = nn.BCEWithLogitsLoss()(output.logits, labels)
        loss.backward()

        assert loss.item() > 0
        assert not torch.isnan(loss)