"""
Tests for the TITAN-style slide transformer with 2D ALiBi
(``oceanpath.models.vit``), registered as the "vit" aggregator.

Covers:
  * the forward contract: embedding shape for batched and unbatched bags, the
    [B, N] CLS-attention map, a classifier built from configs/model/vit.yaml
  * masking: padded patches change nothing, with and without coordinates, for
    the qk_norm / LayerScale variants
  * coordinates: permuting patches together with their coordinates, shifting
    the grid, or giving pixel coordinates with coord_scale leaves the embedding
    unchanged; the coordinates themselves do change it
  * the explicit-attention path (return_attention) equals the fused SDPA path
  * gradients reach every parameter; gradient checkpointing gives the same ones
  * ALiBi: the slope schedule, bias = -slope * distance with a free CLS row and
    column, masked keys pushed to the fp16-safe floor
"""

import copy
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from oceanpath.models import build_classifier
from oceanpath.models.vit import TITANViT, _get_alibi_slopes, compute_2d_alibi_bias

IN_DIM = 16
MASK_VALUE = -65504.0
VIT_YAML = Path(__file__).resolve().parents[1] / "configs" / "model" / "vit.yaml"


def make_vit(**kwargs) -> TITANViT:
    torch.manual_seed(0)
    defaults = {
        "in_dim": IN_DIM,
        "embed_dim": 16,
        "depth": 2,
        "num_heads": 4,
        "drop_path_rate": 0.0,
        "gradient_checkpointing": False,
    }
    defaults.update(kwargs)
    return TITANViT(**defaults).eval()


def make_bag(n=10, batch=1, seed=0):
    """Features and distinct grid coordinates (in grid units)."""
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, n, IN_DIM, generator=gen)
    cells = torch.stack([torch.randperm(64, generator=gen)[:n] for _ in range(batch)])
    coords = torch.stack([cells // 8, cells % 8], dim=-1).float()
    return features, coords


def embed(model, features, **kwargs):
    with torch.no_grad():
        return model(features, **kwargs).slide_embedding


# ═════════════════════════════════════════════════════════════════════════════
# Forward contract
# ═════════════════════════════════════════════════════════════════════════════


class TestForwardContract:
    def test_embedding_shape_for_batched_and_unbatched_bags(self):
        model = make_vit()
        features, coords = make_bag(n=12, batch=3)
        out = model(features, coords=coords)
        assert out.slide_embedding.shape == (3, 16)
        assert out.logits is None
        assert out.extras == {}
        single = embed(model, features[1], coords=coords[1])
        assert single.shape == (1, 16)
        torch.testing.assert_close(single[0], out.slide_embedding[1].detach())

    def test_attention_map_is_the_cls_row_over_real_patches(self):
        model = make_vit()
        features, coords = make_bag(n=9, batch=2)
        mask = torch.ones(2, 9)
        mask[1, 6:] = 0
        with torch.no_grad():
            attn = model(features, mask=mask, coords=coords, return_attention=True).extras[
                "attention_weights"
            ]
        assert attn.shape == (2, 9)
        assert (attn >= 0).all()
        total = attn.sum(dim=-1)  # the CLS token keeps the rest for itself
        assert ((total > 0) & (total < 1)).all()
        assert attn[1, 6:].abs().max() < 1e-12

    def test_classifier_from_the_shipped_vit_config(self):
        cfg = OmegaConf.to_container(OmegaConf.load(VIT_YAML))
        cfg.update(embed_dim=16, depth=2, num_heads=4)
        torch.manual_seed(0)
        classifier = build_classifier("vit", in_dim=IN_DIM, num_classes=3, model_cfg=cfg)
        assert isinstance(classifier.aggregator, TITANViT)
        assert classifier.aggregator.coord_scale == cfg["coord_scale"]
        features, coords = make_bag(n=7, batch=2)
        classifier.train()  # gradient checkpointing (on in the config) only runs in training
        out = classifier(features, coords=coords * cfg["coord_scale"])
        assert out.logits.shape == (2, 3)
        out.logits.sum().backward()
        assert classifier.aggregator.cls_token.grad is not None
        assert classifier.aggregator.no_weight_decay() == {"cls_token"}


# ═════════════════════════════════════════════════════════════════════════════
# Masking and coordinates
# ═════════════════════════════════════════════════════════════════════════════


VARIANTS = [{}, {"qk_norm": True}, {"init_values": 1e-1}]


class TestMaskAndCoords:
    @pytest.mark.parametrize("variant", VARIANTS, ids=["plain", "qk_norm", "layerscale"])
    @pytest.mark.parametrize("use_coords", [True, False], ids=["coords", "no_coords"])
    def test_padding_under_the_mask_changes_nothing(self, variant, use_coords):
        model = make_vit(**variant)
        features, coords = make_bag(n=10)
        padded = torch.cat([features, 50 * torch.randn(1, 4, IN_DIM)], dim=1)
        padded_coords = torch.cat([coords, torch.full((1, 4, 2), 7.0)], dim=1)
        mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 4)], dim=1)
        ref = embed(model, features, coords=coords if use_coords else None)
        out = embed(model, padded, mask=mask, coords=padded_coords if use_coords else None)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)

    def test_an_all_ones_mask_equals_no_mask(self):
        model = make_vit()
        features, coords = make_bag(n=8)
        torch.testing.assert_close(
            embed(model, features, mask=torch.ones(1, 8), coords=coords),
            embed(model, features, coords=coords),
        )

    def test_permuting_patches_with_their_coords_changes_nothing(self):
        model = make_vit()
        features, coords = make_bag(n=11)
        perm = torch.randperm(11, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(
            embed(model, features[:, perm], coords=coords[:, perm]),
            embed(model, features, coords=coords),
            atol=1e-5,
            rtol=1e-4,
        )

    def test_shifting_the_grid_changes_nothing(self):
        model = make_vit()
        features, coords = make_bag(n=11)
        shifted = coords + torch.tensor([40.0, 7.0])
        torch.testing.assert_close(
            embed(model, features, coords=shifted), embed(model, features, coords=coords)
        )

    def test_coord_scale_turns_pixel_coordinates_into_grid_units(self):
        grid_model = make_vit()
        pixel_model = make_vit(coord_scale=256.0)
        pixel_model.load_state_dict(grid_model.state_dict())
        features, coords = make_bag(n=9)
        torch.testing.assert_close(
            embed(pixel_model, features, coords=coords * 256),
            embed(grid_model, features, coords=coords),
        )

    def test_coordinates_change_the_embedding(self):
        model = make_vit()
        features, coords = make_bag(n=9)
        with_coords = embed(model, features, coords=coords)
        assert not torch.allclose(with_coords, embed(model, features), atol=1e-4)
        assert not torch.allclose(with_coords, embed(model, features, coords=coords * 3), atol=1e-4)

    def test_explicit_attention_path_matches_the_fused_path(self):
        model = make_vit(depth=3)
        features, coords = make_bag(n=10, batch=2)
        mask = torch.ones(2, 10)
        mask[0, 7:] = 0
        with torch.no_grad():
            fused = model(features, mask=mask, coords=coords).slide_embedding
            explicit = model(features, mask=mask, coords=coords, return_attention=True)
        torch.testing.assert_close(explicit.slide_embedding, fused, atol=1e-5, rtol=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# Gradients
# ═════════════════════════════════════════════════════════════════════════════


class TestGradients:
    def test_every_parameter_gets_a_finite_gradient(self):
        model = make_vit(init_values=1e-1, qk_norm=True).train()
        features, coords = make_bag(n=10, batch=2)
        model(features, coords=coords).slide_embedding.pow(2).sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, name

    def test_gradient_checkpointing_gives_the_same_gradients(self):
        plain = make_vit().train()
        checkpointed = copy.deepcopy(plain)
        checkpointed.gradient_checkpointing = True
        features, coords = make_bag(n=10, batch=2)
        mask = torch.ones(2, 10)
        mask[1, 8:] = 0
        for model in (plain, checkpointed):
            model(features, mask=mask, coords=coords).slide_embedding.pow(2).sum().backward()
        for (name, a), (_, b) in zip(
            plain.named_parameters(), checkpointed.named_parameters(), strict=True
        ):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5, msg=name)


# ═════════════════════════════════════════════════════════════════════════════
# 2D ALiBi
# ═════════════════════════════════════════════════════════════════════════════


class TestAlibi:
    @pytest.mark.parametrize(
        "n_heads, exponents",
        [
            (8, [1, 2, 3, 4, 5, 6, 7, 8]),  # 2^(-8/n * h), h = 1..n
            (4, [2, 4, 6, 8]),
            (6, [2, 4, 6, 8, 1, 3]),  # 4-head schedule + every other slope of the 8-head one
        ],
    )
    def test_slope_schedule(self, n_heads, exponents):
        assert _get_alibi_slopes(n_heads) == pytest.approx([2.0**-e for e in exponents])

    def test_bias_is_minus_slope_times_euclidean_distance(self):
        coords = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]]])
        slopes = torch.tensor([0.5, 0.25])
        bias = compute_2d_alibi_bias(coords, slopes)
        assert bias.shape == (1, 2, 4, 4)
        dist = [[0.0, 5.0, 1.0], [5.0, 0.0, math.sqrt(18)], [1.0, math.sqrt(18), 0.0]]
        for h, slope in enumerate([0.5, 0.25]):
            expected = torch.tensor([[-slope * d for d in row] for row in dist])
            torch.testing.assert_close(bias[0, h, 1:, 1:], expected)
        assert (bias[:, :, 0, :] == 0).all()  # the CLS token is position-free
        assert (bias[:, :, :, 0] == 0).all()

    def test_masked_keys_get_the_fp16_floor(self):
        coords = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]]])
        slopes = torch.tensor([0.5])
        plain = compute_2d_alibi_bias(coords, slopes)
        masked = compute_2d_alibi_bias(coords, slopes, mask=torch.tensor([[1.0, 1.0, 0.0]]))
        offset = (masked - plain)[0, 0]  # token order: CLS, patch 0, patch 1, padded patch 2
        expected = torch.zeros(4, 4)
        expected[3, :3] = MASK_VALUE
        expected[:3, 3] = MASK_VALUE
        torch.testing.assert_close(offset, expected)  # the padded token still sees itself
        torch.testing.assert_close(
            compute_2d_alibi_bias(coords, slopes, mask=torch.ones(1, 3)), plain
        )
