"""
Tests for oceanpath.models.components — shared model building blocks.

Covers:
  - create_mlp layer layout
  - hilbert_index_2d against the textbook scalar algorithm, bijection and locality
  - normalize_coords_to_grid: uniform min-max scaling, padding, input validation
  - hilbert_sort: patches follow their coordinates, padding last, order-independent
"""

import pytest
import torch
import torch.nn as nn

from oceanpath.models.components import (
    create_mlp,
    hilbert_index_2d,
    hilbert_sort,
    normalize_coords_to_grid,
)

STRIDE = 256  # coords as level-0 patch origins


def _xy2d_reference(n: int, x: int, y: int) -> int:
    """The scalar xy2d of Wikipedia's "Hilbert curve" article for an n x n grid."""
    d = 0
    s = n // 2
    while s > 0:
        rx = int((x & s) > 0)
        ry = int((y & s) > 0)
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:  # rotate the quadrant
            if rx == 1:
                x, y = n - 1 - x, n - 1 - y
            x, y = y, x
        s //= 2
    return d


def _grid(order: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = 1 << order
    ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    return xs.reshape(-1), ys.reshape(-1)


# ═════════════════════════════════════════════════════════════════════════════
# MLP factory
# ═════════════════════════════════════════════════════════════════════════════


class TestCreateMLP:
    def test_single_linear_without_hidden_layers(self):
        mlp = create_mlp(in_dim=8, out_dim=4)
        assert [type(m) for m in mlp] == [nn.Linear]
        assert (mlp[0].in_features, mlp[0].out_features) == (8, 4)

    def test_hidden_layers_dropout_and_trailing_activation(self):
        mlp = create_mlp(
            in_dim=8, hid_dims=[6, 5], out_dim=4, act=nn.GELU(), dropout=0.1, end_with_fc=False
        )
        kinds = [type(m) for m in mlp]
        assert kinds == [nn.Linear, nn.GELU, nn.Dropout] * 2 + [nn.Linear, nn.GELU]
        dims = [(m.in_features, m.out_features) for m in mlp if isinstance(m, nn.Linear)]
        assert dims == [(8, 6), (6, 5), (5, 4)]

    def test_no_bias_and_no_dropout_layer_at_zero_rate(self):
        mlp = create_mlp(in_dim=8, hid_dims=[6], out_dim=4, bias=False)
        assert not any(isinstance(m, nn.Dropout) for m in mlp)
        assert all(m.bias is None for m in mlp if isinstance(m, nn.Linear))


# ═════════════════════════════════════════════════════════════════════════════
# Hilbert index
# ═════════════════════════════════════════════════════════════════════════════


class TestHilbertIndex:
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_matches_the_textbook_algorithm(self, order):
        xs, ys = _grid(order)
        cells = zip(xs.tolist(), ys.tolist(), strict=True)
        expected = [_xy2d_reference(1 << order, x, y) for x, y in cells]
        assert hilbert_index_2d(xs, ys, order).tolist() == expected

    def test_first_order_curve_visits_the_four_cells_in_u_order(self):
        x = torch.tensor([0, 0, 1, 1])
        y = torch.tensor([0, 1, 1, 0])
        assert hilbert_index_2d(x, y, order=1).tolist() == [0, 1, 2, 3]

    def test_is_a_bijection_onto_the_index_range(self):
        xs, ys = _grid(4)
        h = hilbert_index_2d(xs, ys, order=4)
        assert h.dtype == torch.int64
        assert sorted(h.tolist()) == list(range(16 * 16))

    def test_consecutive_indices_are_neighbouring_cells(self):
        """The defining property: the curve moves one cell at a time (raster jumps rows)."""
        xs, ys = _grid(4)
        order = hilbert_index_2d(xs, ys, order=4).argsort()
        steps = (xs[order].diff().abs() + ys[order].diff().abs()).tolist()
        assert set(steps) == {1}

    def test_batched_input_matches_each_row_alone(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 1 << 10, (4, 50), generator=g)
        y = torch.randint(0, 1 << 10, (4, 50), generator=g)
        batched = hilbert_index_2d(x, y, order=10)
        for b in range(4):
            assert torch.equal(batched[b], hilbert_index_2d(x[b], y[b], order=10))

    def test_inputs_are_not_modified(self):
        x = torch.tensor([3, 1, 2])
        y = torch.tensor([0, 2, 3])
        hilbert_index_2d(x, y, order=2)
        assert x.tolist() == [3, 1, 2]
        assert y.tolist() == [0, 2, 3]


# ═════════════════════════════════════════════════════════════════════════════
# Coordinate normalisation
# ═════════════════════════════════════════════════════════════════════════════


class TestNormalizeCoordsToGrid:
    def test_uniform_min_max_scaling_keeps_the_aspect_ratio(self):
        coords = torch.tensor([[[100, 50], [612, 306], [1124, 50]]])  # x span 1024, y span 256
        x, y = normalize_coords_to_grid(coords, order=4)  # grid_max 15
        assert x.tolist() == [[0, 7, 15]]  # 0, 7.5, 15 truncated
        assert y.tolist() == [[0, 3, 0]]  # the shorter axis is scaled by the same span

    def test_each_slide_is_normalised_on_its_own_range(self):
        base = torch.tensor([[0, 0], [256, 512], [512, 256]])
        coords = torch.stack([base, base * 3 + 10_000])
        x, y = normalize_coords_to_grid(coords, order=8)
        assert torch.equal(x[0], x[1])
        assert torch.equal(y[0], y[1])

    def test_padding_is_left_out_of_the_range_and_set_to_the_origin(self):
        coords = torch.tensor([[[0, 0], [512, 512], [10**6, 10**6], [256, 0]]])
        mask = torch.tensor([[1, 1, 0, 1]])
        x, y = normalize_coords_to_grid(coords, mask, order=4)
        unmasked_x, unmasked_y = normalize_coords_to_grid(coords[:, [0, 1, 3]], order=4)
        assert x[0, [0, 1, 3]].tolist() == unmasked_x[0].tolist() == [0, 15, 7]
        assert y[0, [0, 1, 3]].tolist() == unmasked_y[0].tolist()
        assert (x[0, 2].item(), y[0, 2].item()) == (0, 0)

    def test_a_single_patch_maps_to_the_origin(self):
        x, y = normalize_coords_to_grid(torch.tensor([[[4096, 8192]]]), order=16)
        assert (x.item(), y.item()) == (0, 0)

    @pytest.mark.parametrize(
        ("coords", "mask", "order"),
        [
            (torch.zeros(1, 3, 3), None, 4),  # three coordinate columns
            (torch.zeros(3, 2), None, 4),  # no batch dimension
            (torch.zeros(1, 3, 2), None, 0),
            (torch.zeros(1, 3, 2), None, 32),
            (torch.zeros(1, 3, 2), torch.ones(1, 4), 4),  # mask of another length
        ],
    )
    def test_rejects_malformed_input(self, coords, mask, order):
        with pytest.raises(ValueError):
            normalize_coords_to_grid(coords, mask, order=order)


# ═════════════════════════════════════════════════════════════════════════════
# Hilbert sort
# ═════════════════════════════════════════════════════════════════════════════


def _shuffled_grid_bag(order: int = 3, seed: int = 0):
    """A slide whose patches tile a 2^order grid in random order; feature = (x, y, 1)."""
    xs, ys = _grid(order)
    perm = torch.randperm(len(xs), generator=torch.Generator().manual_seed(seed))
    coords = (torch.stack([xs, ys], dim=1)[perm] * STRIDE).unsqueeze(0)  # [1, N, 2]
    feats = torch.cat([coords[0].float(), torch.ones(len(xs), 1)], dim=1).unsqueeze(0)
    return feats, coords


class TestHilbertSort:
    def test_sorted_patches_walk_neighbouring_cells(self):
        feats, coords = _shuffled_grid_bag()
        _, sorted_coords, _, _ = hilbert_sort(feats, coords, order=8)
        steps = sorted_coords[0].diff(dim=0).abs().sum(dim=1)
        assert set(steps.tolist()) == {STRIDE}

    def test_features_follow_their_coordinates(self):
        feats, coords = _shuffled_grid_bag()
        out_feats, out_coords, out_mask, sort_idx = hilbert_sort(feats, coords, order=8)
        assert out_mask is None
        assert torch.equal(out_feats[0, :, :2], out_coords[0].float())
        assert torch.equal(out_feats[0], feats[0, sort_idx[0]])
        assert sorted(sort_idx[0].tolist()) == list(range(feats.shape[1]))

    def test_order_does_not_depend_on_the_input_order(self):
        feats_a, coords_a = _shuffled_grid_bag(seed=1)
        feats_b, coords_b = _shuffled_grid_bag(seed=2)
        assert not torch.equal(coords_a, coords_b)
        assert torch.equal(hilbert_sort(feats_a, coords_a)[1], hilbert_sort(feats_b, coords_b)[1])

    def test_padding_goes_last_and_leaves_the_valid_order_alone(self):
        feats, coords = _shuffled_grid_bag()
        n = feats.shape[1]
        mask = torch.ones(1, n)
        pad = torch.tensor([3, 17, 40])
        mask[0, pad] = 0
        coords = coords.clone()
        coords[0, pad] = torch.tensor([10**6, 10**6])  # garbage in the padded rows

        _, out_coords, out_mask, _ = hilbert_sort(feats, coords, mask, order=8)
        n_valid = n - len(pad)
        assert out_mask is not None
        assert out_mask[0].tolist() == [1.0] * n_valid + [0.0] * len(pad)

        keep = mask[0].bool()
        _, valid_only, _, _ = hilbert_sort(feats[:, keep], coords[:, keep], order=8)
        assert torch.equal(out_coords[0, :n_valid], valid_only[0])

    def test_rejects_a_slide_without_valid_patches(self):
        feats, coords = _shuffled_grid_bag()
        with pytest.raises(ValueError, match="zero valid patches"):
            hilbert_sort(feats, coords, torch.zeros(1, feats.shape[1]))

    def test_rejects_mismatched_shapes(self):
        feats, coords = _shuffled_grid_bag()
        with pytest.raises(ValueError):
            hilbert_sort(feats, coords[:, :-1])
        with pytest.raises(ValueError):
            hilbert_sort(feats, coords, torch.ones(1, feats.shape[1] + 1))
