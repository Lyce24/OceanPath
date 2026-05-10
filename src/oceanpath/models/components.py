"""
Shared building blocks for MIL models.

Contains reusable components that are composed by ABMIL, TransMIL, etc.
Kept separate from model classes to avoid circular imports and enable
independent testing.
"""

import torch
import torch.nn as nn

# ── MLP factory ───────────────────────────────────────────────────────────────


def create_mlp(
    in_dim: int = 768,
    hid_dims: list[int] | None = None,
    out_dim: int = 512,
    act: nn.Module | None = None,
    dropout: float = 0.0,
    end_with_fc: bool = True,
    bias: bool = True,
) -> nn.Module:
    """
    Build a sequential MLP with configurable depth, activation, and dropout.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dims : list[int]
        Hidden layer dimensions. Empty list = single linear layer.
    out_dim : int
        Output dimension.
    act : nn.Module
        Activation function (default: ReLU).
    dropout : float
        Dropout rate after each hidden layer.
    end_with_fc : bool
        If True, the last layer is a bare Linear (no activation).
        If False, activation is appended after the final Linear.
    bias : bool
        Whether Linear layers have bias.

    Returns
    -------
    nn.Sequential
    """
    if hid_dims is None:
        hid_dims = []
    if act is None:
        act = nn.ReLU()

    layers: list[nn.Module] = []
    current_dim = in_dim

    for hid_dim in hid_dims:
        layers.append(nn.Linear(current_dim, hid_dim, bias=bias))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hid_dim

    layers.append(nn.Linear(current_dim, out_dim, bias=bias))
    if not end_with_fc:
        layers.append(act)

    return nn.Sequential(*layers)


# ── Attention modules ─────────────────────────────────────────────────────────


class GlobalAttention(nn.Module):
    """
    Standard attention network (2 FC layers): Linear → Tanh → Dropout → Linear.

    Maps each instance embedding to a scalar attention logit.

    Parameters
    ----------
    L : int
        Input feature dimension.
    D : int
        Hidden attention dimension.
    dropout : float
        Dropout rate.
    num_classes : int
        Number of attention heads (typically 1).
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, L] → [B, N, num_classes]"""
        return self.module(x)


class GlobalGatedAttention(nn.Module):
    """
    Gated attention network (3 FC layers): two parallel branches (Tanh + Sigmoid)
    element-wise multiplied, then projected to attention logits.

    Parameters
    ----------
    L : int
        Input feature dimension.
    D : int
        Hidden attention dimension.
    dropout : float
        Dropout rate.
    num_classes : int
        Number of attention heads (typically 1).
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, L] → [B, N, num_classes]"""
        a = self.attention_a(x)
        b = self.attention_b(x)
        return self.attention_c(a * b)


"""
Hilbert curve ordering for coordinate-aware Mamba aggregators.

Converts 2D patch coordinates into a 1D Hilbert curve index, then sorts
patches so spatially nearby patches are sequentially adjacent in the
Mamba input sequence. This gives the SSM's causal recurrence natural
access to spatial context without any architectural changes.

Why Hilbert over raster scan:
    Raster scan (row-major) breaks spatial continuity at every row boundary.
    The last patch in row k and the first patch in row k+1 are spatially
    far apart but sequentially adjacent. Hilbert curves guarantee that
    patches nearby in 2D are nearby in 1D (bounded locality distance).

Why Hilbert over Z-order (Morton):
    Z-order also preserves some locality but has discontinuities at
    quadrant boundaries. Hilbert is strictly better for locality
    preservation (the curve is continuous, Z-order jumps).

Complexity:
    O(order x B x N) element-wise ops, fully vectorized on GPU.
    For order=16, B=128, N=4800: ~0.2ms on RTX 3090. Negligible vs Mamba.

Usage:
    Called inside Mamba2MIL.forward_features() AFTER patch_embed,
    BEFORE the SSM layers. No changes to dataset or datamodule needed.
"""

# ═════════════════════════════════════════════════════════════════════════════
# Core Hilbert index computation
# ═════════════════════════════════════════════════════════════════════════════


def hilbert_index_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    order: int = 16,
) -> torch.Tensor:
    """Compute Hilbert curve index for 2D integer coordinates.

    Vectorized, works on any tensor shape (including batched [B, N]).
    Implements the standard xy-to-d algorithm (Wikipedia / Numerical Recipes).

    Parameters
    ----------
    x, y : torch.Tensor
        Integer coordinates in [0, 2^order - 1]. Same shape.
    order : int
        Hilbert curve order. Grid is 2^order x 2^order.
        order=16 covers 65536x65536, sufficient for any WSI grid.

    Returns
    -------
    torch.Tensor
        Hilbert curve indices (int64), same shape as inputs.
    """
    x = x.long().clone()
    y = y.long().clone()
    d = torch.zeros_like(x)

    s = 1 << (order - 1)
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d += s * s * ((3 * rx) ^ ry)

        # Rotate quadrant
        need_rotate = ry == 0
        need_flip = need_rotate & (rx == 1)

        x = torch.where(need_flip, s - 1 - x, x)
        y = torch.where(need_flip, s - 1 - y, y)

        x_new = torch.where(need_rotate, y, x)
        y_new = torch.where(need_rotate, x, y)
        x, y = x_new, y_new

        s >>= 1

    return d


# ═════════════════════════════════════════════════════════════════════════════
# Coordinate normalization
# ═════════════════════════════════════════════════════════════════════════════


def normalize_coords_to_grid(
    coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    order: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize raw patch coordinates to [0, 2^order - 1] per batch element.

    Handles irregular coordinate ranges across slides by per-element
    min-max normalization. Padding tokens (mask=0) are excluded from
    range computation and assigned coordinate (0, 0).

    Parameters
    ----------
    coords : torch.Tensor [B, N, 2]
        Raw spatial coordinates (pixel-space or grid-space).
    mask : torch.Tensor [B, N] or None
        1 = valid token, 0 = padding.
    order : int
        Hilbert curve order. Output range is [0, 2^order - 1].

    Returns
    -------
    x_norm, y_norm : torch.Tensor [B, N]
        Normalized integer coordinates, ready for hilbert_index_2d.
    """
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be [B, N, 2], got {tuple(coords.shape)}")
    if not (1 <= order <= 31):
        raise ValueError(f"Hilbert order must be in [1, 31], got {order}")
    if mask is not None and mask.shape != coords.shape[:2]:
        raise ValueError(
            f"mask must match coords batch/sequence shape {tuple(coords.shape[:2])}, "
            f"got {tuple(mask.shape)}"
        )

    grid_max = (1 << order) - 1

    x = coords[..., 0].float()  # [B, N]
    y = coords[..., 1].float()  # [B, N]

    if mask is not None:
        invalid = ~mask.bool()
        # Exclude padding from range computation
        x_for_range = x.masked_fill(invalid, float("inf"))
        y_for_range = y.masked_fill(invalid, float("inf"))
        x_min = x_for_range.min(dim=1, keepdim=True).values  # [B, 1]
        y_min = y_for_range.min(dim=1, keepdim=True).values

        x_for_range = x.masked_fill(invalid, float("-inf"))
        y_for_range = y.masked_fill(invalid, float("-inf"))
        x_max = x_for_range.max(dim=1, keepdim=True).values
        y_max = y_for_range.max(dim=1, keepdim=True).values
    else:
        x_min = x.min(dim=1, keepdim=True).values
        y_min = y.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        y_max = y.max(dim=1, keepdim=True).values

    # Normalize to [0, grid_max] — uniform scale preserves aspect ratio
    x_range = (x_max - x_min).clamp(min=1.0)
    y_range = (y_max - y_min).clamp(min=1.0)
    span = torch.max(x_range, y_range)  # uniform scale

    x_norm = ((x - x_min) / span * grid_max).long().clamp(0, grid_max)
    y_norm = ((y - y_min) / span * grid_max).long().clamp(0, grid_max)

    # Zero out padding coords (their Hilbert index won't matter — we push
    # them to the end separately)
    if mask is not None:
        x_norm = x_norm.masked_fill(invalid, 0)
        y_norm = y_norm.masked_fill(invalid, 0)

    return x_norm, y_norm


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point: sort by Hilbert index
# ═════════════════════════════════════════════════════════════════════════════


def hilbert_sort(
    features: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    order: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Reorder patches along a Hilbert space-filling curve.

    Padding tokens are pushed to the END of each sequence (after all
    valid tokens), preserving the mask contract with downstream layers.

    Parameters
    ----------
    features : torch.Tensor [B, N, D]
        Patch features (after embedding, before SSM).
    coords : torch.Tensor [B, N, 2]
        Spatial coordinates.
    mask : torch.Tensor [B, N] or None
        1 = valid, 0 = padding.
    order : int
        Hilbert curve order.

    Returns
    -------
    features, coords, mask, sort_idx : reordered tensors (same shapes).
    """
    B, N, _ = features.shape
    if coords.shape[:2] != (B, N) or coords.shape[-1] != 2:
        raise ValueError(
            f"coords must be [B, N, 2] matching features, got {tuple(coords.shape)} "
            f"for features {tuple(features.shape)}"
        )

    if mask is not None:
        if mask.shape != (B, N):
            raise ValueError(
                f"mask must be [B, N] matching features, got {tuple(mask.shape)} "
                f"for features {tuple(features.shape)}"
            )
        valid_counts = mask.sum(dim=1)
        if (valid_counts == 0).any():
            raise ValueError("hilbert_sort received a slide with zero valid patches.")

    # Step 1: Normalize coordinates to grid
    x_norm, y_norm = normalize_coords_to_grid(coords, mask, order)

    # Step 2: Compute Hilbert index
    h_idx = hilbert_index_2d(x_norm, y_norm, order)  # [B, N]

    # Step 3: Push padding to end (assign index larger than any valid index)
    if mask is not None:
        pad_idx = 1 << (2 * order)
        h_idx = h_idx.masked_fill(~mask.bool(), pad_idx)

    # Step 4: Argsort — stable sort preserves relative order of tied indices
    sort_idx = h_idx.argsort(dim=1, stable=True)  # [B, N]

    # Step 5: Gather reordered tensors
    features = features.gather(1, sort_idx.unsqueeze(-1).expand_as(features))
    coords = coords.gather(1, sort_idx.unsqueeze(-1).expand_as(coords))
    if mask is not None:
        mask = mask.gather(1, sort_idx)

    return features, coords, mask, sort_idx


"""
Quick verification that Hilbert ordering works correctly.
Run standalone: python test_hilbert.py
"""

import time  # noqa: E402  # script-style benchmark block below


def test_locality():
    """Verify Hilbert curve preserves spatial locality better than raster."""
    print("=" * 60)
    print("TEST 1: Locality preservation")
    print("=" * 60)

    order = 4  # 16x16 grid
    n = 1 << order

    # Create all grid points
    ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    xs_flat = xs.reshape(-1)
    ys_flat = ys.reshape(-1)

    # Hilbert indices
    h_idx = hilbert_index_2d(xs_flat, ys_flat, order)

    # Sort by Hilbert index
    sort_order = h_idx.argsort()
    xs_sorted = xs_flat[sort_order]
    ys_sorted = ys_flat[sort_order]

    # Measure average step distance along the curve
    dx = (xs_sorted[1:] - xs_sorted[:-1]).float().abs()
    dy = (ys_sorted[1:] - ys_sorted[:-1]).float().abs()
    hilbert_step = (dx + dy).mean().item()

    # Compare with raster scan (row-major)
    xs_raster = xs_flat  # already in raster order
    ys_raster = ys_flat
    dx_r = (xs_raster[1:] - xs_raster[:-1]).float().abs()
    dy_r = (ys_raster[1:] - ys_raster[:-1]).float().abs()
    raster_step = (dx_r + dy_r).mean().item()

    print(f"  Grid: {n}x{n} = {n * n} points")
    print(f"  Hilbert avg step distance: {hilbert_step:.3f}")
    print(f"  Raster  avg step distance: {raster_step:.3f}")
    print(f"  Hilbert is {raster_step / hilbert_step:.1f}x better locality")

    # Hilbert curve should have avg step ~1 (always adjacent cells)
    assert hilbert_step < raster_step, "Hilbert should beat raster!"
    print("  ✓ PASSED\n")


def test_uniqueness():
    """Verify all Hilbert indices are unique (bijective mapping)."""
    print("=" * 60)
    print("TEST 2: Uniqueness (bijection)")
    print("=" * 60)

    order = 4
    n = 1 << order
    ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    h_idx = hilbert_index_2d(xs.reshape(-1), ys.reshape(-1), order)

    n_unique = h_idx.unique().numel()
    n_total = n * n
    print(f"  {n_unique} unique indices out of {n_total} points")
    assert n_unique == n_total, "Hilbert should be a bijection!"
    print("  ✓ PASSED\n")


def test_batched():
    """Verify batched computation matches per-element computation."""
    print("=" * 60)
    print("TEST 3: Batched consistency")
    print("=" * 60)

    B, N = 4, 100
    order = 10
    grid_max = (1 << order) - 1

    torch.manual_seed(42)
    x = torch.randint(0, grid_max, (B, N))
    y = torch.randint(0, grid_max, (B, N))

    # Batched
    h_batched = hilbert_index_2d(x, y, order)

    # Per-element
    for b in range(B):
        h_single = hilbert_index_2d(x[b], y[b], order)
        assert torch.equal(h_batched[b], h_single), f"Mismatch at batch {b}!"

    print(f"  Batched ({B}x{N}) matches per-element computation")
    print("  ✓ PASSED\n")


def test_performance():
    """Benchmark on realistic sizes."""
    print("=" * 60)
    print("TEST 4: Performance benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    for B, N in [(32, 4800), (64, 4800), (128, 4800), (32, 10000)]:
        order = 16
        grid_max = (1 << order) - 1

        x = torch.randint(0, grid_max, (B, N), device=device)
        y = torch.randint(0, grid_max, (B, N), device=device)

        # Warmup
        _ = hilbert_index_2d(x, y, order)
        if device == "cuda":
            torch.cuda.synchronize()

        # Time it
        t0 = time.perf_counter()
        for _ in range(10):
            _ = hilbert_index_2d(x, y, order)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / 10 * 1000
        print(f"  B={B:>3}, N={N:>5}: {ms:.2f} ms")

    print("  ✓ DONE\n")


def test_sort_with_mask():
    """Verify padding tokens end up at the end after sorting."""
    print("=" * 60)
    print("TEST 5: Masked sorting")
    print("=" * 60)

    B, N, _D = 2, 10, 4
    torch.manual_seed(0)

    coords = torch.randint(0, 100, (B, N, 2))
    mask = torch.ones(B, N)
    mask[0, 7:] = 0  # slide 0: 7 valid, 3 padding
    mask[1, 5:] = 0  # slide 1: 5 valid, 5 padding

    # Normalize
    grid_max = (1 << 8) - 1
    x = coords[..., 0].float()
    y = coords[..., 1].float()
    invalid = ~mask.bool()

    x_for_min = x.masked_fill(invalid, float("inf"))
    y_for_min = y.masked_fill(invalid, float("inf"))
    x_min = x_for_min.min(dim=1, keepdim=True).values
    y_min = y_for_min.min(dim=1, keepdim=True).values

    x_for_max = x.masked_fill(invalid, float("-inf"))
    y_for_max = y.masked_fill(invalid, float("-inf"))
    x_max = x_for_max.max(dim=1, keepdim=True).values
    y_max = y_for_max.max(dim=1, keepdim=True).values

    x_range = (x_max - x_min).clamp(min=1)
    y_range = (y_max - y_min).clamp(min=1)
    x_norm = ((x - x_min) / x_range * grid_max).long().clamp(0, grid_max)
    y_norm = ((y - y_min) / y_range * grid_max).long().clamp(0, grid_max)
    x_norm.masked_fill_(invalid, 0)
    y_norm.masked_fill_(invalid, 0)

    h_idx = hilbert_index_2d(x_norm, y_norm, order=8)
    h_idx.masked_fill_(invalid, h_idx.max() + 1)

    sort_idx = h_idx.argsort(dim=1, stable=True)
    mask_sorted = mask.gather(1, sort_idx)

    # Check: padding should be at the end
    for b in range(B):
        m = mask_sorted[b]
        n_valid = int(mask[b].sum())
        assert m[:n_valid].all(), f"Valid tokens not at start for batch {b}"
        assert not m[n_valid:].any(), f"Padding not at end for batch {b}"

    print("  Slide 0: 7 valid + 3 padding → padding at end ✓")
    print("  Slide 1: 5 valid + 5 padding → padding at end ✓")
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    test_locality()
    test_uniqueness()
    test_batched()
    test_sort_with_mask()
    test_performance()
    print("All tests passed!")
