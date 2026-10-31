"""
Shared building blocks for MIL models.

Contains reusable components that are composed by ABMIL, TransMIL, etc.
Kept separate from model classes to avoid circular imports and enable
independent testing.
"""

from collections.abc import Callable
from typing import TypeVar, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

_T = TypeVar("_T")

# ── Activation checkpointing ──────────────────────────────────────────────────


def run_checkpointed(fn: Callable[..., _T], *args: object) -> _T:
    """
    Run ``fn(*args)`` under non-reentrant activation checkpointing.

    ``torch.utils.checkpoint.checkpoint`` returns ``fn``'s output, but it is
    unannotated and its body ends in an implicit ``None``, so its inferred
    return type is ``Any | None``; this states the real one.
    """
    return cast(_T, cp.checkpoint(fn, *args, use_reentrant=False))


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

    invalid = None if mask is None else ~mask.bool()
    if invalid is not None:
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
    if invalid is not None:
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
