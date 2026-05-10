"""
WSI feature-space dual-view and multi-crop augmentation for frozen patch embeddings.

Two augmentor classes:
  * WSIDualViewAugmentor — two views with optional inter-view split
    (used by VICReg, JEPA, LeJEPA-2C).
  * WSIMultiCropAugmentor — V_g global + V_l local views (used by LeJEPA-MC).

Pipeline (dual-view):
    full N tokens
    -> shared D4 coordinate transform (optional, min-shifted to non-negative)
    -> split into two view index sets
    -> coordinate crop per view
    -> random token masking per view
    -> hierarchical refill to exactly fixed_n unique tokens per view

Pipeline (multi-crop):
    full N tokens
    -> shared D4 coordinate transform (optional, applied ONCE)
    -> for each of V_g + V_l views, independently:
         coordinate crop -> random token masking -> hierarchical refill
       (no inter-view split; every view samples from the full slide)

Hierarchical refill order (per view):
    1. post-mask tokens
    2. if < fixed_n: add unique tokens from post-crop pool
    3. if still < fixed_n: add unique tokens from post-split pool
    4. if still < fixed_n: add unique tokens from full slide
    5. last resort: oversample with replacement (only fires if
       n_full_slide < fixed_n)

Guarantees:
    - Feature values are never modified.
    - All shape-changing operations are index selections.
    - Features are sliced exactly once per view at the end.
    - Each view always has exactly fixed_n tokens.
    - Tokens within a view are unique unless the slide itself has fewer than
      fixed_n patches (should not happen if manifest enforces min_patches >=
      fixed_n; pass `manifest_min_patches` to the augmentor for a hard guard).
    - D4 output coordinates are non-negative (min-shifted to bbox origin).
    - Hilbert sorting must happen downstream (in the model), not here.

Stats:
    `return_stats=True` returns a per-view breakdown with both the fallback
    level reached AND the actual composition of the final view:

        n_from_mask         tokens kept from post-mask pool
        n_from_crop         tokens added from post-crop pool (refill)
        n_from_split        tokens added from post-split pool (refill)
        n_from_full         tokens added from full-slide pool (refill)
        n_from_replacement  tokens added with replacement (last resort)

    Plus cumulative fractions:

        mask_fraction              n_from_mask / fixed_n
        crop_or_better_fraction    (mask + crop) / fixed_n
        view_or_better_fraction    (mask + crop + split) / fixed_n
        full_refill_fraction       full / fixed_n
        replacement_fraction       replacement / fixed_n

    Track `mask_fraction` during pretraining — it answers the question
    "what fraction of the final view actually obeyed crop/mask augmentation?"
    which `unique_fraction` cannot answer when min_patches >= fixed_n.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

Array = np.ndarray


# =============================================================================
# Config
# =============================================================================


@dataclass
class WSIDualViewAugmentConfig:
    """Single configuration object for the dual-view augmentor.

    The same config is used for both views. View asymmetry is *not* configured
    via per-view "strength" knobs. If you need asymmetric views (e.g. for an
    EMA target encoder), build two augmentors with different configs.
    """

    # ---- Final fixed-N batching ----
    fixed_n: int = 2048

    # ---- Shared D4 coordinate transform ----
    # "none"   = no coordinate transform
    # "shared" = one D4 transform (uniform from 8 elements) applied to coords
    #            before view generation; both views see the same orientation
    d4_mode: Literal["none", "shared"] = "shared"

    # ---- Split ----
    # 1.0 = no split; both views start from all tokens.
    # 0.3 = SPT-style partial overlap (30% shared, ~65% per view).
    # 0.0 = fully disjoint 50/50 partition.
    split_overlap: float = 0.3

    # ---- Spatial crop ----
    crop_prob: float = 1.0
    crop_area_range: tuple[float, float] = (0.5, 1.0)
    crop_aspect_range: tuple[float, float] = (0.5, 2.0)
    crop_min_keep_frac: float = 0.40
    crop_max_tries: int = 10

    # ---- Random token masking ----
    # mask_ratio = fraction *dropped* (NOT kept) before final fixed_n step.
    mask_ratio_range: tuple[float, float] = (0.10, 0.25)
    mask_min_keep_frac: float = 0.60

    # ---- Safety floor ----
    min_tokens: int = 256


# =============================================================================
# RNG / validation helpers
# =============================================================================


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _spawn_rngs(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Spawn n child RNGs from a parent. Falls back to seeded children on
    older numpy versions that lack Generator.spawn.
    """
    if hasattr(rng, "spawn"):
        return list(rng.spawn(n))

    seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n,
        dtype=np.uint32,
    )
    return [np.random.default_rng(int(seed)) for seed in seeds]


def _validate_features_coords(features: Array, coords: Array | None) -> None:
    if not isinstance(features, np.ndarray):
        raise TypeError(f"features must be np.ndarray, got {type(features)}")

    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {features.shape}")

    if features.shape[0] <= 0:
        raise ValueError("features must contain at least one token")

    if coords is not None:
        if not isinstance(coords, np.ndarray):
            raise TypeError(f"coords must be np.ndarray or None, got {type(coords)}")

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape [N, 2], got {coords.shape}")

        if coords.shape[0] != features.shape[0]:
            raise ValueError(
                f"features and coords must have same N, got "
                f"{features.shape[0]} and {coords.shape[0]}"
            )


def _validate_cfg(cfg: WSIDualViewAugmentConfig) -> None:
    if cfg.fixed_n <= 0:
        raise ValueError(f"fixed_n must be positive, got {cfg.fixed_n}")

    if cfg.min_tokens <= 0:
        raise ValueError(f"min_tokens must be positive, got {cfg.min_tokens}")

    if cfg.min_tokens > cfg.fixed_n:
        raise ValueError(
            f"min_tokens should not exceed fixed_n, got "
            f"min_tokens={cfg.min_tokens}, fixed_n={cfg.fixed_n}"
        )

    if cfg.d4_mode not in {"none", "shared"}:
        raise ValueError(f"d4_mode must be 'none' or 'shared', got {cfg.d4_mode}")

    if not (0.0 <= cfg.split_overlap <= 1.0):
        raise ValueError(f"split_overlap must be in [0, 1], got {cfg.split_overlap}")

    if not (0.0 <= cfg.crop_prob <= 1.0):
        raise ValueError(f"crop_prob must be in [0, 1], got {cfg.crop_prob}")

    area_lo, area_hi = cfg.crop_area_range
    if area_lo <= 0 or area_hi <= 0 or area_lo > area_hi or area_hi > 1.0:
        raise ValueError(
            f"crop_area_range must be positive and within (0, 1], got {cfg.crop_area_range}"
        )

    aspect_lo, aspect_hi = cfg.crop_aspect_range
    if aspect_lo <= 0 or aspect_hi <= 0 or aspect_lo > aspect_hi:
        raise ValueError(f"invalid crop_aspect_range={cfg.crop_aspect_range}")

    if not (0.0 <= cfg.crop_min_keep_frac <= 1.0):
        raise ValueError(f"crop_min_keep_frac must be in [0, 1], got {cfg.crop_min_keep_frac}")

    mask_lo, mask_hi = cfg.mask_ratio_range
    if mask_lo < 0 or mask_hi < 0 or mask_lo > mask_hi or mask_hi >= 1.0:
        raise ValueError(
            f"mask_ratio_range must satisfy 0 <= lo <= hi < 1, got {cfg.mask_ratio_range}"
        )

    if not (0.0 <= cfg.mask_min_keep_frac <= 1.0):
        raise ValueError(f"mask_min_keep_frac must be in [0, 1], got {cfg.mask_min_keep_frac}")


# =============================================================================
# 1. Shared D4 transform
# =============================================================================


def _apply_shared_d4(
    coords: Array | None,
    rng: np.random.Generator,
) -> Array | None:
    """Uniformly sample one of the 8 D4 transforms and apply it around the
    coordinate bounding-box center.

    D4 = rotations {0, 90, 180, 270} x optional reflection.
    Each of the 8 elements has probability 1/8.

    Output coordinates are min-shifted to be non-negative — required for
    downstream code (e.g. Hilbert sort) that may assume non-negative
    coordinates. For non-square slides, rotation around the bbox center
    can otherwise push coords below zero.
    """
    if coords is None:
        return None

    original_dtype = coords.dtype
    coords_f = coords.astype(np.float32, copy=True)

    cmin = coords_f.min(axis=0)
    cmax = coords_f.max(axis=0)
    center = (cmin + cmax) / 2.0

    x = coords_f[:, 0] - center[0]
    y = coords_f[:, 1] - center[1]

    # Rotation by k * 90 degrees.
    k = int(rng.integers(0, 4))
    if k == 0:
        xr, yr = x, y
    elif k == 1:
        xr, yr = -y, x
    elif k == 2:
        xr, yr = -x, -y
    else:
        xr, yr = y, -x

    # Optional reflection across x-axis in rotated coordinates.
    if bool(rng.integers(0, 2)):
        xr = -xr

    out = np.empty_like(coords_f)
    out[:, 0] = xr + center[0]
    out[:, 1] = yr + center[1]

    # Min-shift to keep coords non-negative without changing relative geometry.
    out -= out.min(axis=0, keepdims=True)

    if np.issubdtype(original_dtype, np.integer):
        return np.round(out).astype(original_dtype)

    return out.astype(original_dtype, copy=False)


# =============================================================================
# 2. Split
# =============================================================================


def _split_indices(
    n_tokens: int,
    overlap: float,
    min_tokens: int,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    """Split full token index set into two view index sets.

    overlap = 1.0:
        both views receive all tokens.

    overlap = 0.0:
        views are disjoint 50/50 partitions.

    overlap = 0.3:
        30% shared tokens, remaining 70% split between views.
        Expected per-view size = 0.3N + 0.35N = 0.65N.

    Falls back to (all_idx, all_idx) for slides too small to support a
    meaningful split.
    """
    all_idx = np.arange(n_tokens, dtype=np.int64)

    if overlap >= 1.0:
        return all_idx, all_idx

    if n_tokens < 2 * min_tokens:
        return all_idx, all_idx

    perm = rng.permutation(n_tokens)

    shared_n = round(overlap * n_tokens)
    shared = perm[:shared_n]
    rest = perm[shared_n:]

    half = rest.size // 2
    v1 = np.concatenate([shared, rest[:half]])
    v2 = np.concatenate([shared, rest[half:]])

    if v1.size < min_tokens or v2.size < min_tokens:
        return all_idx, all_idx

    return v1.astype(np.int64, copy=False), v2.astype(np.int64, copy=False)


# =============================================================================
# 3. Crop
# =============================================================================


def _crop_indices(
    indices: Array,
    coords: Array | None,
    cfg: WSIDualViewAugmentConfig,
    rng: np.random.Generator,
) -> tuple[Array, float]:
    """Coordinate-based rectangular crop.

    Returns ``(out_indices, area_frac)`` where ``area_frac`` is the realized
    crop area as a fraction of the input bounding-box area: the sampled
    ``area`` on a successful crop, or ``1.0`` when the crop step is a no-op
    (no coords, too few tokens, skipped by ``crop_prob``, degenerate span,
    or all ``crop_max_tries`` attempts rejected).
    """
    if coords is None:
        return indices, 1.0

    n = indices.size
    if n <= cfg.min_tokens:
        return indices, 1.0

    if rng.random() > cfg.crop_prob:
        return indices, 1.0

    min_keep = max(cfg.min_tokens, int(np.ceil(n * cfg.crop_min_keep_frac)))
    if n <= min_keep:
        return indices, 1.0

    sub_coords = coords[indices].astype(np.float32, copy=False)

    cmin = sub_coords.min(axis=0)
    cmax = sub_coords.max(axis=0)
    span = cmax - cmin

    if span[0] <= 1e-6 or span[1] <= 1e-6:
        return indices, 1.0

    area_lo, area_hi = cfg.crop_area_range
    aspect_lo, aspect_hi = cfg.crop_aspect_range

    log_aspect_lo = np.log(aspect_lo)
    log_aspect_hi = np.log(aspect_hi)

    for _ in range(cfg.crop_max_tries):
        area = float(rng.uniform(area_lo, area_hi))
        aspect = float(np.exp(rng.uniform(log_aspect_lo, log_aspect_hi)))

        crop_w_frac = float(np.sqrt(area * aspect))
        crop_h_frac = float(np.sqrt(area / aspect))

        if crop_w_frac > 1.0 or crop_h_frac > 1.0:
            continue

        crop_w = span[0] * crop_w_frac
        crop_h = span[1] * crop_h_frac

        max_x0 = cmax[0] - crop_w
        max_y0 = cmax[1] - crop_h

        x0 = float(rng.uniform(cmin[0], max_x0)) if max_x0 > cmin[0] else cmin[0]
        y0 = float(rng.uniform(cmin[1], max_y0)) if max_y0 > cmin[1] else cmin[1]

        x1 = x0 + crop_w
        y1 = y0 + crop_h

        keep = (
            (sub_coords[:, 0] >= x0)
            & (sub_coords[:, 0] <= x1)
            & (sub_coords[:, 1] >= y0)
            & (sub_coords[:, 1] <= y1)
        )

        if int(keep.sum()) >= min_keep:
            return indices[keep], area

    return indices, 1.0


# =============================================================================
# 4. Mask
# =============================================================================


def _mask_indices(
    indices: Array,
    cfg: WSIDualViewAugmentConfig,
    rng: np.random.Generator,
) -> tuple[Array, float]:
    """Random token masking.

    Keeps exactly keep_n tokens (deterministic given the sampled mask_ratio).
    Sampling without replacement guarantees uniqueness.

    Returns ``(out_indices, mask_ratio_realized)`` where the realized ratio is
    ``1 - keep_n / n`` after the ``mask_min_keep_frac`` floor is applied — i.e.
    the actual fraction of input tokens dropped. ``0.0`` when masking is a
    no-op (too few tokens, or the floor forces ``keep_n >= n``).
    """
    n = indices.size
    if n <= cfg.min_tokens:
        return indices, 0.0

    mask_lo, mask_hi = cfg.mask_ratio_range
    mask_ratio = float(rng.uniform(mask_lo, mask_hi))

    keep_frac = 1.0 - mask_ratio
    keep_n = round(n * keep_frac)

    min_keep = max(cfg.min_tokens, int(np.ceil(n * cfg.mask_min_keep_frac)))
    keep_n = max(keep_n, min_keep)
    keep_n = min(keep_n, n)

    if keep_n >= n:
        return indices, 0.0

    realized = 1.0 - (keep_n / n)
    return rng.choice(indices, size=keep_n, replace=False), realized


# =============================================================================
# 5. Hierarchical refill to exactly fixed_n unique tokens
# =============================================================================


def _hierarchical_refill_indices(
    mask_idx: Array,
    crop_idx: Array,
    split_idx: Array,
    all_idx: Array,
    fixed_n: int,
    rng: np.random.Generator,
) -> tuple[Array, str, dict]:
    """Pick exactly fixed_n indices, preferring tokens closer to the augmented view.

    Hierarchy of pools (each is a strict superset of the previous):
        post_mask  ⊆  post_crop  ⊆  post_split  ⊆  full_slide

    Algorithm:
        1. If post_mask >= fixed_n: random subsample without replacement.
        2. Else: keep all post_mask tokens, then greedily refill from post_crop
           (unique tokens not yet selected), then post_split, then full_slide.
        3. If full_slide is exhausted and we still need more (only possible if
           n_full < fixed_n), oversample with replacement from current selection.

    Returns:
        (indices, fallback_level, n_from)
        - fallback_level: where refill stopped, one of:
            "subsample"      - post_mask had >= fixed_n (no refill needed)
            "post_mask_only" - post_mask < fixed_n but no refill happened
            "post_crop"      - refill stopped after post_crop
            "post_split"     - refill stopped after post_split
            "full_slide"     - refill needed full slide (split disjointness broken)
            "oversample"     - even full slide insufficient (should never fire if
                               manifest enforces min_patches >= fixed_n)
        - n_from: dict counting tokens from each pool. Sums to fixed_n.
          Use these for "how much of the view actually obeyed augmentation"
          diagnostics — fallback_level alone is not enough.
    """
    n_from = {
        "mask": 0,
        "crop": 0,
        "split": 0,
        "full": 0,
        "replacement": 0,
    }

    if mask_idx.size >= fixed_n:
        out = rng.choice(mask_idx, size=fixed_n, replace=False)
        n_from["mask"] = fixed_n
        return out.astype(np.int64, copy=False), "subsample", n_from

    selected = mask_idx.astype(np.int64, copy=True)
    n_from["mask"] = int(mask_idx.size)
    fallback = "post_mask_only"

    for name, pool, key in (
        ("post_crop", crop_idx, "crop"),
        ("post_split", split_idx, "split"),
        ("full_slide", all_idx, "full"),
    ):
        if selected.size >= fixed_n:
            break
        # assume_unique=True is safe: every pool is sampled without replacement
        # upstream, and `selected` is built by concatenating disjoint slices.
        available = np.setdiff1d(pool, selected, assume_unique=True)
        if available.size == 0:
            continue
        need = fixed_n - selected.size
        take = min(need, available.size)
        picked = rng.choice(available, size=take, replace=False)
        selected = np.concatenate([selected, picked])
        n_from[key] = int(take)
        fallback = name

    if selected.size < fixed_n:
        # Last resort: full slide < fixed_n. Should not happen under proper
        # manifest filtering. Oversample from what we have.
        short = fixed_n - selected.size
        extra = rng.choice(selected, size=short, replace=True)
        selected = np.concatenate([selected, extra])
        n_from["replacement"] = int(short)
        fallback = "oversample"

    # Shuffle so refilled tokens are not all clustered at the tail.
    # Hilbert sort downstream will reorder anyway, but this keeps any
    # pre-Hilbert step order-agnostic.
    rng.shuffle(selected)
    return selected.astype(np.int64, copy=False), fallback, n_from


# =============================================================================
# Dual-view augmentor
# =============================================================================


class WSIDualViewAugmentor:
    """Dual-view augmentor for frozen WSI patch embeddings.

    Each call returns two views, each with exactly fixed_n unique tokens
    (subject to the slide having enough total tokens).

    Parameters
    ----------
    cfg : WSIDualViewAugmentConfig, optional
        Augmentation configuration. If None, uses dataclass defaults.
    manifest_min_patches : int, optional
        If provided, asserts that manifest_min_patches >= cfg.fixed_n at
        construction. Catches the bug where someone silently lowers the
        manifest filter or raises fixed_n without updating the other,
        which would silently introduce oversample-with-replacement.
    """

    def __init__(
        self,
        cfg: WSIDualViewAugmentConfig | None = None,
        manifest_min_patches: int | None = None,
    ):
        self.cfg = cfg or WSIDualViewAugmentConfig()
        _validate_cfg(self.cfg)

        if manifest_min_patches is not None and manifest_min_patches < self.cfg.fixed_n:
            raise ValueError(
                f"manifest_min_patches ({manifest_min_patches}) < fixed_n "
                f"({self.cfg.fixed_n}). Slides smaller than fixed_n will trigger "
                f"oversample-with-replacement. Either raise the manifest filter "
                f"or lower fixed_n."
            )

        self.manifest_min_patches = manifest_min_patches

    def __call__(
        self,
        features: Array,
        coords: Array | None = None,
        rng: np.random.Generator | None = None,
        *,
        return_stats: bool = False,
    ):
        rng = _as_rng(rng)
        _validate_features_coords(features, coords)

        cfg = self.cfg
        n_tokens = features.shape[0]

        # 1. Shared D4 (or no-op).
        if cfg.d4_mode == "shared":
            coords_aug = _apply_shared_d4(coords, rng)
        else:
            coords_aug = coords

        # 2. Split into two view index sets.
        v1_init, v2_init = _split_indices(
            n_tokens=n_tokens,
            overlap=cfg.split_overlap,
            min_tokens=cfg.min_tokens,
            rng=rng,
        )

        rng1, rng2 = _spawn_rngs(rng, 2)

        # 3-5. Per-view crop -> mask -> hierarchical refill.
        v1_idx, v1_stats = self._augment_view_indices(
            init_idx=v1_init,
            coords=coords_aug,
            n_tokens=n_tokens,
            rng=rng1,
            view_name="v1",
        )
        v2_idx, v2_stats = self._augment_view_indices(
            init_idx=v2_init,
            coords=coords_aug,
            n_tokens=n_tokens,
            rng=rng2,
            view_name="v2",
        )

        # Single feature slice per view.
        f1 = features[v1_idx]
        f2 = features[v2_idx]

        if coords_aug is not None:
            c1 = coords_aug[v1_idx]
            c2 = coords_aug[v2_idx]
        else:
            c1 = None
            c2 = None

        output = ((f1, c1), (f2, c2))

        extra = []
        if return_stats:
            stats = {
                "n_tokens": int(n_tokens),
                "d4_mode": cfg.d4_mode,
                "split_overlap": float(cfg.split_overlap),
                "fixed_n": int(cfg.fixed_n),
                "v1": v1_stats,
                "v2": v2_stats,
            }
            extra.append(stats)

        if extra:
            return output, *extra

        return output

    def _augment_view_indices(
        self,
        init_idx: Array,
        coords: Array | None,
        n_tokens: int,
        rng: np.random.Generator,
        view_name: str,
    ) -> tuple[Array, dict]:
        cfg = self.cfg

        n_after_split = int(init_idx.size)

        # 3. Crop.
        crop_idx, crop_area_frac = _crop_indices(init_idx, coords, cfg, rng)
        n_after_crop = int(crop_idx.size)

        # 4. Mask.
        mask_idx, mask_ratio = _mask_indices(crop_idx, cfg, rng)
        n_after_mask = int(mask_idx.size)

        # 5. Hierarchical refill to exactly fixed_n unique tokens.
        all_idx = np.arange(n_tokens, dtype=np.int64)

        final_idx, fallback_level, n_from = _hierarchical_refill_indices(
            mask_idx=mask_idx,
            crop_idx=crop_idx,
            split_idx=init_idx,
            all_idx=all_idx,
            fixed_n=cfg.fixed_n,
            rng=rng,
        )

        unique_final = int(np.unique(final_idx).size)
        fixed_n = cfg.fixed_n

        # Cumulative composition fractions.
        mask_n = n_from["mask"]
        crop_n = n_from["crop"]
        split_n = n_from["split"]
        full_n = n_from["full"]
        replacement_n = n_from["replacement"]

        stats = {
            "view": view_name,
            "after_split": n_after_split,
            "after_crop": n_after_crop,
            "after_mask": n_after_mask,
            "final_n": int(final_idx.size),
            "unique_final": unique_final,
            "unique_fraction": float(unique_final / max(1, final_idx.size)),
            "fallback_level": fallback_level,
            # Realized augmentation parameters (the values actually applied
            # this call, not the configured ranges). Use these to monitor
            # whether augmentation is "biting" — if mask_ratio collapses to
            # 0 or crop_area_frac to 1, the views are effectively identical
            # at the spatial-augmentation stage.
            "crop_area_frac": float(crop_area_frac),
            "mask_ratio": float(mask_ratio),
            # Refill composition: token counts.
            "n_from_mask": mask_n,
            "n_from_crop": crop_n,
            "n_from_split": split_n,
            "n_from_full": full_n,
            "n_from_replacement": replacement_n,
            # Refill composition: cumulative fractions.
            "mask_fraction": float(mask_n / fixed_n),
            "crop_or_better_fraction": float((mask_n + crop_n) / fixed_n),
            "view_or_better_fraction": float((mask_n + crop_n + split_n) / fixed_n),
            "full_refill_fraction": float(full_n / fixed_n),
            "replacement_fraction": float(replacement_n / fixed_n),
        }

        return final_idx, stats


# =============================================================================
# Multi-crop augmentor (LeJEPA-MC)
# =============================================================================


@dataclass
class WSIViewSpec:
    """Per-view augmentation spec used by WSIMultiCropAugmentor.

    Each view is sampled independently from the full slide (no inter-view
    split, unlike WSIDualViewAugmentor's 0.3 default). The shared D4 transform
    is still applied once before view sampling — controlled by the augmentor,
    not by this spec.

    All fields mirror the per-view subset of WSIDualViewAugmentConfig so the
    same private crop/mask/refill helpers can be reused unchanged.
    """

    fixed_n: int = 2048

    crop_prob: float = 1.0
    crop_area_range: tuple[float, float] = (0.5, 1.0)
    crop_aspect_range: tuple[float, float] = (0.5, 2.0)
    crop_min_keep_frac: float = 0.40
    crop_max_tries: int = 10

    mask_ratio_range: tuple[float, float] = (0.10, 0.25)
    mask_min_keep_frac: float = 0.60

    min_tokens: int = 256


def _validate_view_spec(spec: WSIViewSpec, name: str = "view") -> None:
    """Mirror of _validate_cfg, restricted to per-view fields."""
    if spec.fixed_n <= 0:
        raise ValueError(f"{name}: fixed_n must be positive, got {spec.fixed_n}")
    if spec.min_tokens <= 0:
        raise ValueError(f"{name}: min_tokens must be positive, got {spec.min_tokens}")
    if spec.min_tokens > spec.fixed_n:
        raise ValueError(f"{name}: min_tokens ({spec.min_tokens}) > fixed_n ({spec.fixed_n})")
    if not (0.0 <= spec.crop_prob <= 1.0):
        raise ValueError(f"{name}: crop_prob must be in [0, 1], got {spec.crop_prob}")
    a_lo, a_hi = spec.crop_area_range
    if a_lo <= 0 or a_hi <= 0 or a_lo > a_hi or a_hi > 1.0:
        raise ValueError(f"{name}: crop_area_range invalid: {spec.crop_area_range}")
    asp_lo, asp_hi = spec.crop_aspect_range
    if asp_lo <= 0 or asp_hi <= 0 or asp_lo > asp_hi:
        raise ValueError(f"{name}: crop_aspect_range invalid: {spec.crop_aspect_range}")
    if not (0.0 <= spec.crop_min_keep_frac <= 1.0):
        raise ValueError(f"{name}: crop_min_keep_frac must be in [0, 1]")
    m_lo, m_hi = spec.mask_ratio_range
    if m_lo < 0 or m_hi < 0 or m_lo > m_hi or m_hi >= 1.0:
        raise ValueError(f"{name}: mask_ratio_range invalid: {spec.mask_ratio_range}")
    if not (0.0 <= spec.mask_min_keep_frac <= 1.0):
        raise ValueError(f"{name}: mask_min_keep_frac must be in [0, 1]")


def _sample_single_view_indices(
    n_tokens: int,
    coords: Array | None,
    spec: WSIViewSpec,
    rng: np.random.Generator,
) -> tuple[Array, dict]:
    """Sample one view from the full slide via crop -> mask -> hierarchical refill.

    Returns ``(indices, stats)`` where indices has shape [spec.fixed_n].
    The stats dict mirrors the per-view stats produced by the dual-view
    augmentor so a single downstream consumer can read both code paths.
    """
    # Build a synthetic dual-view-style cfg so we can reuse _crop_indices /
    # _mask_indices / _hierarchical_refill_indices unchanged. ``split_overlap``
    # is irrelevant here — multi-crop has no inter-view split — but the
    # helpers expect the field, so we set it to 1.0 ("no split").
    tmp_cfg = WSIDualViewAugmentConfig(
        fixed_n=spec.fixed_n,
        d4_mode="none",  # D4 is applied once at the augmentor level.
        split_overlap=1.0,
        crop_prob=spec.crop_prob,
        crop_area_range=spec.crop_area_range,
        crop_aspect_range=spec.crop_aspect_range,
        crop_min_keep_frac=spec.crop_min_keep_frac,
        crop_max_tries=spec.crop_max_tries,
        mask_ratio_range=spec.mask_ratio_range,
        mask_min_keep_frac=spec.mask_min_keep_frac,
        min_tokens=spec.min_tokens,
    )

    init_idx = np.arange(n_tokens, dtype=np.int64)

    crop_idx, crop_area_frac = _crop_indices(init_idx, coords, tmp_cfg, rng)
    n_after_crop = int(crop_idx.size)

    mask_idx, mask_ratio = _mask_indices(crop_idx, tmp_cfg, rng)
    n_after_mask = int(mask_idx.size)

    final_idx, fallback_level, n_from = _hierarchical_refill_indices(
        mask_idx=mask_idx,
        crop_idx=crop_idx,
        # No inter-view split; the "split pool" is the full slide.
        split_idx=init_idx,
        all_idx=init_idx,
        fixed_n=spec.fixed_n,
        rng=rng,
    )

    fixed_n = spec.fixed_n
    mask_n = n_from["mask"]
    crop_n = n_from["crop"]
    split_n = n_from["split"]
    full_n = n_from["full"]
    replacement_n = n_from["replacement"]

    stats = {
        "after_crop": n_after_crop,
        "after_mask": n_after_mask,
        "final_n": int(final_idx.size),
        "unique_final": int(np.unique(final_idx).size),
        # Match the dual-view stats schema so a single consumer reads both.
        "unique_fraction": float(np.unique(final_idx).size / max(1, final_idx.size)),
        "fallback_level": fallback_level,
        # Realized augmentation parameters (see _augment_view_indices for
        # rationale). Same schema as dual-view per-view stats.
        "crop_area_frac": float(crop_area_frac),
        "mask_ratio": float(mask_ratio),
        # Refill composition: token counts.
        # NOTE for multi-crop: ``_sample_single_view_indices`` calls the
        # hierarchical refill with ``split_idx == all_idx`` (there is no
        # split stage), so any "split-level" refills are functionally
        # full-slide refills. ``view_or_better_fraction`` therefore
        # over-reports "view or better" relative to its dual-view meaning.
        # Training is unaffected; only diagnostics need this caveat.
        "n_from_mask": mask_n,
        "n_from_crop": crop_n,
        "n_from_split": split_n,
        "n_from_full": full_n,
        "n_from_replacement": replacement_n,
        "mask_fraction": float(mask_n / fixed_n),
        "crop_or_better_fraction": float((mask_n + crop_n) / fixed_n),
        "view_or_better_fraction": float((mask_n + crop_n + split_n) / fixed_n),
        # Multi-crop alias for `view_or_better_fraction`: in MC there is no
        # split stage, so this fraction is exactly "tokens NOT pulled from
        # a full-slide refill". Use this name in MC reports for clarity.
        "no_split_full_refill_fraction": float((mask_n + crop_n + split_n) / fixed_n),
        "full_refill_fraction": float(full_n / fixed_n),
        "replacement_fraction": float(replacement_n / fixed_n),
    }
    return final_idx, stats


class WSIMultiCropAugmentor:
    """LeJEPA-MC augmentor: V_g global + V_l local views per slide.

    Pipeline per call:
        1. Optional shared D4 coordinate transform (applied ONCE before any
           view is sampled — matches the dual-view augmentor's behaviour).
        2. For each of V_g + V_l views, independently:
             crop  -> mask  -> hierarchical refill to spec.fixed_n unique tokens.
           No inter-view split: every view samples from the full slide.

    Returns a dict with separate global and local entries because their
    fixed_n typically differ (2048 vs 1024 in the proposal), which prevents
    a single homogeneous tensor representation. The dataset is responsible
    for emitting a sample dict that the multi-crop collator can stack.

    Parameters
    ----------
    num_global : int
        V_g. Must be >= 1 (the predictive centroid is computed over global views).
    num_local : int
        V_l. May be 0 (degenerates to "multi-global" if num_global > 1).
    global_spec : WSIViewSpec
        Per-view config for global views (default: fixed_n=2048, crop=[0.5,1.0],
        mask=[0.10,0.25] — matches Section 9 of the proposal).
    local_spec : WSIViewSpec
        Per-view config for local views (default: fixed_n=1024, crop=[0.25,0.60],
        mask=[0.05,0.20]).
    d4_mode : "none" or "shared"
        Whether to apply one D4 transform shared across ALL views.
    manifest_min_patches : int, optional
        Asserted >= max(global_spec.fixed_n, local_spec.fixed_n). Catches
        the same misconfiguration the dual-view augmentor catches.
    """

    def __init__(
        self,
        num_global: int = 2,
        num_local: int = 2,
        global_spec: WSIViewSpec | None = None,
        local_spec: WSIViewSpec | None = None,
        d4_mode: Literal["none", "shared"] = "shared",
        # When V_g == 2, ``split_overlap`` controls inter-global coupling:
        #   1.0 -> independent global sampling (paper / DINO-style MC default).
        #   <1.0 -> dual-view-style split, identical to WSIDualViewAugmentor.
        # This is the knob that makes "LeJEPA-MC = LeJEPA-2C + locals" exact:
        # set split_overlap to base.yaml's value so the global pair is sampled
        # the same way the dual-view methods sample their two views.
        # For V_g != 2, only split_overlap=1.0 (independent) is supported.
        split_overlap: float = 1.0,
        manifest_min_patches: int | None = None,
    ):
        if num_global < 1:
            raise ValueError(
                f"num_global must be >= 1 (centroid needs at least one global view), "
                f"got {num_global}"
            )
        if num_local < 0:
            raise ValueError(f"num_local must be >= 0, got {num_local}")
        if d4_mode not in {"none", "shared"}:
            raise ValueError(f"d4_mode must be 'none' or 'shared', got {d4_mode}")
        if not (0.0 <= float(split_overlap) <= 1.0):
            raise ValueError(f"split_overlap must be in [0, 1], got {split_overlap}")

        self.num_global = int(num_global)
        self.num_local = int(num_local)

        self.global_spec = global_spec or WSIViewSpec(
            fixed_n=2048,
            crop_prob=1.0,
            crop_area_range=(0.5, 1.0),
            crop_aspect_range=(0.5, 2.0),
            crop_min_keep_frac=0.40,
            mask_ratio_range=(0.10, 0.25),
            mask_min_keep_frac=0.60,
            min_tokens=256,
        )
        self.local_spec = local_spec or WSIViewSpec(
            fixed_n=1024,
            crop_prob=1.0,
            crop_area_range=(0.25, 0.60),
            crop_aspect_range=(0.5, 2.0),
            crop_min_keep_frac=0.40,
            mask_ratio_range=(0.05, 0.20),
            mask_min_keep_frac=0.60,
            min_tokens=128,
        )
        _validate_view_spec(self.global_spec, name="global_spec")
        _validate_view_spec(self.local_spec, name="local_spec")

        self.d4_mode = d4_mode
        self.split_overlap = float(split_overlap)
        self.manifest_min_patches = manifest_min_patches

        # If split is requested, we need V_g==2 (the split helper produces a
        # pair of index sets). For V_g>2 with split, fail loud rather than
        # silently fall back to independent sampling.
        if self.split_overlap < 1.0 and self.num_global != 2:
            raise ValueError(
                f"split_overlap<1.0 (dual-view-style global coupling) only "
                f"supports num_global=2, got num_global={self.num_global}. "
                f"Set split_overlap=1.0 to use independent global sampling."
            )

        # Build a nested dual-view augmentor for the V_g=2 split-globals path.
        # Built lazily so the cost is only paid when the user actually opts in.
        # `d4_mode='none'` because the outer MC augmentor already applies D4;
        # `fixed_n` is taken from the global spec so global views land at
        # `global_spec.fixed_n` tokens (consistent with the independent path).
        self._global_dual_aug: WSIDualViewAugmentor | None = None
        if self.split_overlap < 1.0 and self.num_global == 2:
            dv_cfg = WSIDualViewAugmentConfig(
                fixed_n=self.global_spec.fixed_n,
                d4_mode="none",
                split_overlap=self.split_overlap,
                crop_prob=self.global_spec.crop_prob,
                crop_area_range=self.global_spec.crop_area_range,
                crop_aspect_range=self.global_spec.crop_aspect_range,
                crop_min_keep_frac=self.global_spec.crop_min_keep_frac,
                crop_max_tries=self.global_spec.crop_max_tries,
                mask_ratio_range=self.global_spec.mask_ratio_range,
                mask_min_keep_frac=self.global_spec.mask_min_keep_frac,
                min_tokens=self.global_spec.min_tokens,
            )
            self._global_dual_aug = WSIDualViewAugmentor(
                dv_cfg, manifest_min_patches=manifest_min_patches
            )

        max_fixed_n = max(self.global_spec.fixed_n, self.local_spec.fixed_n)
        if manifest_min_patches is not None and manifest_min_patches < max_fixed_n:
            raise ValueError(
                f"manifest_min_patches ({manifest_min_patches}) < max view "
                f"fixed_n ({max_fixed_n}). Slides smaller than the largest view "
                f"will trigger oversample-with-replacement. Either raise the "
                f"manifest filter or lower the view's fixed_n."
            )

    # -- Backward-compat shim for code that reads view_gen.cfg.fixed_n etc. -
    @property
    def fixed_n(self) -> int:
        """Largest view's fixed_n. The multi-crop dataset path uses
        spec-aware finalization rather than this single value, but some
        utility code still expects an augmentor-level fixed_n."""
        return max(self.global_spec.fixed_n, self.local_spec.fixed_n)

    # -- Main entry point -------------------------------------------------
    def __call__(
        self,
        features: Array,
        coords: Array | None = None,
        rng: np.random.Generator | None = None,
        *,
        return_stats: bool = False,
    ):
        rng = _as_rng(rng)
        _validate_features_coords(features, coords)

        # 1. Shared D4 (applied once, before any view sampling).
        if self.d4_mode == "shared":
            coords_aug = _apply_shared_d4(coords, rng)
        else:
            coords_aug = coords

        n_tokens = features.shape[0]

        global_views: list[tuple[Array, Array | None]] = []
        local_views: list[tuple[Array, Array | None]] = []
        global_stats: list[dict] = []
        local_stats: list[dict] = []

        # 2. Global views.
        if self._global_dual_aug is not None:
            # V_g=2 + split_overlap<1.0: use the dual-view augmentor so the
            # global pair is sampled identically to LeJEPA-2C / VICReg / JEPA.
            # `d4_mode="none"` on the inner augmentor; outer D4 already applied.
            inner_rng = _spawn_rngs(rng, 1)[0]
            ((f1, c1), (f2, c2)), gstats = self._global_dual_aug(
                features,
                coords_aug,
                rng=inner_rng,
                return_stats=True,
            )
            global_views.append((f1, c1))
            global_views.append((f2, c2))
            for view_idx, dv_key in ((0, "v1"), (1, "v2")):
                s = dict(gstats[dv_key])
                s["view_type"] = "global"
                s["view_index"] = view_idx
                global_stats.append(s)
            local_rngs = _spawn_rngs(rng, self.num_local)
        else:
            # Independent global sampling (paper convention; required for V_g != 2).
            rngs = _spawn_rngs(rng, self.num_global + self.num_local)
            for i in range(self.num_global):
                idx, stats = _sample_single_view_indices(
                    n_tokens=n_tokens,
                    coords=coords_aug,
                    spec=self.global_spec,
                    rng=rngs[i],
                )
                f_v = features[idx]
                c_v = coords_aug[idx] if coords_aug is not None else None
                global_views.append((f_v, c_v))
                stats["view_type"] = "global"
                stats["view_index"] = i
                global_stats.append(stats)
            local_rngs = rngs[self.num_global :]

        # 3. Local views (always independent from full slide — DINO convention).
        for i in range(self.num_local):
            idx, stats = _sample_single_view_indices(
                n_tokens=n_tokens,
                coords=coords_aug,
                spec=self.local_spec,
                rng=local_rngs[i],
            )
            f_v = features[idx]
            c_v = coords_aug[idx] if coords_aug is not None else None
            local_views.append((f_v, c_v))
            stats["view_type"] = "local"
            stats["view_index"] = i
            local_stats.append(stats)

        output = {
            "global": global_views,  # list[(features, coords)] of length V_g
            "local": local_views,  # list[(features, coords)] of length V_l
        }

        if return_stats:
            stats_payload = {
                "n_tokens": int(n_tokens),
                "d4_mode": self.d4_mode,
                "num_global": self.num_global,
                "num_local": self.num_local,
                "global_views": global_stats,
                "local_views": local_stats,
            }
            return output, stats_payload

        return output


_VALID_METHODS = frozenset(
    {
        "default",
        "vicreg",
        "jepa",  # alias for pooled_jepa (slide-level JEPA augmentor)
        "pooled_jepa",  # internal name for the slide-level JEPA augmentor
        "lejepa",
        "lejepa_mc",  # multi-crop — built via build_multicrop_augmentor
    }
)


def build_augmentor(
    method: str = "default",
    cfg: WSIDualViewAugmentConfig | None = None,
    manifest_min_patches: int | None = None,
    **overrides,
) -> WSIDualViewAugmentor:
    """Build a dual-view augmentor by SSL method name.

    All dual-view methods use the SAME augmentation config by design — this is
    required for a fair regularizer-axis / prediction-signal-axis comparison.
    The method argument is used only for validation/logging; per-method
    augmentation differences would confound the controlled comparison.

    For multi-crop methods (lejepa_mc) use ``build_multicrop_augmentor`` instead.

    Parameters
    ----------
    method : str
        SSL method name (validation only, no augmentation effect).
    cfg : WSIDualViewAugmentConfig, optional
        Base config. If provided, a copy is made via ``dataclasses.replace``;
        the caller's instance is never mutated.
    manifest_min_patches : int, optional
        Asserted at augmentor construction.
    **overrides
        Field-level overrides applied on top of the base config.

    Examples
    --------
        aug = build_augmentor("vicreg", manifest_min_patches=2048)
        aug = build_augmentor("lejepa", split_overlap=0.0)
        aug = build_augmentor("vicreg", crop_area_range=(0.3, 0.9))
    """
    method = method.lower()
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid: {sorted(_VALID_METHODS)}")
    if method == "lejepa_mc":
        raise ValueError(
            "build_augmentor is for dual-view methods only. "
            "Use build_multicrop_augmentor for lejepa_mc."
        )

    # Defensive copy so we never mutate the caller's cfg.
    if cfg is None:
        cfg = WSIDualViewAugmentConfig()
    else:
        cfg = replace(cfg)

    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(
                f"Unknown config field '{k}'. "
                f"Valid fields: {sorted(WSIDualViewAugmentConfig.__dataclass_fields__)}"
            )
        setattr(cfg, k, v)

    return WSIDualViewAugmentor(cfg, manifest_min_patches=manifest_min_patches)


def build_multicrop_augmentor(
    num_global: int = 2,
    num_local: int = 2,
    *,
    # Per-view fixed_n
    global_fixed_n: int = 2048,
    local_fixed_n: int = 1024,
    # Crop knobs
    global_crop_area_range: tuple[float, float] = (0.5, 1.0),
    local_crop_area_range: tuple[float, float] = (0.25, 0.60),
    global_crop_aspect_range: tuple[float, float] = (0.5, 2.0),
    local_crop_aspect_range: tuple[float, float] = (0.5, 2.0),
    global_crop_min_keep_frac: float = 0.40,
    local_crop_min_keep_frac: float = 0.40,
    # Mask knobs
    global_mask_ratio_range: tuple[float, float] = (0.10, 0.25),
    local_mask_ratio_range: tuple[float, float] = (0.05, 0.20),
    global_mask_min_keep_frac: float = 0.60,
    local_mask_min_keep_frac: float = 0.60,
    # Floor knobs
    global_min_tokens: int = 256,
    local_min_tokens: int | None = None,
    d4_mode: Literal["none", "shared"] = "shared",
    # Inter-global coupling. <1.0 routes the global pair through the dual-view
    # split-style sampler so MC = 2C + locals exactly. Only V_g=2 supported.
    split_overlap: float = 1.0,
    manifest_min_patches: int | None = None,
    **extra_overrides,
) -> WSIMultiCropAugmentor:
    """Build a LeJEPA-MC augmentor with the proposal's recommended defaults.

    Every per-view knob is wired through so the multi-crop GLOBAL views can be
    configured to match the dual-view augmentation strength used by VICReg /
    JEPA / LeJEPA-2C — preserving the controlled four-method comparison
    contract documented in `configs/pretrain_training/base.yaml`.

    ``extra_overrides`` is accepted for forward compatibility — unknown keys
    are silently ignored so the same augmentation_cfg dict can be passed for
    both 2-crop and multi-crop methods (the data module does this).
    """
    global_spec = WSIViewSpec(
        fixed_n=global_fixed_n,
        crop_area_range=global_crop_area_range,
        crop_aspect_range=global_crop_aspect_range,
        crop_min_keep_frac=global_crop_min_keep_frac,
        mask_ratio_range=global_mask_ratio_range,
        mask_min_keep_frac=global_mask_min_keep_frac,
        min_tokens=global_min_tokens,
    )
    local_spec = WSIViewSpec(
        fixed_n=local_fixed_n,
        crop_area_range=local_crop_area_range,
        crop_aspect_range=local_crop_aspect_range,
        crop_min_keep_frac=local_crop_min_keep_frac,
        mask_ratio_range=local_mask_ratio_range,
        mask_min_keep_frac=local_mask_min_keep_frac,
        # Default local floor scales with local_fixed_n if caller didn't override.
        min_tokens=local_min_tokens
        if local_min_tokens is not None
        else min(128, local_fixed_n // 8),
    )
    return WSIMultiCropAugmentor(
        num_global=num_global,
        num_local=num_local,
        global_spec=global_spec,
        local_spec=local_spec,
        d4_mode=d4_mode,
        split_overlap=split_overlap,
        manifest_min_patches=manifest_min_patches,
    )


# =============================================================================
# Legacy aliases (for backward compatibility with existing imports)
# =============================================================================

DualViewAugmentor = WSIDualViewAugmentor
SimpleAugmentConfig = WSIDualViewAugmentConfig
