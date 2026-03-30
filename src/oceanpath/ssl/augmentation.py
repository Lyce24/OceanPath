"""
Feature-space augmentation for SSL pretraining.

Four categories of augmentation, ordered from strongest to subtlest:

1. **Patch-selection augmentations** (which patches each view sees):
   - Patch subsampling — uniform random subset
   - SpatialCrop — contiguous rectangular region
   - SpatialDensitySubsample — density-aware thinning
   - SpatialRegionDrop — rectangular cutout in coordinate space
   - TokenSplit — partition into disjoint subsets (SPT-style)          [NEW]
   - Instance dropout — random per-patch drop

2. **Coords-aware feature augmentations** (perturb features using spatial structure):
   - LocalFeatureSmooth — blend with k-nearest spatial neighbors
   - SpatialFeatureInterpolation — lerp toward nearest neighbor
   - TissueRegionMixup — cross-region prototype mixing
   - SpatialBlockMask — contiguous block masking for JEPA              [NEW]

3. **Feature-only augmentations** (channel-level noise):
   - Feature noise — additive Gaussian
   - Feature posterize — quantize to discrete levels                   [NEW]
   - Feature dropout — zero random dimensions

4. **Coordinate augmentations** (change spatial layout, features unchanged):
   - CoordAffine — rotation, flip, scale
   - SpatialGridShuffle — permute within grid cells
   - Coordinate jitter — random offset

View generators (compose augmentors for multi-view SSL):
   - DualViewAugmentor — two independent views (overlapping)
   - SplitDualViewAugmentor — disjoint token split then augment        [NEW]
   - AsymmetricDualViewAugmentor — teacher (global) / student (local)  [NEW]
   - JEPAViewAugmentor — context + target spatial blocks               [NEW]
   - MultiCropAugmentor — N global + M local views (DINOv2-style)

Design principles for feature embedding augmentation
─────────────────────────────────────────────────────
- Feature embeddings are HIGH-LEVEL (ViT-H output). Heavy noise destroys
  semantics. Primary view diversity comes from WHICH patches each view sees
  (subsampling/cropping), not from corrupting embedding values.
- Coords provide SPATIAL STRUCTURE. Coords-aware augmentation lets us create
  views that differ in spatial coverage and local feature relationships,
  encouraging the aggregator to learn representations that are robust to
  which tissue regions are visible and how features vary locally.
- All augmentations preserve the invariant: both views come from the SAME
  SLIDE. The SSL loss teaches the aggregator to produce similar embeddings
  despite seeing different spatial subsets or feature perturbations.
"""

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# Coords-Aware Spatial Transforms (patch selection)
# ═════════════════════════════════════════════════════════════════════════════


class SpatialRegionDrop:
    """
    Drop all patches within k random rectangular regions (tissue cutout).

    Simulates missing tissue regions — forces the aggregator to build
    slide representations from distributed evidence, not just one hot spot.

    Parameters
    ----------
    n_regions : int
        Number of rectangular regions to drop per call.
    region_frac : tuple[float, float]
        Each region covers uniform[lo, hi] fraction of the coordinate
        bounding box in EACH axis.
    min_keep_frac : float
        Never drop more than (1 - min_keep_frac) of total patches.
    """

    def __init__(
        self,
        n_regions: int = 2,
        region_frac: tuple[float, float] = (0.05, 0.2),
        min_keep_frac: float = 0.3,
    ):
        self.n_regions = n_regions
        self.region_lo, self.region_hi = region_frac
        self.min_keep_frac = min_keep_frac

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = features.shape[0]
        min_keep = max(1, int(N * self.min_keep_frac))

        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6

        keep_mask = np.ones(N, dtype=bool)

        for _ in range(self.n_regions):
            frac_x = rng.uniform(self.region_lo, self.region_hi)
            frac_y = rng.uniform(self.region_lo, self.region_hi)
            w = span[0] * frac_x
            h = span[1] * frac_y
            cx = rng.uniform(cmin[0], cmax[0])
            cy = rng.uniform(cmin[1], cmax[1])
            in_box = (
                (coords[:, 0] >= cx - w / 2)
                & (coords[:, 0] <= cx + w / 2)
                & (coords[:, 1] >= cy - h / 2)
                & (coords[:, 1] <= cy + h / 2)
            )
            keep_mask &= ~in_box

        if keep_mask.sum() < min_keep:
            dropped = np.where(~keep_mask)[0]
            n_restore = min_keep - keep_mask.sum()
            restore_idx = rng.choice(dropped, size=min(n_restore, len(dropped)), replace=False)
            keep_mask[restore_idx] = True

        return features[keep_mask], coords[keep_mask]


class SpatialCrop:
    """
    Crop a contiguous rectangular region from coordinate space.

    Parameters
    ----------
    crop_frac : tuple[float, float]
        Each crop covers uniform[lo, hi] fraction of the bounding box PER AXIS.
    min_patches : int
        Fall back to random subsampling if crop yields fewer patches.
    """

    def __init__(
        self,
        crop_frac: tuple[float, float] = (0.4, 0.8),
        min_patches: int = 16,
    ):
        self.crop_lo, self.crop_hi = crop_frac
        self.min_patches = min_patches

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = features.shape[0]
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6

        fx = rng.uniform(self.crop_lo, self.crop_hi)
        fy = rng.uniform(self.crop_lo, self.crop_hi)
        crop_w = span[0] * fx
        crop_h = span[1] * fy

        x0 = rng.uniform(cmin[0], cmin[0] + span[0] - crop_w)
        y0 = rng.uniform(cmin[1], cmin[1] + span[1] - crop_h)

        in_crop = (
            (coords[:, 0] >= x0)
            & (coords[:, 0] <= x0 + crop_w)
            & (coords[:, 1] >= y0)
            & (coords[:, 1] <= y0 + crop_h)
        )

        if in_crop.sum() < self.min_patches:
            k = max(self.min_patches, int(N * fx * fy))
            k = min(k, N)
            idx = rng.permutation(N)[:k]
            return features[idx], coords[idx]

        return features[in_crop], coords[in_crop]


class SpatialDensitySubsample:
    """
    Subsample patches inversely proportional to local spatial density.

    Parameters
    ----------
    target_frac : tuple[float, float]
        Fraction of patches to retain, sampled uniformly from [lo, hi].
    grid_size : int
        Grid resolution for density estimation.
    temperature : float
        Controls thinning aggressiveness. 0=uniform, 1.0=full density correction.
    min_patches : int
        Minimum patches to retain.
    """

    def __init__(
        self,
        target_frac: tuple[float, float] = (0.5, 0.9),
        grid_size: int = 8,
        temperature: float = 0.7,
        min_patches: int = 16,
    ):
        self.frac_lo, self.frac_hi = target_frac
        self.grid_size = grid_size
        self.temperature = temperature
        self.min_patches = min_patches

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = features.shape[0]
        target_frac = rng.uniform(self.frac_lo, self.frac_hi)
        target_n = max(self.min_patches, int(N * target_frac))
        target_n = min(target_n, N)

        if self.min_patches >= N:
            return features, coords

        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6
        cell_size = span / self.grid_size

        cell_x = np.clip(
            ((coords[:, 0] - cmin[0]) / cell_size[0]).astype(int), 0, self.grid_size - 1
        )
        cell_y = np.clip(
            ((coords[:, 1] - cmin[1]) / cell_size[1]).astype(int), 0, self.grid_size - 1
        )
        cell_ids = cell_x * self.grid_size + cell_y

        _unique_cells, inverse, counts = np.unique(
            cell_ids, return_inverse=True, return_counts=True
        )
        patch_density = counts[inverse].astype(np.float64)

        weights = 1.0 / np.power(patch_density, self.temperature)
        weights /= weights.sum()

        idx = rng.choice(N, size=target_n, replace=False, p=weights)
        idx.sort()

        return features[idx], coords[idx]


class TokenSplit:
    """
    Partition patch tokens into two disjoint subsets (SPT-style).

    This is the key insight from SPT: by giving each view a MUTUALLY
    EXCLUSIVE subset of tokens, we reduce mutual information between
    views far more effectively than random subsampling (which allows
    overlap). The aggregator must reconstruct slide-level semantics
    from incomplete, non-overlapping evidence.

    Used at the VIEW GENERATOR level (before per-view augmentation),
    not inside FeatureAugmentor. The split happens once, then each
    subset is independently augmented.

    Parameters
    ----------
    split_ratio : float
        Fraction of tokens going to set A (rest go to set B).
        0.5 = equal split. Use 0.5 for SimCLR/VICReg (symmetric),
        or asymmetric (e.g., 0.6/0.4) for BYOL-style.
    """

    def __init__(self, split_ratio: float = 0.5):
        if not 0.1 <= split_ratio <= 0.9:
            raise ValueError(f"split_ratio must be in [0.1, 0.9], got {split_ratio}")
        self.split_ratio = split_ratio

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """
        Split (N, D) features + (N, 2) coords into two disjoint sets.

        Returns
        -------
        (set_a_features, set_a_coords), (set_b_features, set_b_coords)
        """
        N = features.shape[0]
        n_a = max(1, int(N * self.split_ratio))
        n_a = min(n_a, N - 1)  # ensure set B gets at least 1

        perm = rng.permutation(N)
        idx_a = np.sort(perm[:n_a])
        idx_b = np.sort(perm[n_a:])

        return (features[idx_a], coords[idx_a]), (features[idx_b], coords[idx_b])


class SpatialBlockMask:
    """
    Mask contiguous spatial blocks for JEPA-style predictive pretraining.

    Fundamentally different from SpatialRegionDrop:
    - SpatialRegionDrop DISCARDS patches (they vanish from both views).
    - SpatialBlockMask PARTITIONS patches into context (visible) and
      target (masked) sets. The predictor must predict target block
      representations from the context representation.

    Block sampling: divide coordinate space into a grid, randomly select
    contiguous rectangular groups of cells as target blocks. Each block
    is a spatially contiguous region of tissue.

    Parameters
    ----------
    n_blocks : int
        Number of target blocks to mask.
    block_size_range : tuple[int, int]
        Each block spans [lo, hi] grid cells per axis.
    grid_size : int
        Grid resolution for block sampling.
    min_context_frac : float
        Minimum fraction of patches that must remain as context.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        block_size_range: tuple[int, int] = (1, 3),
        grid_size: int = 8,
        min_context_frac: float = 0.5,
    ):
        self.n_blocks = n_blocks
        self.block_lo, self.block_hi = block_size_range
        self.grid_size = grid_size
        self.min_context_frac = min_context_frac

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],  # context (features, coords)
        tuple[np.ndarray, np.ndarray],  # target  (features, coords)
    ]:
        """
        Partition patches into context (visible) and target (masked) sets.

        Returns
        -------
        (context_features, context_coords), (target_features, target_coords)

        The target coords are essential — they tell the predictor WHERE
        to predict, which is how positional queries work in I-JEPA.
        """
        N = features.shape[0]
        min_context = max(1, int(N * self.min_context_frac))

        # Grid-bin all patches
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6
        cell_size = span / self.grid_size

        cell_x = np.clip(
            ((coords[:, 0] - cmin[0]) / cell_size[0]).astype(int), 0, self.grid_size - 1
        )
        cell_y = np.clip(
            ((coords[:, 1] - cmin[1]) / cell_size[1]).astype(int), 0, self.grid_size - 1
        )

        # Track which patches are masked (target)
        target_mask = np.zeros(N, dtype=bool)

        for _ in range(self.n_blocks):
            bw = rng.integers(self.block_lo, self.block_hi + 1)
            bh = rng.integers(self.block_lo, self.block_hi + 1)

            x0 = rng.integers(0, max(1, self.grid_size - bw + 1))
            y0 = rng.integers(0, max(1, self.grid_size - bh + 1))

            in_block = (cell_x >= x0) & (cell_x < x0 + bw) & (cell_y >= y0) & (cell_y < y0 + bh)

            # Snapshot before merge so undo doesn't leak
            prev_mask = target_mask.copy()
            target_mask |= in_block

            if (N - target_mask.sum()) < min_context:
                target_mask = prev_mask
                break

        # If no patches were masked, force at least one small block
        if target_mask.sum() == 0:
            # Pick a random occupied cell and mask it
            occupied = np.unique(cell_x * self.grid_size + cell_y)
            if len(occupied) > 1:
                chosen_cell = occupied[rng.integers(len(occupied))]
                cx_target = chosen_cell // self.grid_size
                cy_target = chosen_cell % self.grid_size
                target_mask = (cell_x == cx_target) & (cell_y == cy_target)

        context_mask = ~target_mask

        # Safety: ensure context has enough patches
        if context_mask.sum() < min_context:
            # Reduce target to satisfy constraint
            target_indices = np.where(target_mask)[0]
            n_give_back = min_context - context_mask.sum()
            give_back = rng.choice(
                target_indices, size=min(n_give_back, len(target_indices)), replace=False
            )
            target_mask[give_back] = False
            context_mask[give_back] = True

        context_feat = features[context_mask]
        context_coord = coords[context_mask]
        target_feat = features[target_mask]
        target_coord = coords[target_mask]

        # Ensure non-empty returns
        if target_feat.shape[0] == 0:
            # Fallback: take 10% random as target
            n_target = max(1, N // 10)
            perm = rng.permutation(N)
            target_idx = np.sort(perm[:n_target])
            context_idx = np.sort(perm[n_target:])
            return (features[context_idx], coords[context_idx]), (
                features[target_idx],
                coords[target_idx],
            )

        return (context_feat, context_coord), (target_feat, target_coord)


class CoordAffine:
    """
    Apply random affine transforms to coordinates.

    Transforms: random rotation (90° multiples), random flip, random scale.
    Features stay unchanged — only spatial layout changes.

    Parameters
    ----------
    enable_rotation : bool
        Random rotation by a multiple of 90°.
    enable_flip : bool
        Random horizontal and/or vertical flip.
    enable_scale : tuple[float, float] or None
        Random scale factor per axis: uniform[lo, hi]. None = no scaling.
    """

    def __init__(
        self,
        enable_rotation: bool = True,
        enable_flip: bool = True,
        enable_scale: tuple[float, float] | None = (0.8, 1.2),
    ):
        self.enable_rotation = enable_rotation
        self.enable_flip = enable_flip
        self.enable_scale = enable_scale

    def __call__(self, coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = coords.astype(np.float64)
        center = c.mean(axis=0)
        c = c - center

        if self.enable_rotation:
            k = rng.integers(0, 4)
            if k == 1:
                c = np.stack([-c[:, 1], c[:, 0]], axis=1)
            elif k == 2:
                c = -c
            elif k == 3:
                c = np.stack([c[:, 1], -c[:, 0]], axis=1)

        if self.enable_flip:
            if rng.random() < 0.5:
                c[:, 0] = -c[:, 0]
            if rng.random() < 0.5:
                c[:, 1] = -c[:, 1]

        if self.enable_scale is not None:
            sx = rng.uniform(*self.enable_scale)
            sy = rng.uniform(*self.enable_scale)
            c[:, 0] *= sx
            c[:, 1] *= sy

        c = c + center
        return c.astype(coords.dtype)


# ═════════════════════════════════════════════════════════════════════════════
# Coords-Aware Feature Transforms (perturb features using spatial structure)
# ═════════════════════════════════════════════════════════════════════════════


class LocalFeatureSmooth:
    """
    Blend each patch's features with its k-nearest spatial neighbors.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest spatial neighbors to blend with.
    alpha_range : tuple[float, float]
        Blend weight: feat = (1-alpha)*feat + alpha*mean(neighbors).
    frac_patches : float
        Fraction of patches to apply smoothing to.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        alpha_range: tuple[float, float] = (0.05, 0.3),
        frac_patches: float = 0.5,
    ):
        self.k_neighbors = k_neighbors
        self.alpha_lo, self.alpha_hi = alpha_range
        self.frac_patches = frac_patches

    def __call__(
        self, features: np.ndarray, coords: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        from scipy.spatial import cKDTree

        N, _D = features.shape
        if self.k_neighbors + 1 >= N:
            return features

        k = min(self.k_neighbors, N - 1)
        alpha = rng.uniform(self.alpha_lo, self.alpha_hi)

        tree = cKDTree(coords.astype(np.float64))
        # query k+1 because the point itself is the nearest neighbor at dist 0
        _, neighbor_idx = tree.query(coords.astype(np.float64), k=k + 1)
        neighbor_idx = neighbor_idx[:, 1:]  # drop self

        neighbor_mean = features[neighbor_idx].mean(axis=1)

        n_smooth = max(1, int(N * self.frac_patches))
        smooth_idx = rng.choice(N, size=n_smooth, replace=False)

        out = features.copy()
        out[smooth_idx] = (1 - alpha) * features[smooth_idx] + alpha * neighbor_mean[smooth_idx]
        return out.astype(features.dtype)


class SpatialFeatureInterpolation:
    """
    Lerp each selected patch's feature toward its nearest spatial neighbor.

    Parameters
    ----------
    t_range : tuple[float, float]
        Interpolation weight sampled from uniform[lo, hi].
    frac_patches : float
        Fraction of patches to interpolate.
    neighbor_rank_range : tuple[int, int]
        Randomly pick between the k-th nearest neighbor.
    """

    def __init__(
        self,
        t_range: tuple[float, float] = (0.05, 0.25),
        frac_patches: float = 0.5,
        neighbor_rank_range: tuple[int, int] = (1, 3),
    ):
        self.t_lo, self.t_hi = t_range
        self.frac_patches = frac_patches
        self.rank_lo, self.rank_hi = neighbor_rank_range

    def __call__(
        self, features: np.ndarray, coords: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        from scipy.spatial import cKDTree

        N, _D = features.shape
        max_rank = min(self.rank_hi, N - 1)
        if N < 2 or max_rank < 1:
            return features

        topk = min(max_rank + 1, N)
        tree = cKDTree(coords.astype(np.float64))
        # k=topk+1 to include self, then drop column 0 (self)
        _, nn_idx = tree.query(coords.astype(np.float64), k=topk + 1)
        nn_idx = nn_idx[:, 1:]  # (N, topk), already sorted by distance

        n_interp = max(1, int(N * self.frac_patches))
        interp_idx = rng.choice(N, size=n_interp, replace=False)

        # Vectorized: sample a random rank per selected patch
        ranks = rng.integers(self.rank_lo, max_rank + 1, size=n_interp)
        col_idx = np.clip(ranks - 1, 0, topk - 1)
        t_vals = rng.uniform(self.t_lo, self.t_hi, size=n_interp)

        nn_selected = nn_idx[interp_idx, col_idx]
        out = features.copy()
        out[interp_idx] = (1 - t_vals[:, None]) * features[interp_idx] + t_vals[:, None] * features[
            nn_selected
        ]
        return out.astype(features.dtype)


class TissueRegionMixup:
    """
    Mix features across spatial regions to break region-specific signatures.

    Parameters
    ----------
    grid_size : int
        Grid resolution for region definition.
    alpha_range : tuple[float, float]
        Blend weight with foreign region prototype.
    frac_patches : float
        Fraction of patches to apply mixup to.
    min_region_patches : int
        Minimum patches in a region for a valid prototype.
    """

    def __init__(
        self,
        grid_size: int = 4,
        alpha_range: tuple[float, float] = (0.05, 0.2),
        frac_patches: float = 0.3,
        min_region_patches: int = 4,
    ):
        self.grid_size = grid_size
        self.alpha_lo, self.alpha_hi = alpha_range
        self.frac_patches = frac_patches
        self.min_region_patches = min_region_patches

    def __call__(
        self, features: np.ndarray, coords: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        N, _D = features.shape
        if self.min_region_patches * 2 > N:
            return features

        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6
        cell_size = span / self.grid_size

        cell_x = np.clip(
            ((coords[:, 0] - cmin[0]) / cell_size[0]).astype(int), 0, self.grid_size - 1
        )
        cell_y = np.clip(
            ((coords[:, 1] - cmin[1]) / cell_size[1]).astype(int), 0, self.grid_size - 1
        )
        cell_ids = cell_x * self.grid_size + cell_y

        unique_cells = np.unique(cell_ids)
        prototypes = {}
        for cid in unique_cells:
            mask = cell_ids == cid
            if mask.sum() >= self.min_region_patches:
                prototypes[cid] = features[mask].astype(np.float64).mean(axis=0)

        valid_cells = list(prototypes.keys())
        if len(valid_cells) < 2:
            return features

        n_mix = max(1, int(N * self.frac_patches))
        mix_idx = rng.choice(N, size=n_mix, replace=False)
        alpha = rng.uniform(self.alpha_lo, self.alpha_hi)

        out = features.copy()
        for idx in mix_idx:
            own_cell = cell_ids[idx]
            candidates = [c for c in valid_cells if c != own_cell]
            if not candidates:
                continue
            foreign_cell = candidates[rng.integers(len(candidates))]
            foreign_proto = prototypes[foreign_cell]
            out[idx] = (1 - alpha) * features[idx] + alpha * foreign_proto

        return out.astype(features.dtype)


class FeaturePosterize:
    """
    Quantize feature values to discrete levels (feature-space posterize).

    Reduces the precision of embedding values, forcing the aggregator to
    be invariant to fine-grained feature differences. Analogous to image
    posterization but applied in the learned feature space.

    Used by TITAN as a feature-space augmentation. Should be applied
    AFTER spatial feature transforms (smooth/interpolate) but BEFORE
    feature noise, since posterize is a deterministic quantization and
    noise is additive.

    Parameters
    ----------
    n_levels : int
        Number of quantization levels. Lower = stronger augmentation.
        8-16 for subtle, 4-6 for moderate, 2-3 for aggressive.
    prob : float
        Probability of applying posterize on each call.
    """

    def __init__(self, n_levels: int = 8, prob: float = 0.3):
        self.n_levels = n_levels
        self.prob = prob

    def __call__(self, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.prob:
            return features

        f = features.astype(np.float32)

        # Per-dimension min/max for quantization range
        f_min = f.min(axis=0, keepdims=True)
        f_max = f.max(axis=0, keepdims=True)
        f_range = f_max - f_min + 1e-8

        # Normalize to [0, 1], quantize, denormalize
        normalized = (f - f_min) / f_range
        quantized = np.floor(normalized * self.n_levels).clip(0, self.n_levels - 1)
        denormalized = ((quantized + 0.5) / self.n_levels) * f_range + f_min

        return denormalized


class SpatialGridShuffle:
    """
    Divide coordinate space into a grid, then permute patches within each cell.

    Parameters
    ----------
    grid_size : int
        Number of grid cells per axis.
    shuffle_prob : float
        Probability that each cell gets shuffled.
    """

    def __init__(self, grid_size: int = 4, shuffle_prob: float = 0.7):
        self.grid_size = grid_size
        self.shuffle_prob = shuffle_prob

    def __call__(self, coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6
        cell_size = span / self.grid_size

        cell_x = np.clip(
            ((coords[:, 0] - cmin[0]) / cell_size[0]).astype(int), 0, self.grid_size - 1
        )
        cell_y = np.clip(
            ((coords[:, 1] - cmin[1]) / cell_size[1]).astype(int), 0, self.grid_size - 1
        )
        cell_ids = cell_x * self.grid_size + cell_y

        new_coords = coords.copy()
        for cell_id in np.unique(cell_ids):
            if rng.random() > self.shuffle_prob:
                continue
            mask = cell_ids == cell_id
            indices = np.where(mask)[0]
            if len(indices) < 2:
                continue
            perm = rng.permutation(len(indices))
            new_coords[indices] = coords[indices[perm]]

        return new_coords


# ═════════════════════════════════════════════════════════════════════════════
# Main Augmentor (composes all transforms for a SINGLE view)
# ═════════════════════════════════════════════════════════════════════════════


class FeatureAugmentor:
    """
    Stochastic feature-space augmentation for one view.

    Pipeline order:
      PATCH SELECTION  → FEATURE PERTURBATION → COORDINATE PERTURBATION

      ┌─ PATCH SELECTION ─────────────────────────────────────────────────┐
      │ 1. Spatial crop OR density subsample OR random subsampling        │
      │ 2. Spatial region drop (cutout)                                   │
      │ 3. Instance dropout                                               │
      └───────────────────────────────────────────────────────────────────┘
      ┌─ FEATURE PERTURBATION ────────────────────────────────────────────┐
      │ 4. Local feature smoothing (blend with spatial neighbors)         │
      │ 5. Spatial feature interpolation (lerp toward nearest neighbor)   │
      │ 6. Tissue region mixup (cross-region prototype mixing)            │
      │ 7. Feature posterize (quantize to discrete levels)        [NEW]   │
      │ 8. Feature noise (Gaussian)                                       │
      │ 9. Feature dropout (zero dimensions)                              │
      └───────────────────────────────────────────────────────────────────┘
      ┌─ COORDINATE PERTURBATION ─────────────────────────────────────────┐
      │ 10. Coord affine (rotation/flip/scale)                            │
      │ 11. Spatial grid shuffle                                          │
      │ 12. Coordinate jitter                                             │
      └───────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        subsample_frac: tuple[float, float] = (0.5, 1.0),
        instance_dropout: float = 0.1,
        feature_noise_std: float = 0.02,
        feature_dropout: float = 0.05,
        coord_jitter: int = 0,
        # Coords-aware patch selection
        spatial_crop: SpatialCrop | None = None,
        spatial_density_sub: SpatialDensitySubsample | None = None,
        spatial_region_drop: SpatialRegionDrop | None = None,
        # Coords-aware feature transforms
        coord_affine: CoordAffine | None = None,
        local_smooth: LocalFeatureSmooth | None = None,
        spatial_interpolation: SpatialFeatureInterpolation | None = None,
        region_mixup: TissueRegionMixup | None = None,
        feature_posterize: FeaturePosterize | None = None,
        # Coords-aware coord transforms
        grid_shuffle: SpatialGridShuffle | None = None,
        # Selection strategy flags
        use_spatial_crop: bool = False,
        spatial_crop_prob: float = 0.5,
        use_density_sub: bool = False,
        density_sub_prob: float = 0.3,
    ):
        self.subsample_lo, self.subsample_hi = subsample_frac
        self.instance_dropout = instance_dropout
        self.feature_noise_std = feature_noise_std
        self.feature_dropout = feature_dropout
        self.coord_jitter = coord_jitter

        self.spatial_crop = spatial_crop
        self.spatial_density_sub = spatial_density_sub
        self.spatial_region_drop = spatial_region_drop
        self.coord_affine = coord_affine
        self.local_smooth = local_smooth
        self.spatial_interpolation = spatial_interpolation
        self.region_mixup = region_mixup
        self.feature_posterize = feature_posterize
        self.grid_shuffle = grid_shuffle

        self.use_spatial_crop = use_spatial_crop
        self.spatial_crop_prob = spatial_crop_prob
        self.use_density_sub = use_density_sub
        self.density_sub_prob = density_sub_prob

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if rng is None:
            rng = np.random.default_rng()

        N, D = features.shape
        original_dtype = features.dtype
        has_coords = coords is not None

        # ── Step 1: Patch selection ──────────────────────────────────────
        used_density = False
        used_crop = False

        if (
            has_coords
            and self.use_density_sub
            and self.spatial_density_sub is not None
            and rng.random() < self.density_sub_prob
        ):
            features, coords = self.spatial_density_sub(features, coords, rng)
            used_density = True

        if (
            not used_density
            and has_coords
            and self.use_spatial_crop
            and self.spatial_crop is not None
            and rng.random() < self.spatial_crop_prob
        ):
            features, coords = self.spatial_crop(features, coords, rng)
            used_crop = True

        if not used_density and not used_crop:
            frac = rng.uniform(self.subsample_lo, self.subsample_hi)
            k = max(1, int(N * frac))
            indices = rng.permutation(N)[:k]
            features = features[indices].copy()
            if coords is not None:
                coords = coords[indices].copy()

        N_sub = features.shape[0]

        # ── Step 2: Spatial region drop ──────────────────────────────────
        if has_coords and self.spatial_region_drop is not None and coords is not None:
            features, coords = self.spatial_region_drop(features, coords, rng)
            N_sub = features.shape[0]

        # ── Step 3: Instance dropout ─────────────────────────────────────
        if self.instance_dropout > 0 and N_sub > 1:
            keep = rng.random(N_sub) > self.instance_dropout
            # Guarantee at least 16 patches survive (or all if N_sub < 16)
            min_keep = min(16, N_sub)
            if keep.sum() < min_keep:
                dropped = np.where(~keep)[0]
                n_restore = min_keep - keep.sum()
                restore = rng.choice(dropped, size=min(n_restore, len(dropped)), replace=False)
                keep[restore] = True
            features = features[keep]
            if coords is not None:
                coords = coords[keep]

        # ── Cast to working dtype once for all feature perturbation ──
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        # ── Step 4: Local feature smoothing ──────────────────────────
        if has_coords and self.local_smooth is not None and coords is not None:
            features = self.local_smooth(features, coords, rng)

        # ── Step 5: Spatial feature interpolation ────────────────────
        if has_coords and self.spatial_interpolation is not None and coords is not None:
            features = self.spatial_interpolation(features, coords, rng)

        # ── Step 6: Tissue region mixup ──────────────────────────────
        if has_coords and self.region_mixup is not None and coords is not None:
            features = self.region_mixup(features, coords, rng)

        # ── Step 7: Feature posterize ────────────────────────────────
        if self.feature_posterize is not None:
            features = self.feature_posterize(features, rng)

        # ── Step 8: Feature noise ────────────────────────────────────
        if self.feature_noise_std > 0:
            features += (
                rng.standard_normal(features.shape, dtype=np.float32) * self.feature_noise_std
            )

        # ── Step 9: Feature dropout ──────────────────────────────────
        if self.feature_dropout > 0:
            dim_mask = rng.random(D) > self.feature_dropout
            features *= dim_mask.astype(np.float32)

        # ── Cast back once ───────────────────────────────────────────
        if features.dtype != original_dtype:
            features = features.astype(original_dtype)

        # ── Step 10: Coord affine ────────────────────────────────────────
        if has_coords and self.coord_affine is not None and coords is not None:
            coords = self.coord_affine(coords, rng)

        # ── Step 11: Spatial grid shuffle ────────────────────────────────
        if has_coords and self.grid_shuffle is not None and coords is not None:
            coords = self.grid_shuffle(coords, rng)

        # ── Step 12: Coordinate jitter ───────────────────────────────────
        if self.coord_jitter > 0 and coords is not None:
            jitter = rng.integers(-self.coord_jitter, self.coord_jitter + 1, size=coords.shape)
            coords = coords + jitter

        return features, coords


# ═════════════════════════════════════════════════════════════════════════════
# View Generators (compose FeatureAugmentor for multi-view SSL)
# ═════════════════════════════════════════════════════════════════════════════


class DualViewAugmentor:
    """
    Two augmented views from one slide (independent, overlapping).

    Both views are drawn from the full slide independently — they CAN
    share patches. Suitable as a baseline or when token splitting is
    not desired.
    """

    def __init__(self, augmentor: FeatureAugmentor):
        self.augmentor = augmentor

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]]:
        if rng is None:
            rng = np.random.default_rng()
        child_rngs = rng.spawn(2)
        view1 = self.augmentor(features, coords, rng=child_rngs[0])
        view2 = self.augmentor(features, coords, rng=child_rngs[1])
        return view1, view2


class SplitDualViewAugmentor:
    """
    Disjoint token split → independent augmentation per subset.

    SPT-style view generation: first partition ALL tokens into two
    mutually exclusive sets, then apply per-view augmentation to each
    subset independently. This guarantees zero overlap between views,
    maximizing information content of the contrastive signal.

    Recommended for: SimCLR, VICReg (symmetric objectives).

    Parameters
    ----------
    augmentor : FeatureAugmentor
        Per-view augmentation pipeline (applied AFTER split).
    split_ratio : float
        Fraction of tokens to set A. Default 0.5 (equal split).
    """

    def __init__(self, augmentor: FeatureAugmentor, split_ratio: float = 0.5):
        self.augmentor = augmentor
        self.splitter = TokenSplit(split_ratio=split_ratio)

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]]:
        if rng is None:
            rng = np.random.default_rng()

        child_rngs = rng.spawn(3)  # 1 for split, 2 for per-view augmentation

        if coords is None:
            # Fallback: no split possible without coords, use DualView
            view1 = self.augmentor(features, coords, rng=child_rngs[1])
            view2 = self.augmentor(features, coords, rng=child_rngs[2])
            return view1, view2

        (feat_a, coord_a), (feat_b, coord_b) = self.splitter(features, coords, child_rngs[0])

        view1 = self.augmentor(feat_a, coord_a, rng=child_rngs[1])
        view2 = self.augmentor(feat_b, coord_b, rng=child_rngs[2])
        return view1, view2


class AsymmetricDualViewAugmentor:
    """
    Asymmetric teacher/student views for BYOL / DINO-style pretraining.

    Teacher sees a LARGE, lightly augmented view (more context).
    Student sees a SMALL, heavily augmented view (harder task).

    The teacher augmentor should use large crop fractions, low noise,
    and no posterize. The student augmentor should use small crop
    fractions, higher noise, and posterize.

    Optionally applies token splitting before the asymmetric augmentation,
    further reducing mutual information between views.

    Parameters
    ----------
    teacher_augmentor : FeatureAugmentor
        Large crop, light augmentation.
    student_augmentor : FeatureAugmentor
        Small crop, heavy augmentation.
    use_split : bool
        If True, split tokens first, then apply teacher/student
        augmentors to the two subsets. If False, both augmentors
        operate on the full slide independently.
    split_ratio : float
        Only used when use_split=True. Fraction for teacher set.
    """

    def __init__(
        self,
        teacher_augmentor: FeatureAugmentor,
        student_augmentor: FeatureAugmentor,
        use_split: bool = False,
        split_ratio: float = 0.6,
    ):
        self.teacher_aug = teacher_augmentor
        self.student_aug = student_augmentor
        self.use_split = use_split
        self.splitter = TokenSplit(split_ratio=split_ratio) if use_split else None

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]]:
        if rng is None:
            rng = np.random.default_rng()

        child_rngs = rng.spawn(3)

        if self.use_split and self.splitter is not None and coords is not None:
            (feat_t, coord_t), (feat_s, coord_s) = self.splitter(features, coords, child_rngs[0])
            teacher_view = self.teacher_aug(feat_t, coord_t, rng=child_rngs[1])
            student_view = self.student_aug(feat_s, coord_s, rng=child_rngs[2])
        else:
            teacher_view = self.teacher_aug(features, coords, rng=child_rngs[1])
            student_view = self.student_aug(features, coords, rng=child_rngs[2])

        # Convention: view1 = teacher, view2 = student
        return teacher_view, student_view


class JEPAViewAugmentor:
    """
    Context + target spatial block generation for I-JEPA pretraining.

    Fundamentally different from contrastive view generators:
    - Does NOT produce two "similar" views for a contrastive loss.
    - Instead PARTITIONS patches into context (visible) and target (masked).
    - The predictor must predict target representations from context.

    Pipeline:
      1. Optional pre-subsample (if slide is very large)
      2. SpatialBlockMask → context patches, target patches
      3. CoordAffine on both (same transform for spatial consistency)
      4. Light feature augmentation on CONTEXT only (target stays clean)

    Parameters
    ----------
    block_mask : SpatialBlockMask
        Spatial block masking strategy.
    context_augmentor : FeatureAugmentor or None
        Light augmentation for context view. If None, context is unaugmented.
    coord_affine : CoordAffine or None
        Applied to BOTH context and target consistently (same transform).
    pre_subsample : int or None
        If slide > pre_subsample patches, random subsample first.
    """

    def __init__(
        self,
        block_mask: SpatialBlockMask | None = None,
        context_augmentor: FeatureAugmentor | None = None,
        coord_affine: CoordAffine | None = None,
        pre_subsample: int | None = 4096,
    ):
        self.block_mask = block_mask or SpatialBlockMask()
        self.context_aug = context_augmentor
        self.coord_affine = coord_affine
        self.pre_subsample = pre_subsample

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]]:
        """
        Returns (context_features, context_coords), (target_features, target_coords).

        In the training loop:
          - context → context_encoder → context_repr
          - target  → target_encoder (EMA) → target_repr
          - predictor(context_repr, target_coords) → predicted_target_repr
          - loss = MSE(predicted_target_repr, target_repr.detach())
        """
        if rng is None:
            rng = np.random.default_rng()

        child_rngs = rng.spawn(3)
        N = features.shape[0]

        # Pre-subsample for large slides
        if self.pre_subsample is not None and self.pre_subsample < N and coords is not None:
            idx = np.sort(child_rngs[0].permutation(N)[: self.pre_subsample])
            features = features[idx]
            coords = coords[idx]

        if coords is None:
            # Fallback without coords: random split 70/30
            n_ctx = max(1, int(features.shape[0] * 0.7))
            perm = child_rngs[0].permutation(features.shape[0])
            ctx_idx = np.sort(perm[:n_ctx])
            tgt_idx = np.sort(perm[n_ctx:])
            return (features[ctx_idx], None), (features[tgt_idx], None)

        # Spatial block masking
        (ctx_feat, ctx_coord), (tgt_feat, tgt_coord) = self.block_mask(
            features, coords, child_rngs[0]
        )

        # Apply SAME coord affine to both (spatial consistency)
        if self.coord_affine is not None:
            # Use same rng state for both so they get the same transform
            affine_rng = child_rngs[1]
            state = affine_rng.bit_generator.state
            ctx_coord = self.coord_affine(ctx_coord, affine_rng)
            affine_rng.bit_generator.state = state  # reset to replay same transform
            tgt_coord = self.coord_affine(tgt_coord, affine_rng)

        # Light augmentation on context ONLY (target must stay clean for prediction)
        if self.context_aug is not None:
            ctx_feat, ctx_coord = self.context_aug(ctx_feat, ctx_coord, rng=child_rngs[2])

        return (ctx_feat, ctx_coord), (tgt_feat, tgt_coord)


class MultiCropAugmentor:
    """
    Multiple views with different subset sizes (DINOv2 multi-crop).

    Global views: large patch subsets. Local views: small patch subsets.

    Parameters
    ----------
    n_global, n_local : int
        Number of global and local views.
    global_frac, local_frac : tuple[float, float]
        Subsample fraction ranges for each view type.
    """

    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 4,
        global_frac: tuple[float, float] = (0.7, 1.0),
        local_frac: tuple[float, float] = (0.2, 0.5),
        instance_dropout: float = 0.1,
        feature_noise_std: float = 0.02,
        feature_dropout: float = 0.05,
        coord_affine: CoordAffine | None = None,
        local_smooth: LocalFeatureSmooth | None = None,
        spatial_region_drop: SpatialRegionDrop | None = None,
        spatial_interpolation: SpatialFeatureInterpolation | None = None,
        region_mixup: TissueRegionMixup | None = None,
        feature_posterize: FeaturePosterize | None = None,
    ):
        self.n_global = n_global
        self.n_local = n_local

        self.global_aug = FeatureAugmentor(
            subsample_frac=global_frac,
            instance_dropout=instance_dropout,
            feature_noise_std=feature_noise_std,
            feature_dropout=feature_dropout,
            coord_affine=coord_affine,
            local_smooth=local_smooth,
            spatial_region_drop=spatial_region_drop,
            spatial_interpolation=spatial_interpolation,
            region_mixup=region_mixup,
            feature_posterize=feature_posterize,
        )
        self.local_aug = FeatureAugmentor(
            subsample_frac=local_frac,
            instance_dropout=instance_dropout,
            feature_noise_std=feature_noise_std * 1.5,
            feature_dropout=feature_dropout,
            coord_affine=coord_affine,
            local_smooth=local_smooth,
            spatial_interpolation=spatial_interpolation,
            feature_posterize=feature_posterize,
        )

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        if rng is None:
            rng = np.random.default_rng()
        n_total = self.n_global + self.n_local
        child_rngs = rng.spawn(n_total)
        views = []
        for i in range(self.n_global):
            views.append(self.global_aug(features, coords, rng=child_rngs[i]))
        for i in range(self.n_local):
            views.append(self.local_aug(features, coords, rng=child_rngs[self.n_global + i]))
        return views


# ═════════════════════════════════════════════════════════════════════════════
# Factory: build augmentor + view generator from config
# ═════════════════════════════════════════════════════════════════════════════


def _build_single_augmentor(aug_cfg: dict, coords_aware: bool = True) -> FeatureAugmentor:
    """Build one FeatureAugmentor from config dict."""
    kwargs = {
        "subsample_frac": tuple(aug_cfg.get("subsample_frac", [0.5, 1.0])),
        "instance_dropout": aug_cfg.get("instance_dropout", 0.1),
        "feature_noise_std": aug_cfg.get("feature_noise_std", 0.02),
        "feature_dropout": aug_cfg.get("feature_dropout", 0.05),
        "coord_jitter": aug_cfg.get("coord_jitter", 0),
    }

    if coords_aware:
        kwargs.update(
            spatial_crop=SpatialCrop(
                crop_frac=tuple(aug_cfg.get("spatial_crop_frac", [0.4, 0.8])),
                min_patches=aug_cfg.get("spatial_crop_min_patches", 16),
            ),
            spatial_density_sub=SpatialDensitySubsample(
                target_frac=tuple(aug_cfg.get("density_sub_frac", [0.5, 0.9])),
                grid_size=aug_cfg.get("density_sub_grid", 8),
                temperature=aug_cfg.get("density_sub_temp", 0.7),
                min_patches=aug_cfg.get("density_sub_min_patches", 16),
            ),
            spatial_region_drop=SpatialRegionDrop(
                n_regions=aug_cfg.get("region_drop_n", 2),
                region_frac=tuple(aug_cfg.get("region_drop_frac", [0.05, 0.2])),
                min_keep_frac=aug_cfg.get("region_drop_min_keep", 0.3),
            ),
            coord_affine=CoordAffine(
                enable_rotation=aug_cfg.get("coord_rotation", True),
                enable_flip=aug_cfg.get("coord_flip", True),
                enable_scale=(
                    tuple(aug_cfg.get("coord_scale", [0.8, 1.2]))
                    if aug_cfg.get("coord_scale_enabled", True)
                    else None
                ),
            ),
            local_smooth=(
                LocalFeatureSmooth(
                    k_neighbors=aug_cfg.get("smooth_k_neighbors", 5),
                    alpha_range=tuple(aug_cfg.get("smooth_alpha", [0.05, 0.3])),
                    frac_patches=aug_cfg.get("smooth_frac_patches", 0.5),
                )
                if aug_cfg.get("enable_local_smooth", False)
                else None
            ),
            spatial_interpolation=(
                SpatialFeatureInterpolation(
                    t_range=tuple(aug_cfg.get("interp_t_range", [0.05, 0.25])),
                    frac_patches=aug_cfg.get("interp_frac_patches", 0.5),
                    neighbor_rank_range=tuple(aug_cfg.get("interp_neighbor_ranks", [1, 3])),
                )
                if aug_cfg.get("enable_spatial_interpolation", False)
                else None
            ),
            region_mixup=(
                TissueRegionMixup(
                    grid_size=aug_cfg.get("region_mixup_grid", 4),
                    alpha_range=tuple(aug_cfg.get("region_mixup_alpha", [0.05, 0.2])),
                    frac_patches=aug_cfg.get("region_mixup_frac", 0.3),
                    min_region_patches=aug_cfg.get("region_mixup_min_patches", 4),
                )
                if aug_cfg.get("enable_region_mixup", False)
                else None
            ),
            feature_posterize=FeaturePosterize(
                n_levels=aug_cfg.get("posterize_levels", 8),
                prob=aug_cfg.get("posterize_prob", 0.3),
            ),
            grid_shuffle=SpatialGridShuffle(
                grid_size=aug_cfg.get("grid_shuffle_size", 4),
                shuffle_prob=aug_cfg.get("grid_shuffle_prob", 0.7),
            ),
            use_spatial_crop=aug_cfg.get("use_spatial_crop", True),
            spatial_crop_prob=aug_cfg.get("spatial_crop_prob", 0.5),
            use_density_sub=aug_cfg.get("use_density_sub", True),
            density_sub_prob=aug_cfg.get("density_sub_prob", 0.3),
        )

    return FeatureAugmentor(**kwargs)


def build_augmentor(aug_cfg: dict, coords_aware: bool = True) -> FeatureAugmentor:
    """Build a single FeatureAugmentor (backward-compatible API)."""
    return _build_single_augmentor(aug_cfg, coords_aware)


def build_view_generator(
    aug_cfg: dict,
    view_strategy: str = "dual",
    coords_aware: bool = True,
):
    """
    Build a view generator from config.

    Parameters
    ----------
    aug_cfg : dict
        Augmentation config from YAML.
    view_strategy : str
        One of: "dual", "split_dual", "asymmetric", "jepa", "multicrop".
    coords_aware : bool
        Enable coordinate-aware transforms.

    Returns
    -------
    View generator callable with signature:
        (features, coords, rng) -> (view1, view2) or (context, target)
    """
    if view_strategy == "dual":
        augmentor = _build_single_augmentor(aug_cfg, coords_aware)
        return DualViewAugmentor(augmentor)

    if view_strategy == "split_dual":
        augmentor = _build_single_augmentor(aug_cfg, coords_aware)
        return SplitDualViewAugmentor(
            augmentor,
            split_ratio=aug_cfg.get("split_ratio", 0.5),
        )

    if view_strategy == "asymmetric":
        # Teacher: large crops, light augmentation
        teacher_cfg = dict(aug_cfg)
        teacher_cfg.update(
            {
                "subsample_frac": aug_cfg.get("teacher_subsample_frac", [0.7, 1.0]),
                "spatial_crop_frac": aug_cfg.get("teacher_crop_frac", [0.6, 1.0]),
                "instance_dropout": aug_cfg.get("teacher_instance_dropout", 0.05),
                "feature_noise_std": aug_cfg.get("teacher_feature_noise_std", 0.01),
                "feature_dropout": aug_cfg.get("teacher_feature_dropout", 0.02),
                "posterize_prob": 0.0,  # no posterize on teacher
            }
        )
        # Student: small crops, heavy augmentation
        student_cfg = dict(aug_cfg)
        student_cfg.update(
            {
                "subsample_frac": aug_cfg.get("student_subsample_frac", [0.2, 0.5]),
                "spatial_crop_frac": aug_cfg.get("student_crop_frac", [0.2, 0.5]),
                "instance_dropout": aug_cfg.get("student_instance_dropout", 0.15),
                "feature_noise_std": aug_cfg.get("student_feature_noise_std", 0.03),
                "feature_dropout": aug_cfg.get("student_feature_dropout", 0.08),
                "posterize_prob": aug_cfg.get("student_posterize_prob", 0.4),
            }
        )

        teacher_aug = _build_single_augmentor(teacher_cfg, coords_aware)
        student_aug = _build_single_augmentor(student_cfg, coords_aware)

        return AsymmetricDualViewAugmentor(
            teacher_augmentor=teacher_aug,
            student_augmentor=student_aug,
            use_split=aug_cfg.get("asymmetric_use_split", False),
            split_ratio=aug_cfg.get("asymmetric_split_ratio", 0.6),
        )

    if view_strategy == "jepa":
        block_mask = SpatialBlockMask(
            n_blocks=aug_cfg.get("jepa_n_blocks", 4),
            block_size_range=tuple(aug_cfg.get("jepa_block_size", [1, 3])),
            grid_size=aug_cfg.get("jepa_grid_size", 8),
            min_context_frac=aug_cfg.get("jepa_min_context_frac", 0.5),
        )

        # Light context augmentation (no heavy noise, no posterize)
        context_cfg = dict(aug_cfg)
        context_cfg.update(
            {
                "subsample_frac": [1.0, 1.0],  # don't subsample context further
                "instance_dropout": 0.0,
                "feature_noise_std": aug_cfg.get("jepa_context_noise", 0.01),
                "feature_dropout": 0.0,
                "posterize_prob": 0.0,
                "use_spatial_crop": False,
                "use_density_sub": False,
                "enable_local_smooth": False,
                "enable_spatial_interpolation": False,
                "enable_region_mixup": False,
            }
        )
        context_aug = _build_single_augmentor(context_cfg, coords_aware)

        affine = (
            CoordAffine(
                enable_rotation=aug_cfg.get("coord_rotation", True),
                enable_flip=aug_cfg.get("coord_flip", True),
                enable_scale=None,  # no scale for JEPA (block positions must be consistent)
            )
            if coords_aware
            else None
        )

        return JEPAViewAugmentor(
            block_mask=block_mask,
            context_augmentor=context_aug,
            coord_affine=affine,
            pre_subsample=aug_cfg.get("jepa_pre_subsample", 4096),
        )

    if view_strategy == "multicrop":
        affine = CoordAffine() if coords_aware else None
        smooth = LocalFeatureSmooth() if coords_aware else None
        interp = SpatialFeatureInterpolation() if coords_aware else None
        posterize = FeaturePosterize() if coords_aware else None
        region_drop = SpatialRegionDrop() if coords_aware else None
        mixup = TissueRegionMixup() if coords_aware else None

        return MultiCropAugmentor(
            n_global=aug_cfg.get("multicrop_n_global", 2),
            n_local=aug_cfg.get("multicrop_n_local", 4),
            global_frac=tuple(aug_cfg.get("multicrop_global_frac", [0.7, 1.0])),
            local_frac=tuple(aug_cfg.get("multicrop_local_frac", [0.2, 0.5])),
            coord_affine=affine,
            local_smooth=smooth,
            spatial_region_drop=region_drop,
            spatial_interpolation=interp,
            region_mixup=mixup,
            feature_posterize=posterize,
        )

    raise ValueError(
        f"Unknown view_strategy '{view_strategy}'. "
        f"Valid: dual, split_dual, asymmetric, jepa, multicrop"
    )
