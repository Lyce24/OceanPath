"""
Feature-space augmentation for SSL pretraining.

Three categories of augmentation, ordered from strongest to subtlest:

1. **Patch-selection augmentations** (which patches each view sees):
   - Patch subsampling — uniform random subset
   - SpatialCrop — contiguous rectangular region
   - SpatialDensitySubsample — density-aware thinning (NEW)
   - SpatialRegionDrop — rectangular cutout in coordinate space
   - Instance dropout — random per-patch drop

2. **Coords-aware feature augmentations** (perturb features using spatial structure):
   - LocalFeatureSmooth — blend with k-nearest spatial neighbors
   - SpatialFeatureInterpolation — lerp toward nearest neighbor (NEW)
   - TissueRegionMixup — cross-region prototype mixing (NEW)

3. **Feature-only augmentations** (channel-level noise):
   - Feature noise — additive Gaussian
   - Feature dropout — zero random dimensions

4. **Coordinate augmentations** (change spatial layout, features unchanged):
   - CoordAffine — rotation, flip, scale
   - SpatialGridShuffle — permute within grid cells
   - Coordinate jitter — random offset

Design principles for feature embedding augmentation
─────────────────────────────────────────────────────
- Feature embeddings are HIGH-LEVEL (ViT-H output). Heavy noise destroys
  semantics. Primary view diversity comes from WHICH patches each view sees
  (subsampling/cropping), not from corrupting embedding values.
- Coords provide SPATIAL STRUCTURE. Coords-aware augmentation lets us create
  views that differ in spatial coverage and local feature relationships,
  encouraging the aggregator to learn representations that are robust to
  which tissue regions are visible and how features vary locally.
- The three NEW transforms exploit the feature↔coord relationship:
    • SpatialFeatureInterpolation: perturbs features ALONG the tissue manifold
      by blending toward spatial neighbors. More semantically meaningful than
      Gaussian noise because perturbation follows tissue topology.
    • TissueRegionMixup: mixes features across spatial regions, breaking
      region-specific signatures. Forces the aggregator to use distributed
      evidence rather than memorizing one hot-spot's feature profile.
    • SpatialDensitySubsample: normalizes spatial coverage by thinning
      dense regions. Prevents the aggregator from over-representing
      heavily-sampled tissue areas.
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
        """
        features: (N, D), coords: (N, 2) → filtered features, coords.
        """
        N = features.shape[0]
        min_keep = max(1, int(N * self.min_keep_frac))

        # Bounding box of all patches
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6

        keep_mask = np.ones(N, dtype=bool)

        for _ in range(self.n_regions):
            # Random region size as fraction of bounding box
            frac_x = rng.uniform(self.region_lo, self.region_hi)
            frac_y = rng.uniform(self.region_lo, self.region_hi)
            w = span[0] * frac_x
            h = span[1] * frac_y

            # Random center within the bounding box
            cx = rng.uniform(cmin[0], cmax[0])
            cy = rng.uniform(cmin[1], cmax[1])

            # Drop patches inside the rectangle
            in_box = (
                (coords[:, 0] >= cx - w / 2)
                & (coords[:, 0] <= cx + w / 2)
                & (coords[:, 1] >= cy - h / 2)
                & (coords[:, 1] <= cy + h / 2)
            )
            keep_mask &= ~in_box

        # Safety: keep at least min_keep patches
        if keep_mask.sum() < min_keep:
            dropped = np.where(~keep_mask)[0]
            n_restore = min_keep - keep_mask.sum()
            restore_idx = rng.choice(dropped, size=min(n_restore, len(dropped)), replace=False)
            keep_mask[restore_idx] = True

        return features[keep_mask], coords[keep_mask]


class SpatialCrop:
    """
    Crop a contiguous rectangular region from coordinate space.

    Instead of random uniform subsampling, this creates a view that sees
    a spatially coherent tissue region. Different views get different crops,
    so the aggregator must learn representations from partial spatial coverage.

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
        """
        features: (N, D), coords: (N, 2) → cropped features, coords.
        """
        N = features.shape[0]
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6

        # Random crop size
        fx = rng.uniform(self.crop_lo, self.crop_hi)
        fy = rng.uniform(self.crop_lo, self.crop_hi)
        crop_w = span[0] * fx
        crop_h = span[1] * fy

        # Random top-left corner (ensure crop fits within bounding box)
        x0 = rng.uniform(cmin[0], cmin[0] + span[0] - crop_w)
        y0 = rng.uniform(cmin[1], cmin[1] + span[1] - crop_h)

        in_crop = (
            (coords[:, 0] >= x0)
            & (coords[:, 0] <= x0 + crop_w)
            & (coords[:, 1] >= y0)
            & (coords[:, 1] <= y0 + crop_h)
        )

        if in_crop.sum() < self.min_patches:
            # Fallback: random subsample to avoid degenerate views
            k = max(self.min_patches, int(N * fx * fy))
            k = min(k, N)
            idx = rng.permutation(N)[:k]
            return features[idx], coords[idx]

        return features[in_crop], coords[in_crop]


class SpatialDensitySubsample:
    """
    Subsample patches inversely proportional to local spatial density.

    Dense tissue regions (e.g., densely packed tumor nests) produce many
    overlapping patches with redundant features. Sparse regions (e.g.,
    stroma, tissue edges) contribute fewer but often more structurally
    informative patches.

    This transform normalizes spatial coverage by thinning dense clusters
    and preserving sparse regions, preventing the aggregator from
    over-representing heavily-sampled areas.

    Implementation: grid-based density estimation for O(N) efficiency.
    Divide coordinate space into a grid, count patches per cell, then
    assign each patch a keep-probability inversely proportional to its
    cell's count.

    Parameters
    ----------
    target_frac : tuple[float, float]
        Fraction of patches to retain, sampled uniformly from [lo, hi].
    grid_size : int
        Grid resolution for density estimation. Higher = finer density
        map but more cells with count=1 (less thinning effect).
    temperature : float
        Controls how aggressively to thin dense regions.
        0 → uniform random (ignore density). Higher → stronger density
        correction. 1.0 = keep_prob ∝ 1/count. Recommended: 0.5-1.0.
    min_patches : int
        Minimum patches to retain regardless of density.
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
        """
        features: (N, D), coords: (N, 2) → density-subsampled features, coords.
        """
        N = features.shape[0]
        target_frac = rng.uniform(self.frac_lo, self.frac_hi)
        target_n = max(self.min_patches, int(N * target_frac))
        target_n = min(target_n, N)

        if self.min_patches >= N:
            return features, coords

        # Grid-based density estimation
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6
        cell_size = span / self.grid_size

        cell_x = np.clip(
            ((coords[:, 0] - cmin[0]) / cell_size[0]).astype(int),
            0,
            self.grid_size - 1,
        )
        cell_y = np.clip(
            ((coords[:, 1] - cmin[1]) / cell_size[1]).astype(int),
            0,
            self.grid_size - 1,
        )
        cell_ids = cell_x * self.grid_size + cell_y

        # Count patches per cell
        _unique_cells, inverse, counts = np.unique(
            cell_ids, return_inverse=True, return_counts=True
        )
        patch_density = counts[inverse].astype(np.float64)  # (N,) density at each patch

        # Inverse-density weights with temperature control
        # weight = 1 / density^temperature → higher temp = stronger thinning
        weights = 1.0 / np.power(patch_density, self.temperature)
        weights /= weights.sum()

        # Weighted sampling without replacement
        idx = rng.choice(N, size=target_n, replace=False, p=weights)
        idx.sort()  # preserve spatial ordering for downstream transforms

        return features[idx], coords[idx]


class CoordAffine:
    """
    Apply random affine transforms to coordinates.

    Transforms: random rotation, random flip, random scale.
    Features stay unchanged — only spatial layout changes.

    This matters for position-aware aggregators (TransMIL with PPEG):
    the model must produce the same slide embedding regardless of how
    the slide was oriented on the scanner.

    Parameters
    ----------
    enable_rotation : bool
        Random rotation by a multiple of 90°.
    enable_flip : bool
        Random horizontal and/or vertical flip.
    enable_scale : tuple[float, float] or None
        Random scale factor per axis: uniform[lo, hi].
        None = no scaling.
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

    def __call__(
        self,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        coords: (N, 2) → transformed coords (N, 2).
        """
        c = coords.astype(np.float64)

        # Center around origin for rotation/flip
        center = c.mean(axis=0)
        c = c - center

        # Random 90° rotation (0, 90, 180, 270)
        if self.enable_rotation:
            k = rng.integers(0, 4)
            if k == 1:
                c = np.stack([-c[:, 1], c[:, 0]], axis=1)
            elif k == 2:
                c = -c
            elif k == 3:
                c = np.stack([c[:, 1], -c[:, 0]], axis=1)

        # Random flip
        if self.enable_flip:
            if rng.random() < 0.5:
                c[:, 0] = -c[:, 0]
            if rng.random() < 0.5:
                c[:, 1] = -c[:, 1]

        # Random scale
        if self.enable_scale is not None:
            sx = rng.uniform(*self.enable_scale)
            sy = rng.uniform(*self.enable_scale)
            c[:, 0] *= sx
            c[:, 1] *= sy

        # Re-center and convert back to original dtype
        c = c + center
        return c.astype(coords.dtype)


# ═════════════════════════════════════════════════════════════════════════════
# Coords-Aware Feature Transforms (perturb features using spatial structure)
# ═════════════════════════════════════════════════════════════════════════════


class LocalFeatureSmooth:
    """
    Blend each patch's features with its k-nearest spatial neighbors.

    Creates a smoothed feature view that respects tissue topology.
    Different smoothing per view → the aggregator learns representations
    invariant to local feature perturbations along the tissue manifold.

    This is more semantically meaningful than random Gaussian noise
    because the perturbation follows the tissue's spatial structure:
    a patch's features become more like its neighbors', simulating
    the effect of slightly shifted patch extraction boundaries.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest spatial neighbors to blend with.
    alpha_range : tuple[float, float]
        Blend weight: feat = (1-alpha)*feat + alpha*mean(neighbors).
        alpha is sampled from uniform[lo, hi].
    frac_patches : float
        Fraction of patches to apply smoothing to (rest unchanged).
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
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        features: (N, D), coords: (N, 2) → smoothed features (N, D).
        Coords are NOT modified.
        """
        N, _D = features.shape
        if self.k_neighbors + 1 >= N:
            return features

        k = min(self.k_neighbors, N - 1)
        alpha = rng.uniform(self.alpha_lo, self.alpha_hi)

        # Pairwise squared distances (O(N²) — slides are pre-capped)
        diff = coords[:, None, :].astype(np.float64) - coords[None, :, :].astype(np.float64)
        dists = (diff * diff).sum(axis=2)  # (N, N)

        # k+1 nearest (includes self at distance 0)
        neighbor_idx = np.argpartition(dists, k + 1, axis=1)[:, 1 : k + 1]  # (N, k)

        # Compute neighbor mean features
        neighbor_feats = features[neighbor_idx]  # (N, k, D)
        neighbor_mean = neighbor_feats.mean(axis=1)  # (N, D)

        # Select random subset of patches to smooth
        n_smooth = max(1, int(N * self.frac_patches))
        smooth_idx = rng.choice(N, size=n_smooth, replace=False)

        # Blend
        out = features.copy()
        out[smooth_idx] = (1 - alpha) * features[smooth_idx] + alpha * neighbor_mean[smooth_idx]

        return out.astype(features.dtype)


class SpatialFeatureInterpolation:
    """
    Lerp each selected patch's feature toward its nearest spatial neighbor.

    Whereas LocalFeatureSmooth blends with the MEAN of k neighbors (a
    low-pass filter), SpatialFeatureInterpolation blends each patch with
    ONE specific neighbor. This creates sharper, more diverse perturbations
    that follow the tissue manifold:

        feat_i' = (1 - t) * feat_i + t * feat_nn(i)

    where nn(i) is the nearest spatial neighbor and t ~ Uniform[lo, hi].

    Biological motivation: if the patch extraction grid shifted by a few
    pixels, each patch's features would interpolate toward the adjacent
    tile's features. This transform simulates that shift without re-running
    the encoder, producing views that are invariant to exact extraction
    boundaries.

    Efficiency: O(N²) distance matrix, but slides are pre-capped by
    dataset_max_instances (typically ≤ 4096 patches).

    Parameters
    ----------
    t_range : tuple[float, float]
        Interpolation weight sampled from uniform[lo, hi].
        Keep low (0.05-0.25) to preserve patch identity.
    frac_patches : float
        Fraction of patches to interpolate (rest unchanged).
    neighbor_rank_range : tuple[int, int]
        Instead of always using the 1st-nearest neighbor, randomly pick
        between the k-th nearest, where k ~ uniform[lo, hi].
        (1, 1) = always nearest, (1, 3) = nearest to 3rd-nearest.
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
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        features: (N, D), coords: (N, 2) → interpolated features (N, D).
        Coords are NOT modified.
        """
        N, _D = features.shape
        max_rank = min(self.rank_hi, N - 1)
        if N < 2 or max_rank < 1:
            return features

        # Pairwise squared distances
        diff = coords[:, None, :].astype(np.float64) - coords[None, :, :].astype(np.float64)
        dists = (diff * diff).sum(axis=2)  # (N, N)

        # Find the top max_rank+1 nearest neighbors per patch (including self)
        # argpartition is O(N) per row, much cheaper than full argsort
        topk = min(max_rank + 1, N)
        nn_idx = np.argpartition(dists, topk, axis=1)[:, :topk]  # (N, topk)

        # Sort within the partitioned block to get ranked neighbors
        # nn_idx[:, 0] is self, nn_idx[:, 1] is nearest, etc.
        for i in range(N):
            order = np.argsort(dists[i, nn_idx[i]])
            nn_idx[i] = nn_idx[i, order]

        # Select patches to interpolate
        n_interp = max(1, int(N * self.frac_patches))
        interp_idx = rng.choice(N, size=n_interp, replace=False)

        out = features.copy()
        for idx in interp_idx:
            # Random neighbor rank (skip rank 0 which is self)
            rank = rng.integers(self.rank_lo, max_rank + 1)
            rank = min(rank, topk - 1)
            nn = nn_idx[idx, rank]

            # Random interpolation weight
            t = rng.uniform(self.t_lo, self.t_hi)
            out[idx] = (1 - t) * features[idx] + t * features[nn]

        return out.astype(features.dtype)


class TissueRegionMixup:
    """
    Mix features across spatial regions to break region-specific signatures.

    Procedure:
      1. Partition patches into spatial regions via grid binning.
      2. Compute a prototype feature (mean) for each region.
      3. For a random subset of patches, blend their feature with the
         prototype of a DIFFERENT randomly-selected region.

    This forces the aggregator to build slide representations that don't
    rely on one region's specific feature profile. If the tumor signature
    is concentrated in one spatial area, TissueRegionMixup dilutes it
    into other regions, requiring the aggregator to learn features that
    generalize across spatial contexts.

    Different from LocalFeatureSmooth:
      - LocalFeatureSmooth blends with NEARBY patches (local smoothing).
      - TissueRegionMixup blends with DISTANT region prototypes (global mixing).
    The combination of both gives perturbations at two spatial scales.

    Parameters
    ----------
    grid_size : int
        Grid resolution for region definition.
    alpha_range : tuple[float, float]
        Blend weight: feat = (1-alpha)*feat + alpha*foreign_prototype.
        alpha sampled from uniform[lo, hi]. Keep low to preserve patch identity.
    frac_patches : float
        Fraction of patches to apply mixup to.
    min_region_patches : int
        Minimum patches in a region for it to have a valid prototype.
        Regions smaller than this are excluded from the prototype pool.
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
        self,
        features: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        features: (N, D), coords: (N, 2) → region-mixed features (N, D).
        Coords are NOT modified.
        """
        N, _D = features.shape
        if self.min_region_patches * 2 > N:
            return features

        # Grid-based region assignment
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6
        cell_size = span / self.grid_size

        cell_x = np.clip(
            ((coords[:, 0] - cmin[0]) / cell_size[0]).astype(int),
            0,
            self.grid_size - 1,
        )
        cell_y = np.clip(
            ((coords[:, 1] - cmin[1]) / cell_size[1]).astype(int),
            0,
            self.grid_size - 1,
        )
        cell_ids = cell_x * self.grid_size + cell_y  # (N,)

        # Compute region prototypes (mean feature per cell)
        unique_cells = np.unique(cell_ids)
        prototypes = {}  # cell_id → mean feature (D,)
        for cid in unique_cells:
            mask = cell_ids == cid
            if mask.sum() >= self.min_region_patches:
                prototypes[cid] = features[mask].astype(np.float64).mean(axis=0)

        # Need at least 2 valid regions for cross-region mixing
        valid_cells = list(prototypes.keys())
        if len(valid_cells) < 2:
            return features

        # Select patches to mix
        n_mix = max(1, int(N * self.frac_patches))
        mix_idx = rng.choice(N, size=n_mix, replace=False)

        alpha = rng.uniform(self.alpha_lo, self.alpha_hi)
        np.array([prototypes[c] for c in valid_cells])  # (R, D)

        out = features.copy()
        for idx in mix_idx:
            own_cell = cell_ids[idx]
            # Pick a DIFFERENT region's prototype
            candidates = [c for c in valid_cells if c != own_cell]
            if not candidates:
                continue
            foreign_cell = candidates[rng.integers(len(candidates))]
            foreign_proto = prototypes[foreign_cell]
            out[idx] = (1 - alpha) * features[idx] + alpha * foreign_proto

        return out.astype(features.dtype)


class SpatialGridShuffle:
    """
    Divide coordinate space into a grid, then permute patches within each cell.

    Preserves GLOBAL spatial structure (which grid cell each patch belongs to)
    but destroys FINE-GRAINED spatial ordering within cells.

    Forces the aggregator to learn from coarse spatial patterns rather than
    memorizing exact patch positions. Particularly useful for TransMIL where
    PPEG encodes positional information.

    Parameters
    ----------
    grid_size : int
        Number of grid cells per axis (grid_size x grid_size).
    shuffle_prob : float
        Probability that each cell gets shuffled (vs. left unchanged).
    """

    def __init__(self, grid_size: int = 4, shuffle_prob: float = 0.7):
        self.grid_size = grid_size
        self.shuffle_prob = shuffle_prob

    def __call__(
        self,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        coords: (N, 2) → permuted coords (N, 2).
        Only coordinates are shuffled within grid cells; features stay in place.
        """
        coords.shape[0]
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = (cmax - cmin).astype(np.float64) + 1e-6
        cell_size = span / self.grid_size

        cell_x = np.clip(
            ((coords[:, 0] - cmin[0]) / cell_size[0]).astype(int),
            0,
            self.grid_size - 1,
        )
        cell_y = np.clip(
            ((coords[:, 1] - cmin[1]) / cell_size[1]).astype(int),
            0,
            self.grid_size - 1,
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
# Main Augmentor (composes all transforms)
# ═════════════════════════════════════════════════════════════════════════════


class FeatureAugmentor:
    """
    Stochastic feature-space augmentation for one view.

    When coords are provided, applies coords-aware transforms in addition
    to the standard feature-only augmentations.

    Pipeline order (rationale: selection → feature perturbation → coord perturbation):

      ┌─ PATCH SELECTION (which patches this view sees) ──────────────────┐
      │ 1. Spatial crop OR density subsample OR random subsampling         │
      │ 2. Spatial region drop (cutout)                                    │
      │ 3. Instance dropout                                                │
      └────────────────────────────────────────────────────────────────────┘
      ┌─ FEATURE PERTURBATION (modify embedding values) ──────────────────┐
      │ 4. Local feature smoothing (blend with spatial neighbors)          │
      │ 5. Spatial feature interpolation (lerp toward nearest neighbor)    │
      │ 6. Tissue region mixup (cross-region prototype mixing)             │
      │ 7. Feature noise (Gaussian)                                        │
      │ 8. Feature dropout (zero dimensions)                               │
      └────────────────────────────────────────────────────────────────────┘
      ┌─ COORDINATE PERTURBATION (change spatial layout) ─────────────────┐
      │ 9. Coord affine (rotation/flip/scale)                              │
      │ 10. Spatial grid shuffle                                           │
      │ 11. Coordinate jitter (legacy)                                     │
      └────────────────────────────────────────────────────────────────────┘

    Selection runs first because it determines which patches the feature
    transforms operate on (and reduces N for the O(N²) distance matrix
    in LocalFeatureSmooth/SpatialFeatureInterpolation).

    Feature perturbation runs before coordinate perturbation because
    LocalFeatureSmooth and SpatialFeatureInterpolation use ORIGINAL coords
    to define spatial neighborhoods. If coords were shuffled first, the
    neighbor relationships would be wrong.

    Parameters
    ----------
    subsample_frac : tuple[float, float]
        Uniform[lo, hi] fraction of patches to keep (random subsampling).
    instance_dropout : float
        Probability of dropping each remaining patch.
    feature_noise_std : float
        Std of additive Gaussian noise on features.
    feature_dropout : float
        Probability of zeroing each feature dimension.
    coord_jitter : int
        Max random offset for spatial coordinates.
    spatial_crop : SpatialCrop or None
        Coords-aware spatial cropping.
    spatial_density_sub : SpatialDensitySubsample or None
        Coords-aware density-normalized subsampling.
    spatial_region_drop : SpatialRegionDrop or None
        Coords-aware cutout.
    coord_affine : CoordAffine or None
        Coords-aware rotation/flip/scale.
    local_smooth : LocalFeatureSmooth or None
        Blend features with spatial neighbors.
    spatial_interpolation : SpatialFeatureInterpolation or None
        Lerp features toward spatial neighbors.
    region_mixup : TissueRegionMixup or None
        Cross-region feature mixing.
    grid_shuffle : SpatialGridShuffle or None
        Coords-aware local patch permutation.
    use_spatial_crop : bool
        Enable SpatialCrop as an alternative to random subsampling.
    spatial_crop_prob : float
        P(spatial_crop) vs P(random_subsample) when use_spatial_crop=True.
    use_density_sub : bool
        Enable SpatialDensitySubsample as an alternative to random subsampling.
    density_sub_prob : float
        P(density_sub) vs P(other) when use_density_sub=True.
        When both spatial_crop and density_sub are enabled, the selection
        logic is: density_sub (p=density_sub_prob) → spatial_crop
        (p=spatial_crop_prob of remainder) → random_subsample.
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

        # Coords-aware
        self.spatial_crop = spatial_crop
        self.spatial_density_sub = spatial_density_sub
        self.spatial_region_drop = spatial_region_drop
        self.coord_affine = coord_affine
        self.local_smooth = local_smooth
        self.spatial_interpolation = spatial_interpolation
        self.region_mixup = region_mixup
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
        """
        Produce one augmented view of a slide.

        Parameters
        ----------
        features : (N, D) patch features.
        coords : (N, 2) spatial coordinates, optional.
        rng : numpy random generator (for reproducibility in workers).

        Returns
        -------
        aug_features : (M, D) augmented subset, M ≤ N.
        aug_coords : (M, 2) or None.
        """
        if rng is None:
            rng = np.random.default_rng()

        N, D = features.shape
        original_dtype = features.dtype
        has_coords = coords is not None

        # ── Step 1: Patch selection strategy ─────────────────────────────
        #
        # Three mutually exclusive strategies, chosen stochastically:
        #   a) SpatialDensitySubsample — density-normalized thinning
        #   b) SpatialCrop — contiguous rectangular region
        #   c) Random subsampling — uniform random (original fallback)
        #
        # Priority: density_sub → spatial_crop → random_subsample.
        # Each has an independent probability gate.

        used_density = False
        used_crop = False

        # Try density subsampling first
        if (
            has_coords
            and self.use_density_sub
            and self.spatial_density_sub is not None
            and rng.random() < self.density_sub_prob
        ):
            features, coords = self.spatial_density_sub(features, coords, rng)
            used_density = True

        # Try spatial crop (only if density_sub didn't fire)
        if (
            not used_density
            and has_coords
            and self.use_spatial_crop
            and self.spatial_crop is not None
            and rng.random() < self.spatial_crop_prob
        ):
            features, coords = self.spatial_crop(features, coords, rng)
            used_crop = True

        # Fallback: random subsampling (if neither spatial method was used)
        if not used_density and not used_crop:
            frac = rng.uniform(self.subsample_lo, self.subsample_hi)
            k = max(1, int(N * frac))
            indices = rng.permutation(N)[:k]
            features = features[indices].copy()
            if coords is not None:
                coords = coords[indices].copy()

        N_sub = features.shape[0]

        # ── Step 2: Spatial region drop (cutout) ─────────────────────────
        if has_coords and self.spatial_region_drop is not None and coords is not None:
            features, coords = self.spatial_region_drop(features, coords, rng)
            N_sub = features.shape[0]

        # ── Step 3: Instance dropout ─────────────────────────────────────
        if self.instance_dropout > 0 and N_sub > 1:
            keep = rng.random(N_sub) > self.instance_dropout
            if not keep.any():
                keep[rng.integers(N_sub)] = True
            features = features[keep]
            if coords is not None:
                coords = coords[keep]

        # ── Step 4: Local feature smoothing ──────────────────────────────
        if has_coords and self.local_smooth is not None and coords is not None:
            features = self.local_smooth(features, coords, rng)

        # ── Step 5: Spatial feature interpolation ────────────────────────
        if has_coords and self.spatial_interpolation is not None and coords is not None:
            features = self.spatial_interpolation(features, coords, rng)

        # ── Step 6: Tissue region mixup ──────────────────────────────────
        if has_coords and self.region_mixup is not None and coords is not None:
            features = self.region_mixup(features, coords, rng)

        # ── Step 7: Feature noise (Gaussian) ─────────────────────────────
        if self.feature_noise_std > 0:
            noise = rng.standard_normal(features.shape).astype(np.float32)
            features = features.astype(np.float32) + noise * self.feature_noise_std

        # ── Step 8: Feature dropout (zero dimensions) ────────────────────
        if self.feature_dropout > 0:
            dim_mask = rng.random(D) > self.feature_dropout
            features = features * dim_mask.astype(np.float32)

        # Cast back to original dtype (e.g. float16) after float32 ops.
        # This keeps the full pipeline in native dtype when using AMP.
        if features.dtype != original_dtype:
            features = features.astype(original_dtype)

        # ── Step 9: Coord affine (rotation/flip/scale) ───────────────────
        if has_coords and self.coord_affine is not None and coords is not None:
            coords = self.coord_affine(coords, rng)

        # ── Step 10: Spatial grid shuffle ────────────────────────────────
        if has_coords and self.grid_shuffle is not None and coords is not None:
            coords = self.grid_shuffle(coords, rng)

        # ── Step 11: Coordinate jitter (legacy, simple) ──────────────────
        if self.coord_jitter > 0 and coords is not None:
            jitter = rng.integers(-self.coord_jitter, self.coord_jitter + 1, size=coords.shape)
            coords = coords + jitter

        return features, coords


# ═════════════════════════════════════════════════════════════════════════════
# View Generators (compose FeatureAugmentor for multi-view SSL)
# ═════════════════════════════════════════════════════════════════════════════


class DualViewAugmentor:
    """
    Produces two augmented views from one slide's features.

    Each view uses a separate FeatureAugmentor call with independent
    randomness, ensuring the two views are different.
    """

    def __init__(self, augmentor: FeatureAugmentor):
        self.augmentor = augmentor

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray | None],
        tuple[np.ndarray, np.ndarray | None],
    ]:
        """
        Returns (view1_features, view1_coords), (view2_features, view2_coords).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Spawn child generators for independent randomness
        child_rngs = rng.spawn(2)
        view1 = self.augmentor(features, coords, rng=child_rngs[0])
        view2 = self.augmentor(features, coords, rng=child_rngs[1])
        return view1, view2


class MultiCropAugmentor:
    """
    Produces multiple views with different subset sizes (DINOv2 multi-crop).

    Global views: large patch subsets (e.g., 70-100%).
    Local views: small patch subsets (e.g., 20-50%).
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
        # Coords-aware options (applied to both global and local)
        coord_affine: CoordAffine | None = None,
        local_smooth: LocalFeatureSmooth | None = None,
        spatial_region_drop: SpatialRegionDrop | None = None,
        spatial_interpolation: SpatialFeatureInterpolation | None = None,
        region_mixup: TissueRegionMixup | None = None,
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
        )
        self.local_aug = FeatureAugmentor(
            subsample_frac=local_frac,
            instance_dropout=instance_dropout,
            feature_noise_std=feature_noise_std * 1.5,  # slightly more noise for local
            feature_dropout=feature_dropout,
            coord_affine=coord_affine,
            local_smooth=local_smooth,
            # No region drop for local views — they're already small
            spatial_interpolation=spatial_interpolation,
            # No region mixup for local — too few patches per region
        )

    def __call__(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Returns list of (features, coords) views: globals first, then locals."""
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
# Factory: build augmentor from config dict
# ═════════════════════════════════════════════════════════════════════════════


def build_augmentor(aug_cfg: dict, coords_aware: bool = True) -> FeatureAugmentor:
    """
    Construct a FeatureAugmentor from a flat config dict.

    Parameters
    ----------
    aug_cfg : dict
        Keys from pretrain.yaml → pretrain_training.augmentation.
    coords_aware : bool
        If True, enable coords-aware transforms with sensible defaults.
        If False, behave like the original feature-only augmentor.

    Returns
    -------
    FeatureAugmentor
        Fully configured augmentor.
    """
    kwargs = {
        "subsample_frac": tuple(aug_cfg.get("subsample_frac", [0.5, 1.0])),
        "instance_dropout": aug_cfg.get("instance_dropout", 0.1),
        "feature_noise_std": aug_cfg.get("feature_noise_std", 0.02),
        "feature_dropout": aug_cfg.get("feature_dropout", 0.05),
        "coord_jitter": aug_cfg.get("coord_jitter", 0),
    }

    if coords_aware:
        kwargs.update(
            # ── Patch selection ───────────────────────────────────────────
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
            # ── Feature transforms ────────────────────────────────────────
            coord_affine=CoordAffine(
                enable_rotation=aug_cfg.get("coord_rotation", True),
                enable_flip=aug_cfg.get("coord_flip", True),
                enable_scale=(
                    tuple(aug_cfg.get("coord_scale", [0.8, 1.2]))
                    if aug_cfg.get("coord_scale_enabled", True)
                    else None
                ),
            ),
            local_smooth=LocalFeatureSmooth(
                k_neighbors=aug_cfg.get("smooth_k_neighbors", 5),
                alpha_range=tuple(aug_cfg.get("smooth_alpha", [0.05, 0.3])),
                frac_patches=aug_cfg.get("smooth_frac_patches", 0.5),
            ),
            spatial_interpolation=SpatialFeatureInterpolation(
                t_range=tuple(aug_cfg.get("interp_t_range", [0.05, 0.25])),
                frac_patches=aug_cfg.get("interp_frac_patches", 0.5),
                neighbor_rank_range=tuple(aug_cfg.get("interp_neighbor_ranks", [1, 3])),
            ),
            region_mixup=TissueRegionMixup(
                grid_size=aug_cfg.get("region_mixup_grid", 4),
                alpha_range=tuple(aug_cfg.get("region_mixup_alpha", [0.05, 0.2])),
                frac_patches=aug_cfg.get("region_mixup_frac", 0.3),
                min_region_patches=aug_cfg.get("region_mixup_min_patches", 4),
            ),
            grid_shuffle=SpatialGridShuffle(
                grid_size=aug_cfg.get("grid_shuffle_size", 4),
                shuffle_prob=aug_cfg.get("grid_shuffle_prob", 0.7),
            ),
            # ── Selection strategy ────────────────────────────────────────
            use_spatial_crop=aug_cfg.get("use_spatial_crop", True),
            spatial_crop_prob=aug_cfg.get("spatial_crop_prob", 0.5),
            use_density_sub=aug_cfg.get("use_density_sub", True),
            density_sub_prob=aug_cfg.get("density_sub_prob", 0.3),
        )

    return FeatureAugmentor(**kwargs)
