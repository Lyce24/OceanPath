"""
Memmap-backed dataset for SSL pretraining.

Supports two augmentation output formats:

1. **Dual-view** (VICReg, JEPA, LeJEPA-2C):
   view_generator returns ((v1f, v1c), (v2f, v2c)) — emits view{1,2}_features.

2. **Multi-crop** (LeJEPA-MC):
   view_generator returns {"global": [...], "local": [...]} — emits
   global_features / local_features as lists, ready for multicrop_stack_collate.

Data flow per __getitem__
═══════════════════════════════════════════════════════════════════════════
  1. Read features [N_raw, D] + coords [N_raw, 2] from memmap
  2. Pre-cap: if N_raw > dataset_max_instances, subsample per pre_cap_mode
  3. view_generator → either dual-view tuple or multi-crop dict
  4. Format into per-format sample dict
  5. Return sample dict with slide_id

Each view always has exactly the augmentor's per-view fixed_n. coords_aware=False
suppresses coords keys without affecting augmentation.
"""

import logging
from collections.abc import Callable

import numpy as np
import torch

from oceanpath.data.dataset import BaseMmapDataset
from oceanpath.data.mmap_builder import _spatial_stratified_subsample
from oceanpath.ssl.data.augmentation import (
    DualViewAugmentor,
    WSIDualViewAugmentor,
)

logger = logging.getLogger(__name__)

_VALID_PRE_CAP_MODES = frozenset({"random", "contiguous", "head", "spatial_stratified"})


def _to_feature_tensor(arr: np.ndarray, force_float32: bool = False) -> torch.Tensor:
    arr = np.ascontiguousarray(arr)
    if force_float32:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def _to_int_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr).astype(np.int32))


class PretrainDataset(BaseMmapDataset):
    """Slide-level dataset for SSL pretraining (no labels).

    Parameters
    ----------
    mmap_dir : str
        Directory containing features_*.bin, coords_*.bin, index_arrays.npz.
    fixed_n : int
        Target patch count per view for dual-view methods. Multi-crop methods
        derive per-view fixed_n from the augmentor's specs and ignore this
        value (it remains useful as a sanity floor on the dataset side).
    slide_ids : list[str] or None
        Slide IDs to include. None = all slides in the mmap index.
    view_generator : callable or None
        Either a WSIDualViewAugmentor (returns a 2-tuple of views) or a
        WSIMultiCropAugmentor (returns a dict {"global": [...], "local": [...]}).
        If None, a default DualViewAugmentor is built.
    dataset_max_instances : int or None
        Cap patches per slide BEFORE augmentation.
    pre_cap_mode : str
        How to pick patches when slide exceeds dataset_max_instances:
          - "spatial_stratified" (default), "random", "contiguous", "head".
    pre_cap_grid_size : int
        Grid resolution for the spatial_stratified pre-cap mode.
    force_float32 : bool
        If True, cast features to float32. Default False preserves the mmap dtype.
    coords_aware : bool
        If True, emit per-view coords keys so the aggregator receives spatial
        coordinates. False strips coords from the sample dict.
    """

    def __init__(
        self,
        mmap_dir: str,
        fixed_n: int,
        slide_ids: list[str] | None = None,
        view_generator: Callable | None = None,
        dataset_max_instances: int | None = None,
        pre_cap_mode: str = "spatial_stratified",
        pre_cap_grid_size: int = 16,
        force_float32: bool = False,
        coords_aware: bool = True,
    ):
        super().__init__(mmap_dir=mmap_dir, slide_ids=slide_ids, load_coords=True)

        if fixed_n < 1:
            raise ValueError(f"fixed_n must be >= 1, got {fixed_n}")
        if pre_cap_mode not in _VALID_PRE_CAP_MODES:
            raise ValueError(
                f"Unknown pre_cap_mode '{pre_cap_mode}'. Valid: {sorted(_VALID_PRE_CAP_MODES)}"
            )

        self.fixed_n = fixed_n
        self.dataset_max_instances = dataset_max_instances
        self.pre_cap_mode = pre_cap_mode
        self.pre_cap_grid_size = pre_cap_grid_size
        self.force_float32 = force_float32
        self.coords_aware = bool(coords_aware)

        # Default generator: vanilla DualViewAugmentor with dataclass defaults.
        if view_generator is None:
            self.view_generator = DualViewAugmentor()
        else:
            self.view_generator = view_generator

        logger.info(
            "PretrainDataset: %d slides, feat_dim=%d, fixed_n=%d, "
            "dataset_max_instances=%s, pre_cap_mode=%s, coords_aware=%s",
            len(self.slide_ids),
            self.feat_dim,
            self.fixed_n,
            self.dataset_max_instances,
            self.pre_cap_mode,
            self.coords_aware,
        )

    # -- Fixed-N finalization -------------------------------------------------

    def _finalize_fixed(self, feats, coords):
        """No-op safety check. Augmentor guarantees feats.shape[0] == fixed_n."""
        n = self.fixed_n
        if feats.shape[0] != n:
            raise RuntimeError(
                f"augmentor returned {feats.shape[0]} tokens, expected {n}. "
                f"Check that the augmentor's cfg.fixed_n matches the dataset's "
                f"fixed_n."
            )
        return feats, coords, n

    # -- Reading with pre-cap -------------------------------------------------

    def _read_precapped(self, idx: int, n_patches: int) -> tuple[np.ndarray, np.ndarray]:
        """Read features + coords, applying dataset_max_instances pre-cap if needed."""
        cap = self.dataset_max_instances
        if cap is None or n_patches <= cap:
            return self._read_features(idx, n_patches), self._read_coords(idx, n_patches)

        if self.pre_cap_mode == "spatial_stratified":
            coords = self._read_coords(idx, n_patches)
            chosen = _spatial_stratified_subsample(
                coords,
                cap,
                self.pre_cap_grid_size,
                rng=np.random.RandomState(self.rng.integers(2**31)),
            )
            features = self._read_features_indices(idx, n_patches, chosen)
            return features, coords[chosen]

        if self.pre_cap_mode == "random":
            chosen = np.sort(self.rng.choice(n_patches, size=cap, replace=False))
            features = self._read_features_indices(idx, n_patches, chosen)
            coords = self._read_coords_indices(idx, chosen)
            return features, coords

        # "contiguous" or "head"
        start = (
            int(self.rng.integers(0, n_patches - cap + 1))
            if self.pre_cap_mode == "contiguous"
            else 0
        )
        return (
            self._read_features_window(idx, start, cap),
            self._read_coords_window(idx, start, cap),
        )

    # -- __getitem__ ----------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        features, coords = self._read_precapped(idx, self.lengths[idx])

        # Request per-view stats from the dual-view augmentor so the training
        # step can log realized crop/mask parameters (view/* metrics). The
        # multi-crop augmentor's stats schema is per-view-list; we don't
        # surface those here yet (the dual-view-only metric names use _1/_2).
        # Custom callables that don't support return_stats fall through to
        # the no-stats path below.
        wants_stats = isinstance(self.view_generator, WSIDualViewAugmentor)
        if wants_stats:
            output, stats = self.view_generator(features, coords, rng=self.rng, return_stats=True)
        else:
            output = self.view_generator(features, coords, rng=self.rng)
            stats = None

        # Multi-crop path: dict with "global"/"local" view lists.
        if isinstance(output, dict) and "global" in output and "local" in output:
            return self._format_multicrop_sample(output, idx)

        # Dual-view path: tuple ((v1f, v1c), (v2f, v2c)).
        (v1_feat, v1_coord), (v2_feat, v2_coord) = output
        v1_feat, v1_coord, n1 = self._finalize_fixed(v1_feat, v1_coord)
        v2_feat, v2_coord, n2 = self._finalize_fixed(v2_feat, v2_coord)

        result: dict = {
            "view1_features": _to_feature_tensor(v1_feat, self.force_float32),
            "view2_features": _to_feature_tensor(v2_feat, self.force_float32),
            "view1_length": n1,
            "view2_length": n2,
            "slide_id": self.slide_ids[idx],
        }
        if self.coords_aware and v1_coord is not None:
            result["view1_coords"] = _to_int_tensor(v1_coord)
        if self.coords_aware and v2_coord is not None:
            result["view2_coords"] = _to_int_tensor(v2_coord)
        if stats is not None:
            # Per-view aug stats forwarded to the collator → training step
            # for `aug/*` logging. Two groups:
            #   1. Realized augmentation parameters (what was *attempted*):
            #        crop_area_frac, mask_ratio
            #   2. Refill composition fractions (what the final fixed-N view
            #      actually became):
            #        mask_fraction, crop_or_better_fraction,
            #        view_or_better_fraction, full_refill_fraction,
            #        replacement_fraction
            _aug_keys = (
                "crop_area_frac",
                "mask_ratio",
                "mask_fraction",
                "crop_or_better_fraction",
                "view_or_better_fraction",
                "full_refill_fraction",
                "replacement_fraction",
            )
            for key in _aug_keys:
                result[f"view1_{key}"] = float(stats["v1"][key])
                result[f"view2_{key}"] = float(stats["v2"][key])
        return result

    def _format_multicrop_sample(self, output: dict, idx: int) -> dict:
        """Format a multi-crop sample for ``multicrop_stack_collate``.

        Each global / local view becomes its own list entry in the sample dict.
        We use list values (not per-view keys) because the number of views is
        fixed per epoch — set by the augmentor at construction — so the
        collator can safely index ``sample["global_features"][v]``.

        Defensive checks: every global view must have the augmentor's
        ``global_spec.fixed_n`` rows, and likewise for local. The augmentor's
        hierarchical refill guarantees this; the assertion turns a silent
        shape bug into a loud one.
        """
        aug = self.view_generator
        g_n = aug.global_spec.fixed_n
        l_n = aug.local_spec.fixed_n if aug.num_local > 0 else 0

        global_features: list[torch.Tensor] = []
        global_coords: list[torch.Tensor] = []
        for v_i, (f_v, c_v) in enumerate(output["global"]):
            if f_v.shape[0] != g_n:
                raise RuntimeError(
                    f"global view {v_i}: augmentor returned {f_v.shape[0]} tokens, expected {g_n}"
                )
            global_features.append(_to_feature_tensor(f_v, self.force_float32))
            if self.coords_aware and c_v is not None:
                global_coords.append(_to_int_tensor(c_v))

        local_features: list[torch.Tensor] = []
        local_coords: list[torch.Tensor] = []
        for v_i, (f_v, c_v) in enumerate(output["local"]):
            if f_v.shape[0] != l_n:
                raise RuntimeError(
                    f"local view {v_i}: augmentor returned {f_v.shape[0]} tokens, expected {l_n}"
                )
            local_features.append(_to_feature_tensor(f_v, self.force_float32))
            if self.coords_aware and c_v is not None:
                local_coords.append(_to_int_tensor(c_v))

        result: dict = {
            "global_features": global_features,  # list[Tensor[N_g, D]] of len V_g
            "local_features": local_features,  # list[Tensor[N_l, D]] of len V_l
            "num_global": len(global_features),
            "num_local": len(local_features),
            "slide_id": self.slide_ids[idx],
        }
        if self.coords_aware:
            if global_coords:
                result["global_coords"] = global_coords
            if local_coords:
                result["local_coords"] = local_coords
        return result
