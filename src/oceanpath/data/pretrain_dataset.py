"""
Memmap-backed dataset for SSL pretraining.

Returns TWO augmented views of each slide's features for contrastive /
distillation losses.  No labels -- this is unsupervised.

Extends BaseMmapDataset for shared memmap reading logic.

Key differences from the supervised MmapDataset:

  - No labels (returns slide_id, not label).
  - Always loads coords (required for coords-aware augmentation).
  - Returns two views per __getitem__, produced by a DualViewAugmentor.
  - No cache (views are always stochastic -- caching would freeze them).
  - dataset_max_instances caps the raw read BEFORE the augmentor runs,
    preventing O(N^2) blowup in LocalFeatureSmooth for very large slides.

Data flow per __getitem__:
  1. Read features (N, D) + coords (N, 2) from memmap
  2. Pre-cap to dataset_max_instances (mode-dependent if N > cap)
  3. DualViewAugmentor -> view1 (M1, D), view2 (M2, D) with independent randomness
  4. Convert to tensors -> dict with both views
"""

import logging

import numpy as np
import torch

from oceanpath.data.dataset import BaseMmapDataset
from oceanpath.ssl.augmentation import DualViewAugmentor, FeatureAugmentor

logger = logging.getLogger(__name__)

_VALID_PRE_CAP_MODES = frozenset({"random", "contiguous", "head"})


def _to_feature_tensor(arr: np.ndarray, force_float32: bool = False) -> torch.Tensor:
    """Convert numpy array to contiguous tensor, preserving dtype by default."""
    arr = np.ascontiguousarray(arr)
    if force_float32:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def _to_int_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to contiguous int32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr).astype(np.int32))


class PretrainDataset(BaseMmapDataset):
    """Slide-level dataset for SSL pretraining (dual-view, no labels).

    Parameters
    ----------
    mmap_dir : str
        Directory containing features_*.bin, coords_*.bin, index_arrays.npz.
    slide_ids : list[str] or None
        Slide IDs to include. None = use ALL slides in the mmap index.
    augmentor : FeatureAugmentor or None
        Single-view augmentor (wrapped in DualViewAugmentor internally).
    dataset_max_instances : int or None
        Cap patches per slide BEFORE augmentation. Prevents O(N^2) blowup
        in spatial-neighbor augmentations for very large slides.
        None = use all patches.
    pre_cap_mode : str
        Strategy when ``dataset_max_instances`` is active and slide exceeds it:
          - "random": uniform random subset (sorted for sequential disk access)
          - "contiguous": random contiguous window (HDD-friendly)
          - "head": first-N window (deterministic, fastest)
    force_float32 : bool
        If True, cast features to float32 in __getitem__. Default False keeps
        the native mmap dtype (typically float16), saving memory bandwidth.
        With mixed-precision training (AMP), float16 inputs are handled
        automatically by autocast.
    """

    def __init__(
        self,
        mmap_dir: str,
        slide_ids: list[str] | None = None,
        augmentor: FeatureAugmentor | None = None,
        dataset_max_instances: int | None = None,
        pre_cap_mode: str = "random",
        force_float32: bool = False,
    ):
        super().__init__(mmap_dir=mmap_dir, slide_ids=slide_ids, load_coords=True)
        self.dataset_max_instances = dataset_max_instances
        self.pre_cap_mode = pre_cap_mode
        self.force_float32 = force_float32
        if self.pre_cap_mode not in _VALID_PRE_CAP_MODES:
            raise ValueError(
                f"Unknown pre_cap_mode '{self.pre_cap_mode}'. Valid: {sorted(_VALID_PRE_CAP_MODES)}"
            )

        if augmentor is None:
            augmentor = FeatureAugmentor()
        self.dual_view = DualViewAugmentor(augmentor)

        logger.info(
            "PretrainDataset: %d slides, feat_dim=%d, dataset_max_instances=%s, pre_cap_mode=%s",
            len(self.slide_ids),
            self.feat_dim,
            self.dataset_max_instances,
            self.pre_cap_mode,
        )

    # -- Reading with pre-cap --------------------------------------------------

    def _read_precapped(self, idx: int, n_patches: int) -> tuple[np.ndarray, np.ndarray]:
        """Read features + coords, applying dataset_max_instances pre-cap if needed.

        Returns (features, coords) numpy arrays.
        """
        cap = self.dataset_max_instances
        if cap is not None and n_patches > cap:
            if self.pre_cap_mode == "random":
                chosen = np.sort(self.rng.permutation(n_patches)[:cap])
                features = self._read_features_indices(idx, n_patches, chosen)
                coords = self._read_coords_indices(idx, chosen)
            else:
                # "contiguous" picks a random start; "head" always starts at 0
                start = (
                    int(self.rng.integers(0, n_patches - cap + 1))
                    if self.pre_cap_mode == "contiguous"
                    else 0
                )
                features = self._read_features_window(idx, start, cap)
                coords = self._read_coords_window(idx, start, cap)
        else:
            features = self._read_features(idx, n_patches)
            coords = self._read_coords(idx, n_patches)
        return features, coords

    # -- __getitem__ -----------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        """Load one slide and produce two augmented views.

        Pipeline: read -> pre-cap -> dual-view augmentation -> tensor conversion

        Returns
        -------
        dict with keys:
            view1_features : Tensor [M1, D]
            view1_coords   : Tensor [M1, 2]
            view2_features : Tensor [M2, D]
            view2_coords   : Tensor [M2, 2]
            slide_id       : str
        """
        features, coords = self._read_precapped(idx, self.lengths[idx])

        (v1_feat, v1_coord), (v2_feat, v2_coord) = self.dual_view(features, coords, rng=self.rng)

        result: dict = {
            "view1_features": _to_feature_tensor(v1_feat, self.force_float32),
            "view2_features": _to_feature_tensor(v2_feat, self.force_float32),
            "slide_id": self.slide_ids[idx],
        }

        if v1_coord is not None:
            result["view1_coords"] = _to_int_tensor(v1_coord)
        if v2_coord is not None:
            result["view2_coords"] = _to_int_tensor(v2_coord)

        return result
