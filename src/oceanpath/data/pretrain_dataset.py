"""
Memmap-backed dataset for SSL pretraining.

Returns TWO augmented views of each slide's features for contrastive /
distillation losses.  No labels — this is unsupervised.

Key differences from the supervised MmapDataset:

  - No labels (returns slide_id, not label).
  - Always loads coords (required for coords-aware augmentation).
  - Returns two views per __getitem__, produced by a DualViewAugmentor.
  - No cache (views are always stochastic — caching would freeze them).
  - dataset_max_instances caps the raw read BEFORE the augmentor runs,
    preventing O(N²) blowup in LocalFeatureSmooth for very large slides.

Data flow per __getitem__:
  1. Read features (N, D) + coords (N, 2) from memmap
  2. Pre-cap to dataset_max_instances (random subsample if N > cap)
  3. DualViewAugmentor → view1 (M₁, D), view2 (M₂, D) with independent randomness
  4. Convert to tensors → dict with both views
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from oceanpath.data.mmap_builder import validate_mmap_dir
from oceanpath.ssl.augmentation import DualViewAugmentor, FeatureAugmentor

logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """
    Slide-level dataset for SSL pretraining (dual-view, no labels).

    Parameters
    ----------
    mmap_dir : str
        Directory containing features_*.bin, coords_*.bin, index_arrays.npz.
    slide_ids : list[str] or None
        Slide IDs to include. None = use ALL slides in the mmap index.
    augmentor : FeatureAugmentor
        Single-view augmentor (wrapped in DualViewAugmentor internally).
    dataset_max_instances : int or None
        Cap patches per slide BEFORE augmentation. Prevents O(N²) blowup
        in spatial-neighbor augmentations for very large slides.
        None = use all patches.
    """

    def __init__(
        self,
        mmap_dir: str,
        slide_ids: list[str] | None = None,
        augmentor: FeatureAugmentor | None = None,
        dataset_max_instances: int | None = None,
    ):
        super().__init__()
        self.mmap_dir = Path(mmap_dir)
        self.dataset_max_instances = dataset_max_instances
        self.rng = np.random.default_rng()

        # Wrap single augmentor in dual-view generator
        if augmentor is None:
            augmentor = FeatureAugmentor()
        self.dual_view = DualViewAugmentor(augmentor)

        # Validate mmap directory
        meta = validate_mmap_dir(mmap_dir)
        self.feat_dim = meta["feat_dim"]
        self.coord_dim = meta["coord_dim"]

        # Load index
        idx = np.load(str(self.mmap_dir / "index_arrays.npz"), allow_pickle=True)
        all_slide_ids = idx["slide_ids"].tolist()
        self.feat_dtype = str(idx["feat_dtype"][0])
        self.coord_dtype_str = str(idx["coord_dtype"][0])
        self.bytes_per_feat = int(idx["bytes_per_feat"])
        n_feat_chunks = int(idx["num_feat_chunks"])
        n_coord_chunks = int(idx["num_coord_chunks"])

        # Build slide_id → index mapping for the full mmap
        id_to_idx = {sid: i for i, sid in enumerate(all_slide_ids)}

        # Filter to requested slide_ids (or use all)
        if slide_ids is None:
            slide_ids = all_slide_ids

        self.slide_ids: list[str] = []
        self.lengths: list[int] = []
        self.feat_chunk_ids: list[int] = []
        self.feat_offsets: list[int] = []
        self.coord_chunk_ids: list[int] = []
        self.coord_offsets: list[int] = []

        missing = []
        for sid in slide_ids:
            if sid not in id_to_idx:
                missing.append(sid)
                continue
            i = id_to_idx[sid]
            n = int(idx["lengths"][i])
            if n == 0:
                logger.warning(f"Slide '{sid}' has 0 patches — skipping.")
                continue
            self.slide_ids.append(sid)
            self.lengths.append(n)
            self.feat_chunk_ids.append(int(idx["feat_chunk_ids"][i]))
            self.feat_offsets.append(int(idx["feat_offsets"][i]))
            self.coord_chunk_ids.append(int(idx["coord_chunk_ids"][i]))
            self.coord_offsets.append(int(idx["coord_offsets"][i]))

        if missing:
            logger.warning(
                f"{len(missing)}/{len(slide_ids)} slide_ids not found in mmap index. "
                f"First 5: {missing[:5]}"
            )

        logger.info(
            f"PretrainDataset: {len(self.slide_ids)} slides, "
            f"feat_dim={self.feat_dim}, "
            f"dataset_max_instances={self.dataset_max_instances}"
        )

        # Open memmaps (read-only, OS shares physical pages across workers)
        self.feat_mmaps: list[np.memmap] = []
        for c in range(n_feat_chunks):
            path = self.mmap_dir / f"features_{c:03d}.bin"
            self.feat_mmaps.append(np.memmap(str(path), dtype=self.feat_dtype, mode="r"))

        # Always open coords (coords-aware augmentation needs them)
        self.coord_mmaps: list[np.memmap] = []
        for c in range(n_coord_chunks):
            path = self.mmap_dir / f"coords_{c:03d}.bin"
            self.coord_mmaps.append(np.memmap(str(path), dtype=self.coord_dtype_str, mode="r"))

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Load one slide and produce two augmented views.

        Pipeline: read → pre-cap → dual-view augmentation → tensor conversion

        Returns
        -------
        dict with keys:
            view1_features : Tensor [M₁, D]
            view1_coords   : Tensor [M₁, 2]
            view2_features : Tensor [M₂, D]
            view2_coords   : Tensor [M₂, 2]
            slide_id       : str
        """
        n_patches = self.lengths[idx]

        # 1. Read from memmap
        features = self._read_features(idx, n_patches)
        coords = self._read_coords(idx, n_patches)

        # 2. Pre-cap to dataset_max_instances (before augmentor for memory safety)
        if self.dataset_max_instances is not None and n_patches > self.dataset_max_instances:
            perm = self.rng.permutation(n_patches)[: self.dataset_max_instances]
            features = features[perm]
            coords = coords[perm]

        # 3. Dual-view augmentation (independent randomness per view)
        (v1_feat, v1_coord), (v2_feat, v2_coord) = self.dual_view(features, coords, rng=self.rng)

        # 4. Convert to tensors (float32 for features, int32 for coords)
        result = {
            "view1_features": torch.from_numpy(np.ascontiguousarray(v1_feat).astype(np.float32)),
            "view2_features": torch.from_numpy(np.ascontiguousarray(v2_feat).astype(np.float32)),
            "slide_id": self.slide_ids[idx],
        }

        # Coords may be None if augmentor drops all (shouldn't happen, but be safe)
        if v1_coord is not None:
            result["view1_coords"] = torch.from_numpy(
                np.ascontiguousarray(v1_coord).astype(np.int32)
            )
        if v2_coord is not None:
            result["view2_coords"] = torch.from_numpy(
                np.ascontiguousarray(v2_coord).astype(np.int32)
            )

        return result

    # ── Memmap reading ────────────────────────────────────────────────────

    def _read_features(self, idx: int, n_patches: int) -> np.ndarray:
        chunk_id = self.feat_chunk_ids[idx]
        byte_offset = self.feat_offsets[idx]
        mm = self.feat_mmaps[chunk_id]
        elem_size = np.dtype(self.feat_dtype).itemsize
        start_elem = byte_offset // elem_size
        n_elems = n_patches * self.feat_dim
        flat = mm[start_elem : start_elem + n_elems]
        return flat.reshape(n_patches, self.feat_dim).copy()

    def _read_coords(self, idx: int, n_patches: int) -> np.ndarray:
        chunk_id = self.coord_chunk_ids[idx]
        byte_offset = self.coord_offsets[idx]
        mm = self.coord_mmaps[chunk_id]
        elem_size = np.dtype(self.coord_dtype_str).itemsize
        start_elem = byte_offset // elem_size
        n_elems = n_patches * self.coord_dim
        flat = mm[start_elem : start_elem + n_elems]
        return flat.reshape(n_patches, self.coord_dim).copy()

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_bag_sizes(self) -> np.ndarray:
        """Return array of raw patch counts per slide (pre-augmentation)."""
        return np.array(self.lengths)
