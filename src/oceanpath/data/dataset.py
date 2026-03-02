"""
Memmap-backed datasets for MIL training.

Reads slide features from the chunked binary stores produced by build_mmap.py.

BaseMmapDataset
===============
Shared memmap reading logic: index loading, slide-ID filtering, chunk-level
memmap access. Subclassed by MmapDataset (supervised) and PretrainDataset (SSL).

MmapDataset
===========
Supervised single-view dataset. Each __getitem__ returns one slide's features,
coords, label, and metadata.

Subsampling contract
=======================================================================
ALL patch-level subsampling happens in the DATA LAYER (here + collator).
The model receives exactly what it gets -- no further truncation.

  - max_instances (dataset level):
      Train: stochastic random sampling (different view each epoch)
      Val/Test: deterministic first-N (reproducible)
  - instance_dropout: randomly drops patches AFTER subsampling (train only)
  - The collator then pads/truncates to a uniform batch dimension.

The model has NO subsampling logic. This keeps the model clean and
makes the data pipeline the single source of truth for what enters forward().

File layout expected:
  {mmap_dir}/
  |- features_000.bin [, ...]
  |- coords_000.bin   [, ...]
  +- index_arrays.npz
"""

import logging
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from oceanpath.data.mmap_builder import validate_mmap_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Base dataset
# =============================================================================


class BaseMmapDataset(Dataset):
    """Base class for memmap-backed slide-level datasets.

    Handles common setup: mmap validation, index loading, slide-ID filtering,
    chunk-level memmap opening, and raw feature/coord reads.

    Subclasses implement ``__getitem__`` with their own augmentation and
    return format.

    Parameters
    ----------
    mmap_dir : str
        Directory containing features_*.bin, coords_*.bin, index_arrays.npz.
    slide_ids : list[str] or None
        Slide IDs to include. None = use ALL slides in the mmap index.
    load_coords : bool
        Whether to preload coords into RAM at init.
    """

    def __init__(
        self,
        mmap_dir: str,
        slide_ids: list[str] | None = None,
        load_coords: bool = False,
    ):
        super().__init__()
        self.mmap_dir = Path(mmap_dir)
        self.rng = np.random.default_rng()

        meta = validate_mmap_dir(mmap_dir)
        self.feat_dim: int = meta["feat_dim"]
        self.coord_dim: int = meta["coord_dim"]

        # Load index
        idx = np.load(str(self.mmap_dir / "index_arrays.npz"), allow_pickle=True)
        all_slide_ids = idx["slide_ids"].tolist()
        self.feat_dtype: str = str(idx["feat_dtype"][0])
        self.coord_dtype_str: str = str(idx["coord_dtype"][0])
        self._feat_elem_size: int = np.dtype(self.feat_dtype).itemsize
        self._coord_elem_size: int = np.dtype(self.coord_dtype_str).itemsize
        n_feat_chunks = int(idx["num_feat_chunks"])
        n_coord_chunks = int(idx["num_coord_chunks"])

        # Filter to requested slide_ids (or use all)
        id_to_idx = {sid: i for i, sid in enumerate(all_slide_ids)}
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
                logger.warning("Slide '%s' has 0 patches — skipping.", sid)
                continue
            self.slide_ids.append(sid)
            self.lengths.append(n)
            self.feat_chunk_ids.append(int(idx["feat_chunk_ids"][i]))
            self.feat_offsets.append(int(idx["feat_offsets"][i]))
            self.coord_chunk_ids.append(int(idx["coord_chunk_ids"][i]))
            self.coord_offsets.append(int(idx["coord_offsets"][i]))

        if missing:
            logger.warning(
                "%d/%d slide_ids not found in mmap index. First 5: %s",
                len(missing),
                len(slide_ids),
                missing[:5],
            )

        # Open feature memmaps (read-only -> OS shares physical pages across workers)
        self.feat_mmaps: list[np.memmap] = []
        for c in range(n_feat_chunks):
            path = self.mmap_dir / f"features_{c:03d}.bin"
            self.feat_mmaps.append(np.memmap(str(path), dtype=self.feat_dtype, mode="r"))

        # Preload coords into RAM (conditional).
        # Coords are small (~8 bytes/patch for int32 x 2D). Preloading
        # eliminates coord disk seeks during __getitem__.
        self._preloaded_coords: list[np.ndarray] | None = None
        if load_coords:
            self._preload_coords(n_coord_chunks)

    def __len__(self) -> int:
        return len(self.slide_ids)

    # -- Coord preloading ------------------------------------------------------

    def _preload_coords(self, n_coord_chunks: int) -> None:
        """Read all coords for selected slides into RAM at init time.

        Opens coord memmaps temporarily, reads each slide's coords into a
        plain numpy array, then releases the mmap handles. The preloaded
        arrays are shared across DataLoader workers via COW fork semantics.
        """
        coord_mmaps = []
        for c in range(n_coord_chunks):
            path = self.mmap_dir / f"coords_{c:03d}.bin"
            coord_mmaps.append(np.memmap(str(path), dtype=self.coord_dtype_str, mode="r"))

        self._preloaded_coords = []
        total_bytes = 0

        for i in range(len(self.slide_ids)):
            chunk_id = self.coord_chunk_ids[i]
            byte_offset = self.coord_offsets[i]
            n_patches = self.lengths[i]
            mm = coord_mmaps[chunk_id]
            start_elem = byte_offset // self._coord_elem_size
            n_elems = n_patches * self.coord_dim
            flat = mm[start_elem : start_elem + n_elems]
            arr = flat.reshape(n_patches, self.coord_dim).copy()
            self._preloaded_coords.append(arr)
            total_bytes += arr.nbytes

        del coord_mmaps
        logger.info(
            "Preloaded coords: %d slides, %.1f MB",
            len(self._preloaded_coords),
            total_bytes / 1e6,
        )

    def _require_coords(self) -> list[np.ndarray]:
        """Return preloaded coords or raise if not loaded."""
        if self._preloaded_coords is not None:
            return self._preloaded_coords
        raise RuntimeError("Coords not loaded. Set load_coords=True in BaseMmapDataset.__init__.")

    # -- Feature memmap reading ------------------------------------------------

    def _feature_view(self, idx: int, start_patch: int, n_patches: int) -> np.ndarray:
        """Return a view into feature memmap for [start_patch : start_patch + n_patches]."""
        chunk_id = self.feat_chunk_ids[idx]
        byte_offset = self.feat_offsets[idx] + start_patch * self.feat_dim * self._feat_elem_size
        mm = self.feat_mmaps[chunk_id]
        start_elem = byte_offset // self._feat_elem_size
        n_elems = n_patches * self.feat_dim
        return mm[start_elem : start_elem + n_elems].reshape(n_patches, self.feat_dim)

    def _read_features(self, idx: int, n_patches: int) -> np.ndarray:
        """Read all features for one slide (copy from memmap)."""
        return self._feature_view(idx, start_patch=0, n_patches=n_patches).copy()

    def _read_features_window(self, idx: int, start_patch: int, n_patches: int) -> np.ndarray:
        """Read a contiguous window of features for one slide."""
        return self._feature_view(idx, start_patch=start_patch, n_patches=n_patches).copy()

    def _read_features_indices(
        self,
        idx: int,
        total_patches: int,
        row_indices: np.ndarray,
    ) -> np.ndarray:
        """Read selected feature rows without copying the full slide first."""
        view = self._feature_view(idx, start_patch=0, n_patches=total_patches)
        return np.ascontiguousarray(view[np.asarray(row_indices, dtype=np.int64)])

    # -- Coord reading (from preloaded RAM cache) ------------------------------

    def _read_coords(self, idx: int, n_patches: int) -> np.ndarray:
        """Return coords for slide idx from preloaded RAM cache."""
        return self._require_coords()[idx].copy()

    def _read_coords_window(self, idx: int, start_patch: int, n_patches: int) -> np.ndarray:
        """Read a contiguous window of preloaded coords."""
        return self._require_coords()[idx][start_patch : start_patch + n_patches].copy()

    def _read_coords_indices(self, idx: int, row_indices: np.ndarray) -> np.ndarray:
        """Read selected preloaded coord rows."""
        return np.ascontiguousarray(
            self._require_coords()[idx][np.asarray(row_indices, dtype=np.int64)]
        )

    # -- Utilities -------------------------------------------------------------

    def get_bag_sizes(self) -> np.ndarray:
        """Return array of patch counts per slide."""
        return np.array(self.lengths)


# =============================================================================
# Supervised dataset
# =============================================================================


class MmapDataset(BaseMmapDataset):
    """Supervised slide-level dataset backed by chunked memmap binary stores.

    Parameters
    ----------
    mmap_dir : str
        Directory containing features_*.bin, coords_*.bin, index_arrays.npz.
    slide_ids : list[str]
        Slide IDs to include (subset for this split/fold).
    labels : dict[str, int]
        Mapping slide_id -> integer label.
    max_instances : int or None
        Cap patches per slide at load time. None = use all patches.
        Train: stochastic random sampling. Val/Test: deterministic first-N.
    is_train : bool
        If True, subsampling is stochastic and augmentations apply.
    instance_dropout : float
        Probability of dropping each patch after subsampling (train only).
    feature_noise_std : float
        Std of additive Gaussian noise on features (train only).
    cache_size_mb : int
        LRU cache size in MB for val/test. 0 = disabled. Ignored for train.
    return_coords : bool
        Whether to return spatial coordinates alongside features.
    force_float32 : bool
        If True, cast features to float32 in __getitem__. Default True
        for backward compatibility with supervised collators that
        pre-allocate float32 buffers.
    """

    def __init__(
        self,
        mmap_dir: str,
        slide_ids: list[str],
        labels: dict[str, int],
        max_instances: int | None = None,
        is_train: bool = True,
        instance_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
        cache_size_mb: int = 0,
        return_coords: bool = False,
        force_float32: bool = True,
    ):
        super().__init__(mmap_dir=mmap_dir, slide_ids=slide_ids, load_coords=return_coords)
        self.max_instances = max_instances
        self.is_train = is_train
        self.instance_dropout = instance_dropout
        self.feature_noise_std = feature_noise_std
        self.return_coords = return_coords
        self.force_float32 = force_float32

        self.labels_list: list[int] = [labels.get(sid, -1) for sid in self.slide_ids]

        n_unlabeled = sum(1 for label in self.labels_list if label < 0)
        if n_unlabeled > 0:
            logger.warning(
                "%d/%d slides have no label (label=-1). Check your CSV mapping.",
                n_unlabeled,
                len(self.labels_list),
            )

        logger.info(
            "MmapDataset: %d slides, feat_dim=%d, dtype=%s, max_instances=%s, is_train=%s",
            len(self.slide_ids),
            self.feat_dim,
            self.feat_dtype,
            self.max_instances,
            self.is_train,
        )

        # LRU cache (val/test only — train is stochastic, caching freezes augmentation)
        self._cache: OrderedDict | None = None
        self._cache_max_bytes = int(cache_size_mb * 1e6) if cache_size_mb > 0 else 0
        self._cache_current_bytes = 0
        if self._cache_max_bytes > 0 and not is_train:
            self._cache = OrderedDict()

    def __getitem__(self, idx: int) -> dict:
        """Load one slide's features + label.

        Pipeline: read -> subsample -> augment -> tensor conversion

        Returns dict: features [N, D], label, slide_id, length, [coords]
        """
        if self._cache is not None and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        slide_id = self.slide_ids[idx]
        label = self.labels_list[idx]
        n_patches = self.lengths[idx]

        # 1. Read from memmap
        features = self._read_features(idx, n_patches)
        coords = self._read_coords(idx, n_patches) if self.return_coords else None

        # 2. Subsample to max_instances
        if self.max_instances is not None and n_patches > self.max_instances:
            features, coords = self._subsample(features, coords, n_patches)
            n_patches = features.shape[0]

        # 3. Augmentation (train only)
        if self.is_train:
            features, coords, n_patches = self._augment(features, coords, n_patches)

        # 4. Convert to tensors
        features = np.ascontiguousarray(features)
        if self.force_float32:
            features = features.astype(np.float32)
        features_t = torch.from_numpy(features)

        result = {
            "features": features_t,
            "label": label,
            "slide_id": slide_id,
            "length": n_patches,
        }
        if coords is not None:
            result["coords"] = torch.from_numpy(np.ascontiguousarray(coords).astype(np.int32))

        # Cache (val/test only)
        if self._cache is not None:
            item_bytes = features_t.nbytes
            while self._cache_current_bytes + item_bytes > self._cache_max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._cache_current_bytes -= evicted["features"].nbytes
            if item_bytes <= self._cache_max_bytes:
                self._cache[idx] = result
                self._cache_current_bytes += item_bytes

        return result

    # -- Subsampling -----------------------------------------------------------

    def _subsample(
        self,
        features: np.ndarray,
        coords: np.ndarray | None,
        n_patches: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Subsample to self.max_instances patches."""
        k = self.max_instances
        if self.is_train:
            indices = self.rng.permutation(n_patches)[:k]
        else:
            indices = np.arange(k)
        features = features[indices]
        if coords is not None:
            coords = coords[indices]
        return features, coords

    # -- Augmentation ----------------------------------------------------------

    def _augment(
        self,
        features: np.ndarray,
        coords: np.ndarray | None,
        n_patches: int,
    ) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Apply instance dropout and feature noise (train only)."""
        if self.instance_dropout > 0 and n_patches > 1:
            keep = self.rng.random(n_patches) > self.instance_dropout
            if not keep.any():
                keep[self.rng.integers(n_patches)] = True
            features = features[keep]
            if coords is not None:
                coords = coords[keep]
            n_patches = int(keep.sum())

        if self.feature_noise_std > 0:
            noise = self.rng.standard_normal(features.shape).astype(np.float32)
            features = features.astype(np.float32) + noise * self.feature_noise_std

        return features, coords, n_patches

    # -- Label utilities -------------------------------------------------------

    def get_label_counts(self) -> dict[int, int]:
        """Return {label: count} for all slides in this dataset."""
        return dict(Counter(self.labels_list))

    def get_all_labels(self) -> np.ndarray:
        """Return flat array of all labels (for sampler construction)."""
        return np.array(self.labels_list)
