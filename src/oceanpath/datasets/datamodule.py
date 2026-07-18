"""
Lightning DataModule for MIL training.

Subsampling architecture
═══════════════════════════════════════════════════════════════════════════

Two distinct caps control how many patches each slide uses:

  1. dataset_max_instances (MmapDataset level)
     - TRAIN ONLY. Stochastic random subsampling for regularization.
       Each epoch sees a different random subset of patches.
     - Val/Test: None (use all patches for best predictions).
     - Default: None (no dataset-level subsampling).

  2. max_instances (Collator level)
     - Both train and eval. Hard ceiling on the padded tensor dimension.
       Prevents OOM from slides with >10k patches.
     - MILCollator (train): pre-allocated [B, max_instances, D] buffer.
     - SimpleMILCollator (eval): dynamic padding but capped at max_instances.
     - Default: 8000.

The model has NO subsampling logic — it receives pre-subsampled, padded
tensors with a mask and processes them as-is.

Collation:
  MILCollator: pre-allocated pinned-memory buffer (training, fixed-size)
  SimpleMILCollator: dynamic padding to batch-max, capped (val/test)
"""

import logging

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from oceanpath.contracts import normalize_slide_id
from oceanpath.datasets.mmap import MmapDataset
from oceanpath.splitting import get_slide_ids_for_fold, load_splits

logger = logging.getLogger(__name__)


def _cuda_pin_memory_available() -> bool:
    """Return True only when CUDA pinned-memory transfer is genuinely usable."""
    try:
        if not torch.cuda.is_available():
            return False
        _ = torch.empty(1).pin_memory()
        return True
    except Exception:
        return False


# ── Collation ─────────────────────────────────────────────────────────────────


class MILCollator:
    """
    Pre-allocated collator for fixed-size batches.

    Allocates a single (batch_size, max_instances, D) buffer in pinned
    memory at init. Each __call__ zeroes and fills it, eliminating
    per-batch malloc + GC pressure.

    Used for TRAINING where bag sizes are bounded by dataset subsampling.
    """

    def __init__(
        self,
        max_instances: int,
        feat_dim: int,
        batch_size: int,
        pin_memory: bool = True,
    ):
        self.max_instances = max_instances
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.feat_buffer: torch.Tensor | None = None
        self.mask_buffer: torch.Tensor | None = None

    def _ensure_buffers(self, feat_dtype: torch.dtype) -> None:
        if self.feat_buffer is not None and self.feat_buffer.dtype == feat_dtype:
            return

        self.feat_buffer = torch.zeros(
            self.batch_size,
            self.max_instances,
            self.feat_dim,
            dtype=feat_dtype,
            pin_memory=self.pin_memory,
        )
        self.mask_buffer = torch.zeros(
            self.batch_size,
            self.max_instances,
            dtype=torch.float32,
            pin_memory=self.pin_memory,
        )

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        self._ensure_buffers(batch[0]["features"].dtype)
        features = self.feat_buffer[:B]
        mask = self.mask_buffer[:B]
        features.zero_()
        mask.zero_()

        labels = []
        lengths = []
        slide_ids = []
        has_coords = "coords" in batch[0]
        coords_list = [] if has_coords else None

        for i, sample in enumerate(batch):
            feat = sample["features"]  # [N, D], already subsampled by dataset
            n = min(feat.shape[0], self.max_instances)
            features[i, :n] = feat[:n]
            mask[i, :n] = 1.0
            labels.append(sample["label"])
            lengths.append(n)
            slide_ids.append(sample["slide_id"])
            if has_coords:
                coords_list.append(sample["coords"][:n])

        result = {
            "features": features.clone(),  # clone detaches from buffer
            "mask": mask.clone(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.int32),
            "slide_ids": slide_ids,
        }

        if has_coords:
            coord_dim = coords_list[0].shape[-1]
            coords_t = torch.zeros(B, self.max_instances, coord_dim, dtype=torch.int32)
            for i, c in enumerate(coords_list):
                n = min(c.shape[0], self.max_instances)
                coords_t[i, :n] = c[:n]
            result["coords"] = coords_t

        return result


_VALID_SAMPLING_MODES = ("contiguous", "random", "spatial_stratified")


def _seed_for_slide(seed: int, slide_id) -> int:
    """Stable per-slide seed derived from (collator_seed, slide_id).

    Ensures the same slide gets the same subsample across DataLoader
    workers, across DataLoader rebuilds, and across epochs — so the LP
    callback's score reflects representation drift, not sampling drift.
    """
    import hashlib

    h = hashlib.sha256(f"{int(seed)}:{slide_id}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def _spatial_stratified_indices(
    coords: torch.Tensor,
    n_target: int,
    generator: torch.Generator,
    grid_size: int = 32,
) -> torch.Tensor:
    """Sample ``n_target`` indices spread evenly across an ``grid_size`` x
    ``grid_size`` grid over the bag's bounding box.

    Falls back gracefully when bag size <= n_target (returns all indices)
    or when coords are degenerate (single point).
    """
    n = coords.shape[0]
    if n <= n_target:
        return torch.arange(n, dtype=torch.long)

    xy = coords[:, :2].to(torch.float32)
    mn = xy.min(dim=0).values
    mx = xy.max(dim=0).values
    rng = (mx - mn).clamp(min=1.0)

    cells = ((xy - mn) / rng * grid_size).floor().clamp(max=grid_size - 1).to(torch.long)
    cell_id = (cells[:, 0] * grid_size + cells[:, 1]).tolist()

    bins: dict[int, list[int]] = {}
    for i, c in enumerate(cell_id):
        bins.setdefault(c, []).append(i)

    non_empty = [k for k, v in bins.items() if v]
    k = len(non_empty)
    if k == 0:
        return torch.arange(min(n, n_target), dtype=torch.long)

    base = n_target // k
    extra = n_target - base * k

    # Random cell order — distributes the `extra` slots stochastically
    # so we don't always over-sample the same grid cells.
    cell_order = torch.randperm(k, generator=generator).tolist()

    chosen: list[int] = []
    for rank, cell_idx in enumerate(cell_order):
        bucket = bins[non_empty[cell_idx]]
        n_take = base + (1 if rank < extra else 0)
        n_take = min(n_take, len(bucket))
        if n_take == 0:
            continue
        sel = torch.randperm(len(bucket), generator=generator)[:n_take].tolist()
        chosen.extend(bucket[s] for s in sel)

    # Top up with random remaining indices if some cells were too small.
    if len(chosen) < n_target:
        chosen_set = set(chosen)
        remaining = [i for i in range(n) if i not in chosen_set]
        if remaining:
            need = n_target - len(chosen)
            sel = torch.randperm(len(remaining), generator=generator)[:need].tolist()
            chosen.extend(remaining[s] for s in sel)

    return torch.tensor(chosen[:n_target], dtype=torch.long)


class SimpleMILCollator:
    """
    Dynamic collator — pads to the max bag size within each batch.

    No pre-allocation. Used for val/test where batch composition varies
    and the full bag should be used for the best predictions.

    Parameters
    ----------
    max_instances : int or None
        Hard ceiling on the sequence dimension. If a bag exceeds this,
        it is reduced to ``max_instances`` according to ``sampling_mode``.
        None = no ceiling (dynamic padding to batch-max), and
        ``sampling_mode`` is irrelevant.
    sampling_mode : str
        How to choose which tokens to keep when a bag exceeds
        ``max_instances``.
          - ``contiguous`` (default): deterministic first-N. Matches the
            ``contiguous`` pretraining pre-cap; fastest.
          - ``random``: uniform without replacement, deterministic per
            slide via a slide-id-derived seed.
          - ``spatial_stratified``: bucket tokens into a grid over the
            bag's bounding box and sample evenly across cells. Requires
            ``coords`` in the sample dict; falls back to ``random`` when
            coords are absent.
    seed : int
        Base seed combined with each slide_id to derive the per-slide
        RNG. The default keeps eval-time sampling reproducible across
        runs / epochs / workers.
    spatial_grid_size : int
        Grid resolution for ``spatial_stratified`` (default 32 → 1024 cells,
        matching MmapDataset.cap_grid_size used during pretraining).
    """

    def __init__(
        self,
        max_instances: int | None = None,
        sampling_mode: str = "contiguous",
        seed: int = 42,
        spatial_grid_size: int = 32,
    ):
        if sampling_mode not in _VALID_SAMPLING_MODES:
            raise ValueError(f"sampling_mode={sampling_mode!r} not in {_VALID_SAMPLING_MODES}")
        self.max_instances = max_instances
        self.sampling_mode = sampling_mode
        self.seed = int(seed)
        self.spatial_grid_size = int(spatial_grid_size)

    def _select_indices(
        self,
        n: int,
        n_target: int,
        coords: torch.Tensor | None,
        slide_id,
    ) -> torch.Tensor:
        if n <= n_target:
            return torch.arange(n, dtype=torch.long)
        if self.sampling_mode == "contiguous":
            return torch.arange(n_target, dtype=torch.long)

        gen = torch.Generator()
        gen.manual_seed(_seed_for_slide(self.seed, slide_id))

        if self.sampling_mode == "spatial_stratified" and coords is not None:
            return _spatial_stratified_indices(
                coords, n_target, gen, grid_size=self.spatial_grid_size
            )
        # random (also covers spatial_stratified without coords)
        return torch.randperm(n, generator=gen)[:n_target].clone()

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        feat_dim = batch[0]["features"].shape[1]
        feat_dtype = batch[0]["features"].dtype
        has_coords = "coords" in batch[0]

        # Per-sample subsample indices
        sample_indices: list[torch.Tensor] = []
        for s in batch:
            n_full = s["features"].shape[0]
            n_target = n_full if self.max_instances is None else min(n_full, self.max_instances)
            idx = self._select_indices(
                n=n_full,
                n_target=n_target,
                coords=s.get("coords") if has_coords else None,
                slide_id=s.get("slide_id"),
            )
            sample_indices.append(idx)

        actual_lengths = [int(idx.shape[0]) for idx in sample_indices]
        max_n = max(actual_lengths)

        features = torch.zeros(B, max_n, feat_dim, dtype=feat_dtype)
        mask = torch.zeros(B, max_n, dtype=torch.float32)
        labels = []
        lengths = []
        slide_ids = []
        coords_list = [] if has_coords else None

        for i, sample in enumerate(batch):
            idx = sample_indices[i]
            n = idx.shape[0]
            features[i, :n] = sample["features"][idx]
            mask[i, :n] = 1.0
            labels.append(sample["label"])
            lengths.append(n)
            slide_ids.append(sample["slide_id"])
            if has_coords:
                coords_list.append(sample["coords"][idx])

        result = {
            "features": features,
            "mask": mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.int32),
            "slide_ids": slide_ids,
        }

        if has_coords:
            coord_dim = coords_list[0].shape[-1]
            coords_t = torch.zeros(B, max_n, coord_dim, dtype=torch.int32)
            for i, c in enumerate(coords_list):
                n = min(c.shape[0], max_n)
                coords_t[i, :n] = c[:n]
            result["coords"] = coords_t

        return result


# ── Length-bucket batch sampler ────────────────────────────────────────────────


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group slides by patch count so that batches have similar lengths.

    Sorts slides by bag size, chunks them into buckets of ``bucket_size``
    slides, shuffles buckets, then yields batches of ``batch_size`` within
    each bucket.  This minimises wasted padding while preserving
    stochasticity across epochs.

    Compatible with ``WeightedRandomSampler`` via *pre-sorted indices* — the
    sampler itself handles shuffling, so set ``shuffle=False`` on DataLoader.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        bucket_size: int = 64,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        # Sort by length, chunk into buckets, shuffle within bucket
        order = np.argsort(self.lengths)
        n = len(order)
        # Chunk into buckets
        buckets = [order[i : i + self.bucket_size] for i in range(0, n, self.bucket_size)]
        # Shuffle bucket order
        self.rng.shuffle(buckets)
        for bucket in buckets:
            # Shuffle within bucket
            self.rng.shuffle(bucket)
            # Yield batches
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


# ── Distributed sampler wrapper ───────────────────────────────────────────────


class DistributedSamplerWrapper(DistributedSampler):
    """Shard the output of an arbitrary inner Sampler across DDP ranks.

    Replays the inner sampler at every ``__iter__`` so a stochastic inner
    sampler (e.g. ``WeightedRandomSampler``) draws a fresh epoch each time.
    The inner indices are then sliced rank-wise via the standard
    ``DistributedSampler`` machinery — each rank sees a disjoint chunk and
    the per-epoch sample budget across all ranks matches the inner sampler's
    own budget (no replication).

    Required for DDP because Lightning will not auto-wrap a user-provided
    sampler with ``DistributedSampler``; without this wrapper every rank
    would draw the same global indices and effectively replicate the data.
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset=range(len(sampler)),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.inner_sampler = sampler

    def __iter__(self):
        # One full pass of the inner sampler this epoch.
        inner_indices = list(self.inner_sampler)
        for pos in super().__iter__():
            yield inner_indices[pos]


def _ddp_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# ── DataModule ────────────────────────────────────────────────────────────────


class MILDataModule(L.LightningDataModule):
    """
    Lightning DataModule for MIL training from memmap stores.

    Parameters
    ----------
    mmap_dir : str
        Path to mmap directory.
    splits_dir : str
        Path containing splits.parquet.
    csv_path : str
        Manifest CSV with labels.
    label_column : str
        Column name for integer labels.
    filename_column : str
        Column used to derive slide_id (stem of filename).
    scheme : str
        Split scheme (kfold, holdout, custom_kfold, etc.).
    fold : int
        Current fold index.
    batch_size : int
        Training batch size.
    max_instances : int or None
        Collation cap — hard ceiling on padded tensor dimension.
        Both train and eval. Prevents OOM on large bags.
    dataset_max_instances : int or None
        Dataset-level subsampling. TRAIN ONLY.
        None = no subsampling (dataset returns all patches).
    num_workers : int
        DataLoader workers.
    prefetch_factor : int
        Number of prefetched batches per worker when num_workers > 0.
    pin_memory : bool or None
        Whether to pin host-memory batches for faster H2D copies.
        None = auto-detect CUDA safety.
    persistent_workers : bool or None
        Keep DataLoader workers alive across epochs. None = enabled iff
        num_workers > 0.
    class_weighted_sampling : bool
        Inverse-frequency weighted random sampling for training.
    instance_dropout : float
        Patch dropout rate (train only).
    feature_noise_std : float
        Feature noise std (train only).
    cache_size_mb : int
        LRU cache size per dataset (val/test only).
    return_coords : bool
        Return spatial coordinates.
    verify_splits : bool
        Verify split integrity hash before loading.
    use_preallocated_collator : bool
        Use pre-allocated collation buffers for training.
    force_float32 : bool
        Cast mmap features to fp32 in the dataset. False preserves the mmap
        dtype (typically float16) to reduce RAM usage in the worker queue.
    """

    def __init__(
        self,
        mmap_dir: str,
        splits_dir: str,
        csv_path: str,
        label_column: str = "label",
        filename_column: str = "filename",
        scheme: str = "kfold",
        fold: int = 0,
        batch_size: int = 1,
        max_instances: int | None = None,
        dataset_max_instances: int | None = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        class_weighted_sampling: bool = True,
        instance_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
        cache_size_mb: int = 0,
        return_coords: bool = False,
        verify_splits: bool = True,
        use_preallocated_collator: bool = True,
        force_float32: bool = True,
        refit_mode: bool = False,
        cap_strategy: str = "random",
        cap_grid_size: int = 32,
        eval_n_crops: int = 1,
        length_bucket: bool = False,
        length_bucket_size: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

        self.mmap_dir = mmap_dir
        self.splits_dir = splits_dir
        self.csv_path = csv_path
        self.label_column = label_column
        self.filename_column = filename_column
        self.scheme = scheme
        self.fold = fold
        self.batch_size = batch_size
        self.max_instances = max_instances
        self.dataset_max_instances = dataset_max_instances
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.class_weighted_sampling = class_weighted_sampling
        self.instance_dropout = instance_dropout
        self.feature_noise_std = feature_noise_std
        self.cache_size_mb = cache_size_mb
        self.return_coords = return_coords
        self.verify_splits = verify_splits
        self.use_preallocated_collator = use_preallocated_collator
        self.force_float32 = force_float32
        self.refit_mode = refit_mode
        self.cap_strategy = cap_strategy
        self.cap_grid_size = cap_grid_size
        self.eval_n_crops = eval_n_crops
        self.length_bucket = length_bucket
        self.length_bucket_size = length_bucket_size
        self.train_dataset: MmapDataset | None = None
        self.val_dataset: MmapDataset | None = None
        self.test_dataset: MmapDataset | None = None
        self._labels_map: dict | None = None
        self._feat_dim: int | None = None

    @property
    def feat_dim(self) -> int:
        if self._feat_dim is None:
            from oceanpath.storage.mmap import validate_mmap_dir

            meta = validate_mmap_dir(self.mmap_dir)
            self._feat_dim = meta["feat_dim"]
        return self._feat_dim

    @property
    def num_classes(self) -> int:
        labels = self._load_labels()
        return len(set(labels.values()))

    def _load_labels(self) -> dict[str, int]:
        if self._labels_map is not None:
            return self._labels_map
        df = pd.read_csv(self.csv_path)
        if self.filename_column not in df.columns:
            raise ValueError(
                f"filename_column='{self.filename_column}' not found. Columns: {list(df.columns)}"
            )
        df["slide_id"] = df[self.filename_column].astype(str).map(normalize_slide_id)
        self._labels_map = dict(zip(df["slide_id"], df[self.label_column].astype(int), strict=True))
        return self._labels_map

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        labels = self._load_labels()

        splits_df = load_splits(
            self.splits_dir,
            csv_path=self.csv_path if self.verify_splits else None,
            verify=self.verify_splits,
        )
        fold_ids = get_slide_ids_for_fold(
            splits_df,
            self.fold,
            scheme=self.scheme,
        )

        if stage in ("fit", None):
            if self.refit_mode:
                # REFIT MODE: ALL non-test slides go to train, no val
                all_train_ids = fold_ids["train"] + fold_ids["val"]
                logger.info(
                    f"Refit mode: {len(all_train_ids)} slides "
                    f"(train={len(fold_ids['train'])} + val={len(fold_ids['val'])})"
                )
            else:
                all_train_ids = fold_ids["train"]

            # TRAIN: stochastic subsampling + augmentation
            # dataset_max_instances provides regularization via random views
            self.train_dataset = MmapDataset(
                mmap_dir=self.mmap_dir,
                slide_ids=all_train_ids,  # <── was fold_ids["train"]
                labels=labels,
                max_instances=self.dataset_max_instances,
                is_train=True,
                cap_strategy=self.cap_strategy,
                cap_grid_size=self.cap_grid_size,
                instance_dropout=self.instance_dropout,
                feature_noise_std=self.feature_noise_std,
                cache_size_mb=0,  # never cache train (stochastic)
                return_coords=self.return_coords,
                force_float32=self.force_float32,
            )

            if self.refit_mode:
                # No validation in refit mode
                self.val_dataset = None
            else:
                # VAL: deterministic subsampling with the same dataset cap as train.
                # Collation is still dynamic unless use_preallocated_collator=True,
                # so batches are [B, N_batch, D] with N_batch <= max_instances.
                self.val_dataset = MmapDataset(
                    mmap_dir=self.mmap_dir,
                    slide_ids=fold_ids["val"],
                    labels=labels,
                    max_instances=self.dataset_max_instances,
                    is_train=False,
                    cap_strategy=self.cap_strategy,
                    cap_grid_size=self.cap_grid_size,
                    eval_crop_seed=0,
                    cache_size_mb=self.cache_size_mb if self.eval_n_crops <= 1 else 0,
                    return_coords=self.return_coords,
                    force_float32=self.force_float32,
                )

            # Log bag size statistics
            train_sizes = self.train_dataset.get_bag_sizes()
            logger.info(
                f"Fold {self.fold}: "
                f"train={len(self.train_dataset)} ({self.train_dataset.get_label_counts()})"
                + (
                    f", val={len(self.val_dataset)} ({self.val_dataset.get_label_counts()})"
                    if self.val_dataset
                    else ", val=None (refit mode)"
                )
            )
            if len(train_sizes) > 0:
                logger.info(
                    f"Bag sizes — train: mean={train_sizes.mean():.0f}, max={train_sizes.max()}"
                )
                if self.val_dataset:
                    val_sizes = self.val_dataset.get_bag_sizes()
                    logger.info(
                        f"Bag sizes — val: mean={val_sizes.mean():.0f}, max={val_sizes.max()}"
                    )

        if stage in ("test", "predict", None):
            test_ids = fold_ids.get("test", [])
            if test_ids:
                self.test_dataset = MmapDataset(
                    mmap_dir=self.mmap_dir,
                    slide_ids=test_ids,
                    labels=labels,
                    max_instances=self.dataset_max_instances,
                    is_train=False,
                    cap_strategy=self.cap_strategy,
                    cap_grid_size=self.cap_grid_size,
                    eval_crop_seed=0,
                    cache_size_mb=self.cache_size_mb if self.eval_n_crops <= 1 else 0,
                    return_coords=self.return_coords,
                    force_float32=self.force_float32,
                )
                logger.info(f"Test set: {len(self.test_dataset)} slides")

    # ── Collators ─────────────────────────────────────────────────────────

    def _resolve_pin_memory(self) -> bool:
        return _cuda_pin_memory_available() if self.pin_memory is None else bool(self.pin_memory)

    def _resolve_persistent_workers(self) -> bool:
        if self.persistent_workers is None:
            return self.num_workers > 0
        return bool(self.persistent_workers) and self.num_workers > 0

    def _loader_kwargs(self) -> dict:
        kw = {
            "num_workers": self.num_workers,
            "pin_memory": self._resolve_pin_memory(),
            "persistent_workers": self._resolve_persistent_workers(),
        }
        if self.num_workers > 0:
            kw["prefetch_factor"] = self.prefetch_factor
            kw["worker_init_fn"] = self._worker_init_fn
        return kw

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Reset per-worker RNG after fork so slide sampling diverges by worker."""
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        ds = info.dataset
        if hasattr(ds, "rng"):
            ds.rng = np.random.default_rng(info.seed & 0xFFFF_FFFF)

    def _train_collator(self):
        """Pre-allocated collator for training (fixed max_instances cap)."""
        if self.use_preallocated_collator and self.max_instances is not None:
            return MILCollator(
                max_instances=self.max_instances,
                feat_dim=self.feat_dim,
                batch_size=self.batch_size,
                pin_memory=self._resolve_pin_memory(),
            )
        return SimpleMILCollator(max_instances=self.max_instances)

    def _eval_collator(self):
        """Collator for eval.

        Uses pre-allocated fixed-size buffers only when
        use_preallocated_collator=True. Otherwise pads dynamically to the
        max bag size in the current batch, capped at max_instances.
        """
        if self.use_preallocated_collator and self.max_instances is not None:
            return MILCollator(
                max_instances=self.max_instances,
                feat_dim=self.feat_dim,
                batch_size=self.batch_size,
                pin_memory=self._resolve_pin_memory(),
            )
        return SimpleMILCollator(max_instances=self.max_instances)

    # ── Sampler ───────────────────────────────────────────────────────────

    def _get_sampler(self, dataset: MmapDataset):
        if not self.class_weighted_sampling:
            return None
        labels = dataset.get_all_labels()
        if len(labels) == 0:
            return None
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            logger.warning(
                f"Only {len(unique)} class(es) in training set — "
                f"class-weighted sampling may not be meaningful."
            )
        weights_per_class = 1.0 / counts.astype(float)
        class_to_weight = dict(zip(unique, weights_per_class, strict=True))
        sample_weights = np.array([class_to_weight[label] for label in labels])
        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        if self.length_bucket:
            if _ddp_active():
                # LengthBucketBatchSampler yields whole batches; sharding it
                # across ranks would require a custom DistributedBatchSampler
                # that we do not have. Fail loudly rather than silently
                # replicating batches across ranks.
                raise NotImplementedError(
                    "length_bucket=True is not compatible with DDP. "
                    "Set training.length_bucket=false (the supervised "
                    "fixed-N regime already pads to a constant length)."
                )
            if self.class_weighted_sampling:
                logger.warning(
                    "length_bucket=True overrides class_weighted_sampling. "
                    "Set class_weighted_sampling=false to silence this warning, "
                    "or set length_bucket=false for imbalanced tasks."
                )
            # Length-bucketed batching: group similar-length slides together
            batch_sampler = LengthBucketBatchSampler(
                lengths=self.train_dataset.get_bag_sizes(),
                batch_size=self.batch_size,
                bucket_size=self.length_bucket_size,
                drop_last=False,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self._train_collator(),
                **self._loader_kwargs(),
            )

        sampler = self._get_sampler(self.train_dataset)
        # Under DDP, Lightning auto-wraps eval loaders (no sampler) but skips
        # train loaders that already carry a custom sampler. Wrap manually so
        # WeightedRandomSampler (or any future custom sampler) shards across
        # ranks instead of replicating its full draw on every rank.
        if _ddp_active():
            if sampler is None:
                # Vanilla random shuffle — DistributedSampler does the work.
                sampler = DistributedSampler(self.train_dataset, shuffle=True)
            else:
                sampler = DistributedSamplerWrapper(sampler, shuffle=False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=self._train_collator(),
            drop_last=False,
            **self._loader_kwargs(),
        )

    def set_eval_crop_seed(self, dataset: MmapDataset, seed: int) -> None:
        """Set the eval crop seed on a dataset (for multi-crop evaluation)."""
        dataset.eval_crop_seed = seed
        # Invalidate cache since different seed → different crops
        if dataset._cache is not None:
            dataset._cache.clear()
            dataset._cache_current_bytes = 0

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None  # refit mode has no validation set
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._eval_collator(),
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("No test set for this scheme/fold")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._eval_collator(),
            **self._loader_kwargs(),
        )
