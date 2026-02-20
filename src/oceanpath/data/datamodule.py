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
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from oceanpath.data.dataset import MmapDataset
from oceanpath.data.splits import get_slide_ids_for_fold, load_splits

logger = logging.getLogger(__name__)


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

        use_pin = pin_memory and torch.cuda.is_available()
        self.feat_buffer = torch.zeros(
            batch_size,
            max_instances,
            feat_dim,
            dtype=torch.float32,
            pin_memory=use_pin,
        )
        self.mask_buffer = torch.zeros(
            batch_size,
            max_instances,
            dtype=torch.float32,
            pin_memory=use_pin,
        )

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
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


class SimpleMILCollator:
    """
    Dynamic collator — pads to the max bag size within each batch.

    No pre-allocation. Used for val/test where batch composition varies
    and the full bag should be used for the best predictions.

    Parameters
    ----------
    max_instances : int or None
        Hard ceiling on the sequence dimension. If a bag exceeds this,
        it is truncated to max_instances (deterministic first-N).
        None = no ceiling (dynamic padding to batch-max).
    """

    def __init__(self, max_instances: int | None = None):
        self.max_instances = max_instances

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        feat_dim = batch[0]["features"].shape[1]

        # Compute actual sequence length per sample (respecting cap)
        actual_lengths = []
        for s in batch:
            n = s["features"].shape[0]
            if self.max_instances is not None:
                n = min(n, self.max_instances)
            actual_lengths.append(n)

        max_n = max(actual_lengths)

        features = torch.zeros(B, max_n, feat_dim, dtype=torch.float32)
        mask = torch.zeros(B, max_n, dtype=torch.float32)
        labels = []
        lengths = []
        slide_ids = []
        has_coords = "coords" in batch[0]
        coords_list = [] if has_coords else None

        for i, sample in enumerate(batch):
            n = actual_lengths[i]
            features[i, :n] = sample["features"][:n]
            mask[i, :n] = 1.0
            labels.append(sample["label"])
            lengths.append(n)
            slide_ids.append(sample["slide_id"])
            if has_coords:
                coords_list.append(sample["coords"][:n])

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
    outer_fold : int or None
        Outer fold for nested_cv.
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
        outer_fold: int | None = None,
        batch_size: int = 1,
        max_instances: int | None = None,
        dataset_max_instances: int | None = None,
        num_workers: int = 4,
        class_weighted_sampling: bool = True,
        instance_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
        cache_size_mb: int = 0,
        return_coords: bool = False,
        verify_splits: bool = True,
        use_preallocated_collator: bool = True,
        refit_mode: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mmap_dir = mmap_dir
        self.splits_dir = splits_dir
        self.csv_path = csv_path
        self.label_column = label_column
        self.filename_column = filename_column
        self.scheme = scheme
        self.fold = fold
        self.outer_fold = outer_fold
        self.batch_size = batch_size
        self.max_instances = max_instances
        self.dataset_max_instances = dataset_max_instances
        self.num_workers = num_workers
        self.class_weighted_sampling = class_weighted_sampling
        self.instance_dropout = instance_dropout
        self.feature_noise_std = feature_noise_std
        self.cache_size_mb = cache_size_mb
        self.return_coords = return_coords
        self.verify_splits = verify_splits
        self.use_preallocated_collator = use_preallocated_collator
        self.refit_mode = refit_mode
        self.train_dataset: MmapDataset | None = None
        self.val_dataset: MmapDataset | None = None
        self.test_dataset: MmapDataset | None = None
        self._labels_map: dict | None = None
        self._feat_dim: int | None = None

    @property
    def feat_dim(self) -> int:
        if self._feat_dim is None:
            from oceanpath.data.mmap_builder import validate_mmap_dir

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
        df["slide_id"] = df[self.filename_column].astype(str).apply(lambda x: Path(x).stem)
        self._labels_map = dict(zip(df["slide_id"], df[self.label_column].astype(int)))
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
            outer_fold=self.outer_fold,
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
                instance_dropout=self.instance_dropout,
                feature_noise_std=self.feature_noise_std,
                cache_size_mb=0,  # never cache train (stochastic)
                return_coords=self.return_coords,
            )

            if self.refit_mode:
                # No validation in refit mode
                self.val_dataset = None
            else:
                # VAL: NO subsampling — use all patches for best predictions.
                # The collator caps at max_instances to prevent OOM.
                self.val_dataset = MmapDataset(
                    mmap_dir=self.mmap_dir,
                    slide_ids=fold_ids["val"],
                    labels=labels,
                    max_instances=None,  # val sees ALL patches
                    is_train=False,
                    cache_size_mb=self.cache_size_mb,
                    return_coords=self.return_coords,
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
                    max_instances=None,  # test sees ALL patches
                    is_train=False,
                    cache_size_mb=self.cache_size_mb,
                    return_coords=self.return_coords,
                )
                logger.info(f"Test set: {len(self.test_dataset)} slides")

    # ── Collators ─────────────────────────────────────────────────────────

    def _train_collator(self):
        """Pre-allocated collator for training (fixed max_instances cap)."""
        if self.use_preallocated_collator and self.max_instances is not None:
            return MILCollator(
                max_instances=self.max_instances,
                feat_dim=self.feat_dim,
                batch_size=self.batch_size,
                pin_memory=torch.cuda.is_available(),
            )
        return SimpleMILCollator(max_instances=self.max_instances)

    def _eval_collator(self):
        """Dynamic collator for eval — capped at max_instances to prevent OOM."""
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
        class_to_weight = dict(zip(unique, weights_per_class))
        sample_weights = np.array([class_to_weight[label] for label in labels])
        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        sampler = self._get_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            collate_fn=self._train_collator(),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None  # refit mode has no validation set
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._eval_collator(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("No test set for this scheme/fold")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._eval_collator(),
            pin_memory=torch.cuda.is_available(),
        )
