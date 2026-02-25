"""
Lightning DataModule for SSL pretraining.

Key differences from the supervised MILDataModule:

  - No labels.  Everything is unlabeled.
  - No external splits.  Slide IDs come from the mmap index, optionally
    filtered by a pretrain.csv manifest.  Train/val split is a simple
    random percentage (default 90/10).
  - No test set.  SSL pretraining quality is monitored via loss + RankMe /
    alpha-ReQ callbacks, not held-out metrics.
  - No class-weighted sampling.  Just shuffle.
  - Custom DualViewCollator that pads two variable-length views independently,
    producing the {view1, mask1, view2, mask2} interface that
    SSLPretrainModule.training_step() expects.
  - Builds the coords-aware FeatureAugmentor from config and passes it to
    PretrainDataset.

Collation contract
═══════════════════════════════════════════════════════════════════════════
Each sample from PretrainDataset contains:
  view1_features (M₁, D)   view2_features (M₂, D)
  view1_coords   (M₁, 2)   view2_coords   (M₂, 2)

DualViewCollator pads each view INDEPENDENTLY to its own batch-max,
capped at max_instances:
  view1 [B, M₁_max, D]  mask1 [B, M₁_max]
  view2 [B, M₂_max, D]  mask2 [B, M₂_max]

This avoids wasting memory padding short views to the longer view's max.
"""

import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from oceanpath.data.mmap_builder import validate_mmap_dir
from oceanpath.data.pretrain_dataset import PretrainDataset
from oceanpath.ssl.augmentation import (
    build_augmentor,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Dual-View Collator
# ═════════════════════════════════════════════════════════════════════════════


class DualViewCollator:
    """
    Collate dual-view samples into padded batches.

    Each view is padded independently to its own batch-max length,
    capped at max_instances.  Produces the {view1, mask1, view2, mask2}
    dict that SSLPretrainModule expects.

    Parameters
    ----------
    max_instances : int
        Hard ceiling on sequence length per view (prevents OOM).
    """

    def __init__(self, max_instances: int = 4096):
        self.max_instances = max_instances

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        D = batch[0]["view1_features"].shape[-1]

        # Compute actual lengths per view (respecting cap)
        lengths1, lengths2 = [], []
        for s in batch:
            lengths1.append(min(s["view1_features"].shape[0], self.max_instances))
            lengths2.append(min(s["view2_features"].shape[0], self.max_instances))

        max_n1 = max(lengths1)
        max_n2 = max(lengths2)

        # Allocate padded tensors
        view1 = torch.zeros(B, max_n1, D, dtype=torch.float32)
        mask1 = torch.zeros(B, max_n1, dtype=torch.float32)
        view2 = torch.zeros(B, max_n2, D, dtype=torch.float32)
        mask2 = torch.zeros(B, max_n2, dtype=torch.float32)

        slide_ids = []
        has_coords = "view1_coords" in batch[0]
        coords1_list = [] if has_coords else None
        coords2_list = [] if has_coords else None

        for i, sample in enumerate(batch):
            n1 = lengths1[i]
            n2 = lengths2[i]

            view1[i, :n1] = sample["view1_features"][:n1]
            mask1[i, :n1] = 1.0
            view2[i, :n2] = sample["view2_features"][:n2]
            mask2[i, :n2] = 1.0

            slide_ids.append(sample["slide_id"])

            if has_coords:
                coords1_list.append(sample.get("view1_coords", None))
                coords2_list.append(sample.get("view2_coords", None))

        result = {
            "view1": view1,
            "mask1": mask1,
            "view2": view2,
            "mask2": mask2,
            "lengths1": torch.tensor(lengths1, dtype=torch.int32),
            "lengths2": torch.tensor(lengths2, dtype=torch.int32),
            "slide_ids": slide_ids,
        }

        # Optionally pack coords (needed if aggregator uses PPEG)
        if has_coords and coords1_list[0] is not None:
            coord_dim = coords1_list[0].shape[-1]
            c1 = torch.zeros(B, max_n1, coord_dim, dtype=torch.int32)
            c2 = torch.zeros(B, max_n2, coord_dim, dtype=torch.int32)
            for i in range(B):
                n1 = lengths1[i]
                n2 = lengths2[i]
                if coords1_list[i] is not None:
                    c1[i, :n1] = coords1_list[i][:n1]
                if coords2_list[i] is not None:
                    c2[i, :n2] = coords2_list[i][:n2]
            result["coords1"] = c1
            result["coords2"] = c2

        return result


# ═════════════════════════════════════════════════════════════════════════════
# DataModule
# ═════════════════════════════════════════════════════════════════════════════


class PretrainDataModule(L.LightningDataModule):
    """
    Lightning DataModule for SSL pretraining.

    Parameters
    ----------
    mmap_dir : str
        Path to mmap directory.
    csv_path : str or None
        Path to pretrain.csv manifest.  If provided, only slides listed
        in this CSV are used.  None = use ALL slides in the mmap index.
    filename_column : str
        Column name in pretrain.csv containing slide IDs.
    augmentation_cfg : dict
        Augmentation config from pretrain.yaml → pretrain_training.augmentation.
    coords_aware : bool
        If True, enable coords-aware augmentation transforms.
    batch_size : int
        Training batch size (typically 32 for SSL).
    max_instances : int
        Hard ceiling on padded sequence length per view (collator cap).
    dataset_max_instances : int or None
        Cap patches per slide before augmentation (memory safety).
    num_workers : int
        DataLoader workers.
    val_frac : float
        Fraction of slides held out for validation (default 0.1).
    seed : int
        Random seed for reproducible train/val split.
    """

    def __init__(
        self,
        mmap_dir: str,
        csv_path: str | None = None,
        filename_column: str = "slide_id",
        augmentation_cfg: dict | None = None,
        coords_aware: bool = True,
        batch_size: int = 32,
        max_instances: int = 4096,
        dataset_max_instances: int | None = None,
        num_workers: int = 8,
        val_frac: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mmap_dir = mmap_dir
        self.csv_path = csv_path
        self.filename_column = filename_column
        self.augmentation_cfg = augmentation_cfg or {}
        self.coords_aware = coords_aware
        self.batch_size = batch_size
        self.max_instances = max_instances
        self.dataset_max_instances = dataset_max_instances
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.seed = seed

        self.train_dataset: PretrainDataset | None = None
        self.val_dataset: PretrainDataset | None = None
        self._feat_dim: int | None = None

    @property
    def feat_dim(self) -> int:
        if self._feat_dim is None:
            meta = validate_mmap_dir(self.mmap_dir)
            self._feat_dim = meta["feat_dim"]
        return self._feat_dim

    # ── Slide ID loading ──────────────────────────────────────────────────

    def _load_slide_ids(self) -> list[str]:
        """
        Load slide IDs from mmap index, optionally filtered by pretrain.csv.

        Returns sorted list (deterministic ordering for reproducible splits).
        """
        # All slide IDs in the mmap store
        idx = np.load(str(Path(self.mmap_dir) / "index_arrays.npz"), allow_pickle=True)
        mmap_ids = set(idx["slide_ids"].tolist())

        if self.csv_path is not None and Path(self.csv_path).exists():
            # Filter to slides listed in pretrain.csv
            df = pd.read_csv(self.csv_path)
            csv_ids = set(
                df[self.filename_column].astype(str).apply(lambda x: Path(x).stem).tolist()
            )
            # Intersection: must be in BOTH csv and mmap
            slide_ids = sorted(mmap_ids & csv_ids)
            n_csv_only = len(csv_ids - mmap_ids)
            n_mmap_only = len(mmap_ids - csv_ids)
            if n_csv_only > 0:
                logger.warning(
                    f"{n_csv_only} slides in pretrain.csv but not in mmap (missing features?)"
                )
            if n_mmap_only > 0:
                logger.info(
                    f"{n_mmap_only} slides in mmap but not in pretrain.csv (excluded by filter)"
                )
        else:
            slide_ids = sorted(mmap_ids)
            if self.csv_path is not None:
                logger.warning(
                    f"pretrain.csv not found at {self.csv_path} — "
                    f"using ALL {len(slide_ids)} slides from mmap"
                )

        logger.info(f"Loaded {len(slide_ids)} slide IDs for pretraining")
        return slide_ids

    # ── Train/Val split ───────────────────────────────────────────────────

    def _split_slide_ids(self, slide_ids: list[str]) -> tuple[list[str], list[str]]:
        """
        Random train/val split.

        Uses a seeded RNG for reproducibility.  The split is stable
        across runs with the same seed + same slide set.
        """
        rng = np.random.default_rng(self.seed)
        n = len(slide_ids)
        n_val = max(1, int(n * self.val_frac))

        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_ids = [slide_ids[i] for i in train_idx]
        val_ids = [slide_ids[i] for i in val_idx]

        logger.info(
            f"Train/val split: {len(train_ids)} train, {len(val_ids)} val "
            f"(val_frac={self.val_frac}, seed={self.seed})"
        )
        return train_ids, val_ids

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return  # already set up

        slide_ids = self._load_slide_ids()
        train_ids, val_ids = self._split_slide_ids(slide_ids)

        # Build augmentor from config
        augmentor = build_augmentor(self.augmentation_cfg, coords_aware=self.coords_aware)

        # Train: full augmentation, stochastic views
        self.train_dataset = PretrainDataset(
            mmap_dir=self.mmap_dir,
            slide_ids=train_ids,
            augmentor=augmentor,
            dataset_max_instances=self.dataset_max_instances,
        )

        # Val: same augmentor (still produces dual views for val loss),
        # but the loss comparison is meaningful because both train and val
        # use identical augmentation pipelines.
        self.val_dataset = PretrainDataset(
            mmap_dir=self.mmap_dir,
            slide_ids=val_ids,
            augmentor=augmentor,
            dataset_max_instances=self.dataset_max_instances,
        )

        # Log bag-size stats
        train_sizes = self.train_dataset.get_bag_sizes()
        val_sizes = self.val_dataset.get_bag_sizes()
        if len(train_sizes) > 0:
            logger.info(
                f"Bag sizes — train: mean={train_sizes.mean():.0f}, "
                f"median={np.median(train_sizes):.0f}, max={train_sizes.max()}"
            )
        if len(val_sizes) > 0:
            logger.info(
                f"Bag sizes — val: mean={val_sizes.mean():.0f}, "
                f"median={np.median(val_sizes):.0f}, max={val_sizes.max()}"
            )

    # ── Collators ─────────────────────────────────────────────────────────

    def _collator(self) -> DualViewCollator:
        return DualViewCollator(max_instances=self.max_instances)

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collator(),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,  # drop last for stable batch-norm in projector
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collator(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
