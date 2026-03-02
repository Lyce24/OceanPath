"""
Lightning DataModule for SSL pretraining.

Key differences from the supervised MILDataModule:

  - No labels.  Everything is unlabeled.
  - Optional external split manifest. Slide IDs come from the mmap index,
    optionally filtered by pretrain.csv. Train/val can be:
      (a) explicit from split_manifest.json, or
      (b) random percentage split (default 90/10).
  - No test set.  SSL pretraining quality is monitored via loss + RankMe /
    alpha-ReQ callbacks, not held-out metrics.
  - No class-weighted sampling.  Just shuffle.
  - DualViewCollator (from batching.py) pads two variable-length views
    independently, producing the {view1, mask1, view2, mask2} interface
    that SSLPretrainModule.training_step() expects.
  - Builds the coords-aware FeatureAugmentor from config and passes it to
    PretrainDataset.

Collation contract
═══════════════════════════════════════════════════════════════════════════
Each sample from PretrainDataset contains:
  view1_features (M1, D)   view2_features (M2, D)
  view1_coords   (M1, 2)   view2_coords   (M2, 2)

DualViewCollator pads each view INDEPENDENTLY to its own batch-max,
capped at max_instances:
  view1 [B, M1_max, D]  mask1 [B, M1_max]
  view2 [B, M2_max, D]  mask2 [B, M2_max]

This avoids wasting memory padding short views to the longer view's max.
"""

import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from oceanpath.data.batching import (
    BatchingConfig,
    DualViewCollator,
    build_batch_sampler,
    build_collator,
)
from oceanpath.data.mmap_builder import validate_mmap_dir
from oceanpath.data.pretrain_dataset import PretrainDataset
from oceanpath.ssl.augmentation import build_augmentor

logger = logging.getLogger(__name__)


def _cuda_pin_memory_available() -> bool:
    """Return True only when CUDA is genuinely usable for pinned-memory transfer.

    Some environments expose a CUDA build but fail at runtime device queries.
    In that case, pinning triggers DataLoader runtime errors, so we disable it.
    """
    try:
        if not torch.cuda.is_available():
            return False
        _ = torch.empty(1).pin_memory()
        return True
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# DataModule
# ═════════════════════════════════════════════════════════════════════════════


class PretrainDataModule(L.LightningDataModule):
    """Lightning DataModule for SSL pretraining.

    Parameters
    ----------
    mmap_dir : str
        Path to mmap directory.
    csv_path : str or None
        Path to pretrain.csv manifest.  If provided, only slides listed
        in this CSV are used.  None = use ALL slides in the mmap index.
    filename_column : str
        Column name in pretrain.csv containing slide IDs.
    split_manifest_path : str or None
        Optional path to a JSON split manifest containing explicit
        ``train_ids`` and ``val_ids``. If provided, this deterministic split
        is used instead of random splitting.
    augmentation_cfg : dict
        Augmentation config from pretrain.yaml -> pretrain_training.augmentation.
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
    dataset_pre_cap_mode : str
        Pre-cap strategy for slides longer than dataset_max_instances.
        One of: {"random", "contiguous", "head"}.
    prefetch_factor : int
        DataLoader prefetch factor when num_workers > 0.
    pin_memory : bool or None
        If None, auto-detect CUDA-safe pinning. Otherwise force on/off.
    persistent_workers : bool or None
        If None, enabled when num_workers > 0. Otherwise force on/off.
    val_frac : float
        Fraction of slides held out for validation (default 0.1).
    seed : int
        Random seed for reproducible train/val split.
    batching_strategy : str
        Name of the batching strategy (see batching.py).
    batching_cfg : dict or None
        Extra overrides for BatchingConfig.
    force_float32 : bool
        If True, cast features to float32 in the dataset. Default False
        preserves the native mmap dtype (typically float16), halving memory
        bandwidth. With mixed-precision training, float16 is handled by autocast.
    """

    def __init__(
        self,
        mmap_dir: str,
        csv_path: str | None = None,
        filename_column: str = "slide_id",
        split_manifest_path: str | None = None,
        augmentation_cfg: dict | None = None,
        coords_aware: bool = True,
        batch_size: int = 32,
        max_instances: int = 4096,
        dataset_max_instances: int | None = None,
        dataset_pre_cap_mode: str = "random",
        num_workers: int = 8,
        prefetch_factor: int = 2,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        val_frac: float = 0.1,
        seed: int = 42,
        batching_strategy: str = "pad_to_max_in_batch",
        batching_cfg: dict | None = None,
        force_float32: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mmap_dir = mmap_dir
        self.csv_path = csv_path
        self.filename_column = filename_column
        self.split_manifest_path = split_manifest_path
        self.augmentation_cfg = augmentation_cfg or {}
        self.coords_aware = coords_aware
        self.batch_size = batch_size
        self.max_instances = max_instances
        self.dataset_max_instances = dataset_max_instances
        self.dataset_pre_cap_mode = dataset_pre_cap_mode
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_frac = val_frac
        self.seed = seed
        self.batching_strategy = batching_strategy
        self.batching_cfg = batching_cfg or {}
        self.force_float32 = force_float32

        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")

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
        """Load slide IDs from mmap index, optionally filtered by pretrain.csv.

        Returns sorted list (deterministic ordering for reproducible splits).
        """
        idx = np.load(str(Path(self.mmap_dir) / "index_arrays.npz"), allow_pickle=True)
        mmap_ids = set(idx["slide_ids"].tolist())

        if self.csv_path is not None and Path(self.csv_path).exists():
            df = pd.read_csv(self.csv_path)
            csv_ids = set(df[self.filename_column].astype(str).tolist())
            slide_ids = sorted(mmap_ids & csv_ids)
            n_csv_only = len(csv_ids - mmap_ids)
            n_mmap_only = len(mmap_ids - csv_ids)
            if n_csv_only > 0:
                logger.warning(
                    "%d slides in pretrain.csv but not in mmap (missing features?)",
                    n_csv_only,
                )
            if n_mmap_only > 0:
                logger.info(
                    "%d slides in mmap but not in pretrain.csv (excluded by filter)",
                    n_mmap_only,
                )
        else:
            slide_ids = sorted(mmap_ids)
            if self.csv_path is not None:
                logger.warning(
                    "pretrain.csv not found at %s — using ALL %d slides from mmap",
                    self.csv_path,
                    len(slide_ids),
                )

        logger.info("Loaded %d slide IDs for pretraining", len(slide_ids))
        return slide_ids

    # ── Train/Val split ───────────────────────────────────────────────────

    def _split_slide_ids(self, slide_ids: list[str]) -> tuple[list[str], list[str]]:
        """Train/val split from manifest (if provided), else random split.

        Manifest format::

            {"train_ids": [...], "val_ids": [...], "seed": 42, "val_frac": 0.1}
        """
        if getattr(self, "split_manifest_path", None) is not None:
            manifest_path = Path(self.split_manifest_path)
            if manifest_path.exists():
                payload = json.loads(manifest_path.read_text())
                train_ids = payload.get("train_ids", [])
                val_ids = payload.get("val_ids", [])

                slide_set = set(slide_ids)
                train_ids = [sid for sid in train_ids if sid in slide_set]
                val_ids = [sid for sid in val_ids if sid in slide_set]

                if train_ids and val_ids:
                    logger.info(
                        "Using split manifest: %s (%d train, %d val)",
                        manifest_path,
                        len(train_ids),
                        len(val_ids),
                    )
                    return train_ids, val_ids

                logger.warning(
                    "Split manifest found at %s but resolved split is empty "
                    "(train=%d, val=%d). Falling back to random split.",
                    manifest_path,
                    len(train_ids),
                    len(val_ids),
                )
            else:
                logger.warning(
                    "Split manifest path does not exist: %s. Falling back to random split.",
                    manifest_path,
                )

        rng = np.random.default_rng(self.seed)
        n = len(slide_ids)
        n_val = max(1, int(n * self.val_frac))

        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_ids = [slide_ids[i] for i in train_idx]
        val_ids = [slide_ids[i] for i in val_idx]

        logger.info(
            "Train/val split: %d train, %d val (val_frac=%.2f, seed=%d)",
            len(train_ids),
            len(val_ids),
            self.val_frac,
            self.seed,
        )
        return train_ids, val_ids

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return  # already set up

        slide_ids = self._load_slide_ids()
        train_ids, val_ids = self._split_slide_ids(slide_ids)

        augmentor = build_augmentor(self.augmentation_cfg, coords_aware=self.coords_aware)

        self.train_dataset = PretrainDataset(
            mmap_dir=self.mmap_dir,
            slide_ids=train_ids,
            augmentor=augmentor,
            dataset_max_instances=self.dataset_max_instances,
            pre_cap_mode=self.dataset_pre_cap_mode,
            force_float32=self.force_float32,
        )

        self.val_dataset = PretrainDataset(
            mmap_dir=self.mmap_dir,
            slide_ids=val_ids,
            augmentor=augmentor,
            dataset_max_instances=self.dataset_max_instances,
            pre_cap_mode=self.dataset_pre_cap_mode,
            force_float32=self.force_float32,
        )

        for name, ds in [("train", self.train_dataset), ("val", self.val_dataset)]:
            sizes = ds.get_bag_sizes()
            if len(sizes) > 0:
                logger.info(
                    "Bag sizes — %s: mean=%.0f, median=%.0f, max=%d",
                    name,
                    sizes.mean(),
                    np.median(sizes),
                    sizes.max(),
                )

    # ── Batching ───────────────────────────────────────────────────────────

    def _build_batching_config(self) -> BatchingConfig:
        """Build BatchingConfig from strategy name + override dict."""
        params = {
            "strategy": self.batching_strategy,
            "batch_size": self.batch_size,
            "max_instances": self.max_instances,
            "seed": self.seed,
        }
        params.update(self.batching_cfg)
        return BatchingConfig(**params)

    # ── DataLoader kwarg helpers ──────────────────────────────────────────

    def _resolve_pin_memory(self) -> bool:
        if self.pin_memory is None:
            return _cuda_pin_memory_available()
        return bool(self.pin_memory)

    def _resolve_persistent_workers(self) -> bool:
        if self.persistent_workers is None:
            return self.num_workers > 0
        return bool(self.persistent_workers) and self.num_workers > 0

    def _loader_kwargs(self) -> dict:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self._resolve_pin_memory(),
            "persistent_workers": self._resolve_persistent_workers(),
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        config = self._build_batching_config()
        batch_sampler = build_batch_sampler(
            config,
            self.train_dataset.get_bag_sizes(),
        )
        collate_fn = build_collator(config)
        loader_kwargs = self._loader_kwargs()

        if batch_sampler is not None:
            logger.info(
                "Using batching strategy '%s' with custom sampler (%s)",
                config.strategy,
                type(batch_sampler).__name__,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **loader_kwargs,
            )

        # Default shuffle strategies (pad_to_max_in_batch, pad_to_global, subsample_fixed_n, regional_crops)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            **loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=DualViewCollator(max_instances=self.max_instances),
            **self._loader_kwargs(),
        )
