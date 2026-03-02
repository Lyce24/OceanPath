"""
Pluggable batching strategies for SSL pretraining.

Nine named strategies control how variable-length slide bags are grouped
into batches and how padding/subsampling is applied:

  1. Pad-to-max-in-batch
  2. Pad-to-global-16k
  3. Token budget (dynamic B)
  4. Bucket Batching (Fixed Batch Size per Bucket)
  5. subsample-fixed-n (Fixed Batch Size, Fixed Sequence Length)
  6. Fixed-Size Regional Crops (TITAN-style)
  7. Sequence Packing (with Block-Diagonal Attention)
  8. Multi-Crop (DINOv2/iBOT: 2 global + N local views)
  9. JEPA (context/target masking partition)

All strategies produce the same base output dict format so the SSL module
requires zero changes.  Sequence packing adds optional ``segment_ids``
keys; multi-crop adds ``global_views``/``local_views`` lists; JEPA adds
``context_pos``/``target_pos`` indices.

Coordinate padding uses sentinel value -1 (``COORD_PAD_VALUE``) so that
padded positions are distinguishable from real grid location (0, 0).
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Sampler

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""


logger = logging.getLogger(__name__)

# Sentinel value for padded coordinate positions.  Real grid coordinates
# are non-negative integers, so -1 is never a valid position.
COORD_PAD_VALUE: int = -1


# ═════════════════════════════════════════════════════════════════════════════
# Strategy Enum & Config
# ═════════════════════════════════════════════════════════════════════════════


class BatchingStrategy(StrEnum):
    """Named batching strategies for SSL pretraining."""

    PAD_TO_MAX_IN_BATCH = "pad_to_max_in_batch"
    PAD_TO_GLOBAL = "pad_to_global"
    TOKEN_BUDGET = "token_budget"
    BUCKET_BATCHING = "bucket_batching"
    SUBSAMPLE_FIXED_N = "subsample_fixed_n"
    REGIONAL_CROPS = "regional_crops"
    SEQUENCE_PACKING = "sequence_packing"
    MULTI_CROP = "multi_crop"
    JEPA = "jepa"


@dataclass
class BatchingConfig:
    """Configuration for a batching strategy.

    Parameters
    ----------
    strategy : str
        One of the BatchingStrategy values.
    n_buckets : int
        Number of buckets for BucketBatchSampler (bucket_batching).
    token_budget : int
        Max total tokens per batch (token_budget, sequence_packing).
    max_batch_size : int
        Hard cap on samples per batch (token budget strategies).
    min_batch_size : int
        Minimum samples to emit a batch (remainder dropped if below).
    fixed_n : int
        Target sequence length for FixedNCollator / RegionalCropCollator.
    global_max : int
        Fixed padded length for pad_to_global strategy.
    crop_frac : float
        Fraction of spatial extent for each regional crop (0, 1].
    sort_within_batch : bool
        Sort samples within each batch by descending length.
    seed : int
        Base random seed for reproducibility.
    batch_size : int
        Nominal batch size (used by bucket sampler, pad-to-max, etc.).
    max_instances : int
        Hard ceiling on padded sequence length per view (collator cap).
    drop_last : bool
        Whether to drop the last incomplete batch.
    global_crop_n : int
        Patch count per global crop (multi_crop strategy).
    local_crop_n : int
        Patch count per local crop (multi_crop strategy).
    n_local_crops : int
        Number of local crops to generate (multi_crop strategy).
    context_ratio : float
        Fraction of patches used as context (jepa strategy), in (0, 1).
    max_context : int or None
        Optional cap on context length (jepa strategy).
    max_target : int or None
        Optional cap on target length (jepa strategy).
    """

    strategy: str = "pad_to_max_in_batch"
    n_buckets: int = 10
    token_budget: int = 32768
    max_batch_size: int = 64
    min_batch_size: int = 2
    fixed_n: int = 512
    global_max: int = 16384
    crop_frac: float = 0.5
    sort_within_batch: bool = True
    seed: int = 42
    batch_size: int = 32
    max_instances: int | None = 4096
    drop_last: bool = True
    global_crop_n: int = 512
    local_crop_n: int = 128
    n_local_crops: int = 6
    context_ratio: float = 0.5
    max_context: int | None = None
    max_target: int | None = None

    def __post_init__(self) -> None:
        valid = {s.value for s in BatchingStrategy}
        if self.strategy not in valid:
            raise ValueError(
                f"Unknown batching strategy '{self.strategy}'. Valid strategies: {sorted(valid)}"
            )
        if self.fixed_n < 1:
            raise ValueError(f"fixed_n must be >= 1, got {self.fixed_n}")
        if self.token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {self.token_budget}")
        if self.min_batch_size < 1:
            raise ValueError(f"min_batch_size must be >= 1, got {self.min_batch_size}")
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.global_max < 1:
            raise ValueError(f"global_max must be >= 1, got {self.global_max}")
        if not (0.0 < self.crop_frac <= 1.0):
            raise ValueError(f"crop_frac must be in (0, 1], got {self.crop_frac}")
        if self.global_crop_n < 1:
            raise ValueError(f"global_crop_n must be >= 1, got {self.global_crop_n}")
        if self.local_crop_n < 1:
            raise ValueError(f"local_crop_n must be >= 1, got {self.local_crop_n}")
        if self.n_local_crops < 1:
            raise ValueError(f"n_local_crops must be >= 1, got {self.n_local_crops}")
        if not (0.0 < self.context_ratio < 1.0):
            raise ValueError(f"context_ratio must be in (0, 1), got {self.context_ratio}")


# ═════════════════════════════════════════════════════════════════════════════
# Batch Samplers
# ═════════════════════════════════════════════════════════════════════════════


class BucketBatchSampler(Sampler[list[int]]):
    """Sort indices by bag size, divide into buckets, yield batches from buckets.

    Algorithm:
        1. Sort dataset indices by bag_size.
        2. Divide sorted indices into ``n_buckets`` roughly-equal groups.
        3. Shuffle within each bucket.
        4. Yield consecutive ``batch_size`` chunks from each bucket.
        5. Iterate buckets in random order each epoch.

    Parameters
    ----------
    bag_sizes : array-like
        Per-sample bag sizes (length N).
    batch_size : int
        Number of samples per batch.
    n_buckets : int
        Number of length-similarity buckets.
    drop_last : bool
        If True, drop the last incomplete batch per bucket.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bag_sizes: np.ndarray,
        batch_size: int,
        n_buckets: int = 10,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.bag_sizes = np.asarray(bag_sizes)
        self.batch_size = batch_size
        self.n_buckets = max(1, min(n_buckets, len(self.bag_sizes)))
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        sorted_indices = np.argsort(self.bag_sizes)
        buckets = np.array_split(sorted_indices, self.n_buckets)

        for bucket in buckets:
            rng.shuffle(bucket)

        bucket_order = rng.permutation(len(buckets))

        for bi in bucket_order:
            bucket = buckets[bi]
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size].tolist()
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        n = len(self.bag_sizes)
        if n == 0:
            return 0
        base_size = n // self.n_buckets
        remainder = n % self.n_buckets
        total = 0
        for i in range(self.n_buckets):
            bucket_size = base_size + (1 if i < remainder else 0)
            if self.drop_last:
                total += bucket_size // self.batch_size
            else:
                total += math.ceil(bucket_size / self.batch_size)
        return total


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Variable batch size to fit a fixed token budget.

    Iterates through (optionally shuffled) indices and accumulates samples
    until the total token count would exceed ``token_budget`` or
    ``max_batch_size`` is reached. Uses raw ``bag_sizes`` as token-count
    proxy (pre-augmentation).

    After greedy accumulation the **batch list is shuffled** so that
    correlated batch-size runs (large-slide spillover) don't produce
    systematic patterns that harm methods sensitive to batch statistics
    (VICReg, SimCLR).

    A single sample exceeding ``token_budget`` is yielded as a solo batch
    (never skipped).

    Parameters
    ----------
    bag_sizes : array-like
        Per-sample bag sizes (length N).
    token_budget : int
        Maximum total tokens per batch.
    max_batch_size : int
        Hard cap on number of samples per batch.
    min_batch_size : int
        Minimum samples to emit a remainder batch (dropped if below).
    sort_within_batch : bool
        Sort samples within each batch by descending bag size.
    sort : bool
        If True, iterate indices sorted by bag size (ascending) instead of
        shuffled. Useful for packing strategies.
    drop_last : bool
        If True, drop remainder batch below min_batch_size.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bag_sizes: np.ndarray,
        token_budget: int = 32768,
        max_batch_size: int = 64,
        min_batch_size: int = 2,
        sort_within_batch: bool = True,
        sort: bool = False,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.bag_sizes = np.asarray(bag_sizes)
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.sort_within_batch = sort_within_batch
        self.sort = sort
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0
        self._cached_len: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._cached_len = None

    def _build_batches(self, rng: np.random.Generator) -> list[list[int]]:
        if self.sort:
            indices = np.argsort(self.bag_sizes).tolist()
        else:
            indices = rng.permutation(len(self.bag_sizes)).tolist()

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_tokens = 0

        for idx in indices:
            size = int(self.bag_sizes[idx])

            if current_batch and (
                current_tokens + size > self.token_budget
                or len(current_batch) >= self.max_batch_size
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(idx)
            current_tokens += size

        # Handle remainder
        if current_batch and (not self.drop_last or len(current_batch) >= self.min_batch_size):
            batches.append(current_batch)

        if self.sort_within_batch:
            for i, batch in enumerate(batches):
                batches[i] = sorted(batch, key=lambda x: self.bag_sizes[x], reverse=True)

        return batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        batches = self._build_batches(rng)

        # ── Issue 1 fix: shuffle batch ORDER ───────────────────────────
        # Greedy accumulation creates runs of correlated batch sizes
        # (large-slide spillover → consecutive small-B batches).
        # Shuffling breaks these runs.
        if not self.sort:
            rng.shuffle(batches)

        # ── Issue 6 fix: log effective batch size distribution ─────────
        if batches:
            sizes = [len(b) for b in batches]
            logger.info(
                "TokenBudget epoch %d: %d batches, B mean=%.1f min=%d max=%d std=%.1f",
                self._epoch,
                len(sizes),
                np.mean(sizes),
                min(sizes),
                max(sizes),
                np.std(sizes),
            )

        yield from batches

    def __len__(self) -> int:
        if self._cached_len is None:
            rng = np.random.default_rng(self.seed + self._epoch)
            self._cached_len = len(self._build_batches(rng))
        return self._cached_len


# ═════════════════════════════════════════════════════════════════════════════
# Coordinate padding helper
# ═════════════════════════════════════════════════════════════════════════════


def _make_coord_buffer(*shape: int, coord_dim: int = 2) -> torch.Tensor:
    """Allocate a coordinate tensor filled with the padding sentinel."""
    return torch.full((*shape, coord_dim), COORD_PAD_VALUE, dtype=torch.int32)


# ═════════════════════════════════════════════════════════════════════════════
# Collators
# ═════════════════════════════════════════════════════════════════════════════


class DualViewCollator:
    """Collate dual-view samples into padded batches (pad-to-max-in-batch).

    Each view is padded independently to its own batch-max length,
    capped at ``max_instances``.  Produces the ``{view1, mask1, view2, mask2}``
    dict that ``SSLPretrainModule`` expects.

    Coordinate padding uses ``COORD_PAD_VALUE`` (-1) so that padded
    positions are distinguishable from real grid location (0, 0).

    Parameters
    ----------
    max_instances : int or None
        Hard ceiling on sequence length per view (prevents OOM).
    """

    def __init__(self, max_instances: int | None = 4096):
        self.max_instances = max_instances

    def _cap(self, n: int) -> int:
        return min(n, self.max_instances) if self.max_instances is not None else n

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        feat_dtype = batch[0]["view1_features"].dtype
        d = batch[0]["view1_features"].shape[-1]

        lengths1 = [self._cap(s["view1_features"].shape[0]) for s in batch]
        lengths2 = [self._cap(s["view2_features"].shape[0]) for s in batch]

        max_n1 = max(lengths1)
        max_n2 = max(lengths2)

        view1 = torch.zeros(b, max_n1, d, dtype=feat_dtype)
        mask1 = torch.zeros(b, max_n1, dtype=torch.float32)
        view2 = torch.zeros(b, max_n2, d, dtype=feat_dtype)
        mask2 = torch.zeros(b, max_n2, dtype=torch.float32)

        slide_ids = []
        has_coords = "view1_coords" in batch[0]
        coords1_list: list | None = [] if has_coords else None
        coords2_list: list | None = [] if has_coords else None

        for i, sample in enumerate(batch):
            n1 = lengths1[i]
            n2 = lengths2[i]

            view1[i, :n1] = sample["view1_features"][:n1]
            mask1[i, :n1] = 1.0
            view2[i, :n2] = sample["view2_features"][:n2]
            mask2[i, :n2] = 1.0

            slide_ids.append(sample["slide_id"])

            if has_coords:
                coords1_list.append(sample.get("view1_coords"))
                coords2_list.append(sample.get("view2_coords"))

        result = {
            "view1": view1,
            "mask1": mask1,
            "view2": view2,
            "mask2": mask2,
            "lengths1": torch.tensor(lengths1, dtype=torch.int32),
            "lengths2": torch.tensor(lengths2, dtype=torch.int32),
            "slide_ids": slide_ids,
        }

        if has_coords and coords1_list and coords1_list[0] is not None:
            coord_dim = coords1_list[0].shape[-1]
            c1 = _make_coord_buffer(b, max_n1, coord_dim=coord_dim)
            c2 = _make_coord_buffer(b, max_n2, coord_dim=coord_dim)
            for i in range(b):
                n1 = lengths1[i]
                n2 = lengths2[i]
                if coords1_list[i] is not None:
                    c1[i, :n1] = coords1_list[i][:n1]
                if coords2_list[i] is not None:
                    c2[i, :n2] = coords2_list[i][:n2]
            result["coords1"] = c1
            result["coords2"] = c2

        return result


class PadToGlobalCollator:
    """Pad every view to a fixed global maximum length.

    Unlike ``DualViewCollator`` which pads to the batch-max, this collator
    pads ALL views to ``global_max``.  This produces perfectly rectangular
    tensors of a known size, simplifying compiled/fused kernels at the cost
    of higher padding overhead.

    Parameters
    ----------
    global_max : int
        Fixed padded length for every view (e.g. 16384).
    max_instances : int or None
        Hard ceiling applied before padding (prevents OOM on huge slides).
    """

    def __init__(self, global_max: int = 16384, max_instances: int | None = None):
        self.global_max = global_max
        self.max_instances = max_instances

    def _cap(self, n: int) -> int:
        n = min(n, self.global_max)
        if self.max_instances is not None:
            n = min(n, self.max_instances)
        return n

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        feat_dtype = batch[0]["view1_features"].dtype
        d = batch[0]["view1_features"].shape[-1]
        g = self.global_max

        view1 = torch.zeros(b, g, d, dtype=feat_dtype)
        mask1 = torch.zeros(b, g, dtype=torch.float32)
        view2 = torch.zeros(b, g, d, dtype=feat_dtype)
        mask2 = torch.zeros(b, g, dtype=torch.float32)

        has_coords = "view1_coords" in batch[0]
        coord_dim = batch[0]["view1_coords"].shape[-1] if has_coords else 0
        coords1 = _make_coord_buffer(b, g, coord_dim=coord_dim) if has_coords else None
        coords2 = _make_coord_buffer(b, g, coord_dim=coord_dim) if has_coords else None

        slide_ids = []
        lengths1 = []
        lengths2 = []

        for i, sample in enumerate(batch):
            n1 = self._cap(sample["view1_features"].shape[0])
            n2 = self._cap(sample["view2_features"].shape[0])

            view1[i, :n1] = sample["view1_features"][:n1]
            mask1[i, :n1] = 1.0
            view2[i, :n2] = sample["view2_features"][:n2]
            mask2[i, :n2] = 1.0

            lengths1.append(n1)
            lengths2.append(n2)
            slide_ids.append(sample["slide_id"])

            if has_coords:
                c1 = sample.get("view1_coords")
                c2 = sample.get("view2_coords")
                if c1 is not None:
                    coords1[i, :n1] = c1[:n1]
                if c2 is not None:
                    coords2[i, :n2] = c2[:n2]

        result = {
            "view1": view1,
            "mask1": mask1,
            "view2": view2,
            "mask2": mask2,
            "lengths1": torch.tensor(lengths1, dtype=torch.int32),
            "lengths2": torch.tensor(lengths2, dtype=torch.int32),
            "slide_ids": slide_ids,
        }

        if has_coords and coords1 is not None:
            result["coords1"] = coords1
            result["coords2"] = coords2

        return result


class FixedNCollator:
    """Subsample/pad every view to exactly N patches.

    For each view independently:
      - If view has more than ``fixed_n`` patches, randomly subsample
        (using a random permutation via an optional ``torch.Generator``
        for reproducibility).
      - If view has fewer, zero-pad to ``fixed_n``.

    Produces perfectly rectangular ``[B, fixed_n, D]`` tensors.

    Parameters
    ----------
    fixed_n : int
        Target number of patches per view.
    max_instances : int or None
        Hard ceiling (applied before fixed_n logic). Views are first
        truncated to max_instances, then subsampled/padded to fixed_n.
    generator : torch.Generator or None
        Optional RNG for reproducible subsampling.  When ``None``,
        ``torch.randperm`` uses the global PyTorch RNG state.
    """

    def __init__(
        self,
        fixed_n: int = 512,
        max_instances: int | None = 4096,
        generator: torch.Generator | None = None,
    ):
        self.fixed_n = fixed_n
        self.max_instances = max_instances
        self.generator = generator

    def _cap(self, n: int) -> int:
        return min(n, self.max_instances) if self.max_instances is not None else n

    def _fill_view(
        self,
        sample: dict,
        view_key: str,
        i: int,
        view_out: torch.Tensor,
        mask_out: torch.Tensor,
        coords_out: torch.Tensor | None,
    ) -> int:
        """Fill one view for one sample. Returns the effective length."""
        n = self.fixed_n
        feats = sample[view_key]
        n_capped = self._cap(feats.shape[0])

        coord_key = view_key.replace("_features", "_coords")
        sample_coords = sample.get(coord_key) if coords_out is not None else None

        if n_capped > n:
            perm = torch.randperm(n_capped, generator=self.generator)[:n]
            view_out[i] = feats[perm]
            mask_out[i, :n] = 1.0
            if coords_out is not None and sample_coords is not None:
                coords_out[i] = sample_coords[perm]
            return n

        view_out[i, :n_capped] = feats[:n_capped]
        mask_out[i, :n_capped] = 1.0
        if coords_out is not None and sample_coords is not None:
            coords_out[i, :n_capped] = sample_coords[:n_capped]
        return n_capped

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        feat_dtype = batch[0]["view1_features"].dtype
        d = batch[0]["view1_features"].shape[-1]
        n = self.fixed_n

        view1 = torch.zeros(b, n, d, dtype=feat_dtype)
        mask1 = torch.zeros(b, n, dtype=torch.float32)
        view2 = torch.zeros(b, n, d, dtype=feat_dtype)
        mask2 = torch.zeros(b, n, dtype=torch.float32)

        has_coords = "view1_coords" in batch[0]
        coord_dim = batch[0]["view1_coords"].shape[-1] if has_coords else 0
        coords1 = _make_coord_buffer(b, n, coord_dim=coord_dim) if has_coords else None
        coords2 = _make_coord_buffer(b, n, coord_dim=coord_dim) if has_coords else None

        slide_ids = []
        lengths1 = []
        lengths2 = []

        for i, sample in enumerate(batch):
            slide_ids.append(sample["slide_id"])
            lengths1.append(self._fill_view(sample, "view1_features", i, view1, mask1, coords1))
            lengths2.append(self._fill_view(sample, "view2_features", i, view2, mask2, coords2))

        result = {
            "view1": view1,
            "mask1": mask1,
            "view2": view2,
            "mask2": mask2,
            "lengths1": torch.tensor(lengths1, dtype=torch.int32),
            "lengths2": torch.tensor(lengths2, dtype=torch.int32),
            "slide_ids": slide_ids,
        }

        if has_coords and coords1 is not None:
            result["coords1"] = coords1
            result["coords2"] = coords2

        return result


class RegionalCropCollator:
    """TITAN-style fixed-size regional crops.

    For each view, selects a random spatial region (defined by coordinates),
    keeps only patches within that region, then subsamples/pads to
    ``fixed_n``.  This produces fixed-size ``[B, fixed_n, D]`` tensors
    with spatial coherence (nearby patches grouped together).

    Falls back to random subsampling when coordinates are unavailable.

    Parameters
    ----------
    fixed_n : int
        Target number of patches per crop.
    crop_frac : float
        Fraction of spatial extent to crop in each dimension (0, 1].
        Smaller values = tighter crops with stronger locality.
    max_instances : int or None
        Hard ceiling before cropping.
    """

    def __init__(
        self,
        fixed_n: int = 512,
        crop_frac: float = 0.5,
        max_instances: int | None = 4096,
    ):
        self.fixed_n = fixed_n
        self.crop_frac = crop_frac
        self.max_instances = max_instances

    def _cap(self, n: int) -> int:
        return min(n, self.max_instances) if self.max_instances is not None else n

    def _crop_view(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor | None,
        fixed_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Crop a spatial region, subsample/pad to fixed_n.

        Returns (cropped_feats [fixed_n, D], cropped_coords [fixed_n, C] or None, n_real).
        """
        n_raw = feats.shape[0]
        n_capped = self._cap(n_raw)
        feats = feats[:n_capped]
        if coords is not None:
            coords = coords[:n_capped]

        d = feats.shape[-1]

        # If coords available, do spatial crop
        if coords is not None and len(coords) > 0:
            feats, coords, n_in_region = self._spatial_crop(feats, coords)
        else:
            n_in_region = len(feats)

        # Subsample or pad to fixed_n
        out_feats = torch.zeros(fixed_n, d, dtype=feats.dtype)
        out_coords = None
        if coords is not None:
            out_coords = torch.full(
                (fixed_n, coords.shape[-1]), COORD_PAD_VALUE, dtype=coords.dtype
            )

        if n_in_region >= fixed_n:
            perm = torch.randperm(n_in_region)[:fixed_n]
            out_feats[:] = feats[perm]
            if out_coords is not None:
                out_coords[:] = coords[perm]
            return out_feats, out_coords, fixed_n

        # Pad
        actual = min(n_in_region, fixed_n)
        out_feats[:actual] = feats[:actual]
        if out_coords is not None and coords is not None:
            out_coords[:actual] = coords[:actual]
        return out_feats, out_coords, actual

    def _spatial_crop(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Select patches within a random spatial region."""
        n = len(feats)
        if n == 0:
            return feats, coords, 0

        x_min = coords[:, 0].min().item()
        x_max = coords[:, 0].max().item()
        y_min = coords[:, 1].min().item()
        y_max = coords[:, 1].max().item()

        x_span = x_max - x_min + 1
        y_span = y_max - y_min + 1
        crop_w = max(1, int(x_span * self.crop_frac))
        crop_h = max(1, int(y_span * self.crop_frac))

        # Random crop origin
        x_start = x_min + torch.randint(0, max(1, x_span - crop_w + 1), (1,)).item()
        y_start = y_min + torch.randint(0, max(1, y_span - crop_h + 1), (1,)).item()

        in_region = (
            (coords[:, 0] >= x_start)
            & (coords[:, 0] < x_start + crop_w)
            & (coords[:, 1] >= y_start)
            & (coords[:, 1] < y_start + crop_h)
        )

        region_feats = feats[in_region]
        region_coords = coords[in_region]
        n_in = len(region_feats)

        # If region is empty, fall back to full set
        if n_in == 0:
            return feats, coords, n

        return region_feats, region_coords, n_in

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        feat_dtype = batch[0]["view1_features"].dtype
        d = batch[0]["view1_features"].shape[-1]
        n = self.fixed_n

        view1 = torch.zeros(b, n, d, dtype=feat_dtype)
        mask1 = torch.zeros(b, n, dtype=torch.float32)
        view2 = torch.zeros(b, n, d, dtype=feat_dtype)
        mask2 = torch.zeros(b, n, dtype=torch.float32)

        has_coords = "view1_coords" in batch[0]
        coord_dim = batch[0]["view1_coords"].shape[-1] if has_coords else 0
        coords1 = _make_coord_buffer(b, n, coord_dim=coord_dim) if has_coords else None
        coords2 = _make_coord_buffer(b, n, coord_dim=coord_dim) if has_coords else None

        slide_ids = []
        lengths1 = []
        lengths2 = []

        for i, sample in enumerate(batch):
            slide_ids.append(sample["slide_id"])

            c1 = sample.get("view1_coords") if has_coords else None
            c2 = sample.get("view2_coords") if has_coords else None

            f1, cr1, n1 = self._crop_view(sample["view1_features"], c1, n)
            f2, cr2, n2 = self._crop_view(sample["view2_features"], c2, n)

            view1[i] = f1
            mask1[i, :n1] = 1.0
            view2[i] = f2
            mask2[i, :n2] = 1.0
            lengths1.append(n1)
            lengths2.append(n2)

            if coords1 is not None and cr1 is not None:
                coords1[i] = cr1
            if coords2 is not None and cr2 is not None:
                coords2[i] = cr2

        result = {
            "view1": view1,
            "mask1": mask1,
            "view2": view2,
            "mask2": mask2,
            "lengths1": torch.tensor(lengths1, dtype=torch.int32),
            "lengths2": torch.tensor(lengths2, dtype=torch.int32),
            "slide_ids": slide_ids,
        }

        if has_coords and coords1 is not None:
            result["coords1"] = coords1
            result["coords2"] = coords2

        return result


class SequencePackingCollator:
    """Pack multiple samples into concatenated sequences.

    Concatenates all samples in a batch along the sequence dimension,
    producing a single packed sequence per view.  Adds ``segment_ids``
    tensors so models can apply block-diagonal attention (tokens from
    different slides never attend to each other).

    The upstream ``TokenBudgetBatchSampler`` (with ``sort=True``) ensures
    that samples assigned to a batch fit within the token budget.

    Output format
    -------------
    Standard keys (``view1``, ``mask1``, etc.) have batch dimension B=1.
    Additional keys:

    - ``segment_ids1`` / ``segment_ids2``: ``[1, L]`` int tensor mapping
      each token to its original sample index (0-based, -1 for padding).
    - ``n_segments``: number of original samples packed together.

    Parameters
    ----------
    max_instances : int or None
        Hard ceiling on per-sample sequence length before packing.
    """

    def __init__(self, max_instances: int | None = 4096):
        self.max_instances = max_instances

    def _cap(self, n: int) -> int:
        return min(n, self.max_instances) if self.max_instances is not None else n

    def __call__(self, batch: list[dict]) -> dict:
        has_coords = "view1_coords" in batch[0]
        coord_dim = batch[0]["view1_coords"].shape[-1] if has_coords else 0

        # Collect per-sample tensors
        feats1_list = []
        feats2_list = []
        coords1_list = []
        coords2_list = []
        seg1_list = []
        seg2_list = []
        slide_ids = []
        lengths1 = []
        lengths2 = []

        for seg_idx, sample in enumerate(batch):
            n1 = self._cap(sample["view1_features"].shape[0])
            n2 = self._cap(sample["view2_features"].shape[0])

            feats1_list.append(sample["view1_features"][:n1])
            feats2_list.append(sample["view2_features"][:n2])
            seg1_list.append(torch.full((n1,), seg_idx, dtype=torch.int32))
            seg2_list.append(torch.full((n2,), seg_idx, dtype=torch.int32))
            lengths1.append(n1)
            lengths2.append(n2)
            slide_ids.append(sample["slide_id"])

            if has_coords:
                c1 = sample.get("view1_coords")
                c2 = sample.get("view2_coords")
                coords1_list.append(
                    c1[:n1]
                    if c1 is not None
                    else torch.full((n1, coord_dim), COORD_PAD_VALUE, dtype=torch.int32)
                )
                coords2_list.append(
                    c2[:n2]
                    if c2 is not None
                    else torch.full((n2, coord_dim), COORD_PAD_VALUE, dtype=torch.int32)
                )

        # Concatenate into packed sequences
        packed1 = torch.cat(feats1_list, dim=0)  # [total_L1, D]
        packed2 = torch.cat(feats2_list, dim=0)  # [total_L2, D]
        seg_ids1 = torch.cat(seg1_list, dim=0)  # [total_L1]
        seg_ids2 = torch.cat(seg2_list, dim=0)  # [total_L2]

        total_l1 = packed1.shape[0]
        total_l2 = packed2.shape[0]

        # Wrap in batch dim (B=1)
        result = {
            "view1": packed1.unsqueeze(0),  # [1, L1, D]
            "mask1": torch.ones(1, total_l1, dtype=torch.float32),
            "view2": packed2.unsqueeze(0),  # [1, L2, D]
            "mask2": torch.ones(1, total_l2, dtype=torch.float32),
            "lengths1": torch.tensor(lengths1, dtype=torch.int32),
            "lengths2": torch.tensor(lengths2, dtype=torch.int32),
            "slide_ids": slide_ids,
            "segment_ids1": seg_ids1.unsqueeze(0),  # [1, L1]
            "segment_ids2": seg_ids2.unsqueeze(0),  # [1, L2]
            "n_segments": len(batch),
        }

        if has_coords and coords1_list:
            packed_c1 = torch.cat(coords1_list, dim=0)
            packed_c2 = torch.cat(coords2_list, dim=0)
            result["coords1"] = packed_c1.unsqueeze(0)  # [1, L1, C]
            result["coords2"] = packed_c2.unsqueeze(0)  # [1, L2, C]

        return result


class MultiCropCollator:
    """Multi-crop collator for DINOv2/iBOT-style training.

    Produces 2 global views (subsampled/padded to ``global_crop_n``) and
    ``n_local_crops`` local views (subsampled to ``local_crop_n``).  Local
    crops are created by further subsampling from the input views.

    Output format
    -------------
    Backward-compatible keys (``view1``/``mask1``/``view2``/``mask2``) map
    to the two global views.  Additional keys:

    - ``global_views``:  list of 2 tensors ``[B, global_crop_n, D]``
    - ``global_masks``:  list of 2 tensors ``[B, global_crop_n]``
    - ``local_views``:   list of ``n_local_crops`` tensors ``[B, local_crop_n, D]``
    - ``local_masks``:   list of ``n_local_crops`` tensors ``[B, local_crop_n]``

    Parameters
    ----------
    global_crop_n : int
        Patch count per global crop.
    local_crop_n : int
        Patch count per local crop.
    n_local_crops : int
        Number of local crops to generate.
    max_instances : int or None
        Hard ceiling on input sequence length before cropping.
    generator : torch.Generator or None
        Optional RNG for reproducible subsampling.
    """

    def __init__(
        self,
        global_crop_n: int = 512,
        local_crop_n: int = 128,
        n_local_crops: int = 6,
        max_instances: int | None = 4096,
        generator: torch.Generator | None = None,
    ):
        self.global_crop_n = global_crop_n
        self.local_crop_n = local_crop_n
        self.n_local_crops = n_local_crops
        self.max_instances = max_instances
        self.generator = generator

    def _cap(self, n: int) -> int:
        return min(n, self.max_instances) if self.max_instances is not None else n

    def _subsample_pad(
        self,
        feats: torch.Tensor,
        target_n: int,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        """Subsample or pad *feats* to *target_n*.

        Returns (out_feats, out_mask, out_coords_or_None, n_real).
        """
        d = feats.shape[-1]
        n = feats.shape[0]
        n_capped = self._cap(n)
        feats = feats[:n_capped]
        if coords is not None:
            coords = coords[:n_capped]
        n = n_capped

        out_feats = torch.zeros(target_n, d, dtype=feats.dtype)
        out_mask = torch.zeros(target_n, dtype=torch.float32)
        out_coords: torch.Tensor | None = None
        if coords is not None:
            out_coords = torch.full(
                (target_n, coords.shape[-1]), COORD_PAD_VALUE, dtype=coords.dtype
            )

        if n >= target_n:
            perm = torch.randperm(n, generator=self.generator)[:target_n]
            out_feats[:] = feats[perm]
            out_mask[:] = 1.0
            if out_coords is not None and coords is not None:
                out_coords[:] = coords[perm]
            return out_feats, out_mask, out_coords, target_n

        out_feats[:n] = feats
        out_mask[:n] = 1.0
        if out_coords is not None and coords is not None:
            out_coords[:n] = coords[:n]
        return out_feats, out_mask, out_coords, n

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        feat_dtype = batch[0]["view1_features"].dtype
        d = batch[0]["view1_features"].shape[-1]
        has_coords = "view1_coords" in batch[0]
        coord_dim = batch[0]["view1_coords"].shape[-1] if has_coords else 0

        gn = self.global_crop_n
        ln = self.local_crop_n

        # Pre-allocate global views
        gv1 = torch.zeros(b, gn, d, dtype=feat_dtype)
        gm1 = torch.zeros(b, gn, dtype=torch.float32)
        gv2 = torch.zeros(b, gn, d, dtype=feat_dtype)
        gm2 = torch.zeros(b, gn, dtype=torch.float32)
        gc1 = _make_coord_buffer(b, gn, coord_dim=coord_dim) if has_coords else None
        gc2 = _make_coord_buffer(b, gn, coord_dim=coord_dim) if has_coords else None

        # Pre-allocate local views
        local_feats = [torch.zeros(b, ln, d, dtype=feat_dtype) for _ in range(self.n_local_crops)]
        local_masks = [torch.zeros(b, ln, dtype=torch.float32) for _ in range(self.n_local_crops)]
        local_coords = (
            [_make_coord_buffer(b, ln, coord_dim=coord_dim) for _ in range(self.n_local_crops)]
            if has_coords
            else None
        )

        slide_ids = []
        lengths1 = []
        lengths2 = []

        for i, sample in enumerate(batch):
            slide_ids.append(sample["slide_id"])

            v1_feats = sample["view1_features"]
            v2_feats = sample["view2_features"]
            v1_coords = sample.get("view1_coords") if has_coords else None
            v2_coords = sample.get("view2_coords") if has_coords else None

            # Global crops from the two augmented views
            f1, m1, c1, n1 = self._subsample_pad(v1_feats, gn, v1_coords)
            f2, m2, c2, n2 = self._subsample_pad(v2_feats, gn, v2_coords)

            gv1[i] = f1
            gm1[i] = m1
            gv2[i] = f2
            gm2[i] = m2
            lengths1.append(n1)
            lengths2.append(n2)
            if gc1 is not None and c1 is not None:
                gc1[i] = c1
            if gc2 is not None and c2 is not None:
                gc2[i] = c2

            # Local crops: alternate subsampling from view1 and view2
            for k in range(self.n_local_crops):
                src_feats = v1_feats if k % 2 == 0 else v2_feats
                src_coords = v1_coords if k % 2 == 0 else v2_coords
                lf, lm, lc, _ = self._subsample_pad(src_feats, ln, src_coords)
                local_feats[k][i] = lf
                local_masks[k][i] = lm
                if local_coords is not None and lc is not None:
                    local_coords[k][i] = lc

        result = {
            # Backward-compatible dual-view keys (global views)
            "view1": gv1,
            "mask1": gm1,
            "view2": gv2,
            "mask2": gm2,
            "lengths1": torch.tensor(lengths1, dtype=torch.int32),
            "lengths2": torch.tensor(lengths2, dtype=torch.int32),
            "slide_ids": slide_ids,
            # Multi-crop specific
            "global_views": [gv1, gv2],
            "global_masks": [gm1, gm2],
            "local_views": local_feats,
            "local_masks": local_masks,
        }

        if has_coords and gc1 is not None:
            result["coords1"] = gc1
            result["coords2"] = gc2
            result["global_coords"] = [gc1, gc2]
            if local_coords is not None:
                result["local_coords"] = local_coords

        return result


class JEPACollator:
    """Collator for JEPA-style predictive pretraining.

    Instead of dual independent augmented views, takes a single view
    (``view1_features``) and partitions its patches into a **context** set
    (visible to the encoder) and a **target** set (to be predicted).

    The partition is a random split controlled by ``context_ratio``.

    Output format
    -------------
    Backward-compatible keys are aliased so existing training loops work:

    - ``view1`` / ``mask1`` = context
    - ``view2`` / ``mask2`` = target

    Additional keys:

    - ``context_pos``:  ``[B, N_ctx]``  original position indices
    - ``target_pos``:   ``[B, N_tgt]``  original position indices

    Parameters
    ----------
    context_ratio : float
        Fraction of patches assigned to context, in (0, 1).
    max_instances : int or None
        Hard ceiling on input sequence length before partitioning.
    max_context : int or None
        Optional cap on context length (pad/truncate).
    max_target : int or None
        Optional cap on target length (pad/truncate).
    generator : torch.Generator or None
        Optional RNG for reproducible partitioning.
    """

    def __init__(
        self,
        context_ratio: float = 0.5,
        max_instances: int | None = 4096,
        max_context: int | None = None,
        max_target: int | None = None,
        generator: torch.Generator | None = None,
    ):
        self.context_ratio = context_ratio
        self.max_instances = max_instances
        self.max_context = max_context
        self.max_target = max_target
        self.generator = generator

    def _cap(self, n: int) -> int:
        return min(n, self.max_instances) if self.max_instances is not None else n

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        feat_dtype = batch[0]["view1_features"].dtype
        d = batch[0]["view1_features"].shape[-1]
        has_coords = "view1_coords" in batch[0]
        coord_dim = batch[0]["view1_coords"].shape[-1] if has_coords else 0

        # First pass: determine per-sample context/target sizes and batch maxes
        ctx_lengths = []
        tgt_lengths = []
        for sample in batch:
            n = self._cap(sample["view1_features"].shape[0])
            n_ctx = max(1, int(n * self.context_ratio))
            n_tgt = n - n_ctx
            if self.max_context is not None:
                n_ctx = min(n_ctx, self.max_context)
            if self.max_target is not None:
                n_tgt = min(n_tgt, self.max_target)
            ctx_lengths.append(n_ctx)
            tgt_lengths.append(max(1, n_tgt))

        max_ctx = max(ctx_lengths)
        max_tgt = max(tgt_lengths)

        # Allocate output tensors
        context = torch.zeros(b, max_ctx, d, dtype=feat_dtype)
        ctx_mask = torch.zeros(b, max_ctx, dtype=torch.float32)
        target = torch.zeros(b, max_tgt, d, dtype=feat_dtype)
        tgt_mask = torch.zeros(b, max_tgt, dtype=torch.float32)
        ctx_pos = torch.full((b, max_ctx), -1, dtype=torch.int32)
        tgt_pos = torch.full((b, max_tgt), -1, dtype=torch.int32)
        ctx_coords = _make_coord_buffer(b, max_ctx, coord_dim=coord_dim) if has_coords else None
        tgt_coords = _make_coord_buffer(b, max_tgt, coord_dim=coord_dim) if has_coords else None

        slide_ids = []

        for i, sample in enumerate(batch):
            feats = sample["view1_features"]
            n = self._cap(feats.shape[0])
            feats = feats[:n]
            coords = None
            if has_coords:
                coords = sample.get("view1_coords")
                if coords is not None:
                    coords = coords[:n]

            n_ctx = ctx_lengths[i]
            n_tgt = tgt_lengths[i]

            # Random partition
            perm = torch.randperm(n, generator=self.generator)
            ctx_idx = perm[:n_ctx]
            tgt_idx = perm[n_ctx : n_ctx + n_tgt]

            context[i, :n_ctx] = feats[ctx_idx]
            ctx_mask[i, :n_ctx] = 1.0
            ctx_pos[i, :n_ctx] = ctx_idx.to(torch.int32)

            target[i, :n_tgt] = feats[tgt_idx]
            tgt_mask[i, :n_tgt] = 1.0
            tgt_pos[i, :n_tgt] = tgt_idx.to(torch.int32)

            if has_coords and coords is not None:
                if ctx_coords is not None:
                    ctx_coords[i, :n_ctx] = coords[ctx_idx]
                if tgt_coords is not None:
                    tgt_coords[i, :n_tgt] = coords[tgt_idx]

            slide_ids.append(sample["slide_id"])

        result = {
            # Backward-compatible (context = view1, target = view2)
            "view1": context,
            "mask1": ctx_mask,
            "view2": target,
            "mask2": tgt_mask,
            "lengths1": torch.tensor(ctx_lengths, dtype=torch.int32),
            "lengths2": torch.tensor(tgt_lengths, dtype=torch.int32),
            "slide_ids": slide_ids,
            # JEPA-specific
            "context_pos": ctx_pos,
            "target_pos": tgt_pos,
        }

        if has_coords:
            if ctx_coords is not None:
                result["coords1"] = ctx_coords
            if tgt_coords is not None:
                result["coords2"] = tgt_coords

        return result


# ═════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═════════════════════════════════════════════════════════════════════════════


def build_batch_sampler(
    config: BatchingConfig,
    bag_sizes: np.ndarray,
) -> Sampler | None:
    """Return a custom batch sampler, or None for default DataLoader shuffle.

    Parameters
    ----------
    config : BatchingConfig
        Batching configuration.
    bag_sizes : array-like
        Per-sample bag sizes from the training dataset.

    Returns
    -------
    Sampler | None
        A batch sampler instance, or None if the strategy uses default shuffling.
    """
    strategy = BatchingStrategy(config.strategy)

    if strategy == BatchingStrategy.BUCKET_BATCHING:
        return BucketBatchSampler(
            bag_sizes=bag_sizes,
            batch_size=config.batch_size,
            n_buckets=config.n_buckets,
            drop_last=config.drop_last,
            seed=config.seed,
        )

    if strategy == BatchingStrategy.TOKEN_BUDGET:
        return TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=config.token_budget,
            max_batch_size=config.max_batch_size,
            min_batch_size=config.min_batch_size,
            sort_within_batch=config.sort_within_batch,
            sort=False,
            drop_last=config.drop_last,
            seed=config.seed,
        )

    if strategy == BatchingStrategy.SEQUENCE_PACKING:
        return TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=config.token_budget,
            max_batch_size=config.max_batch_size,
            min_batch_size=config.min_batch_size,
            sort_within_batch=False,
            sort=True,
            drop_last=config.drop_last,
            seed=config.seed,
        )

    # Strategies using default DataLoader shuffle:
    # PAD_TO_MAX_IN_BATCH, PAD_TO_GLOBAL, SUBSAMPLE_FIXED_N,
    # REGIONAL_CROPS, MULTI_CROP, JEPA
    return None


def build_collator(config: BatchingConfig):
    """Return the appropriate collate function for the strategy.

    Parameters
    ----------
    config : BatchingConfig
        Batching configuration.

    Returns
    -------
    callable
        A collator instance.
    """
    strategy = BatchingStrategy(config.strategy)

    if strategy == BatchingStrategy.PAD_TO_GLOBAL:
        return PadToGlobalCollator(
            global_max=config.global_max,
            max_instances=config.max_instances,
        )

    if strategy == BatchingStrategy.SUBSAMPLE_FIXED_N:
        return FixedNCollator(
            fixed_n=config.fixed_n,
            max_instances=config.max_instances,
        )

    if strategy == BatchingStrategy.REGIONAL_CROPS:
        return RegionalCropCollator(
            fixed_n=config.fixed_n,
            crop_frac=config.crop_frac,
            max_instances=config.max_instances,
        )

    if strategy == BatchingStrategy.SEQUENCE_PACKING:
        return SequencePackingCollator(max_instances=config.max_instances)

    if strategy == BatchingStrategy.MULTI_CROP:
        return MultiCropCollator(
            global_crop_n=config.global_crop_n,
            local_crop_n=config.local_crop_n,
            n_local_crops=config.n_local_crops,
            max_instances=config.max_instances,
        )

    if strategy == BatchingStrategy.JEPA:
        return JEPACollator(
            context_ratio=config.context_ratio,
            max_instances=config.max_instances,
            max_context=config.max_context,
            max_target=config.max_target,
        )

    # PAD_TO_MAX_IN_BATCH, TOKEN_BUDGET, BUCKET_BATCHING
    return DualViewCollator(max_instances=config.max_instances)
