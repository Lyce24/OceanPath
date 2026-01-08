"""
SSL Data Module for Bag-Level Self-Supervised Learning.

OPTIMIZED version - fixes from original:
1. Subsample BEFORE copying from memmap (don't load 50k rows to keep 5k)
2. Remove unnecessary .clone() calls 
3. Lazy float32 casting (only when needed for noise/math)
4. Reduced prefetch_factor to avoid RAM explosion
5. Keep float16 through pipeline where possible

Expected memmap structure:
    features.bin      - [total_patches, D] float16/float32 binary
    index_arrays.npz  - offsets, lengths, slide_ids, feat_dim, dtype, total_patches
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


# =============================================================================
# Bag Augmentations (OPTIMIZED - lazy float32, no unnecessary copies)
# =============================================================================
@dataclass
class BagAugmentationConfig:
    """Configuration for bag-level augmentations."""
    
    subsample_ratio: Tuple[float, float] = (0.5, 0.9)
    instance_dropout_prob: float = 0.1
    noise_std: float = 0.1
    feature_dropout_prob: float = 0.05
    shuffle: bool = True
    min_instances: int = 16
    
    def __post_init__(self):
        if not (0 < self.subsample_ratio[0] <= self.subsample_ratio[1] <= 1.0):
            raise ValueError(f"Invalid subsample_ratio: {self.subsample_ratio}")
        if not (0 <= self.instance_dropout_prob < 1.0):
            raise ValueError(f"Invalid instance_dropout_prob: {self.instance_dropout_prob}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative: {self.noise_std}")
        if not (0 <= self.feature_dropout_prob < 1.0):
            raise ValueError(f"Invalid feature_dropout_prob: {self.feature_dropout_prob}")
        if self.min_instances < 1:
            raise ValueError(f"min_instances must be >= 1: {self.min_instances}")


class BagAugmentation:
    """
    Augmentations for bags of patch features.
    
    OPTIMIZED: 
    - Only upcast to float32 when doing noise/dropout operations
    - Avoids unnecessary copies
    - Returns original dtype
    """
    
    def __init__(self, config: Optional[BagAugmentationConfig] = None):
        self.config = config or BagAugmentationConfig()
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a bag of features.
        
        Args:
            features: [N, D] patch features (will be modified in-place for index ops)
            
        Returns:
            Augmented features [N', D] where N' <= N
        """
        cfg = self.config
        
        if features.ndim != 2:
            raise ValueError(f"Expected 2D tensor [N, D], got shape {features.shape}")
        
        N, D = features.shape
        device = features.device
        original_dtype = features.dtype
        
        # Check if we need float32 for numerical operations
        needs_float32 = (cfg.noise_std > 0) or (cfg.feature_dropout_prob > 0)
        
        # Handle edge case: very small bags
        if N <= cfg.min_instances:
            if needs_float32:
                features = features.float()
                if cfg.noise_std > 0:
                    features = features + torch.randn_like(features) * cfg.noise_std
                if cfg.feature_dropout_prob > 0:
                    feat_mask = torch.rand(D, device=device) > cfg.feature_dropout_prob
                    features = features * feat_mask.float().unsqueeze(0)
                return features.to(original_dtype)
            return features
        
        # 1. Subsample: keep random fraction of instances (index operation, no cast needed)
        if cfg.subsample_ratio[1] < 1.0 or cfg.subsample_ratio[0] < 1.0:
            ratio = random.uniform(cfg.subsample_ratio[0], cfg.subsample_ratio[1])
            n_keep = max(cfg.min_instances, int(N * ratio))
            n_keep = min(n_keep, N)
            
            if n_keep < N:
                indices = torch.randperm(N, device=device)[:n_keep]
                features = features[indices]
                N = features.shape[0]
        
        # 2. Instance dropout: randomly drop instances (index operation, no cast needed)
        if cfg.instance_dropout_prob > 0 and N > cfg.min_instances:
            keep_mask = torch.rand(N, device=device) > cfg.instance_dropout_prob
            n_kept = keep_mask.sum().item()
            
            if n_kept < cfg.min_instances:
                false_indices = (~keep_mask).nonzero(as_tuple=True)[0]
                n_needed = cfg.min_instances - int(n_kept)
                if len(false_indices) >= n_needed:
                    extra_keep = false_indices[torch.randperm(len(false_indices), device=device)[:n_needed]]
                    keep_mask[extra_keep] = True
            
            if keep_mask.sum() > 0:
                features = features[keep_mask]
                N = features.shape[0]
        
        # 3. Shuffle order (index operation, no cast needed)
        if cfg.shuffle and N > 1:
            perm = torch.randperm(N, device=device)
            features = features[perm]
        
        # 4 & 5: Only cast to float32 if we need noise or feature dropout
        if needs_float32:
            features = features.float()
            
            # Add Gaussian noise
            if cfg.noise_std > 0:
                features = features + torch.randn_like(features) * cfg.noise_std
            
            # Feature dimension dropout
            if cfg.feature_dropout_prob > 0:
                feat_mask = torch.rand(D, device=device) > cfg.feature_dropout_prob
                features = features * feat_mask.float().unsqueeze(0)
            
            return features.to(original_dtype)
        
        return features
    
    @classmethod
    def weak(cls) -> "BagAugmentation":
        """Weak augmentation (for BYOL target or evaluation)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.8, 0.95),
            instance_dropout_prob=0.02,
            noise_std=0.02,
            feature_dropout_prob=0.01,
            min_instances=32,
        ))
    
    @classmethod
    def strong(cls) -> "BagAugmentation":
        """Strong augmentation (for contrastive learning)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.4, 0.8),
            instance_dropout_prob=0.15,
            noise_std=0.15,
            feature_dropout_prob=0.1,
            min_instances=16,
        ))
    
    @classmethod
    def medium(cls) -> "BagAugmentation":
        """Medium augmentation (default, balanced)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.5, 0.9),
            instance_dropout_prob=0.1,
            noise_std=0.1,
            feature_dropout_prob=0.05,
            min_instances=16,
        ))
    
    @classmethod
    def index_only(cls) -> "BagAugmentation":
        """Index-only augmentation (no noise, no dropout - stays in original dtype)."""
        return cls(BagAugmentationConfig(
            subsample_ratio=(0.5, 0.9),
            instance_dropout_prob=0.1,
            noise_std=0.0,  # No noise = no float32 needed
            feature_dropout_prob=0.0,  # No dropout = no float32 needed
            min_instances=16,
        ))


# =============================================================================
# Memmap SSL Dataset (OPTIMIZED)
# =============================================================================
class MemmapSSLDataset(Dataset):
    """
    SSL Dataset using memory-mapped binary file.
    
    OPTIMIZED:
    - Subsample BEFORE copying from memmap (critical for large slides)
    - No unnecessary .clone() calls
    - Keeps original dtype through pipeline
    """
    
    def __init__(
        self,
        data_dir: str,
        indices: Optional[List[int]] = None,
        augmentation: Optional[BagAugmentation] = None,
        augmentation_view2: Optional[BagAugmentation] = None,
        max_instances: Optional[int] = None,
        return_key: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing features.bin and index_arrays.npz
            indices: Subset of slide indices to use (for train/val split)
            augmentation: Augmentation for view 1 (and view 2 if augmentation_view2 is None)
            augmentation_view2: Optional separate augmentation for view 2 (asymmetric)
            max_instances: Max patches to sample per slide (CRITICAL: now samples BEFORE loading)
            return_key: Whether to return slide ID with each sample
        """
        self.data_dir = data_dir
        self.max_instances = max_instances
        self.return_key = return_key
        
        # Load index
        index_path = os.path.join(data_dir, "index_arrays.npz")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        index = np.load(index_path, allow_pickle=True)
        
        self._offsets = index['offsets']
        self._lengths = index['lengths']
        self._slide_ids = index['slide_ids']
        
        # Handle different ways feat_dim might be stored
        feat_dim = index['feat_dim']
        self._feat_dim = int(feat_dim.item() if feat_dim.ndim == 0 else feat_dim)
        
        # Handle dtype
        dtype_arr = index['dtype']
        if dtype_arr.ndim == 0:
            self._dtype_str = str(dtype_arr.item())
        else:
            self._dtype_str = str(dtype_arr[0])
        
        # Total patches
        total_patches = index['total_patches']
        self._total_patches = int(total_patches.item() if total_patches.ndim == 0 else total_patches)
        
        # Filter valid indices (offset >= 0) and apply subset
        all_valid = [i for i in range(len(self._offsets)) if self._offsets[i] >= 0]
        
        if indices is not None:
            self.indices = [i for i in indices if i in all_valid or self._offsets[i] >= 0]
        else:
            self.indices = all_valid
        
        # Memory-map the binary file (read-only, OS handles caching)
        bin_path = os.path.join(data_dir, "features.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Features file not found: {bin_path}")
        
        self.data = np.memmap(
            bin_path,
            dtype=self._dtype_str,
            mode='r',
            shape=(self._total_patches, self._feat_dim)
        )
        
        # Augmentations
        self.augmentation = augmentation or BagAugmentation.medium()
        self.augmentation_view2 = augmentation_view2
    
    def __len__(self) -> int:
        return len(self.indices)
    
    @property
    def feature_dim(self) -> int:
        return self._feat_dim
    
    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
    ]:
        """
        Returns:
            view1: Augmented bag [N1, D]
            view2: Differently augmented bag [N2, D]
            slide_id: (if return_key=True) Slide identifier
        """
        # Map to real index
        real_idx = self.indices[idx]
        
        offset = int(self._offsets[real_idx])
        length = int(self._lengths[real_idx])
        slide_id = str(self._slide_ids[real_idx])
        
        # Handle empty/failed slides
        if offset < 0 or length == 0:
            dummy = torch.zeros((1, self._feat_dim), dtype=torch.float32)
            if self.return_key:
                return dummy, dummy, slide_id
            return dummy, dummy
        
        # =================================================================
        # CRITICAL OPTIMIZATION: Subsample BEFORE copying from memmap
        # This is the biggest win - don't load 50k rows to keep 5k
        # =================================================================
        k = length
        if self.max_instances and length > self.max_instances:
            k = self.max_instances
        
        if k < length:
            start = np.random.randint(0, length - k + 1)
            features_np = np.array(self.data[offset + start : offset + start + k], copy=True)
        else:
            features_np = np.array(self.data[offset : offset + length], copy=True)

        # Convert to torch - keep original dtype (float16 if stored as float16)
        features = torch.from_numpy(features_np)
        
        # =================================================================
        # CRITICAL OPTIMIZATION: No .clone() - augmentation doesn't need it
        # The augmentation only does indexing ops (which create new tensors)
        # and additive ops (which also create new tensors)
        # =================================================================
        view1 = self.augmentation(features)
        
        aug2 = self.augmentation_view2 if self.augmentation_view2 is not None else self.augmentation
        view2 = aug2(features)
        
        if self.return_key:
            return view1, view2, slide_id
        return view1, view2


# =============================================================================
# Collate Functions (unchanged)
# =============================================================================
def collate_ssl_bags(
    batch: List[Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
    ]]
) -> Union[
    Tuple[List[torch.Tensor], List[torch.Tensor]],
    Tuple[List[torch.Tensor], List[torch.Tensor], List[str]],
]:
    """Collate variable-size bags into lists."""
    if len(batch[0]) == 3:
        views1, views2, keys = zip(*batch)
        return list(views1), list(views2), list(keys)
    else:
        views1, views2 = zip(*batch)
        return list(views1), list(views2)


def collate_ssl_bags_padded(
    batch: List[Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
    ]],
    pad_value: float = 0.0,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
]:
    """Collate with padding for batched processing."""
    has_keys = len(batch[0]) == 3
    
    if has_keys:
        views1, views2, keys = zip(*batch)
        keys = list(keys)
    else:
        views1, views2 = zip(*batch)
        keys = None
    
    def pad_bags(bags: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_n = max(b.shape[0] for b in bags)
        D = bags[0].shape[1]
        B = len(bags)
        
        padded = torch.full((B, max_n, D), pad_value, dtype=bags[0].dtype)
        mask = torch.zeros(B, max_n, dtype=torch.bool)
        
        for i, bag in enumerate(bags):
            n = bag.shape[0]
            padded[i, :n] = bag
            mask[i, :n] = True
        
        return padded, mask
    
    v1_padded, mask1 = pad_bags(list(views1))
    v2_padded, mask2 = pad_bags(list(views2))
    
    if has_keys:
        return v1_padded, v2_padded, mask1, mask2, keys
    return v1_padded, v2_padded, mask1, mask2


# =============================================================================
# Memmap SSL Data Module (OPTIMIZED)
# =============================================================================
class MemmapSSLDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for bag-level SSL using memory-mapped files.
    
    OPTIMIZED:
    - Reduced prefetch_factor (1-2 instead of 4) to avoid RAM explosion
    - Subsample before loading in dataset
    """
    
    def __init__(
        self,
        data_dir: str,
        # Augmentation
        augmentation_config: Optional[BagAugmentationConfig] = None,
        augmentation_strength: str = "medium",
        asymmetric: bool = False,
        # Data loading
        batch_size: int = 32,
        num_workers: int = 8,
        max_instances: Optional[int] = 5000,
        pin_memory: bool = True,
        # Collate mode
        use_padded_collate: bool = False,
        # Split
        val_fraction: float = 0.05,
        seed: int = 42,
        # Misc
        return_key: bool = False,
        verbose: bool = True,
        # NEW: control prefetch (default reduced from 4 to 2)
        prefetch_factor: int = 2,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.prefetch_factor = prefetch_factor
        
        # Augmentation
        if augmentation_config is not None:
            self.augmentation = BagAugmentation(augmentation_config)
        else:
            aug_map = {
                "weak": BagAugmentation.weak,
                "medium": BagAugmentation.medium,
                "strong": BagAugmentation.strong,
                "index_only": BagAugmentation.index_only,  # NEW: for benchmarking
            }
            if augmentation_strength not in aug_map:
                raise ValueError(f"Unknown augmentation_strength: {augmentation_strength}")
            self.augmentation = aug_map[augmentation_strength]()
        
        self.augmentation_view2 = BagAugmentation.weak() if asymmetric else None
        
        # Data loading
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_instances = max_instances
        self.pin_memory = pin_memory
        self.use_padded_collate = use_padded_collate
        
        # Split
        self.val_fraction = val_fraction
        self.seed = seed
        
        # Misc
        self.return_key = return_key
        self.verbose = verbose
        
        # State
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.train_dataset: Optional[MemmapSSLDataset] = None
        self.val_dataset: Optional[MemmapSSLDataset] = None
        self._feature_dim: Optional[int] = None
        self._metadata: Optional[Dict] = None
    
    def prepare_data(self):
        """Verify data files exist."""
        bin_path = os.path.join(self.data_dir, "features.bin")
        index_path = os.path.join(self.data_dir, "index_arrays.npz")
        
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"Features file not found: {bin_path}")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
    
    def setup(self, stage: Optional[str] = None):
        """Load index and create train/val split."""
        
        index_path = os.path.join(self.data_dir, "index_arrays.npz")
        index = np.load(index_path, allow_pickle=True)
        
        offsets = index['offsets']
        lengths = index['lengths']
        
        valid_indices = [
            i for i in range(len(offsets)) 
            if offsets[i] >= 0 and lengths[i] > 0
        ]
        
        n_total = len(valid_indices)
        n_failed = len(offsets) - n_total
        
        if n_total == 0:
            raise RuntimeError("No valid slides found in memmap!")
        
        # Train/val split
        rng = np.random.default_rng(self.seed)
        shuffled_indices = valid_indices.copy()
        rng.shuffle(shuffled_indices)
        
        n_val = max(1, int(n_total * self.val_fraction))
        self.val_indices = shuffled_indices[:n_val]
        self.train_indices = shuffled_indices[n_val:]
        
        if self.verbose:
            print(f"[MemmapSSLDataModule] Data dir: {self.data_dir}")
            print(f"[MemmapSSLDataModule] Valid slides: {n_total} ({n_failed} failed/empty)")
            print(f"[MemmapSSLDataModule] Split: train={len(self.train_indices)}, val={len(self.val_indices)}")
        
        # Create datasets
        self.train_dataset = MemmapSSLDataset(
            data_dir=self.data_dir,
            indices=self.train_indices,
            augmentation=self.augmentation,
            augmentation_view2=self.augmentation_view2,
            max_instances=self.max_instances,
            return_key=self.return_key,
        )
        
        self.val_dataset = MemmapSSLDataset(
            data_dir=self.data_dir,
            indices=self.val_indices,
            augmentation=self.augmentation,
            augmentation_view2=self.augmentation_view2,
            max_instances=self.max_instances,
            return_key=self.return_key,
        )
        
        self._feature_dim = self.train_dataset.feature_dim
        
        if self.verbose:
            print(f"[MemmapSSLDataModule] Feature dim: {self._feature_dim}")
            print(f"[MemmapSSLDataModule] Max instances: {self.max_instances}")
            print(f"[MemmapSSLDataModule] Prefetch factor: {self.prefetch_factor}")
    
    def _get_collate_fn(self):
        if self.use_padded_collate:
            return collate_ssl_bags_padded
        return collate_ssl_bags
    
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._get_collate_fn(),
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            # REDUCED from 4 to self.prefetch_factor (default 2)
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() first")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(4, self.num_workers),
            pin_memory=self.pin_memory,
            collate_fn=self._get_collate_fn(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
    
    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("Call setup() first to get feature_dim")
        return self._feature_dim
    
    @property
    def num_train_samples(self) -> int:
        return len(self.train_indices)
    
    @property
    def num_val_samples(self) -> int:
        return len(self.val_indices)


# =============================================================================
# Benchmark Utility (IMPROVED - separate components)
# =============================================================================
def benchmark_memmap_loading(data_dir: str, n_samples: int = 500):
    """
    Benchmark memmap loading speed with component breakdown.
    
    Tests:
    1. Raw memmap read (no augmentation)
    2. With index-only augmentation (no float32 ops)
    3. Full pipeline with medium augmentation
    """
    import time
    
    print(f"Benchmarking memmap loading from: {data_dir}\n")
    
    # Load index to check slide sizes
    index = np.load(os.path.join(data_dir, "index_arrays.npz"), allow_pickle=True)
    lengths = index['lengths']
    valid_lengths = lengths[lengths > 0]
    
    print(f"Slide size stats:")
    print(f"  Min: {valid_lengths.min()}")
    print(f"  Max: {valid_lengths.max()}")
    print(f"  Mean: {valid_lengths.mean():.0f}")
    print(f"  Median: {np.median(valid_lengths):.0f}")
    print()
    
    # Test 1: Raw loading only (no augmentation)
    print("--- Test 1: Raw memmap read (max_instances=5000, no augmentation) ---")
    
    class RawDataset(Dataset):
        def __init__(self, data_dir, max_instances=5000):
            index = np.load(os.path.join(data_dir, "index_arrays.npz"), allow_pickle=True)
            self._offsets = index['offsets']
            self._lengths = index['lengths']
            feat_dim = index['feat_dim']
            self._feat_dim = int(feat_dim.item() if feat_dim.ndim == 0 else feat_dim)
            dtype_arr = index['dtype']
            dtype_str = str(dtype_arr.item() if dtype_arr.ndim == 0 else dtype_arr[0])
            total_patches = index['total_patches']
            total = int(total_patches.item() if total_patches.ndim == 0 else total_patches)
            
            self.indices = [i for i in range(len(self._offsets)) if self._offsets[i] >= 0]
            self.data = np.memmap(
                os.path.join(data_dir, "features.bin"),
                dtype=dtype_str, mode='r', shape=(total, self._feat_dim)
            )
            self.max_instances = max_instances
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            offset = int(self._offsets[real_idx])
            length = int(self._lengths[real_idx])
            
            k = min(length, self.max_instances) if self.max_instances else length
            
            if k < length:
                rel = np.random.choice(length, size=k, replace=False)
                features_np = np.array(self.data[offset + rel], copy=True)
            else:
                features_np = np.array(self.data[offset:offset + length], copy=True)
            
            return torch.from_numpy(features_np)
    
    raw_ds = RawDataset(data_dir, max_instances=5000)
    
    # Warmup
    for i in range(min(10, len(raw_ds))):
        _ = raw_ds[i]
    
    n_test = min(n_samples, len(raw_ds))
    indices = np.random.permutation(len(raw_ds))[:n_test]
    
    start = time.time()
    for idx in indices:
        _ = raw_ds[idx]
    elapsed = time.time() - start
    
    raw_speed = n_test / elapsed
    print(f"  Samples: {n_test}, Time: {elapsed:.2f}s")
    print(f"  Speed: {raw_speed:.0f} samples/sec")
    
    # Test 2: Index-only augmentation (no float32)
    print("\n--- Test 2: With index-only augmentation (no noise/dropout) ---")
    
    ds_index = MemmapSSLDataset(
        data_dir=data_dir,
        augmentation=BagAugmentation.index_only(),
        max_instances=5000,
    )
    
    for i in range(min(10, len(ds_index))):
        _ = ds_index[i]
    
    start = time.time()
    for idx in indices:
        _ = ds_index[idx]
    elapsed = time.time() - start
    
    index_speed = n_test / elapsed
    print(f"  Samples: {n_test}, Time: {elapsed:.2f}s")
    print(f"  Speed: {index_speed:.0f} samples/sec")
    
    # Test 3: Full pipeline with medium augmentation
    print("\n--- Test 3: Full pipeline with medium augmentation ---")
    
    ds_full = MemmapSSLDataset(
        data_dir=data_dir,
        augmentation=BagAugmentation.medium(),
        max_instances=5000,
    )
    
    for i in range(min(10, len(ds_full))):
        _ = ds_full[i]
    
    start = time.time()
    for idx in indices:
        _ = ds_full[idx]
    elapsed = time.time() - start
    
    full_speed = n_test / elapsed
    print(f"  Samples: {n_test}, Time: {elapsed:.2f}s")
    print(f"  Speed: {full_speed:.0f} samples/sec")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Raw read:           {raw_speed:>6.0f} samples/sec")
    print(f"Index-only aug:     {index_speed:>6.0f} samples/sec")
    print(f"Full aug (medium):  {full_speed:>6.0f} samples/sec")
    print(f"\nProjected epoch times (17385 slides):")
    print(f"  Raw:        {17385 / raw_speed / 60:>5.1f} min")
    print(f"  Index-only: {17385 / index_speed / 60:>5.1f} min")
    print(f"  Full aug:   {17385 / full_speed / 60:>5.1f} min")
    
    return {
        'raw': raw_speed,
        'index_only': index_speed,
        'full': full_speed,
    }


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./src/univ2_mmap/")
    args = parser.parse_args()
    
    print("=== Testing MemmapSSLDataModule ===\n")
    
    start_time = time.time()
    dm = MemmapSSLDataModule(
        data_dir=args.data_dir,
        batch_size=32,
        num_workers=8,
        max_instances=5000,
        use_padded_collate=False,
        return_key=True,
        verbose=True,
        prefetch_factor=2,  # Reduced from 4
    )
    
    dm.prepare_data()
    dm.setup()
    
    print(f"\nFeature dim: {dm.feature_dim}")
    print(f"Train samples: {dm.num_train_samples}")
    print(f"Val samples: {dm.num_val_samples}")
    
    # Test train loader
    print("\n--- Testing Train DataLoader ---")
    train_loader = dm.train_dataloader()
    
    for i, batch in enumerate(train_loader):
        views1, views2, keys = batch
        print(f"Batch {i}:")
        print(f"  Num bags: {len(views1)}")
        print(f"  View1 shapes: {[v.shape for v in views1[:3]]}")
        print(f"  View1 dtype: {views1[0].dtype}")
        print(f"  Keys: {keys[:3]}")
        
        if i >= 2:
            break
    end_time = time.time()
    print(f"Train loader test time: {end_time - start_time:.2f}s")
    print("\n✓ All tests passed!")

    benchmark_memmap_loading(args.data_dir)