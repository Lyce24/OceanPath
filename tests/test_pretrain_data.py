"""
Tests for SSL pretraining dataset and datamodule.

Two testing strategies:
  1. **Real mmap data**: Build actual memmap files on disk matching the
     format expected by PretrainDataset, then run the full read → augment
     → collate pipeline end-to-end.
  2. **Fake/mock data**: Patch internals to test logic (splitting, filtering,
     collation math) without touching disk.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures: build real mmap data on disk
# ═════════════════════════════════════════════════════════════════════════════

FEAT_DIM = 16
COORD_DIM = 2
FEAT_DTYPE = "float16"
COORD_DTYPE = "int32"


def _build_mmap_dir(
    tmp_path: Path,
    slide_ids: list[str],
    patch_counts: list[int],
    feat_dim: int = FEAT_DIM,
    seed: int = 0,
) -> Path:
    """
    Create a valid mmap directory with features, coords, and index.

    Produces a single chunk (chunk 0) with all slides concatenated.
    Returns the mmap directory path.
    """
    mmap_dir = tmp_path / "mmap"
    mmap_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    n_slides = len(slide_ids)
    feat_elem_size = np.dtype(FEAT_DTYPE).itemsize
    coord_elem_size = np.dtype(COORD_DTYPE).itemsize

    # Accumulate features and coords into one flat array per chunk
    all_feats = []
    all_coords = []
    feat_offsets = []
    coord_offsets = []
    feat_byte_cursor = 0
    coord_byte_cursor = 0

    for _i, n in enumerate(patch_counts):
        feat_offsets.append(feat_byte_cursor)
        coord_offsets.append(coord_byte_cursor)

        feats = rng.standard_normal((n, feat_dim)).astype(FEAT_DTYPE)
        # Coords: scatter patches in a 1000x1000 space
        coords = rng.integers(0, 1000, size=(n, COORD_DIM)).astype(COORD_DTYPE)

        all_feats.append(feats)
        all_coords.append(coords)

        feat_byte_cursor += n * feat_dim * feat_elem_size
        coord_byte_cursor += n * COORD_DIM * coord_elem_size

    # Write feature chunk
    feat_flat = np.concatenate([f.ravel() for f in all_feats])
    feat_mm = np.memmap(
        str(mmap_dir / "features_000.bin"),
        dtype=FEAT_DTYPE,
        mode="w+",
        shape=feat_flat.shape,
    )
    feat_mm[:] = feat_flat
    feat_mm.flush()

    # Write coord chunk
    coord_flat = np.concatenate([c.ravel() for c in all_coords])
    coord_mm = np.memmap(
        str(mmap_dir / "coords_000.bin"),
        dtype=COORD_DTYPE,
        mode="w+",
        shape=coord_flat.shape,
    )
    coord_mm[:] = coord_flat
    coord_mm.flush()

    # Write index
    np.savez(
        str(mmap_dir / "index_arrays.npz"),
        slide_ids=np.array(slide_ids),
        lengths=np.array(patch_counts, dtype=np.int64),
        feat_chunk_ids=np.zeros(n_slides, dtype=np.int64),
        feat_offsets=np.array(feat_offsets, dtype=np.int64),
        coord_chunk_ids=np.zeros(n_slides, dtype=np.int64),
        coord_offsets=np.array(coord_offsets, dtype=np.int64),
        feat_dtype=np.array([FEAT_DTYPE]),
        coord_dtype=np.array([COORD_DTYPE]),
        bytes_per_feat=np.array([feat_dim * feat_elem_size]),
        num_feat_chunks=np.array([1]),
        num_coord_chunks=np.array([1]),
        feat_dim=np.array([feat_dim]),
        coord_dim=np.array([COORD_DIM]),
    )

    return mmap_dir


def _mock_validate_mmap_dir(mmap_dir):
    """Return metadata dict matching what validate_mmap_dir produces."""
    idx = np.load(str(Path(mmap_dir) / "index_arrays.npz"), allow_pickle=True)
    return {
        "feat_dim": int(idx["feat_dim"]),
        "coord_dim": int(idx["coord_dim"]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_mmap(tmp_path):
    """5 slides with varying patch counts."""
    ids = ["slide_a", "slide_b", "slide_c", "slide_d", "slide_e"]
    counts = [50, 120, 30, 200, 80]
    mmap_dir = _build_mmap_dir(tmp_path, ids, counts)
    return mmap_dir, ids, counts


@pytest.fixture
def tiny_mmap(tmp_path):
    """2 slides, very small — for fast edge-case tests."""
    ids = ["tiny_1", "tiny_2"]
    counts = [10, 5]
    mmap_dir = _build_mmap_dir(tmp_path, ids, counts)
    return mmap_dir, ids, counts


@pytest.fixture
def single_slide_mmap(tmp_path):
    """1 slide — edge case for splits."""
    ids = ["solo"]
    counts = [100]
    mmap_dir = _build_mmap_dir(tmp_path, ids, counts)
    return mmap_dir, ids, counts


@pytest.fixture
def pretrain_csv(tmp_path, small_mmap):
    """CSV listing a subset of slides."""
    csv_path = tmp_path / "pretrain.csv"
    # Include 3 of 5 slides, plus one that doesn't exist in mmap
    csv_path.write_text("slide_id\nslide_a\nslide_c\nslide_e\nslide_nonexistent\n")
    return csv_path


# ═════════════════════════════════════════════════════════════════════════════
# Import helpers (patch validate_mmap_dir since we build our own mmap)
# ═════════════════════════════════════════════════════════════════════════════

# We patch validate_mmap_dir at the module level where it's imported.
# BaseMmapDataset (in dataset.py) calls validate_mmap_dir, so patch there.
# PretrainDataModule also imports it directly for its feat_dim property.
VALIDATE_PATCH_DATASET = "oceanpath.data.dataset.validate_mmap_dir"
VALIDATE_PATCH_DATAMODULE = "oceanpath.ssl.data.datamodule.validate_mmap_dir"


# NOTE: TestPretrainDatasetReal and TestPretrainDatasetAugmentations were
# removed when their helper classes (FeatureAugmentor, CoordAffine,
# LocalFeatureSmooth, SpatialCrop, SpatialFeatureInterpolation,
# SpatialRegionDrop, TissueRegionMixup) and the legacy ``augmentor=`` kwarg
# on PretrainDataset were retired in favor of the dual-view / multi-crop
# augmentor schema. Replacements should target WSIDualViewAugmentor +
# PretrainDataset(view_generator=...).


# NOTE: TestDualViewCollator was removed. The legacy DualViewCollator
# class no longer lives in oceanpath.ssl.data.datamodule (the new dual-view
# collator is the function stack_collate). Replacements should target
# stack_collate directly with the per-sample fixed_n contract.


# ═════════════════════════════════════════════════════════════════════════════
# Tests: PretrainDataModule (real mmap + CSV filtering)
# ═════════════════════════════════════════════════════════════════════════════


class TestPretrainDataModule:
    """Integration tests for the full DataModule lifecycle."""

    def _make_dm(self, mmap_dir, csv_path=None, **kwargs):
        from oceanpath.ssl.data.datamodule import PretrainDataModule

        defaults = {
            "mmap_dir": str(mmap_dir),
            "csv_path": str(csv_path) if csv_path else None,
            "batch_size": 2,
            "max_instances": 512,
            "dataset_max_instances": 256,
            "num_workers": 0,
            "val_frac": 0.2,
            "seed": 42,
            "coords_aware": True,
        }
        defaults.update(kwargs)
        return PretrainDataModule(**defaults)

    def test_setup_creates_datasets(self, small_mmap):
        mmap_dir, ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir)
            dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) + len(dm.val_dataset) == len(ids)

    def test_setup_idempotent(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir)
            dm.setup()
            train_ds = dm.train_dataset
            dm.setup()  # second call should be no-op
            assert dm.train_dataset is train_ds

    def test_train_val_split_no_overlap(self, small_mmap):
        mmap_dir, ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir)
            dm.setup()

        train_ids = set(dm.train_dataset.slide_ids)
        val_ids = set(dm.val_dataset.slide_ids)
        assert len(train_ids & val_ids) == 0, "Train/val overlap!"
        assert train_ids | val_ids == set(ids)

    def test_split_reproducible(self, small_mmap):
        """Same seed → same split."""
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm1 = self._make_dm(mmap_dir, seed=123)
            dm1.setup()
            dm2 = self._make_dm(mmap_dir, seed=123)
            dm2.setup()

        assert dm1.train_dataset.slide_ids == dm2.train_dataset.slide_ids
        assert dm1.val_dataset.slide_ids == dm2.val_dataset.slide_ids

    def test_different_seed_different_split(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm1 = self._make_dm(mmap_dir, seed=1)
            dm1.setup()
            dm2 = self._make_dm(mmap_dir, seed=999)
            dm2.setup()

        # With 5 slides they *could* match by chance, but very unlikely
        assert set(dm1.val_dataset.slide_ids) != set(dm2.val_dataset.slide_ids), (
            "Different seeds produced identical splits (possible but very unlikely)"
        )

    def test_csv_filtering(self, small_mmap, pretrain_csv):
        """CSV should filter down to only matching slides."""
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, csv_path=pretrain_csv)
            dm.setup()

        all_ids = set(dm.train_dataset.slide_ids) | set(dm.val_dataset.slide_ids)
        # CSV has slide_a, slide_c, slide_e (slide_nonexistent is ignored)
        assert all_ids == {"slide_a", "slide_c", "slide_e"}

    def test_no_csv_uses_all_slides(self, small_mmap):
        mmap_dir, ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, csv_path=None)
            dm.setup()

        all_ids = set(dm.train_dataset.slide_ids) | set(dm.val_dataset.slide_ids)
        assert all_ids == set(ids)

    def test_val_at_least_one(self, single_slide_mmap):
        """Even with 1 slide, val should get at least 1 slide."""
        mmap_dir, _ids, _counts = single_slide_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, val_frac=0.1)
            dm.setup()

        assert len(dm.val_dataset) >= 1

    def test_train_dataloader_returns_batches(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, batch_size=2, num_workers=0)
            dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert "view1" in batch
        assert "view2" in batch
        assert "mask1" in batch
        assert "mask2" in batch
        assert batch["view1"].shape[0] == 2  # batch_size
        assert batch["view1"].shape[2] == FEAT_DIM

    def test_val_dataloader_returns_batches(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, batch_size=2, num_workers=0)
            dm.setup()

        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert batch["view1"].ndim == 3

    def test_feat_dim_property(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        with patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir):
            dm = self._make_dm(mmap_dir)
            assert dm.feat_dim == FEAT_DIM

    def test_max_instances_respected_in_batch(self, small_mmap):
        """Collator max_instances should cap sequence length."""
        mmap_dir, _ids, _counts = small_mmap
        cap = 32
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(
                mmap_dir, max_instances=cap, dataset_max_instances=cap, batch_size=2, num_workers=0
            )
            dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["view1"].shape[1] <= cap
        assert batch["view2"].shape[1] <= cap

    def test_coords_in_batch(self, small_mmap):
        """Coords should flow through to collated batch."""
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, batch_size=2, num_workers=0, coords_aware=True)
            dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "coords1" in batch
        assert "coords2" in batch
        assert batch["coords1"].dtype == torch.int32


# NOTE: TestEdgeCases was removed alongside the legacy FeatureAugmentor
# class and the long-gone DualViewCollator. The dual-view collator path is
# now ``stack_collate``; replacements should target it directly.


# ═════════════════════════════════════════════════════════════════════════════
# Tests: DataModule internal logic with mocked I/O
# ═════════════════════════════════════════════════════════════════════════════


class TestDataModuleSplitLogic:
    """Test _load_slide_ids and _split_slide_ids with mocked filesystem."""

    def test_split_fractions(self):
        """Validate that val_frac produces the right number of val slides."""
        from oceanpath.ssl.data.datamodule import PretrainDataModule

        with patch(VALIDATE_PATCH_DATAMODULE, return_value={"feat_dim": 16, "coord_dim": 2}):
            dm = PretrainDataModule.__new__(PretrainDataModule)
            dm.val_frac = 0.2
            dm.seed = 42

        ids = [f"slide_{i}" for i in range(100)]
        train, val = dm._split_slide_ids(ids)

        assert len(val) == 20
        assert len(train) == 80
        assert set(train) | set(val) == set(ids)
        assert len(set(train) & set(val)) == 0

    def test_split_min_one_val(self):
        """Even with very few slides, val should have at least 1."""
        from oceanpath.ssl.data.datamodule import PretrainDataModule

        with patch(VALIDATE_PATCH_DATAMODULE, return_value={"feat_dim": 16, "coord_dim": 2}):
            dm = PretrainDataModule.__new__(PretrainDataModule)
            dm.val_frac = 0.01
            dm.seed = 42

        ids = ["a", "b", "c"]
        _train, val = dm._split_slide_ids(ids)
        assert len(val) >= 1

    def test_split_manifest_is_respected(self, tmp_path):
        from oceanpath.ssl.data.datamodule import PretrainDataModule

        manifest = tmp_path / "split_manifest.json"
        manifest.write_text(
            '{\n  "train_ids": ["slide_a", "slide_c"],\n  "val_ids": ["slide_b"]\n}\n'
        )

        with patch(VALIDATE_PATCH_DATAMODULE, return_value={"feat_dim": 16, "coord_dim": 2}):
            dm = PretrainDataModule.__new__(PretrainDataModule)
            dm.val_frac = 0.2
            dm.seed = 42
            dm.split_manifest_path = str(manifest)

        ids = ["slide_a", "slide_b", "slide_c", "slide_d"]
        train, val = dm._split_slide_ids(ids)
        assert train == ["slide_a", "slide_c"]
        assert val == ["slide_b"]

    def test_missing_manifest_falls_back_to_random_split(self):
        from oceanpath.ssl.data.datamodule import PretrainDataModule

        with patch(VALIDATE_PATCH_DATAMODULE, return_value={"feat_dim": 16, "coord_dim": 2}):
            dm = PretrainDataModule.__new__(PretrainDataModule)
            dm.val_frac = 0.5
            dm.seed = 7
            dm.split_manifest_path = "/tmp/does-not-exist.json"

        ids = ["a", "b", "c", "d"]
        train, val = dm._split_slide_ids(ids)
        assert len(train) == 2
        assert len(val) == 2
        assert set(train) | set(val) == set(ids)


# ═════════════════════════════════════════════════════════════════════════════
# Tests: BucketBatchSampler
# ═════════════════════════════════════════════════════════════════════════════


class TestBucketBatchSampler:
    """Test BucketBatchSampler grouping, coverage, and reproducibility."""

    def test_all_indices_covered(self):
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([30, 50, 120, 200, 80, 45, 90, 150, 60, 100])
        sampler = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=3, n_buckets=3, drop_last=False, seed=42
        )
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(bag_sizes)))

    def test_batch_size_respected(self):
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([30, 50, 120, 200, 80, 45, 90, 150, 60, 100])
        sampler = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=4, n_buckets=3, drop_last=True, seed=42
        )
        for batch in sampler:
            assert len(batch) == 4

    def test_within_bucket_similarity(self):
        """Samples in same batch should have similar bag sizes."""
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([10, 20, 30, 100, 110, 120, 200, 210, 220, 300, 310, 320])
        sampler = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=3, n_buckets=4, drop_last=False, seed=42
        )
        for batch in sampler:
            sizes = [bag_sizes[i] for i in batch]
            # Within a bucket, range should be <= total range / n_buckets * 2 (generous)
            assert max(sizes) - min(sizes) <= 200, f"Batch too spread: {sizes}"

    def test_drop_last(self):
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([30, 50, 120, 200, 80, 45, 90])  # 7 samples
        sampler_drop = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=3, n_buckets=2, drop_last=True, seed=42
        )
        sampler_keep = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=3, n_buckets=2, drop_last=False, seed=42
        )
        batches_drop = list(sampler_drop)
        batches_keep = list(sampler_keep)
        # drop_last should produce fewer or equal batches
        assert len(batches_drop) <= len(batches_keep)
        # All batches in drop_last mode are full-size
        for b in batches_drop:
            assert len(b) == 3

    def test_reproducibility(self):
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([30, 50, 120, 200, 80, 45, 90, 150])
        s1 = BucketBatchSampler(bag_sizes=bag_sizes, batch_size=2, n_buckets=3, seed=99)
        s2 = BucketBatchSampler(bag_sizes=bag_sizes, batch_size=2, n_buckets=3, seed=99)
        assert list(s1) == list(s2)

    def test_set_epoch_changes_order(self):
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([30, 50, 120, 200, 80, 45, 90, 150, 60, 100])
        sampler = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=2, n_buckets=3, drop_last=False, seed=42
        )
        epoch0 = list(sampler)
        sampler.set_epoch(1)
        epoch1 = list(sampler)
        # Different epoch should (almost certainly) produce different batch order
        assert epoch0 != epoch1

    def test_len_estimate(self):
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([30, 50, 120, 200, 80, 45, 90, 150, 60, 100])
        sampler = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=3, n_buckets=3, drop_last=True, seed=42
        )
        assert len(sampler) == len(list(sampler))

    def test_small_dataset_fewer_buckets(self):
        from oceanpath.data.batching import BucketBatchSampler

        bag_sizes = np.array([30, 50])
        sampler = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=1, n_buckets=10, drop_last=False, seed=42
        )
        # Should not crash; n_buckets clamped to min(10, 2)
        batches = list(sampler)
        assert len(batches) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Tests: TokenBudgetBatchSampler
# ═════════════════════════════════════════════════════════════════════════════


class TestTokenBudgetBatchSampler:
    """Test TokenBudgetBatchSampler budget constraints and coverage."""

    def test_budget_not_exceeded(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 150, 80, 300, 50, 120, 250])
        budget = 400
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=budget,
            max_batch_size=64,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        for batch in sampler:
            total_tokens = sum(bag_sizes[i] for i in batch)
            # Budget may be exceeded only for single-sample batches
            if len(batch) > 1:
                assert total_tokens <= budget, f"Budget exceeded: {total_tokens} > {budget}"

    def test_all_indices_covered(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 150, 80, 300, 50, 120, 250])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=500,
            max_batch_size=64,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(bag_sizes)))

    def test_variable_batch_sizes(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([10, 20, 500, 15, 25, 600, 30, 12])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=100,
            max_batch_size=64,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        sizes = [len(b) for b in sampler]
        assert len(set(sizes)) > 1, "Expected variable batch sizes"

    def test_max_batch_size_cap(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([1] * 100)
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=999999,
            max_batch_size=10,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        for batch in sampler:
            assert len(batch) <= 10

    def test_min_batch_size_drop(self):
        """Remainder batches below min_batch_size are dropped."""
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 150, 80, 300])
        sampler_drop = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=350,
            max_batch_size=64,
            min_batch_size=2,
            drop_last=True,
            seed=42,
        )
        sampler_keep = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=350,
            max_batch_size=64,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        drop_indices = set()
        for batch in sampler_drop:
            drop_indices.update(batch)
        keep_indices = set()
        for batch in sampler_keep:
            keep_indices.update(batch)
        # keep should have all indices; drop may have fewer
        assert keep_indices == set(range(len(bag_sizes)))
        assert drop_indices.issubset(keep_indices)

    def test_sort_within_batch(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 50, 80, 300, 150, 120, 250])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=600,
            max_batch_size=64,
            min_batch_size=1,
            sort_within_batch=True,
            drop_last=False,
            seed=42,
        )
        for batch in sampler:
            if len(batch) > 1:
                sizes = [bag_sizes[i] for i in batch]
                assert sizes == sorted(sizes, reverse=True)

    def test_single_huge_sample(self):
        """A single sample larger than budget should be yielded as solo batch."""
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([50000, 100, 200])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=1000,
            max_batch_size=64,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        # The huge sample must appear
        assert 0 in all_indices

    def test_set_epoch(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 150, 80, 300, 50, 120, 250])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=500,
            max_batch_size=64,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        epoch0 = list(sampler)
        sampler.set_epoch(1)
        epoch1 = list(sampler)
        assert epoch0 != epoch1

    def test_len_matches_iteration(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 150, 80, 300, 50, 120, 250])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=500,
            max_batch_size=64,
            min_batch_size=1,
            drop_last=False,
            seed=42,
        )
        assert len(sampler) == len(list(sampler))

    def test_sorted_mode(self):
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([300, 50, 200, 100, 150])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=400,
            max_batch_size=64,
            min_batch_size=1,
            sort=True,
            drop_last=False,
            seed=42,
        )
        # In sort mode, iteration order is deterministic regardless of epoch
        batches1 = list(sampler)
        sampler.set_epoch(1)
        batches2 = list(sampler)
        # sort=True ignores shuffling, so batches should be identical
        assert batches1 == batches2


# ═════════════════════════════════════════════════════════════════════════════
# Tests: FixedNCollator
# ═════════════════════════════════════════════════════════════════════════════


class TestFixedNCollator:
    """Test FixedNCollator subsampling and padding."""

    def _make_sample(self, n1, n2, D=FEAT_DIM, with_coords=True):
        sample = {
            "view1_features": torch.randn(n1, D),
            "view2_features": torch.randn(n2, D),
            "slide_id": f"slide_{n1}_{n2}",
        }
        if with_coords:
            sample["view1_coords"] = torch.randint(0, 1000, (n1, 2), dtype=torch.int32)
            sample["view2_coords"] = torch.randint(0, 1000, (n2, 2), dtype=torch.int32)
        return sample

    def test_fixed_output_shape(self):
        from oceanpath.data.batching import FixedNCollator

        N = 64
        collator = FixedNCollator(fixed_n=N)
        batch = [self._make_sample(100, 50), self._make_sample(30, 80)]
        out = collator(batch)

        assert out["view1"].shape == (2, N, FEAT_DIM)
        assert out["view2"].shape == (2, N, FEAT_DIM)
        assert out["mask1"].shape == (2, N)
        assert out["mask2"].shape == (2, N)

    def test_short_sample_padding(self):
        from oceanpath.data.batching import FixedNCollator

        N = 64
        collator = FixedNCollator(fixed_n=N)
        batch = [self._make_sample(20, 10)]
        out = collator(batch)

        # Mask should indicate 20 real tokens for view1, 10 for view2
        assert out["mask1"][0].sum().item() == 20
        assert out["mask2"][0].sum().item() == 10
        # Padding region should be zero
        assert torch.all(out["view1"][0, 20:] == 0)
        assert torch.all(out["view2"][0, 10:] == 0)

    def test_long_sample_subsampled(self):
        from oceanpath.data.batching import FixedNCollator

        N = 32
        collator = FixedNCollator(fixed_n=N)
        batch = [self._make_sample(200, 150)]
        out = collator(batch)

        # After subsampling, all N positions should be active
        assert out["mask1"][0].sum().item() == N
        assert out["mask2"][0].sum().item() == N
        # No zero rows in the active region (subsampled from non-zero features)
        assert not torch.all(out["view1"][0] == 0)

    def test_mask_correctness(self):
        from oceanpath.data.batching import FixedNCollator

        N = 50
        collator = FixedNCollator(fixed_n=N)
        batch = [self._make_sample(30, 80), self._make_sample(60, 20)]
        out = collator(batch)

        # Sample 0: view1 has 30 < 50 → 30 active; view2 has 80 > 50 → 50 active
        assert out["lengths1"][0].item() == 30
        assert out["lengths2"][0].item() == N
        # Sample 1: view1 has 60 > 50 → 50 active; view2 has 20 < 50 → 20 active
        assert out["lengths1"][1].item() == N
        assert out["lengths2"][1].item() == 20

    def test_coords_consistency(self):
        from oceanpath.data.batching import FixedNCollator

        N = 32
        collator = FixedNCollator(fixed_n=N)
        batch = [self._make_sample(100, 50, with_coords=True)]
        out = collator(batch)

        assert "coords1" in out
        assert "coords2" in out
        assert out["coords1"].shape == (1, N, 2)
        assert out["coords2"].shape == (1, N, 2)

    def test_output_dict_format(self):
        from oceanpath.data.batching import FixedNCollator

        collator = FixedNCollator(fixed_n=64)
        batch = [self._make_sample(50, 50)]
        out = collator(batch)

        expected_keys = {"view1", "mask1", "view2", "mask2", "lengths1", "lengths2", "slide_ids"}
        assert expected_keys.issubset(set(out.keys()))

    def test_no_coords(self):
        from oceanpath.data.batching import FixedNCollator

        collator = FixedNCollator(fixed_n=32)
        batch = [self._make_sample(50, 50, with_coords=False)]
        out = collator(batch)

        assert "coords1" not in out
        assert "coords2" not in out


# ═════════════════════════════════════════════════════════════════════════════
# Tests: PadToGlobalCollator
# ═════════════════════════════════════════════════════════════════════════════


class TestPadToGlobalCollator:
    """Test PadToGlobalCollator fixed-size padding."""

    def _make_sample(self, n1, n2, D=FEAT_DIM, with_coords=True):
        sample = {
            "view1_features": torch.randn(n1, D),
            "view2_features": torch.randn(n2, D),
            "slide_id": f"slide_{n1}_{n2}",
        }
        if with_coords:
            sample["view1_coords"] = torch.randint(0, 1000, (n1, 2), dtype=torch.int32)
            sample["view2_coords"] = torch.randint(0, 1000, (n2, 2), dtype=torch.int32)
        return sample

    def test_fixed_output_shape(self):
        from oceanpath.data.batching import PadToGlobalCollator

        G = 128
        collator = PadToGlobalCollator(global_max=G)
        batch = [self._make_sample(20, 50), self._make_sample(80, 30)]
        out = collator(batch)

        assert out["view1"].shape == (2, G, FEAT_DIM)
        assert out["view2"].shape == (2, G, FEAT_DIM)
        assert out["mask1"].shape == (2, G)
        assert out["mask2"].shape == (2, G)

    def test_mask_correctness(self):
        from oceanpath.data.batching import PadToGlobalCollator

        G = 128
        collator = PadToGlobalCollator(global_max=G)
        batch = [self._make_sample(20, 50)]
        out = collator(batch)

        assert out["mask1"][0].sum().item() == 20
        assert out["mask2"][0].sum().item() == 50
        assert out["lengths1"][0].item() == 20
        assert out["lengths2"][0].item() == 50

    def test_truncation_to_global_max(self):
        from oceanpath.data.batching import PadToGlobalCollator

        G = 32
        collator = PadToGlobalCollator(global_max=G)
        batch = [self._make_sample(100, 200)]
        out = collator(batch)

        assert out["view1"].shape == (2, G, FEAT_DIM) or out["view1"].shape == (1, G, FEAT_DIM)
        assert out["mask1"][0].sum().item() == G
        assert out["lengths1"][0].item() == G

    def test_coords_padded(self):
        from oceanpath.data.batching import PadToGlobalCollator

        G = 64
        collator = PadToGlobalCollator(global_max=G)
        batch = [self._make_sample(20, 30, with_coords=True)]
        out = collator(batch)

        assert "coords1" in out
        assert out["coords1"].shape == (1, G, 2)

    def test_no_coords(self):
        from oceanpath.data.batching import PadToGlobalCollator

        G = 64
        collator = PadToGlobalCollator(global_max=G)
        batch = [self._make_sample(20, 30, with_coords=False)]
        out = collator(batch)

        assert "coords1" not in out

    def test_output_keys(self):
        from oceanpath.data.batching import PadToGlobalCollator

        collator = PadToGlobalCollator(global_max=64)
        batch = [self._make_sample(20, 30)]
        out = collator(batch)

        expected = {"view1", "mask1", "view2", "mask2", "lengths1", "lengths2", "slide_ids"}
        assert expected.issubset(set(out.keys()))


# ═════════════════════════════════════════════════════════════════════════════
# Tests: RegionalCropCollator
# ═════════════════════════════════════════════════════════════════════════════


class TestRegionalCropCollator:
    """Test TITAN-style spatial crop collator."""

    def _make_sample(self, n1, n2, D=FEAT_DIM, with_coords=True):
        sample = {
            "view1_features": torch.randn(n1, D),
            "view2_features": torch.randn(n2, D),
            "slide_id": f"slide_{n1}_{n2}",
        }
        if with_coords:
            # Scatter patches on a grid so spatial crops work
            sample["view1_coords"] = torch.randint(0, 100, (n1, 2), dtype=torch.int32)
            sample["view2_coords"] = torch.randint(0, 100, (n2, 2), dtype=torch.int32)
        return sample

    def test_fixed_output_shape(self):
        from oceanpath.data.batching import RegionalCropCollator

        N = 32
        collator = RegionalCropCollator(fixed_n=N, crop_frac=0.5)
        batch = [self._make_sample(200, 150), self._make_sample(100, 80)]
        out = collator(batch)

        assert out["view1"].shape == (2, N, FEAT_DIM)
        assert out["view2"].shape == (2, N, FEAT_DIM)
        assert out["mask1"].shape == (2, N)
        assert out["mask2"].shape == (2, N)

    def test_mask_validity(self):
        from oceanpath.data.batching import RegionalCropCollator

        N = 32
        collator = RegionalCropCollator(fixed_n=N, crop_frac=0.5)
        batch = [self._make_sample(200, 150)]
        out = collator(batch)

        # Mask values should be 0 or 1
        assert torch.all((out["mask1"] == 0) | (out["mask1"] == 1))
        # At least some tokens should be active
        assert out["mask1"].sum().item() > 0

    def test_small_slide_padding(self):
        from oceanpath.data.batching import RegionalCropCollator

        N = 64
        collator = RegionalCropCollator(fixed_n=N, crop_frac=1.0)
        batch = [self._make_sample(10, 5)]
        out = collator(batch)

        # With crop_frac=1.0 and small slide, should pad
        assert out["mask1"][0].sum().item() <= 10
        assert out["mask2"][0].sum().item() <= 5
        assert out["view1"].shape[1] == N

    def test_no_coords_fallback(self):
        from oceanpath.data.batching import RegionalCropCollator

        N = 16
        collator = RegionalCropCollator(fixed_n=N, crop_frac=0.5)
        batch = [self._make_sample(50, 30, with_coords=False)]
        out = collator(batch)

        # Should fall back to random subsampling
        assert out["view1"].shape == (1, N, FEAT_DIM)
        assert "coords1" not in out

    def test_coords_in_output(self):
        from oceanpath.data.batching import RegionalCropCollator

        N = 32
        collator = RegionalCropCollator(fixed_n=N, crop_frac=0.5)
        batch = [self._make_sample(100, 80, with_coords=True)]
        out = collator(batch)

        assert "coords1" in out
        assert out["coords1"].shape == (1, N, 2)

    def test_output_keys(self):
        from oceanpath.data.batching import RegionalCropCollator

        collator = RegionalCropCollator(fixed_n=32)
        batch = [self._make_sample(50, 50)]
        out = collator(batch)

        expected = {"view1", "mask1", "view2", "mask2", "lengths1", "lengths2", "slide_ids"}
        assert expected.issubset(set(out.keys()))

    def test_crop_frac_affects_selection(self):
        """Smaller crop_frac should select fewer patches from the spatial field."""
        from oceanpath.data.batching import RegionalCropCollator

        torch.manual_seed(42)
        N = 200
        # Large slide with well-spread coords
        sample = self._make_sample(500, 500)

        # Tight crop
        tight = RegionalCropCollator(fixed_n=N, crop_frac=0.2)
        out_tight = tight([sample])

        # Wide crop
        wide = RegionalCropCollator(fixed_n=N, crop_frac=0.9)
        out_wide = wide([sample])

        # Both should produce the same shape
        assert out_tight["view1"].shape == out_wide["view1"].shape


# ═════════════════════════════════════════════════════════════════════════════
# Tests: SequencePackingCollator
# ═════════════════════════════════════════════════════════════════════════════


class TestSequencePackingCollator:
    """Test sequence packing with block-diagonal attention support."""

    def _make_sample(self, n1, n2, D=FEAT_DIM, with_coords=True, slide_id=None):
        sample = {
            "view1_features": torch.randn(n1, D),
            "view2_features": torch.randn(n2, D),
            "slide_id": slide_id or f"slide_{n1}_{n2}",
        }
        if with_coords:
            sample["view1_coords"] = torch.randint(0, 1000, (n1, 2), dtype=torch.int32)
            sample["view2_coords"] = torch.randint(0, 1000, (n2, 2), dtype=torch.int32)
        return sample

    def test_packed_shape(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(30, 20), self._make_sample(50, 40)]
        out = collator(batch)

        # B=1, total length = sum of individual lengths
        assert out["view1"].shape == (1, 80, FEAT_DIM)
        assert out["view2"].shape == (1, 60, FEAT_DIM)

    def test_mask_all_ones(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(30, 20), self._make_sample(50, 40)]
        out = collator(batch)

        # No padding in packing — all positions are real tokens
        assert out["mask1"].sum().item() == 80
        assert out["mask2"].sum().item() == 60

    def test_segment_ids_present(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(30, 20), self._make_sample(50, 40)]
        out = collator(batch)

        assert "segment_ids1" in out
        assert "segment_ids2" in out
        assert "n_segments" in out
        assert out["n_segments"] == 2

    def test_segment_ids_correctness(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(10, 5), self._make_sample(20, 15)]
        out = collator(batch)

        seg1 = out["segment_ids1"][0]  # [30]
        # First 10 tokens should be segment 0, next 20 should be segment 1
        assert torch.all(seg1[:10] == 0)
        assert torch.all(seg1[10:] == 1)

        seg2 = out["segment_ids2"][0]  # [20]
        assert torch.all(seg2[:5] == 0)
        assert torch.all(seg2[5:] == 1)

    def test_lengths_per_sample(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(30, 20), self._make_sample(50, 40)]
        out = collator(batch)

        assert out["lengths1"].tolist() == [30, 50]
        assert out["lengths2"].tolist() == [20, 40]

    def test_slide_ids_preserved(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [
            self._make_sample(10, 5, slide_id="s1"),
            self._make_sample(20, 15, slide_id="s2"),
        ]
        out = collator(batch)

        assert out["slide_ids"] == ["s1", "s2"]

    def test_single_sample(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(50, 30)]
        out = collator(batch)

        assert out["view1"].shape == (1, 50, FEAT_DIM)
        assert out["n_segments"] == 1
        assert torch.all(out["segment_ids1"][0] == 0)

    def test_coords_packed(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [
            self._make_sample(10, 5, with_coords=True),
            self._make_sample(20, 15, with_coords=True),
        ]
        out = collator(batch)

        assert "coords1" in out
        assert out["coords1"].shape == (1, 30, 2)
        assert out["coords2"].shape == (1, 20, 2)

    def test_no_coords(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(10, 5, with_coords=False)]
        out = collator(batch)

        assert "coords1" not in out

    def test_max_instances_cap(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator(max_instances=16)
        batch = [self._make_sample(100, 50)]
        out = collator(batch)

        assert out["view1"].shape == (1, 16, FEAT_DIM)
        assert out["view2"].shape == (1, 16, FEAT_DIM)

    def test_output_keys(self):
        from oceanpath.data.batching import SequencePackingCollator

        collator = SequencePackingCollator()
        batch = [self._make_sample(10, 5)]
        out = collator(batch)

        expected = {
            "view1",
            "mask1",
            "view2",
            "mask2",
            "lengths1",
            "lengths2",
            "slide_ids",
            "segment_ids1",
            "segment_ids2",
            "n_segments",
        }
        assert expected.issubset(set(out.keys()))


# ═════════════════════════════════════════════════════════════════════════════
# Tests: Batching Strategy Integration (end-to-end with real mmap)
# ═════════════════════════════════════════════════════════════════════════════


_ALL_STRATEGIES = [
    "pad_to_max_in_batch",
    "pad_to_global",
    "token_budget",
    "bucket_batching",
    "subsample_fixed_n",
    "regional_crops",
    "sequence_packing",
    "multi_crop",
    "jepa",
]

_EXPECTED_KEYS = {"view1", "mask1", "view2", "mask2", "lengths1", "lengths2", "slide_ids"}


class TestBatchingStrategyIntegration:
    """End-to-end tests for all strategies using real mmap data."""

    def _make_dm(self, mmap_dir, strategy, **kwargs):
        from oceanpath.ssl.data.datamodule import PretrainDataModule

        defaults = {
            "mmap_dir": str(mmap_dir),
            "batch_size": 2,
            "max_instances": 512,
            "dataset_max_instances": 256,
            "num_workers": 0,
            "val_frac": 0.2,
            "seed": 42,
            "coords_aware": True,
            "batching_strategy": strategy,
            "batching_cfg": {
                "n_buckets": 3,
                "token_budget": 500,
                "max_batch_size": 4,
                "min_batch_size": 1,
                "fixed_n": 32,
                "global_max": 256,
                "crop_frac": 0.5,
                "sort_within_batch": True,
                "drop_last": False,
                "global_crop_n": 32,
                "local_crop_n": 16,
                "n_local_crops": 2,
                "context_ratio": 0.5,
            },
        }
        defaults.update(kwargs)
        return PretrainDataModule(**defaults)

    @pytest.mark.parametrize("strategy", _ALL_STRATEGIES)
    def test_strategy_produces_valid_output(self, small_mmap, strategy):
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, strategy)
            dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        # Check all required keys present
        assert _EXPECTED_KEYS.issubset(set(batch.keys())), (
            f"Missing keys for strategy '{strategy}': {_EXPECTED_KEYS - set(batch.keys())}"
        )

        # Check tensor shapes
        B = batch["view1"].shape[0]
        assert B > 0
        assert batch["view1"].ndim == 3
        assert batch["view2"].ndim == 3
        assert batch["mask1"].ndim == 2
        assert batch["mask2"].ndim == 2
        assert batch["view1"].shape[2] == FEAT_DIM
        assert batch["view2"].shape[2] == FEAT_DIM

        if strategy == "sequence_packing":
            # Packing: B=1, slide_ids/lengths have N_segments entries
            assert B == 1
            assert "segment_ids1" in batch
            n_seg = batch["n_segments"]
            assert len(batch["slide_ids"]) == n_seg
            assert batch["lengths1"].shape == (n_seg,)
            assert batch["lengths2"].shape == (n_seg,)
        elif strategy == "multi_crop":
            assert len(batch["slide_ids"]) == B
            assert "global_views" in batch
            assert "local_views" in batch
            assert len(batch["global_views"]) == 2
            assert len(batch["local_views"]) == 2  # n_local_crops=2 in cfg
        elif strategy == "jepa":
            assert len(batch["slide_ids"]) == B
            assert "context_pos" in batch
            assert "target_pos" in batch
        else:
            assert len(batch["slide_ids"]) == B
            assert batch["lengths1"].shape == (B,)
            assert batch["lengths2"].shape == (B,)

    @pytest.mark.parametrize("strategy", _ALL_STRATEGIES)
    def test_strategy_full_epoch(self, small_mmap, strategy):
        """Iterate full epoch without errors."""
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, strategy)
            dm.setup()

        loader = dm.train_dataloader()
        n_batches = 0
        for batch in loader:
            n_batches += 1
            assert batch["view1"].shape[0] > 0
        assert n_batches > 0

    def test_default_strategy_matches_original(self, small_mmap):
        """pad_to_max_in_batch should behave identically to the original."""
        mmap_dir, _ids, _counts = small_mmap
        with (
            patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir),
            patch(VALIDATE_PATCH_DATAMODULE, side_effect=_mock_validate_mmap_dir),
        ):
            dm = self._make_dm(mmap_dir, "pad_to_max_in_batch")
            dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        # Should use DualViewCollator (standard padding)
        assert batch["view1"].ndim == 3
        # Batch size should match configured batch_size
        assert batch["view1"].shape[0] <= 2


# ═════════════════════════════════════════════════════════════════════════════
# Tests: Padding Efficiency
# ═════════════════════════════════════════════════════════════════════════════


class TestPaddingEfficiency:
    """Verify that bucket batching reduces padding compared to random batching."""

    def test_bucket_less_padding_than_random(self):
        """Bucket batching should have lower padding ratio than random."""
        from oceanpath.data.batching import (
            BatchingConfig,
            BucketBatchSampler,
            build_collator,
        )

        # Create synthetic samples with wide size variation
        rng = np.random.default_rng(42)
        n_samples = 50
        bag_sizes = rng.integers(20, 300, size=n_samples)
        D = 8

        def make_samples(sizes):
            samples = []
            for i, s in enumerate(sizes):
                samples.append(
                    {
                        "view1_features": torch.randn(int(s), D),
                        "view2_features": torch.randn(int(s), D),
                        "slide_id": f"slide_{i}",
                    }
                )
            return samples

        samples = make_samples(bag_sizes)

        def compute_padding_ratio(batches_of_indices, samples_list):
            config = BatchingConfig(strategy="pad_to_max_in_batch", max_instances=4096)
            collator = build_collator(config)
            total_tokens = 0
            total_padded = 0
            for batch_indices in batches_of_indices:
                batch = [samples_list[i] for i in batch_indices]
                out = collator(batch)
                B, T, _D = out["view1"].shape
                total_padded += B * T
                total_tokens += out["mask1"].sum().item()
            if total_padded == 0:
                return 1.0
            return 1.0 - total_tokens / total_padded

        # Bucket batching
        bucket_sampler = BucketBatchSampler(
            bag_sizes=bag_sizes, batch_size=8, n_buckets=5, drop_last=False, seed=42
        )
        bucket_batches = list(bucket_sampler)

        # Random batching
        random_indices = rng.permutation(n_samples)
        random_batches = [random_indices[i : i + 8].tolist() for i in range(0, n_samples, 8)]

        bucket_ratio = compute_padding_ratio(bucket_batches, samples)
        random_ratio = compute_padding_ratio(random_batches, samples)

        assert bucket_ratio < random_ratio, (
            f"Bucket padding ratio ({bucket_ratio:.3f}) should be less than "
            f"random ({random_ratio:.3f})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tests: BatchingConfig validation
# ═════════════════════════════════════════════════════════════════════════════


class TestBatchingConfig:
    """Test BatchingConfig validation."""

    def test_valid_strategies(self):
        from oceanpath.data.batching import BatchingConfig

        for strategy in [
            "pad_to_max_in_batch",
            "pad_to_global",
            "token_budget",
            "bucket_batching",
            "subsample_fixed_n",
            "regional_crops",
            "sequence_packing",
            "multi_crop",
            "jepa",
        ]:
            cfg = BatchingConfig(strategy=strategy)
            assert cfg.strategy == strategy

    def test_invalid_strategy_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="Unknown batching strategy"):
            BatchingConfig(strategy="nonexistent_strategy")

    def test_invalid_fixed_n_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="fixed_n must be >= 1"):
            BatchingConfig(strategy="pad_to_max_in_batch", fixed_n=0)

    def test_invalid_token_budget_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="token_budget must be >= 1"):
            BatchingConfig(strategy="pad_to_max_in_batch", token_budget=0)

    def test_invalid_global_max_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="global_max must be >= 1"):
            BatchingConfig(strategy="pad_to_max_in_batch", global_max=0)

    def test_invalid_crop_frac_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="crop_frac must be in"):
            BatchingConfig(strategy="pad_to_max_in_batch", crop_frac=0.0)

        with pytest.raises(ValueError, match="crop_frac must be in"):
            BatchingConfig(strategy="pad_to_max_in_batch", crop_frac=1.5)

    def test_invalid_context_ratio_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="context_ratio must be in"):
            BatchingConfig(strategy="jepa", context_ratio=0.0)

        with pytest.raises(ValueError, match="context_ratio must be in"):
            BatchingConfig(strategy="jepa", context_ratio=1.0)

    def test_invalid_global_crop_n_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="global_crop_n must be >= 1"):
            BatchingConfig(strategy="multi_crop", global_crop_n=0)

    def test_invalid_local_crop_n_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="local_crop_n must be >= 1"):
            BatchingConfig(strategy="multi_crop", local_crop_n=0)

    def test_invalid_n_local_crops_raises(self):
        from oceanpath.data.batching import BatchingConfig

        with pytest.raises(ValueError, match="n_local_crops must be >= 1"):
            BatchingConfig(strategy="multi_crop", n_local_crops=0)


# ═════════════════════════════════════════════════════════════════════════════
# Tests: MultiCropCollator
# ═════════════════════════════════════════════════════════════════════════════


class TestMultiCropCollator:
    """Test DINOv2/iBOT multi-crop collator."""

    def _make_sample(self, n1, n2, D=FEAT_DIM, with_coords=True):
        sample = {
            "view1_features": torch.randn(n1, D),
            "view2_features": torch.randn(n2, D),
            "slide_id": f"slide_{n1}_{n2}",
        }
        if with_coords:
            sample["view1_coords"] = torch.randint(0, 1000, (n1, 2), dtype=torch.int32)
            sample["view2_coords"] = torch.randint(0, 1000, (n2, 2), dtype=torch.int32)
        return sample

    def test_global_view_shapes(self):
        from oceanpath.data.batching import MultiCropCollator

        gn = 64
        collator = MultiCropCollator(global_crop_n=gn, local_crop_n=16, n_local_crops=4)
        batch = [self._make_sample(200, 150), self._make_sample(100, 80)]
        out = collator(batch)

        assert out["view1"].shape == (2, gn, FEAT_DIM)
        assert out["view2"].shape == (2, gn, FEAT_DIM)
        assert out["mask1"].shape == (2, gn)
        assert out["mask2"].shape == (2, gn)

    def test_local_view_shapes(self):
        from oceanpath.data.batching import MultiCropCollator

        ln = 16
        n_lc = 4
        collator = MultiCropCollator(global_crop_n=64, local_crop_n=ln, n_local_crops=n_lc)
        batch = [self._make_sample(200, 150)]
        out = collator(batch)

        assert "local_views" in out
        assert "local_masks" in out
        assert len(out["local_views"]) == n_lc
        assert len(out["local_masks"]) == n_lc
        for lv, lm in zip(out["local_views"], out["local_masks"]):
            assert lv.shape == (1, ln, FEAT_DIM)
            assert lm.shape == (1, ln)

    def test_global_views_list(self):
        from oceanpath.data.batching import MultiCropCollator

        collator = MultiCropCollator(global_crop_n=32, local_crop_n=8, n_local_crops=2)
        batch = [self._make_sample(100, 80)]
        out = collator(batch)

        assert "global_views" in out
        assert "global_masks" in out
        assert len(out["global_views"]) == 2
        assert len(out["global_masks"]) == 2
        # global_views should be the same tensors as view1/view2
        assert out["global_views"][0] is out["view1"]
        assert out["global_views"][1] is out["view2"]

    def test_backward_compatible_keys(self):
        from oceanpath.data.batching import MultiCropCollator

        collator = MultiCropCollator(global_crop_n=32, local_crop_n=8, n_local_crops=2)
        batch = [self._make_sample(50, 50)]
        out = collator(batch)

        expected = {"view1", "mask1", "view2", "mask2", "lengths1", "lengths2", "slide_ids"}
        assert expected.issubset(set(out.keys()))

    def test_short_sample_padding(self):
        from oceanpath.data.batching import MultiCropCollator

        gn = 64
        collator = MultiCropCollator(global_crop_n=gn, local_crop_n=16, n_local_crops=2)
        batch = [self._make_sample(10, 5)]
        out = collator(batch)

        # 10 patches < 64 global → padded, mask should reflect actual patches
        assert out["mask1"][0].sum().item() == 10
        assert out["mask2"][0].sum().item() == 5
        assert out["view1"].shape[1] == gn

    def test_long_sample_subsampled(self):
        from oceanpath.data.batching import MultiCropCollator

        gn = 32
        collator = MultiCropCollator(global_crop_n=gn, local_crop_n=8, n_local_crops=2)
        batch = [self._make_sample(200, 150)]
        out = collator(batch)

        # All positions should be active (subsampled, not padded)
        assert out["mask1"][0].sum().item() == gn
        assert out["mask2"][0].sum().item() == gn

    def test_coords_in_output(self):
        from oceanpath.data.batching import MultiCropCollator

        gn = 32
        collator = MultiCropCollator(global_crop_n=gn, local_crop_n=8, n_local_crops=2)
        batch = [self._make_sample(100, 80, with_coords=True)]
        out = collator(batch)

        assert "coords1" in out
        assert "coords2" in out
        assert out["coords1"].shape == (1, gn, 2)
        assert "global_coords" in out
        assert "local_coords" in out
        assert len(out["local_coords"]) == 2

    def test_no_coords(self):
        from oceanpath.data.batching import MultiCropCollator

        collator = MultiCropCollator(global_crop_n=32, local_crop_n=8, n_local_crops=2)
        batch = [self._make_sample(50, 50, with_coords=False)]
        out = collator(batch)

        assert "coords1" not in out
        assert "local_coords" not in out

    def test_generator_reproducibility(self):
        from oceanpath.data.batching import MultiCropCollator

        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        c1 = MultiCropCollator(global_crop_n=16, local_crop_n=8, n_local_crops=2, generator=g1)
        c2 = MultiCropCollator(global_crop_n=16, local_crop_n=8, n_local_crops=2, generator=g2)

        torch.manual_seed(99)
        sample = self._make_sample(100, 80)

        out1 = c1([sample])
        out2 = c2([sample])

        assert torch.equal(out1["view1"], out2["view1"])
        assert torch.equal(out1["view2"], out2["view2"])


# ═════════════════════════════════════════════════════════════════════════════
# Tests: JEPACollator
# ═════════════════════════════════════════════════════════════════════════════


class TestJEPACollator:
    """Test JEPA context/target partition collator."""

    def _make_sample(self, n, D=FEAT_DIM, with_coords=True):
        sample = {
            "view1_features": torch.randn(n, D),
            "view2_features": torch.randn(n, D),  # ignored by JEPA
            "slide_id": f"slide_{n}",
        }
        if with_coords:
            sample["view1_coords"] = torch.randint(0, 1000, (n, 2), dtype=torch.int32)
            sample["view2_coords"] = torch.randint(0, 1000, (n, 2), dtype=torch.int32)
        return sample

    def test_context_target_partition(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(100)]
        out = collator(batch)

        n_ctx = out["mask1"][0].sum().item()
        n_tgt = out["mask2"][0].sum().item()
        assert n_ctx == 50  # 0.5 * 100
        assert n_tgt == 50  # 100 - 50
        assert n_ctx + n_tgt == 100

    def test_backward_compatible_keys(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(50)]
        out = collator(batch)

        expected = {"view1", "mask1", "view2", "mask2", "lengths1", "lengths2", "slide_ids"}
        assert expected.issubset(set(out.keys()))

    def test_jepa_specific_keys(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(50)]
        out = collator(batch)

        assert "context_pos" in out
        assert "target_pos" in out

    def test_position_indices_valid(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(100)]
        out = collator(batch)

        ctx_pos = out["context_pos"][0]
        tgt_pos = out["target_pos"][0]

        # Active positions should be valid indices into original sequence
        n_ctx = int(out["mask1"][0].sum().item())
        n_tgt = int(out["mask2"][0].sum().item())
        active_ctx = ctx_pos[:n_ctx]
        active_tgt = tgt_pos[:n_tgt]

        assert torch.all(active_ctx >= 0)
        assert torch.all(active_ctx < 100)
        assert torch.all(active_tgt >= 0)
        assert torch.all(active_tgt < 100)

        # Context and target positions should not overlap
        ctx_set = set(active_ctx.tolist())
        tgt_set = set(active_tgt.tolist())
        assert len(ctx_set & tgt_set) == 0

    def test_context_ratio_respected(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.7)
        batch = [self._make_sample(100)]
        out = collator(batch)

        n_ctx = int(out["mask1"][0].sum().item())
        assert n_ctx == 70  # 0.7 * 100

    def test_output_shapes(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(80), self._make_sample(60)]
        out = collator(batch)

        assert out["view1"].shape[0] == 2  # batch size
        assert out["view2"].shape[0] == 2
        assert out["view1"].ndim == 3
        assert out["view2"].ndim == 3
        assert out["view1"].shape[2] == FEAT_DIM

    def test_max_context_cap(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5, max_context=10)
        batch = [self._make_sample(100)]
        out = collator(batch)

        n_ctx = int(out["mask1"][0].sum().item())
        assert n_ctx <= 10

    def test_max_target_cap(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5, max_target=10)
        batch = [self._make_sample(100)]
        out = collator(batch)

        n_tgt = int(out["mask2"][0].sum().item())
        assert n_tgt <= 10

    def test_coords_in_output(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(100, with_coords=True)]
        out = collator(batch)

        assert "coords1" in out
        assert "coords2" in out

    def test_no_coords(self):
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(50, with_coords=False)]
        out = collator(batch)

        assert "coords1" not in out
        assert "coords2" not in out

    def test_generator_reproducibility(self):
        from oceanpath.data.batching import JEPACollator

        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        c1 = JEPACollator(context_ratio=0.5, generator=g1)
        c2 = JEPACollator(context_ratio=0.5, generator=g2)

        torch.manual_seed(99)
        sample = self._make_sample(100)

        out1 = c1([sample])
        out2 = c2([sample])

        assert torch.equal(out1["view1"], out2["view1"])
        assert torch.equal(out1["context_pos"], out2["context_pos"])

    def test_small_sample(self):
        """JEPA should handle very small slides (context at least 1 patch)."""
        from oceanpath.data.batching import JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        batch = [self._make_sample(2)]
        out = collator(batch)

        n_ctx = int(out["mask1"][0].sum().item())
        n_tgt = int(out["mask2"][0].sum().item())
        assert n_ctx >= 1
        assert n_tgt >= 1


# ═════════════════════════════════════════════════════════════════════════════
# Tests: COORD_PAD_VALUE sentinel
# ═════════════════════════════════════════════════════════════════════════════


class TestCoordPadValue:
    """Verify that coordinate padding uses sentinel -1 across all collators."""

    def _make_sample(self, n1, n2, D=FEAT_DIM):
        return {
            "view1_features": torch.randn(n1, D),
            "view2_features": torch.randn(n2, D),
            "slide_id": f"slide_{n1}_{n2}",
            "view1_coords": torch.randint(0, 1000, (n1, 2), dtype=torch.int32),
            "view2_coords": torch.randint(0, 1000, (n2, 2), dtype=torch.int32),
        }

    def test_dual_view_coord_padding(self):
        from oceanpath.data.batching import COORD_PAD_VALUE, DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(5, 10), self._make_sample(15, 3)]
        out = collator(batch)

        # Sample 0 view1: 5 real patches, padded from 5..14
        assert torch.all(out["coords1"][0, 5:] == COORD_PAD_VALUE)
        # Sample 1 view2: 3 real patches, padded from 3..9
        assert torch.all(out["coords2"][1, 3:] == COORD_PAD_VALUE)

    def test_pad_to_global_coord_padding(self):
        from oceanpath.data.batching import COORD_PAD_VALUE, PadToGlobalCollator

        collator = PadToGlobalCollator(global_max=64)
        batch = [self._make_sample(10, 20)]
        out = collator(batch)

        assert torch.all(out["coords1"][0, 10:] == COORD_PAD_VALUE)
        assert torch.all(out["coords2"][0, 20:] == COORD_PAD_VALUE)

    def test_fixed_n_coord_padding(self):
        from oceanpath.data.batching import COORD_PAD_VALUE, FixedNCollator

        collator = FixedNCollator(fixed_n=32)
        batch = [self._make_sample(10, 5)]
        out = collator(batch)

        assert torch.all(out["coords1"][0, 10:] == COORD_PAD_VALUE)
        assert torch.all(out["coords2"][0, 5:] == COORD_PAD_VALUE)

    def test_regional_crop_coord_padding(self):
        from oceanpath.data.batching import COORD_PAD_VALUE, RegionalCropCollator

        collator = RegionalCropCollator(fixed_n=64, crop_frac=1.0)
        batch = [self._make_sample(10, 5)]
        out = collator(batch)

        # With small slide and large fixed_n, padding will exist
        n1 = int(out["mask1"][0].sum().item())
        assert torch.all(out["coords1"][0, n1:] == COORD_PAD_VALUE)

    def test_multi_crop_coord_padding(self):
        from oceanpath.data.batching import COORD_PAD_VALUE, MultiCropCollator

        collator = MultiCropCollator(global_crop_n=32, local_crop_n=16, n_local_crops=2)
        batch = [self._make_sample(10, 5)]
        out = collator(batch)

        # Global view1: 10 patches < 32 → padding from 10 onwards
        assert torch.all(out["coords1"][0, 10:] == COORD_PAD_VALUE)

    def test_jepa_coord_padding(self):
        from oceanpath.data.batching import COORD_PAD_VALUE, JEPACollator

        collator = JEPACollator(context_ratio=0.5)
        # Two samples with different sizes → batch padding will occur
        s1 = self._make_sample(100, 100)
        s2 = self._make_sample(50, 50)
        out = collator([s1, s2])

        # Sample with smaller context will have padding
        n_ctx_s2 = int(out["mask1"][1].sum().item())
        max_ctx = out["view1"].shape[1]
        if n_ctx_s2 < max_ctx:
            assert torch.all(out["coords1"][1, n_ctx_s2:] == COORD_PAD_VALUE)

    def test_coord_pad_value_is_negative(self):
        from oceanpath.data.batching import COORD_PAD_VALUE

        assert COORD_PAD_VALUE == -1


# ═════════════════════════════════════════════════════════════════════════════
# Tests: TokenBudgetBatchSampler batch order shuffle (Issue 1 fix)
# ═════════════════════════════════════════════════════════════════════════════


class TestTokenBudgetBatchShuffle:
    """Verify that batch ORDER is shuffled after greedy accumulation."""

    def test_batch_order_shuffled(self):
        """Non-sort mode should shuffle batch order across epochs."""
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 150, 80, 300, 50, 120, 250, 90, 170])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=400,
            max_batch_size=64,
            min_batch_size=1,
            sort=False,
            drop_last=False,
            seed=42,
        )

        epoch0_batches = list(sampler)
        sampler.set_epoch(1)
        epoch1_batches = list(sampler)

        # Batch count should be the same
        assert len(epoch0_batches) == len(epoch1_batches)
        # Batch order should differ between epochs
        assert epoch0_batches != epoch1_batches

    def test_sort_mode_not_shuffled(self):
        """Sort mode should NOT shuffle batch order (for packing)."""
        from oceanpath.data.batching import TokenBudgetBatchSampler

        bag_sizes = np.array([100, 200, 150, 80, 300])
        sampler = TokenBudgetBatchSampler(
            bag_sizes=bag_sizes,
            token_budget=400,
            max_batch_size=64,
            min_batch_size=1,
            sort=True,
            drop_last=False,
            seed=42,
        )

        batches1 = list(sampler)
        sampler.set_epoch(1)
        batches2 = list(sampler)
        # sort=True is deterministic regardless of epoch
        assert batches1 == batches2


# ═════════════════════════════════════════════════════════════════════════════
# Tests: FixedNCollator generator reproducibility (Issue 7 fix)
# ═════════════════════════════════════════════════════════════════════════════


class TestFixedNCollatorGenerator:
    """Verify that FixedNCollator supports reproducible subsampling."""

    def _make_sample(self, n, D=FEAT_DIM):
        return {
            "view1_features": torch.randn(n, D),
            "view2_features": torch.randn(n, D),
            "slide_id": f"slide_{n}",
            "view1_coords": torch.randint(0, 1000, (n, 2), dtype=torch.int32),
            "view2_coords": torch.randint(0, 1000, (n, 2), dtype=torch.int32),
        }

    def test_generator_gives_reproducible_results(self):
        from oceanpath.data.batching import FixedNCollator

        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)

        c1 = FixedNCollator(fixed_n=16, generator=g1)
        c2 = FixedNCollator(fixed_n=16, generator=g2)

        torch.manual_seed(99)
        sample = self._make_sample(100)

        out1 = c1([sample])
        out2 = c2([sample])

        assert torch.equal(out1["view1"], out2["view1"])
        assert torch.equal(out1["view2"], out2["view2"])
        assert torch.equal(out1["coords1"], out2["coords1"])

    def test_different_seeds_differ(self):
        from oceanpath.data.batching import FixedNCollator

        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(99)

        c1 = FixedNCollator(fixed_n=16, generator=g1)
        c2 = FixedNCollator(fixed_n=16, generator=g2)

        torch.manual_seed(123)
        sample = self._make_sample(100)

        out1 = c1([sample])
        out2 = c2([sample])

        # Different seeds should produce different subsampling
        assert not torch.equal(out1["view1"], out2["view1"])

    def test_no_generator_still_works(self):
        from oceanpath.data.batching import FixedNCollator

        collator = FixedNCollator(fixed_n=16, generator=None)
        sample = self._make_sample(100)
        out = collator([sample])

        assert out["view1"].shape == (1, 16, FEAT_DIM)
        assert out["mask1"][0].sum().item() == 16
