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

# We patch validate_mmap_dir at the module level where it's imported
VALIDATE_PATCH_DATASET = "oceanpath.data.pretrain_dataset.validate_mmap_dir"
VALIDATE_PATCH_DATAMODULE = "oceanpath.data.pretrain_datamodule.validate_mmap_dir"


# ═════════════════════════════════════════════════════════════════════════════
# Tests: PretrainDataset (real mmap data)
# ═════════════════════════════════════════════════════════════════════════════


class TestPretrainDatasetReal:
    """Tests using real mmap files on disk."""

    def _make_dataset(self, mmap_dir, slide_ids=None, max_inst=None):
        from oceanpath.data.pretrain_dataset import PretrainDataset
        from oceanpath.ssl.augmentation import FeatureAugmentor

        with patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir):
            aug = FeatureAugmentor(
                subsample_frac=(0.8, 1.0),
                instance_dropout=0.0,
                feature_noise_std=0.0,
                feature_dropout=0.0,
            )
            ds = PretrainDataset(
                mmap_dir=str(mmap_dir),
                slide_ids=slide_ids,
                augmentor=aug,
                dataset_max_instances=max_inst,
            )
        return ds

    def test_len_all_slides(self, small_mmap):
        mmap_dir, ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        assert len(ds) == len(ids)

    def test_len_filtered_slides(self, small_mmap):
        mmap_dir, ids, _counts = small_mmap
        subset = ids[:3]
        ds = self._make_dataset(mmap_dir, slide_ids=subset)
        assert len(ds) == 3

    def test_getitem_returns_correct_keys(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        sample = ds[0]

        assert "view1_features" in sample
        assert "view2_features" in sample
        assert "view1_coords" in sample
        assert "view2_coords" in sample
        assert "slide_id" in sample

    def test_getitem_feature_shape(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        sample = ds[0]

        v1 = sample["view1_features"]
        v2 = sample["view2_features"]
        assert v1.ndim == 2
        assert v2.ndim == 2
        assert v1.shape[1] == FEAT_DIM
        assert v2.shape[1] == FEAT_DIM

    def test_getitem_coord_shape(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        sample = ds[0]

        c1 = sample["view1_coords"]
        c2 = sample["view2_coords"]
        assert c1.shape == (sample["view1_features"].shape[0], COORD_DIM)
        assert c2.shape == (sample["view2_features"].shape[0], COORD_DIM)

    def test_getitem_tensor_dtypes(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        sample = ds[0]

        assert sample["view1_features"].dtype == torch.float32
        assert sample["view2_features"].dtype == torch.float32
        assert sample["view1_coords"].dtype == torch.int32
        assert sample["view2_coords"].dtype == torch.int32

    def test_slide_id_matches(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)

        for i in range(len(ds)):
            sample = ds[i]
            assert sample["slide_id"] == ds.slide_ids[i]

    def test_two_views_differ(self, small_mmap):
        """Dual views should differ (different augmentation randomness)."""
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        sample = ds[1]  # slide_b has 120 patches — enough room to differ

        v1 = sample["view1_features"]
        v2 = sample["view2_features"]
        # Views may have different lengths or different content
        sizes_differ = v1.shape[0] != v2.shape[0]
        if not sizes_differ:
            content_differs = not torch.allclose(v1, v2, atol=1e-5)
        else:
            content_differs = True
        assert sizes_differ or content_differs

    def test_dataset_max_instances_caps(self, small_mmap):
        """Pre-cap should limit patches before augmentor runs."""
        mmap_dir, _ids, _counts = small_mmap
        cap = 25
        ds = self._make_dataset(mmap_dir, max_inst=cap)

        # slide_b has 120 patches; after cap + subsample(0.8-1.0), max ≤ cap
        sample = ds[1]
        assert sample["view1_features"].shape[0] <= cap
        assert sample["view2_features"].shape[0] <= cap

    def test_dataset_max_instances_no_effect_on_small(self, tiny_mmap):
        """Cap larger than slide should not affect output."""
        mmap_dir, _ids, _counts = tiny_mmap
        ds_no_cap = self._make_dataset(mmap_dir, max_inst=None)
        ds_capped = self._make_dataset(mmap_dir, max_inst=9999)
        # Both should load same number of slides
        assert len(ds_no_cap) == len(ds_capped)

    def test_missing_slide_ids_skipped(self, small_mmap):
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir, slide_ids=["slide_a", "nonexistent", "slide_b"])
        assert len(ds) == 2
        assert set(ds.slide_ids) == {"slide_a", "slide_b"}

    def test_get_bag_sizes(self, small_mmap):
        mmap_dir, _ids, counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        sizes = ds.get_bag_sizes()
        np.testing.assert_array_equal(sizes, np.array(counts))

    def test_all_slides_iterable(self, small_mmap):
        """Every index in range should return a valid sample."""
        mmap_dir, _ids, _counts = small_mmap
        ds = self._make_dataset(mmap_dir)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["view1_features"].shape[0] > 0
            assert sample["view2_features"].shape[0] > 0


# ═════════════════════════════════════════════════════════════════════════════
# Tests: PretrainDataset with augmentation variations
# ═════════════════════════════════════════════════════════════════════════════


class TestPretrainDatasetAugmentations:
    """Test dataset with various augmentor configs to catch integration bugs."""

    def _make_dataset_with_aug(self, mmap_dir, aug):
        from oceanpath.data.pretrain_dataset import PretrainDataset

        with patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir):
            ds = PretrainDataset(
                mmap_dir=str(mmap_dir),
                augmentor=aug,
                dataset_max_instances=200,
            )
        return ds

    def test_full_coords_aware_augmentor(self, small_mmap):
        """Full pipeline with all coords-aware transforms enabled."""
        from oceanpath.ssl.augmentation import build_augmentor

        mmap_dir, _ids, _counts = small_mmap
        aug = build_augmentor({}, coords_aware=True)
        ds = self._make_dataset_with_aug(mmap_dir, aug)

        sample = ds[0]
        assert sample["view1_features"].shape[1] == FEAT_DIM
        assert sample["view2_features"].shape[1] == FEAT_DIM

    def test_feature_only_augmentor(self, small_mmap):
        """Coords-unaware mode (original baseline)."""
        from oceanpath.ssl.augmentation import build_augmentor

        mmap_dir, _ids, _counts = small_mmap
        aug = build_augmentor({}, coords_aware=False)
        ds = self._make_dataset_with_aug(mmap_dir, aug)

        sample = ds[0]
        assert sample["view1_features"].shape[1] == FEAT_DIM

    def test_heavy_augmentation_doesnt_crash(self, small_mmap):
        """Aggressive settings: high dropout, noise, smoothing."""
        from oceanpath.ssl.augmentation import (
            CoordAffine,
            FeatureAugmentor,
            LocalFeatureSmooth,
            SpatialCrop,
            SpatialFeatureInterpolation,
            SpatialRegionDrop,
            TissueRegionMixup,
        )

        mmap_dir, _ids, _counts = small_mmap
        aug = FeatureAugmentor(
            subsample_frac=(0.3, 0.5),
            instance_dropout=0.3,
            feature_noise_std=0.1,
            feature_dropout=0.2,
            spatial_crop=SpatialCrop(crop_frac=(0.3, 0.5)),
            spatial_region_drop=SpatialRegionDrop(n_regions=3),
            local_smooth=LocalFeatureSmooth(k_neighbors=3, alpha_range=(0.1, 0.5)),
            spatial_interpolation=SpatialFeatureInterpolation(t_range=(0.1, 0.4)),
            region_mixup=TissueRegionMixup(alpha_range=(0.1, 0.4)),
            coord_affine=CoordAffine(),
            use_spatial_crop=True,
            spatial_crop_prob=0.8,
        )
        ds = self._make_dataset_with_aug(mmap_dir, aug)

        # Should not crash on any slide
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["view1_features"].shape[0] >= 1
            assert sample["view2_features"].shape[0] >= 1

    def test_no_augmentation(self, tiny_mmap):
        """Minimal augmentor (identity-ish): patches should be preserved."""
        from oceanpath.ssl.augmentation import FeatureAugmentor

        mmap_dir, _ids, counts = tiny_mmap
        aug = FeatureAugmentor(
            subsample_frac=(1.0, 1.0),
            instance_dropout=0.0,
            feature_noise_std=0.0,
            feature_dropout=0.0,
        )
        ds = self._make_dataset_with_aug(mmap_dir, aug)
        sample = ds[0]
        # With frac=1.0 and no dropout, view length should equal raw patch count
        assert sample["view1_features"].shape[0] == counts[0]


# ═════════════════════════════════════════════════════════════════════════════
# Tests: DualViewCollator (synthetic tensor data)
# ═════════════════════════════════════════════════════════════════════════════


class TestDualViewCollator:
    """Test collation logic with hand-crafted fake samples."""

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

    def test_basic_collation_shape(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(20, 30), self._make_sample(40, 10)]
        out = collator(batch)

        assert out["view1"].shape == (2, 40, FEAT_DIM)  # max of 20, 40
        assert out["view2"].shape == (2, 30, FEAT_DIM)  # max of 30, 10
        assert out["mask1"].shape == (2, 40)
        assert out["mask2"].shape == (2, 30)

    def test_padding_is_zero(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(5, 10), self._make_sample(15, 3)]
        out = collator(batch)

        # First sample, view1 has 5 patches → positions 5..14 should be zero
        assert torch.all(out["view1"][0, 5:] == 0)
        assert torch.all(out["mask1"][0, 5:] == 0)
        assert torch.all(out["mask1"][0, :5] == 1)

    def test_mask_correctness(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(8, 12), self._make_sample(3, 20)]
        out = collator(batch)

        # Check mask sums match actual lengths
        assert out["mask1"][0].sum().item() == 8
        assert out["mask1"][1].sum().item() == 3
        assert out["mask2"][0].sum().item() == 12
        assert out["mask2"][1].sum().item() == 20

    def test_max_instances_caps(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        cap = 10
        collator = DualViewCollator(max_instances=cap)
        batch = [self._make_sample(50, 100)]
        out = collator(batch)

        assert out["view1"].shape[1] <= cap
        assert out["view2"].shape[1] <= cap
        assert out["mask1"][0].sum().item() == cap
        assert out["mask2"][0].sum().item() == cap

    def test_lengths_tensor(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(7, 13), self._make_sample(22, 5)]
        out = collator(batch)

        assert torch.equal(out["lengths1"], torch.tensor([7, 22], dtype=torch.int32))
        assert torch.equal(out["lengths2"], torch.tensor([13, 5], dtype=torch.int32))

    def test_slide_ids_preserved(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(5, 5), self._make_sample(10, 10)]
        out = collator(batch)

        assert len(out["slide_ids"]) == 2
        assert all(isinstance(s, str) for s in out["slide_ids"])

    def test_coords_padded(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [
            self._make_sample(5, 8, with_coords=True),
            self._make_sample(12, 3, with_coords=True),
        ]
        out = collator(batch)

        assert "coords1" in out
        assert "coords2" in out
        assert out["coords1"].shape == (2, 12, 2)
        assert out["coords2"].shape == (2, 8, 2)
        # Padding region should be zero
        assert torch.all(out["coords1"][0, 5:] == 0)

    def test_no_coords(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(5, 5, with_coords=False)]
        out = collator(batch)

        assert "coords1" not in out
        assert "coords2" not in out

    def test_single_sample_batch(self):
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(25, 30)]
        out = collator(batch)

        assert out["view1"].shape[0] == 1
        assert out["mask1"][0].sum().item() == 25

    def test_equal_length_no_padding(self):
        """When all samples have the same length, no padding needed."""
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        batch = [self._make_sample(10, 10) for _ in range(4)]
        out = collator(batch)

        assert out["view1"].shape == (4, 10, FEAT_DIM)
        # All mask entries should be 1
        assert torch.all(out["mask1"] == 1)


# ═════════════════════════════════════════════════════════════════════════════
# Tests: PretrainDataModule (real mmap + CSV filtering)
# ═════════════════════════════════════════════════════════════════════════════


class TestPretrainDataModule:
    """Integration tests for the full DataModule lifecycle."""

    def _make_dm(self, mmap_dir, csv_path=None, **kwargs):
        from oceanpath.data.pretrain_datamodule import PretrainDataModule

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


# ═════════════════════════════════════════════════════════════════════════════
# Tests: Edge cases and error handling
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_zero_patch_slide_skipped(self, tmp_path):
        """Slides with 0 patches should be excluded."""
        from oceanpath.data.pretrain_dataset import PretrainDataset
        from oceanpath.ssl.augmentation import FeatureAugmentor

        ids = ["good_slide", "empty_slide"]
        counts = [50, 0]
        mmap_dir = _build_mmap_dir(tmp_path, ids, counts)

        with patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir):
            ds = PretrainDataset(
                mmap_dir=str(mmap_dir),
                augmentor=FeatureAugmentor(),
            )
        assert len(ds) == 1
        assert ds.slide_ids == ["good_slide"]

    def test_all_slides_missing_yields_empty_dataset(self, small_mmap):
        from oceanpath.data.pretrain_dataset import PretrainDataset
        from oceanpath.ssl.augmentation import FeatureAugmentor

        mmap_dir, _ids, _counts = small_mmap
        with patch(VALIDATE_PATCH_DATASET, side_effect=_mock_validate_mmap_dir):
            ds = PretrainDataset(
                mmap_dir=str(mmap_dir),
                slide_ids=["nonexistent_1", "nonexistent_2"],
                augmentor=FeatureAugmentor(),
            )
        assert len(ds) == 0

    def test_collator_handles_length_one_views(self):
        """Views with a single patch should collate correctly."""
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        sample = {
            "view1_features": torch.randn(1, FEAT_DIM),
            "view2_features": torch.randn(1, FEAT_DIM),
            "slide_id": "tiny",
            "view1_coords": torch.zeros(1, 2, dtype=torch.int32),
            "view2_coords": torch.zeros(1, 2, dtype=torch.int32),
        }
        out = collator([sample])
        assert out["view1"].shape == (1, 1, FEAT_DIM)
        assert out["mask1"].sum().item() == 1

    def test_collator_mixed_coord_presence(self):
        """If first sample has coords, all should — but test graceful handling."""
        from oceanpath.data.pretrain_datamodule import DualViewCollator

        collator = DualViewCollator(max_instances=4096)
        s1 = {
            "view1_features": torch.randn(5, FEAT_DIM),
            "view2_features": torch.randn(5, FEAT_DIM),
            "slide_id": "a",
            "view1_coords": torch.zeros(5, 2, dtype=torch.int32),
            "view2_coords": torch.zeros(5, 2, dtype=torch.int32),
        }
        s2 = {
            "view1_features": torch.randn(3, FEAT_DIM),
            "view2_features": torch.randn(3, FEAT_DIM),
            "slide_id": "b",
            "view1_coords": torch.zeros(3, 2, dtype=torch.int32),
            "view2_coords": torch.zeros(3, 2, dtype=torch.int32),
        }
        out = collator([s1, s2])
        assert "coords1" in out


# ═════════════════════════════════════════════════════════════════════════════
# Tests: DataModule internal logic with mocked I/O
# ═════════════════════════════════════════════════════════════════════════════


class TestDataModuleSplitLogic:
    """Test _load_slide_ids and _split_slide_ids with mocked filesystem."""

    def test_split_fractions(self):
        """Validate that val_frac produces the right number of val slides."""
        from oceanpath.data.pretrain_datamodule import PretrainDataModule

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
        from oceanpath.data.pretrain_datamodule import PretrainDataModule

        with patch(VALIDATE_PATCH_DATAMODULE, return_value={"feat_dim": 16, "coord_dim": 2}):
            dm = PretrainDataModule.__new__(PretrainDataModule)
            dm.val_frac = 0.01
            dm.seed = 42

        ids = ["a", "b", "c"]
        _train, val = dm._split_slide_ids(ids)
        assert len(val) >= 1
