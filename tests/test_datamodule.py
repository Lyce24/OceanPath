"""
Tests for the parts of ``oceanpath.data.datamodule`` that tests/test_data.py
leaves out: eval-time subsampling in SimpleMILCollator, the length-bucket and
distributed samplers, refit and test-stage setup, multi-crop evaluation and
DataLoader workers. Slides come from conftest's ``build_feature_store``, whose
feature column 0 is the patch's row index, so every test can tell exactly
which patches a collator or dataset kept.

Covers:
  * SimpleMILCollator sampling modes (contiguous / random / spatial_stratified):
    per-slide determinism, features and coords kept aligned, the coord-less
    fallback, an invalid mode
  * the collator's spatial-stratified sampler: even spread over occupied grid
    cells, top-up when cells run dry, small bags
  * LengthBucketBatchSampler: every slide once per epoch, similar lengths in a
    batch, len() equal to the number of batches it yields, seeded order
  * DistributedSamplerWrapper: rank shards partition one draw of the inner
    sampler, which is redrawn every epoch
  * MILDataModule: refit mode, holdout test stage, multi-crop crop seeds and the
    eval cache, the length-bucketed train loader (and its DDP guard), workers
    drawing independent subsamples, constructor / manifest validation
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from oceanpath.data import datamodule as datamodule_module
from oceanpath.data.datamodule import (
    DistributedSamplerWrapper,
    LengthBucketBatchSampler,
    MILDataModule,
    SimpleMILCollator,
    _spatial_stratified_indices,
)

FEAT_DIM = 8
PATCH_COUNTS = [40, 40, 25, 31, 18, 22, 35, 12]


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def store(build_feature_store):
    return build_feature_store(PATCH_COUNTS, feat_dim=FEAT_DIM, seed=11)


def write_manifest(root, store, splits: dict, labels=None):
    """Manifest CSV + splits.parquet; ``splits`` maps column name -> per-slide values."""
    labels = labels or [i % 2 for i in range(len(store.slide_ids))]
    csv_path = root / "manifest.csv"
    pd.DataFrame({"filename": [f"{s}.svs" for s in store.slide_ids], "label": labels}).to_csv(
        csv_path, index=False
    )
    splits_dir = root / "splits"
    splits_dir.mkdir()
    pd.DataFrame({"slide_id": store.slide_ids, **splits}).to_parquet(
        splits_dir / "splits.parquet", index=False
    )
    return csv_path, splits_dir


def make_dm(store, csv_path, splits_dir, **kwargs) -> MILDataModule:
    defaults = {
        "mmap_dir": str(store.mmap_dir),
        "splits_dir": str(splits_dir),
        "csv_path": str(csv_path),
        "scheme": "kfold",
        "fold": 0,
        "batch_size": 2,
        "max_instances": 64,
        "num_workers": 0,
        "class_weighted_sampling": False,
        "verify_splits": False,
        "use_preallocated_collator": False,
    }
    defaults.update(kwargs)
    return MILDataModule(**defaults)


# folds: train = folds 1-2, val = fold 0, test = -1 (for fold 0)
KFOLD = [1, 1, 0, 0, 0, 2, -1, -1]


@pytest.fixture
def kfold_env(store, tmp_path):
    csv_path, splits_dir = write_manifest(tmp_path, store, {"fold": KFOLD})
    ids = store.slide_ids
    return SimpleNamespace(
        csv=csv_path,
        splits_dir=splits_dir,
        train=[s for s, f in zip(ids, KFOLD, strict=True) if f > 0],
        val=[s for s, f in zip(ids, KFOLD, strict=True) if f == 0],
        test=[s for s, f in zip(ids, KFOLD, strict=True) if f == -1],
    )


def store_sample(store, slide_id, with_coords=True):
    sample = {
        "features": torch.from_numpy(store.features[slide_id]),
        "label": 0,
        "slide_id": slide_id,
    }
    if with_coords:
        sample["coords"] = torch.from_numpy(store.coords[slide_id])
    return sample


def kept_rows(batch, i=0) -> list[int]:
    """Row indices (feature column 0) of the patches kept for sample i."""
    n = int(batch["lengths"][i])
    return batch["features"][i, :n, 0].long().tolist()


# ═════════════════════════════════════════════════════════════════════════════
# SimpleMILCollator sampling modes
# ═════════════════════════════════════════════════════════════════════════════


class TestCollatorSampling:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="sampling_mode"):
            SimpleMILCollator(max_instances=4, sampling_mode="first_n")

    def test_contiguous_keeps_the_first_rows(self, store):
        batch = SimpleMILCollator(max_instances=5)([store_sample(store, "slide_02")])
        assert kept_rows(batch) == [0, 1, 2, 3, 4]

    def test_random_mode_is_a_reproducible_subset_per_slide(self, store):
        collator = SimpleMILCollator(max_instances=6, sampling_mode="random", seed=42)
        sample = store_sample(store, "slide_00")
        rows = kept_rows(collator([sample]))
        assert rows == kept_rows(collator([sample]))  # same slide, same crop every epoch
        assert len(set(rows)) == 6
        assert set(rows) <= set(range(PATCH_COUNTS[0]))
        assert rows != list(range(6))
        # same features under another slide id, or another seed: another crop
        renamed = dict(sample, slide_id="slide_99")
        assert kept_rows(collator([renamed])) != rows
        reseeded = SimpleMILCollator(max_instances=6, sampling_mode="random", seed=7)
        assert kept_rows(reseeded([sample])) != rows

    def test_random_mode_keeps_coords_aligned_with_features(self, store):
        collator = SimpleMILCollator(max_instances=6, sampling_mode="random")
        batch = collator([store_sample(store, "slide_03")])
        rows = kept_rows(batch)
        np.testing.assert_array_equal(batch["coords"][0].numpy(), store.coords["slide_03"][rows])

    def test_spatial_mode_without_coords_falls_back_to_random(self, store):
        sample = store_sample(store, "slide_06", with_coords=False)
        spatial = SimpleMILCollator(max_instances=5, sampling_mode="spatial_stratified")
        random = SimpleMILCollator(max_instances=5, sampling_mode="random")
        assert kept_rows(spatial([sample])) == kept_rows(random([sample]))

    def test_spatial_mode_samples_every_quadrant(self):
        """30 patches in one corner, 2 in each of the other three: a cap of 8 takes 2 per corner."""
        corners = [(0, 0)] * 30 + [(0, 1000)] * 2 + [(1000, 0)] * 2 + [(1000, 1000)] * 2
        coords = torch.tensor(corners, dtype=torch.int32) + torch.arange(36)[:, None] % 5
        features = torch.arange(36, dtype=torch.float32)[:, None].repeat(1, FEAT_DIM)
        sample = {"features": features, "coords": coords, "label": 1, "slide_id": "s"}
        collator = SimpleMILCollator(
            max_instances=8, sampling_mode="spatial_stratified", spatial_grid_size=2
        )
        rows = kept_rows(collator([sample]))
        assert len(set(rows)) == 8
        corner_of = [0] * 30 + [1, 1, 2, 2, 3, 3]
        assert sorted(np.bincount([corner_of[r] for r in rows], minlength=4)) == [2, 2, 2, 2]


class TestSpatialStratifiedIndices:
    def test_small_bag_is_returned_whole(self):
        idx = _spatial_stratified_indices(torch.zeros(5, 2), 8, torch.Generator())
        assert idx.tolist() == [0, 1, 2, 3, 4]

    def test_tops_up_from_the_remaining_patches_when_cells_run_dry(self):
        # three occupied cells holding 1, 1 and 10 patches; 9 wanted, 3 per cell
        coords = torch.tensor([[0, 0], [0, 100]] + [[100, 100]] * 10, dtype=torch.int32)
        idx = _spatial_stratified_indices(coords, 9, torch.Generator().manual_seed(0), 2)
        assert len(idx) == len(set(idx.tolist())) == 9
        assert {0, 1} <= set(idx.tolist())

    def test_degenerate_coordinates_still_give_distinct_patches(self):
        idx = _spatial_stratified_indices(torch.ones(20, 2), 7, torch.Generator().manual_seed(1))
        assert len(set(idx.tolist())) == 7
        assert max(idx.tolist()) < 20


# ═════════════════════════════════════════════════════════════════════════════
# LengthBucketBatchSampler
# ═════════════════════════════════════════════════════════════════════════════


class TestLengthBucketBatchSampler:
    def test_every_slide_appears_once_per_epoch(self):
        lengths = np.random.default_rng(0).integers(10, 500, size=23)
        sampler = LengthBucketBatchSampler(lengths, batch_size=4, bucket_size=8, seed=0)
        for _ in range(2):
            flat = [i for batch in sampler for i in batch]
            assert sorted(flat) == list(range(23))

    def test_a_batch_holds_slides_of_similar_length(self):
        lengths = np.random.default_rng(1).permutation(40)  # length == rank
        sampler = LengthBucketBatchSampler(lengths, batch_size=4, bucket_size=8, seed=0)
        for batch in sampler:
            spread = lengths[batch].max() - lengths[batch].min()
            assert spread < 8  # all from one bucket of 8 consecutive lengths

    @pytest.mark.parametrize(
        "n, batch_size, bucket_size, drop_last",
        [
            (10, 3, 4, False),
            (10, 3, 4, True),
            (100, 32, 64, False),
            (23, 4, 8, True),
            (5, 8, 2, False),
        ],
    )
    def test_len_is_the_number_of_batches_yielded(self, n, batch_size, bucket_size, drop_last):
        """Lightning ends the epoch after len(batch_sampler) batches; a short len drops slides."""
        sampler = LengthBucketBatchSampler(
            np.arange(n), batch_size=batch_size, bucket_size=bucket_size, drop_last=drop_last
        )
        batches = list(sampler)
        assert len(sampler) == len(batches)
        if drop_last:
            assert all(len(b) == batch_size for b in batches)

    def test_order_is_reproducible_for_a_seed(self):
        lengths = np.arange(30)

        def batches(seed):
            return list(LengthBucketBatchSampler(lengths, 4, bucket_size=8, seed=seed))

        assert batches(7) == batches(7)
        assert batches(7) != batches(8)


# ═════════════════════════════════════════════════════════════════════════════
# DistributedSamplerWrapper
# ═════════════════════════════════════════════════════════════════════════════


def weighted_sampler(seed: int) -> WeightedRandomSampler:
    weights = [1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 3.0, 1.0]
    return WeightedRandomSampler(
        weights, num_samples=10, replacement=True, generator=torch.Generator().manual_seed(seed)
    )


class TestDistributedSamplerWrapper:
    def test_rank_shards_partition_one_draw_of_the_inner_sampler(self):
        """Each DDP rank holds an identically seeded copy of the inner sampler."""
        full_draw = list(weighted_sampler(seed=3))
        shards = [
            list(DistributedSamplerWrapper(weighted_sampler(seed=3), num_replicas=2, rank=r))
            for r in range(2)
        ]
        assert [len(s) for s in shards] == [5, 5]
        assert shards[0] == full_draw[0::2]
        assert shards[1] == full_draw[1::2]

    def test_a_stochastic_inner_sampler_is_redrawn_every_epoch(self):
        wrapper = DistributedSamplerWrapper(weighted_sampler(seed=0), num_replicas=1, rank=0)
        assert len(wrapper) == 10
        assert list(wrapper) != list(wrapper)


# ═════════════════════════════════════════════════════════════════════════════
# MILDataModule
# ═════════════════════════════════════════════════════════════════════════════


class TestDataModuleSetup:
    def test_refit_mode_trains_on_train_and_val_without_validation(self, store, kfold_env):
        dm = make_dm(store, kfold_env.csv, kfold_env.splits_dir, refit_mode=True)
        dm.setup(stage="fit")
        assert sorted(dm.train_dataset.slide_ids) == sorted(kfold_env.train + kfold_env.val)
        assert dm.val_dataset is None
        assert dm.val_dataloader() is None

    def test_holdout_test_stage_builds_the_test_set(self, store, tmp_path):
        split = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
        csv_path, splits_dir = write_manifest(tmp_path, store, {"split": split})
        dm = make_dm(store, csv_path, splits_dir, scheme="holdout")
        dm.setup(stage="fit")
        with pytest.raises(RuntimeError, match="No test set"):
            dm.test_dataloader()
        dm.setup(stage="test")
        seen = [sid for batch in dm.test_dataloader() for sid in batch["slide_ids"]]
        assert seen == store.slide_ids[6:]

    def test_multi_crop_eval_turns_the_eval_cache_off(self, store, kfold_env):
        """A cached crop would hide the next crop seed."""
        cached = make_dm(store, kfold_env.csv, kfold_env.splits_dir, cache_size_mb=1)
        multi = make_dm(store, kfold_env.csv, kfold_env.splits_dir, cache_size_mb=1, eval_n_crops=3)
        cached.setup(stage="fit")
        multi.setup(stage="fit")
        assert cached.val_dataset._cache is not None
        assert multi.val_dataset._cache is None

    def test_a_new_crop_seed_gives_a_new_crop_and_empties_the_cache(self, store, kfold_env):
        dm = make_dm(
            store, kfold_env.csv, kfold_env.splits_dir, dataset_max_instances=4, cache_size_mb=1
        )
        dm.setup(stage="fit")
        ds = dm.val_dataset
        first = ds[0]["features"][:, 0].tolist()
        assert ds[0]["features"][:, 0].tolist() == first  # served from the cache
        dm.set_eval_crop_seed(ds, seed=1000)
        assert len(ds._cache) == 0
        assert ds._cache_current_bytes == 0
        second = ds[0]["features"][:, 0].tolist()
        assert second != first
        assert len(second) == 4

    def test_length_bucketed_train_loader_batches_similar_bags(self, store, kfold_env):
        dm = make_dm(
            store, kfold_env.csv, kfold_env.splits_dir, length_bucket=True, length_bucket_size=2
        )
        dm.setup(stage="fit")
        loader = dm.train_dataloader()
        assert isinstance(loader.batch_sampler, LengthBucketBatchSampler)
        batches = [batch["slide_ids"] for batch in loader]
        assert sorted(s for b in batches for s in b) == sorted(kfold_env.train)
        assert len(batches) == len(loader)

    def test_length_bucketing_is_refused_under_ddp(self, store, kfold_env, monkeypatch):
        monkeypatch.setattr(datamodule_module, "_ddp_active", lambda: True)
        dm = make_dm(store, kfold_env.csv, kfold_env.splits_dir, length_bucket=True)
        dm.setup(stage="fit")
        with pytest.raises(NotImplementedError, match="length_bucket"):
            dm.train_dataloader()

    def test_dataloader_workers_draw_independent_subsamples(self, store, tmp_path):
        """Forked workers inherit one numpy RNG; the worker init must reseed it."""
        folds = [1, 1, 0, -1, -1, -1, -1, -1]  # train = the two 40-patch slides
        csv_path, splits_dir = write_manifest(tmp_path, store, {"fold": folds})
        dm = make_dm(
            store, csv_path, splits_dir, batch_size=1, dataset_max_instances=3, num_workers=2
        )
        dm.setup(stage="fit")
        torch.manual_seed(0)
        batches = list(dm.train_dataloader())
        assert sorted(b["slide_ids"][0] for b in batches) == ["slide_00", "slide_01"]
        rows = [kept_rows(b) for b in batches]
        assert all(len(r) == 3 for r in rows)
        assert rows[0] != rows[1]

    def test_single_class_training_set_is_sampled_uniformly(self, store, tmp_path):
        csv_path, splits_dir = write_manifest(
            tmp_path, store, {"fold": KFOLD}, labels=[1] * len(PATCH_COUNTS)
        )
        dm = make_dm(store, csv_path, splits_dir, class_weighted_sampling=True)
        dm.setup(stage="fit")
        sampler = dm.train_dataloader().sampler
        assert isinstance(sampler, WeightedRandomSampler)
        assert len(set(sampler.weights.tolist())) == 1


class TestDataModuleValidation:
    def test_prefetch_factor_below_one_is_rejected(self, store, kfold_env):
        with pytest.raises(ValueError, match="prefetch_factor"):
            make_dm(store, kfold_env.csv, kfold_env.splits_dir, prefetch_factor=0)

    def test_missing_filename_column_names_the_columns_found(self, store, kfold_env):
        dm = make_dm(store, kfold_env.csv, kfold_env.splits_dir, filename_column="slide_path")
        with pytest.raises(ValueError, match=r"slide_path.*filename"):
            dm.setup(stage="fit")
