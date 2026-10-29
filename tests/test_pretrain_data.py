"""
Tests for the SSL pretraining data pipeline (``oceanpath.ssl.data``).

Covers:
  * collators: ``stack_collate`` (dual view) and ``multicrop_stack_collate``
  * view generation: ``_build_view_generator`` dispatch, ``WSIDualViewAugmentor``,
    ``WSIMultiCropAugmentor``, ``build_augmentor`` / ``build_multicrop_augmentor``,
    and the augmentation blocks shipped in ``configs/pretrain_training``
  * ``PretrainDataset``: pre-cap modes, sample formats, coords and dtype options
  * ``PretrainDataModule``: argument validation, slide splitting, and the train /
    val loaders end to end on a memmap store built with the repo's builder
    (``build_feature_store`` in conftest.py)

These replace the tests of ``oceanpath.data.batching``, which was deleted in
f5a9e0e when batching and collation moved into ``oceanpath.ssl.data``.

Synthetic slides encode each patch's row index in feature column 0, so a test
can tell exactly which patches a sampler or augmentor kept. Coordinates are
distinct grid points, so pairwise distances identify them up to the shared D4
transform (an isometry).
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from oceanpath.ssl.data.augmentation import (
    WSIDualViewAugmentConfig,
    WSIDualViewAugmentor,
    WSIMultiCropAugmentor,
    WSIViewSpec,
    build_augmentor,
    build_multicrop_augmentor,
)
from oceanpath.ssl.data.datamodule import (
    PretrainDataModule,
    _build_view_generator,
    multicrop_stack_collate,
    stack_collate,
)
from oceanpath.ssl.data.dataset import PretrainDataset

FEAT_DIM = 16
PATCH_COUNTS = [300, 260, 220, 400, 280, 350]
BIG_SLIDE = 3  # slide_03, 400 patches
AUG_STAT_KEYS = (
    "crop_area_frac",
    "mask_ratio",
    "mask_fraction",
    "crop_or_better_fraction",
    "view_or_better_fraction",
    "full_refill_fraction",
    "replacement_fraction",
)
DUAL_VIEW_METHODS = ("vicreg", "jepa", "lejepa")
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "pretrain_training"
DATAMODULE_LOGGER = "oceanpath.ssl.data.datamodule"


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def store(build_feature_store):
    """Six slides (220-400 patches) on a 32x32 grid, float32 features."""
    return build_feature_store(PATCH_COUNTS, feat_dim=FEAT_DIM)


@pytest.fixture(scope="module")
def fp16_store(build_feature_store):
    """Two slides stored at the builder's default float16 precision."""
    return build_feature_store([120, 90], feat_dim=FEAT_DIM, feat_precision=16, seed=1)


def _slide(n: int, d: int = 4, grid_side: int = 40, seed: int = 0):
    """In-memory slide: feature column 0 is the row index, coords are grid points."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, d)).astype(np.float32)
    feats[:, 0] = np.arange(n)
    gx, gy = np.meshgrid(np.arange(grid_side), np.arange(grid_side), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1) * 256
    coords = grid[rng.choice(len(grid), size=n, replace=False)].astype(np.int32)
    return feats, coords


def _row_index(view) -> np.ndarray:
    """Row indices of a view's patches (feature column 0)."""
    arr = view.numpy() if isinstance(view, torch.Tensor) else np.asarray(view)
    return arr[:, 0].astype(np.int64)


def _sq_dists(coords) -> np.ndarray:
    """Pairwise squared distances; invariant under the shared D4 transform."""
    c = np.asarray(coords, dtype=np.int64)
    diff = c[:, None, :] - c[None, :, :]
    return (diff**2).sum(axis=-1)


def _dual_aug(fixed_n: int = 64, **overrides) -> WSIDualViewAugmentor:
    overrides.setdefault("min_tokens", min(16, fixed_n))
    return build_augmentor("vicreg", fixed_n=fixed_n, **overrides)


class _RecordingViews:
    """View generator that records the pre-capped pool it receives and returns
    the pool's first ``fixed_n`` rows as both views (no ``return_stats``)."""

    def __init__(self, fixed_n: int):
        self.fixed_n = fixed_n
        self.pools: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, features, coords, rng=None):
        self.pools.append((features.copy(), coords.copy()))
        view = (features[: self.fixed_n], coords[: self.fixed_n])
        return view, view


def _dataset(store, view_generator, *, slide=0, fixed_n=32, cap=128, seed=0, **kwargs):
    ds = PretrainDataset(
        mmap_dir=str(store.mmap_dir),
        fixed_n=fixed_n,
        slide_ids=[store.slide_ids[slide]],
        view_generator=view_generator,
        dataset_max_instances=cap,
        **kwargs,
    )
    ds.rng = np.random.default_rng(seed)
    return ds


def _aug_cfg(method: str) -> dict:
    if method == "lejepa_mc":
        return {
            "num_global_views": 2,
            "num_local_views": 2,
            "local_fixed_n": 16,
            "global_min_tokens": 8,
        }
    return {"min_tokens": 8}


def _datamodule(store, method: str = "vicreg", **kwargs) -> PretrainDataModule:
    params = {
        "mmap_dir": str(store.mmap_dir),
        "fixed_n": 32,
        "batch_size": 2,
        "ssl_method": method,
        "augmentation_cfg": _aug_cfg(method),
        "dataset_max_instances": 128,
        "num_workers": 0,
        "val_frac": 0.34,  # 6 slides -> 2 val, 4 train
        "seed": 42,
    }
    params.update(kwargs)
    return PretrainDataModule(**params)


# ═════════════════════════════════════════════════════════════════════════════
# Collators
# ═════════════════════════════════════════════════════════════════════════════


def _dual_sample(n=8, d=4, slide_id="s", *, coords=True, stats=False, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sample = {
        "view1_features": torch.randn(n, d, generator=g),
        "view2_features": torch.randn(n, d, generator=g),
        "view1_length": n,
        "view2_length": n,
        "slide_id": slide_id,
    }
    if coords:
        for v in ("1", "2"):
            sample[f"view{v}_coords"] = torch.randint(
                0, 1000, (n, 2), generator=g, dtype=torch.int32
            )
    if stats:
        for i, key in enumerate(AUG_STAT_KEYS):
            sample[f"view1_{key}"] = 0.1 * i + seed
            sample[f"view2_{key}"] = 0.2 * i + seed
    return sample


def _mc_sample(v_g=2, v_l=3, n_g=8, n_l=4, d=4, slide_id="s", *, coords=True, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sample = {
        "global_features": [torch.randn(n_g, d, generator=g) for _ in range(v_g)],
        "local_features": [torch.randn(n_l, d, generator=g) for _ in range(v_l)],
        "num_global": v_g,
        "num_local": v_l,
        "slide_id": slide_id,
    }
    if coords:
        sample["global_coords"] = [
            torch.randint(0, 1000, (n_g, 2), generator=g, dtype=torch.int32) for _ in range(v_g)
        ]
        if v_l:
            sample["local_coords"] = [
                torch.randint(0, 1000, (n_l, 2), generator=g, dtype=torch.int32) for _ in range(v_l)
            ]
    return sample


class TestStackCollate:
    def test_stacks_views_in_batch_order(self):
        batch = [_dual_sample(slide_id=f"s{i}", seed=i) for i in range(3)]
        out = stack_collate(batch)
        assert out["view1"].shape == (3, 8, 4)
        assert out["view2"].shape == (3, 8, 4)
        for i, sample in enumerate(batch):
            torch.testing.assert_close(out["view1"][i], sample["view1_features"])
            torch.testing.assert_close(out["view2"][i], sample["view2_features"])
        assert out["slide_ids"] == ["s0", "s1", "s2"]

    def test_masks_and_lengths_cover_the_full_views(self):
        out = stack_collate([_dual_sample(seed=i) for i in range(2)])
        for key in ("mask1", "mask2"):
            assert out[key].dtype == torch.float32
            assert out[key].shape == (2, 8)
            assert torch.all(out[key] == 1)
        for key in ("lengths1", "lengths2"):
            assert out[key].dtype == torch.int32
            assert out[key].tolist() == [8, 8]

    def test_coords_stacked_when_present(self):
        batch = [_dual_sample(seed=i) for i in range(2)]
        out = stack_collate(batch)
        assert out["coords1"].shape == (2, 8, 2)
        assert out["coords1"].dtype == torch.int32
        torch.testing.assert_close(out["coords2"][1], batch[1]["view2_coords"])

    def test_no_coords_keys_without_coords(self):
        out = stack_collate([_dual_sample(coords=False, seed=i) for i in range(2)])
        assert "coords1" not in out
        assert "coords2" not in out

    def test_aug_stats_become_per_sample_float_tensors(self):
        out = stack_collate([_dual_sample(stats=True, seed=i) for i in range(2)])
        for key in AUG_STAT_KEYS:
            for view in ("1", "2"):
                assert out[f"{key}{view}"].dtype == torch.float32
                assert out[f"{key}{view}"].shape == (2,)
        torch.testing.assert_close(out["mask_ratio2"], torch.tensor([0.2, 1.2]))

    def test_no_aug_stat_keys_without_stats(self):
        out = stack_collate([_dual_sample(seed=i) for i in range(2)])
        assert not any(key.startswith(AUG_STAT_KEYS) for key in out)

    def test_empty_batch_rejected(self):
        with pytest.raises(ValueError, match="empty batch"):
            stack_collate([])

    def test_non_uniform_views_rejected(self):
        with pytest.raises(RuntimeError, match="uniform"):
            stack_collate([_dual_sample(n=8), _dual_sample(n=7)])


class TestMulticropStackCollate:
    def test_view_axis_leads_the_batch_axis(self):
        batch = [_mc_sample(slide_id=f"s{i}", seed=i) for i in range(3)]
        out = multicrop_stack_collate(batch)
        assert out["global_views"].shape == (2, 3, 8, 4)
        assert out["local_views"].shape == (3, 3, 4, 4)
        for b, sample in enumerate(batch):
            for v in range(2):
                torch.testing.assert_close(out["global_views"][v, b], sample["global_features"][v])
            for v in range(3):
                torch.testing.assert_close(out["local_views"][v, b], sample["local_features"][v])
        assert out["num_global"] == 2
        assert out["num_local"] == 3
        assert out["slide_ids"] == ["s0", "s1", "s2"]

    def test_masks_lengths_and_coords(self):
        out = multicrop_stack_collate([_mc_sample(seed=i) for i in range(2)])
        assert out["global_masks"].shape == (2, 2, 8)
        assert out["local_masks"].shape == (3, 2, 4)
        assert torch.all(out["global_masks"] == 1)
        assert torch.all(out["local_masks"] == 1)
        assert out["global_lengths"].dtype == torch.int32
        assert torch.all(out["global_lengths"] == 8)
        assert torch.all(out["local_lengths"] == 4)
        assert out["global_coords"].shape == (2, 2, 8, 2)
        assert out["local_coords"].shape == (3, 2, 4, 2)

    def test_zero_local_views_give_empty_placeholders(self):
        out = multicrop_stack_collate([_mc_sample(v_l=0, seed=i) for i in range(2)])
        assert out["local_views"].shape == (0, 2, 0, 4)
        assert out["local_masks"].shape == (0, 2, 0)
        assert out["local_lengths"].shape == (0, 2)
        assert out["num_local"] == 0
        assert "local_coords" not in out

    def test_no_coords_keys_without_coords(self):
        out = multicrop_stack_collate([_mc_sample(coords=False, seed=i) for i in range(2)])
        assert "global_coords" not in out
        assert "local_coords" not in out

    def test_mismatched_view_counts_rejected(self):
        with pytest.raises(RuntimeError, match="V_l=2"):
            multicrop_stack_collate([_mc_sample(v_l=2), _mc_sample(v_l=3)])

    def test_empty_batch_rejected(self):
        with pytest.raises(ValueError, match="empty batch"):
            multicrop_stack_collate([])


# ═════════════════════════════════════════════════════════════════════════════
# View generation
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildViewGenerator:
    """``_build_view_generator`` picks the augmentor for the configured method."""

    @pytest.mark.parametrize("method", ["vicreg", "jepa", "lejepa", "default", None, "VICReg"])
    def test_dual_view_methods_get_a_dual_view_augmentor(self, method):
        aug = _build_view_generator(method, {"min_tokens": 8}, fixed_n=32)
        assert isinstance(aug, WSIDualViewAugmentor)
        assert aug.cfg.fixed_n == 32

    def test_dual_view_overrides_applied(self):
        aug = _build_view_generator(
            "lejepa", {"min_tokens": 8, "split_overlap": 0.5, "d4_mode": "none"}, fixed_n=32
        )
        assert aug.cfg.split_overlap == 0.5
        assert aug.cfg.d4_mode == "none"
        assert aug.cfg.min_tokens == 8

    def test_datamodule_fixed_n_wins_over_augmentation_cfg(self, caplog):
        with caplog.at_level(logging.WARNING, logger=DATAMODULE_LOGGER):
            aug = _build_view_generator("vicreg", {"min_tokens": 8, "fixed_n": 64}, fixed_n=32)
        assert aug.cfg.fixed_n == 32
        assert "augmentation_cfg.fixed_n=64 ignored" in caplog.text

    @pytest.mark.parametrize("method", ["simclr", "byol", "dino"])
    def test_retired_methods_rejected(self, method):
        with pytest.raises(ValueError, match="Unknown ssl_method"):
            _build_view_generator(method, None, fixed_n=32)

    def test_unknown_dual_view_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown config field 'num_local_views'"):
            _build_view_generator("vicreg", {"min_tokens": 8, "num_local_views": 2}, fixed_n=32)

    def test_multicrop_method_gets_a_multicrop_augmentor(self):
        aug = _build_view_generator(
            "lejepa_mc",
            {
                "num_global_views": 2,
                "num_local_views": 3,
                "local_fixed_n": 16,
                "global_min_tokens": 8,
            },
            fixed_n=32,
        )
        assert isinstance(aug, WSIMultiCropAugmentor)
        assert (aug.num_global, aug.num_local) == (2, 3)
        assert aug.global_spec.fixed_n == 32  # global_fixed_n defaults to fixed_n
        assert aug.local_spec.fixed_n == 16
        assert aug.local_spec.min_tokens == 2  # min(128, local_fixed_n // 8)
        assert aug.fixed_n == 32
        assert aug._global_dual_aug is None  # split_overlap defaults to 1.0

    def test_multicrop_drops_inherited_dual_view_keys_quietly(self, caplog):
        cfg = {
            "local_fixed_n": 16,
            "global_min_tokens": 8,
            "crop_area_range": [0.3, 0.9],
            "mask_ratio_range": [0.1, 0.2],
            "min_tokens": 256,
            "fixed_n": 2048,
        }
        with caplog.at_level(logging.WARNING, logger=DATAMODULE_LOGGER):
            aug = _build_view_generator("lejepa_mc", cfg, fixed_n=32)
        assert isinstance(aug, WSIMultiCropAugmentor)
        assert [r for r in caplog.records if r.name == DATAMODULE_LOGGER] == []
        assert cfg["min_tokens"] == 256  # the caller's dict is not consumed

    def test_multicrop_warns_on_unknown_keys(self, caplog):
        with caplog.at_level(logging.WARNING, logger=DATAMODULE_LOGGER):
            _build_view_generator(
                "lejepa_mc",
                {"local_fixed_n": 16, "global_min_tokens": 8, "bogus_knob": 1},
                fixed_n=32,
            )
        assert "bogus_knob" in caplog.text

    def test_multicrop_split_overlap_couples_the_global_pair(self):
        aug = _build_view_generator(
            "lejepa_mc",
            {"local_fixed_n": 16, "global_min_tokens": 8, "split_overlap": 0.3},
            fixed_n=32,
        )
        assert isinstance(aug._global_dual_aug, WSIDualViewAugmentor)
        assert aug._global_dual_aug.cfg.split_overlap == 0.3
        assert aug._global_dual_aug.cfg.fixed_n == 32
        assert aug._global_dual_aug.cfg.d4_mode == "none"  # D4 runs once, outside

    def test_manifest_min_patches_guard(self):
        with pytest.raises(ValueError, match="manifest_min_patches"):
            _build_view_generator("vicreg", {"min_tokens": 8}, fixed_n=32, manifest_min_patches=16)
        with pytest.raises(ValueError, match="manifest_min_patches"):
            _build_view_generator(
                "lejepa_mc",
                {"local_fixed_n": 64, "global_min_tokens": 8},
                fixed_n=32,
                manifest_min_patches=48,
            )


class TestShippedAugmentationConfigs:
    """The augmentation blocks in configs/pretrain_training build valid augmentors.

    Each method file lists ``defaults: [base]``; merging it over base.yaml
    mirrors Hydra's composition (lejepa_mc therefore also inherits the
    dual-view keys, which must be dropped without a warning).
    """

    @staticmethod
    def _compose(name: str):
        base = OmegaConf.load(CONFIG_DIR / "base.yaml")
        cfg = OmegaConf.merge(base, OmegaConf.load(CONFIG_DIR / f"{name}.yaml"))
        return cfg, OmegaConf.to_container(cfg.augmentation, resolve=True)

    @pytest.mark.parametrize("name", ["vicreg", "jepa", "lejepa_2c"])
    def test_dual_view_configs(self, name):
        cfg, aug_cfg = self._compose(name)
        aug = _build_view_generator(
            cfg.ssl_method,
            aug_cfg,
            fixed_n=int(cfg.fixed_n),
            manifest_min_patches=int(cfg.manifest_min_patches),
        )
        assert isinstance(aug, WSIDualViewAugmentor)
        assert aug.cfg.fixed_n == cfg.fixed_n
        assert aug.cfg.split_overlap == cfg.augmentation.split_overlap

    def test_lejepa_mc_config(self, caplog):
        cfg, aug_cfg = self._compose("lejepa_mc")
        with caplog.at_level(logging.WARNING, logger=DATAMODULE_LOGGER):
            aug = _build_view_generator(
                cfg.ssl_method,
                aug_cfg,
                fixed_n=int(cfg.fixed_n),
                manifest_min_patches=int(cfg.manifest_min_patches),
            )
        assert [r for r in caplog.records if r.name == DATAMODULE_LOGGER] == []
        assert isinstance(aug, WSIMultiCropAugmentor)
        assert (aug.num_global, aug.num_local) == (2, 2)
        assert aug.global_spec.fixed_n == cfg.augmentation.global_fixed_n
        assert aug.local_spec.fixed_n == cfg.augmentation.local_fixed_n
        # LeJEPA-MC globals are sampled exactly like the LeJEPA-2C pair.
        assert aug._global_dual_aug.cfg.split_overlap == cfg.augmentation.split_overlap


class TestDualViewAugmentor:
    def test_views_are_fixed_n_unique_unmodified_rows(self):
        feats, coords = _slide(600)
        (f1, _), (f2, _) = _dual_aug(64)(feats, coords, rng=np.random.default_rng(0))
        for f in (f1, f2):
            idx = _row_index(f)
            assert f.shape == (64, 4)
            assert len(np.unique(idx)) == 64
            np.testing.assert_array_equal(f, feats[idx])

    def test_without_d4_coords_are_the_slide_coords(self):
        feats, coords = _slide(600)
        (f1, c1), (f2, c2) = _dual_aug(64, d4_mode="none")(
            feats, coords, rng=np.random.default_rng(0)
        )
        np.testing.assert_array_equal(c1, coords[_row_index(f1)])
        np.testing.assert_array_equal(c2, coords[_row_index(f2)])

    def test_shared_d4_is_one_isometry_for_both_views(self):
        feats, coords = _slide(600)
        aug = _dual_aug(64, split_overlap=1.0)
        moved = 0
        for seed in range(8):
            (f1, c1), (f2, c2) = aug(feats, coords, rng=np.random.default_rng(seed))
            idx = np.concatenate([_row_index(f1), _row_index(f2)])
            both = np.concatenate([c1, c2])
            # Distances across the two views are preserved too, so both views
            # went through the same transform.
            np.testing.assert_array_equal(_sq_dists(both), _sq_dists(coords[idx]))
            assert both.min() >= 0
            assert both.dtype == coords.dtype
            moved += not np.array_equal(both, coords[idx])
        assert moved > 0

    def test_zero_overlap_split_gives_disjoint_views(self):
        feats, coords = _slide(600)
        aug = _dual_aug(64, split_overlap=0.0, crop_prob=0.0, mask_ratio_range=(0.0, 0.0))
        (f1, _), (f2, _) = aug(feats, coords, rng=np.random.default_rng(0))
        assert not set(_row_index(f1)) & set(_row_index(f2))

    def test_small_slide_is_oversampled_to_fixed_n(self):
        feats, coords = _slide(40)
        ((f1, _), _), stats = _dual_aug(64)(
            feats, coords, rng=np.random.default_rng(0), return_stats=True
        )
        assert f1.shape == (64, 4)
        assert len(np.unique(_row_index(f1))) == 40
        assert stats["v1"]["fallback_level"] == "oversample"
        assert stats["v1"]["n_from_replacement"] == 24
        assert stats["v1"]["replacement_fraction"] == pytest.approx(24 / 64)

    def test_stats_account_for_every_token(self):
        feats, coords = _slide(600)
        _, stats = _dual_aug(64)(feats, coords, rng=np.random.default_rng(0), return_stats=True)
        assert stats["n_tokens"] == 600
        assert stats["fixed_n"] == 64
        for view in ("v1", "v2"):
            s = stats[view]
            parts = ("mask", "crop", "split", "full", "replacement")
            assert sum(s[f"n_from_{p}"] for p in parts) == s["final_n"] == 64
            assert s["mask_fraction"] == pytest.approx(s["n_from_mask"] / 64)
            assert 0.0 < s["crop_area_frac"] <= 1.0
            assert 0.0 <= s["mask_ratio"] < 1.0
            assert s["unique_fraction"] == 1.0

    def test_same_seed_gives_the_same_views(self):
        feats, coords = _slide(600)
        aug = _dual_aug(64)
        a = aug(feats, coords, rng=np.random.default_rng(7))
        b = aug(feats, coords, rng=np.random.default_rng(7))
        c = aug(feats, coords, rng=np.random.default_rng(8))
        for (fa, ca), (fb, cb) in zip(a, b, strict=True):
            np.testing.assert_array_equal(fa, fb)
            np.testing.assert_array_equal(ca, cb)
        assert not np.array_equal(a[0][0], c[0][0])

    def test_runs_without_coords(self):
        feats, _ = _slide(600)
        (f1, c1), (f2, c2) = _dual_aug(64)(feats, None, rng=np.random.default_rng(0))
        assert c1 is None
        assert c2 is None
        assert f1.shape == f2.shape == (64, 4)

    def test_manifest_min_patches_below_fixed_n_rejected(self):
        cfg = WSIDualViewAugmentConfig(fixed_n=64, min_tokens=16)
        with pytest.raises(ValueError, match="manifest_min_patches"):
            WSIDualViewAugmentor(cfg, manifest_min_patches=32)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("min_tokens", 128),  # above fixed_n
            ("split_overlap", 1.5),
            ("crop_prob", -0.1),
            ("crop_area_range", (0.0, 0.5)),
            ("crop_aspect_range", (2.0, 0.5)),
            ("mask_ratio_range", (0.2, 1.0)),
            ("d4_mode", "rot90"),
        ],
    )
    def test_invalid_config_rejected(self, field, value):
        cfg = WSIDualViewAugmentConfig(fixed_n=64, min_tokens=16)
        setattr(cfg, field, value)
        with pytest.raises(ValueError):
            WSIDualViewAugmentor(cfg)

    def test_input_validation(self):
        feats, coords = _slide(100)
        aug = _dual_aug(64)
        with pytest.raises(TypeError):
            aug(torch.from_numpy(feats), coords)
        with pytest.raises(ValueError, match=r"\[N, D\]"):
            aug(feats[:, 0], coords)
        with pytest.raises(ValueError, match=r"\[N, 2\]"):
            aug(feats, coords[:, :1])
        with pytest.raises(ValueError, match="same N"):
            aug(feats, coords[:50])


class TestBuildAugmentor:
    def test_does_not_mutate_the_callers_config(self):
        base = WSIDualViewAugmentConfig(fixed_n=64, min_tokens=16)
        aug = build_augmentor("lejepa", cfg=base, split_overlap=0.0)
        assert aug.cfg.split_overlap == 0.0
        assert base.split_overlap == 0.3
        assert aug.cfg is not base

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="Unknown config field 'blur_sigma'"):
            build_augmentor("vicreg", fixed_n=64, min_tokens=16, blur_sigma=1.0)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown method"):
            build_augmentor("simclr")

    def test_multicrop_method_rejected(self):
        with pytest.raises(ValueError, match="build_multicrop_augmentor"):
            build_augmentor("lejepa_mc")


class TestMultiCropAugmentor:
    @staticmethod
    def _aug(**kwargs) -> WSIMultiCropAugmentor:
        kwargs.setdefault("global_fixed_n", 64)
        kwargs.setdefault("local_fixed_n", 32)
        kwargs.setdefault("global_min_tokens", 16)
        return build_multicrop_augmentor(**kwargs)

    def test_view_counts_and_sizes(self):
        feats, coords = _slide(600)
        out = self._aug(num_global=2, num_local=3)(feats, coords, rng=np.random.default_rng(0))
        assert [f.shape for f, _ in out["global"]] == [(64, 4)] * 2
        assert [f.shape for f, _ in out["local"]] == [(32, 4)] * 3
        for f, _ in out["global"] + out["local"]:
            idx = _row_index(f)
            assert len(np.unique(idx)) == len(idx)
            np.testing.assert_array_equal(f, feats[idx])

    def test_one_d4_transform_shared_by_all_views(self):
        feats, coords = _slide(600)
        out = self._aug(num_global=2, num_local=2)(feats, coords, rng=np.random.default_rng(1))
        views = out["global"] + out["local"]
        idx = np.concatenate([_row_index(f) for f, _ in views])
        all_coords = np.concatenate([c for _, c in views])
        np.testing.assert_array_equal(_sq_dists(all_coords), _sq_dists(coords[idx]))

    def test_zero_local_views(self):
        feats, coords = _slide(600)
        out = self._aug(num_global=3, num_local=0)(feats, coords, rng=np.random.default_rng(0))
        assert len(out["global"]) == 3
        assert out["local"] == []

    def test_per_view_stats(self):
        feats, coords = _slide(600)
        _, stats = self._aug(num_global=2, num_local=2)(
            feats, coords, rng=np.random.default_rng(0), return_stats=True
        )
        assert [s["view_type"] for s in stats["global_views"]] == ["global", "global"]
        assert [s["view_type"] for s in stats["local_views"]] == ["local", "local"]
        assert [s["final_n"] for s in stats["local_views"]] == [32, 32]

    def test_split_overlap_samples_globals_like_the_dual_view_augmentor(self):
        spec = WSIViewSpec(fixed_n=64, crop_prob=0.0, mask_ratio_range=(0.0, 0.0), min_tokens=16)
        local = WSIViewSpec(fixed_n=32, min_tokens=8)
        aug = WSIMultiCropAugmentor(
            num_global=2, num_local=1, global_spec=spec, local_spec=local, split_overlap=0.0
        )
        feats, coords = _slide(600)
        out = aug(feats, coords, rng=np.random.default_rng(0))
        (g1, _), (g2, _) = out["global"]
        assert not set(_row_index(g1)) & set(_row_index(g2))

    def test_invalid_arguments_rejected(self):
        with pytest.raises(ValueError, match="num_global"):
            WSIMultiCropAugmentor(num_global=0)
        with pytest.raises(ValueError, match="num_local"):
            WSIMultiCropAugmentor(num_local=-1)
        with pytest.raises(ValueError, match="num_global=3"):
            WSIMultiCropAugmentor(num_global=3, split_overlap=0.5)
        with pytest.raises(ValueError, match="manifest_min_patches"):
            self._aug(manifest_min_patches=48)

    def test_fixed_n_is_the_largest_view(self):
        assert self._aug(global_fixed_n=64, local_fixed_n=96).fixed_n == 96

    def test_default_local_min_tokens_scales_with_local_fixed_n(self):
        assert self._aug(local_fixed_n=32).local_spec.min_tokens == 4
        assert self._aug(local_fixed_n=32, local_min_tokens=12).local_spec.min_tokens == 12


# ═════════════════════════════════════════════════════════════════════════════
# PretrainDataset
# ═════════════════════════════════════════════════════════════════════════════


class TestPretrainDatasetPreCap:
    """dataset_max_instances caps each slide before augmentation."""

    def _pools(self, store, mode, *, cap=128, seed=0, reads=1):
        rec = _RecordingViews(fixed_n=32)
        ds = _dataset(store, rec, slide=BIG_SLIDE, cap=cap, seed=seed, pre_cap_mode=mode)
        for _ in range(reads):
            ds[0]
        sid = store.slide_ids[BIG_SLIDE]
        return rec.pools, store.features[sid], store.coords[sid]

    @pytest.mark.parametrize("cap", [None, 400, 1000])
    def test_slide_within_the_cap_is_passed_whole(self, store, cap):
        pools, feats, coords = self._pools(store, "random", cap=cap)
        np.testing.assert_array_equal(pools[0][0], feats)
        np.testing.assert_array_equal(pools[0][1], coords)

    @pytest.mark.parametrize("mode", ["random", "contiguous", "head", "spatial_stratified"])
    def test_pool_is_capped_and_coords_follow_features(self, store, mode):
        pools, feats, coords = self._pools(store, mode)
        pool_feats, pool_coords = pools[0]
        idx = _row_index(pool_feats)
        assert pool_feats.shape == (128, FEAT_DIM)
        assert len(np.unique(idx)) == 128
        np.testing.assert_array_equal(pool_feats, feats[idx])
        np.testing.assert_array_equal(pool_coords, coords[idx])

    def test_head_takes_the_first_rows(self, store):
        pools, _, _ = self._pools(store, "head", reads=3)
        for pool_feats, _ in pools:
            np.testing.assert_array_equal(_row_index(pool_feats), np.arange(128))

    def test_contiguous_takes_a_random_window(self, store):
        pools, _, _ = self._pools(store, "contiguous", reads=10)
        starts = set()
        for pool_feats, _ in pools:
            idx = _row_index(pool_feats)
            np.testing.assert_array_equal(idx, idx[0] + np.arange(128))
            assert 0 <= idx[0] <= 400 - 128
            starts.add(int(idx[0]))
        assert len(starts) > 1

    def test_random_takes_a_sorted_random_subset(self, store):
        pools, _, _ = self._pools(store, "random", reads=2)
        first, second = (_row_index(pool_feats) for pool_feats, _ in pools)
        assert np.all(np.diff(first) > 0)
        assert not np.array_equal(first, second)

    def test_spatial_stratified_keeps_every_occupied_cell(self, store):
        """With the cap above the occupied-cell count, each occupied cell of the
        sampler's 16x16 grid keeps a patch (binning as in
        mmap_builder._spatial_stratified_subsample)."""
        pools, _, coords = self._pools(store, "spatial_stratified", cap=300)
        cmin = coords.min(axis=0)
        cell_size = (coords.max(axis=0) - cmin + 1e-6) / 16

        def cells(c):
            ij = np.clip(((c - cmin) / cell_size).astype(int), 0, 15)
            return set((ij[:, 0] * 16 + ij[:, 1]).tolist())

        occupied = cells(coords)
        assert len(occupied) < 300  # stratified path, not the uniform fallback
        assert cells(pools[0][1]) == occupied

    def test_pre_cap_draws_from_the_dataset_rng(self, store):
        first, _, _ = self._pools(store, "random", seed=5)
        second, _, _ = self._pools(store, "random", seed=5)
        np.testing.assert_array_equal(first[0][0], second[0][0])

    def test_invalid_mode_rejected(self, store):
        with pytest.raises(ValueError, match="pre_cap_mode"):
            _dataset(store, _RecordingViews(32), pre_cap_mode="nearest")

    def test_fixed_n_must_be_positive(self, store):
        with pytest.raises(ValueError, match="fixed_n"):
            _dataset(store, _RecordingViews(32), fixed_n=0)


class TestPretrainDatasetSamples:
    def test_dual_view_sample(self, store):
        sample = _dataset(store, _dual_aug(32))[0]
        sid = store.slide_ids[0]
        assert sample["slide_id"] == sid
        for v in ("1", "2"):
            feats = sample[f"view{v}_features"]
            coords = sample[f"view{v}_coords"]
            assert feats.shape == (32, FEAT_DIM)
            assert feats.dtype == torch.float32
            assert coords.dtype == torch.int32
            assert sample[f"view{v}_length"] == 32
            idx = _row_index(feats)
            np.testing.assert_array_equal(feats.numpy(), store.features[sid][idx])
            np.testing.assert_array_equal(_sq_dists(coords), _sq_dists(store.coords[sid][idx]))
        for key in AUG_STAT_KEYS:
            assert isinstance(sample[f"view1_{key}"], float)
            assert isinstance(sample[f"view2_{key}"], float)

    def test_coords_aware_false_strips_coords(self, store):
        sample = _dataset(store, _dual_aug(32), coords_aware=False)[0]
        assert "view1_coords" not in sample
        assert "view2_coords" not in sample
        assert sample["view1_features"].shape == (32, FEAT_DIM)

    def test_native_dtype_kept_unless_forced(self, fp16_store):
        assert _dataset(fp16_store, _dual_aug(32))[0]["view1_features"].dtype == torch.float16
        forced = _dataset(fp16_store, _dual_aug(32), force_float32=True)[0]
        assert forced["view1_features"].dtype == torch.float32

    def test_multicrop_sample(self, store):
        aug = build_multicrop_augmentor(
            num_global=2, num_local=3, global_fixed_n=32, local_fixed_n=16, global_min_tokens=8
        )
        sample = _dataset(store, aug)[0]
        assert (sample["num_global"], sample["num_local"]) == (2, 3)
        assert [tuple(f.shape) for f in sample["global_features"]] == [(32, FEAT_DIM)] * 2
        assert [tuple(f.shape) for f in sample["local_features"]] == [(16, FEAT_DIM)] * 3
        assert len(sample["global_coords"]) == 2
        assert len(sample["local_coords"]) == 3
        assert "view1_features" not in sample

    def test_augmentor_size_mismatch_is_loud(self, store):
        ds = _dataset(store, _dual_aug(48), fixed_n=32)
        with pytest.raises(RuntimeError, match="augmentor returned 48 tokens, expected 32"):
            ds[0]

    def test_plain_callable_view_generator(self, store):
        """Callables without return_stats are supported; no stats keys are emitted."""
        sample = _dataset(store, _RecordingViews(32))[0]
        assert "view1_crop_area_frac" not in sample
        torch.testing.assert_close(sample["view1_features"], sample["view2_features"])

    def test_all_slides_by_default(self, store):
        ds = PretrainDataset(str(store.mmap_dir), fixed_n=32, view_generator=_dual_aug(32))
        assert len(ds) == len(PATCH_COUNTS)
        assert ds.slide_ids == store.slide_ids
        assert ds.get_bag_sizes().tolist() == PATCH_COUNTS


# ═════════════════════════════════════════════════════════════════════════════
# PretrainDataModule
# ═════════════════════════════════════════════════════════════════════════════


class TestPretrainDataModuleArguments:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"fixed_n": 512, "dataset_max_instances": 256}, "Pre-cap cannot be smaller"),
            ({"fixed_n": 64, "manifest_min_patches": 32}, "manifest_min_patches=32"),
            ({"fixed_n": 0}, "fixed_n must be >= 1"),
            ({"batch_size": 0}, "batch_size must be >= 1"),
            ({"prefetch_factor": 0}, "prefetch_factor must be >= 1"),
            ({"val_frac": 0.0}, "val_frac"),
            ({"val_frac": 1.0}, "val_frac"),
        ],
    )
    def test_invalid_arguments_rejected(self, tmp_path, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PretrainDataModule(mmap_dir=str(tmp_path), **kwargs)

    def test_max_instances_overrides_fixed_n(self, tmp_path):
        assert PretrainDataModule(mmap_dir=str(tmp_path), max_instances=48).fixed_n == 48

    def test_multicrop_pre_cap_must_cover_the_largest_view(self, store):
        aug = {"global_fixed_n": 256, "local_fixed_n": 16, "global_min_tokens": 8}
        dm = _datamodule(store, "lejepa_mc", augmentation_cfg=aug)
        with pytest.raises(ValueError, match="max view fixed_n=256"):
            dm.setup()

    def test_loader_kwargs_without_workers(self, tmp_path):
        dm = PretrainDataModule(mmap_dir=str(tmp_path), num_workers=0, pin_memory=False)
        assert dm._loader_kwargs() == {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }

    def test_loader_kwargs_with_workers(self, tmp_path):
        dm = PretrainDataModule(
            mmap_dir=str(tmp_path), num_workers=2, prefetch_factor=3, pin_memory=False
        )
        kwargs = dm._loader_kwargs()
        assert kwargs["prefetch_factor"] == 3
        assert kwargs["persistent_workers"] is True
        assert kwargs["worker_init_fn"] is PretrainDataModule._worker_init_fn
        dm.persistent_workers = False
        assert dm._loader_kwargs()["persistent_workers"] is False

    def test_pin_memory_auto_detect_is_off_without_cuda(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert PretrainDataModule(mmap_dir=str(tmp_path))._resolve_pin_memory() is False
        explicit = PretrainDataModule(mmap_dir=str(tmp_path), pin_memory=True)
        assert explicit._resolve_pin_memory() is True

    def test_worker_init_reseeds_the_dataset_rng(self, monkeypatch):
        dataset = SimpleNamespace(rng=None)
        info = SimpleNamespace(dataset=dataset, seed=2**40 + 7)
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)
        PretrainDataModule._worker_init_fn(0)
        expected = np.random.default_rng((2**40 + 7) & 0xFFFF_FFFF).integers(1 << 30, size=4)
        np.testing.assert_array_equal(dataset.rng.integers(1 << 30, size=4), expected)


class TestPretrainDataModuleSplits:
    def test_setup_splits_every_slide_once(self, store):
        dm = _datamodule(store)
        dm.setup()
        train, val = set(dm.train_dataset.slide_ids), set(dm.val_dataset.slide_ids)
        assert not train & val
        assert train | val == set(store.slide_ids)
        assert len(val) == 2  # max(1, int(6 * 0.34))

    def test_setup_is_idempotent(self, store):
        dm = _datamodule(store)
        dm.setup()
        train_ds = dm.train_dataset
        dm.setup()
        assert dm.train_dataset is train_ds

    def test_split_reproducible_for_a_seed(self, store):
        first, second = _datamodule(store, seed=7), _datamodule(store, seed=7)
        first.setup()
        second.setup()
        assert first.train_dataset.slide_ids == second.train_dataset.slide_ids
        assert first.val_dataset.slide_ids == second.val_dataset.slide_ids

    def test_split_depends_on_the_seed(self):
        ids = [f"slide_{i:02d}" for i in range(20)]
        splits = set()
        for seed in range(5):
            dm = PretrainDataModule.__new__(PretrainDataModule)
            dm.val_frac, dm.seed, dm.split_manifest_path = 0.2, seed, None
            splits.add(tuple(sorted(dm._split_slide_ids(ids)[1])))
        assert len(splits) > 1

    @pytest.mark.parametrize(
        "n, val_frac, expected_val", [(100, 0.2, 20), (3, 0.01, 1), (4, 0.5, 2), (1, 0.1, 1)]
    )
    def test_split_sizes(self, n, val_frac, expected_val):
        dm = PretrainDataModule.__new__(PretrainDataModule)
        dm.val_frac, dm.seed, dm.split_manifest_path = val_frac, 42, None
        ids = [f"s{i}" for i in range(n)]
        train, val = dm._split_slide_ids(ids)
        assert len(val) == expected_val
        assert sorted(train + val) == sorted(ids)

    def test_no_slides_rejected(self):
        dm = PretrainDataModule.__new__(PretrainDataModule)
        with pytest.raises(ValueError, match="No slide IDs"):
            dm._split_slide_ids([])

    def test_csv_filters_slides(self, store, tmp_path, caplog):
        csv = tmp_path / "pretrain.csv"
        csv.write_text("slide_id\nslide_00\nslide_02\nslide_04\nslide_missing\n")
        dm = _datamodule(store, csv_path=str(csv))
        with caplog.at_level(logging.WARNING, logger=DATAMODULE_LOGGER):
            dm.setup()
        used = set(dm.train_dataset.slide_ids) | set(dm.val_dataset.slide_ids)
        assert used == {"slide_00", "slide_02", "slide_04"}
        assert "1 slides listed in CSV but missing from mmap" in caplog.text

    def test_csv_filename_column(self, store, tmp_path):
        csv = tmp_path / "pretrain.csv"
        csv.write_text("filename\nslide_01\nslide_03\n")
        dm = _datamodule(store, csv_path=str(csv), filename_column="filename", val_frac=0.5)
        dm.setup()
        used = set(dm.train_dataset.slide_ids) | set(dm.val_dataset.slide_ids)
        assert used == {"slide_01", "slide_03"}

    def test_missing_csv_falls_back_to_all_slides(self, store, tmp_path, caplog):
        dm = _datamodule(store, csv_path=str(tmp_path / "absent.csv"))
        with caplog.at_level(logging.WARNING, logger=DATAMODULE_LOGGER):
            dm.setup()
        used = set(dm.train_dataset.slide_ids) | set(dm.val_dataset.slide_ids)
        assert used == set(store.slide_ids)
        assert "CSV not found" in caplog.text

    def test_split_manifest_is_respected(self, store, tmp_path):
        manifest = tmp_path / "split.json"
        manifest.write_text(
            json.dumps(
                {
                    "train_ids": ["slide_00", "slide_01", "slide_02", "slide_03"],
                    "val_ids": ["slide_04", "slide_05", "slide_unknown"],
                }
            )
        )
        dm = _datamodule(store, split_manifest_path=str(manifest))
        dm.setup()
        assert dm.train_dataset.slide_ids == ["slide_00", "slide_01", "slide_02", "slide_03"]
        assert dm.val_dataset.slide_ids == ["slide_04", "slide_05"]

    def test_overlapping_manifest_rejected(self, store, tmp_path):
        manifest = tmp_path / "split.json"
        manifest.write_text(json.dumps({"train_ids": ["slide_00"], "val_ids": ["slide_00"]}))
        with pytest.raises(ValueError, match="overlapping"):
            _datamodule(store, split_manifest_path=str(manifest)).setup()

    @pytest.mark.parametrize("payload", [None, {"train_ids": ["slide_00"], "val_ids": []}])
    def test_unusable_manifest_falls_back_to_a_random_split(self, store, tmp_path, payload):
        manifest = tmp_path / "split.json"
        if payload is not None:
            manifest.write_text(json.dumps(payload))
        dm = _datamodule(store, split_manifest_path=str(manifest))
        dm.setup()
        assert len(dm.val_dataset) == 2
        assert len(dm.train_dataset) == 4


class TestPretrainDataModuleLoaders:
    @pytest.mark.parametrize("method", DUAL_VIEW_METHODS)
    def test_dual_view_train_batches(self, store, method):
        dm = _datamodule(store, method)
        dm.setup()
        batches = list(dm.train_dataloader())
        assert len(batches) == 2  # 4 train slides, batch_size=2, drop_last
        for batch in batches:
            assert batch["view1"].shape == batch["view2"].shape == (2, 32, FEAT_DIM)
            assert batch["view1"].dtype == torch.float32
            assert torch.all(batch["mask1"] == 1)
            assert torch.all(batch["mask2"] == 1)
            assert batch["lengths1"].tolist() == [32, 32]
            assert batch["coords1"].shape == (2, 32, 2)
            assert batch["coords1"].dtype == torch.int32
            for key in AUG_STAT_KEYS:
                assert batch[f"{key}1"].shape == (2,)
            # Every row is an unmodified patch of its slide, and the coords still
            # belong to those patches (up to the shared D4 isometry).
            for b, sid in enumerate(batch["slide_ids"]):
                for v in ("1", "2"):
                    view = batch[f"view{v}"][b].numpy()
                    idx = _row_index(view)
                    np.testing.assert_array_equal(view, store.features[sid][idx])
                    np.testing.assert_array_equal(
                        _sq_dists(batch[f"coords{v}"][b]), _sq_dists(store.coords[sid][idx])
                    )

    def test_train_drops_and_val_keeps_the_partial_batch(self, store):
        dm = _datamodule(store, batch_size=3)
        dm.setup()
        train = list(dm.train_dataloader())
        val = list(dm.val_dataloader())
        assert [b["view1"].shape[0] for b in train] == [3]  # 4 slides, last one dropped
        assert [b["view1"].shape[0] for b in val] == [2]

    def test_val_loader_is_not_shuffled(self, store):
        dm = _datamodule(store, batch_size=1)
        dm.setup()
        ids = [b["slide_ids"][0] for b in dm.val_dataloader()]
        assert ids == dm.val_dataset.slide_ids

    def test_coords_aware_false_drops_coords(self, store):
        dm = _datamodule(store, coords_aware=False)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert "coords1" not in batch
        assert "coords2" not in batch

    def test_native_float16_kept_unless_forced(self, fp16_store):
        for force, dtype in ((False, torch.float16), (True, torch.float32)):
            dm = _datamodule(fp16_store, batch_size=1, val_frac=0.5, force_float32=force)
            dm.setup()
            assert next(iter(dm.train_dataloader()))["view1"].dtype == dtype

    def test_multicrop_batches(self, store):
        dm = _datamodule(store, "lejepa_mc")
        dm.setup()
        assert dm._collate_fn() is multicrop_stack_collate
        batch = next(iter(dm.train_dataloader()))
        assert batch["global_views"].shape == (2, 2, 32, FEAT_DIM)
        assert batch["local_views"].shape == (2, 2, 16, FEAT_DIM)
        assert batch["global_coords"].shape == (2, 2, 32, 2)
        assert batch["local_coords"].shape == (2, 2, 16, 2)
        assert (batch["num_global"], batch["num_local"]) == (2, 2)
        for b, sid in enumerate(batch["slide_ids"]):
            for v in range(2):
                view = batch["local_views"][v, b].numpy()
                np.testing.assert_array_equal(view, store.features[sid][_row_index(view)])

    def test_feat_dim_read_from_the_store(self, store):
        assert _datamodule(store).feat_dim == FEAT_DIM

    def test_worker_processes_produce_the_same_batch_contract(self, store):
        dm = _datamodule(store, num_workers=2, persistent_workers=False)
        dm.setup()
        batches = list(dm.train_dataloader())
        assert len(batches) == 2
        for batch in batches:
            assert batch["view1"].shape == (2, 32, FEAT_DIM)
