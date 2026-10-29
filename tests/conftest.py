"""
Shared fixtures and path setup for all OceanPath tests.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable (for running pytest from project root)
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


PATCH_STRIDE = 256  # coords are multiples of this, like level-0 patch origins


@dataclass
class FeatureStore:
    """A memmap store built by ``oceanpath.data.mmap_builder.build_mmap``.

    ``features`` / ``coords`` hold the arrays written to each slide's H5 file
    so tests can check exactly which patches a sampler returned. Column 0 of
    every feature row is the patch's row index within its slide.
    """

    mmap_dir: Path
    slide_ids: list[str]
    feat_dim: int
    features: dict[str, np.ndarray] = field(default_factory=dict)
    coords: dict[str, np.ndarray] = field(default_factory=dict)


@pytest.fixture(scope="session")
def build_feature_store(tmp_path_factory):
    """Factory: write per-slide H5 files and build a memmap store from them.

    Each slide's patches sit on distinct points of a ``grid_side`` x
    ``grid_side`` grid (stride ``PATCH_STRIDE``) in shuffled order. Feature
    column 0 is the row index (exact for float32 stores); the remaining
    columns are standard-normal noise.
    """
    import h5py

    from oceanpath.data.mmap_builder import MmapBuildConfig, build_mmap

    def _build(
        patch_counts: list[int],
        *,
        feat_dim: int = 16,
        feat_precision: int = 32,
        grid_side: int = 32,
        seed: int = 0,
    ) -> FeatureStore:
        root = tmp_path_factory.mktemp("feature_store")
        h5_dir = root / "h5"
        h5_dir.mkdir()
        rng = np.random.default_rng(seed)
        gx, gy = np.meshgrid(np.arange(grid_side), np.arange(grid_side), indexing="ij")
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1) * PATCH_STRIDE

        store = FeatureStore(mmap_dir=root / "mmap", slide_ids=[], feat_dim=feat_dim)
        for i, n in enumerate(patch_counts):
            slide_id = f"slide_{i:02d}"
            feats = rng.standard_normal((n, feat_dim)).astype(np.float32)
            feats[:, 0] = np.arange(n)
            coords = grid[rng.choice(len(grid), size=n, replace=False)].astype(np.int32)
            with h5py.File(h5_dir / f"{slide_id}.h5", "w") as f:
                f["features"] = feats
                f["coords"] = coords
            store.slide_ids.append(slide_id)
            store.features[slide_id] = feats
            store.coords[slide_id] = coords

        build_mmap(
            MmapBuildConfig(
                h5_dir=str(h5_dir),
                output_dir=str(store.mmap_dir),
                feat_precision=feat_precision,
                save_coverage_thumbnails=False,
            )
        )
        return store

    return _build
