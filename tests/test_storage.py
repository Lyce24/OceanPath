"""Integrity tests for H5-to-mmap storage."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from oceanpath.storage.mmap import (
    ChunkedBinaryWriter,
    MmapBuildConfig,
    build_mmap,
    compute_source_hash,
    scan_h5_dir,
    validate_mmap_dir,
)


def _write_h5(path: Path, *, n_patches: int = 4, feat_dim: int = 3, coord_dim: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "features",
            data=np.arange(n_patches * feat_dim, dtype=np.float32).reshape(n_patches, feat_dim),
        )
        handle.create_dataset(
            "coords",
            data=np.arange(n_patches * coord_dim, dtype=np.int32).reshape(n_patches, coord_dim),
        )


def _config(h5_dir: Path, output_dir: Path, **overrides: object) -> MmapBuildConfig:
    values: dict[str, object] = {
        "h5_dir": str(h5_dir),
        "output_dir": str(output_dir),
        "save_coverage_thumbnails": False,
        "stream_chunk_size": 2,
        "max_chunk_gb": 1.0,
    }
    values.update(overrides)
    return MmapBuildConfig(**values)  # type: ignore[arg-type]


def _rewrite_index(index_path: Path, **changes: np.ndarray) -> None:
    with np.load(index_path, allow_pickle=True) as index:
        values = {name: index[name] for name in index.files}
    values.update(changes)
    np.savez(index_path, **values)


def test_csv_filter_normalizes_wsi_extensions_and_path_separators(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    _write_h5(h5_dir / "case_a.h5")
    _write_h5(h5_dir / "nested" / "case_b.h5")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"filename": ["case_a.SVS", r"nested\case_b.NDPI"]}).to_csv(manifest, index=False)

    cfg = _config(
        h5_dir,
        tmp_path / "mmap",
        csv_path=str(manifest),
        csv_filename_col="filename",
    )
    slides, errors = scan_h5_dir(cfg)

    assert errors == []
    assert [slide.slide_id for slide in slides] == ["case_a", "nested/case_b"]


def test_csv_filter_requires_complete_manifest_by_default(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    _write_h5(h5_dir / "present.h5")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"slide_id": ["present.svs", "missing.tiff"]}).to_csv(manifest, index=False)
    cfg = _config(h5_dir, tmp_path / "mmap", csv_path=str(manifest))

    with pytest.raises(ValueError, match="manifest slide IDs have no valid H5"):
        scan_h5_dir(cfg)

    cfg.require_all_csv_slides = False
    slides, _ = scan_h5_dir(cfg)
    assert [slide.slide_id for slide in slides] == ["present"]


def test_scan_rejects_inconsistent_coordinate_dimensions(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    _write_h5(h5_dir / "xy.h5", coord_dim=2)
    _write_h5(h5_dir / "xyz.h5", coord_dim=3)

    with pytest.raises(ValueError, match="Inconsistent coordinate dimensions"):
        scan_h5_dir(_config(h5_dir, tmp_path / "mmap"))


def test_source_hash_includes_chunk_layout_config(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    _write_h5(h5_dir / "case.h5")
    cfg = _config(h5_dir, tmp_path / "mmap", max_chunk_gb=1.0)
    slides, _ = scan_h5_dir(cfg)

    baseline = compute_source_hash(slides, cfg)
    cfg.max_chunk_gb = 2.0

    assert compute_source_hash(slides, cfg) != baseline


def test_write_failure_aborts_without_publishing_an_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5_dir = tmp_path / "h5"
    output_dir = tmp_path / "mmap"
    _write_h5(h5_dir / "case.h5")
    cfg = _config(h5_dir, output_dir)
    build_mmap(cfg)
    assert (output_dir / "index_arrays.npz").is_file()
    original = ChunkedBinaryWriter.write_streamed

    def fail_coord_write(self: ChunkedBinaryWriter, *args: object, **kwargs: object):
        if self.prefix == "coords":
            raise OSError("simulated disk failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ChunkedBinaryWriter, "write_streamed", fail_coord_write)

    with pytest.raises(RuntimeError, match="build was aborted"):
        build_mmap(cfg, force=True)

    assert not (output_dir / "index_arrays.npz").exists()
    assert not (output_dir / ".schema_version").exists()
    assert not (output_dir / ".source_hash").exists()


def test_validator_rejects_truncated_binary_chunk(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    output_dir = tmp_path / "mmap"
    _write_h5(h5_dir / "case.h5")
    build_mmap(_config(h5_dir, output_dir))
    coord_path = output_dir / "coords_000.bin"
    coord_path.write_bytes(coord_path.read_bytes()[:-4])

    with pytest.raises(ValueError, match="Truncated coords chunk"):
        validate_mmap_dir(output_dir)


def test_validator_rejects_invalid_offset(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    output_dir = tmp_path / "mmap"
    _write_h5(h5_dir / "case.h5")
    build_mmap(_config(h5_dir, output_dir))
    _rewrite_index(output_dir / "index_arrays.npz", feat_offsets=np.array([2], dtype=np.int64))

    with pytest.raises(ValueError, match="Invalid features offset"):
        validate_mmap_dir(output_dir)


def test_validator_rejects_nonpositive_length(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    output_dir = tmp_path / "mmap"
    _write_h5(h5_dir / "case.h5")
    build_mmap(_config(h5_dir, output_dir))
    _rewrite_index(output_dir / "index_arrays.npz", lengths=np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="lengths must be positive"):
        validate_mmap_dir(output_dir)


def test_matching_source_hash_does_not_hide_a_corrupt_store(tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    output_dir = tmp_path / "mmap"
    _write_h5(h5_dir / "case.h5")
    cfg = _config(h5_dir, output_dir)
    build_mmap(cfg)
    feature_path = output_dir / "features_000.bin"
    expected_size = feature_path.stat().st_size
    feature_path.write_bytes(feature_path.read_bytes()[:-2])

    result = build_mmap(cfg)

    assert feature_path.stat().st_size == expected_size
    assert result.feat_chunks == 1
    validate_mmap_dir(output_dir)
