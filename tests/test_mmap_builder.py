"""
Tests for the memmap store builder (``oceanpath.data.mmap_builder``).

Stores are built from synthetic per-slide H5 files whose feature column 0 holds
the patch's row index, so a test can tell exactly which source rows a store kept.
``_read_store`` reads a store back from the index format alone (chunk id + byte
offset per slide), independently of the training-time ``MmapDataset``.

Covers:
  * build_mmap round trip: float32 reads back identical and float16 holds the
    source rounded to float16; (1, N, D) TRIDENT layout, nested slide ids, small
    streaming blocks, several chunk files under a small chunk limit, and
    MmapDataset reading the same rows
  * scan_h5_dir: invalid files reported per slide, inconsistent feature dims,
    allowed-slide filtering from a CSV manifest
  * ChunkedBinaryWriter offsets and chunk roll-over
  * source hash: skip on unchanged sources, rebuild on force, changed settings or
    a new slide; stale chunk files removed by a rebuild
  * build-time capping (random and spatial_stratified): cap held, kept rows keep
    their own coords, sparse regions kept, reproducible builds
  * coverage QC: stats CSV, thumbnails and montage written as PNG files
  * load_mmap_index / read_mmap_coords / read_h5_coords, validate_mmap_dir on
    damaged stores, and per-slide failures recorded in errors.txt
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from oceanpath.data.mmap_builder import (
    MMAP_SCHEMA_VERSION,
    ChunkedBinaryWriter,
    MmapBuildConfig,
    SlideInfo,
    build_coverage_grid,
    build_mmap,
    compute_source_hash,
    infer_patch_size,
    load_mmap_index,
    read_h5_coords,
    read_mmap_coords,
    render_summary_montage,
    scan_h5_dir,
    slide_id_to_h5_path,
    validate_mmap_dir,
)

STRIDE = 256  # coords are level-0 patch origins on this lattice
FEAT_DIM = 8
GB = 1024**3


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _slide_arrays(n: int, rng, feat_dim: int = FEAT_DIM, grid_side: int = 16):
    """n patches on distinct lattice points; feature column 0 is the row index."""
    feats = rng.standard_normal((n, feat_dim)).astype(np.float32)
    feats[:, 0] = np.arange(n)
    gx, gy = np.meshgrid(np.arange(grid_side), np.arange(grid_side), indexing="ij")
    lattice = np.stack([gx.ravel(), gy.ravel()], axis=1) * STRIDE
    coords = lattice[rng.choice(len(lattice), size=n, replace=False)].astype(np.int32)
    return feats, coords


def _write_h5(
    path: Path,
    feats,
    coords,
    *,
    leading_dim: bool = False,
    feat_key: str = "features",
    coord_key: str = "coords",
) -> None:
    """Write one slide; ``leading_dim`` stores (1, N, D) as TRIDENT does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f[feat_key] = feats[None] if leading_dim else feats
        f[coord_key] = coords[None] if leading_dim else coords


def _cfg(h5_dir, out_dir, **kwargs) -> MmapBuildConfig:
    kwargs.setdefault("feat_precision", 32)
    kwargs.setdefault("save_coverage_thumbnails", False)
    return MmapBuildConfig(h5_dir=str(h5_dir), output_dir=str(out_dir), **kwargs)


def _read_store(mmap_dir) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{slide_id: (features, coords)} read from the index and the .bin chunks."""
    mmap_dir = Path(mmap_dir)
    with np.load(mmap_dir / "index_arrays.npz", allow_pickle=True) as idx:
        index = dict(idx)
    feat_dtype = np.dtype(str(index["feat_dtype"][0]))
    coord_dtype = np.dtype(str(index["coord_dtype"][0]))
    feat_dim, coord_dim = int(index["feat_dim"]), int(index["coord_dim"])
    out = {}
    for i, sid in enumerate(index["slide_ids"]):
        n = int(index["lengths"][i])
        if n == 0:
            out[str(sid)] = (np.empty((0, feat_dim), feat_dtype), np.empty((0, 2), coord_dtype))
            continue
        feats = np.fromfile(
            mmap_dir / f"features_{int(index['feat_chunk_ids'][i]):03d}.bin",
            dtype=feat_dtype,
            count=n * feat_dim,
            offset=int(index["feat_offsets"][i]),
        )
        coords = np.fromfile(
            mmap_dir / f"coords_{int(index['coord_chunk_ids'][i]):03d}.bin",
            dtype=coord_dtype,
            count=n * coord_dim,
            offset=int(index["coord_offsets"][i]),
        )
        out[str(sid)] = (feats.reshape(n, feat_dim), coords.reshape(n, coord_dim))
    return out


def _assert_png(path: Path) -> None:
    assert path.is_file() and path.stat().st_size > 0
    with Image.open(path) as img:
        assert img.format == "PNG"
        img.verify()  # raises on a truncated or corrupt file


# ═════════════════════════════════════════════════════════════════════════════
# Round trip
# ═════════════════════════════════════════════════════════════════════════════

# float32 bytes per slide at FEAT_DIM=8: a 640, b 800, big 2560, nested/c 960, nested/d 640.
ROUND_TRIP_COUNTS = {"a": 20, "b": 25, "big": 80, "nested/c": 30, "nested/d": 20}
CHUNK_LIMIT_GB = 2048 / GB  # 2 KiB, exactly representable


@pytest.fixture(scope="module")
def round_trip(tmp_path_factory):
    """float32 store with 2 KiB feature chunks and 7-row streaming reads.

    "b" is stored in the (1, N, D) layout; "nested/*" live in a subdirectory.
    """
    root = tmp_path_factory.mktemp("round_trip")
    rng = np.random.default_rng(0)
    source = {}
    for sid, n in ROUND_TRIP_COUNTS.items():
        feats, coords = _slide_arrays(n, rng)
        _write_h5(root / "h5" / f"{sid}.h5", feats, coords, leading_dim=sid == "b")
        source[sid] = (feats, coords)
    result = build_mmap(
        _cfg(root / "h5", root / "mmap", max_chunk_gb=CHUNK_LIMIT_GB, stream_chunk_size=7)
    )
    return SimpleNamespace(h5_dir=root / "h5", mmap_dir=root / "mmap", source=source, result=result)


class TestRoundTrip:
    def test_features_and_coords_read_back_identical(self, round_trip):
        stored = _read_store(round_trip.mmap_dir)
        assert set(stored) == set(round_trip.source)
        for sid, (feats, coords) in round_trip.source.items():
            assert stored[sid][0].dtype == np.float32
            np.testing.assert_array_equal(stored[sid][0], feats)
            np.testing.assert_array_equal(stored[sid][1], coords)

    def test_slide_ids_are_relative_posix_paths(self, round_trip):
        index = load_mmap_index(round_trip.mmap_dir)
        assert index["slide_ids"] == ["a", "b", "big", "nested/c", "nested/d"]
        for sid in index["slide_ids"]:
            assert slide_id_to_h5_path(round_trip.h5_dir, sid).is_file()

    def test_small_chunk_limit_spreads_slides_over_several_files(self, round_trip):
        """A slide opens a new file when it would overflow the current one; a slide
        larger than the limit ("big", 2560 B) still goes whole into one file."""
        with np.load(round_trip.mmap_dir / "index_arrays.npz", allow_pickle=True) as idx:
            assert idx["feat_chunk_ids"].tolist() == [0, 0, 1, 2, 2]
            assert idx["feat_offsets"].tolist() == [0, 640, 0, 0, 960]
        sizes = [(round_trip.mmap_dir / f"features_{i:03d}.bin").stat().st_size for i in range(3)]
        assert sizes == [640 + 800, 2560, 960 + 640]
        assert round_trip.result.feat_chunks == 3
        assert round_trip.result.coord_chunks == 1

    def test_index_records_counts_and_metadata(self, round_trip):
        assert validate_mmap_dir(round_trip.mmap_dir) == {
            "schema_version": MMAP_SCHEMA_VERSION,
            "n_slides": 5,
            "total_patches": sum(ROUND_TRIP_COUNTS.values()),
            "feat_dim": FEAT_DIM,
            "feat_dtype": "float32",
            "coord_dim": 2,
            "num_feat_chunks": 3,
            "num_coord_chunks": 1,
        }
        index = load_mmap_index(round_trip.mmap_dir)
        counts = [ROUND_TRIP_COUNTS[sid] for sid in index["slide_ids"]]
        assert index["lengths"].tolist() == counts
        assert index["raw_lengths"].tolist() == counts

    def test_mmap_dataset_reads_the_same_rows(self, round_trip):
        """The training-time reader agrees with the builder across chunk files."""
        from oceanpath.data.dataset import MmapDataset

        ids = list(ROUND_TRIP_COUNTS)
        ds = MmapDataset(
            mmap_dir=str(round_trip.mmap_dir),
            slide_ids=ids,
            labels=dict.fromkeys(ids, 0),
            is_train=False,
            return_coords=True,
        )
        assert len(ds) == len(ids)
        for i in range(len(ds)):
            sample = ds[i]
            feats, coords = round_trip.source[sample["slide_id"]]
            np.testing.assert_array_equal(sample["features"].numpy(), feats)
            np.testing.assert_array_equal(sample["coords"].numpy(), coords)

    def test_read_mmap_coords_matches_the_source(self, round_trip):
        index = load_mmap_index(round_trip.mmap_dir)
        for i, sid in enumerate(index["slide_ids"]):
            coords = read_mmap_coords(round_trip.mmap_dir, index, i)
            np.testing.assert_array_equal(coords, round_trip.source[sid][1])

    def test_index_without_raw_lengths_falls_back_to_lengths(self, round_trip, tmp_path):
        """Stores built before raw_lengths existed report the stored count as raw."""
        store = tmp_path / "old_store"
        shutil.copytree(round_trip.mmap_dir, store)
        with np.load(store / "index_arrays.npz", allow_pickle=True) as idx:
            arrays = {k: v for k, v in idx.items() if k != "raw_lengths"}
        np.savez(store / "index_arrays.npz", **arrays)
        index = load_mmap_index(store)
        np.testing.assert_array_equal(index["raw_lengths"], index["lengths"])


def test_float16_store_holds_the_float16_rounding(build_feature_store):
    store = build_feature_store([30, 20], feat_dim=FEAT_DIM, feat_precision=16, seed=3)
    stored = _read_store(store.mmap_dir)
    for sid in store.slide_ids:
        feats = store.features[sid]
        assert stored[sid][0].dtype == np.float16
        np.testing.assert_array_equal(stored[sid][0], feats.astype(np.float16))
        # The noise columns are not float16-exact: the rounding happened and is within
        # float16's half-ulp (relative 2**-11).
        err = np.abs(stored[sid][0].astype(np.float32) - feats)
        assert 0 < err.max() <= np.abs(feats).max() * 2**-11
        np.testing.assert_array_equal(stored[sid][1], store.coords[sid])


# ═════════════════════════════════════════════════════════════════════════════
# Scan
# ═════════════════════════════════════════════════════════════════════════════


class TestScan:
    def test_missing_or_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            scan_h5_dir(_cfg(tmp_path / "nowhere", tmp_path / "out"))
        (tmp_path / "empty").mkdir()
        (tmp_path / "empty" / "notes.txt").write_text("not a slide")
        with pytest.raises(FileNotFoundError, match="No .h5 files"):
            scan_h5_dir(_cfg(tmp_path / "empty", tmp_path / "out"))

    def test_invalid_files_are_reported_per_slide(self, tmp_path):
        feats, coords = _slide_arrays(6, np.random.default_rng(0))
        h5 = tmp_path / "h5"
        _write_h5(h5 / "good.h5", feats, coords)
        _write_h5(h5 / "no_feats.h5", feats, coords, feat_key="embeddings")
        _write_h5(h5 / "no_coords.h5", feats, coords, coord_key="xy")
        _write_h5(h5 / "flat.h5", feats[:, 0], coords)
        _write_h5(h5 / "stacked.h5", np.stack([feats, feats]), coords)
        _write_h5(h5 / "flat_coords.h5", feats, coords[:, 0])
        _write_h5(h5 / "mismatch.h5", feats, coords[:5])
        _write_h5(h5 / "empty.h5", feats[:0], coords[:0])
        (h5 / "corrupt.h5").write_bytes(b"not an hdf5 file")

        slides, errors = scan_h5_dir(_cfg(h5, tmp_path / "out"))

        assert [(s.slide_id, s.n_patches, s.feat_dim, s.coord_dim) for s in slides] == [
            ("good", 6, FEAT_DIM, 2)
        ]
        reasons = {e.slide_id: e.reason for e in errors}
        assert set(reasons) == {
            "no_feats",
            "no_coords",
            "flat",
            "stacked",
            "flat_coords",
            "mismatch",
            "empty",
            "corrupt",
        }
        assert "'features' not found" in reasons["no_feats"]
        assert "'coords' not found" in reasons["no_coords"]
        assert "Features expected 2D" in reasons["flat"]
        assert "Features expected 2D" in reasons["stacked"]
        assert "Coords expected 2D" in reasons["flat_coords"]
        assert "feats=6, coords=5" in reasons["mismatch"]
        assert reasons["empty"] == "Zero patches"

    def test_inconsistent_feature_dims_are_fatal(self, tmp_path):
        rng = np.random.default_rng(0)
        _write_h5(tmp_path / "h5" / "a.h5", *_slide_arrays(5, rng, feat_dim=8))
        _write_h5(tmp_path / "h5" / "b.h5", *_slide_arrays(5, rng, feat_dim=12))
        with pytest.raises(ValueError, match="Inconsistent feature dimensions"):
            scan_h5_dir(_cfg(tmp_path / "h5", tmp_path / "out"))

    @pytest.mark.parametrize(
        "manifest",
        [
            {"slide_id": ["site1/s1", "s3", "not_on_disk"]},
            {"filename": ["site1/s1.h5", "s3.h5", "not_on_disk.h5"]},
            {"filename": ["site1/s1", "s3", "not_on_disk"]},
        ],
        ids=["slide_id", "filename_with_h5", "bare_filename"],
    )
    def test_csv_filter_keeps_only_listed_slides(self, tmp_path, manifest):
        rng = np.random.default_rng(0)
        for sid in ("site1/s1", "site1/s2", "s3"):
            _write_h5(tmp_path / "h5" / f"{sid}.h5", *_slide_arrays(5, rng))
        pd.DataFrame(manifest).to_csv(tmp_path / "manifest.csv", index=False)
        cfg = _cfg(tmp_path / "h5", tmp_path / "out", csv_path=str(tmp_path / "manifest.csv"))
        slides, errors = scan_h5_dir(cfg)
        assert [s.slide_id for s in slides] == ["s3", "site1/s1"]
        assert errors == []

    def test_csv_without_an_id_column_is_rejected(self, tmp_path):
        _write_h5(tmp_path / "h5" / "s.h5", *_slide_arrays(5, np.random.default_rng(0)))
        pd.DataFrame({"name": ["s"]}).to_csv(tmp_path / "manifest.csv", index=False)
        cfg = _cfg(tmp_path / "h5", tmp_path / "out", csv_path=str(tmp_path / "manifest.csv"))
        with pytest.raises(ValueError, match="neither 'slide_id' nor 'filename'"):
            scan_h5_dir(cfg)


# ═════════════════════════════════════════════════════════════════════════════
# Chunked writer
# ═════════════════════════════════════════════════════════════════════════════


def test_writer_offsets_and_chunk_roll_over(tmp_path):
    writer = ChunkedBinaryWriter(str(tmp_path), "features", max_bytes=100)
    blocks = [np.full(n, i, dtype=np.uint8) for i, n in enumerate([60, 30, 20, 250, 5])]
    placements = [writer.write(block) for block in blocks]
    writer.close()
    # 60 + 30 fit in 100 bytes; 20 more would not, so it opens file 1; a 250-byte
    # block is larger than any file may be and goes whole into its own file.
    assert placements == [(0, 0), (0, 60), (1, 0), (2, 0), (3, 0)]
    assert writer.num_chunks == 4
    for (chunk, offset), block in zip(placements, blocks, strict=True):
        data = np.fromfile(tmp_path / f"features_{chunk:03d}.bin", dtype=np.uint8)
        np.testing.assert_array_equal(data[offset : offset + len(block)], block)


# ═════════════════════════════════════════════════════════════════════════════
# Source hash and rebuilds
# ═════════════════════════════════════════════════════════════════════════════


class TestSourceHashAndRebuild:
    @pytest.fixture
    def sources(self, tmp_path):
        rng = np.random.default_rng(1)
        feats = {}
        for sid, n in {"a": 12, "b": 9}.items():
            feats[sid], coords = _slide_arrays(n, rng)
            _write_h5(tmp_path / "h5" / f"{sid}.h5", feats[sid], coords)
        return tmp_path / "h5", feats

    def test_hash_tracks_the_slides_and_the_stored_format(self, tmp_path):
        slides = [SlideInfo("a", "a.h5", 10, 10, 8, 2), SlideInfo("b", "b.h5", 5, 5, 8, 2)]

        def source_hash(slide_list=slides, **settings):
            return compute_source_hash(slide_list, _cfg(tmp_path, tmp_path / "out", **settings))

        base = source_hash()
        assert source_hash() == base
        # The streaming block size does not change what is stored.
        assert source_hash(stream_chunk_size=3) == base
        variants = [
            source_hash(slides[:1]),
            source_hash([slides[0], SlideInfo("b", "b.h5", 6, 6, 8, 2)]),
            source_hash(feat_precision=16),
            source_hash(max_instances=4),
            source_hash(cap_strategy="random"),
            source_hash(cap_grid_size=8),
            source_hash(coord_dtype="int64"),
            source_hash(h5_feat_key="embeddings"),
            source_hash(h5_coord_key="xy"),
        ]
        assert base not in variants
        assert len(set(variants)) == len(variants)

    def test_unchanged_sources_skip_the_rebuild_unless_forced(self, tmp_path, sources):
        h5_dir, feats = sources
        cfg = _cfg(h5_dir, tmp_path / "mmap")
        first = build_mmap(cfg)
        chunk = tmp_path / "mmap" / "features_000.bin"
        size = chunk.stat().st_size
        chunk.write_bytes(bytes(size))  # damage the store behind the builder's back

        skipped = build_mmap(cfg)
        assert skipped.source_hash == first.source_hash
        assert skipped.feat_chunks == 0
        assert chunk.read_bytes() == bytes(size)  # untouched

        rebuilt = build_mmap(cfg, force=True)
        assert rebuilt.feat_chunks == 1
        stored = _read_store(tmp_path / "mmap")
        for sid, f in feats.items():
            np.testing.assert_array_equal(stored[sid][0], f)

    def test_changed_settings_or_new_slides_rebuild_without_force(self, tmp_path, sources):
        h5_dir, _ = sources
        build_mmap(_cfg(h5_dir, tmp_path / "mmap"))

        build_mmap(_cfg(h5_dir, tmp_path / "mmap", feat_precision=16))
        assert validate_mmap_dir(tmp_path / "mmap")["feat_dtype"] == "float16"

        _write_h5(h5_dir / "c.h5", *_slide_arrays(7, np.random.default_rng(9)))
        build_mmap(_cfg(h5_dir, tmp_path / "mmap", feat_precision=16))
        assert load_mmap_index(tmp_path / "mmap")["slide_ids"] == ["a", "b", "c"]

    def test_rebuild_removes_chunk_files_the_new_layout_does_not_use(self, tmp_path, sources):
        h5_dir, feats = sources
        build_mmap(_cfg(h5_dir, tmp_path / "mmap", max_chunk_gb=512 / GB))  # a: 384 B, b: 288 B
        assert len(list((tmp_path / "mmap").glob("features_*.bin"))) == 2

        build_mmap(_cfg(h5_dir, tmp_path / "mmap"), force=True)
        assert sorted(p.name for p in (tmp_path / "mmap").glob("*.bin")) == [
            "coords_000.bin",
            "features_000.bin",
        ]
        stored = _read_store(tmp_path / "mmap")
        for sid, f in feats.items():
            np.testing.assert_array_equal(stored[sid][0], f)


# ═════════════════════════════════════════════════════════════════════════════
# Build-time capping
# ═════════════════════════════════════════════════════════════════════════════

CAP = 24
CAP_COUNTS = {"large": 120, "medium": 60, "small": 15}


@pytest.fixture(scope="module")
def cap_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("cap_sources")
    rng = np.random.default_rng(2)
    source = {}
    for sid, n in CAP_COUNTS.items():
        feats, coords = _slide_arrays(n, rng)
        _write_h5(root / f"{sid}.h5", feats, coords, leading_dim=sid == "medium")
        source[sid] = (feats, coords)
    return root, source


@pytest.mark.parametrize("strategy", ["random", "spatial_stratified"])
class TestBuildTimeCap:
    def test_capped_rows_keep_their_own_coords(self, cap_sources, tmp_path, strategy):
        """1 KiB feature chunks hold one capped slide (768 B) each, so the capped
        writes also open new files."""
        h5_dir, source = cap_sources
        cfg = _cfg(
            h5_dir,
            tmp_path,
            max_instances=CAP,
            cap_strategy=strategy,
            stream_chunk_size=5,
            max_chunk_gb=1024 / GB,
        )
        build_mmap(cfg)
        index = load_mmap_index(tmp_path)
        assert index["slide_ids"] == ["large", "medium", "small"]
        assert index["lengths"].tolist() == [CAP, CAP, 15]
        assert index["raw_lengths"].tolist() == [120, 60, 15]
        assert validate_mmap_dir(tmp_path)["num_feat_chunks"] == 3

        stored = _read_store(tmp_path)
        for sid, (feats, coords) in source.items():
            kept_feats, kept_coords = stored[sid]
            rows = kept_feats[:, 0].astype(int)  # column 0 = source row index
            assert len(np.unique(rows)) == len(rows) == min(len(feats), CAP)
            np.testing.assert_array_equal(kept_feats, feats[rows])
            np.testing.assert_array_equal(kept_coords, coords[rows])
        # A slide under the cap is stored whole, in its original order.
        np.testing.assert_array_equal(stored["small"][0], source["small"][0])

    def test_cap_is_reproducible(self, cap_sources, tmp_path, strategy):
        h5_dir, _ = cap_sources
        for out in ("first", "second"):
            build_mmap(_cfg(h5_dir, tmp_path / out, max_instances=CAP, cap_strategy=strategy))
        first, second = _read_store(tmp_path / "first"), _read_store(tmp_path / "second")
        for sid in CAP_COUNTS:
            np.testing.assert_array_equal(first[sid][0], second[sid][0])
            np.testing.assert_array_equal(first[sid][1], second[sid][1])


def test_spatial_stratified_cap_keeps_sparse_regions(tmp_path):
    """A 15x15 cluster in one corner holds 225 of 257 patches; 16 two-patch islands
    spread over the rest of the slide hold the other 32.

    A uniform cap of 40 would keep about five island patches and lose most islands;
    the stratified cap keeps a patch on every island and gives the cluster well
    under its 88% share.
    """
    cluster = [(x, y) for x in range(15) for y in range(15)]
    island_of = {}  # lattice point -> island number
    for i, (x, y) in enumerate((x, y) for x in (20, 32, 44, 56) for y in (20, 32, 44, 56)):
        island_of[(x, y)] = island_of[(x + 1, y)] = i
    coords = np.array(cluster + list(island_of), dtype=np.int32) * STRIDE
    feats = np.zeros((len(coords), 4), np.float32)
    feats[:, 0] = np.arange(len(coords))
    _write_h5(tmp_path / "h5" / "slide.h5", feats, coords)

    build_mmap(_cfg(tmp_path / "h5", tmp_path / "mmap", max_instances=40, cap_grid_size=8))

    kept_feats, kept_coords = _read_store(tmp_path / "mmap")["slide"]
    assert len(np.unique(kept_feats[:, 0])) == 40
    lattice = kept_coords // STRIDE
    in_cluster = (lattice < 15).all(axis=1)
    assert in_cluster.mean() < 0.75
    assert {island_of[(int(x), int(y))] for x, y in lattice[~in_cluster]} == set(range(16))


def test_spatial_stratified_cap_keeps_every_isolated_patch(tmp_path):
    """400 patches crowd one cell of a 4x4 grid; each other cell holds one patch.
    With a cap of 20 all 15 single patches are kept: the crowded cell gives way."""
    cluster = [(x, y) for x in range(0, 40, 2) for y in range(0, 40, 2)]
    singles = [(50 + 100 * i, 50 + 100 * j) for i in range(4) for j in range(4) if i or j]
    coords = np.array(cluster + singles, dtype=np.int32)
    feats = np.zeros((len(coords), 4), np.float32)
    feats[:, 0] = np.arange(len(coords))
    _write_h5(tmp_path / "h5" / "slide.h5", feats, coords)

    build_mmap(_cfg(tmp_path / "h5", tmp_path / "mmap", max_instances=20, cap_grid_size=4))

    kept_feats, kept_coords = _read_store(tmp_path / "mmap")["slide"]
    assert len(np.unique(kept_feats[:, 0])) == 20
    assert set(singles) <= {(int(x), int(y)) for x, y in kept_coords}


# ═════════════════════════════════════════════════════════════════════════════
# Coverage QC
# ═════════════════════════════════════════════════════════════════════════════


class TestCoverageQC:
    @pytest.fixture(scope="class")
    def qc_dir(self, tmp_path_factory):
        """Four slides capped at 20 patches and one ("u") under the cap.

        coverage_montage_n=1 bounds how many coverage grids the build keeps in
        memory. The worst-covered slide, "a/s0" (22%), is written first, so it has
        to survive the pruning that the later slides trigger.
        """
        root = tmp_path_factory.mktemp("qc")
        rng = np.random.default_rng(4)
        for sid, n in {"a/s0": 90, "s1": 30, "s2": 40, "s3": 60, "u": 10}.items():
            _write_h5(root / "h5" / f"{sid}.h5", *_slide_arrays(n, rng))
        cfg = _cfg(
            root / "h5",
            root / "mmap",
            max_instances=20,
            save_coverage_thumbnails=True,
            coverage_montage_n=1,
        )
        build_mmap(cfg)
        return root / "mmap" / "coverage_qc"

    def test_stats_csv_has_one_row_per_slide(self, qc_dir):
        stats = pd.read_csv(qc_dir / "coverage_stats.csv").set_index("slide_id")
        assert sorted(stats.index) == ["a/s0", "s1", "s2", "s3", "u"]
        assert stats.loc["a/s0"].to_dict() == {
            "n_total": 90,
            "n_kept": 20,
            "n_discarded": 70,
            "coverage_pct": 22.22,
            "was_capped": True,
        }
        assert stats.loc["s1", "coverage_pct"] == 66.67
        assert stats.loc["u"].to_dict() == {
            "n_total": 10,
            "n_kept": 10,
            "n_discarded": 0,
            "coverage_pct": 100.0,
            "was_capped": False,
        }

    def test_thumbnails_for_the_worst_capped_slides_and_a_montage(self, qc_dir):
        thumbs = sorted((qc_dir / "thumbnails").iterdir())
        names = {p.name for p in thumbs}
        assert "a_s0.png" in names  # worst slide, nested id flattened into the file name
        assert "u.png" not in names  # not capped
        assert len(names) < 4  # bounded: not every capped slide keeps its grid
        for path in [*thumbs, qc_dir / "coverage_summary.png"]:
            _assert_png(path)

    def test_nothing_is_rendered_when_no_slide_is_capped(self, tmp_path):
        _write_h5(tmp_path / "h5" / "s.h5", *_slide_arrays(10, np.random.default_rng(5)))
        build_mmap(_cfg(tmp_path / "h5", tmp_path / "mmap", save_coverage_thumbnails=True))
        qc = tmp_path / "mmap" / "coverage_qc"
        assert pd.read_csv(qc / "coverage_stats.csv")["was_capped"].tolist() == [False]
        assert list((qc / "thumbnails").iterdir()) == []
        assert not (qc / "coverage_summary.png").exists()


class TestCoverageGrid:
    def test_patch_size_is_the_most_common_step(self):
        cols = np.array([0, 1, 2, 4, 5]) * 512  # one missing column
        xs, ys = np.meshgrid(cols, np.arange(3) * 512, indexing="ij")
        assert infer_patch_size(np.stack([xs.ravel(), ys.ravel()], axis=1)) == 512
        # A single column of patches: only y varies.
        assert infer_patch_size(np.array([[0, 0], [0, 224], [0, 448], [0, 896]])) == 224
        # Nothing to measure: fall back to 256.
        assert infer_patch_size(np.array([[100, 100]])) == 256
        assert infer_patch_size(np.array([[7, 7], [7, 7]])) == 256

    def test_grid_marks_kept_discarded_and_background(self):
        coords = np.array([[0, 0], [256, 0], [512, 256], [0, 512]]) + 1000
        grid, patch_size = build_coverage_grid(coords, kept_coords=coords[[0, 2]])
        assert patch_size == 256  # inferred
        # rows are y, columns x; 2 = kept, 1 = discarded, 0 = background
        expected = np.array([[2, 1, 0], [0, 0, 2], [1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(grid, expected)
        all_kept, _ = build_coverage_grid(coords, kept_coords=None, patch_size=256)
        np.testing.assert_array_equal(all_kept, np.where(expected > 0, 2, 0))

    def test_montage_of_several_rows_with_a_missing_grid(self, tmp_path):
        stats = [
            {
                "slide_id": f"dir/s{i}",
                "n_total": 100,
                "n_kept": 10 * (i + 1),
                "coverage_pct": 10.0 * (i + 1),
                "was_capped": True,
            }
            for i in range(6)
        ]
        grids = {f"dir/s{i}": np.eye(4, dtype=np.uint8) * 2 for i in range(5)}  # s5 has none
        render_summary_montage(stats, grids, tmp_path / "montage.png", n_worst=6)
        _assert_png(tmp_path / "montage.png")

        uncapped = [{**s, "was_capped": False} for s in stats]
        render_summary_montage(uncapped, grids, tmp_path / "none.png")
        assert not (tmp_path / "none.png").exists()


# ═════════════════════════════════════════════════════════════════════════════
# Readers, validation and failures
# ═════════════════════════════════════════════════════════════════════════════


def test_read_h5_coords_squeezes_and_casts(tmp_path):
    with h5py.File(tmp_path / "s.h5", "w") as f:
        f["coords"] = np.array([[[0.0, 256.0], [512.0, 768.0]]])  # (1, N, 2) floats
        f["flat"] = np.arange(4)
    coords = read_h5_coords(tmp_path / "s.h5")
    assert coords.dtype == np.int32
    np.testing.assert_array_equal(coords, [[0, 256], [512, 768]])
    with pytest.raises(ValueError, match="Expected 2D"):
        read_h5_coords(tmp_path / "s.h5", coord_key="flat")


def _drop(name):
    return lambda store: (store / name).unlink()


def _bump_schema(store):
    (store / ".schema_version").write_text(str(MMAP_SCHEMA_VERSION + 1))


def _numeric_slide_ids(store):
    with np.load(store / "index_arrays.npz", allow_pickle=True) as idx:
        arrays = dict(idx)
    arrays["slide_ids"] = np.arange(len(arrays["slide_ids"])).astype(object)
    np.savez(store / "index_arrays.npz", **arrays)


class TestValidateMmapDir:
    @pytest.mark.parametrize(
        ("damage", "error", "match"),
        [
            (_drop("features_002.bin"), FileNotFoundError, "features_002"),
            (_drop("coords_000.bin"), FileNotFoundError, "coords_000"),
            (_drop("index_arrays.npz"), FileNotFoundError, "Index not found"),
            (_bump_schema, ValueError, "schema version mismatch"),
            (_numeric_slide_ids, TypeError, "not all strings"),
        ],
        ids=["feature_chunk", "coord_chunk", "index", "schema", "slide_ids"],
    )
    def test_damaged_store_is_rejected(self, round_trip, tmp_path, damage, error, match):
        store = tmp_path / "store"
        shutil.copytree(round_trip.mmap_dir, store)
        validate_mmap_dir(store)  # intact copy passes
        damage(store)
        with pytest.raises(error, match=match):
            validate_mmap_dir(store)

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_mmap_dir(tmp_path / "nowhere")


class TestBuildFailures:
    def test_unknown_cap_strategy_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="cap_strategy"):
            build_mmap(_cfg(tmp_path, tmp_path / "out", cap_strategy="grid"))

    def test_no_valid_slide_is_an_error(self, tmp_path):
        _write_h5(tmp_path / "h5" / "empty.h5", np.zeros((0, 4)), np.zeros((0, 2)))
        with pytest.raises(RuntimeError, match="No valid slides"):
            build_mmap(_cfg(tmp_path / "h5", tmp_path / "out"))

    def test_slide_failing_mid_build_is_logged_and_the_rest_intact(self, tmp_path):
        """'b' passes the scan (shapes only) but its feature data lives in an external
        file that is gone, so reading it fails during the write pass; 'd' fails the
        scan. Both go to errors.txt and the slides around them read back intact."""
        rng = np.random.default_rng(7)
        h5 = tmp_path / "h5"
        source = {sid: _slide_arrays(n, rng) for sid, n in {"a": 9, "c": 11}.items()}
        for sid, (feats, coords) in source.items():
            _write_h5(h5 / f"{sid}.h5", feats, coords)
        with h5py.File(h5 / "b.h5", "w") as f:
            f.create_dataset(
                "features",
                shape=(5, FEAT_DIM),
                dtype="f4",
                external=[(str(tmp_path / "missing.bin"), 0, h5py.h5f.UNLIMITED)],
            )
            f["coords"] = np.zeros((5, 2), np.int32)
        _write_h5(h5 / "d.h5", np.zeros((0, FEAT_DIM)), np.zeros((0, 2)))

        result = build_mmap(_cfg(h5, tmp_path / "mmap"))

        assert result.n_errors == 2
        logged = (tmp_path / "mmap" / "errors.txt").read_text().splitlines()
        assert sorted(line.split("\t")[0] for line in logged) == ["b", "d"]
        index = load_mmap_index(tmp_path / "mmap")
        assert index["slide_ids"] == ["a", "b", "c"]
        assert index["lengths"].tolist() == [9, 0, 11]
        stored = _read_store(tmp_path / "mmap")
        for sid, (feats, coords) in source.items():
            np.testing.assert_array_equal(stored[sid][0], feats)
            np.testing.assert_array_equal(stored[sid][1], coords)
