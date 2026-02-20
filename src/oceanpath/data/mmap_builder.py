"""
Build chunked memmap stores from per-slide H5 feature files.

Converts many per-slide H5 files into contiguous binary stores + a
schema-versioned index for fast training-time loading.

Design:
  - Two-pass: scan (validate shapes, count patches) → write (stream data)
  - Streaming: never loads a full H5 into memory; reads in fixed-size chunks
  - Chunked output: splits .bin files at configurable size limits
  - Schema-versioned: downstream stages check version on read
  - Source-hashed: detects when H5 inputs change and mmap needs rebuild

File layout produced:
  {mmap_dir}/
  ├── features_000.bin [, features_001.bin, ...]
  ├── coords_000.bin   [, coords_001.bin, ...]
  ├── index_arrays.npz
  ├── .schema_version
  └── .source_hash
"""

import gc
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Bump when index format, binary layout, or metadata schema changes.
# Downstream stages (Dataset, DataModule) check this on read.
MMAP_SCHEMA_VERSION = 1


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class MmapBuildConfig:
    """All parameters needed to build a memmap store."""

    # Input
    h5_dir: str  # directory of per-slide .h5 files
    output_dir: str  # where to write .bin + index

    # H5 keys
    h5_feat_key: str = "features"
    h5_coord_key: str = "coords"

    # Precision
    feat_precision: int = 16  # 16 or 32
    coord_dtype: str = "int32"

    # Chunked output
    max_chunk_gb: float = 9.5

    # Streaming
    stream_chunk_size: int = 4096  # patches per H5 read

    # Optional
    max_instances: int | None = None  # cap patches per slide at build time
    csv_path: str | None = None  # filter to slides in this CSV
    csv_id_col: str = "slide_id"  # column name for slide ID in CSV
    csv_filename_col: str = "filename"  # column name for filename (stem → slide_id)

    @property
    def feat_dtype_str(self) -> str:
        return "float16" if self.feat_precision == 16 else "float32"

    @property
    def feat_np_dtype(self) -> np.dtype:
        return np.dtype("float16") if self.feat_precision == 16 else np.dtype("float32")

    @property
    def coord_np_dtype(self) -> np.dtype:
        return np.dtype(self.coord_dtype)

    @property
    def bytes_per_feat_element(self) -> int:
        return 2 if self.feat_precision == 16 else 4

    @property
    def max_chunk_bytes(self) -> int:
        return int(self.max_chunk_gb * 1024**3)


# ── Slide scanner (pass 1) ────────────────────────────────────────────────────


@dataclass
class SlideInfo:
    """Metadata collected during scan pass."""

    slide_id: str
    h5_path: str
    n_patches: int
    feat_dim: int
    coord_dim: int


class ScanError:
    """A slide that failed validation during scan."""

    def __init__(self, slide_id: str, h5_path: str, reason: str):
        self.slide_id = slide_id
        self.h5_path = h5_path
        self.reason = reason


def scan_h5_dir(cfg: MmapBuildConfig) -> tuple[list[SlideInfo], list[ScanError]]:
    """
    Pass 1: scan all H5 files, collect shapes, validate consistency.

    Only reads H5 dataset shapes — never loads feature data into memory.

    Returns
    -------
    slides : list[SlideInfo]
        Successfully validated slides, sorted by slide_id.
    errors : list[ScanError]
        Slides that failed validation.
    """
    h5_dir = Path(cfg.h5_dir)

    if not h5_dir.is_dir():
        raise FileNotFoundError(f"H5 directory does not exist: {h5_dir}")

    # Discover H5 files
    h5_paths = sorted(h5_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    # Optional CSV filter: only include slides present in the manifest
    allowed_ids = None
    if cfg.csv_path:
        allowed_ids = _load_allowed_slide_ids(cfg)
        logger.info(f"CSV filter active: {len(allowed_ids)} slide IDs from {cfg.csv_path}")

    slides: list[SlideInfo] = []
    errors: list[ScanError] = []

    for h5_path in tqdm(h5_paths, desc="Scanning H5s", unit="file"):
        slide_id = h5_path.stem

        # CSV filter
        if allowed_ids is not None and slide_id not in allowed_ids:
            continue

        try:
            with h5py.File(str(h5_path), "r") as f:
                if cfg.h5_feat_key not in f:
                    errors.append(
                        ScanError(slide_id, str(h5_path), f"Key '{cfg.h5_feat_key}' not found")
                    )
                    continue
                if cfg.h5_coord_key not in f:
                    errors.append(
                        ScanError(slide_id, str(h5_path), f"Key '{cfg.h5_coord_key}' not found")
                    )
                    continue

                feat_shape = f[cfg.h5_feat_key].shape
                coord_shape = f[cfg.h5_coord_key].shape

            # Normalize shapes: handle (1, N, D) → (N, D)
            if len(feat_shape) == 3 and feat_shape[0] == 1:
                feat_shape = (feat_shape[1], feat_shape[2])
            if len(coord_shape) == 3 and coord_shape[0] == 1:
                coord_shape = (coord_shape[1], coord_shape[2])

            if len(feat_shape) != 2:
                errors.append(
                    ScanError(
                        slide_id, str(h5_path), f"Features expected 2D, got shape {feat_shape}"
                    )
                )
                continue
            if len(coord_shape) != 2:
                errors.append(
                    ScanError(
                        slide_id, str(h5_path), f"Coords expected 2D, got shape {coord_shape}"
                    )
                )
                continue

            n_patches_f, feat_dim = feat_shape
            n_patches_c, coord_dim = coord_shape

            if n_patches_f != n_patches_c:
                errors.append(
                    ScanError(
                        slide_id,
                        str(h5_path),
                        f"Patch count mismatch: feats={n_patches_f}, coords={n_patches_c}",
                    )
                )
                continue

            if n_patches_f == 0:
                errors.append(ScanError(slide_id, str(h5_path), "Zero patches"))
                continue

            # Apply max_instances cap (determines stored count)
            n_stored = n_patches_f
            if cfg.max_instances and n_patches_f > cfg.max_instances:
                n_stored = cfg.max_instances

            slides.append(
                SlideInfo(
                    slide_id=slide_id,
                    h5_path=str(h5_path),
                    n_patches=n_stored,
                    feat_dim=feat_dim,
                    coord_dim=coord_dim,
                )
            )

        except Exception as e:
            errors.append(ScanError(slide_id, str(h5_path), repr(e)))

    # Validate consistent feature dim across all slides
    if slides:
        dims = {s.feat_dim for s in slides}
        if len(dims) > 1:
            raise ValueError(
                f"Inconsistent feature dimensions across slides: {dims}. "
                f"All H5 files must have the same D."
            )

    slides.sort(key=lambda s: s.slide_id)
    return slides, errors


def _load_allowed_slide_ids(cfg: MmapBuildConfig) -> set[str]:
    """Load slide IDs from a CSV manifest for filtering."""
    import pandas as pd

    df = pd.read_csv(cfg.csv_path)

    # Try slide_id column first, fall back to stem(filename)
    if cfg.csv_id_col in df.columns:
        return set(df[cfg.csv_id_col].astype(str).tolist())
    if cfg.csv_filename_col in df.columns:
        return {Path(fn).stem for fn in df[cfg.csv_filename_col].astype(str)}
    raise ValueError(
        f"CSV {cfg.csv_path} has neither '{cfg.csv_id_col}' nor "
        f"'{cfg.csv_filename_col}' column. Available: {list(df.columns)}"
    )


# ── Chunked binary writer ────────────────────────────────────────────────────


class ChunkedBinaryWriter:
    """
    Writes numpy arrays to chunked binary files with size limits.

    Each write returns (chunk_id, byte_offset) for the index.
    When current file would exceed max_bytes, a new chunk is opened.
    """

    def __init__(self, output_dir: str, prefix: str, max_bytes: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes

        self.current_chunk_id = 0
        self.current_file = None
        self.current_bytes = 0
        self.chunk_files: list[str] = []

        self._open_new_chunk()

    def _chunk_path(self, chunk_id: int) -> str:
        return os.path.join(self.output_dir, f"{self.prefix}_{chunk_id:03d}.bin")

    def _open_new_chunk(self) -> None:
        if self.current_file is not None:
            self.current_file.flush()
            self.current_file.close()

        path = self._chunk_path(self.current_chunk_id)
        # Suppress SIM115: File lifecycle is managed by the class instance
        self.current_file = open(path, "wb")  # noqa: SIM115
        self.chunk_files.append(path)
        self.current_bytes = 0

    def write(self, data: np.ndarray) -> tuple[int, int]:
        """
        Write array to current chunk.

        Returns (chunk_id, byte_offset_within_chunk).
        Starts a new chunk if adding data would exceed the limit.
        """
        data_bytes = data.nbytes

        # Start new chunk if needed (never split a single slide)
        if self.current_bytes > 0 and (self.current_bytes + data_bytes) > self.max_bytes:
            self.current_chunk_id += 1
            self._open_new_chunk()

        chunk_id = self.current_chunk_id
        offset = self.current_bytes

        data.tofile(self.current_file)
        self.current_bytes += data_bytes

        return chunk_id, offset

    def write_streamed(
        self,
        h5_dataset,
        offset_start: int,
        n_rows: int,
        dtype: np.dtype,
        chunk_size: int,
        subsample_indices: np.ndarray | None = None,
    ) -> tuple[int, int]:
        """
        Stream data from an H5 dataset directly to the binary file.

        Reads `chunk_size` rows at a time from the H5 dataset,
        converts dtype, and writes immediately. Never holds more
        than `chunk_size` rows in memory.

        Parameters
        ----------
        h5_dataset : h5py.Dataset
            Open H5 dataset to read from.
        offset_start : int
            Row offset in the H5 dataset (for pre-squeezed data).
        n_rows : int
            Number of rows to read total.
        dtype : np.dtype
            Target dtype for type conversion.
        chunk_size : int
            Rows per streaming read.
        subsample_indices : np.ndarray, optional
            If provided, only these row indices are written (sorted).
            Used when max_instances < n_patches.

        Returns
        -------
        chunk_id : int
            Binary chunk this slide landed in.
        byte_offset : int
            Byte offset within that chunk.
        """
        if subsample_indices is not None:
            # Subsampled: read only selected rows in sorted order.
            # For small max_instances, this is memory-efficient enough.
            sorted_idx = np.sort(subsample_indices)
            total_bytes = int(sorted_idx.shape[0]) * h5_dataset.shape[-1] * dtype.itemsize

            # Check if we need a new chunk
            if self.current_bytes > 0 and (self.current_bytes + total_bytes) > self.max_bytes:
                self.current_chunk_id += 1
                self._open_new_chunk()

            chunk_id = self.current_chunk_id
            byte_offset = self.current_bytes

            # Read in batches of indices to avoid huge fancy-indexing overhead
            for batch_start in range(0, len(sorted_idx), chunk_size):
                batch_idx = sorted_idx[batch_start : batch_start + chunk_size]
                block = h5_dataset[batch_idx]
                if block.ndim == 3 and block.shape[0] == 1:
                    block = block[0]
                block = np.ascontiguousarray(block.astype(dtype, copy=False))
                block.tofile(self.current_file)
                self.current_bytes += block.nbytes

            return chunk_id, byte_offset

        # Full slide: stream in sequential chunks
        D = h5_dataset.shape[-1]
        total_bytes = n_rows * D * dtype.itemsize

        if self.current_bytes > 0 and (self.current_bytes + total_bytes) > self.max_bytes:
            self.current_chunk_id += 1
            self._open_new_chunk()

        chunk_id = self.current_chunk_id
        byte_offset = self.current_bytes

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            block = h5_dataset[offset_start + start : offset_start + end]
            if block.ndim == 3 and block.shape[0] == 1:
                block = block[0]
            block = np.ascontiguousarray(block.astype(dtype, copy=False))
            block.tofile(self.current_file)
            self.current_bytes += block.nbytes

        return chunk_id, byte_offset

    def close(self) -> None:
        if self.current_file is not None:
            self.current_file.flush()
            self.current_file.close()
            self.current_file = None

    @property
    def num_chunks(self) -> int:
        return self.current_chunk_id + 1


# ── Source hashing ────────────────────────────────────────────────────────────


def compute_source_hash(slides: list[SlideInfo], cfg: MmapBuildConfig) -> str:
    """
    Deterministic hash of the H5 source + build config.

    Changes when:
      - H5 files are added/removed
      - Build precision or max_instances changes
      - H5 file modification times change (optional)

    Used for cache invalidation: if source hash matches existing
    mmap, skip rebuild.
    """
    h = hashlib.sha256()

    # Hash slide list (sorted IDs + patch counts)
    for s in slides:
        h.update(f"{s.slide_id}:{s.n_patches}:{s.feat_dim}".encode())

    # Hash build config
    h.update(f"precision={cfg.feat_precision}".encode())
    h.update(f"max_instances={cfg.max_instances}".encode())
    h.update(f"coord_dtype={cfg.coord_dtype}".encode())
    h.update(f"h5_feat_key={cfg.h5_feat_key}".encode())
    h.update(f"h5_coord_key={cfg.h5_coord_key}".encode())

    return h.hexdigest()[:20]


# ── Main build function ──────────────────────────────────────────────────────


@dataclass
class BuildResult:
    """Summary returned after a successful build."""

    output_dir: str
    n_slides: int
    n_errors: int
    total_patches: int
    feat_dim: int
    feat_chunks: int
    coord_chunks: int
    elapsed_seconds: float
    source_hash: str


def build_mmap(cfg: MmapBuildConfig, force: bool = False) -> BuildResult:
    """
    Build chunked memmap store from H5 feature files.

    Two-pass approach:
      1. Scan: validate all H5s, collect shapes, check consistency
      2. Write: stream data from H5s into chunked binary files

    Parameters
    ----------
    cfg : MmapBuildConfig
        Complete build configuration.
    force : bool
        Rebuild even if source hash matches existing mmap.

    Returns
    -------
    BuildResult
        Summary of the build.
    """
    start_time = time.monotonic()
    output_dir = Path(cfg.output_dir)

    # ── Pass 1: Scan ──────────────────────────────────────────────────────
    logger.info(f"Scanning H5 files in {cfg.h5_dir} ...")
    slides, errors = scan_h5_dir(cfg)

    if not slides:
        raise RuntimeError(f"No valid slides found in {cfg.h5_dir}. Errors: {len(errors)}")

    feat_dim = slides[0].feat_dim
    total_patches = sum(s.n_patches for s in slides)

    logger.info(
        f"Scan complete: {len(slides)} slides, {total_patches:,} patches, "
        f"D={feat_dim}, {len(errors)} errors"
    )

    # ── Check source hash (skip if unchanged) ─────────────────────────────
    source_hash = compute_source_hash(slides, cfg)
    existing_hash_path = output_dir / ".source_hash"

    if not force and existing_hash_path.is_file():
        existing_hash = existing_hash_path.read_text().strip()
        if existing_hash == source_hash:
            logger.info(
                f"Source hash matches ({source_hash[:12]}...) — mmap is up to date. "
                f"Use force=true to rebuild."
            )
            return BuildResult(
                output_dir=str(output_dir),
                n_slides=len(slides),
                n_errors=len(errors),
                total_patches=total_patches,
                feat_dim=feat_dim,
                feat_chunks=0,
                coord_chunks=0,
                elapsed_seconds=time.monotonic() - start_time,
                source_hash=source_hash,
            )

    # ── Prepare output directory ──────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old chunk files (avoid stale leftovers from previous builds)
    for old_bin in output_dir.glob("features_*.bin"):
        old_bin.unlink()
    for old_bin in output_dir.glob("coords_*.bin"):
        old_bin.unlink()

    # ── Pass 2: Stream + Write ────────────────────────────────────────────
    logger.info(f"Writing memmap to {output_dir} ...")

    feat_writer = ChunkedBinaryWriter(str(output_dir), "features", cfg.max_chunk_bytes)
    coord_writer = ChunkedBinaryWriter(str(output_dir), "coords", cfg.max_chunk_bytes)

    n_slides = len(slides)
    feat_chunk_ids = np.full(n_slides, -1, dtype=np.int16)
    feat_offsets = np.full(n_slides, -1, dtype=np.int64)
    coord_chunk_ids = np.full(n_slides, -1, dtype=np.int16)
    coord_offsets = np.full(n_slides, -1, dtype=np.int64)
    lengths = np.zeros(n_slides, dtype=np.int32)
    slide_ids_out = np.array([s.slide_id for s in slides], dtype=object)

    rng = np.random.RandomState(42)  # deterministic subsampling

    for i, slide in enumerate(tqdm(slides, desc="Writing chunks", unit="slide")):
        try:
            with h5py.File(slide.h5_path, "r") as f:
                feat_ds = f[cfg.h5_feat_key]
                coord_ds = f[cfg.h5_coord_key]

                # Determine actual patch count in H5
                raw_n = feat_ds.shape[0]
                if feat_ds.ndim == 3 and feat_ds.shape[0] == 1:
                    raw_n = feat_ds.shape[1]

                # Subsample indices if needed
                subsample_idx = None
                if cfg.max_instances and raw_n > cfg.max_instances:
                    subsample_idx = rng.permutation(raw_n)[: cfg.max_instances]

                # Stream features
                f_chunk, f_offset = feat_writer.write_streamed(
                    h5_dataset=feat_ds,
                    offset_start=0,
                    n_rows=raw_n,
                    dtype=cfg.feat_np_dtype,
                    chunk_size=cfg.stream_chunk_size,
                    subsample_indices=subsample_idx,
                )

                # Stream coords (same subsample indices for sync)
                c_chunk, c_offset = coord_writer.write_streamed(
                    h5_dataset=coord_ds,
                    offset_start=0,
                    n_rows=raw_n,
                    dtype=cfg.coord_np_dtype,
                    chunk_size=cfg.stream_chunk_size,
                    subsample_indices=subsample_idx,
                )

            feat_chunk_ids[i] = f_chunk
            feat_offsets[i] = f_offset
            coord_chunk_ids[i] = c_chunk
            coord_offsets[i] = c_offset
            lengths[i] = slide.n_patches

        except Exception as e:
            errors.append(ScanError(slide.slide_id, slide.h5_path, repr(e)))
            logger.warning(f"Failed to write slide {slide.slide_id}: {e}")

        # Periodic GC to prevent memory creep
        if i % 200 == 0 and i > 0:
            gc.collect()

    feat_writer.close()
    coord_writer.close()

    # ── Write index ───────────────────────────────────────────────────────
    index_path = output_dir / "index_arrays.npz"
    np.savez(
        str(index_path),
        # Schema
        schema_version=np.int32(MMAP_SCHEMA_VERSION),
        # Slide info
        slide_ids=slide_ids_out,
        lengths=lengths,
        # Feature indexing (byte offsets within chunks)
        feat_chunk_ids=feat_chunk_ids,
        feat_offsets=feat_offsets,
        # Coord indexing
        coord_chunk_ids=coord_chunk_ids,
        coord_offsets=coord_offsets,
        # Metadata
        feat_dim=np.int32(feat_dim),
        feat_dtype=np.array([cfg.feat_dtype_str], dtype=object),
        coord_dtype=np.array([cfg.coord_dtype], dtype=object),
        coord_dim=np.int32(slides[0].coord_dim),
        bytes_per_feat=np.int32(cfg.bytes_per_feat_element),
        # Chunk counts
        num_feat_chunks=np.int32(feat_writer.num_chunks),
        num_coord_chunks=np.int32(coord_writer.num_chunks),
        # Totals
        total_patches=np.int64(total_patches),
        n_slides=np.int32(n_slides),
    )
    logger.info(f"Index saved: {index_path}")

    # ── Write metadata files ──────────────────────────────────────────────
    (output_dir / ".schema_version").write_text(str(MMAP_SCHEMA_VERSION))
    (output_dir / ".source_hash").write_text(source_hash)

    # ── Write error log ───────────────────────────────────────────────────
    if errors:
        err_path = output_dir / "errors.txt"
        with open(err_path, "w") as f:
            for e in errors:
                f.write(f"{e.slide_id}: {e.reason}\n")
        logger.warning(f"{len(errors)} errors logged to {err_path}")

    elapsed = time.monotonic() - start_time

    # ── Summary ───────────────────────────────────────────────────────────
    result = BuildResult(
        output_dir=str(output_dir),
        n_slides=len(slides),
        n_errors=len(errors),
        total_patches=total_patches,
        feat_dim=feat_dim,
        feat_chunks=feat_writer.num_chunks,
        coord_chunks=coord_writer.num_chunks,
        elapsed_seconds=elapsed,
        source_hash=source_hash,
    )

    logger.info(
        f"Build complete in {elapsed:.1f}s: "
        f"{result.n_slides} slides, {result.total_patches:,} patches, "
        f"D={result.feat_dim}, "
        f"{result.feat_chunks} feat chunks, {result.coord_chunks} coord chunks"
    )

    # Log chunk file sizes
    for path in feat_writer.chunk_files:
        size_gb = os.path.getsize(path) / (1024**3)
        logger.info(f"  {os.path.basename(path)}: {size_gb:.2f} GB")

    return result


# ── Validation (for downstream stages) ────────────────────────────────────────


def validate_mmap_dir(mmap_dir: str | Path) -> dict:
    """
    Validate that a memmap directory contains a valid, complete store.

    Returns metadata dict if valid, raises on failure.
    Used by Dataset/DataModule at init time.
    """
    mmap_dir = Path(mmap_dir)

    if not mmap_dir.is_dir():
        raise FileNotFoundError(f"Memmap directory not found: {mmap_dir}")

    # Check schema version
    schema_path = mmap_dir / ".schema_version"
    if schema_path.is_file():
        version = int(schema_path.read_text().strip())
        if version != MMAP_SCHEMA_VERSION:
            raise ValueError(
                f"Memmap schema version mismatch: expected {MMAP_SCHEMA_VERSION}, "
                f"got {version}. Rebuild with: python scripts/build_mmap.py force=true"
            )

    # Check index exists and load metadata
    index_path = mmap_dir / "index_arrays.npz"
    if not index_path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")

    idx = np.load(str(index_path), allow_pickle=True)

    n_feat_chunks = int(idx["num_feat_chunks"])
    n_coord_chunks = int(idx["num_coord_chunks"])

    # Check all chunk files exist
    for i in range(n_feat_chunks):
        p = mmap_dir / f"features_{i:03d}.bin"
        if not p.is_file():
            raise FileNotFoundError(f"Missing feature chunk: {p}")

    for i in range(n_coord_chunks):
        p = mmap_dir / f"coords_{i:03d}.bin"
        if not p.is_file():
            raise FileNotFoundError(f"Missing coord chunk: {p}")

    return {
        "schema_version": int(idx.get("schema_version", 0)),
        "n_slides": int(idx["n_slides"]),
        "total_patches": int(idx["total_patches"]),
        "feat_dim": int(idx["feat_dim"]),
        "feat_dtype": str(idx["feat_dtype"][0]),
        "coord_dim": int(idx.get("coord_dim", 2)),
        "num_feat_chunks": n_feat_chunks,
        "num_coord_chunks": n_coord_chunks,
    }
