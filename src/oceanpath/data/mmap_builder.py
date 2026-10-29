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
  - Inline QC: coverage thumbnails generated during write pass (zero extra I/O)

Slide ID convention:
  slide_id = relative path from h5_dir to the .h5 file, minus the .h5 extension.
  Examples:
    h5_dir/foo.h5           → slide_id = "foo"
    h5_dir/a/b/foo.h5       → slide_id = "a/b/foo"
  To reconstruct the H5 path: Path(h5_dir) / f"{slide_id}.h5"
  POSIX separators are always used regardless of OS.

File layout produced:
  {mmap_dir}/
  ├── features_000.bin [, features_001.bin, ...]
  ├── coords_000.bin   [, coords_001.bin, ...]
  ├── index_arrays.npz
  ├── .schema_version
  ├── .source_hash
  └── coverage_qc/                    (if save_coverage_thumbnails=True)
      ├── thumbnails/{slide_id}.png
      ├── coverage_stats.csv
      └── coverage_summary.png
"""

from __future__ import annotations

import csv as csv_mod
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

_VALID_CAP_STRATEGIES = frozenset({"random", "spatial_stratified"})


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════


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

    # Build-time capping strategy (only used when max_instances is set)
    #   "spatial_stratified" — √n-proportional grid allocation (preserves coverage + density)
    #   "random"             — uniform random subsample (original behavior)
    cap_strategy: str = "spatial_stratified"
    cap_grid_size: int = 32  # grid resolution for spatial_stratified

    # Coverage QC: generate per-slide spatial thumbnails during build
    save_coverage_thumbnails: bool = True
    coverage_montage_n: int = 20  # slides in summary montage

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


# ═════════════════════════════════════════════════════════════════════════════
# Slide scanner (pass 1)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class SlideInfo:
    """Metadata collected during scan pass."""

    slide_id: str
    h5_path: str
    n_patches: int  # stored count (after cap)
    raw_patches: int  # original count in H5
    feat_dim: int
    coord_dim: int


class ScanError:
    """A slide that failed validation during scan."""

    def __init__(self, slide_id: str, h5_path: str, reason: str):
        self.slide_id = slide_id
        self.h5_path = h5_path
        self.reason = reason


def _derive_slide_id(h5_path: Path, h5_dir: Path) -> str:
    """
    Derive slide_id from the relative path of the H5 file under h5_dir.

    Always uses POSIX separators for cross-platform consistency.
    """
    return h5_path.relative_to(h5_dir).with_suffix("").as_posix()


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

    h5_paths = sorted(h5_dir.rglob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    # Optional CSV filter
    allowed_ids = None
    if cfg.csv_path:
        allowed_ids = _load_allowed_slide_ids(cfg)
        logger.info(f"CSV filter active: {len(allowed_ids)} slide IDs from {cfg.csv_path}")

    slides: list[SlideInfo] = []
    errors: list[ScanError] = []
    seen_ids: set[str] = set()

    for h5_path in tqdm(h5_paths, desc="Scanning H5s", unit="file"):
        slide_id = _derive_slide_id(h5_path, h5_dir)

        if slide_id in seen_ids:
            errors.append(
                ScanError(slide_id, str(h5_path), "Duplicate slide_id (possible symlink loop)")
            )
            continue
        seen_ids.add(slide_id)

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

            n_stored = n_patches_f
            if cfg.max_instances and n_patches_f > cfg.max_instances:
                n_stored = cfg.max_instances

            slides.append(
                SlideInfo(
                    slide_id=slide_id,
                    h5_path=str(h5_path),
                    n_patches=n_stored,
                    raw_patches=n_patches_f,
                    feat_dim=feat_dim,
                    coord_dim=coord_dim,
                )
            )

        except Exception as e:
            errors.append(ScanError(slide_id, str(h5_path), repr(e)))

    # Validate consistent feature dim
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

    if cfg.csv_id_col in df.columns:
        return {Path(sid).as_posix() for sid in df[cfg.csv_id_col].astype(str)}
    if cfg.csv_filename_col in df.columns:
        ids = set()
        for fn in df[cfg.csv_filename_col].astype(str):
            p = Path(fn)
            if p.suffix == ".h5":
                ids.add(p.with_suffix("").as_posix())
            else:
                ids.add(p.as_posix())
        return ids
    raise ValueError(
        f"CSV {cfg.csv_path} has neither '{cfg.csv_id_col}' nor "
        f"'{cfg.csv_filename_col}' column. Available: {list(df.columns)}"
    )


# ── Helper: reconstruct H5 path from slide_id ────────────────────────────────


def slide_id_to_h5_path(h5_dir: str | Path, slide_id: str) -> Path:
    """Reconstruct the H5 file path from a slide_id and h5_dir."""
    return Path(h5_dir) / f"{slide_id}.h5"


def _read_h5_2d(dataset) -> np.ndarray:
    """
    Read an H5 dataset and squeeze to 2D.

    All H5 files in the pipeline store features as (1, N, D) and coords
    as (1, N, 2).  This helper reads the full dataset and squeezes the
    leading dimension, returning a contiguous (N, D) array.

    Also handles the rare (N, D) case for forward compatibility.
    """
    data = dataset[:]
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D after squeeze, got shape {data.shape} (raw shape: {dataset.shape})"
        )
    return np.ascontiguousarray(data)


# ═════════════════════════════════════════════════════════════════════════════
# Chunked binary writer
# ═════════════════════════════════════════════════════════════════════════════


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
        self.current_file = open(path, "wb")  # noqa: SIM115
        self.chunk_files.append(path)
        self.current_bytes = 0

    def write(self, data: np.ndarray) -> tuple[int, int]:
        """Write array to current chunk. Returns (chunk_id, byte_offset)."""
        data_bytes = data.nbytes

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

        Reads `chunk_size` rows at a time, converts dtype, writes immediately.
        Never holds more than `chunk_size` rows in memory.

        Handles both (N, D) and (1, N, D) shaped H5 datasets transparently.
        """
        # ── Detect leading-1 dimension ────────────────────────────────
        # Some feature extractors write (1, N, D) instead of (N, D).
        # We need to know this BEFORE indexing to avoid out-of-range errors.
        has_leading_dim = h5_dataset.ndim == 3 and h5_dataset.shape[0] == 1

        if subsample_indices is not None:
            sorted_idx = np.sort(subsample_indices)
            D = h5_dataset.shape[-1]
            total_bytes = int(sorted_idx.shape[0]) * D * dtype.itemsize

            if self.current_bytes > 0 and (self.current_bytes + total_bytes) > self.max_bytes:
                self.current_chunk_id += 1
                self._open_new_chunk()

            chunk_id = self.current_chunk_id
            byte_offset = self.current_bytes

            for batch_start in range(0, len(sorted_idx), chunk_size):
                batch_idx = sorted_idx[batch_start : batch_start + chunk_size]
                if has_leading_dim:
                    # Index as dataset[0, [indices]] to skip the size-1 axis
                    block = h5_dataset[0, batch_idx]
                else:
                    block = h5_dataset[batch_idx]
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
            if has_leading_dim:
                block = h5_dataset[0, offset_start + start : offset_start + end]
            else:
                block = h5_dataset[offset_start + start : offset_start + end]
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


# ═════════════════════════════════════════════════════════════════════════════
# Source hashing
# ═════════════════════════════════════════════════════════════════════════════


def compute_source_hash(slides: list[SlideInfo], cfg: MmapBuildConfig) -> str:
    """Deterministic hash of the H5 source + build config for cache invalidation."""
    h = hashlib.sha256()

    for s in slides:
        h.update(f"{s.slide_id}:{s.n_patches}:{s.feat_dim}".encode())

    h.update(f"precision={cfg.feat_precision}".encode())
    h.update(f"max_instances={cfg.max_instances}".encode())
    h.update(f"cap_strategy={cfg.cap_strategy}".encode())
    h.update(f"cap_grid_size={cfg.cap_grid_size}".encode())
    h.update(f"coord_dtype={cfg.coord_dtype}".encode())
    h.update(f"h5_feat_key={cfg.h5_feat_key}".encode())
    h.update(f"h5_coord_key={cfg.h5_coord_key}".encode())

    return h.hexdigest()[:20]


# ═════════════════════════════════════════════════════════════════════════════
# Build-time capping strategies
# ═════════════════════════════════════════════════════════════════════════════


def _spatial_stratified_subsample(
    coords: np.ndarray,
    n_samples: int,
    grid_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Spatially stratified subsampling with sqrt(n)-proportional allocation.

    Divides the coordinate space into a grid_size x grid_size grid and
    allocates per-cell budgets proportional to sqrt(cell_count).  Each
    occupied cell is guaranteed at least 1 patch.

    Returns sorted indices of selected patches (length == n_samples).
    """
    N = len(coords)
    if n_samples >= N:
        return np.arange(N)

    # Grid binning
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = (cmax - cmin).astype(np.float64) + 1e-6
    cell_sz = span / grid_size

    cx = np.clip(((coords[:, 0] - cmin[0]) / cell_sz[0]).astype(int), 0, grid_size - 1)
    cy = np.clip(((coords[:, 1] - cmin[1]) / cell_sz[1]).astype(int), 0, grid_size - 1)
    cell_keys = cx * grid_size + cy

    unique_cells = np.unique(cell_keys)
    cell_groups = {int(c): np.where(cell_keys == c)[0] for c in unique_cells}
    n_cells = len(cell_groups)

    # Edge case: more occupied cells than budget
    if n_cells >= n_samples:
        return np.sort(rng.permutation(N)[:n_samples])

    # sqrt(n)-proportional budget allocation
    cells = list(cell_groups.keys())
    cell_sizes = np.array([len(cell_groups[c]) for c in cells])
    weights = np.sqrt(cell_sizes.astype(np.float64))

    ideal = n_samples * weights / weights.sum()
    budgets = np.clip(np.floor(ideal).astype(int), 1, cell_sizes)

    # Balance to exactly n_samples
    deficit = n_samples - int(budgets.sum())

    if deficit > 0:
        spare = cell_sizes - budgets
        remainders = ideal - budgets.astype(np.float64)
        remainders[spare == 0] = -1.0
        order = np.argsort(-remainders)
        for j in order:
            if deficit == 0:
                break
            give = min(deficit, int(spare[j]))
            if give > 0:
                budgets[j] += give
                deficit -= give
    elif deficit < 0:
        surplus = -deficit
        order = np.argsort(-budgets)
        for j in order:
            if surplus == 0:
                break
            trim = min(surplus, int(budgets[j]) - 1)
            if trim > 0:
                budgets[j] -= trim
                surplus -= trim

    # Sample within cells
    selected = []
    for i, c in enumerate(cells):
        idx = cell_groups[c]
        b = int(budgets[i])
        if b >= len(idx):
            selected.append(idx)
        else:
            selected.append(rng.choice(idx, size=b, replace=False))

    return np.sort(np.concatenate(selected))


# ═════════════════════════════════════════════════════════════════════════════
# Coverage QC visualization (shared between build-time and standalone CLI)
# ═════════════════════════════════════════════════════════════════════════════


def infer_patch_size(coords: np.ndarray) -> int:
    """
    Infer the patch stride from coordinate differences.

    Looks at the most common nonzero difference between adjacent x-coords
    when sorted. Falls back to 256 if detection fails.
    """
    if len(coords) < 2:
        return 256

    order = np.lexsort((coords[:, 1], coords[:, 0]))
    sorted_coords = coords[order]

    dx = np.diff(sorted_coords[:, 0])
    dx_nonzero = dx[dx > 0]

    if len(dx_nonzero) == 0:
        dy = np.diff(sorted_coords[:, 1])
        dy_nonzero = dy[dy > 0]
        if len(dy_nonzero) == 0:
            return 256
        vals, counts = np.unique(dy_nonzero, return_counts=True)
        return int(vals[counts.argmax()])

    vals, counts = np.unique(dx_nonzero, return_counts=True)
    return int(vals[counts.argmax()])


def build_coverage_grid(
    all_coords: np.ndarray,
    kept_coords: np.ndarray | None,
    patch_size: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Build a spatial coverage grid from coordinates.

    Parameters
    ----------
    all_coords : (N_total, 2)
        All patch positions from original H5.
    kept_coords : (N_kept, 2) or None
        Patch positions stored in mmap. If None, all patches are kept.
    patch_size : int or None
        Inferred if None.

    Returns
    -------
    grid : np.ndarray, dtype uint8
        2D grid where 0=background, 1=discarded, 2=kept.
    patch_size : int
        The patch stride used.
    """
    if patch_size is None:
        patch_size = infer_patch_size(all_coords)

    xy_min = all_coords.min(axis=0)

    # Vectorized grid coordinate computation
    all_gx = ((all_coords[:, 0] - xy_min[0]) // patch_size).astype(np.intp)
    all_gy = ((all_coords[:, 1] - xy_min[1]) // patch_size).astype(np.intp)

    grid_w = int(all_gx.max()) + 1
    grid_h = int(all_gy.max()) + 1
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    if kept_coords is None:
        # All kept — vectorized scatter
        grid[all_gy, all_gx] = 2
    else:
        # Mark all as discarded, then overwrite kept — fully vectorized
        grid[all_gy, all_gx] = 1
        kept_gx = ((kept_coords[:, 0] - xy_min[0]) // patch_size).astype(np.intp)
        kept_gy = ((kept_coords[:, 1] - xy_min[1]) // patch_size).astype(np.intp)
        grid[kept_gy, kept_gx] = 2

    return grid, patch_size


def _get_coverage_cmap():
    """
    Return a singleton colormap for coverage visualization.

    Creates it once and caches — avoids matplotlib's colormap registry
    leak when ListedColormap() is called repeatedly.
    """
    if not hasattr(_get_coverage_cmap, "_cmap"):
        from matplotlib.colors import ListedColormap

        _get_coverage_cmap._cmap = ListedColormap(
            ["#f8f9fa", "#ef4444", "#22c55e"], name="_coverage_qc"
        )
    return _get_coverage_cmap._cmap


def render_coverage_thumbnail(
    grid: np.ndarray,
    slide_id: str,
    n_total: int,
    n_kept: int,
    cmap=None,
    max_thumb_size: int = 800,
    dpi: int = 100,
):
    """
    Render a spatial coverage map for one slide.

    Parameters
    ----------
    grid : (H, W) uint8
        Coverage grid (0=bg, 1=discarded, 2=kept).
    slide_id : str
    n_total : int
    n_kept : int
    cmap : matplotlib colormap, optional
        Reuse a single colormap across calls to avoid registry leak.
        If None, uses the shared singleton.
    max_thumb_size : int
    dpi : int

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if cmap is None:
        cmap = _get_coverage_cmap()

    n_discarded = n_total - n_kept
    coverage_pct = 100 * n_kept / n_total if n_total > 0 else 0
    was_capped = n_kept < n_total

    grid_h, grid_w = grid.shape
    aspect = grid_h / grid_w if grid_w > 0 else 1
    fig_w = min(max_thumb_size / dpi, 8)
    fig_h = fig_w * aspect

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", origin="upper")
    ax.set_axis_off()

    ax.set_title(
        f"{slide_id}\n"
        f"Total: {n_total:,}  Kept: {n_kept:,}  Discarded: {n_discarded:,}  "
        f"Coverage: {coverage_pct:.1f}%",
        fontsize=8,
        fontfamily="monospace",
        pad=6,
    )

    legend_elements = [
        Patch(facecolor="#22c55e", edgecolor="none", label=f"Kept ({n_kept:,})"),
    ]
    if was_capped:
        legend_elements.append(
            Patch(facecolor="#ef4444", edgecolor="none", label=f"Discarded ({n_discarded:,})")
        )
    legend_elements.append(Patch(facecolor="#f8f9fa", edgecolor="#ddd", label="Background"))
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=6,
        framealpha=0.9,
        handlelength=1,
        handleheight=1,
    )

    fig.tight_layout()
    return fig


def render_summary_montage(
    stats: list[dict],
    grids: dict[str, np.ndarray],
    output_path: Path,
    n_worst: int = 20,
) -> None:
    """
    Render a montage of the N slides with lowest coverage.

    Parameters
    ----------
    stats : list[dict]
        Per-slide coverage statistics.
    grids : dict[str, np.ndarray]
        Pre-built coverage grids keyed by slide_id (only capped slides needed).
    output_path : Path
        Where to save the montage PNG.
    n_worst : int
        Number of worst-coverage slides to include.
    """
    import matplotlib.pyplot as plt

    capped = [s for s in stats if s["was_capped"]]
    capped.sort(key=lambda s: s["coverage_pct"])
    worst = capped[:n_worst]

    if not worst:
        logger.info("No capped slides — skipping summary montage.")
        return

    n_cols = min(5, len(worst))
    n_rows = (len(worst) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=100)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    cmap = _get_coverage_cmap()

    for plot_i, s in enumerate(worst):
        row, col = divmod(plot_i, n_cols)
        ax = axes[row, col]

        slide_id = s["slide_id"]
        if slide_id in grids:
            ax.imshow(
                grids[slide_id],
                cmap=cmap,
                vmin=0,
                vmax=2,
                interpolation="nearest",
                origin="upper",
            )
            ax.set_title(
                f"{Path(slide_id).name}\n{s['coverage_pct']:.1f}% kept "
                f"({s['n_kept']:,}/{s['n_total']:,})",
                fontsize=7,
                fontfamily="monospace",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Grid not\navailable",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_title(f"{Path(slide_id).name}", fontsize=7)

        ax.set_axis_off()

    # Hide unused axes
    for plot_i in range(len(worst), n_rows * n_cols):
        row, col = divmod(plot_i, n_cols)
        axes[row, col].set_axis_off()

    fig.suptitle(
        "Lowest-Coverage Slides (most patches discarded)\nGreen = kept  |  Red = discarded",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Summary montage saved: %s", output_path)


def load_mmap_index(mmap_dir: str | Path) -> dict:
    """Load mmap index arrays (for standalone QC and downstream stages)."""
    idx = np.load(str(Path(mmap_dir) / "index_arrays.npz"), allow_pickle=True)
    result = {
        "slide_ids": idx["slide_ids"].tolist(),
        "lengths": idx["lengths"].astype(int),
        "coord_chunk_ids": idx["coord_chunk_ids"].astype(int),
        "coord_offsets": idx["coord_offsets"].astype(int),
        "coord_dim": int(idx.get("coord_dim", 2)),
        "coord_dtype": str(idx["coord_dtype"][0]),
        "num_coord_chunks": int(idx["num_coord_chunks"]),
    }
    # raw_lengths added in later builds; fall back to lengths if absent
    if "raw_lengths" in idx:
        result["raw_lengths"] = idx["raw_lengths"].astype(int)
    else:
        result["raw_lengths"] = result["lengths"].copy()
    return result


def read_mmap_coords(
    mmap_dir: Path,
    idx: dict,
    slide_idx: int,
) -> np.ndarray:
    """Read stored (kept) coordinates from the mmap binary."""
    chunk_id = idx["coord_chunk_ids"][slide_idx]
    byte_offset = idx["coord_offsets"][slide_idx]
    n_patches = idx["lengths"][slide_idx]
    coord_dim = idx["coord_dim"]
    dtype = np.dtype(idx["coord_dtype"])

    mm_path = mmap_dir / f"coords_{chunk_id:03d}.bin"
    mm = np.memmap(str(mm_path), dtype=dtype, mode="r")

    start = byte_offset // dtype.itemsize
    n_elems = n_patches * coord_dim
    flat = mm[start : start + n_elems]
    return flat.reshape(n_patches, coord_dim).copy()


def read_h5_coords(h5_path: str | Path, coord_key: str = "coords") -> np.ndarray:
    """Read ALL coordinates from an H5 file (original, uncapped)."""
    with h5py.File(str(h5_path), "r") as f:
        coords = _read_h5_2d(f[coord_key])
    return coords.astype(np.int32)


# ═════════════════════════════════════════════════════════════════════════════
# Main build function
# ═════════════════════════════════════════════════════════════════════════════


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
      2. Write: stream data → binary + generate coverage QC inline

    Coverage thumbnails are generated DURING pass 2 while coords are
    still in memory, avoiding a second scan of the H5 directory.

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
    if cfg.cap_strategy not in _VALID_CAP_STRATEGIES:
        raise ValueError(
            f"Unknown cap_strategy '{cfg.cap_strategy}'. Valid: {sorted(_VALID_CAP_STRATEGIES)}"
        )

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

    # ── Check source hash ─────────────────────────────────────────────────
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

    for old_bin in output_dir.glob("features_*.bin"):
        old_bin.unlink()
    for old_bin in output_dir.glob("coords_*.bin"):
        old_bin.unlink()

    # Prepare QC directories
    qc_dir = output_dir / "coverage_qc"
    thumb_dir = qc_dir / "thumbnails"
    if cfg.save_coverage_thumbnails:
        thumb_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 2: Stream + Write + Inline QC ────────────────────────────────
    logger.info(f"Writing memmap to {output_dir} ...")

    feat_writer = ChunkedBinaryWriter(str(output_dir), "features", cfg.max_chunk_bytes)
    coord_writer = ChunkedBinaryWriter(str(output_dir), "coords", cfg.max_chunk_bytes)

    n_slides = len(slides)
    feat_chunk_ids = np.full(n_slides, -1, dtype=np.int16)
    feat_offsets = np.full(n_slides, -1, dtype=np.int64)
    coord_chunk_ids = np.full(n_slides, -1, dtype=np.int16)
    coord_offsets = np.full(n_slides, -1, dtype=np.int64)
    lengths = np.zeros(n_slides, dtype=np.int32)
    raw_lengths = np.zeros(n_slides, dtype=np.int32)  # original count before cap
    slide_ids_out = np.array([s.slide_id for s in slides], dtype=object)

    rng = np.random.RandomState(42)

    # QC accumulators (populated inline during write loop)
    qc_stats: list[dict] = []
    # Keep grids ONLY for worst-coverage slides (bounded memory).
    # We maintain a heap of the N worst so far and discard the rest.
    _qc_montage_n = cfg.coverage_montage_n if cfg.save_coverage_thumbnails else 0
    qc_grids: dict[str, np.ndarray] = {}
    # Track worst coverage to decide whether to keep a new grid
    _qc_worst_coverages: list[tuple[float, str]] = []  # (coverage_pct, slide_id)

    for i, slide in enumerate(tqdm(slides, desc="Writing chunks", unit="slide")):
        # Initialize temporaries for safe cleanup after try/except
        raw_coords = None
        subsample_idx = None
        kept_coords = None

        try:
            with h5py.File(slide.h5_path, "r") as f:
                feat_ds = f[cfg.h5_feat_key]
                coord_ds = f[cfg.h5_coord_key]

                raw_n = slide.raw_patches

                # ── Compute subsample indices ─────────────────────────
                if cfg.max_instances and raw_n > cfg.max_instances:
                    # Read coords once (needed for spatial_stratified + QC grid)
                    raw_coords = _read_h5_2d(coord_ds).astype(np.int32)

                    if cfg.cap_strategy == "spatial_stratified":
                        subsample_idx = _spatial_stratified_subsample(
                            raw_coords,
                            cfg.max_instances,
                            cfg.cap_grid_size,
                            rng,
                        )
                    else:
                        subsample_idx = rng.permutation(raw_n)[: cfg.max_instances]

                # ── Stream features to binary ─────────────────────────
                f_chunk, f_offset = feat_writer.write_streamed(
                    h5_dataset=feat_ds,
                    offset_start=0,
                    n_rows=raw_n,
                    dtype=cfg.feat_np_dtype,
                    chunk_size=cfg.stream_chunk_size,
                    subsample_indices=subsample_idx,
                )

                # ── Write coords to binary ────────────────────────────
                if raw_coords is not None and subsample_idx is not None:
                    # Coords already in memory — write directly, skip HDD re-read
                    kept_coords = raw_coords[subsample_idx]
                    kept_coords_out = np.ascontiguousarray(
                        kept_coords.astype(cfg.coord_np_dtype, copy=False)
                    )
                    c_chunk, c_offset = coord_writer.write(kept_coords_out)
                    del kept_coords_out
                else:
                    # Not capped — stream coords from H5 (sequential, fast)
                    c_chunk, c_offset = coord_writer.write_streamed(
                        h5_dataset=coord_ds,
                        offset_start=0,
                        n_rows=raw_n,
                        dtype=cfg.coord_np_dtype,
                        chunk_size=cfg.stream_chunk_size,
                        subsample_indices=None,
                    )
                    kept_coords = None

            # ── Index bookkeeping ─────────────────────────────────────
            feat_chunk_ids[i] = f_chunk
            feat_offsets[i] = f_offset
            coord_chunk_ids[i] = c_chunk
            coord_offsets[i] = c_offset
            lengths[i] = slide.n_patches
            raw_lengths[i] = raw_n

            # ── Inline QC: collect stats (NO matplotlib here) ─────────
            if cfg.save_coverage_thumbnails:
                was_capped = subsample_idx is not None
                n_total = raw_n
                n_kept = slide.n_patches
                coverage_pct = 100.0 * n_kept / n_total if n_total > 0 else 100.0

                qc_stats.append(
                    {
                        "slide_id": slide.slide_id,
                        "n_total": n_total,
                        "n_kept": n_kept,
                        "n_discarded": n_total - n_kept,
                        "coverage_pct": round(coverage_pct, 2),
                        "was_capped": was_capped,
                    }
                )

                # Build grid for capped slides; only keep worst N for montage
                if was_capped and raw_coords is not None:
                    if kept_coords is None:
                        kept_coords = raw_coords[subsample_idx]
                    grid, _ = build_coverage_grid(raw_coords, kept_coords)

                    # Bounded cache: keep at most 2x montage_n grids,
                    # then prune to montage_n when buffer fills
                    qc_grids[slide.slide_id] = grid
                    _qc_worst_coverages.append((coverage_pct, slide.slide_id))

                    if len(qc_grids) > _qc_montage_n * 2 and _qc_montage_n > 0:
                        # Prune: keep only the worst montage_n
                        _qc_worst_coverages.sort()
                        keep_ids = {sid for _, sid in _qc_worst_coverages[:_qc_montage_n]}
                        drop_ids = [sid for sid in qc_grids if sid not in keep_ids]
                        for sid in drop_ids:
                            del qc_grids[sid]
                        _qc_worst_coverages = _qc_worst_coverages[:_qc_montage_n]

                    del grid

        except Exception as e:
            errors.append(ScanError(slide.slide_id, slide.h5_path, repr(e)))
            logger.warning(f"Failed to write slide {slide.slide_id}: {e}")

        # Explicit cleanup of large temporaries
        del raw_coords, subsample_idx, kept_coords

        # Periodic GC
        if i % 500 == 0 and i > 0:
            gc.collect()

    feat_writer.close()
    coord_writer.close()

    # ── Write index ───────────────────────────────────────────────────────
    index_path = output_dir / "index_arrays.npz"
    np.savez(
        str(index_path),
        schema_version=np.int32(MMAP_SCHEMA_VERSION),
        slide_ids=slide_ids_out,
        lengths=lengths,
        raw_lengths=raw_lengths,
        feat_chunk_ids=feat_chunk_ids,
        feat_offsets=feat_offsets,
        coord_chunk_ids=coord_chunk_ids,
        coord_offsets=coord_offsets,
        feat_dim=np.int32(feat_dim),
        feat_dtype=np.array([cfg.feat_dtype_str], dtype=object),
        coord_dtype=np.array([cfg.coord_dtype], dtype=object),
        coord_dim=np.int32(slides[0].coord_dim),
        bytes_per_feat=np.int32(cfg.bytes_per_feat_element),
        num_feat_chunks=np.int32(feat_writer.num_chunks),
        num_coord_chunks=np.int32(coord_writer.num_chunks),
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
                f.write(f"{e.slide_id}\t{e.reason}\n")
        logger.warning(f"{len(errors)} errors logged to {err_path}")

    # ── Finalize QC: write CSV + batch-render thumbnails ─────────────────
    if cfg.save_coverage_thumbnails and qc_stats:
        # Write stats CSV
        csv_path = qc_dir / "coverage_stats.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=qc_stats[0].keys())
            writer.writeheader()
            writer.writerows(qc_stats)

        capped = [s for s in qc_stats if s["was_capped"]]
        if capped:
            coverages = [s["coverage_pct"] for s in capped]
            logger.info(
                "Coverage QC: %d/%d slides capped — mean=%.1f%%, min=%.1f%%, median=%.1f%%",
                len(capped),
                len(qc_stats),
                float(np.mean(coverages)),
                float(np.min(coverages)),
                float(np.median(coverages)),
            )

            # ── Batch-render thumbnails for capped slides ─────────────
            # Matplotlib leaks ~200-500 KB per figure even after close().
            # Rendering inside the 17k write loop caused unbounded growth.
            # Here we render in batches with explicit cleanup.
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                logger.warning(
                    "matplotlib not installed — skipping coverage thumbnails. "
                    "Install with: pip install matplotlib"
                )
                plt = None

            if plt is not None:
                RENDER_BATCH = 50
                cmap = _get_coverage_cmap()
                n_rendered = 0

                # Index stats for O(1) lookup
                stats_by_id = {s["slide_id"]: s for s in capped}

                logger.info(
                    "Rendering %d coverage thumbnails (batch size %d) ...",
                    len(qc_grids),
                    RENDER_BATCH,
                )

                for slide_id, grid in tqdm(
                    qc_grids.items(),
                    desc="Rendering thumbnails",
                    unit="slide",
                    total=len(qc_grids),
                ):
                    stat = stats_by_id[slide_id]

                    fig = render_coverage_thumbnail(
                        grid=grid,
                        slide_id=slide_id,
                        n_total=stat["n_total"],
                        n_kept=stat["n_kept"],
                        cmap=cmap,
                    )
                    thumb_path = thumb_dir / f"{slide_id.replace('/', '_')}.png"
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(str(thumb_path), bbox_inches="tight", dpi=100)

                    # Aggressive cleanup: clf + close releases Agg renderer memory
                    fig.clf()
                    plt.close(fig)
                    del fig

                    n_rendered += 1
                    if n_rendered % RENDER_BATCH == 0:
                        plt.close("all")  # safety net for any leaked figures
                        gc.collect()

                # Final cleanup after all thumbnails
                plt.close("all")
                gc.collect()
                logger.info("Rendered %d thumbnails.", n_rendered)

                # ── Render montage from cached grids ──────────────────────
                render_summary_montage(
                    stats=qc_stats,
                    grids=qc_grids,
                    output_path=qc_dir / "coverage_summary.png",
                    n_worst=cfg.coverage_montage_n,
                )
                plt.close("all")
                gc.collect()
        else:
            logger.info("Coverage QC: no slides were capped.")

        logger.info("Coverage QC saved to %s", qc_dir)

    # Free montage grids
    del qc_grids

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

    for path in feat_writer.chunk_files:
        size_gb = os.path.getsize(path) / (1024**3)
        logger.info(f"  {os.path.basename(path)}: {size_gb:.2f} GB")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Validation (for downstream stages)
# ═════════════════════════════════════════════════════════════════════════════


def validate_mmap_dir(mmap_dir: str | Path) -> dict:
    """
    Validate that a memmap directory contains a valid, complete store.

    Returns metadata dict if valid, raises on failure.
    """
    mmap_dir = Path(mmap_dir)

    if not mmap_dir.is_dir():
        raise FileNotFoundError(f"Memmap directory not found: {mmap_dir}")

    schema_path = mmap_dir / ".schema_version"
    if schema_path.is_file():
        version = int(schema_path.read_text().strip())
        if version != MMAP_SCHEMA_VERSION:
            raise ValueError(
                f"Memmap schema version mismatch: expected {MMAP_SCHEMA_VERSION}, "
                f"got {version}. Rebuild with: python scripts/build_mmap.py force=true"
            )

    index_path = mmap_dir / "index_arrays.npz"
    if not index_path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")

    idx = np.load(str(index_path), allow_pickle=True)

    slide_ids = idx["slide_ids"]
    if not all(isinstance(s, str) for s in slide_ids):
        raise TypeError(
            "slide_ids in index_arrays.npz are not all strings. "
            "Index may be corrupted — rebuild the mmap."
        )

    n_feat_chunks = int(idx["num_feat_chunks"])
    n_coord_chunks = int(idx["num_coord_chunks"])

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
