#!/usr/bin/env python
"""
QC Visualization: Spatial coverage of mmap patch selection.

Generates per-slide thumbnail images showing which tissue patches were
KEPT (stored in the mmap binary) vs DISCARDED (dropped by max_instances cap)
during build_mmap.

Uses only the (x, y) coordinates from the original H5 files and the mmap
index to reconstruct spatial coverage maps.  No pixel data is needed —
each patch is rendered as a colored cell on a grid.

Outputs:
  {output_dir}/
  ├── thumbnails/
  │   ├── {slide_id}.png          # per-slide spatial maps
  │   └── ...
  ├── coverage_summary.png        # montage of worst-coverage slides
  └── coverage_stats.csv          # per-slide coverage statistics

Usage:
    # After building mmap with max_instances cap:
    python scripts/visualize_mmap_coverage.py \
        --h5_dir /path/to/h5 \
        --mmap_dir /path/to/mmap \
        --output_dir outputs/coverage_qc

    # Only visualize slides that were capped (lost patches):
    python scripts/visualize_mmap_coverage.py \
        --h5_dir /path/to/h5 \
        --mmap_dir /path/to/mmap \
        --output_dir outputs/coverage_qc \
        --only_capped

    # Limit to N slides (for quick inspection):
    python scripts/visualize_mmap_coverage.py \
        --h5_dir /path/to/h5 \
        --mmap_dir /path/to/mmap \
        --output_dir outputs/coverage_qc \
        --max_slides 50
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Core: reconstruct kept vs discarded patches
# ═════════════════════════════════════════════════════════════════════════════


def load_mmap_index(mmap_dir: str | Path) -> dict:
    """Load mmap index arrays."""
    idx = np.load(str(Path(mmap_dir) / "index_arrays.npz"), allow_pickle=True)
    return {
        "slide_ids": idx["slide_ids"].tolist(),
        "lengths": idx["lengths"].astype(int),
        "coord_chunk_ids": idx["coord_chunk_ids"].astype(int),
        "coord_offsets": idx["coord_offsets"].astype(int),
        "coord_dim": int(idx.get("coord_dim", 2)),
        "coord_dtype": str(idx["coord_dtype"][0]),
        "num_coord_chunks": int(idx["num_coord_chunks"]),
    }


def read_h5_coords(h5_path: str | Path, coord_key: str = "coords") -> np.ndarray:
    """Read ALL coordinates from an H5 file (original, uncapped)."""
    with h5py.File(str(h5_path), "r") as f:
        coords = f[coord_key][:]
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords[0]
    return coords.astype(np.int32)


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

    elem_size = dtype.itemsize
    start = byte_offset // elem_size
    n_elems = n_patches * coord_dim
    flat = mm[start : start + n_elems]
    return flat.reshape(n_patches, coord_dim).copy()


# ═════════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════════


def infer_patch_size(coords: np.ndarray) -> int:
    """
    Infer the patch stride from coordinate differences.

    Looks at the most common nonzero difference between adjacent x-coords
    when sorted. Falls back to 256 if detection fails.
    """
    if len(coords) < 2:
        return 256

    # Sort by x then y
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    sorted_coords = coords[order]

    # Look at x-differences between consecutive patches
    dx = np.diff(sorted_coords[:, 0])
    dx_nonzero = dx[dx > 0]

    if len(dx_nonzero) == 0:
        # All same x — try y
        dy = np.diff(sorted_coords[:, 1])
        dy_nonzero = dy[dy > 0]
        if len(dy_nonzero) == 0:
            return 256
        vals, counts = np.unique(dy_nonzero, return_counts=True)
        return int(vals[counts.argmax()])

    vals, counts = np.unique(dx_nonzero, return_counts=True)
    return int(vals[counts.argmax()])


def render_coverage_thumbnail(
    all_coords: np.ndarray,
    kept_coords: np.ndarray,
    slide_id: str,
    patch_size: int | None = None,
    max_thumb_size: int = 800,
    dpi: int = 100,
) -> plt.Figure:
    """
    Render a spatial coverage map for one slide.

    Parameters
    ----------
    all_coords : (N_total, 2) — all patch positions from original H5
    kept_coords : (N_kept, 2) — patch positions stored in mmap
    slide_id : str
    patch_size : int or None — inferred if None
    max_thumb_size : int — max pixel dimension of the thumbnail
    dpi : int

    Returns
    -------
    matplotlib Figure
    """
    if patch_size is None:
        patch_size = infer_patch_size(all_coords)

    # Build coordinate sets for fast lookup
    # Normalize to grid indices
    x_min, y_min = all_coords.min(axis=0)
    all_grid = ((all_coords - [x_min, y_min]) // patch_size).astype(int)
    kept_grid = ((kept_coords - [x_min, y_min]) // patch_size).astype(int)

    grid_w = all_grid[:, 0].max() + 1
    grid_h = all_grid[:, 1].max() + 1

    # 0 = background, 1 = discarded, 2 = kept
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    # Mark all patches as discarded first
    for x, y in all_grid:
        grid[y, x] = 1

    # Overwrite kept patches
    for x, y in kept_grid:
        grid[y, x] = 2

    # Stats
    n_total = len(all_coords)
    n_kept = len(kept_coords)
    n_discarded = n_total - n_kept
    coverage_pct = 100 * n_kept / n_total if n_total > 0 else 0

    # Custom colormap: white=background, red=discarded, green=kept
    cmap = ListedColormap(["#f8f9fa", "#ef4444", "#22c55e"])

    # Figure sizing
    aspect = grid_h / grid_w if grid_w > 0 else 1
    fig_w = min(max_thumb_size / dpi, 8)
    fig_h = fig_w * aspect

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", origin="upper")
    ax.set_axis_off()

    # Title with stats
    ax.set_title(
        f"{slide_id}\n"
        f"Total: {n_total:,}  Kept: {n_kept:,}  Discarded: {n_discarded:,}  "
        f"Coverage: {coverage_pct:.1f}%",
        fontsize=8,
        fontfamily="monospace",
        pad=6,
    )

    # Legend
    legend_elements = [
        Patch(facecolor="#22c55e", edgecolor="none", label=f"Kept ({n_kept:,})"),
        Patch(facecolor="#ef4444", edgecolor="none", label=f"Discarded ({n_discarded:,})"),
        Patch(facecolor="#f8f9fa", edgecolor="#ddd", label="Background"),
    ]
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
    mmap_dir: Path,
    h5_dir: Path,
    idx: dict,
    output_path: Path,
    n_worst: int = 20,
    coord_key: str = "coords",
) -> None:
    """Render a montage of the N slides with lowest coverage (most discarded)."""
    # Sort by coverage ascending (worst coverage first)
    capped = [s for s in stats if s["coverage_pct"] < 100]
    capped.sort(key=lambda s: s["coverage_pct"])
    worst = capped[:n_worst]

    if not worst:
        logger.info("No capped slides found — skipping summary montage.")
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

    sid_to_mmap_idx = {sid: i for i, sid in enumerate(idx["slide_ids"])}

    for i, s in enumerate(worst):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        slide_id = s["slide_id"]
        h5_path = h5_dir / f"{slide_id}.h5"

        try:
            all_coords = read_h5_coords(h5_path, coord_key)
            midx = sid_to_mmap_idx[slide_id]
            kept_coords = read_mmap_coords(mmap_dir, idx, midx)
            patch_size = infer_patch_size(all_coords)

            x_min, y_min = all_coords.min(axis=0)
            all_grid = ((all_coords - [x_min, y_min]) // patch_size).astype(int)
            kept_grid = ((kept_coords - [x_min, y_min]) // patch_size).astype(int)

            grid_w = all_grid[:, 0].max() + 1
            grid_h = all_grid[:, 1].max() + 1
            grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
            for x, y in all_grid:
                grid[y, x] = 1
            for x, y in kept_grid:
                grid[y, x] = 2

            cmap = ListedColormap(["#f8f9fa", "#ef4444", "#22c55e"])
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", origin="upper")
            ax.set_title(
                f"{Path(slide_id).name}\n{s['coverage_pct']:.1f}% kept "
                f"({s['n_kept']:,}/{s['n_total']:,})",
                fontsize=7,
                fontfamily="monospace",
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=6)
            ax.set_title(f"{Path(slide_id).name}", fontsize=7)

        ax.set_axis_off()

    # Hide unused axes
    for i in range(len(worst), n_rows * n_cols):
        row, col = divmod(i, n_cols)
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
    logger.info(f"Summary montage saved: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize spatial coverage of mmap patch selection.")
    p.add_argument("--h5_dir", type=str, required=True, help="Original H5 feature directory.")
    p.add_argument("--mmap_dir", type=str, required=True, help="Built mmap directory.")
    p.add_argument(
        "--output_dir", type=str, default="outputs/coverage_qc", help="Output directory."
    )
    p.add_argument("--coord_key", type=str, default="coords", help="H5 key for coordinates.")
    p.add_argument("--max_slides", type=int, default=None, help="Limit slides to visualize.")
    p.add_argument("--only_capped", action="store_true", help="Only visualize capped slides.")
    p.add_argument("--no_thumbnails", action="store_true", help="Skip per-slide PNGs (stats only).")
    p.add_argument("--montage_n", type=int, default=20, help="Slides in summary montage.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        force=True,
    )
    args = parse_args()

    h5_dir = Path(args.h5_dir)
    mmap_dir = Path(args.mmap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = output_dir / "thumbnails"
    if not args.no_thumbnails:
        thumb_dir.mkdir(parents=True, exist_ok=True)

    # Load mmap index
    idx = load_mmap_index(mmap_dir)
    slide_ids = idx["slide_ids"]
    mmap_lengths = idx["lengths"]

    logger.info(f"Mmap: {len(slide_ids)} slides in {mmap_dir}")
    logger.info(f"H5 source: {h5_dir}")

    sid_to_mmap_idx = {sid: i for i, sid in enumerate(slide_ids)}

    # Collect stats
    stats: list[dict] = []
    slides_to_viz = slide_ids[: args.max_slides] if args.max_slides else slide_ids

    for slide_id in tqdm(slides_to_viz, desc="Processing slides", unit="slide"):
        h5_path = h5_dir / f"{slide_id}.h5"

        if not h5_path.exists():
            logger.warning(f"H5 not found for {slide_id}: {h5_path}")
            continue

        try:
            # Read ALL coords from original H5
            all_coords = read_h5_coords(h5_path, args.coord_key)
            n_total = len(all_coords)

            # Read KEPT coords from mmap
            midx = sid_to_mmap_idx[slide_id]
            n_kept = int(mmap_lengths[midx])
            kept_coords = read_mmap_coords(mmap_dir, idx, midx)

            was_capped = n_kept < n_total
            coverage_pct = 100 * n_kept / n_total if n_total > 0 else 100

            stat = {
                "slide_id": slide_id,
                "n_total": n_total,
                "n_kept": n_kept,
                "n_discarded": n_total - n_kept,
                "coverage_pct": round(coverage_pct, 2),
                "was_capped": was_capped,
            }
            stats.append(stat)

            if args.only_capped and not was_capped:
                continue

            # Render per-slide thumbnail
            if not args.no_thumbnails:
                fig = render_coverage_thumbnail(
                    all_coords=all_coords,
                    kept_coords=kept_coords,
                    slide_id=slide_id,
                )
                # Preserve subdirectory structure in output
                thumb_path = thumb_dir / f"{slide_id.replace('/', '_')}.png"
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(thumb_path), bbox_inches="tight", dpi=100)
                plt.close(fig)

        except Exception as e:
            logger.error(f"Failed on {slide_id}: {e}")
            continue

    # Write CSV stats
    csv_path = output_dir / "coverage_stats.csv"
    if stats:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
        logger.info(f"Stats saved: {csv_path} ({len(stats)} slides)")

        # Log summary
        capped = [s for s in stats if s["was_capped"]]
        if capped:
            coverages = [s["coverage_pct"] for s in capped]
            logger.info(
                f"\nCapped slides: {len(capped)}/{len(stats)} ({100 * len(capped) / len(stats):.1f}%)"
            )
            logger.info(
                f"Coverage of capped slides: "
                f"mean={np.mean(coverages):.1f}%, "
                f"min={np.min(coverages):.1f}%, "
                f"median={np.median(coverages):.1f}%, "
                f"max={np.max(coverages):.1f}%"
            )
        else:
            logger.info("No slides were capped (all within max_instances).")

    # Render summary montage
    if stats:
        render_summary_montage(
            stats=stats,
            mmap_dir=mmap_dir,
            h5_dir=h5_dir,
            idx=idx,
            output_path=output_dir / "coverage_summary.png",
            n_worst=args.montage_n,
            coord_key=args.coord_key,
        )

    logger.info(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
