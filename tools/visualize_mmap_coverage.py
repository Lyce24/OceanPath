"""Render coverage QC from an existing H5 feature directory and mmap store."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from oceanpath.viz.coverage import build_coverage_grid, write_coverage_report

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize spatial coverage of mmap patch selection."
    )
    parser.add_argument("--h5-dir", "--h5_dir", required=True, type=Path)
    parser.add_argument("--mmap-dir", "--mmap_dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=Path("outputs/coverage_qc"),
        type=Path,
    )
    parser.add_argument("--coord-key", "--coord_key", default="coords")
    parser.add_argument("--max-slides", "--max_slides", type=int)
    parser.add_argument("--only-capped", "--only_capped", action="store_true")
    parser.add_argument("--no-thumbnails", "--no_thumbnails", action="store_true")
    parser.add_argument("--montage-n", "--montage_n", default=20, type=int)
    return parser.parse_args()


def _load_index(mmap_dir: Path) -> dict[str, Any]:
    with np.load(mmap_dir / "index_arrays.npz", allow_pickle=True) as index:
        return {
            "slide_ids": [str(value) for value in index["slide_ids"].tolist()],
            "lengths": index["lengths"].astype(int),
            "coord_chunk_ids": index["coord_chunk_ids"].astype(int),
            "coord_offsets": index["coord_offsets"].astype(int),
            "coord_dim": int(index["coord_dim"]),
            "coord_dtype": str(index["coord_dtype"][0]),
        }


def _read_h5_coords(path: Path, coord_key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        coords = np.asarray(handle[coord_key])
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if coords.ndim != 2:
        raise ValueError(f"Expected 2D coordinates in {path}, got {coords.shape}")
    return coords.astype(np.int32, copy=False)


def _read_mmap_coords(mmap_dir: Path, index: dict[str, Any], row: int) -> np.ndarray:
    chunk_id = int(index["coord_chunk_ids"][row])
    byte_offset = int(index["coord_offsets"][row])
    n_patches = int(index["lengths"][row])
    coord_dim = int(index["coord_dim"])
    dtype = np.dtype(index["coord_dtype"])
    start = byte_offset // dtype.itemsize
    mmap = np.memmap(mmap_dir / f"coords_{chunk_id:03d}.bin", dtype=dtype, mode="r")
    return np.asarray(mmap[start : start + n_patches * coord_dim]).reshape(n_patches, coord_dim)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    index = _load_index(args.mmap_dir)
    slide_ids = index["slide_ids"]
    if args.max_slides is not None:
        slide_ids = slide_ids[: args.max_slides]

    row_by_slide = {slide_id: row for row, slide_id in enumerate(index["slide_ids"])}
    stats: list[dict[str, Any]] = []
    grids: dict[str, np.ndarray] = {}

    for slide_id in tqdm(slide_ids, desc="Reading coverage", unit="slide"):
        h5_path = args.h5_dir / f"{slide_id}.h5"
        if not h5_path.is_file():
            logger.warning("Missing H5 for %s: %s", slide_id, h5_path)
            continue
        try:
            all_coords = _read_h5_coords(h5_path, args.coord_key)
            kept_coords = _read_mmap_coords(args.mmap_dir, index, row_by_slide[slide_id])
            n_total = len(all_coords)
            n_kept = len(kept_coords)
            was_capped = n_kept < n_total
            stats.append(
                {
                    "slide_id": slide_id,
                    "n_total": n_total,
                    "n_kept": n_kept,
                    "n_discarded": n_total - n_kept,
                    "coverage_pct": round(100 * n_kept / n_total, 2) if n_total else 100.0,
                    "was_capped": was_capped,
                }
            )
            if was_capped or not args.only_capped:
                grids[slide_id], _ = build_coverage_grid(all_coords, kept_coords)
        except (OSError, KeyError, ValueError) as error:
            logger.error("Failed to read %s: %s", slide_id, error)

    write_coverage_report(
        qc_dir=args.output_dir,
        thumb_dir=args.output_dir / "thumbnails",
        qc_stats=stats,
        qc_grids=grids,
        montage_n=args.montage_n,
        render_thumbnails=not args.no_thumbnails,
        only_capped=args.only_capped,
    )


if __name__ == "__main__":
    main()
