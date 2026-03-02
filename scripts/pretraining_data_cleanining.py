#!/usr/bin/env python
"""
Pretraining data cleaning: scan H5 feature files, filter to TCGA + CPTAC,
discard small slides (< MIN_PATCHES), and produce a curated manifest CSV.

Output CSV columns:
  slide_id      — filename without .h5
  slide_path    — relative POSIX path from ROOT_DIR without .h5
  cancer_type   — e.g. BRCA, LUAD, GBM, ...
  num_patches   — number of patches in the H5 file

Usage:
    python scripts/pretraining_data_cleanining.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR = "/mnt/d/YC.Liu/UNI2_h_features"
OUTPUT_DIR = "/mnt/d/YC.Liu/manifests"
OUTPUT_CSV_NAME = "pretrain_manifest.csv"
OUTPUT_STATS_NAME = "pretrain_manifest_stats.txt"

MIN_PATCHES = 256
FEATURE_KEY = "features"
ALLOWED_SOURCES = {"TCGA", "CPTAC"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_cancer_type(slide_path: str) -> str:
    """
    Extract cancer type from the relative slide_path.

    Examples
    --------
    >>> _extract_cancer_type("TCGA/TCGA-CESC/TCGA-FU-A40J-01Z-00-DX1...")
    'CESC'
    >>> _extract_cancer_type("TCGA/TCGA-BRCA_IDC/slide1")
    'BRCA'
    >>> _extract_cancer_type("CPTAC/cptac_brca/01BR001-...")
    'BRCA'
    """
    parts = slide_path.split("/")
    source = parts[0]

    if source == "TCGA":
        # TCGA/TCGA-CESC/... -> CESC
        subdir = parts[1]  # e.g. "TCGA-CESC" or "TCGA-BRCA_IDC"
        cancer = subdir.split("-", 1)[1].upper()
        if cancer in ("BRCA_IDC", "BRCA_OTHERS"):
            cancer = "BRCA"
    elif source == "CPTAC":
        # CPTAC/cptac_brca/... -> BRCA
        subdir = parts[1]  # e.g. "cptac_brca"
        cancer = subdir.split("_", 1)[1].upper()
    else:
        cancer = "UNKNOWN"

    return cancer


def _read_patch_count(h5_path: str | Path) -> int | None:
    """Read number of patches from an H5 file (shape only, no data loaded)."""
    try:
        with h5py.File(str(h5_path), "r") as f:
            if FEATURE_KEY not in f:
                return None
            shape = f[FEATURE_KEY].shape
            # Handle (1, N, D) -> N or (N, D) -> N
            return shape[1] if len(shape) == 3 else shape[0]
    except Exception as e:
        logger.warning("Failed to read %s: %s", h5_path, e)
        return None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        force=True,
    )

    root = Path(ROOT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_csv = out_dir / OUTPUT_CSV_NAME
    output_stats = out_dir / OUTPUT_STATS_NAME

    if not root.is_dir():
        logger.error("ROOT_DIR does not exist: %s", root)
        return

    # ── Scan all H5 files ─────────────────────────────────────────────────
    h5_files = sorted(root.rglob("*.h5"))
    logger.info("Found %d H5 files in %s", len(h5_files), root)

    rows: list[dict] = []
    skipped_source = 0
    skipped_small = 0
    skipped_error = 0

    for h5_path in tqdm(h5_files, desc="Scanning H5 files", unit="file"):
        rel = h5_path.relative_to(root)
        source = rel.parts[0]

        # Filter: TCGA + CPTAC only
        if source not in ALLOWED_SOURCES:
            skipped_source += 1
            continue

        num_patches = _read_patch_count(h5_path)
        if num_patches is None:
            skipped_error += 1
            continue

        # Filter: discard slides with < MIN_PATCHES
        if num_patches < MIN_PATCHES:
            skipped_small += 1
            continue

        slide_id = h5_path.stem
        slide_path = rel.with_suffix("").as_posix()
        cancer_type = _extract_cancer_type(slide_path)

        rows.append(
            {
                "slide_id": slide_id,
                "slide_path": slide_path,
                "cancer_type": cancer_type,
                "num_patches": num_patches,
            }
        )

    # ── Build DataFrame and save CSV ──────────────────────────────────────
    df = pd.DataFrame(rows)
    df.sort_values("slide_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved %d slides to %s", len(df), output_csv)

    # ── Compute and save stats ────────────────────────────────────────────
    patch_counts = df["num_patches"].values

    lines = [
        "=" * 60,
        "  PRETRAIN MANIFEST STATISTICS",
        "=" * 60,
        f"  Source:              {ROOT_DIR}",
        f"  Allowed sources:     {', '.join(sorted(ALLOWED_SOURCES))}",
        f"  Min patches filter:  {MIN_PATCHES}",
        "",
        f"  Total H5 scanned:    {len(h5_files)}",
        f"  Skipped (source):    {skipped_source}",
        f"  Skipped (< {MIN_PATCHES} patches): {skipped_small}",
        f"  Skipped (read error):{skipped_error}",
        f"  Kept slides:         {len(df)}",
        "",
        f"  Total patches:       {patch_counts.sum():,}",
        f"  Mean patches/slide:  {patch_counts.mean():.1f}",
        f"  Median patches:      {np.median(patch_counts):.0f}",
        f"  Min patches:         {patch_counts.min()}",
        f"  Max patches:         {patch_counts.max()}",
        f"  25th percentile:     {np.percentile(patch_counts, 25):.0f}",
        f"  50th percentile:     {np.percentile(patch_counts, 50):.0f}",
        f"  75th percentile:     {np.percentile(patch_counts, 75):.0f}",
        "",
        "  Cancer type breakdown:",
    ]

    for cancer_type, group in sorted(df.groupby("cancer_type"), key=lambda x: -len(x[1])):
        lines.append(
            f"    {cancer_type:12s}  {len(group):5d} slides  "
            f"{group['num_patches'].sum():>10,} patches  "
            f"mean={group['num_patches'].mean():.0f}  "
            f"median={int(group['num_patches'].median())}"
        )

    lines.append("")
    lines.append("=" * 60)

    stats_text = "\n".join(lines)
    output_stats.write_text(stats_text + "\n")

    print()
    print(stats_text)
    print()
    logger.info("CSV saved to:   %s", output_csv)
    logger.info("Stats saved to: %s", output_stats)


if __name__ == "__main__":
    main()
