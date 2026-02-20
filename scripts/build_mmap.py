"""
Thin entry point for building memmap stores from H5 feature files.

Usage:
    # Build memmap from BLCA UNI v2 features
    python scripts/build_mmap.py platform=local data=blca encoder=uni2

    # Dry run — scan H5s, report stats, don't write anything
    python scripts/build_mmap.py platform=local data=blca encoder=uni2 dry_run=true

    # Force rebuild (ignore source hash cache)
    python scripts/build_mmap.py platform=local data=blca encoder=uni2 force=true

    # Full precision
    python scripts/build_mmap.py platform=local data=blca encoder=uni2 storage.feat_precision=32

    # Cap patches per slide at build time
    python scripts/build_mmap.py platform=local data=blca encoder=uni2 storage.max_instances=8000

    # Filter to slides in CSV manifest only
    python scripts/build_mmap.py platform=local data=blca encoder=uni2 storage.filter_by_csv=true
"""

import logging
import time

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _setup_logging(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    data_name = OmegaConf.select(cfg, "data.name", default="unknown")
    encoder_name = OmegaConf.select(cfg, "encoder.name", default="unknown")
    fmt = (
        "%(asctime)s | %(levelname)-7s | "
        f"stage=build_mmap | data={data_name} | encoder={encoder_name} | "
        "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


def _build_mmap_config(cfg: DictConfig):
    """Build the typed MmapBuildConfig from Hydra's composed config."""
    from oceanpath.data.mmap_builder import MmapBuildConfig

    storage = cfg.storage

    # Resolve CSV path for filtering
    csv_path = None
    csv_id_col = "slide_id"
    csv_filename_col = "filename"
    if storage.get("filter_by_csv", False):
        csv_path = cfg.data.csv_path
        csv_id_col = OmegaConf.select(cfg, "data.filename_column", default="filename")

    return MmapBuildConfig(
        # Input: H5 directory (resolved from data + extraction + encoder configs)
        h5_dir=cfg.data.feature_h5_dir,
        # Output: memmap directory
        output_dir=cfg.data.mmap_dir,
        # H5 keys
        h5_feat_key=storage.h5_feat_key,
        h5_coord_key=storage.h5_coord_key,
        # Precision
        feat_precision=storage.feat_precision,
        coord_dtype=storage.coord_dtype,
        # Chunking
        max_chunk_gb=storage.max_chunk_gb,
        # Streaming
        stream_chunk_size=storage.stream_chunk_size,
        # Optional
        max_instances=OmegaConf.select(cfg, "storage.max_instances", default=None),
        csv_path=csv_path,
        csv_id_col=csv_id_col,
        csv_filename_col=csv_filename_col,
    )


@hydra.main(config_path="../configs", config_name="build_mmap", version_base=None)
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    mmap_cfg = _build_mmap_config(cfg)

    # ── Dry run: scan only ────────────────────────────────────────────────
    if cfg.get("dry_run", False):
        from oceanpath.data.mmap_builder import compute_source_hash, scan_h5_dir

        slides, errors = scan_h5_dir(mmap_cfg)
        source_hash = compute_source_hash(slides, mmap_cfg) if slides else "N/A"

        total_patches = sum(s.n_patches for s in slides)
        feat_dim = slides[0].feat_dim if slides else 0
        bytes_per = mmap_cfg.bytes_per_feat_element

        est_feat_gb = (total_patches * feat_dim * bytes_per) / (1024**3)
        est_coord_gb = (total_patches * 2 * 4) / (1024**3)  # int32, 2 cols

        print(f"\n{'=' * 60}")
        print("  DRY RUN — build_mmap")
        print(f"{'=' * 60}")
        print(f"  H5 directory:     {mmap_cfg.h5_dir}")
        print(f"  Output directory:  {mmap_cfg.output_dir}")
        print(f"  Valid slides:      {len(slides)}")
        print(f"  Errors:            {len(errors)}")
        print(f"  Total patches:     {total_patches:,}")
        print(f"  Feature dim:       {feat_dim}")
        print(f"  Precision:         {mmap_cfg.feat_dtype_str}")
        print(f"  Est. features:     {est_feat_gb:.2f} GB")
        print(f"  Est. coords:       {est_coord_gb:.2f} GB")
        print(f"  Max chunk size:    {mmap_cfg.max_chunk_gb} GB")
        print(f"  Est. feat chunks:  {max(1, int(est_feat_gb / mmap_cfg.max_chunk_gb) + 1)}")
        print(f"  Source hash:       {source_hash}")
        print(f"{'=' * 60}\n")

        if errors:
            print("  Errors (first 10):")
            for e in errors[:10]:
                print(f"    {e.slide_id}: {e.reason}")
            print()

        return

    # ── Full build ────────────────────────────────────────────────────────
    from oceanpath.data.mmap_builder import build_mmap

    start = time.monotonic()
    result = build_mmap(mmap_cfg, force=cfg.get("force", False))
    elapsed = time.monotonic() - start

    print(f"\n{'=' * 60}")
    print("  build_mmap complete")
    print(f"{'=' * 60}")
    print(f"  Output:        {result.output_dir}")
    print(f"  Slides:        {result.n_slides}")
    print(f"  Patches:       {result.total_patches:,}")
    print(f"  Feature dim:   {result.feat_dim}")
    print(f"  Feat chunks:   {result.feat_chunks}")
    print(f"  Coord chunks:  {result.coord_chunks}")
    print(f"  Errors:        {result.n_errors}")
    print(f"  Source hash:   {result.source_hash}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
