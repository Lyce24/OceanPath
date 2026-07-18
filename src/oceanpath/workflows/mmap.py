"""Mmap-build workflow boundary."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from oceanpath.config import FoundationPaths, cfg_select
from oceanpath.contracts import PipelineStage, StageResult
from oceanpath.runtime import run_context, setup_logging
from oceanpath.storage.mmap import (
    MmapBuildConfig,
    build_mmap,
    compute_source_hash,
    scan_h5_dir,
)

logger = logging.getLogger(__name__)


def build_mmap_config(cfg: Any) -> MmapBuildConfig:
    """Translate a composed workflow config into a typed storage input."""
    filter_by_csv = bool(cfg_select(cfg, "storage.filter_by_csv", False))
    return MmapBuildConfig(
        h5_dir=str(cfg_select(cfg, "data.feature_h5_dir")),
        output_dir=str(cfg_select(cfg, "data.mmap_dir")),
        h5_feat_key=str(cfg_select(cfg, "storage.h5_feat_key", "features")),
        h5_coord_key=str(cfg_select(cfg, "storage.h5_coord_key", "coords")),
        feat_precision=int(cfg_select(cfg, "storage.feat_precision", 16)),
        coord_dtype=str(cfg_select(cfg, "storage.coord_dtype", "int32")),
        max_chunk_gb=float(cfg_select(cfg, "storage.max_chunk_gb", 9.5)),
        stream_chunk_size=int(cfg_select(cfg, "storage.stream_chunk_size", 4096)),
        max_instances=cfg_select(cfg, "storage.max_instances", None),
        csv_path=str(cfg_select(cfg, "data.csv_path")) if filter_by_csv else None,
        csv_id_col="slide_id",
        csv_filename_col=str(cfg_select(cfg, "data.filename_column", "filename")),
        require_all_csv_slides=bool(cfg_select(cfg, "storage.require_all_csv_slides", True)),
        cap_strategy=str(cfg_select(cfg, "storage.cap_strategy", "spatial_stratified")),
        cap_grid_size=int(cfg_select(cfg, "storage.cap_grid_size", 32)),
        save_coverage_thumbnails=bool(cfg_select(cfg, "storage.save_coverage_thumbnails", False)),
    )


def _dry_run_details(spec: MmapBuildConfig) -> dict[str, Any]:
    slides, errors = scan_h5_dir(spec)
    total_patches = sum(slide.n_patches for slide in slides)
    feature_dim = slides[0].feat_dim if slides else 0
    return {
        "valid_slides": len(slides),
        "errors": [{"slide_id": error.slide_id, "reason": error.reason} for error in errors],
        "total_patches": total_patches,
        "feature_dim": feature_dim,
        "estimated_feature_gb": (
            total_patches * feature_dim * spec.bytes_per_feat_element / 1024**3
        ),
        "source_hash": compute_source_hash(slides, spec) if slides else None,
    }


def run_mmap_build(cfg: Any) -> StageResult:
    """Scan H5 inputs or build the configured mmap store."""
    stage = PipelineStage.BUILD_MMAP
    setup_logging(cfg, stage=stage.value)
    paths = FoundationPaths.from_config(cfg)
    spec = build_mmap_config(cfg)
    dry_run = bool(cfg_select(cfg, "dry_run", False))
    started = time.monotonic()

    with run_context(cfg, stage=stage.value, output_dir=paths.mmap_dir, persist=not dry_run):
        if dry_run:
            details = _dry_run_details(spec)
            status = "dry_run"
        else:
            built = build_mmap(spec, force=bool(cfg_select(cfg, "force", False)))
            details = {
                "n_slides": built.n_slides,
                "total_patches": built.total_patches,
                "feature_dim": built.feat_dim,
                "feature_chunks": built.feat_chunks,
                "coord_chunks": built.coord_chunks,
                "n_errors": built.n_errors,
                "source_hash": built.source_hash,
            }
            status = "completed"

    result = StageResult(
        stage=stage,
        status=status,
        output_dir=Path(spec.output_dir),
        elapsed_seconds=time.monotonic() - started,
        details=details,
    )
    logger.info("Mmap workflow finished with status=%s", result.status)
    return result
