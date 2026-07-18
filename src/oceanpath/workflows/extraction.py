"""Feature-extraction workflow boundary."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from oceanpath.config import FoundationPaths, cfg_select
from oceanpath.config.access import cfg_list
from oceanpath.contracts import PipelineStage, StageResult
from oceanpath.extraction import TridentExtractionConfig, run_pipeline, validate_outputs
from oceanpath.runtime import run_context, setup_logging

logger = logging.getLogger(__name__)


def build_extraction_config(cfg: Any) -> TridentExtractionConfig:
    """Translate a composed workflow config into a typed extraction input."""
    encoder_name = str(cfg_select(cfg, "encoder.name"))
    encoder_type = str(cfg_select(cfg, "encoder.type", "patch"))
    checkpoint = cfg_select(cfg, "encoder.checkpoint_path", None)
    return TridentExtractionConfig(
        wsi_dir=str(cfg_select(cfg, "data.slide_dir")),
        job_dir=str(cfg_select(cfg, "data.feature_job_dir")),
        custom_list_of_wsis=cfg_select(cfg, "data.custom_list_of_wsis", None),
        wsi_ext=[str(value) for value in cfg_list(cfg, "extraction.wsi_ext")],
        reader_type=cfg_select(cfg, "extraction.reader_type", None),
        search_nested=bool(cfg_select(cfg, "extraction.search_nested", False)),
        segmenter=str(cfg_select(cfg, "extraction.segmenter", "hest")),
        seg_conf_thresh=float(cfg_select(cfg, "extraction.seg_conf_thresh", 0.5)),
        remove_holes=bool(cfg_select(cfg, "extraction.remove_holes", False)),
        remove_artifacts=bool(cfg_select(cfg, "extraction.remove_artifacts", False)),
        remove_penmarks=bool(cfg_select(cfg, "extraction.remove_penmarks", False)),
        mag=int(cfg_select(cfg, "extraction.mag", 20)),
        patch_size=int(cfg_select(cfg, "extraction.patch_size", 256)),
        overlap=int(cfg_select(cfg, "extraction.overlap", 0)),
        min_tissue_proportion=float(cfg_select(cfg, "extraction.min_tissue_proportion", 0.0)),
        coords_dir=cfg_select(cfg, "extraction.coords_dir", None),
        patch_encoder=encoder_name if encoder_type == "patch" else "uni_v1",
        patch_encoder_ckpt_path=checkpoint if encoder_type == "patch" else None,
        slide_encoder=encoder_name if encoder_type == "slide" else None,
        encoder_source=cfg_select(cfg, "encoder.source", None),
        encoder_adapter=str(cfg_select(cfg, "encoder.adapter", "trident")),
        gpu=int(cfg_select(cfg, "extraction.gpu", 0)),
        batch_size=int(cfg_select(cfg, "extraction.batch_size", 64)),
        seg_batch_size=cfg_select(cfg, "extraction.seg_batch_size", None),
        feat_batch_size=cfg_select(cfg, "extraction.feat_batch_size", None),
        max_workers=cfg_select(cfg, "extraction.max_workers", None),
        skip_errors=bool(cfg_select(cfg, "extraction.skip_errors", True)),
        require_complete=bool(cfg_select(cfg, "extraction.require_complete", True)),
        wsi_cache=cfg_select(cfg, "extraction.wsi_cache", None),
        cache_batch_size=int(cfg_select(cfg, "extraction.cache_batch_size", 32)),
    )


def run_extraction(cfg: Any) -> StageResult:
    """Validate and run the configured extraction tasks."""
    stage = PipelineStage.EXTRACT_FEATURES
    setup_logging(cfg, stage=stage.value)
    paths = FoundationPaths.from_config(cfg)
    extraction_cfg = build_extraction_config(cfg)
    tasks = [str(task) for task in cfg_list(cfg, "tasks", ["seg", "coords", "feat"])]
    dry_run = bool(cfg_select(cfg, "dry_run", False))
    started = time.monotonic()

    with run_context(
        cfg,
        stage=stage.value,
        output_dir=paths.extraction_dir,
        persist=not dry_run,
    ):
        if bool(cfg_select(cfg, "validate_only", False)):
            validation = validate_outputs(extraction_cfg, tasks)
            if validation["missing"]:
                raise RuntimeError(
                    f"Extraction validation failed: {len(validation['missing'])} slides missing"
                )
            details = validation
            status = "validated"
        else:
            shard_id = cfg_select(cfg, "platform.slurm_array_task_id", None)
            total_shards = cfg_select(cfg, "platform.slurm_array_task_count", None)
            result_path = run_pipeline(
                cfg=extraction_cfg,
                tasks=tasks,
                diff_mode=bool(cfg_select(cfg, "diff_mode", False)),
                shard_id=int(shard_id) if shard_id is not None else None,
                total_shards=int(total_shards) if total_shards is not None else None,
                dry_run=dry_run,
                force=bool(cfg_select(cfg, "force", False)),
            )
            details = {"tasks": tasks, "manifest_path": str(result_path) if result_path else None}
            status = "dry_run" if dry_run else "completed"

    result = StageResult(
        stage=stage,
        status=status,
        output_dir=Path(extraction_cfg.job_dir),
        elapsed_seconds=time.monotonic() - started,
        details=details,
    )
    logger.info("Extraction workflow finished with status=%s", result.status)
    return result
