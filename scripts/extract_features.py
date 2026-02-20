"""
Thin entry point for TRIDENT feature extraction.

Usage:
    # Full pipeline (seg → coords → feat) on local machine
    python scripts/extract_features.py data=blca platform=local

    # Dry run — validate inputs, estimate work, exit before compute
    python scripts/extract_features.py data=blca platform=local dry_run=true

    # Diff mode — only extract new/changed slides (incremental update)
    python scripts/extract_features.py data=blca platform=local diff_mode=true

    # Resume from feature extraction only (seg + coords already done)
    python scripts/extract_features.py data=blca platform=local tasks=[feat]

    # Different encoder
    python scripts/extract_features.py data=blca platform=local encoder=conch_v15

    # Override GPU
    python scripts/extract_features.py data=blca platform=local extraction.gpu=1

    # Force re-extraction (ignore existing outputs)
    python scripts/extract_features.py data=blca platform=local force=true

    # Validate outputs after Slurm array completes
    python scripts/extract_features.py data=blca platform=hpc validate_only=true
"""

import logging
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _setup_logging(cfg: DictConfig) -> None:
    """Configure structured logging with experiment context."""
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO

    # Safely extract context for log format — don't crash if keys are missing
    data_name = OmegaConf.select(cfg, "data.name", default="unknown")
    encoder_name = OmegaConf.select(cfg, "encoder.name", default="unknown")

    fmt = (
        "%(asctime)s | %(levelname)-7s | "
        f"stage=extract | data={data_name} | encoder={encoder_name} | "
        "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


def _build_extraction_config(cfg: DictConfig):
    """Build the typed TridentExtractionConfig from Hydra's DictConfig."""
    from oceanpath.data.feature_extract import TridentExtractionConfig

    ext = cfg.extraction

    # Resolve encoder config into extraction fields
    encoder_name = cfg.encoder.name
    encoder_type = cfg.encoder.type
    ckpt_path = OmegaConf.select(cfg, "encoder.checkpoint_path", default=None)

    return TridentExtractionConfig(
        # Paths — resolved from platform + data configs
        wsi_dir=cfg.data.slide_dir,
        job_dir=cfg.data.feature_job_dir,
        custom_list_of_wsis=OmegaConf.select(cfg, "data.custom_list_of_wsis", default=None),
        wsi_ext=OmegaConf.to_container(ext.wsi_ext, resolve=True) if ext.wsi_ext else None,
        reader_type=OmegaConf.select(cfg, "extraction.reader_type", default=None),
        search_nested=ext.get("search_nested", False),
        # Segmentation
        segmenter=ext.segmenter,
        seg_conf_thresh=ext.seg_conf_thresh,
        remove_holes=ext.remove_holes,
        remove_artifacts=ext.remove_artifacts,
        remove_penmarks=ext.remove_penmarks,
        # Patching
        mag=ext.mag,
        patch_size=ext.patch_size,
        overlap=ext.overlap,
        min_tissue_proportion=ext.min_tissue_proportion,
        coords_dir=OmegaConf.select(cfg, "extraction.coords_dir", default=None),
        # Feature extraction — route based on encoder type
        patch_encoder=encoder_name if encoder_type == "patch" else "uni_v2",
        patch_encoder_ckpt_path=ckpt_path if encoder_type == "patch" else None,
        slide_encoder=encoder_name if encoder_type == "slide" else None,
        # Runtime
        gpu=ext.gpu,
        batch_size=ext.batch_size,
        seg_batch_size=OmegaConf.select(cfg, "extraction.seg_batch_size", default=None),
        feat_batch_size=OmegaConf.select(cfg, "extraction.feat_batch_size", default=None),
        max_workers=OmegaConf.select(cfg, "extraction.max_workers", default=None),
        skip_errors=ext.skip_errors,
        # Caching
        wsi_cache=OmegaConf.select(cfg, "extraction.wsi_cache", default=None),
        cache_batch_size=ext.cache_batch_size,
    )


@hydra.main(config_path="../configs", config_name="extract", version_base=None)
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)

    # ── Print resolved config ─────────────────────────────────────────────
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # ── Build typed config ────────────────────────────────────────────────
    extraction_cfg = _build_extraction_config(cfg)

    # ── Determine tasks ───────────────────────────────────────────────────
    tasks = list(OmegaConf.to_container(cfg.tasks, resolve=True))

    # ── Handle validate-only mode (post-array-job check) ──────────────────
    if cfg.get("validate_only", False):
        from oceanpath.data.feature_extract import validate_outputs

        result = validate_outputs(extraction_cfg, tasks)
        logger.info(f"Validation result: {result['found']}/{result['total']} H5 files present")
        if result["missing"]:
            logger.warning(f"Missing slides ({len(result['missing'])}): {result['missing'][:20]}")
            sys.exit(1)
        sys.exit(0)

    # ── Resolve Slurm array sharding ──────────────────────────────────────
    shard_id = None
    total_shards = None
    slurm_task_id = OmegaConf.select(cfg, "platform.slurm_array_task_id", default=None)
    if slurm_task_id is not None:
        shard_id = int(slurm_task_id)
        total_shards = int(cfg.platform.slurm_array_task_count)
        logger.info(f"Slurm array mode: shard {shard_id}/{total_shards}")

    # ── Run pipeline ──────────────────────────────────────────────────────
    from oceanpath.data.feature_extract import run_pipeline

    start = time.monotonic()

    result_path = run_pipeline(
        cfg=extraction_cfg,
        tasks=tasks,
        diff_mode=cfg.get("diff_mode", False),
        shard_id=shard_id,
        total_shards=total_shards,
        dry_run=cfg.get("dry_run", False),
        force=cfg.get("force", False),
    )

    elapsed = time.monotonic() - start
    if result_path:
        logger.info(f"Done in {elapsed:.1f}s → {result_path}")
    else:
        logger.info(f"Dry run completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
