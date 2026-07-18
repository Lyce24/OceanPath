"""Split-generation workflow boundary."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from oceanpath.config import FoundationPaths, cfg_select
from oceanpath.contracts import PipelineStage, StageResult
from oceanpath.runtime import run_context, setup_logging
from oceanpath.splitting import SplitConfig, generate_splits, preview_splits

logger = logging.getLogger(__name__)


def _value_or(value: Any, default: Any) -> Any:
    return default if value is None else value


def _optional_float(value: Any) -> float | None:
    """Preserve profile ``null`` values for scheme-specific ratio handling."""
    return None if value is None else float(value)


def build_split_config(cfg: Any) -> SplitConfig:
    """Translate a composed workflow config into a typed split input."""
    paths = FoundationPaths.from_config(cfg)
    verify_features = bool(cfg_select(cfg, "verify_features", False))
    return SplitConfig(
        scheme=str(cfg_select(cfg, "splits.scheme")),
        name=str(cfg_select(cfg, "splits.name")),
        csv_path=str(paths.manifest_path),
        output_dir=str(paths.splits_dir),
        filename_column=str(cfg_select(cfg, "data.filename_column", "filename")),
        label_column=str(cfg_select(cfg, "splits.label_column")),
        group_column=cfg_select(cfg, "splits.group_column", None),
        site_column=cfg_select(cfg, "splits.site_column", None),
        seed=int(cfg_select(cfg, "splits.seed", 42)),
        n_folds=int(_value_or(cfg_select(cfg, "splits.n_folds", 5), 5)),
        n_repeats=int(_value_or(cfg_select(cfg, "splits.n_repeats", 10), 10)),
        train_ratio=_optional_float(cfg_select(cfg, "splits.train_ratio", None)),
        val_ratio=_optional_float(cfg_select(cfg, "splits.val_ratio", None)),
        test_ratio=_optional_float(cfg_select(cfg, "splits.test_ratio", None)),
        test_filter=cfg_select(cfg, "splits.test_filter", None),
        val_filter=cfg_select(cfg, "splits.val_filter", None),
        feature_h5_dir=str(paths.feature_h5_dir) if verify_features else None,
    )


def run_split_generation(cfg: Any) -> StageResult:
    """Preview or generate leakage-safe split assignments."""
    stage = PipelineStage.SPLIT_DATA
    setup_logging(cfg, stage=stage.value)
    paths = FoundationPaths.from_config(cfg)
    spec = build_split_config(cfg)
    dry_run = bool(cfg_select(cfg, "dry_run", False))
    started = time.monotonic()

    with run_context(cfg, stage=stage.value, output_dir=paths.splits_dir, persist=not dry_run):
        if dry_run:
            preview = preview_splits(spec)
            details = {
                "n_slides": preview.n_slides,
                "n_groups": preview.n_groups,
                "label_distribution": preview.label_distribution,
                "test_filter_matches": preview.test_filter_matches,
            }
            status = "dry_run"
        else:
            generated = generate_splits(spec, force=bool(cfg_select(cfg, "force", False)))
            details = {
                "parquet_path": generated.parquet_path,
                "scheme": generated.scheme,
                "n_slides": generated.n_slides,
                "n_groups": generated.n_groups,
                "label_distribution": generated.label_distribution,
                "fold_distribution": generated.fold_distribution,
                "integrity_hash": generated.integrity_hash,
            }
            status = "completed"

    result = StageResult(
        stage=stage,
        status=status,
        output_dir=Path(spec.output_dir),
        elapsed_seconds=time.monotonic() - started,
        details=details,
    )
    logger.info("Split workflow finished with status=%s", result.status)
    return result
