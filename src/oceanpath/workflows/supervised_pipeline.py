"""Build the config-aware six-stage supervised pipeline.

This module owns the boundary between the generic DAG engine and OceanPath's
domain workflows. It resolves foundation paths, attaches domain validators,
and declares which implementation/config inputs fingerprint each stage.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from oceanpath.config import FoundationPaths, cfg_select
from oceanpath.extraction import validate_outputs
from oceanpath.pipeline import PipelineRunner, Stage
from oceanpath.pipeline.transactions import (
    compose_validators,
    validate_files_exist,
    validate_parquet_not_empty,
)
from oceanpath.splitting import verify_split_integrity
from oceanpath.storage import validate_mmap_dir
from oceanpath.workflows.extraction import build_extraction_config
from oceanpath.workflows.training import training_run_fingerprint, validate_training_run_dir


def _stage_run(stage_name: str, stage_runs: dict[str, Callable] | None) -> Callable | None:
    if not stage_runs:
        return None
    return stage_runs.get(stage_name)


def _resolve_supervised_paths(cfg: Any) -> dict[str, Path]:
    paths = FoundationPaths.from_config(cfg)
    return {
        "slide_dir": paths.slide_dir,
        "feature_h5_dir": paths.feature_h5_dir,
        "mmap_dir": paths.mmap_dir,
        "splits_dir": paths.splits_dir,
        "train_dir": paths.train_dir,
        "eval_dir": paths.eval_dir,
        "artifact_dir": paths.artifact_dir,
        "csv_path": paths.manifest_path,
    }


def build_supervised_pipeline(
    cfg: Any,
    stage_runs: dict[str, Callable] | None = None,
) -> PipelineRunner:
    """Build the config-aware six-stage foundation pipeline DAG."""
    paths = _resolve_supervised_paths(cfg)
    runner = PipelineRunner()
    package_root = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[3]
    orchestration_code = [
        Path(__file__).resolve(),
        package_root / "workflows" / "pipeline.py",
    ]

    def _validate_extraction(path: Path) -> None:
        spec = build_extraction_config(cfg)
        if path.resolve() != paths["feature_h5_dir"].resolve():
            raise ValueError(f"Unexpected extraction output path: {path}")
        result = validate_outputs(spec, ["feat"])
        if result["missing"]:
            raise FileNotFoundError(
                f"Extraction is missing {len(result['missing'])}/{result['total']} "
                f"expected slide features; first 10: {result['missing'][:10]}"
            )

    def _validate_mmap(path: Path) -> None:
        validate_mmap_dir(path)

    split_structure_validator = compose_validators(
        validate_files_exist("splits.parquet", ".integrity_hash", "summary.json"),
        validate_parquet_not_empty("splits.parquet"),
    )

    def _validate_splits(path: Path) -> None:
        split_structure_validator(path)
        verify_split_integrity(path, paths["csv_path"])

    def _validate_training(path: Path) -> None:
        validate_training_run_dir(
            path,
            expected_fingerprint=training_run_fingerprint(cfg),
        )

    evaluation_validator = validate_files_exist(
        "oof_evaluation.json",
        "evaluation_scope.json",
    )
    export_required = [
        "export_report.json",
        "model_card.json",
        "model.ckpt",
        "export_config.yaml",
    ]
    export_formats = {str(value) for value in (cfg_select(cfg, "export.formats", []) or [])}
    if "onnx" in export_formats:
        export_required.append("model.onnx")
    if "torchscript" in export_formats:
        export_required.append("model.pt")
    export_validator = validate_files_exist(*export_required)

    runner.register(
        Stage(
            name="extract_features",
            description="Stage 1: feature extraction",
            inputs=[paths["slide_dir"]],
            outputs=[paths["feature_h5_dir"]],
            config_keys=[
                "platform",
                "data",
                "encoder",
                "extraction",
            ],
            code_paths=[
                package_root / "extraction",
                package_root / "workflows" / "extraction.py",
                *orchestration_code,
                repository_root / "scripts" / "extract_features.py",
            ],
            validator=_validate_extraction,
            run=_stage_run("extract_features", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="build_mmap",
            description="Stage 2: build mmap",
            inputs=[paths["feature_h5_dir"]],
            outputs=[paths["mmap_dir"]],
            config_keys=[
                "platform",
                "data",
                "encoder",
                "extraction",
                "storage",
            ],
            code_paths=[
                package_root / "storage",
                package_root / "workflows" / "mmap.py",
                *orchestration_code,
                repository_root / "scripts" / "build_mmap.py",
            ],
            depends_on=["extract_features"],
            validator=_validate_mmap,
            run=_stage_run("build_mmap", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="split_data",
            description="Stage 3: split data",
            inputs=[paths["csv_path"]],
            outputs=[paths["splits_dir"]],
            config_keys=["platform", "data", "splits"],
            code_paths=[
                package_root / "splitting",
                package_root / "workflows" / "splitting.py",
                *orchestration_code,
                repository_root / "scripts" / "make_splits.py",
            ],
            validator=_validate_splits,
            run=_stage_run("split_data", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="train_model",
            description="Stage 4: train model (model/datamodule/optional preload weights)",
            inputs=[paths["mmap_dir"], paths["splits_dir"]],
            outputs=[paths["train_dir"]],
            config_keys=[
                "platform",
                "data",
                "encoder",
                "extraction",
                "storage",
                "splits",
                "model",
                "training",
                "experiment",
            ],
            code_paths=[
                package_root / "datasets",
                package_root / "models",
                package_root / "training",
                package_root / "workflows" / "training.py",
                package_root / "workflows" / "finalize.py",
                *orchestration_code,
                repository_root / "scripts" / "train.py",
            ],
            depends_on=["build_mmap", "split_data"],
            validator=_validate_training,
            run=_stage_run("train_model", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="evaluate",
            description="Stage 5: evaluate",
            inputs=[paths["train_dir"]],
            outputs=[paths["eval_dir"]],
            config_keys=[
                "platform",
                "data",
                "encoder",
                "splits",
                "model",
                "training",
                "experiment",
                "eval",
            ],
            code_paths=[
                package_root / "eval",
                package_root / "workflows" / "evaluation.py",
                *orchestration_code,
                repository_root / "scripts" / "evaluate.py",
            ],
            depends_on=["train_model"],
            validator=evaluation_validator,
            run=_stage_run("evaluate", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="export_model",
            description="Stage 6: export portable model artifacts",
            inputs=[paths["train_dir"], paths["eval_dir"]],
            outputs=[paths["artifact_dir"]],
            config_keys=[
                "platform",
                "data",
                "encoder",
                "splits",
                "model",
                "training",
                "experiment",
                "eval",
                "export",
            ],
            code_paths=[
                package_root / "export",
                package_root / "workflows" / "export.py",
                *orchestration_code,
                repository_root / "scripts" / "export_model.py",
            ],
            depends_on=["evaluate"],
            validator=export_validator,
            run=_stage_run("export_model", stage_runs),
        )
    )

    return runner


__all__ = ["build_supervised_pipeline"]
