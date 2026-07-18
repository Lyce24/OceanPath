"""
Stage 6: Export trained MIL model to deployment-ready artifacts.

Pipeline:
  ┌────────────────────┐
  │  Stage 4 (Train)   │  model.ckpt (Lightning checkpoint)
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Portability       │  CPU ↔ GPU output match? (catches device-specific bugs)
  │  Validation        │
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  ONNX Export       │  model.onnx (cross-platform, quantization-ready)
  │  + TorchScript     │  model.pt   (pure-PyTorch fallback)
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Numerical         │  Exported outputs == PyTorch outputs?
  │  Validation        │  (catches export-time op translation bugs)
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Model Card        │  model_card.json (provenance, metrics, schema)
  └────────┬───────────┘
           ▼
  artifacts/{exp_name}_{fingerprint}/
    ├── model.onnx
    ├── model.pt
    ├── model.ckpt          ← fallback copy
    ├── model_card.json
    └── export_report.json

Usage:
    # Export best_fold model (default)
    python scripts/export_model.py platform=local data=blca encoder=univ1 \\
           splits=kfold5 model=abmil training=default

    # ONNX only
    python scripts/export_model.py ... export.formats=[onnx]

    # Skip validation (faster, not recommended)
    python scripts/export_model.py ... export.skip_validation=true

    # Dry run
    python scripts/export_model.py ... dry_run=true
"""

import logging
import time
from dataclasses import replace
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from oceanpath.config import FoundationPaths, artifact_config_fingerprint
from oceanpath.contracts import PipelineStage, StageResult
from oceanpath.pipeline.transactions import atomic_output, validate_files_exist

logger = logging.getLogger(__name__)

_PORTABLE_MODEL_STRATEGIES = frozenset({"best_fold", "refit"})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _setup_logging(cfg: DictConfig) -> None:
    """Configure logging to match train.py / evaluate.py style."""
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="export")
    fmt = f"%(asctime)s | %(levelname)-7s | exp={exp_name} | %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


def _resolve_model_strategy(requested: object) -> str:
    """Resolve one checkpoint-backed strategy that the portable artifact can represent."""
    strategy = str(requested or "best_fold")
    if strategy == "ensemble":
        raise ValueError(
            "export.model_strategy=ensemble is not supported: an ensemble contains multiple "
            "checkpoints, while the exported ONNX/TorchScript serving contract represents one "
            "model. Use best_fold/refit, or add a dedicated ensemble artifact and serving contract."
        )
    if strategy not in _PORTABLE_MODEL_STRATEGIES:
        supported = ", ".join(sorted(_PORTABLE_MODEL_STRATEGIES))
        raise ValueError(f"Unsupported export.model_strategy={strategy!r}; expected {supported}")
    return strategy


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def _run_export(
    cfg: DictConfig,
    paths: FoundationPaths,
    *,
    public_output_dir: Path | None = None,
) -> None:
    """Stage 6: Export trained model to deployment artifacts."""

    _setup_logging(cfg)

    start_time = time.monotonic()
    e = cfg.export

    logger.info("=" * 60)
    logger.info("  Stage 6: Export Model Artifact")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # ── Resolve paths ─────────────────────────────────────────────────────
    train_dir = paths.train_dir
    eval_dir = paths.eval_dir

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # ── Resolve model strategy + checkpoint ───────────────────────────────
    strategy = _resolve_model_strategy(e.get("model_strategy"))

    model_dir = train_dir / "final" / strategy
    if not model_dir.is_dir():
        final_dir = train_dir / "final"
        available = (
            [d.name for d in final_dir.iterdir() if d.is_dir()] if final_dir.is_dir() else []
        )
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Available: {available}")

    checkpoint = model_dir / "model.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Strategy:   {strategy}")

    # FoundationPaths is the single authority for artifact identity and layout.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    fp = artifact_config_fingerprint(cfg)
    output_dir = paths.artifact_dir
    display_dir = public_output_dir or output_dir
    artifact_name = display_dir.name

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — export_model.py")
        print(f"{'=' * 60}")
        print(f"  Experiment:   {cfg.exp_name}")
        print(f"  Strategy:     {strategy}")
        print(f"  Checkpoint:   {checkpoint}")
        print(f"  Output:       {display_dir}")
        print(f"  Formats:      {list(e.formats)}")
        print(f"  Fingerprint:  {fp}")
        print(f"  Validation:   {'skip' if e.skip_validation else 'full'}")
        print(f"{'=' * 60}\n")
        return

    from oceanpath.workflows.training import (
        training_run_fingerprint,
        validate_training_run_dir,
    )

    validate_training_run_dir(
        train_dir,
        expected_fingerprint=training_run_fingerprint(cfg),
    )

    # ── Run export ────────────────────────────────────────────────────────
    from oceanpath.export.exporter import Exporter
    from oceanpath.export.model_card import (
        ModelCard,
        load_oof_metrics,
        load_threshold_info,
    )

    logger.info("=" * 50)
    logger.info("  Part 1: Export Model")
    logger.info("=" * 50)

    exporter = Exporter(
        checkpoint_path=str(checkpoint),
        output_dir=str(output_dir),
        in_dim=cfg.encoder.feature_dim,
        num_classes=cfg.get("num_classes", 2),
        formats=list(e.formats),
        opset_version=e.opset_version,
        atol=e.atol,
        rtol=e.rtol,
        validation_n_patches=list(e.validation_n_patches),
        device=e.get("device", "cpu"),
        public_output_dir=str(display_dir),
    )

    export_report = exporter.run(skip_validation=e.skip_validation)
    if not export_report.get("success", False):
        raise RuntimeError(
            "Export or numerical validation failed; the partial artifact will not be "
            "committed. See the preceding export log for the failed check."
        )

    # ── Build model card ──────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("  Part 2: Model Card")
    logger.info("=" * 50)

    card = ModelCard(cfg_dict, export_report)

    # Add OOF metrics from training
    oof_metrics = load_oof_metrics(str(train_dir))
    if oof_metrics:
        card.add_metrics(oof_metrics)
        logger.info(f"Added {len(oof_metrics)} metric entries to model card")

    # Add thresholds from evaluation
    thresholds = load_threshold_info(str(eval_dir))
    if thresholds:
        card.add_thresholds(thresholds)
        logger.info("Added threshold info to model card")

    # Add export-specific metadata
    card.add_custom("model_strategy", strategy)
    card.add_custom("artifact_name", artifact_name)

    card.write(str(output_dir / "model_card.json"))

    # ── Save resolved config ──────────────────────────────────────────────
    config_path = output_dir / "export_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    elapsed = time.monotonic() - start_time

    # ── Final summary ─────────────────────────────────────────────────────
    _print_export_report(export_report, display_dir, strategy, elapsed)


def _print_export_report(
    report: dict,
    output_dir,
    strategy: str,
    elapsed: float,
) -> None:
    """Print structured export summary."""
    print(f"\n{'=' * 60}")
    print("  Stage 6 Complete")
    print(f"{'=' * 60}")
    print(f"  Output:       {output_dir}")
    print(f"  Time:         {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Model:        {strategy}")
    print(f"  Params:       {report.get('model_params', '?'):,}")

    # Validation results
    v = report.get("validation", {})
    print("\n  Validation:")
    for name, result in v.items():
        status = result.get("status", "?") if isinstance(result, dict) else "?"
        symbol = "✓" if status == "pass" else ("⚠" if status == "skipped" else "✗")
        print(f"    {name:25s}: {symbol} {status}")

    # Export results
    exports = report.get("exports", {})
    print("\n  Exports:")
    for fmt, info in exports.items():
        status = info.get("status", "?")
        size = info.get("size_mb", "?")
        symbol = "✓" if status == "success" else "✗"
        size_str = f" ({size} MB)" if isinstance(size, (int, float)) else ""
        print(f"    {fmt:15s}: {symbol} {status}{size_str}")

    print(f"\n  Overall:      {'✓ PASSED' if report.get('success') else '✗ FAILED'}")
    print(f"{'=' * 60}")

    # Serving hint
    print("\n  To serve this model:")
    print(f"    python scripts/serve.py export.artifact_dir={output_dir}")
    print()


def run_export(cfg: DictConfig) -> StageResult:
    """Package a trained model as a portable, validated artifact."""
    from oceanpath.runtime import run_context

    paths = FoundationPaths.from_config(cfg)
    started = time.monotonic()
    dry_run = bool(cfg.get("dry_run", False))
    if dry_run:
        with run_context(
            cfg,
            stage="export_model",
            output_dir=paths.artifact_dir,
            persist=False,
        ):
            _run_export(cfg, paths)
    else:
        required_files = [
            "export_report.json",
            "model_card.json",
            "model.ckpt",
            "export_config.yaml",
        ]
        formats = {str(value) for value in cfg.export.formats}
        if "onnx" in formats:
            required_files.append("model.onnx")
        if "torchscript" in formats:
            required_files.append("model.pt")
        validator = validate_files_exist(*required_files)

        with atomic_output(paths.artifact_dir, validator=validator) as staging_dir:
            staging_paths = replace(paths, artifact_dir=staging_dir)
            with run_context(
                cfg,
                stage="export_model",
                output_dir=staging_dir,
                persist=True,
            ):
                _run_export(
                    cfg,
                    staging_paths,
                    public_output_dir=paths.artifact_dir,
                )
    return StageResult(
        stage=PipelineStage.EXPORT_MODEL,
        status="dry_run" if dry_run else "completed",
        output_dir=paths.artifact_dir,
        elapsed_seconds=time.monotonic() - started,
        details={"training_dir": str(paths.train_dir)},
    )
