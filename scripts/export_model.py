"""
Stage 8: Export trained MIL model to deployment-ready artifacts.

Pipeline:
  ┌────────────────────┐
  │  Stage 5 (Train)   │  model.ckpt (Lightning checkpoint)
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
    python scripts/export_model.py platform=local data=gej encoder=univ1 \\
           splits=kfold5 model=abmil training=gej

    # Export ensemble model
    python scripts/export_model.py ... export.model_strategy=ensemble

    # ONNX only
    python scripts/export_model.py ... export.formats=[onnx]

    # Skip validation (faster, not recommended)
    python scripts/export_model.py ... export.skip_validation=true

    # Dry run
    python scripts/export_model.py ... dry_run=true
"""

import json
import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _setup_logging(cfg: DictConfig) -> None:
    """Configure logging to match train.py / evaluate.py style."""
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="export")
    fmt = f"%(asctime)s | %(levelname)-7s | exp={exp_name} | %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="export", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Stage 8: Export trained model to deployment artifacts."""

    _setup_logging(cfg)

    start_time = time.monotonic()
    e = cfg.export

    logger.info("=" * 60)
    logger.info("  Stage 8: Export + Serving Artifact")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # ── Resolve paths ─────────────────────────────────────────────────────
    train_dir = Path(cfg.train_dir)
    eval_dir = Path(cfg.eval_dir)

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # ── Resolve model strategy + checkpoint ───────────────────────────────
    strategy = e.get("model_strategy") or "best_fold"

    # Allow "auto" to read from Stage 6 recommendation
    if strategy == "auto":
        rec_path = eval_dir / "recommended_model.json"
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text())
            strategy = rec.get("recommended_model", "best_fold")
            logger.info(f"Stage 6 recommended: {strategy}")
        else:
            strategy = "best_fold"

    model_dir = train_dir / "final" / strategy
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            f"Available: {[d.name for d in (train_dir / 'final').iterdir() if d.is_dir()]}"
        )

    # Find checkpoint
    if strategy == "ensemble":
        # For ensemble, we export fold_0 as representative (users can also
        # serve ensemble via the PyTorch backend)
        ckpt_paths = sorted(model_dir.glob("fold_*.ckpt"))
        if not ckpt_paths:
            raise FileNotFoundError(f"No fold_*.ckpt in {model_dir}")
        checkpoint = ckpt_paths[0]
        logger.info(
            "Ensemble: exporting representative model (fold 0). "
            "For full ensemble inference, use serve.py with pytorch backend."
        )
    else:
        checkpoint = model_dir / "model.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Strategy:   {strategy}")

    # ── Compute artifact directory name ───────────────────────────────────
    from oceanpath.serving.exporter import config_fingerprint

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    fp = config_fingerprint(cfg_dict)
    artifact_name = f"{cfg.exp_name}_{fp}"
    output_dir = Path(e.artifact_root) / artifact_name

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — export_model.py")
        print(f"{'=' * 60}")
        print(f"  Experiment:   {cfg.exp_name}")
        print(f"  Strategy:     {strategy}")
        print(f"  Checkpoint:   {checkpoint}")
        print(f"  Output:       {output_dir}")
        print(f"  Formats:      {list(e.formats)}")
        print(f"  Fingerprint:  {fp}")
        print(f"  Validation:   {'skip' if e.skip_validation else 'full'}")
        print(f"{'=' * 60}\n")
        return

    # ── Run export ────────────────────────────────────────────────────────
    from oceanpath.serving.exporter import Exporter
    from oceanpath.serving.model_card import (
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
    )

    export_report = exporter.run(skip_validation=e.skip_validation)

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
    _print_export_report(export_report, output_dir, strategy, elapsed)


def _print_export_report(
    report: dict,
    output_dir,
    strategy: str,
    elapsed: float,
) -> None:
    """Print structured export summary."""
    print(f"\n{'=' * 60}")
    print("  Stage 8 Complete")
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


if __name__ == "__main__":
    main()
