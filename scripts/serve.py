"""
Stage 8: Serve an exported model via REST API.

Launches a FastAPI server that accepts precomputed patch features
and returns class predictions. This is the lightweight "last mile"
of the inference pipeline.

Data pipeline:
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │ OFFLINE (heavy, GPU, minutes per slide)                     │
  │                                                             │
  │  Raw WSI ──▶ TRIDENT ──▶ Encoder ──▶ H5 features ──▶ mmap │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼  precomputed [N, D] features
  ┌─────────────────────────────────────────────────────────────┐
  │ ONLINE (lightweight, CPU, ~10ms per slide)                  │
  │                                                             │
  │  Features ──▶  serve.py (this script)  ──▶  Predictions    │
  │               ┌────────────────────┐                        │
  │               │  ONNX or PyTorch   │                        │
  │               │  MIL aggregator    │                        │
  │               └────────────────────┘                        │
  └─────────────────────────────────────────────────────────────┘

Endpoints:
  GET  /health         → liveness + model metadata
  GET  /model-card     → full provenance JSON
  POST /predict        → single-slide inference
  POST /predict/batch  → multi-slide batch inference

Usage:
    # Serve with default settings
    python scripts/serve.py export.artifact_dir=artifacts/my_experiment_abc123

    # Custom port and host
    python scripts/serve.py ... serve.host=0.0.0.0 serve.port=8080

    # Force ONNX backend
    python scripts/serve.py ... serve.backend=onnx

    # GPU inference
    python scripts/serve.py ... serve.device=cuda:0

    # Dry run (print config, don't start server)
    python scripts/serve.py ... dry_run=true

Client example (Python):
    import requests, numpy as np

    features = np.random.randn(500, 1024).tolist()  # 500 patches, 1024-dim
    resp = requests.post(
        "http://localhost:8000/predict",
        json={"features": features, "slide_id": "GEJ_001"},
    )
    print(resp.json())
    # {"pred_class": 1, "pred_prob": 0.87, "probabilities": [...], ...}
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _setup_logging(cfg: DictConfig) -> None:
    """Configure logging to match pipeline style."""
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="serve")
    fmt = f"%(asctime)s | %(levelname)-7s | exp={exp_name} | %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


@hydra.main(config_path="../configs", config_name="export", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Stage 8: Launch model serving API."""

    _setup_logging(cfg)

    s = cfg.serve
    artifact_dir = cfg.export.artifact_dir

    logger.info("=" * 60)
    logger.info("  Stage 8: Model Serving")
    logger.info("=" * 60)

    if not artifact_dir or artifact_dir == "null":
        raise ValueError(
            "No artifact directory specified. Set export.artifact_dir:\n"
            "  python scripts/serve.py "
            "export.artifact_dir=artifacts/my_experiment_abc123"
        )

    logger.info(f"Artifact dir: {artifact_dir}")
    logger.info(f"Backend:      {s.backend}")
    logger.info(f"Device:       {s.device}")
    logger.info(f"Host:         {s.host}:{s.port}")

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — serve.py")
        print(f"{'=' * 60}")
        print(f"  Artifact:  {artifact_dir}")
        print(f"  Backend:   {s.backend}")
        print(f"  Device:    {s.device}")
        print(f"  Address:   {s.host}:{s.port}")
        print(f"  Workers:   {s.workers}")
        print(f"{'=' * 60}\n")
        return

    # ── Create app + launch ───────────────────────────────────────────────
    from oceanpath.serving.server import create_app

    app = create_app(
        artifact_dir=str(artifact_dir),
        backend=s.backend,
        device=s.device,
    )

    import uvicorn

    uvicorn.run(
        app,
        host=s.host,
        port=s.port,
        workers=s.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
