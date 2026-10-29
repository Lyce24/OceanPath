"""
FastAPI serving layer for exported MIL models.

Serves precomputed features → predictions. This is the realistic deployment
path for pathology models because feature extraction (encoder + TRIDENT)
is the expensive step that happens offline.

Architecture:
═════════════
    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Client      │───▶│  /predict    │───▶│  ONNX or PT  │
    │  (features)  │    │  (FastAPI)   │    │  Runtime      │
    └─────────────┘    └──────────────┘    └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  model_card  │  ← provenance
                       │  .json       │
                       └──────────────┘

Endpoints:
  GET  /health         → liveness + model info
  GET  /model-card     → full model card JSON
  POST /predict        → single slide inference
  POST /predict/batch  → multi-slide batch inference

Backends:
  - ONNX (preferred): onnxruntime, CPU or GPU
  - PyTorch: fallback, loads from .ckpt or .pt

Why feature-level serving?
══════════════════════════
Raw WSI → prediction requires the full extraction pipeline (TRIDENT + encoder).
That's a separate, heavier system. This server handles the lightweight
"last mile": precomputed embeddings → class prediction, which is:
  - 10-100ms per slide (vs minutes for full WSI extraction)
  - CPU-deployable (no GPU required for MIL inference)
  - Batch-friendly (process entire cohorts in seconds)
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Inference Backend
# ═════════════════════════════════════════════════════════════════════════════


class InferenceBackend:
    """
    Unified inference backend supporting ONNX and PyTorch models.

    Loads from an artifact directory produced by Exporter.run():
      artifacts/{exp_name}/
        ├── model.onnx          → ONNX backend
        ├── model.pt            → TorchScript backend
        ├── model.ckpt          → PyTorch/Lightning fallback
        └── model_card.json     → metadata
    """

    def __init__(
        self,
        artifact_dir: str,
        backend: str = "auto",
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        artifact_dir : path to exported artifact directory.
        backend : "onnx", "torchscript", "pytorch", or "auto".
        device : "cpu" or "cuda:0" (PyTorch backends only).
        """
        self.artifact_dir = Path(artifact_dir)
        self.device = device
        self._session = None
        self._model = None
        self._backend_name = "none"

        # Load model card
        card_path = self.artifact_dir / "model_card.json"
        if card_path.is_file():
            self.model_card = json.loads(card_path.read_text())
        else:
            self.model_card = {}

        # Auto-detect or load specified backend
        if backend == "auto":
            self._auto_load()
        elif backend == "onnx":
            self._load_onnx()
        elif backend == "torchscript":
            self._load_torchscript()
        elif backend == "pytorch":
            self._load_pytorch()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        logger.info(f"InferenceBackend: {self._backend_name} loaded from {self.artifact_dir}")

    def _auto_load(self):
        """Try ONNX → TorchScript → PyTorch in order."""
        onnx_path = self.artifact_dir / "model.onnx"
        ts_path = self.artifact_dir / "model.pt"
        ckpt_path = self.artifact_dir / "model.ckpt"

        if onnx_path.is_file():
            try:
                self._load_onnx()
                return
            except Exception as e:
                logger.warning(f"ONNX load failed, trying next: {e}")

        if ts_path.is_file():
            try:
                self._load_torchscript()
                return
            except Exception as e:
                logger.warning(f"TorchScript load failed, trying next: {e}")

        if ckpt_path.is_file():
            self._load_pytorch()
            return

        raise FileNotFoundError(
            f"No loadable model found in {self.artifact_dir}. "
            f"Expected model.onnx, model.pt, or model.ckpt."
        )

    def _load_onnx(self):
        """Load ONNX model via onnxruntime."""
        import onnxruntime as ort

        path = self.artifact_dir / "model.onnx"
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in self.device
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(path), providers=providers)
        self._backend_name = "onnx"

    def _load_torchscript(self):
        """Load TorchScript model."""
        import torch

        path = self.artifact_dir / "model.pt"
        self._model = torch.jit.load(str(path), map_location=self.device)
        self._model.eval()
        self._backend_name = "torchscript"

    def _load_pytorch(self):
        """Load from Lightning checkpoint."""
        from oceanpath.modules.train_module import MILTrainModule
        from oceanpath.serving.exporter import _OnnxExportWrapper

        path = self.artifact_dir / "model.ckpt"
        try:
            module = MILTrainModule.load_from_checkpoint(
                str(path),
                weights_only=False,
                map_location=self.device,
            )
        except TypeError:
            module = MILTrainModule.load_from_checkpoint(
                str(path),
                map_location=self.device,
            )

        wrapper = _OnnxExportWrapper(module.model)
        wrapper.eval()
        wrapper.to(self.device)
        self._model = wrapper
        self._backend_name = "pytorch"

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(
        self,
        features: np.ndarray,
    ) -> dict:
        """
        Run inference on precomputed features.

        Parameters
        ----------
        features : [N, D] or [B, N, D] float32 array of patch embeddings.

        Returns
        -------
        dict with logits, probabilities, embedding, pred_class, pred_prob.
        """
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # [1, N, D]

        if self._backend_name == "onnx":
            return self._predict_onnx(features)
        return self._predict_torch(features)

    def _predict_onnx(self, features: np.ndarray) -> dict:
        """Run ONNX inference."""
        features = features.astype(np.float32)
        outputs = self._session.run(
            None,
            {"features": features},
        )
        logits = outputs[0]  # [B, C]
        probs = outputs[1]  # [B, C]
        embedding = outputs[2]  # [B, D]

        return _build_result(logits, probs, embedding)

    def _predict_torch(self, features: np.ndarray) -> dict:
        """Run PyTorch/TorchScript inference."""
        import torch

        x = torch.from_numpy(features.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits, probs, embedding = self._model(x)

        return _build_result(
            logits.cpu().numpy(),
            probs.cpu().numpy(),
            embedding.cpu().numpy(),
        )

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def info(self) -> dict:
        """Summary info for health checks."""
        return {
            "backend": self._backend_name,
            "artifact_dir": str(self.artifact_dir),
            "experiment": self.model_card.get("experiment_name", "unknown"),
            "schema_version": self.model_card.get("schema_version"),
            "model_arch": (self.model_card.get("model", {}).get("architecture", "unknown")),
            "validation_passed": (self.model_card.get("export", {}).get("validation_passed")),
        }


def _build_result(
    logits: np.ndarray,
    probs: np.ndarray,
    embedding: np.ndarray,
) -> dict:
    """Build standardized prediction result dict."""
    # Squeeze batch dim if single slide
    if logits.shape[0] == 1:
        logits = logits.squeeze(0)
        probs = probs.squeeze(0)
        embedding = embedding.squeeze(0)

    pred_class = int(np.argmax(probs, axis=-1)) if probs.ndim == 1 else None
    pred_prob = float(probs.max()) if probs.ndim == 1 else None

    return {
        "logits": logits.tolist(),
        "probabilities": probs.tolist(),
        "embedding": embedding.tolist(),
        "pred_class": pred_class,
        "pred_prob": pred_prob,
    }


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═════════════════════════════════════════════════════════════════════════════


def create_app(
    artifact_dir: str,
    backend: str = "auto",
    device: str = "cpu",
):
    """
    Create a FastAPI application for serving predictions.

    Usage
    ─────
        # In scripts/serve.py:
        app = create_app("artifacts/exp_abc123")
        uvicorn.run(app, host="0.0.0.0", port=8000)

    Or from CLI:
        python scripts/serve.py export.artifact_dir=artifacts/exp_abc123
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    # ── Initialize backend ────────────────────────────────────────────────
    engine = InferenceBackend(
        artifact_dir=artifact_dir,
        backend=backend,
        device=device,
    )

    # ── Request/response schemas ──────────────────────────────────────────

    class PredictRequest(BaseModel):
        features: list[list[float]] = Field(
            ...,
            description=(
                "Patch embeddings as [N, D] nested list. "
                "N = number of patches (variable), D = feature dimension."
            ),
        )
        slide_id: str | None = Field(
            None,
            description="Optional slide identifier for logging.",
        )

    class BatchPredictRequest(BaseModel):
        slides: list[PredictRequest] = Field(
            ...,
            description="List of slides to predict.",
        )

    class PredictResponse(BaseModel):
        slide_id: str | None = None
        pred_class: int | None = None
        pred_prob: float | None = None
        probabilities: list[float] = []
        logits: list[float] = []
        n_patches: int = 0
        latency_ms: float = 0.0

    class HealthResponse(BaseModel):
        status: str
        backend: str
        experiment: str
        model_arch: str
        validation_passed: bool | None = None

    # ── App ───────────────────────────────────────────────────────────────
    app = FastAPI(
        title="OceanPath MIL Inference",
        description=(
            "Serves precomputed patch features → class predictions "
            "for whole-slide image classification."
        ),
        version="1.0.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        info = engine.info
        return HealthResponse(
            status="healthy",
            backend=info["backend"],
            experiment=info["experiment"],
            model_arch=info["model_arch"],
            validation_passed=info["validation_passed"],
        )

    @app.get("/model-card")
    def model_card():
        if not engine.model_card:
            raise HTTPException(404, "No model card available")
        return engine.model_card

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        start = time.monotonic()

        features = np.array(req.features, dtype=np.float32)
        if features.ndim != 2:
            raise HTTPException(
                400,
                f"Expected 2D features [N, D], got shape {features.shape}",
            )

        result = engine.predict(features)
        latency = (time.monotonic() - start) * 1000

        return PredictResponse(
            slide_id=req.slide_id,
            pred_class=result["pred_class"],
            pred_prob=result["pred_prob"],
            probabilities=result["probabilities"],
            logits=result["logits"],
            n_patches=features.shape[0],
            latency_ms=round(latency, 2),
        )

    @app.post("/predict/batch", response_model=list[PredictResponse])
    def predict_batch(req: BatchPredictRequest):
        responses = []
        for slide_req in req.slides:
            start = time.monotonic()

            features = np.array(slide_req.features, dtype=np.float32)
            if features.ndim != 2:
                raise HTTPException(
                    400,
                    f"Slide {slide_req.slide_id}: expected 2D features, got {features.shape}",
                )

            result = engine.predict(features)
            latency = (time.monotonic() - start) * 1000

            responses.append(
                PredictResponse(
                    slide_id=slide_req.slide_id,
                    pred_class=result["pred_class"],
                    pred_prob=result["pred_prob"],
                    probabilities=result["probabilities"],
                    logits=result["logits"],
                    n_patches=features.shape[0],
                    latency_ms=round(latency, 2),
                )
            )

        return responses

    return app
