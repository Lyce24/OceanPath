"""
Stage 8: Export + Serving.

Packages trained models into stable, portability-validated inference artifacts
and provides a lightweight serving layer for precomputed-feature inference.

Public API
──────────
  Exporter           Export models to ONNX / TorchScript with validation
  ModelCard          Build provenance-rich model cards
  InferenceBackend   Unified inference backend (ONNX / TorchScript / PyTorch)
  create_app         FastAPI application factory
"""

from oceanpath.serving.exporter import Exporter
from oceanpath.serving.model_card import ModelCard
from oceanpath.serving.server import InferenceBackend, create_app

__all__ = ["Exporter", "InferenceBackend", "ModelCard", "create_app"]
