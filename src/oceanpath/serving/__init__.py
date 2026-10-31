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

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the names the lazy __getattr__ below provides
    from oceanpath.serving.exporter import Exporter
    from oceanpath.serving.model_card import ModelCard
    from oceanpath.serving.server import InferenceBackend, create_app

# Resolved on first access so that serving an ONNX artifact (oceanpath.serving.server)
# never imports torch, which only the exporter and the PyTorch backends need.
_EXPORTS = {
    "Exporter": "oceanpath.serving.exporter",
    "ModelCard": "oceanpath.serving.model_card",
    "InferenceBackend": "oceanpath.serving.server",
    "create_app": "oceanpath.serving.server",
}

__all__ = ["Exporter", "InferenceBackend", "ModelCard", "create_app"]


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
