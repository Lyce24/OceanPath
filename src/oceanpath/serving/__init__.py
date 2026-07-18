"""Optional HTTP serving for already-exported OceanPath artifacts.

Public API
──────────
  InferenceBackend   Unified inference backend (ONNX / TorchScript / PyTorch)
  create_app         FastAPI application factory
"""

from oceanpath.serving.server import InferenceBackend, create_app

__all__ = ["InferenceBackend", "create_app"]
