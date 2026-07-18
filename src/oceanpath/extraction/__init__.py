"""Patch-feature extraction domain.

The public API accepts typed configuration and contains no Hydra or CLI code.
"""

from oceanpath.extraction.trident import (
    TridentExtractionConfig,
    ValidationError,
    compute_encoder_fingerprint,
    run_pipeline,
    validate_inputs,
    validate_outputs,
)

__all__ = [
    "TridentExtractionConfig",
    "ValidationError",
    "compute_encoder_fingerprint",
    "run_pipeline",
    "validate_inputs",
    "validate_outputs",
]
