"""Stable contracts shared across workflow and orchestration layers."""

from oceanpath.contracts.slide_ids import KNOWN_SLIDE_EXTENSIONS, normalize_slide_id
from oceanpath.contracts.stages import PipelineStage, StageResult

__all__ = [
    "KNOWN_SLIDE_EXTENSIONS",
    "PipelineStage",
    "StageResult",
    "normalize_slide_id",
]
