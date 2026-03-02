"""
DAG-based pipeline orchestration with atomic stage outputs.

Provides make-like incremental builds for the OceanPath pipeline.
"""

from oceanpath.pipeline.dag import (
    PipelineRunner,
    Stage,
    build_oceanpath_pipeline,
    build_pretraining_pipeline,
    build_supervised_pipeline,
)
from oceanpath.pipeline.transactions import atomic_output

__all__ = [
    "PipelineRunner",
    "Stage",
    "atomic_output",
    "build_oceanpath_pipeline",
    "build_pretraining_pipeline",
    "build_supervised_pipeline",
]
