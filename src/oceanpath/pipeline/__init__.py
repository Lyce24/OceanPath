"""
DAG-based pipeline orchestration with atomic stage outputs.

Provides make-like incremental builds for the OceanPath pipeline.
"""

from oceanpath.pipeline.dag import PipelineRunner, Stage
from oceanpath.pipeline.transactions import atomic_output

__all__ = ["PipelineRunner", "Stage", "atomic_output"]
