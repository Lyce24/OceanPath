"""
OceanPath training modules.

Provides Lightning modules and callbacks for:
  - MIL supervised training (MILTrainModule)
  - Model finalization after CV (finalize_models)
  - Training callbacks (BagCurriculum, FoldTiming, etc.)
"""

from oceanpath.modules.callbacks import (
    BagCurriculumCallback,
    BatchIndexLogger,
    FoldTimingCallback,
    MetadataWriter,
    WandbFoldSummary,
)
from oceanpath.modules.finalize import finalize_models
from oceanpath.modules.train_module import MILTrainModule

__all__ = [
    "BagCurriculumCallback",
    "BatchIndexLogger",
    "FoldTimingCallback",
    "MILTrainModule",
    "MetadataWriter",
    "WandbFoldSummary",
    "finalize_models",
]
