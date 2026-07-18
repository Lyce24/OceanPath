"""Pipeline stage names and workflow result contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class PipelineStage(str, Enum):
    """Essential stages shipped by the foundation branch."""

    EXTRACT_FEATURES = "extract_features"
    BUILD_MMAP = "build_mmap"
    SPLIT_DATA = "split_data"
    TRAIN_MODEL = "train_model"
    EVALUATE = "evaluate"
    EXPORT_MODEL = "export_model"


@dataclass(frozen=True)
class StageResult:
    """Serializable result returned by every top-level workflow."""

    stage: PipelineStage
    status: str
    output_dir: Path
    elapsed_seconds: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        data["output_dir"] = str(self.output_dir)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
