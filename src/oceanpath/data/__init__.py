"""
OceanPath data loading pipeline.

Provides memmap-backed datasets and Lightning DataModules for:
  - Supervised MIL training (MmapDataset, MILDataModule)
  - SSL pretraining (PretrainDataset, PretrainDataModule)
"""

from oceanpath.data.batching import (
    COORD_PAD_VALUE,
    BatchingConfig,
    BatchingStrategy,
    DualViewCollator,
    FixedNCollator,
    JEPACollator,
    MultiCropCollator,
    PadToGlobalCollator,
    RegionalCropCollator,
    SequencePackingCollator,
)
from oceanpath.data.datamodule import MILCollator, MILDataModule, SimpleMILCollator
from oceanpath.data.dataset import BaseMmapDataset, MmapDataset
from oceanpath.data.pretrain_datamodule import PretrainDataModule
from oceanpath.data.pretrain_dataset import PretrainDataset

__all__ = [
    "COORD_PAD_VALUE",
    "BaseMmapDataset",
    "BatchingConfig",
    "BatchingStrategy",
    "DualViewCollator",
    "FixedNCollator",
    "JEPACollator",
    "MILCollator",
    "MILDataModule",
    "MmapDataset",
    "MultiCropCollator",
    "PadToGlobalCollator",
    "PretrainDataModule",
    "PretrainDataset",
    "RegionalCropCollator",
    "SequencePackingCollator",
    "SimpleMILCollator",
]
