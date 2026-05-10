"""
OceanPath data loading pipeline (supervised MIL).

Provides memmap-backed datasets and Lightning DataModules for supervised
MIL training. The SSL pretraining dataset / DataModule has moved to
``oceanpath.ssl.data`` — import from there directly:

    from oceanpath.ssl.data import PretrainDataset, PretrainDataModule
"""

from oceanpath.data.datamodule import MILCollator, MILDataModule, SimpleMILCollator
from oceanpath.data.dataset import BaseMmapDataset, MmapDataset

__all__ = [
    "BaseMmapDataset",
    "MILCollator",
    "MILDataModule",
    "MmapDataset",
    "SimpleMILCollator",
]
