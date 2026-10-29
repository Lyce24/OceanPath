"""
SSL data pipeline.

Dataset, DataModule, augmentor, and collators for self-supervised
slide-level pretraining. Lives next to ``oceanpath.ssl.modules`` (the
training pipeline) so the data side and the model side are cleanly
separable.

Public symbols:
  - PretrainDataset, PretrainDataModule
  - WSIDualViewAugmentor, WSIDualViewAugmentConfig, DualViewAugmentor
  - WSIMultiCropAugmentor, WSIViewSpec
  - build_augmentor, build_multicrop_augmentor
  - stack_collate, multicrop_stack_collate
"""

from oceanpath.ssl.data.augmentation import (
    DualViewAugmentor,
    SimpleAugmentConfig,
    WSIDualViewAugmentConfig,
    WSIDualViewAugmentor,
    WSIMultiCropAugmentor,
    WSIViewSpec,
    build_augmentor,
    build_multicrop_augmentor,
)
from oceanpath.ssl.data.datamodule import (
    PretrainDataModule,
    multicrop_stack_collate,
    stack_collate,
)
from oceanpath.ssl.data.dataset import PretrainDataset

__all__ = [
    "DualViewAugmentor",
    "PretrainDataModule",
    "PretrainDataset",
    "SimpleAugmentConfig",
    "WSIDualViewAugmentConfig",
    "WSIDualViewAugmentor",
    "WSIMultiCropAugmentor",
    "WSIViewSpec",
    "build_augmentor",
    "build_multicrop_augmentor",
    "multicrop_stack_collate",
    "stack_collate",
]
