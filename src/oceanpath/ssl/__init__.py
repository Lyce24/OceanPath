"""
OceanPath SSL pretraining package.

Two sub-packages:

  - :mod:`oceanpath.ssl.data`     — PretrainDataset, PretrainDataModule,
                                    augmentors, collators.
  - :mod:`oceanpath.ssl.modules`  — SSLPretrainModule, losses, heads,
                                    quality / linear-probe callbacks.

Four supported SSL methods at slide level:
  VICReg, JEPA (symmetric), LeJEPA-2C, LeJEPA-MC.

Top-level re-exports below cover the common one-liner imports
(``from oceanpath.ssl import VICRegLoss``, etc.). For deeper imports go
through the sub-package directly.
"""

# Re-export from training side.
# Re-export from data side.
from oceanpath.ssl.data import (
    DualViewAugmentor,
    PretrainDataModule,
    PretrainDataset,
    WSIDualViewAugmentConfig,
    WSIDualViewAugmentor,
    WSIMultiCropAugmentor,
    WSIViewSpec,
    build_augmentor,
    build_multicrop_augmentor,
    multicrop_stack_collate,
    stack_collate,
)
from oceanpath.ssl.modules import (
    AlphaReQCallback,
    DINOHead,
    EMANetwork,
    JEPALoss,
    LeJEPALoss,
    LeJEPAMCLoss,
    LinearProbeEvalCallback,
    LinearProbeTask,
    Predictor,
    Projector,
    RankMeCallback,
    SSLPretrainModule,
    SSLQualityCallback,
    VICRegLoss,
)

__all__ = [
    "AlphaReQCallback",
    "DINOHead",
    "DualViewAugmentor",
    "EMANetwork",
    "JEPALoss",
    "LeJEPALoss",
    "LeJEPAMCLoss",
    "LinearProbeEvalCallback",
    "LinearProbeTask",
    "Predictor",
    "PretrainDataModule",
    "PretrainDataset",
    "Projector",
    "RankMeCallback",
    "SSLPretrainModule",
    "SSLQualityCallback",
    "VICRegLoss",
    "WSIDualViewAugmentConfig",
    "WSIDualViewAugmentor",
    "WSIMultiCropAugmentor",
    "WSIViewSpec",
    "build_augmentor",
    "build_multicrop_augmentor",
    "multicrop_stack_collate",
    "stack_collate",
]
