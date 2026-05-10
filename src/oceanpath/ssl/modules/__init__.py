"""
SSL training pipeline.

Lightning module, losses, heads, and quality / linear-probe callbacks for
self-supervised slide-level pretraining. The data side lives next door in
``oceanpath.ssl.data``.

Public symbols:
  - SSLPretrainModule
  - VICRegLoss, JEPALoss, LeJEPALoss, LeJEPAMCLoss
  - Projector, Predictor, DINOHead, EMANetwork
  - RankMeCallback, AlphaReQCallback, SSLQualityCallback
  - LinearProbeEvalCallback, LinearProbeTask
"""

from oceanpath.ssl.modules.callbacks import (
    AlphaReQCallback,
    LinearProbeEvalCallback,
    LinearProbeTask,
    RankMeCallback,
    SSLQualityCallback,
)
from oceanpath.ssl.modules.heads import (
    DINOHead,
    EMANetwork,
    Predictor,
    Projector,
)
from oceanpath.ssl.modules.losses import (
    JEPALoss,
    LeJEPALoss,
    LeJEPAMCLoss,
    VICRegLoss,
)
from oceanpath.ssl.modules.pretrain_module import SSLPretrainModule

__all__ = [
    "AlphaReQCallback",
    "DINOHead",
    "EMANetwork",
    "JEPALoss",
    "LeJEPALoss",
    "LeJEPAMCLoss",
    "LinearProbeEvalCallback",
    "LinearProbeTask",
    "Predictor",
    "Projector",
    "RankMeCallback",
    "SSLPretrainModule",
    "SSLQualityCallback",
    "VICRegLoss",
]
