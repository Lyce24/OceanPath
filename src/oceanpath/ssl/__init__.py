"""
OceanPath SSL pretraining module.

Provides five self-supervised learning methods for slide-level pretraining:
  VICReg, SimCLR, BYOL, DINO, JEPA

Key components:
  - SSLPretrainModule: Unified Lightning module for all SSL methods
  - Losses: VICRegLoss, SimCLRLoss, BYOLLoss, DINOLoss, JEPALoss
  - Heads: Projector, Predictor, DINOHead, EMANetwork
  - Callbacks: RankMeCallback, AlphaReQCallback, SSLQualityCallback
"""

from oceanpath.ssl.callbacks import AlphaReQCallback, RankMeCallback, SSLQualityCallback
from oceanpath.ssl.heads import DINOHead, EMANetwork, Predictor, Projector
from oceanpath.ssl.losses import BYOLLoss, DINOLoss, JEPALoss, SimCLRLoss, VICRegLoss
from oceanpath.ssl.pretrain_module import SSLPretrainModule

__all__ = [
    "AlphaReQCallback",
    "BYOLLoss",
    "DINOHead",
    "DINOLoss",
    "EMANetwork",
    "JEPALoss",
    "Predictor",
    "Projector",
    "RankMeCallback",
    "SSLPretrainModule",
    "SSLQualityCallback",
    "SimCLRLoss",
    "VICRegLoss",
]
