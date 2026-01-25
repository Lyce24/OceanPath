from __future__ import annotations


import gc
import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl

import copy
import os
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping

from data.data_module import WSIMILDataModule


# =============================================================================
# DataModule Variants
# =============================================================================
class FullTrainDataModule(WSIMILDataModule):
    """All data for training, no validation."""

    def _build_splits(self):
        self.train_ids = self.data_df[self.id_col].astype(str).tolist()
        self.val_ids = []
        self.test_ids = []


class TestOnlyDataModule(WSIMILDataModule):
    """All data for testing only."""

    def _build_splits(self):
        self.train_ids = []
        self.val_ids = []
        self.test_ids = self.data_df[self.id_col].astype(str).tolist()

# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def cleanup_memory() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV, Excel, or Parquet."""
    path_lower = path.lower()
    if path_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    elif path_lower.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _is_head_param(name: str) -> bool:
    """
    Check if a parameter belongs to the classification head.
    
    Head parameters typically contain these keywords in their names.
    """
    head_keywords = [
        'head', 'heads', 'classifier', 'fc', 'final', 
        'linear_out', 'output_layer', 'mlp_head'
    ]
    name_lower = name.lower()
    return any(kw in name_lower for kw in head_keywords)


def _is_encoder_param(name: str) -> bool:
    """
    Check if a parameter belongs to the encoder.
    
    Encoder parameters typically contain these keywords.
    """
    encoder_keywords = [
        'encoder', 'feature_encoder', 'aggregator', 'attention',
        'transformer', 'embed', 'projection', 'norm'
    ]
    name_lower = name.lower()
    # It's encoder if it matches encoder keywords OR doesn't match head keywords
    return any(kw in name_lower for kw in encoder_keywords) or not _is_head_param(name)
