"""
Abstract base class for all MIL aggregators.

Defines the standardized forward contract that all models must follow:

    slide_embedding, logits, extras = model(features, mask=..., coords=...)

Where:
    - slide_embedding: [B, D] — single vector per slide
    - logits:          [B, C] — class predictions (None if num_classes=0)
    - extras:          dict   — attention_weights, etc. (always present, may be empty)

All aggregators inherit from this and implement `forward_features()`.
The WSIClassifier wrapper adds classification heads on top.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MILOutput:
    """Standardized output from any MIL aggregator."""

    slide_embedding: torch.Tensor  # [B, D]
    logits: torch.Tensor | None  # [B, C] or None
    extras: dict  # attention_weights, etc.


class BaseMIL(ABC, nn.Module):
    """
    Abstract base for MIL aggregators (ABMIL, TransMIL, MeanPool, etc.).

    Subclasses must implement `forward_features()` which maps
    [B, N, D_in] → [B, D_embed] slide embedding + extras dict.

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension (e.g., 1024 for UNI).
    embed_dim : int
        Internal embedding dimension after patch projection.
    gradient_checkpointing : bool
        If True, subclasses should use torch.utils.checkpoint on
        memory-heavy layers (attention).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.gradient_checkpointing = gradient_checkpointing

    @abstractmethod
    def forward_features(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patch features into a slide-level embedding.

        Parameters
        ----------
        h : torch.Tensor
            [B, N, D] patch embeddings.
        mask : torch.Tensor, optional
            [B, N] binary mask (1=valid, 0=padding). None means all valid.
        coords : torch.Tensor, optional
            [B, N, 2] spatial coordinates (used by some models, e.g., TransMIL).
        return_attention : bool
            If True, include attention weights in the extras dict.

        Returns
        -------
        slide_embedding : torch.Tensor
            [B, D_embed] aggregated slide representation.
        extras : dict
            May contain 'attention_weights': [B, N] or [B, K, N].
        """
        ...

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> MILOutput:
        """
        Full forward pass: optional subsampling → aggregation.

        Parameters
        ----------
        features : torch.Tensor
            [B, N, D] or [N, D] (auto-batched). Patch embeddings.
        mask : torch.Tensor, optional
            [B, N] binary mask.
        coords : torch.Tensor, optional
            [B, N, 2] coordinates.
        return_attention : bool
            Whether to return attention weights.

        Returns
        -------
        MILOutput
            slide_embedding, logits=None, extras.
        """
        # Auto-batch if 2D input
        features = self._ensure_batched(features)
        if mask is not None:
            mask = self._ensure_batched_2d(mask)
        if coords is not None:
            coords = self._ensure_batched(coords)

        slide_embedding, extras = self.forward_features(
            features, mask=mask, coords=coords, return_attention=return_attention
        )

        return MILOutput(
            slide_embedding=slide_embedding,
            logits=None,
            extras=extras,
        )

    # ── Shape helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _ensure_batched(t: torch.Tensor) -> torch.Tensor:
        """[N, D] → [1, N, D]; [B, N, D] → pass through."""
        if t.ndim == 2:
            return t.unsqueeze(0)
        return t

    @staticmethod
    def _ensure_batched_2d(t: torch.Tensor) -> torch.Tensor:
        """[N] → [1, N]; [B, N] → pass through."""
        if t.ndim == 1:
            return t.unsqueeze(0)
        return t

    # ── Weight initialization ─────────────────────────────────────────────

    def initialize_weights(self) -> None:
        """Reset all learnable parameters using PyTorch defaults."""
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()
