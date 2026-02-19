"""
Transformer-Based Multiple Instance Learning (TransMIL).

Shao et al., "TransMIL: Transformer based Correlated Multiple Instance Learning
for Whole Slide Image Classification", NeurIPS 2021.

Architecture:
    patches [B, N, D] → MLP → [B, N, E]
    → prepend CLS token → Nystrom Attention blocks (with PPEG) → CLS token → [B, E]

Requires: pip install nystrom-attention

Supports:
    - Gradient checkpointing per transformer block
    - Float32 attention softmax
    - Variable bag sizes (padding-aware via spatial grid)
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from typing import Optional

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import create_mlp


class TransLayer(nn.Module):
    """Single Nystrom attention transformer block."""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        from nystrom_attention import NystromAttention

        self.norm = nn.LayerNorm(dim)
        self.attention = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] → [B, L, D] with residual."""
        return x + self.attention(self.norm(x))


class PPEG(nn.Module):
    """
    Pyramid Positional Encoding Generator.

    Applies multi-scale depthwise convolutions to the spatial grid
    of patch tokens, then adds back to the token sequence.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: [B, 1 + H*W, D] (CLS token prepended).
        Returns same shape with positional encoding added to patch tokens.
        """
        B, _, D = x.shape
        cls_tok = x[:, :1, :]                                    # [B, 1, D]
        feat_tok = x[:, 1:, :]                                   # [B, H*W, D]
        feat_map = feat_tok.transpose(1, 2).view(B, D, H, W)     # [B, D, H, W]

        y = (
            self.proj(feat_map)
            + feat_map
            + self.proj1(feat_map)
            + self.proj2(feat_map)
        )
        y = y.flatten(2).transpose(1, 2)                         # [B, H*W, D]
        return torch.cat([cls_tok, y], dim=1)                     # [B, 1+H*W, D]


class TransMIL(BaseMIL):
    """
    TransMIL aggregator.

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension.
    embed_dim : int
        Transformer hidden dimension.
    num_fc_layers : int
        FC layers in patch embedding MLP.
    num_attention_layers : int
        Number of Nystrom attention blocks.
    num_heads : int
        Attention heads per block.
    dropout : float
        Dropout rate.
    gradient_checkpointing : bool
        Checkpoint each transformer block.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        num_attention_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.25,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            embed_dim=embed_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_layer = PPEG(dim=embed_dim)

        self.blocks = nn.ModuleList([
            TransLayer(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_attention_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def forward_features(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patches via transformer blocks, extract CLS token.

        Parameters
        ----------
        h : torch.Tensor
            [B, N, D] patch embeddings.
        mask : unused (TransMIL pads to square grid internally).
        coords : unused.
        return_attention : bool
            If True, compute CLS-to-patch dot-product attention from first block.

        Returns
        -------
        slide_embedding : torch.Tensor [B, E]
        extras : dict with optional 'attention_weights' [B, N]
        """
        h = self.patch_embed(h)  # [B, N, E]
        B, N, D = h.shape

        # Pad to square grid for PPEG convolutions
        Hs = int(math.ceil(math.sqrt(N)))
        Ws = Hs
        pad = Hs * Ws - N
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)  # [B, Hs*Ws, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1).to(h.device)  # [B, 1, D]
        h = torch.cat([cls, h], dim=1)                        # [B, 1+Hs*Ws, D]

        # Transformer blocks with optional checkpointing
        extras = {}
        for i, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                h = cp.checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)

            # Capture attention from first block (CLS vs patch tokens)
            if i == 0 and return_attention:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    cls_tok = h[:, :1, :].float()      # [B, 1, D]
                    feats = h[:, 1:, :].float()         # [B, Hs*Ws, D]
                    attn = (feats @ cls_tok.transpose(-1, -2)).squeeze(-1)  # [B, Hs*Ws]
                    extras["attention_weights"] = attn[:, :N]  # trim padding → [B, N]

            # Positional encoding between blocks
            h = self.pos_layer(h, Hs, Ws)

        # Extract CLS token as slide embedding
        slide_embedding = self.norm(h)[:, 0, :]  # [B, D]

        return slide_embedding, extras