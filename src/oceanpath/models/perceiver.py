"""
Perceiver MIL Aggregator.

Architecture:
    patches [B, N, D] → MLP → [B, N, E]
    → Cross-Attention(latents, patches) → Self-Attention(latents)
    → mean pool latents → [B, E]

Uses M learned latent tokens (M << N) that cross-attend to the N input
patches, achieving O(M*N) complexity instead of O(N²).

No external dependencies — pure PyTorch multihead attention.

Supports:
    - Configurable number of latent tokens and layers
    - Gradient checkpointing per cross-attn + self-attn block
    - Float32 attention softmax under AMP
    - Variable bag sizes with attention masking
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import create_mlp


class PerceiverBlock(nn.Module):
    """
    Single Perceiver block: cross-attention (latents ← patches) + self-attention (latents).

    Parameters
    ----------
    dim : int
        Latent/patch embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, dim: int = 512, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # Cross-attention: latents attend to patches
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention among latents
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN after self-attention
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        patches: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        latents : [B, M, D]
        patches : [B, N, D]
        key_padding_mask : [B, N] bool, True = ignored position

        Returns
        -------
        latents : [B, M, D]
        cross_attn_weights : [B, M, N]
        """
        # Cross-attention: latents query patches
        q = self.cross_norm_q(latents)
        kv = self.cross_norm_kv(patches)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            cross_out, cross_weights = self.cross_attn(
                q.float(),
                kv.float(),
                kv.float(),
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,  # [B, M, N] averaged over heads
            )
        latents = latents + cross_out.to(latents.dtype)

        # Self-attention among latents
        h = self.self_norm(latents)
        self_out, _ = self.self_attn(h, h, h, need_weights=False)
        latents = latents + self_out

        # FFN
        latents = latents + self.ffn(self.ffn_norm(latents))

        return latents, cross_weights


class PerceiverMIL(BaseMIL):
    """
    Perceiver MIL aggregator.

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension.
    embed_dim : int
        Embedding dimension after patch MLP.
    num_fc_layers : int
        FC layers in patch embedding MLP.
    num_latents : int
        Number of learned latent tokens (M << N).
    num_layers : int
        Number of cross-attn + self-attn blocks.
    num_heads : int
        Attention heads per block.
    dropout : float
        Dropout rate.
    gradient_checkpointing : bool
        Checkpoint each Perceiver block.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        num_latents: int = 32,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.25,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            embed_dim=embed_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.num_latents = num_latents

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False,
        )

        # Learned latent tokens
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                PerceiverBlock(dim=embed_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def forward_features(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patches via Perceiver cross-attention with learned latents.

        Parameters
        ----------
        h : torch.Tensor
            [B, N, D] patch embeddings.
        mask : torch.Tensor, optional
            [B, N] binary mask (1=valid, 0=padding).
        coords : unused.
        return_attention : bool
            If True, return cross-attention weights averaged over latents.

        Returns
        -------
        slide_embedding : torch.Tensor [B, E]
        extras : dict with optional 'attention_weights' [B, N]
        """
        h = self.patch_embed(h)  # [B, N, E]
        B = h.shape[0]

        # Expand latent tokens for the batch
        latents = self.latents.expand(B, -1, -1)  # [B, M, E]

        # Build key_padding_mask for cross-attention (True = ignored)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (1 - mask).bool()  # [B, N]

        # Perceiver blocks
        all_cross_weights = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # checkpoint doesn't support kwargs, pack into positional args
                latents, cross_weights = cp.checkpoint(
                    block, latents, h, key_padding_mask, use_reentrant=False
                )
            else:
                latents, cross_weights = block(latents, h, key_padding_mask=key_padding_mask)
            if return_attention:
                all_cross_weights.append(cross_weights)

        latents = self.norm(latents)  # [B, M, E]

        # Mean pool over latent tokens
        slide_embedding = latents.mean(dim=1)  # [B, E]

        extras = {}
        if return_attention and all_cross_weights:
            # Average cross-attention weights across layers and latents → [B, N]
            # Each cross_weights: [B, M, N]
            stacked = torch.stack(all_cross_weights, dim=0)  # [L, B, M, N]
            avg_weights = stacked.mean(dim=(0, 2))  # [B, N]
            extras["attention_weights"] = avg_weights

        return slide_embedding, extras
