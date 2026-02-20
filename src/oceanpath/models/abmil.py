"""
Attention-Based Multiple Instance Learning (ABMIL).

Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.

Architecture:
    patches [B, N, D] → MLP → [B, N, E] → Gated Attention → weighted sum → [B, E]

Supports:
    - Standard or gated attention
    - Gradient checkpointing for large bags
    - Float32 attention softmax under AMP
    - Variable bag sizes with attention masking
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import GlobalAttention, GlobalGatedAttention, create_mlp


class ABMIL(BaseMIL):
    """
    ABMIL aggregator.

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension.
    embed_dim : int
        Embedding dimension after patch MLP.
    num_fc_layers : int
        Number of FC layers in the patch embedding MLP.
    attn_dim : int
        Hidden dimension of the attention network.
    gate : bool
        Use gated attention (Tanh x Sigmoid) vs. standard (Tanh only).
    dropout : float
        Dropout rate in MLP and attention.
    gradient_checkpointing : bool
        Checkpoint attention computation to save memory.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        attn_dim: int = 384,
        gate: bool = True,
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
            end_with_fc=False,  # ends with activation
        )

        attn_cls = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_cls(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1,
        )

        self.initialize_weights()

    def forward_features(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patch features into slide embedding via attention pooling.

        Parameters
        ----------
        h : torch.Tensor
            [B, N, D] patch embeddings.
        mask : torch.Tensor, optional
            [B, N] binary mask (1=valid, 0=padding).
        coords : unused (kept for interface compatibility).
        return_attention : bool
            Include attention weights in output.

        Returns
        -------
        slide_embedding : torch.Tensor [B, E]
        extras : dict with optional 'attention_weights' [B, N]
        """
        # Patch embedding MLP
        h = self.patch_embed(h)  # [B, N, E]

        # Attention logits (with optional gradient checkpointing)
        if self.gradient_checkpointing and self.training:
            attn_logits = cp.checkpoint(self.global_attn, h, use_reentrant=False)  # [B, N, 1]
        else:
            attn_logits = self.global_attn(h)  # [B, N, 1]

        attn_logits = attn_logits.transpose(-2, -1)  # [B, 1, N]

        # Apply mask before softmax
        if mask is not None:
            # mask: [B, N] → [B, 1, N]
            attn_logits = attn_logits.masked_fill((1 - mask).unsqueeze(1).bool(), float("-inf"))

        # Float32 softmax for numerical stability under AMP
        with torch.amp.autocast(device_type="cuda", enabled=False):
            attn_weights = F.softmax(attn_logits.float(), dim=-1)  # [B, 1, N]

        # Weighted sum
        slide_embedding = torch.bmm(attn_weights, h).squeeze(1)  # [B, E]

        extras = {}
        if return_attention:
            extras["attention_weights"] = attn_logits.squeeze(1)  # [B, N] (pre-softmax)

        return slide_embedding, extras
