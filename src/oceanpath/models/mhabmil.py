"""
Multihead Attention-Based Multiple Instance Learning (Multihead ABMIL).

Architecture:
    patches [B, N, D] → MLP → [B, N, E]
    → K independent gated attention heads → K weighted sums
    → concat → Linear → [B, E]

Uses K parallel gated attention branches (reusing GlobalGatedAttention /
GlobalAttention from components.py), each producing a [B, E] embedding.
These are concatenated to [B, K*E] and projected back to [B, E].

Supports:
    - Configurable number of independent attention heads
    - Gated or standard attention per head
    - Gradient checkpointing
    - Float32 attention softmax under AMP
    - Variable bag sizes with attention masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import GlobalAttention, GlobalGatedAttention, create_mlp


class MultiheadABMIL(BaseMIL):
    """
    Multihead ABMIL aggregator.

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension.
    embed_dim : int
        Embedding dimension after patch MLP.
    num_fc_layers : int
        FC layers in patch embedding MLP.
    num_heads : int
        Number of independent attention heads (K).
    attn_dim : int
        Hidden dimension of each attention network.
    gate : bool
        Use gated attention (Tanh x Sigmoid) vs. standard (Tanh only).
    dropout : float
        Dropout rate.
    gradient_checkpointing : bool
        Checkpoint the attention computation.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        num_heads: int = 4,
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

        self.num_heads = num_heads

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False,
        )

        # K independent attention branches
        attn_cls = GlobalGatedAttention if gate else GlobalAttention
        self.attention_heads = nn.ModuleList(
            [
                attn_cls(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)
                for _ in range(num_heads)
            ]
        )

        # Projection from concatenated head outputs back to embed_dim
        self.head_proj = nn.Linear(num_heads * embed_dim, embed_dim)

        self.initialize_weights()

    def _compute_attention(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention logits for all heads.

        Parameters
        ----------
        h : [B, N, E]

        Returns
        -------
        attn_logits : [B, K, N] pre-softmax attention logits
        """
        logits_list = []
        for head in self.attention_heads:
            logits = head(h)  # [B, N, 1]
            logits_list.append(logits.squeeze(-1))  # [B, N]
        return torch.stack(logits_list, dim=1)  # [B, K, N]

    def forward_features(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patches via K independent attention heads.

        Parameters
        ----------
        h : torch.Tensor
            [B, N, D] patch embeddings.
        mask : torch.Tensor, optional
            [B, N] binary mask (1=valid, 0=padding).
        coords : unused.
        return_attention : bool
            If True, return per-head attention logits.

        Returns
        -------
        slide_embedding : torch.Tensor [B, E]
        extras : dict with optional 'attention_weights' [B, N] (averaged over heads)
                 and 'attention_weights_per_head' [B, K, N]
        """
        h = self.patch_embed(h)  # [B, N, E]

        # Compute attention logits for all heads (with optional checkpointing)
        if self.gradient_checkpointing and self.training:
            attn_logits = cp.checkpoint(
                self._compute_attention, h, use_reentrant=False
            )  # [B, K, N]
        else:
            attn_logits = self._compute_attention(h)  # [B, K, N]

        # Apply mask before softmax
        if mask is not None:
            # mask: [B, N] → [B, 1, N]
            attn_logits = attn_logits.masked_fill((1 - mask).unsqueeze(1).bool(), float("-inf"))

        # Float32 softmax for numerical stability under AMP
        with torch.amp.autocast(device_type="cuda", enabled=False):
            attn_weights = F.softmax(attn_logits.float(), dim=-1)  # [B, K, N]

        # Weighted sum per head: [B, K, N] @ [B, N, E] → [B, K, E]
        head_embeddings = torch.bmm(attn_weights, h)  # [B, K, E]

        # Concat and project: [B, K*E] → [B, E]
        B = h.shape[0]
        concat = head_embeddings.reshape(B, -1)  # [B, K*E]
        slide_embedding = self.head_proj(concat)  # [B, E]

        extras = {}
        if return_attention:
            extras["attention_weights"] = attn_logits.mean(dim=1)  # [B, N] averaged
            extras["attention_weights_per_head"] = attn_logits  # [B, K, N]

        return slide_embedding, extras
