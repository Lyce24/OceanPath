"""
Bidirectional Mamba-2 MIL Aggregator (BiMamba-2).

Architecture:
    patches [B, N, D] → MLP → [B, N, E] → 2x BiMamba2 layers → masked mean pool → [B, E]

Each BiMamba2Layer runs Mamba2 in forward and reverse directions, sums the
outputs, and applies a residual connection with LayerNorm.

Requires: pip install mamba-ssm  (Dao lab Mamba-2 implementation)

Supports:
    - Bidirectional sequence modeling of patch tokens
    - Gradient checkpointing per layer
    - Float32 attention proxy under AMP
    - Variable bag sizes with masked mean pooling
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import create_mlp


class BiMamba2Layer(nn.Module):
    """
    Single bidirectional Mamba-2 layer.

    Runs Mamba2 on forward and reversed sequences, sums outputs,
    and applies residual + LayerNorm.

    Parameters
    ----------
    dim : int
        Model dimension.
    d_state : int
        SSM state dimension.
    d_conv : int
        Local convolution width.
    expand : int
        Expansion factor for inner dimension.
    dropout : float
        Dropout rate after the bidirectional merge.
    """

    def __init__(
        self,
        dim: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        from mamba_ssm import Mamba2

        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_rev = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D] → [B, N, D] with residual.
        """
        residual = x
        h = self.norm(x)

        # Forward direction
        h_fwd = self.mamba_fwd(h)

        # Reverse direction
        h_rev = self.mamba_rev(h.flip(dims=[1])).flip(dims=[1])

        # Merge and residual
        out = self.dropout(h_fwd + h_rev) + residual
        return out


class BiMamba2MIL(BaseMIL):
    """
    BiMamba-2 MIL aggregator.

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension.
    embed_dim : int
        Embedding dimension after patch MLP.
    num_fc_layers : int
        FC layers in patch embedding MLP.
    num_layers : int
        Number of stacked BiMamba2 layers.
    d_state : int
        SSM state dimension.
    d_conv : int
        Local convolution width.
    expand : int
        Expansion factor for inner dimension.
    dropout : float
        Dropout rate.
    gradient_checkpointing : bool
        Checkpoint each BiMamba2 layer.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        num_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
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

        self.layers = nn.ModuleList(
            [
                BiMamba2Layer(
                    dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
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
        Aggregate patch features via bidirectional Mamba-2 layers + masked mean pool.

        Parameters
        ----------
        h : torch.Tensor
            [B, N, D] patch embeddings.
        mask : torch.Tensor, optional
            [B, N] binary mask (1=valid, 0=padding).
        coords : unused.
        return_attention : bool
            If True, compute dot-product proxy attention scores.

        Returns
        -------
        slide_embedding : torch.Tensor [B, E]
        extras : dict with optional 'attention_weights' [B, N]
        """
        h = self.patch_embed(h)  # [B, N, E]
        B, N, E = h.shape

        # Zero out padding before SSM layers so the recurrence doesn't
        # propagate information from padded positions.
        if mask is not None:
            h = h * mask.unsqueeze(-1)

        # Mamba2 CUDA kernels require strides to be multiples of 8.
        # Pad short sequences and track original length for trimming.
        _MIN_SEQ = 8
        pad_len = 0
        if N < _MIN_SEQ:
            pad_len = _MIN_SEQ - N
            h = torch.cat([h, h.new_zeros(B, pad_len, E)], dim=1)

        # BiMamba-2 layers with optional checkpointing
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = cp.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)

        # Trim padding back to original length
        if pad_len > 0:
            h = h[:, :N, :]

        h = self.norm(h)  # [B, N, E]

        # Masked mean pooling
        if mask is not None:
            h_masked = h * mask.unsqueeze(-1)  # zero out padding
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            slide_embedding = h_masked.sum(dim=1) / valid_counts  # [B, E]
        else:
            slide_embedding = h.mean(dim=1)  # [B, E]

        extras = {}
        if return_attention:
            # Proxy attention: dot-product of each token with the pooled embedding
            with torch.amp.autocast(device_type="cuda", enabled=False):
                attn = torch.bmm(h.float(), slide_embedding.float().unsqueeze(-1)).squeeze(
                    -1
                )  # [B, N]
            extras["attention_weights"] = attn

        return slide_embedding, extras
