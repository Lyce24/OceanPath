"""
Perceiver MIL Aggregator — PRISM2-style.

Architecture (single-block variant, following PRISM2):
    patches [B, N, D] → MLP → [B, N, E]
    → (optional) add 2D Fourier positional encoding from coords
    → Cross-Attention(latents ← patches)            # 1 cross-attn, GQA-capable
    → Self-Attention(latents) x num_sa_layers         # deep self-attn stack
    → AttentionPool(latents) → [B, E]               # learned attention pooling

Key design choices derived from PRISM / PRISM2 / Perceiver:
    - PRISM2: 1 block, 6 self-attn layers, 256 latents, GQA cross-attn, attn pool
    - Original Perceiver: weight sharing across repeated blocks, 1 cross-attn head
    - PathAlign/BLIP-2: small number of learned queries (32-256) as bottleneck

Supports:
    - Configurable block count with deep self-attention per block
    - Weight sharing across blocks (critical regularizer for small datasets)
    - Grouped Query Attention (GQA) in cross-attention
    - Learned attention pooling (replaces naive mean pool)
    - 2D Fourier positional encoding from spatial coordinates
    - Separate head counts for cross-attention vs self-attention
    - Gradient checkpointing per block
    - Float32 attention softmax under AMP
    - Variable bag sizes with attention masking

No external dependencies — pure PyTorch multihead attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import create_mlp

# ---------------------------------------------------------------------------
# Fourier Positional Encoding for 2-D coordinates
# ---------------------------------------------------------------------------


class FourierPositionalEncoding2D(nn.Module):
    """
    Generate Fourier features from (x, y) spatial coordinates.

    Produces sin/cos encodings at multiple frequency bands, concatenated
    into a vector of size ``4 * num_bands`` per coordinate pair.  A linear
    projection maps this to ``embed_dim``.

    Parameters
    ----------
    embed_dim : int
        Target embedding dimension (output of the linear projection).
    num_bands : int
        Number of frequency bands.  Total raw Fourier dim = 4 * num_bands
        (sin + cos for each of x and y at each band).
    max_resolution : float
        Maximum spatial extent used to normalise coordinates.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_bands: int = 64,
        max_resolution: float = 224.0,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution

        # Frequency bands: log-spaced from 1 to max_resolution / 2
        freqs = torch.logspace(
            0.0,
            math.log2(max_resolution / 2),
            steps=num_bands,
            base=2.0,
        )
        self.register_buffer("freqs", freqs)  # [num_bands]

        raw_dim = 4 * num_bands  # sin_x, cos_x, sin_y, cos_y per band
        self.proj = nn.Linear(raw_dim, embed_dim, bias=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : [B, N, 2]  —  (x, y) coordinates for each patch.

        Returns
        -------
        pos_enc : [B, N, embed_dim]
        """
        # Normalise to [0, 1]
        coords = coords.float() / self.max_resolution  # [B, N, 2]
        x, y = coords[..., 0:1], coords[..., 1:2]  # [B, N, 1]

        freqs = self.freqs[None, None, :]  # [1, 1, num_bands]
        # [B, N, num_bands] for each
        x_enc = torch.cat([torch.sin(x * freqs * math.pi), torch.cos(x * freqs * math.pi)], dim=-1)
        y_enc = torch.cat([torch.sin(y * freqs * math.pi), torch.cos(y * freqs * math.pi)], dim=-1)

        enc = torch.cat([x_enc, y_enc], dim=-1)  # [B, N, 4*num_bands]
        return self.proj(enc)


# ---------------------------------------------------------------------------
# Grouped Query Attention (for cross-attention efficiency)
# ---------------------------------------------------------------------------


class GroupedQueryCrossAttention(nn.Module):
    """
    Cross-attention with Grouped Query Attention (GQA).

    Queries come from latents, keys/values come from patches.
    GQA uses fewer KV heads than Q heads to save memory on the
    large-N patch dimension.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_q_heads : int
        Number of query heads (for latents).
    num_kv_heads : int
        Number of key/value heads (for patches).  Must divide num_q_heads.
    dropout : float
        Attention dropout.
    """

    def __init__(
        self,
        dim: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads
        self.scale = self.head_dim**-0.5
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        query : [B, M, D]   — latents
        key_value : [B, N, D] — patches
        key_padding_mask : [B, N] bool, True = ignored
        return_weights : bool

        Returns
        -------
        out : [B, M, D]
        attn_weights : [B, M, N] or None
        """
        B, M, _ = query.shape
        N = key_value.shape[1]

        # Project
        q = self.q_proj(query).view(B, M, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads for GQA: [B, num_kv_heads, ...] -> [B, num_q_heads, ...]
        if self.kv_group_size > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.kv_group_size, -1, -1)
            k = k.reshape(B, self.num_q_heads, N, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.kv_group_size, -1, -1)
            v = v.reshape(B, self.num_q_heads, N, self.head_dim)

        # Compute attention in float32 for numerical stability under AMP
        with torch.amp.autocast(device_type="cuda", enabled=False):
            q_f, k_f, v_f = q.float(), k.float(), v.float()
            attn = torch.matmul(q_f, k_f.transpose(-2, -1)) * self.scale  # [B, H, M, N]

            if key_padding_mask is not None:
                # key_padding_mask: [B, N], True = ignored
                attn = attn.masked_fill(
                    key_padding_mask[:, None, None, :],  # [B, 1, 1, N]
                    float("-inf"),
                )

            attn_weights = F.softmax(attn, dim=-1)
            if self.training and self.dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)

            out = torch.matmul(attn_weights, v_f)  # [B, H, M, head_dim]

        out = out.to(query.dtype)
        out = out.transpose(1, 2).reshape(B, M, self.dim)
        out = self.out_proj(out)

        weights_out = None
        if return_weights:
            # Average over heads → [B, M, N]
            weights_out = attn_weights.mean(dim=1).to(query.dtype)

        return out, weights_out


# ---------------------------------------------------------------------------
# Learned Attention Pooling (replaces naive mean pool)
# ---------------------------------------------------------------------------


class AttentionPool(nn.Module):
    """
    Learned attention pooling over latent tokens.

    A single learnable query cross-attends to the latent array and produces
    a single vector.  Optionally uses multiple heads.

    Parameters
    ----------
    dim : int
        Latent embedding dimension.
    num_heads : int
        Number of attention heads for pooling.
    """

    def __init__(self, dim: int = 512, num_heads: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latents : [B, M, D]

        Returns
        -------
        pooled : [B, D]
        """
        B = latents.shape[0]
        q = self.norm_q(self.query.expand(B, -1, -1))
        kv = self.norm_kv(latents)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            out, _ = self.attn(q.float(), kv.float(), kv.float(), need_weights=False)
        return out.to(latents.dtype).squeeze(1)  # [B, D]


# ---------------------------------------------------------------------------
# Self-Attention Layer (for the deep latent transformer stack)
# ---------------------------------------------------------------------------


class SelfAttentionLayer(nn.Module):
    """
    Standard pre-norm self-attention + FFN block for latent tokens.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    ffn_mult : int
        FFN hidden dim multiplier.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Perceiver Block: 1 cross-attn + N self-attn layers
# ---------------------------------------------------------------------------


class PerceiverBlock(nn.Module):
    """
    Single Perceiver block: cross-attention (latents ← patches) followed by
    a deep stack of self-attention layers over the latents.

    Matches PRISM2 design: 1 cross-attn + ``num_sa_layers`` self-attn layers
    per block.

    Parameters
    ----------
    dim : int
        Latent/patch embedding dimension.
    num_sa_layers : int
        Number of self-attention layers *within this block*.
    num_cross_heads : int
        Number of query heads in cross-attention.
    num_kv_heads : int
        Number of KV heads in cross-attention (GQA).
    num_sa_heads : int
        Number of heads in each self-attention layer.
    dropout : float
        Dropout rate.
    ffn_mult : int
        FFN hidden dim multiplier.
    share_sa_weights : bool
        If True, all self-attention layers share a single set of weights
        (strong regularizer, reduces parameters).
    """

    def __init__(
        self,
        dim: int = 512,
        num_sa_layers: int = 6,
        num_cross_heads: int = 1,
        num_kv_heads: int = 1,
        num_sa_heads: int = 8,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        share_sa_weights: bool = False,
    ):
        super().__init__()
        self.num_sa_layers = num_sa_layers
        self.share_sa_weights = share_sa_weights

        # --- Cross-attention: latents ← patches (GQA) ---
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = GroupedQueryCrossAttention(
            dim=dim,
            num_q_heads=num_cross_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        # Post-cross-attn FFN
        self.cross_ffn_norm = nn.LayerNorm(dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

        # --- Deep self-attention stack over latents ---
        if share_sa_weights:
            # Single layer, applied num_sa_layers times
            self._shared_sa = SelfAttentionLayer(
                dim=dim,
                num_heads=num_sa_heads,
                dropout=dropout,
                ffn_mult=ffn_mult,
            )
        else:
            self.sa_layers = nn.ModuleList(
                [
                    SelfAttentionLayer(
                        dim=dim,
                        num_heads=num_sa_heads,
                        dropout=dropout,
                        ffn_mult=ffn_mult,
                    )
                    for _ in range(num_sa_layers)
                ]
            )

    def _get_sa_layer(self, idx: int) -> SelfAttentionLayer:
        if self.share_sa_weights:
            return self._shared_sa
        return self.sa_layers[idx]

    def forward(
        self,
        latents: torch.Tensor,
        patches: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        latents : [B, M, D]
        patches : [B, N, D]
        key_padding_mask : [B, N] bool, True = ignored position
        return_attention : bool

        Returns
        -------
        latents : [B, M, D]
        cross_attn_weights : [B, M, N] or None
        """
        # --- Cross-attention ---
        q = self.cross_norm_q(latents)
        kv = self.cross_norm_kv(patches)
        cross_out, cross_weights = self.cross_attn(
            q,
            kv,
            key_padding_mask=key_padding_mask,
            return_weights=return_attention,
        )
        latents = latents + cross_out.to(latents.dtype)
        latents = latents + self.cross_ffn(self.cross_ffn_norm(latents))

        # --- Deep self-attention stack ---
        for i in range(self.num_sa_layers):
            latents = self._get_sa_layer(i)(latents)

        return latents, cross_weights


# ---------------------------------------------------------------------------
# Main Model: PerceiverMIL
# ---------------------------------------------------------------------------


class PerceiverMIL(BaseMIL):
    """
    PRISM2-style Perceiver MIL aggregator.

    Default configuration follows PRISM2 findings:
      - 1 block with 6 self-attention layers (deep processing)
      - GQA in cross-attention (memory-efficient)
      - Learned attention pooling (not mean pool)
      - Optional weight sharing (regulariser for small datasets)
      - Optional 2D Fourier positional encoding from coordinates

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension (e.g. 1024 for UNI, 2560 for Virchow).
    embed_dim : int
        Embedding dimension after patch projection.
    num_fc_layers : int
        FC layers in patch embedding MLP.
    num_latents : int
        Number of learned latent tokens (M << N).
    num_blocks : int
        Number of cross-attn + self-attn blocks (PRISM2 uses 1).
    num_sa_layers : int
        Self-attention layers *per block* (PRISM2 uses 6).
    num_cross_heads : int
        Query heads in cross-attention (original Perceiver uses 1).
    num_kv_heads : int
        KV heads in cross-attention (GQA; 1 = maximum sharing).
    num_sa_heads : int
        Heads in each self-attention layer.
    dropout : float
        Dropout rate.
    ffn_mult : int
        FFN hidden dimension multiplier.
    share_sa_weights : bool
        Share self-attention weights within each block.
    share_block_weights : bool
        Share weights across blocks (all blocks after the first share
        weights with the first block — original Perceiver style).
    use_attn_pool : bool
        Use learned attention pooling (True) or mean pooling (False).
    use_pos_encoding : bool
        Add Fourier positional encoding from spatial coordinates.
    pos_num_bands : int
        Number of Fourier frequency bands for positional encoding.
    pos_max_resolution : float
        Max spatial resolution for positional encoding normalisation.
    gradient_checkpointing : bool
        Checkpoint each block to save VRAM.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        num_latents: int = 256,
        num_blocks: int = 1,
        num_sa_layers: int = 6,
        num_cross_heads: int = 1,
        num_kv_heads: int = 1,
        num_sa_heads: int = 8,
        dropout: float = 0.15,
        ffn_mult: int = 4,
        share_sa_weights: bool = False,
        share_block_weights: bool = False,
        use_attn_pool: bool = True,
        use_pos_encoding: bool = False,
        pos_num_bands: int = 64,
        pos_max_resolution: float = 224.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            embed_dim=embed_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.num_latents = num_latents
        self.num_blocks = num_blocks
        self.share_block_weights = share_block_weights
        self.use_attn_pool = use_attn_pool
        self.use_pos_encoding = use_pos_encoding

        # --- Patch embedding MLP ---
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False,
        )

        # --- Optional positional encoding ---
        self.pos_enc: FourierPositionalEncoding2D | None = None
        if use_pos_encoding:
            self.pos_enc = FourierPositionalEncoding2D(
                embed_dim=embed_dim,
                num_bands=pos_num_bands,
                max_resolution=pos_max_resolution,
            )

        # --- Learned latent tokens ---
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)

        # --- Perceiver blocks ---
        if share_block_weights and num_blocks > 1:
            # First block has its own weights; all subsequent share weights
            self.block_first = PerceiverBlock(
                dim=embed_dim,
                num_sa_layers=num_sa_layers,
                num_cross_heads=num_cross_heads,
                num_kv_heads=num_kv_heads,
                num_sa_heads=num_sa_heads,
                dropout=dropout,
                ffn_mult=ffn_mult,
                share_sa_weights=share_sa_weights,
            )
            self.block_shared = PerceiverBlock(
                dim=embed_dim,
                num_sa_layers=num_sa_layers,
                num_cross_heads=num_cross_heads,
                num_kv_heads=num_kv_heads,
                num_sa_heads=num_sa_heads,
                dropout=dropout,
                ffn_mult=ffn_mult,
                share_sa_weights=share_sa_weights,
            )
            self.blocks = None  # use block_first + block_shared instead
        else:
            self.blocks = nn.ModuleList(
                [
                    PerceiverBlock(
                        dim=embed_dim,
                        num_sa_layers=num_sa_layers,
                        num_cross_heads=num_cross_heads,
                        num_kv_heads=num_kv_heads,
                        num_sa_heads=num_sa_heads,
                        dropout=dropout,
                        ffn_mult=ffn_mult,
                        share_sa_weights=share_sa_weights,
                    )
                    for _ in range(num_blocks)
                ]
            )

        # --- Output normalization ---
        self.norm = nn.LayerNorm(embed_dim)

        # --- Pooling ---
        if use_attn_pool:
            self.attn_pool = AttentionPool(dim=embed_dim, num_heads=1)
        else:
            self.attn_pool = None

        self.initialize_weights()

    def _iter_blocks(self):
        """Yield (index, block) pairs respecting weight sharing."""
        if self.blocks is not None:
            yield from enumerate(self.blocks)
        else:
            yield 0, self.block_first
            for i in range(1, self.num_blocks):
                yield i, self.block_shared

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
            [B, N] binary mask (1 = valid, 0 = padding).
        coords : torch.Tensor, optional
            [B, N, 2] spatial (x, y) coordinates for each patch.
        return_attention : bool
            If True, return cross-attention weights averaged over latents.

        Returns
        -------
        slide_embedding : torch.Tensor [B, E]
        extras : dict with optional ``attention_weights`` [B, N]
            and ``latents`` [B, M, E] (useful for LLM adapters / JEPA targets).
        """
        h = self.patch_embed(h)  # [B, N, E]
        B = h.shape[0]

        # Add positional encoding if available
        if self.pos_enc is not None and coords is not None:
            h = h + self.pos_enc(coords)

        # Expand latent tokens for the batch
        latents = self.latents.expand(B, -1, -1)  # [B, M, E]

        # Build key_padding_mask for cross-attention (True = ignored)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # [B, N]

        # --- Perceiver blocks ---
        all_cross_weights = []
        for _idx, block in self._iter_blocks():
            if self.gradient_checkpointing and self.training:
                latents, cross_weights = cp.checkpoint(
                    block,
                    latents,
                    h,
                    key_padding_mask,
                    return_attention,
                    use_reentrant=False,
                )
            else:
                latents, cross_weights = block(
                    latents,
                    h,
                    key_padding_mask=key_padding_mask,
                    return_attention=return_attention,
                )
            if return_attention and cross_weights is not None:
                all_cross_weights.append(cross_weights)

        latents = self.norm(latents)  # [B, M, E]

        # --- Pooling ---
        if self.use_attn_pool:
            slide_embedding = self.attn_pool(latents)  # [B, E]
        else:
            # Fallback mean pool
            slide_embedding = latents.mean(dim=1)  # [B, E]

        # --- Build extras ---
        extras: dict = {}
        # Always return latents — needed for LLM adapter (PRISM2-style)
        # and JEPA targets
        extras["latents"] = latents

        if return_attention and all_cross_weights:
            # Average cross-attention weights across blocks and latents → [B, N]
            stacked = torch.stack(all_cross_weights, dim=0)  # [num_blocks, B, M, N]
            avg_weights = stacked.mean(dim=(0, 2))  # [B, N]
            extras["attention_weights"] = avg_weights

        return slide_embedding, extras
