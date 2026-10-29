"""
TITAN-style ViT with 2D-ALiBi for slide-level aggregation.

Ding et al., Nature Medicine 2025.
Matches the official release at huggingface.co/MahmoodLab/TITAN.

    [B, N, D_in] -> Linear+GELU -> prepend CLS -> LxBlock(ALiBi) -> LN -> CLS -> [B, E]
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.layers import DropPath, Mlp, trunc_normal_

from oceanpath.models.base import BaseMIL

_MASK_VALUE: float = -65504.0  # fp16-safe sentinel


# ═════════════════════════════════════════════════════════════════════════════
# 2D ALiBi
# ═════════════════════════════════════════════════════════════════════════════


def _get_alibi_slopes(n: int) -> list[float]:
    """ALiBi slope schedule (Press et al., ICLR 2022)."""
    if math.log2(n).is_integer():
        p = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [p * (p**i) for i in range(n)]
    k = 2 ** math.floor(math.log2(n))
    base = _get_alibi_slopes(k)
    extra = _get_alibi_slopes(2 * k)[0::2][: n - k]
    return base + extra


def compute_2d_alibi_bias(
    coords: torch.Tensor,
    slopes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D ALiBi bias from grid-unit coordinates.

    coords: [B, N, 2]   slopes: [H]   mask: [B, N] (1=real, 0=pad)
    Returns [B, H, 1+N, 1+N] with CLS at position 0.
    """
    B, N, _ = coords.shape

    dist = torch.cdist(coords.float(), coords.float(), p=2)  # [B, N, N]
    patch_bias = dist[:, None] * (-slopes[None, :, None, None])  # [B, H, N, N]
    full_bias = F.pad(patch_bias, (1, 0, 1, 0), value=0.0)  # [B, H, 1+N, 1+N]

    if mask is not None and not mask.all():
        L = 1 + N
        pad = mask.new_ones(B, L)
        pad[:, 1:] = mask
        attn_mask = pad.unsqueeze(2) * pad.unsqueeze(1)
        attn_mask.diagonal(dim1=-2, dim2=-1).clamp_(min=1.0)
        full_bias = full_bias + (1.0 - attn_mask).unsqueeze(1) * _MASK_VALUE

    return full_bias


# ═════════════════════════════════════════════════════════════════════════════
# Attention
# ═════════════════════════════════════════════════════════════════════════════


class ALiBiAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop

    def forward(self, x, alibi_bias=None, return_attention=False):
        B, N, C = x.shape
        H, d = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_attention:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                qf, kf = q.float(), k.float()
                attn = (qf @ kf.transpose(-2, -1)) * (d**-0.5)
                if alibi_bias is not None:
                    attn = attn + alibi_bias.float()
                attn_w = F.softmax(attn, dim=-1)
            x = (attn_w.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))
            return x, attn_w

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=alibi_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x, None


# ═════════════════════════════════════════════════════════════════════════════
# LayerScale
# ═════════════════════════════════════════════════════════════════════════════


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


# ═════════════════════════════════════════════════════════════════════════════
# Transformer Block
# ═════════════════════════════════════════════════════════════════════════════


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ALiBiAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, alibi_bias=None, return_attention=False):
        attn_out, attn_w = self.attn(
            self.norm1(x), alibi_bias=alibi_bias, return_attention=return_attention
        )
        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, attn_w


# ═════════════════════════════════════════════════════════════════════════════
# TITANViT
# ═════════════════════════════════════════════════════════════════════════════


class TITANViT(BaseMIL):
    """TITAN ViT with 2D-ALiBi.

    Parameters
    ----------
    in_dim         : patch feature dim (1536 UNI2-H, 768 CONCH v1.5)
    embed_dim      : transformer dim
    depth          : number of blocks (4→29M, 6→43M)
    num_heads      : attention heads
    mlp_ratio      : MLP expansion
    qkv_bias       : QKV bias
    qk_norm        : per-head Q/K LayerNorm
    init_values    : LayerScale init (None = off)
    drop_rate      : projection dropout
    attn_drop_rate : attention dropout
    drop_path_rate : stochastic depth max
    coord_scale    : divide coords by this to get grid units (None = already grid)
    gradient_checkpointing : checkpoint blocks
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        coord_scale: float | None = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__(
            in_dim=in_dim, embed_dim=embed_dim, gradient_checkpointing=gradient_checkpointing
        )

        self.coord_scale = coord_scale
        self.depth = depth
        self.num_heads = num_heads

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.GELU())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        slopes = torch.tensor(_get_alibi_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward_features(self, h, mask=None, coords=None, return_attention=False):
        B, N, _ = h.shape

        # Coords → grid units
        if coords is not None and self.coord_scale is not None:
            coords = coords.float() / self.coord_scale

        # Patch embed + CLS
        h = self.patch_embed(h)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = self.pos_drop(h)

        # ALiBi bias (once, shared across layers)
        alibi_bias = None
        if coords is not None:
            alibi_bias = compute_2d_alibi_bias(coords, self.alibi_slopes, mask).to(h.dtype)
        elif mask is not None and not mask.all():
            L = 1 + N
            pad = mask.new_ones(B, L)
            pad[:, 1:] = mask
            attn_mask = pad.unsqueeze(2) * pad.unsqueeze(1)
            attn_mask.diagonal(dim1=-2, dim2=-1).clamp_(min=1.0)
            alibi_bias = ((1.0 - attn_mask) * _MASK_VALUE).unsqueeze(1).to(h.dtype)

        # Transformer
        extras = {}
        for i, blk in enumerate(self.blocks):
            want_attn = return_attention and (i == self.depth - 1)
            if self.gradient_checkpointing and self.training and not want_attn:
                h, _ = cp.checkpoint(_block_forward, blk, h, alibi_bias, False, use_reentrant=False)
            else:
                h, attn_w = blk(h, alibi_bias=alibi_bias, return_attention=want_attn)
                if want_attn and attn_w is not None:
                    extras["attention_weights"] = attn_w[:, :, 0, 1:].mean(dim=1)

        return self.norm(h)[:, 0], extras


def _block_forward(blk, x, alibi_bias, return_attention):
    return blk(x, alibi_bias=alibi_bias, return_attention=return_attention)
