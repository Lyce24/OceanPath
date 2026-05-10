"""
Mamba-2 MIL Aggregator (Prototype).

Minimal Mamba-2 slide encoder for SSL pretraining.
Architecture follows the COBRA/MambaMIL skeleton:

    patches [B, N, D] → Embed MLP → Mamba-2 SSD layers → Gated Attention Pool → [B, E]

No coordinate awareness, no bidirectional scanning — those come later.

Requires: pip install mamba-ssm causal-conv1d
    - mamba-ssm provides Mamba2 block (SSD)
    - causal-conv1d provides the efficient conv1d used inside Mamba blocks
    - Both require CUDA and nvcc.

Design decisions vs. COBRA & MambaMIL:
    - Single FM (UNIv2 1536-dim), so one embedding head (not COBRA's per-FM MLPs)
    - Mamba-2 SSD (not MambaMIL's Mamba-1), following COBRA
    - Residual after each SSD layer (COBRA Eq. 3), not SR-Mamba reordering
    - Single-head gated attention (your existing ABMIL), not COBRA's multi-head
    - COBRA's inference trick (pool raw features using Mamba-derived weights)
      is implemented as an option for downstream use, but disabled during SSL

Note on padding/masking:
    Mamba processes the full sequence including padding tokens through its
    causal conv1d and SSM recurrence. Padding is only masked at the
    attention pooling stage. This is consistent with COBRA and MambaMIL,
    neither of which mask within the SSM layers. For heavily padded batches,
    consider sorting by length and using smaller sub-batches instead.

VRAM budget at B=128, N=4800, fp16:
    embed_dim=384, expand=2: ~13 GiB  (safe on 24GB)
    embed_dim=512, expand=1: ~12 GiB  (safe on 24GB)
    embed_dim=512, expand=2: ~16 GiB  (tight but OK with grad ckpt)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import GlobalGatedAttention

# ── Mamba-2 import with graceful fallback ─────────────────────────────────────

try:
    from mamba_ssm.modules.mamba2 import Mamba2

    MAMBA2_AVAILABLE = True
except ImportError:
    MAMBA2_AVAILABLE = False


# ── Mamba-2 layer wrapper ─────────────────────────────────────────────────────


class Mamba2Layer(nn.Module):
    """
    Single Mamba-2 SSD block with residual connection.

    Follows COBRA's pattern (Eq. 3): residual after each SSD call.

        x → Mamba2 (has internal RMSNorm) → + x (residual)

    The Mamba2 module already contains:
        - Input projection (in_proj): d_model → 2*d_inner + ngroups*d_state*2 + nheads
        - Causal conv1d on the SSM input path
        - SSD (structured state space dual) computation
        - Internal RMSNorm (when rmsnorm=True, the default)
        - Gated output with multiplicative interaction
        - Output projection (out_proj): d_inner → d_model

    We do NOT add external LayerNorm to avoid double-normalization
    (Mamba2's internal RMSNorm already normalizes before gating).

    Parameters
    ----------
    d_model : int
        Model dimension (must match input/output).
    d_state : int
        SSM state dimension. Higher = more memory capacity but slower.
        Mamba2 class default is 128. COBRA uses 64 for efficiency.
    d_conv : int
        Causal conv1d kernel size inside the Mamba block.
    expand : int
        Expansion factor for inner dimension. d_inner = expand * d_model.
        expand=2 is standard. Use expand=1 to halve activation memory.
    headdim : int
        Head dimension for the SSD multi-head structure.
        Must divide (expand * d_model) evenly. Default 64.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()

        if not MAMBA2_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for Mamba2MIL. "
                "Install with: pip install mamba-ssm causal-conv1d --no-build-isolation"
            )

        # Validate headdim divides d_inner
        d_inner = expand * d_model
        if d_inner % headdim != 0:
            raise ValueError(
                f"headdim={headdim} must divide expand*d_model={d_inner}. "
                f"Got d_model={d_model}, expand={expand}."
            )

        # No external norm — Mamba2 has internal RMSNorm (rmsnorm=True default)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] → [B, L, D] with residual."""
        return x + self.mamba(x)


# ── Main model ────────────────────────────────────────────────────────────────


class Mamba2MIL(BaseMIL):
    """
    Mamba-2 based MIL aggregator for slide-level representation learning.

    Architecture:
        1. Embedding:  Linear → ReLU  (project D_in → embed_dim)
        2. SSM:        [Mamba2-SSD + residual] x num_layers → Linear → LayerNorm
        3. Pooling:    Gated Attention → weighted sum → slide embedding

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension (e.g., 1536 for UNIv2).
    embed_dim : int
        Internal embedding dimension. Controls VRAM usage.
        384 recommended for B=128, N=4800 on 24GB GPU.
    num_layers : int
        Number of Mamba-2 SSD layers. COBRA uses 2. MambaMIL uses 2+.
    d_state : int
        SSM state dimension. Higher = more memory per token.
        64 recommended (COBRA uses 64). Mamba2 class defaults to 128.
    d_conv : int
        Causal conv1d kernel size.
    expand : int
        Inner dimension expansion factor. d_inner = expand * embed_dim.
    headdim : int
        Head dimension for SSD. Must divide (expand * embed_dim).
    attn_dim : int
        Hidden dim of the gated attention pooling network.
    dropout : float
        Dropout in the attention network.
    use_cobra_inference : bool
        If True, at eval time: use Mamba-derived attention weights
        but pool over the ORIGINAL (pre-Mamba) embeddings, following
        COBRA Eq. 6. This preserves the FM feature space.
        During training, always pools over Mamba-encoded features.
    gradient_checkpointing : bool
        Checkpoint each Mamba layer to save activation memory.
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 384,
        num_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        attn_dim: int = 256,
        dropout: float = 0.1,
        use_cobra_inference: bool = False,
        gradient_checkpointing: bool = False,
        use_hilbert: bool = True,
        hilbert_order: int = 16,
    ):
        super().__init__(
            in_dim=in_dim,
            embed_dim=embed_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.use_hilbert = use_hilbert
        self.hilbert_order = hilbert_order
        self.use_cobra_inference = use_cobra_inference

        # ── Stage 1: Embedding ────────────────────────────────────────────
        # Simple linear projection + activation.
        # With hid_dims=[] this produces: Linear(in_dim, embed_dim) → ReLU
        # Dropout has no effect with no hidden layers, so set to 0 explicitly.
        self.patch_embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim, bias=True),
            nn.LayerNorm(embed_dim),
        )

        # ── Stage 2: Mamba-2 SSD layers ───────────────────────────────────
        self.blocks = nn.ModuleList(
            [
                Mamba2Layer(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                )
                for _ in range(num_layers)
            ]
        )

        # Post-SSM projection + normalization before attention pooling.
        # Both COBRA (Eq. 3) and MambaMIL (Eq. 9) apply a Linear projection
        # after the SSM layers before feeding into the attention aggregator.
        # This gives the model a learned transform between the SSM feature
        # space and the attention pooling space.
        self.post_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # ── Stage 3: Gated Attention Pooling ──────────────────────────────
        self.global_attn = GlobalGatedAttention(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1,
        )

        # NOTE: We intentionally do NOT call self.initialize_weights().
        # Mamba2 has S4-inherited initialization for A_log, dt_bias, and D
        # that would be destroyed by a blanket reset_parameters() call.
        # The embedding MLP and attention layers use PyTorch's default init.

    def forward_features(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patch features into slide embedding.

        Parameters
        ----------
        h : torch.Tensor
            [B, N, D] patch embeddings from a single FM.
        mask : torch.Tensor, optional
            [B, N] binary mask (1=valid, 0=padding).
            Only applied at attention pooling, not within SSM layers.
        coords : torch.Tensor, optional
            [B, N, 2] spatial coordinates. Used for Hilbert ordering when
            ``use_hilbert=True``.
        return_attention : bool
            Include pre-softmax attention logits in output
            (consistent with ABMIL convention).

        Returns
        -------
        slide_embedding : torch.Tensor [B, embed_dim]
        extras : dict
            If return_attention=True, contains 'attention_weights': [B, N]
            as pre-softmax logits (consistent with ABMIL).
        """
        # ── Stage 1: Embed ────────────────────────────────────────────────
        h_embed = self.patch_embed(h)  # [B, N, E]
        if mask is not None and (mask.sum(dim=1) == 0).any():
            raise ValueError("Mamba2MIL received a slide with zero valid patches.")
        if self.use_hilbert and coords is not None:
            from oceanpath.models.components import hilbert_sort

            h_embed, coords, mask, sort_idx = hilbert_sort(
                h_embed,
                coords,
                mask,
                order=self.hilbert_order,
            )
        else:
            sort_idx = None

        # ── Stage 2: Mamba-2 SSD layers ───────────────────────────────────
        # Note on residual pattern:
        # COBRA (Eq. 3) uses flat skip connections from H_E for all residuals:
        #   H_S = Lin(SSD(SSD(H_E) + H_E) + H_E)
        # Our implementation uses cascading residuals (each block adds to its
        # own input), which generalizes naturally to num_layers > 2.
        # For exactly 2 layers, the difference is minor.
        h_ssm = h_embed
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                h_ssm = cp.checkpoint(blk, h_ssm, use_reentrant=False)
            else:
                h_ssm = blk(h_ssm)

        # Post-SSM projection + norm (COBRA Eq. 3: outer Lin(); MambaMIL Eq. 9: Linear)
        h_ssm = self.post_proj(h_ssm)  # [B, N, E]
        h_ssm = self.norm(h_ssm)  # [B, N, E]

        # ── Stage 3: Gated Attention Pooling ──────────────────────────────
        attn_logits = self.global_attn(h_ssm)  # [B, N, 1]
        attn_logits = attn_logits.transpose(-2, -1)  # [B, 1, N]

        # Mask padding tokens before softmax
        if mask is not None:
            attn_logits = attn_logits.masked_fill((1 - mask).unsqueeze(1).bool(), float("-inf"))

        # Float32 softmax + pooling for numerical stability under AMP.
        # Without this, autocast silently downcasts fp32 attention weights
        # back to bf16 for the bmm, undoing the precision gain from the
        # fp32 softmax.
        device_type = h_ssm.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            attn_weights = F.softmax(attn_logits.float(), dim=-1)

            # Choose what to pool over
            if self.use_cobra_inference and not self.training:
                # COBRA trick (Eq. 6): attention from Mamba, but pool raw embeddings
                pool_features = h_embed  # [B, N, E]
            else:
                # Standard: pool Mamba-encoded features (used during SSL training)
                pool_features = h_ssm  # [B, N, E]

            slide_embedding = torch.bmm(attn_weights, pool_features.float()).squeeze(1)  # [B, E]

        # ── Extras ────────────────────────────────────────────────────────
        extras = {}
        if return_attention:
            if sort_idx is not None:
                B, N = sort_idx.shape  # ← add this
                inv_idx = torch.empty_like(sort_idx)
                inv_idx.scatter_(1, sort_idx, torch.arange(N, device=sort_idx.device).expand(B, -1))
                extras["attention_weights"] = attn_logits.squeeze(1).gather(1, inv_idx)
            else:
                extras["attention_weights"] = attn_logits.squeeze(1)  # [B, N]

        return slide_embedding, extras
