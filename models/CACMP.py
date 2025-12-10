import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL

class CACMP(MIL):
    """
    Cancer-Aware Competitive Multi-Phenotype ABMIL (simple, drop-in).

    - K phenotype heads (default 3): {neoplastic, stroma/immune, necrosis/other}
    - Instance->phenotype competitive routing (softmax over heads per instance)
    - Per-head attention over instances (softmax over instances per head)
    - Effective weights = (routing * attention), normalized per head
    - Slide embedding = concat([mu_1, ..., mu_K, mu_1 - mu_2]) -> linear projection
    - Optional: classification head
    - Returns a log_dict with attentions and regularizers you can add to the loss.

    Args:
        in_dim:   instance feature dimension (e.g., 1024 from FM)
        embed_dim: patch embedding dimension
        attn_dim:  attention hidden width
        dropout:   dropout in MLP/attention
        gate:      use gated attention (Ilse et al.) or vanilla
        num_heads: number of phenotype pools (default 3)
        num_classes: if >0, add a linear classifier head
        proj_out_dim: output slide-embedding dim after projection (default embed_dim)
        init_temp: initial temperature for per-head attention sharpening
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        attn_dim: int = 384,
        dropout: float = 0.25,
        gate: bool = True,
        num_heads: int = 3,
        num_classes: int = 2,
        proj_out_dim: int = 512,
        init_temp: float = 0.7,  # <1 sharpens top patches slightly
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        assert num_heads >= 2, "Use at least 2 heads to form an interface vector."
        self.K = num_heads
        self.proj_out_dim = proj_out_dim

        # Patch embed (lightweight MLP)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[],
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        # Per-head attention modules (1 head each, no multi-head transformer)
        Attn = GlobalGatedAttention if gate else GlobalAttention
        self.heads = nn.ModuleList([
            Attn(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)
            for _ in range(self.K)
        ])

        # Competitive router: instance -> phenotype logits (B, M, K)
        self.router = nn.Linear(embed_dim, self.K)

        # Learnable per-head temperatures for instance softmax (attention sharpness)
        # Store and transform via softplus to keep >0
        init = torch.log(torch.exp(torch.tensor(init_temp)) - 1.)
        self._raw_temps = nn.Parameter(init.repeat(self.K))

        # Simple projector to final slide embedding (kept tiny)
        in_proj = (self.K + 1) * embed_dim  # concat K mus + 1 interface (mu0 - mu1)
        self.project = nn.Sequential(
            nn.Linear(in_proj, proj_out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_out_dim),
        )

        # Optional classifier head
        self.classifier = nn.Linear(proj_out_dim, num_classes) if num_classes and num_classes > 0 else None

        self.initialize_weights()

    @property
    def temps(self) -> torch.Tensor:
        # softplus to ensure positivity; add tiny floor for numerical stability
        return F.softplus(self._raw_temps) + 1e-6  # (K,)

    def _phenotype_pool(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        h: [B, M, D] embedded instances
        attn_mask: [B, M] with 1 valid, 0 masked (optional)

        Returns:
            Z: [B, (K+1)*D] concatenated phenotype means + (mu0 - mu1)
            log: dict with attentions, assignments, regularizer terms (no loss combined)
        """
        B, M, D = h.shape

        # Competitive routing: per-instance phenotype assignment
        router_logits = self.router(h)                          # [B, M, K]
        gamma = torch.softmax(router_logits, dim=-1)            # [B, M, K] over heads

        # Per-head attention over instances
        alphas = []
        raw_logits = []
        for k, head in enumerate(self.heads):
            logits_k = head(h).squeeze(-1)                      # [B, M]
            if attn_mask is not None:
                logits_k = logits_k.masked_fill((attn_mask == 0), float("-inf"))
            # temperature per head for instance softmax
            T = self.temps[k]
            alpha_k = torch.softmax(logits_k / T, dim=1)        # [B, M] over instances
            raw_logits.append(logits_k)
            alphas.append(alpha_k)

        alphas = torch.stack(alphas, dim=2)                     # [B, M, K]
        # Effective weights: instance must both (i) belong to phenotype k and (ii) be salient within that head
        w = alphas * gamma                                      # [B, M, K]
        # Normalize per head across instances to sum to 1
        denom = w.sum(dim=1, keepdim=True) + 1e-8               # [B, 1, K]
        w_norm = w / denom                                      # [B, M, K]

        # Per-phenotype pooled embeddings
        h_exp = h.unsqueeze(-1)                                 # [B, M, D, 1]
        w_exp = w_norm.unsqueeze(2)                             # [B, M, 1, K]
        mu = (h_exp * w_exp).sum(dim=1)                         # [B, D, K]
        mu = mu.transpose(1, 2).contiguous()                    # [B, K, D]

        # Interface vector (mu0 - mu1)
        delta = (mu[:, 0, :] - mu[:, 1, :]).unsqueeze(1)        # [B, 1, D]
        slide_stack = torch.cat([mu, delta], dim=1).reshape(B, -1)  # [B, (K+1)*D]

        # Fractions of assignments per head (use gamma, not alpha)
        frac = gamma.mean(dim=1)                                # [B, K]

        # Regularizers (returned; you decide weights in your loss)
        # 1) Assignment entropy per instance (encourage confident routing)
        # mean entropy across instances and batch
        ent = -(gamma.clamp_min(1e-8) * gamma.clamp_min(1e-8).log()).sum(dim=-1).mean()

        # 2) Balance across heads within a slide (avoid collapse)
        # target ~ uniform(1/K); MSE averaged over batch
        balance = ((frac - frac.new_full(frac.shape, 1.0 / self.K)) ** 2).mean()

        # 3) Phenotype diversity (minimize cosine similarity between pooled mu vectors)
        # sum of squared cosine similarities across distinct pairs
        mu_norm = F.normalize(mu, dim=-1)                       # [B, K, D]
        cos2 = 0.0
        cnt = 0
        for i in range(self.K):
            for j in range(i + 1, self.K):
                cos2 = cos2 + (mu_norm[:, i] * mu_norm[:, j]).sum(dim=-1).pow(2).mean()
                cnt += 1
        diversity = cos2 / max(cnt, 1)

        log = dict(
            router_gamma=gamma,            # [B, M, K] per-instance head assignment
            alphas=alphas,                 # [B, M, K] per-head instance attention
            eff_weights=w_norm,            # [B, M, K] effective normalized weights
            mu=mu,                         # [B, K, D]
            frac=frac,                     # [B, K]
            assign_entropy=ent,            # scalar
            balance_mse=balance,           # scalar
            diversity_cos2=diversity,      # scalar
            raw_attn_logits=torch.stack(raw_logits, dim=2),  # [B, M, K]
            temps=self.temps.detach().cpu()
        )
        return slide_stack, log

    def forward_attention(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_only: bool = True
    ):
        """
        For parity with your ABMIL signature. Returns router/attn tensors.
        """
        h = self.patch_embed(h)                                 # [B, M, D]
        Z_stack, log = self._phenotype_pool(h, attn_mask=attn_mask)
        if attn_only:
            return log["eff_weights"].transpose(1, 2)           # [B, K, M]
        return h, log["eff_weights"].transpose(1, 2)

    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Produces the cancer-aware slide embedding and a rich log_dict.
        """
        h = self.patch_embed(h)                                 # [B, M, D]
        Z_stack, log = self._phenotype_pool(h, attn_mask=attn_mask)
        z = self.project(Z_stack)                               # [B, proj_out_dim]
        if not return_attention:
            log = {}
        return z, log

    def forward(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        z, log = self.forward_features(h, attn_mask=attn_mask, return_attention=True)
        if self.classifier is None:
            return z, log
        logits = self.classifier(z)
        return logits, log


def cancer_regularizer(
    log: Dict,
    lambda_assign: float = 0.02,
    lambda_balance: float = 0.05,
    lambda_diversity: float = 0.01,
    edges: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    lambda_spatial: float = 0.0
) -> torch.Tensor:
    """
    Combines optional unsupervised regularizers.
      - Assignment entropy   : encourage crisper per-patch phenotype choice
      - Balance MSE          : avoid head collapse (useful with strong FMs)
      - Diversity (cos^2)    : discourage redundant phenotype pools
      - Spatial coherence    : OPTIONAL; if edges=(src,dst) over instances are provided,
                               penalize differences in router assignments along edges.
    Returns a scalar you can add to the CE/BCE loss.
    """
    reg = (
        lambda_assign   * log["assign_entropy"]
      + lambda_balance  * log["balance_mse"]
      + lambda_diversity* log["diversity_cos2"]
    )
    if lambda_spatial > 0.0 and edges is not None:
        src, dst = edges  # [E], [E] indices into instance axis
        gamma = log["router_gamma"]  # [B, M, K]
        # Apply per-bag, average over batch
        B, M, K = gamma.shape
        spatial = 0.0
        for b in range(B):
            g = gamma[b]  # [M, K]
            diff = (g[src] - g[dst]).pow(2).sum(dim=-1).mean()
            spatial = spatial + diff
        spatial = spatial / B
        reg = reg + lambda_spatial * spatial
    return reg
