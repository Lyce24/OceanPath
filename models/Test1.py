# fusion_mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL


class CancerCHAMPFusionMIL(MIL):
    """
    Cancer-CHAMP-ABMIL (Fusion):
      - K phenotype heads with competitive routing (instance -> head)
      - Per-head attention = softmax(s_k / tau_k(bag)) * prior^gamma
      - Effective per-head weights = routing_k * attention_k  (renormalized per head)
      - Per-head heterogeneity pooling: concat [mu | std | quantile-contrast] -> small compressor
      - Slide embedding = concat( z_1, ..., z_K, (mu_0 - mu_1) ) -> projector -> classifier

    Assumptions:
      head 0 ~ neoplastic (tumor), head 1 ~ stroma/immune, head 2 ~ necrosis/other (when K=3).
      Change `interface_heads=(0,1)` if you order differently.

    Key knobs to ablate:
      - gamma=0.0 disables prior modulation (pure ABMIL per head)
      - target_prevalence sets the dynamic temperature pivot (None disables bag-tempering)
      - use_quantile=False disables quantile contrast (keeps mean/std only)
      - include_interface=False disables (mu_0 - mu_1)

    Shapes:
      h: [B, M, D_in]; mask: [B, M] {1/0}
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
        # CHAMP bits
        target_prevalence: Optional[float] = 0.10,  # None -> disable dynamic tau
        base_tau: float = 1.0,                      # init per-head learnable tau
        gamma: float = 1.0,                         # strength of prior modulation
        top_frac: float = 0.10,
        bottom_frac: float = 0.10,
        use_quantile: bool = True,
        # Cancer-ABMIL bits
        proj_out_dim: int = 512,
        include_interface: bool = True,
        interface_heads: Tuple[int, int] = (0, 1),
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        assert num_heads >= 2, "Need at least 2 heads for an interface vector."
        self.K = num_heads
        self.gamma = gamma
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.use_quantile = use_quantile
        self.include_interface = include_interface
        self.interface_heads = interface_heads
        self.target_prev = target_prevalence

        # -------- Patch embedding (light MLP) --------
        self.patch_embed = create_mlp(
            in_dim=in_dim, hid_dims=[], dropout=dropout, out_dim=embed_dim, end_with_fc=False
        )

        # -------- Per-head (gated) attention providers --------
        Attn = GlobalGatedAttention if gate else GlobalAttention
        self.attn_heads = nn.ModuleList([
            Attn(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1) for _ in range(self.K)
        ])

        # -------- Competitive router (instance -> phenotype/head logits) --------
        self.router = nn.Linear(embed_dim, self.K)

        # -------- Prior ("canceriness") per instance --------
        self.prior_head = nn.Linear(embed_dim, 1)

        # -------- Learnable base tau per head (combined with bag-temper) --------
        init = torch.log(torch.tensor(base_tau).exp() - 1.0)  # softplus^-1
        self._raw_tau = nn.Parameter(init.repeat(self.K))     # (K,)

        # -------- Per-head compressor for heterogeneity pooling (3*D -> D) --------
        comp_in = embed_dim * (3 if use_quantile else 2)
        self.compressor = nn.Sequential(
            nn.Linear(comp_in, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -------- Projector & classifier --------
        final_in = self.K * embed_dim + (embed_dim if include_interface else 0)
        self.project = nn.Sequential(
            nn.Linear(final_in, proj_out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_out_dim),
        )
        self.classifier = nn.Linear(proj_out_dim, num_classes) if num_classes and num_classes > 0 else None

        self.register_buffer("eps", torch.tensor(1e-6), persistent=False)
        self.initialize_weights()

    # ------------------- Utilities -------------------

    @property
    def base_taus(self) -> torch.Tensor:
        # >0
        return F.softplus(self._raw_tau) + 1e-6  # (K,)

    def _bag_tau_factor(self, prior_probs: torch.Tensor) -> torch.Tensor:
        """ prior_probs: [B,M] -> factor [B,1]; None if dynamic tau disabled. """
        if self.target_prev is None:
            return None
        mean_prev = prior_probs.mean(dim=1, keepdim=True)  # [B,1]
        ratio = mean_prev / (self.target_prev + float(self.eps))
        return torch.clamp(ratio, 0.5, 2.0)               # [B,1]

    @staticmethod
    def _safe_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return logits

    # ------------------- Core blocks -------------------

    def _per_head_attention(
        self,
        H: torch.Tensor,                 # [B,M,D]
        prior: torch.Tensor,             # [B,M]
        mask: Optional[torch.Tensor],    # [B,M] 1=valid
        bag_tau_factor: Optional[torch.Tensor],  # [B,1] or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          attn_logits: [B,M,K] (pre-softmax s_k)
          attn_weights: [B,M,K] (softmax(s_k/tau_k) * prior^gamma, normalized per head)
        """
        B, M, D = H.shape
        attn_logits = []
        attn_weights = []

        for k, head in enumerate(self.attn_heads):
            s_k = head(H).squeeze(-1)                       # [B,M]
            s_k = self._safe_mask(s_k, mask)
            # dynamic tau: per-head learnable * bag factor (if provided)
            tau_k = self.base_taus[k]                       # scalar >0
            if bag_tau_factor is not None:
                tau_k = tau_k * bag_tau_factor              # [B,1]
            # scale logits and softmax over instances
            s_scaled = s_k / torch.clamp(tau_k, min=1e-6)
            a_k = F.softmax(s_scaled, dim=-1)               # [B,M]

            # prior modulation (product-of-experts)
            if self.gamma != 0.0:
                a_k = a_k * torch.clamp(prior, min=1e-6).pow(self.gamma)
                a_k = a_k / (a_k.sum(dim=-1, keepdim=True) + 1e-6)

            attn_logits.append(s_k)
            attn_weights.append(a_k)

        attn_logits = torch.stack(attn_logits, dim=-1)      # [B,M,K]
        attn_weights = torch.stack(attn_weights, dim=-1)    # [B,M,K]
        return attn_logits, attn_weights

    def _weighted_stats(
        self,
        H: torch.Tensor,                 # [B,M,D]
        w: torch.Tensor,                 # [B,M] (sum=1)
        top_frac: float,
        bot_frac: float,
        use_quantile: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Returns:
          mu: [B,D], std: [B,D], delta_q (or None): [B,D], aux idx dict
        """
        B, M, D = H.shape
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)        # normalize
        mu = torch.bmm(w.unsqueeze(1), H).squeeze(1)        # [B,D]

        dev = H - mu.unsqueeze(1)
        var = torch.bmm(w.unsqueeze(1), dev.pow(2)).squeeze(1)
        std = torch.sqrt(var + 1e-6)

        delta_q, aux = None, {"top_idx": None, "bot_idx": None}
        if use_quantile:
            k_top = max(1, min(M, int(round(top_frac * M))))
            k_bot = max(1, min(M, int(round(bot_frac * M))))

            top_vals, top_idx = torch.topk(w, k=k_top, dim=-1, largest=True, sorted=False)   # [B,k_top]
            bot_vals, bot_idx = torch.topk(w, k=k_bot, dim=-1, largest=False, sorted=False)  # [B,k_bot]

            H_top = H.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
            H_bot = H.gather(1, bot_idx.unsqueeze(-1).expand(-1, -1, D))

            mu_top = (top_vals.unsqueeze(-1) * H_top).sum(dim=1) / (top_vals.sum(dim=1, keepdim=True) + 1e-6)
            mu_bot = (bot_vals.unsqueeze(-1) * H_bot).sum(dim=1) / (bot_vals.sum(dim=1, keepdim=True) + 1e-6)
            delta_q = mu_top - mu_bot

            aux["top_idx"], aux["bot_idx"] = top_idx, bot_idx

        return mu, std, delta_q, aux

    def _phenotype_pool(
        self,
        H: torch.Tensor,                 # [B,M,D]
        mask: Optional[torch.Tensor],    # [B,M]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
          slide_stack: [B, K*D + (include_interface? D:0)] (pre-projector)
          log: dict of diagnostics & regularizers
        """
        B, M, D = H.shape

        # Router (instance -> phenotype)
        router_logits = self.router(H)                             # [B,M,K]
        router_gamma = torch.softmax(router_logits, dim=-1)        # [B,M,K]

        # Prior & bag temper
        prior = torch.sigmoid(self.prior_head(H)).squeeze(-1)      # [B,M]
        if mask is not None:
            prior = torch.where(mask.bool(), prior, torch.zeros_like(prior))
        bag_tau_factor = self._bag_tau_factor(prior)               # [B,1] or None

        # Per-head attention (CHAMP-style) then combine with routing (Cancer-ABMIL)
        attn_logits, attn_weights = self._per_head_attention(H, prior, mask, bag_tau_factor)  # [B,M,K], [B,M,K]
        eff_w = attn_weights * router_gamma                        # [B,M,K]
        eff_w = eff_w / (eff_w.sum(dim=1, keepdim=True) + 1e-6)    # normalize per head

        # Per-head heterogeneity pooling -> compressor (3*D -> D)
        per_head_vecs = []
        per_head_mu = []
        top_idx_all, bot_idx_all = [], []
        for k in range(self.K):
            mu_k, std_k, dq_k, aux = self._weighted_stats(
                H, eff_w[..., k], self.top_frac, self.bottom_frac, self.use_quantile
            )
            per_head_mu.append(mu_k)                                # [B,D]
            if self.use_quantile:
                z_k = torch.cat([mu_k, std_k, dq_k], dim=-1)       # [B,3D]
            else:
                z_k = torch.cat([mu_k, std_k], dim=-1)             # [B,2D]
            per_head_vecs.append(self.compressor(z_k))              # [B,D]
            top_idx_all.append(aux["top_idx"])
            bot_idx_all.append(aux["bot_idx"])

        Z_heads = torch.cat(per_head_vecs, dim=-1)                  # [B,K*D]
        mu_stack = torch.stack(per_head_mu, dim=1)                  # [B,K,D]

        # Interface vector (mu_tumor - mu_stroma)
        slide_stack = Z_heads
        if self.include_interface:
            i, j = self.interface_heads
            delta_mu = (mu_stack[:, i, :] - mu_stack[:, j, :])     # [B,D]
            slide_stack = torch.cat([Z_heads, delta_mu], dim=-1)    # [B,K*D + D]

        # Regularizers
        frac = router_gamma.mean(dim=1)                             # [B,K] usage per head
        assign_entropy = -(router_gamma.clamp_min(1e-8) * router_gamma.clamp_min(1e-8).log()).sum(dim=-1).mean()
        balance_mse = ((frac - frac.new_full(frac.shape, 1.0 / self.K)) ** 2).mean()

        # Phenotype diversity (cos^2 between pooled mu vectors)
        mu_norm = F.normalize(mu_stack, dim=-1)                     # [B,K,D]
        cos2, cnt = 0.0, 0
        for a in range(self.K):
            for b in range(a + 1, self.K):
                cos2 = cos2 + (mu_norm[:, a] * mu_norm[:, b]).sum(dim=-1).pow(2).mean()
                cnt += 1
        diversity_cos2 = cos2 / max(cnt, 1)

        log = dict(
            router_gamma=router_gamma,          # [B,M,K]
            attn_logits=attn_logits,            # [B,M,K]
            attn_weights=attn_weights,          # [B,M,K]
            eff_weights=eff_w,                   # [B,M,K]
            prior=prior,                         # [B,M]
            bag_tau_factor=bag_tau_factor,       # [B,1] or None
            mu=mu_stack,                         # [B,K,D]
            head_usage=frac,                     # [B,K]
            assign_entropy=assign_entropy,       # scalar
            balance_mse=balance_mse,             # scalar
            diversity_cos2=diversity_cos2,       # scalar
            top_idx=top_idx_all,                 # list(len=K) of [B,k_top] or None
            bot_idx=bot_idx_all,                 # list(len=K) of [B,k_bot] or None
            base_taus=self.base_taus.detach().cpu(),
        )
        return slide_stack, log

    # ------------------- Public APIs -------------------

    def forward_attention(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, attn_only: bool = True):
        H = self.patch_embed(h)                                   # [B,M,D]
        _, log = self._phenotype_pool(H, mask=attn_mask)
        if attn_only:
            # return final effective weights per head: [B,K,M]
            return log["eff_weights"].transpose(1, 2)
        return H, log["eff_weights"].transpose(1, 2)

    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        H = self.patch_embed(h)                                   # [B,M,D]
        Z_stack, log = self._phenotype_pool(H, mask=attn_mask)    # [B, *]
        z = self.project(Z_stack)                                 # [B, proj_out_dim]
        if not return_attention:
            log = {}
        return z, log

    def forward(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        z, log = self.forward_features(h, attn_mask=attn_mask, return_attention=True)
        if self.classifier is None:
            return z, log
        logits = self.classifier(z)
        return logits, log


def cancer_fusion_regularizer(
    log: Dict,
    lambda_assign: float = 0.02,
    lambda_balance: float = 0.05,
    lambda_diversity: float = 0.01,
    edges: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    lambda_spatial: float = 0.0,
) -> torch.Tensor:
    """
    Unsupervised regularization for the fusion head.
      - assignment entropy:   encourage confident router assignments
      - balance MSE:          avoid head collapse
      - diversity (cos^2):    discourage redundant phenotype pools
      - spatial coherence:    optional Laplacian smoothing of router assignments across adjacency edges
    """
    reg = (
        lambda_assign   * log["assign_entropy"]
      + lambda_balance  * log["balance_mse"]
      + lambda_diversity* log["diversity_cos2"]
    )
    if lambda_spatial > 0.0 and edges is not None:
        src, dst = edges  # [E], [E] indices along the instance axis
        gamma = log["router_gamma"]  # [B,M,K]
        B = gamma.shape[0]
        spatial = 0.0
        for b in range(B):
            g = gamma[b]                           # [M,K]
            spatial += (g[src] - g[dst]).pow(2).sum(dim=-1).mean()
        spatial = spatial / B
        reg = reg + lambda_spatial * spatial
    return reg
