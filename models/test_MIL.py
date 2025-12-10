# fusion_mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL

# fusion_mil_can.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL


class CIPHER(MIL):
    """
    Cancer-CHAMP-ABMIL + CAN-Pool integration.

    Ingredients
    ----------
    (A) Multi-phenotype pooling with competitive routing (K heads):
        - Router assigns patches -> heads (soft)
        - Per-head attention is CHAMP-style: softmax(s_k / τ_k(bag)) * prior^γ
        - Effective weights per head = router * attention  (renormalized per head)
        - Per-head heterogeneity pooling: [μ | σ | quantile-contrast] -> compressor

    (B) Interface signal: append (μ_tumor - μ_stroma) before projector.

    (C) CAN-Pool extensions (all optional, cheap):
        - Heterogeneity-tempered attention: τ_k(bag) *= (1 + α * H_bag),
          where H_bag is a scalar feature spread per bag.
        - Lesion-purity prior (dynamic top-p nucleus):
          predict bag-wise p̂ ∈ [p_min, p_max]; after softmax produce a hard top-p
          mask on selected head(s), renormalize.
        - Compartment debias: learn tumor/stroma prototypes; subtract a small
          projection onto the stroma direction from patch embeddings before pooling.

    Notes
    -----
    - Set `enable_can_*` flags to ablate each CAN-Pool addition independently.
    - By default, top-p nucleus is applied only to the `tumor` head (interface_heads[0]).
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
        target_prevalence: Optional[float] = 0.10,  # None -> disable bag-temper (prior)
        base_tau: float = 1.0,
        gamma: float = 1.0,                         # prior modulation strength
        top_frac: float = 0.10,
        bottom_frac: float = 0.10,
        use_quantile: bool = True,
        # Cancer-ABMIL bits
        proj_out_dim: int = 512,
        include_interface: bool = True,
        interface_heads: Tuple[int, int] = (0, 1),  # (tumor_idx, stroma_idx)
        # CAN-Pool: heterogeneity-tempered attention
        enable_can_hetero: bool = True,
        hetero_alpha: float = 0.75,                 # scales H_bag inside (1 + αH)
        hetero_scale: float = 0.10,                 # tanh(hetero_scale * raw_spread)
        # CAN-Pool: dynamic top-p nucleus
        enable_can_nucleus: bool = True,
        pmin: float = 0.02,
        pmax: float = 0.30,
        nucleus_apply_to: str = "tumor",            # {'tumor', 'all', 'none'}
        # CAN-Pool: compartment debias
        enable_can_debias: bool = True,
        debias_beta: float = 0.20,                  # magnitude of stroma projection subtraction
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        assert num_heads >= 2, "Need at least 2 heads for interface vector."
        self.K = num_heads
        self.gamma = gamma
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.use_quantile = use_quantile
        self.include_interface = include_interface
        self.interface_heads = interface_heads
        self.target_prev = target_prevalence

        # ---- CAN flags / params ----
        self.enable_can_hetero = enable_can_hetero
        self.hetero_alpha = hetero_alpha
        self.hetero_scale = hetero_scale

        self.enable_can_nucleus = enable_can_nucleus
        self.pmin, self.pmax = pmin, pmax
        assert nucleus_apply_to in {"tumor", "all", "none"}
        self.nucleus_apply_to = nucleus_apply_to

        self.enable_can_debias = enable_can_debias
        self.debias_beta = debias_beta

        # -------- Patch embedding --------
        self.patch_embed = create_mlp(
            in_dim=in_dim, hid_dims=[], dropout=dropout, out_dim=embed_dim, end_with_fc=False
        )

        # -------- Per-head (gated) attention --------
        Attn = GlobalGatedAttention if gate else GlobalAttention
        self.attn_heads = nn.ModuleList([
            Attn(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1) for _ in range(self.K)
        ])

        # -------- Competitive router --------
        self.router = nn.Linear(embed_dim, self.K)

        # -------- Prior (per-instance "canceriness") --------
        self.prior_head = nn.Linear(embed_dim, 1)

        # -------- Learnable base tau per head --------
        init = torch.log(torch.tensor(base_tau).exp() - 1.0)  # softplus^-1
        self._raw_tau = nn.Parameter(init.repeat(self.K))     # (K,)

        # -------- Per-head compressor for [μ | σ | Δq] --------
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

        # -------- CAN-Pool: bag purity head (for top-p nucleus) --------
        # Tiny probe on unweighted bag mean of embedded patches.
        self.purity_head = nn.Linear(embed_dim, 1)

        # -------- CAN-Pool: compartment prototypes (unit vectors learned) --------
        self.proto_tumor = nn.Parameter(torch.randn(embed_dim))
        self.proto_stroma = nn.Parameter(torch.randn(embed_dim))

        self.register_buffer("eps", torch.tensor(1e-6), persistent=False)
        self.initialize_weights()

    # =================== Utilities ===================

    @property
    def base_taus(self) -> torch.Tensor:
        return F.softplus(self._raw_tau) + 1e-6  # (K,)

    def _normalize_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v / (v.norm(p=2) + 1e-6)

    def _bag_tau_factor_prior(self, prior_probs: torch.Tensor) -> Optional[torch.Tensor]:
        """Prior-based bag temper (CHAMP)."""
        if self.target_prev is None:
            return None
        mean_prev = prior_probs.mean(dim=1, keepdim=True)  # [B,1]
        ratio = mean_prev / (self.target_prev + float(self.eps))
        return torch.clamp(ratio, 0.5, 2.0)                # [B,1]

    def _bag_heterogeneity(self, H: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Scalar spread per bag, normalized to ~[0,1] via tanh.
        H: [B,M,D]; mask: [B,M] (1 valid)
        """
        if mask is not None:
            # approximate masked mean
            w = mask.float() / (mask.float().sum(dim=1, keepdim=True) + 1e-6)  # [B,M]
            mu = torch.bmm(w.unsqueeze(1), H).squeeze(1)                       # [B,D]
            dev = H - mu.unsqueeze(1)
            var = torch.bmm(w.unsqueeze(1), dev.pow(2)).squeeze(1).mean(dim=1, keepdim=True)  # [B,1] mean over D
        else:
            mu = H.mean(dim=1)
            dev = H - mu.unsqueeze(1)
            var = dev.pow(2).mean(dim=(1, 2), keepdim=True)  # [B,1]
        spread = torch.sqrt(var + 1e-6)                      # [B,1]
        H_bag = torch.tanh(self.hetero_scale * spread)       # squash to ~[0,1]
        return H_bag  # [B,1]

    def _bag_purity(self, H: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Tiny probe for p̂ in [pmin, pmax]; input is unweighted mean embedding.
        """
        if mask is not None:
            w = mask.float() / (mask.float().sum(dim=1, keepdim=True) + 1e-6)
            mu = torch.bmm(w.unsqueeze(1), H).squeeze(1)  # [B,D]
        else:
            mu = H.mean(dim=1)                            # [B,D]
        raw = torch.sigmoid(self.purity_head(mu))         # [B,1] in (0,1)
        p_hat = self.pmin + (self.pmax - self.pmin) * raw
        return p_hat  # [B,1]

    @staticmethod
    def _safe_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return logits

    # =================== Core blocks ===================

    def _apply_compartment_debias(self, H: torch.Tensor) -> torch.Tensor:
        """
        Subtract a small projection onto stroma prototype: H' = H - β * <H, v_s> v_s
        Stop background/stroma from diluting tumor signal.
        """
        if not self.enable_can_debias or self.debias_beta <= 0:
            return H
        v_s = self._normalize_vec(self.proto_stroma)           # [D]
        proj = torch.einsum("bmd,d->bm", H, v_s).unsqueeze(-1) # [B,M,1]
        H_deb = H - self.debias_beta * proj * v_s              # [B,M,D]
        return H_deb

    def _per_head_attention(
        self,
        H: torch.Tensor,                 # [B,M,D]
        prior: torch.Tensor,             # [B,M]
        mask: Optional[torch.Tensor],    # [B,M]
        bag_tau_prior: Optional[torch.Tensor],   # [B,1] or None
        bag_tau_hetero: Optional[torch.Tensor],  # [B,1] or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          attn_logits:  [B,M,K]
          attn_weights: [B,M,K]  (CHAMP: softmax(s/tau) * prior^gamma, norm per head)
        """
        attn_logits, attn_weights = [], []
        for k, head in enumerate(self.attn_heads):
            s_k = head(H).squeeze(-1)                       # [B,M]
            s_k = self._safe_mask(s_k, mask)

            # τ_k(bag) = base_tau_k * bag_tau_prior * (1 + α H_bag)
            tau_k = self.base_taus[k]                       # scalar
            if bag_tau_prior is not None:
                tau_k = tau_k * bag_tau_prior               # [B,1]
            if self.enable_can_hetero and bag_tau_hetero is not None:
                tau_k = tau_k * (1.0 + self.hetero_alpha * bag_tau_hetero)  # [B,1]

            s_scaled = s_k / torch.clamp(tau_k, min=1e-6)
            a_k = F.softmax(s_scaled, dim=-1)               # [B,M]

            if self.gamma != 0.0:
                a_k = a_k * torch.clamp(prior, min=1e-6).pow(self.gamma)
                a_k = a_k / (a_k.sum(dim=-1, keepdim=True) + 1e-6)

            attn_logits.append(s_k)
            attn_weights.append(a_k)

        return torch.stack(attn_logits, dim=-1), torch.stack(attn_weights, dim=-1)

    def _top_p_nucleus_mask(
        self,
        weights: torch.Tensor,        # [B,M] or [B,M,1]
        p_hat: torch.Tensor,          # [B,1]
        mask: Optional[torch.Tensor]  # [B,M] or [B,M,1] or None
    ) -> torch.Tensor:
        """
        Hard top-p (nucleus) selection per bag:
        - Sort weights desc
        - Keep the smallest set whose cumulative mass >= p̂
        - Map back to original order
        Returns: binary mask [B,M] (float).
        """
        # ---- normalize shapes ----
        if weights.ndim == 3 and weights.shape[-1] == 1:
            weights = weights.squeeze(-1)              # [B,M]
        if weights.ndim != 2:
            raise ValueError(f"Expected [B,M] or [B,M,1], got {tuple(weights.shape)}")

        B, M = weights.shape

        if p_hat.ndim == 2 and p_hat.shape[1] == 1:
            pass
        else:
            raise ValueError(f"p_hat must be [B,1], got {tuple(p_hat.shape)}")

        if mask is not None:
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)                # [B,M]
            if mask.ndim != 2:
                raise ValueError(f"mask must be [B,M] (or [B,M,1]), got {tuple(mask.shape)}")
            # zero out invalid positions so they go to the end after sorting
            weights = weights * mask.float()

        # ---- sort & cumulative sum ----
        vals, idx = torch.sort(weights, dim=1, descending=True)   # [B,M]
        csum = torch.cumsum(vals, dim=1)                          # [B,M]

        # index of first position where csum >= p_hat
        # (guaranteed to exist since csum[:, -1] == sum(weights) ~ 1 for a head)
        cross_idx = (csum >= p_hat).float().argmax(dim=1)         # [B]

        # keep prefix up to cross_idx (inclusive)
        arangeM = torch.arange(M, device=weights.device).unsqueeze(0).expand(B, M)  # [B,M]
        keep_sorted = (arangeM <= cross_idx.unsqueeze(1)).float()                   # [B,M]

        # ensure at least 1 kept even if all weights are ~0
        keep_sorted[:, 0] = 1.0

        # map back to original positions
        keep = torch.zeros_like(weights)                          # [B,M]
        keep.scatter_(1, idx, keep_sorted)

        if mask is not None:
            keep = keep * mask.float()                            # respect original mask

        return keep


    def _weighted_stats(
        self,
        H: torch.Tensor,                 # [B,M,D]
        w: torch.Tensor,                 # [B,M] (sum=1 where valid)
        top_frac: float,
        bot_frac: float,
        use_quantile: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        B, M, D = H.shape
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        mu = torch.bmm(w.unsqueeze(1), H).squeeze(1)        # [B,D]

        dev = H - mu.unsqueeze(1)
        var = torch.bmm(w.unsqueeze(1), dev.pow(2)).squeeze(1)
        std = torch.sqrt(var + 1e-6)

        delta_q, aux = None, {"top_idx": None, "bot_idx": None}
        if use_quantile:
            k_top = max(1, min(M, int(round(top_frac * M))))
            k_bot = max(1, min(M, int(round(bot_frac * M))))

            top_vals, top_idx = torch.topk(w, k=k_top, dim=-1, largest=True, sorted=False)
            bot_vals, bot_idx = torch.topk(w, k=k_bot, dim=-1, largest=False, sorted=False)

            H_top = H.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
            H_bot = H.gather(1, bot_idx.unsqueeze(-1).expand(-1, -1, D))

            mu_top = (top_vals.unsqueeze(-1) * H_top).sum(dim=1) / (top_vals.sum(dim=1, keepdim=True) + 1e-6)
            mu_bot = (bot_vals.unsqueeze(-1) * H_bot).sum(dim=1) / (bot_vals.sum(dim=1, keepdim=True) + 1e-6)
            delta_q = mu_top - mu_bot

            aux["top_idx"], aux["bot_idx"] = top_idx, bot_idx

        return mu, std, delta_q, aux

    def _phenotype_pool(
        self,
        H_in: torch.Tensor,                # [B,M,D]
        mask: Optional[torch.Tensor],      # [B,M]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
          slide_stack: [B, K*D + (include_interface? D:0)] (pre-projector)
          log: dict of diagnostics & regularizers
        """
        # ---- CAN-Pool: compartment debias (pre-attention) ----
        H = self._apply_compartment_debias(H_in)   # [B,M,D]

        B, M, D = H.shape

        # Router (instance -> phenotype)
        router_logits = self.router(H)                             # [B,M,K]
        router_gamma = torch.softmax(router_logits, dim=-1)        # [B,M,K]

        # Prior & bag temper (CHAMP)
        prior = torch.sigmoid(self.prior_head(H)).squeeze(-1)      # [B,M]
        if mask is not None:
            prior = torch.where(mask.bool(), prior, torch.zeros_like(prior))
        bag_tau_prior = self._bag_tau_factor_prior(prior)          # [B,1] or None

        # CAN-Pool: heterogeneity scalar H_bag for extra broadening
        bag_tau_hetero = None
        H_bag = None
        if self.enable_can_hetero:
            H_bag = self._bag_heterogeneity(H, mask)               # [B,1]
            bag_tau_hetero = H_bag

        # Per-head attention
        attn_logits, attn_weights = self._per_head_attention(
            H, prior, mask, bag_tau_prior, bag_tau_hetero
        )  # [B,M,K], [B,M,K]

        # Effective weights before (optional) nucleus filtering
        eff_w = attn_weights * router_gamma                         # [B,M,K]

        # CAN-Pool: dynamic top-p nucleus
        kept_counts = None
        p_hat = None
        if self.enable_can_nucleus and self.nucleus_apply_to != "none":
            # bag-wise fraction p̂ ∈ [pmin,pmax]
            p_hat = self._bag_purity(H, mask)                      # [B,1]
            kept_counts = torch.zeros(B, self.K, device=H.device)

            # choose which heads get nucleus filtering
            heads_to_filter = range(self.K) if self.nucleus_apply_to == "all" else [self.interface_heads[0]]

            # apply per selected head, then renormalize that head across instances
            for k in heads_to_filter:
                wk = eff_w[..., k]                     # could be [B,M] or [B,M,1] depending on upstream ops
                wk = wk.squeeze(-1) if wk.ndim == 3 else wk  # ensure [B,M]
                nucleus_mask = self._top_p_nucleus_mask(wk.detach(), p_hat, mask)  # [B,M] in {0,1}
                wk = wk * nucleus_mask
                wk = wk / (wk.sum(dim=1, keepdim=True) + 1e-6)
                eff_w[..., k] = wk if eff_w.ndim == 3 else wk.unsqueeze(-1)

        # Final per-head pooling -> compressor
        per_head_vecs, per_head_mu = [], []
        top_idx_all, bot_idx_all = [], []
        for k in range(self.K):
            mu_k, std_k, dq_k, aux = self._weighted_stats(
                H, eff_w[..., k], self.top_frac, self.bottom_frac, self.use_quantile
            )
            per_head_mu.append(mu_k)                                # [B,D]
            z_k = torch.cat([mu_k, std_k, dq_k], dim=-1) if self.use_quantile else torch.cat([mu_k, std_k], dim=-1)
            per_head_vecs.append(self.compressor(z_k))              # [B,D]
            top_idx_all.append(aux["top_idx"])
            bot_idx_all.append(aux["bot_idx"])

        Z_heads = torch.cat(per_head_vecs, dim=-1)                  # [B,K*D]
        mu_stack = torch.stack(per_head_mu, dim=1)                  # [B,K,D]

        # Interface vector
        slide_stack = Z_heads
        if self.include_interface:
            i, j = self.interface_heads
            delta_mu = (mu_stack[:, i, :] - mu_stack[:, j, :])     # [B,D]
            slide_stack = torch.cat([Z_heads, delta_mu], dim=-1)

        # Regularizers (as before)
        frac = router_gamma.mean(dim=1)
        assign_entropy = -(router_gamma.clamp_min(1e-8) * router_gamma.clamp_min(1e-8).log()).sum(dim=-1).mean()
        balance_mse = ((frac - frac.new_full(frac.shape, 1.0 / self.K)) ** 2).mean()

        mu_norm = F.normalize(mu_stack, dim=-1)
        cos2, cnt = 0.0, 0
        for a in range(self.K):
            for b in range(a + 1, self.K):
                cos2 = cos2 + (mu_norm[:, a] * mu_norm[:, b]).sum(dim=-1).pow(2).mean()
                cnt += 1
        diversity_cos2 = cos2 / max(cnt, 1)

        log = dict(
            router_gamma=router_gamma,          # [B,M,K]
            attention=attn_logits,            # [B,M,K]
            attn_weights=attn_weights,          # [B,M,K]
            eff_weights=eff_w,                   # [B,M,K]
            prior=prior,                         # [B,M]
            bag_tau_prior=bag_tau_prior,         # [B,1] or None
            bag_tau_hetero=bag_tau_hetero,       # [B,1] or None
            H_bag=H_bag,                         # [B,1] or None
            mu=mu_stack,                         # [B,K,D]
            head_usage=frac,                     # [B,K]
            assign_entropy=assign_entropy,       # scalar
            balance_mse=balance_mse,             # scalar
            diversity_cos2=diversity_cos2,       # scalar
            top_idx=top_idx_all,                 # list of [B,k_top] or None
            bot_idx=bot_idx_all,                 # list of [B,k_bot] or None
            base_taus=self.base_taus.detach().cpu(),
            p_hat=p_hat,                         # [B,1] or None
            kept_counts=kept_counts,             # [B,K] or None
        )
        return slide_stack, log

    # =================== Public APIs ===================
    def forward_attention(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, attn_only: bool = True):
        H0 = self.patch_embed(h)                                  # [B,M,D]
        slide_stack, log = self._phenotype_pool(H0, mask=attn_mask)
        if attn_only:
            return log["eff_weights"].transpose(1, 2)             # [B,K,M]
        return H0, log["eff_weights"].transpose(1, 2)

    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        H0 = self.patch_embed(h)                                   # [B,M,D]
        Z_stack, log = self._phenotype_pool(H0, mask=attn_mask)    # [B,*]
        z = self.project(Z_stack)                                  # [B, proj_out_dim]
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
    Same regularizer as earlier (assignment entropy, head balance, diversity),
    plus optional spatial coherence on router assignments.
    """
    reg = (
        lambda_assign   * log["assign_entropy"]
      + lambda_balance  * log["balance_mse"]
      + lambda_diversity* log["diversity_cos2"]
    )
    if lambda_spatial > 0.0 and edges is not None:
        src, dst = edges  # [E], [E]
        gamma = log["router_gamma"]  # [B,M,K]
        B = gamma.shape[0]
        spatial = 0.0
        for b in range(B):
            g = gamma[b]                           # [M,K]
            spatial += (g[src] - g[dst]).pow(2).sum(dim=-1).mean()
        spatial = spatial / B
        reg = reg + lambda_spatial * spatial
    return reg


if __name__ == "__main__":
    x = torch.randn(1, 5, 1024)
    model = CIPHER()
    output, log_dict = model.forward_features(x, return_attention=True)
    print(output.shape)
    print(log_dict['attention'].shape)