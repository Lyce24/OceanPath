import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL

class CHAMP(MIL):
    """
    CHAMP: Cancer Heterogeneity-Aware MIL Pooling.

    Adds three elements atop ABMIL:
      (1) Prior-modulated attention via a learned scalar "canceriness" per instance.
      (2) Dynamic softmax temperature based on estimated tumor fraction in the bag.
      (3) Heterogeneity-aware pooling: concat of weighted mean, weighted std, and
          top-vs-bottom quantile feature contrast, then a tiny compressor MLP.

    Args (notable):
        target_prevalence (float): typical positive area fraction (e.g., 0.05~0.15).
        base_tau (float): base attention temperature (1.0 = standard softmax).
        gamma (float): strength of prior modulation (>=0).
        top_frac, bottom_frac (float): quantile fractions for contrast feature.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        attn_dim: int = 384,
        gate: bool = True,
        num_classes: int = 2,
        # CHAMP extras
        target_prevalence: float = 0.10,
        base_tau: float = 1.0,
        gamma: float = 1.0,
        top_frac: float = 0.10,
        bottom_frac: float = 0.10,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        # Patch embedding (same spirit as ABMIL)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (max(num_fc_layers - 1, 0)),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )

        # Base (gated) attention to get logits s
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)

        # Prior: scalar "canceriness" per instance
        self.prior_head = nn.Linear(embed_dim, 1)

        # Compressor for heterogeneity-aware pooled vector [mu | std | delta]
        self.compressor = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Simple classifier head
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Hyperparams
        self.register_buffer("eps", torch.tensor(1e-6), persistent=False)
        self.base_tau = base_tau
        self.target_prev = target_prevalence
        self.gamma = gamma
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.use_quantile = True

        self.initialize_weights()

    # ---------- Utilities ----------

    def _compute_temperature(self, prior_probs: torch.Tensor) -> torch.Tensor:
        """
        prior_probs: [B, M] (sigmoid outputs).
        Returns tau per bag: [B, 1]
        """
        mean_prev = prior_probs.mean(dim=1, keepdim=True)  # [B,1]
        # Smaller mean_prev -> sharper softmax (tau down)
        ratio = mean_prev / (self.target_prev + float(self.eps))
        tau = self.base_tau * torch.clamp(ratio, 0.5, 2.0)
        return tau  # [B,1]

    @staticmethod
    def _safe_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return F.softmax(logits, dim=dim)

    # ---------- Attention ----------

    def forward_attention(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None, return_all=False):
        """
        h: [B, M, D] raw instance features.
        attn_mask: [B, M] with 1 valid, 0 masked (optional).
        return_all: if True, also return prior and tau for logging.

        Returns:
            A (weights) [B, M], optional dict with 'logits', 'prior', 'tau'
        """
        H = self.patch_embed(h)                     # [B, M, D]
        s = self.global_attn(H).squeeze(-1)        # [B, M] (attention logits)
        prior = torch.sigmoid(self.prior_head(H)).squeeze(-1)  # [B, M]

        if attn_mask is not None:
            s = s.masked_fill(~attn_mask.bool(), float("-inf"))
            prior = torch.where(attn_mask.bool(), prior, torch.zeros_like(prior))

        tau = self._compute_temperature(prior)     # [B,1]

        # Product-of-experts style: softmax(s/tau) * prior^gamma
        s_scaled = s / torch.clamp(tau, min=1e-6)
        weights = self._safe_softmax(s_scaled, None, dim=-1)   # [B, M]
        if self.gamma != 0.0:
            weights = weights * torch.clamp(prior, min=1e-6).pow(self.gamma)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        if return_all:
            return weights, {"logits": s, "prior": prior, "tau": tau, "H": H}
        return weights

    # ---------- Heterogeneity-aware pooling ----------
    def _weighted_stats(
        self,
        H: torch.Tensor,                 # [B,M,D]
        w: torch.Tensor,                 # [B,M] (sum=1 where valid)
        top_frac: float,
        bot_frac: float,
        use_quantile: bool,
    ):
        """
        H: [B, M, D], A: [B, M] (sum to 1)
        Returns:
            mu: [B, D], std: [B, D], delta_q: [B, D]
        """
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
    
    # ---------- Public features/logits APIs ----------

    def forward_features(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None, return_attention: bool = True):
        """
        Returns:
            bag_embed: [B, D]
            log_dict:  dict with 'attention', 'attention_logits', 'prior', 'tau', 'top_idx', 'bot_idx'
        """
        A, aux = self.forward_attention(h, attn_mask=attn_mask, return_all=True)  # A: [B,M]
        H = aux.pop("H")  # [B,M,D]

        mu, std, delta_q, idx_dict = self._weighted_stats(H, A, self.top_frac, self.bottom_frac, self.use_quantile)
        z = torch.cat([mu, std, delta_q], dim=-1)             # [B, 3D]
        bag_embed = self.compressor(z)                         # [B, D]

        log_dict = {
            "attention": A if return_attention else None,
            "attention_logits": aux.get("logits", None) if return_attention else None,
            "prior": aux.get("prior", None) if return_attention else None,
            "tau": aux.get("tau", None),
            "top_idx": idx_dict["top_idx"] if return_attention else None,
            "bot_idx": idx_dict["bot_idx"] if return_attention else None,
        }
        return bag_embed, log_dict

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None):
        bag_embed, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=True)
        logits = self.classifier(bag_embed)
        return logits, log_dict


# ---- quick sanity check ----
if __name__ == "__main__":
    x = torch.randn(2, 137, 1024)     # B, M, D_in
    model = CHAMP()
    with torch.no_grad():
        logits, logs = model(x)
    print("logits:", logits.shape)              # [2, num_classes]
    be, _ = model.forward_features(x, return_attention=True)
    print("bag embedding:", be.shape)           # [2, embed_dim]
