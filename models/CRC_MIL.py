import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL

class CRC_ABMIL(MIL):
    """
    Phenotype-Gated, Interface-Aware ABMIL (PGIA) for CRC.
    - Adds a tiny phenotype probe and an invasive-front prior to ABMIL pooling.
    - Keeps aggregation simple, O(P*M), no pairwise patch ops.

    Phenotypes (default order): [tumor, stroma, lymphoid, mucin, necrosis]
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
        phenotypes: int = 5,
        tumor_idx: int = 0,      # which channel in phenotype probe is "tumor"
        use_interface_prior: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.eps = eps
        self.P = phenotypes
        self.tumor_idx = tumor_idx
        self.use_interface_prior = use_interface_prior

        # Patch embedding MLP (same spirit as ABMIL)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )

        # ABMIL attention (1 head)
        Attn = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = Attn(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)

        # Tiny phenotype probe on embedded features -> [B, M, P]
        self.pheno_probe = nn.Linear(embed_dim, self.P)

        # Slide-level gating over P phenotypes + 1 interface channel
        self.use_if_channel = use_interface_prior
        self.K = self.P + (1 if self.use_if_channel else 0)
        self.gate_head = nn.Linear(embed_dim, self.K)

        # Minimal classification head on the final slide embedding
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def _masked_normalize(self, w, mask=None, dim=-1, eps=1e-6):
        # w: [B, *, M]; mask: [B, M] with 1 valid, 0 masked
        if mask is not None:
            w = w * mask.unsqueeze(1 if w.dim()==3 else 0)
        s = w.sum(dim=dim, keepdim=True).clamp_min(eps)
        return w / s

    def forward_attention(self, h, attn_mask=None, attn_only=True):
        """
        Returns:
          if attn_only: A_base [B, 1, M] (pre-softmax)
          else: (h_emb [B,M,D], A_base [B,1,M])
        """
        h_emb = self.patch_embed(h)       # [B, M, D]
        A_base = self.global_attn(h_emb)  # [B, M, 1]
        A_base = A_base.transpose(1, 2)   # [B, 1, M]
        if attn_mask is not None:
            A_base = A_base.masked_fill((1 - attn_mask).unsqueeze(1).bool(), float('-inf'))
        return A_base if attn_only else (h_emb, A_base)

    def forward_features(self, h, attn_mask=None, return_attention=True):
        """
        Aggregates to a slide embedding using phenotype gating + interface prior.
        Returns:
          bag [B, D], log_dict with interpretable maps
        """
        B, M, _ = h.shape
        h_emb, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # [B,M,D], [B,1,M]

        # Standard ABMIL weights
        A = F.softmax(A_base, dim=-1)  # [B,1,M]
        if attn_mask is not None:
            A = A * attn_mask.unsqueeze(1)
            A = self._masked_normalize(A, attn_mask, dim=-1, eps=self.eps)

        # Phenotype soft assignment per patch
        pheno_logits = self.pheno_probe(h_emb)       # [B,M,P]
        pheno_prob   = F.softmax(pheno_logits, dim=-1)  # [B,M,P]
        pheno_prob_T = pheno_prob.transpose(1, 2)    # [B,P,M]

        # Phenotype-gated attention per channel: w_c ∝ a_i * p_{i,c}
        w_pheno = A * pheno_prob_T                   # [B,P,M] (broadcast A over P)
        if attn_mask is not None:
            w_pheno = w_pheno * attn_mask.unsqueeze(1)
        w_pheno = self._masked_normalize(w_pheno, attn_mask, dim=-1, eps=self.eps)

        # Pool per phenotype
        m_pheno = torch.bmm(w_pheno, h_emb)          # [B,P,D]

        # Optional interface channel using u_i = t_i*(1 - t_i)
        m_if = None
        if self.use_if_channel:
            t = pheno_prob[..., self.tumor_idx]      # [B,M]
            u = t * (1.0 - t)                        # peaks at boundary -> invasive front proxy
            w_if = A.squeeze(1) * u                  # [B,M]
            if attn_mask is not None:
                w_if = w_if * attn_mask
            w_if = self._masked_normalize(w_if, attn_mask, dim=-1, eps=self.eps)  # [B,M]
            m_if = torch.bmm(w_if.unsqueeze(1), h_emb).squeeze(1)                 # [B,D]

        # Concatenate phenotype bags (+ interface) then gate them with a single slide-level gate
        # Gate is computed from slide context (mean embedding)
        h_mean = h_emb.mean(dim=1)                   # [B,D]
        g = F.softmax(self.gate_head(h_mean), dim=-1)  # [B,K]

        bags = m_pheno
        if self.use_if_channel:
            bags = torch.cat([m_pheno, m_if.unsqueeze(1)], dim=1)  # [B,K,D]

        bag = (g.unsqueeze(-1) * bags).sum(dim=1)    # [B,D]
        logs = {
            "attn_logits": A_base.squeeze(1) if return_attention else None,   # [B,M]
            "attn": A.squeeze(1) if return_attention else None,               # [B,M]
            "pheno_prob": pheno_prob if return_attention else None,           # [B,M,P]
            "pheno_weights": w_pheno if return_attention else None,           # [B,P,M]
            "interface_weight": (w_if if (self.use_if_channel and return_attention) else None),  # [B,M]
            "gate": g if return_attention else None,                          # [B,K]
        }
        return bag, logs

    def forward(self, h, attn_mask=None):
        bag, logs = self.forward_features(h, attn_mask=attn_mask, return_attention=False)
        logits = self.classifier(bag)
        return logits
