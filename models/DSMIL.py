import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_modules import create_mlp, MIL


class IClassifier(nn.Module):
    """Instance-level classifier."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.inst_classifier = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B x M x D]
        c = self.inst_classifier(h)  # B x M x C
        return c

class BClassifier(nn.Module):
    """Bag-level classifier with attention."""

    def __init__(self, in_dim: int, attn_dim: int = 384, layernorm = True, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
        )
        
        self.layernorm = layernorm
        if layernorm:
            self.norm = nn.LayerNorm(in_dim)

    # h: [B x M x D], c: [B x M x C]
    def forward(self, h: torch.Tensor, c: torch.Tensor, attn_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, D = h.shape
        device = h.device
        V = self.v(h)  # B x M x D
        Q = self.q(h)  # B x M x D_attn
        
        # Sort instances by class scores to find critical instances
        top_idx = torch.argmax(c, dim=1)              # B x C

        # Select features of top instances for each class
        b_idx = torch.arange(B, device=device)[:, None]    # B x 1
        m_feats = h[b_idx, top_idx, :]                 # B x C x D
        
        q_max = self.q(m_feats)  # B x C x D_attn
        
        # Q: [B x M x D_attn], q_max: [B x C x D_attn]
        # q_max.transpose(-1, -2) -> [B x D_attn x C]
        # Attention score: (B x M x D_attn) x (B x D_attn x C) -> (B x M x C)
        A = torch.bmm(Q, q_max.transpose(-1, -2))  # B x M x C
        
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=2) * torch.finfo(A.dtype).min

        A = F.softmax(A / (Q.size(-1) ** 0.5), dim=1)            # softmax over instances N

        # Aggregate features
        # Bag representation: (B x C x N) @ (B x N x V) -> B x C x V
        B_bag = torch.matmul(A.transpose(1, 2), V)         # B x C x V

        if self.layernorm:
            B_bag = self.norm(B_bag)

        # A = A.mean(dim=2)  # Aggregate attention scores over classes

        return B_bag, A

class DSMIL(MIL):
    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            dropout_v: float = 0.0,
            num_classes: int = 2,
            layernorm: bool = True,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False
        )
        self.i_classifier = IClassifier(in_dim=embed_dim, num_classes=num_classes)
        self.b_classifier = BClassifier(in_dim=embed_dim, attn_dim=attn_dim, dropout=dropout_v, layernorm=layernorm)
        self.initialize_weights()

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = False) -> tuple[
        torch.Tensor, dict]:
        h = self.patch_embed(h)
        instance_classes = self.i_classifier(h)
        slide_feats, attention = self.b_classifier(h, instance_classes, attn_mask=attn_mask)
        # attention = attention.mean(dim=2)
        intermeds = {'instance_classes': instance_classes}
        if return_attention:
            intermeds['attention'] = attention if return_attention else None
                
        return slide_feats, intermeds
    
    
if __name__ == "__main__":
    x = torch.randn(1, 8, 1024)  # Example input tensor
    model = DSMIL(in_dim=1024, embed_dim=512, num_fc_layers=1, dropout=0.1,
                  attn_dim=384, dropout_v=0.0, num_classes=2, layernorm=True)
    output, ld = model.forward_features(x, return_attention=True)
    print(output.shape)
    print(ld['attention'].shape)