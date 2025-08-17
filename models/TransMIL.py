import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
import math

from models.base_modules import create_mlp, GlobalAttention, GlobalGatedAttention, MIL

class TransLayer(nn.Module):
    def __init__(self, norm_layer: nn.Module = nn.LayerNorm, dim: int = 512, num_heads: int = 8):
        """
        Transformer Layer with Nystrom Attention.

        Args:
            norm_layer (nn.Module): Normalization layer, default is nn.LayerNorm.
            dim (int): Dimension for the transformer layer, default is 512.
        """
        super().__init__()
        self.norm = norm_layer(dim)
        self.attention = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and normalization.
        """
        z = self.attention(self.norm(x))
        x = x + z  # Residual connection
        return x

class PPEG(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, padding=7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, padding=5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, padding=3//2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, 1 + H*W, D]
        B, _, D = x.shape
        cls_tok, feat_tok = x[:, :1, :], x[:, 1:, :]     # [B,1,D], [B,H*W,D]
        feat_map = feat_tok.transpose(1,2).view(B, D, H, W)
        y = self.proj(feat_map) + feat_map \
          + self.proj1(feat_map) + self.proj2(feat_map)
        y = y.flatten(2).transpose(1,2)                  # [B, H*W, D]
        return torch.cat([cls_tok, y], dim=1)            # [B, 1+H*W, D]
    
class TransMIL(MIL):
    def __init__(self, in_dim: int, embed_dim: int,
                 num_fc_layers: int, dropout: float,
                 num_attention_layers: int, num_classes: int, num_heads: int=8):
        """
        TransMIL model with transformer-based Multi-instance Learning.

        Args:
            in_dim (int): Input dimension for the MLP.
            embed_dim (int): Embedding dimension for all layers.
            n_fc_layers (int): Number of fully connected layers in the MLP.
            dropout (float): Dropout rate for MLP.
            n_attention_layers (int): Number of transformer attention layers.
            n_classes (int): Number of output classes for classification.
        """
        super(TransMIL, self).__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed: nn.Module = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        self.pos_layer: nn.Module = PPEG(dim=embed_dim)
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.blocks: nn.ModuleList = nn.ModuleList(
            [TransLayer(dim=embed_dim, num_heads=num_heads) for _ in range(num_attention_layers)]
        )

        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def forward_features(self, h: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Get slide-level features from cls token.

        Args:
            h (torch.Tensor): The input tensor of shape (features, dim) or
                              (batch_size, features, dim).

        Returns:
            torch.Tensor: Slide-level feature of cls token. Output shape will be
                          (1, 1, embed_dim) if input was 2D or
                          (batch_size, 1, embed_dim) if input was 3D.
        """
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        h = self.patch_embed(h)
        B, N, D = h.shape
        
        # padding
        Hs = int(math.ceil(math.sqrt(N)))
        Ws = Hs
        pad = Hs*Ws - N
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)  # [B, Hs*Ws, D]
        
        # cls token
        cls = self.cls_token.expand(B, -1, -1).to(h.device)  # [B,1,D]
        h   = torch.cat([cls, h], dim=1)                      # [B, 1+Hs*Ws, D]

        # apply trans layers + capture attn from first block
        attn_dict = {"attention": None}
        for i, blk in enumerate(self.blocks):
            h = blk(h)  # [B,1+Hs*Ws,D]
            if i == 0 and return_attention:
                # dot‐prod CLS vs tokens
                cls_tok = h[:, :1, :]            # [B,1,D]
                feats   = h[:, 1:, :]            # [B,Hs*Ws,D]
                full_attn = (feats @ cls_tok.transpose(-1,-2)).squeeze(-1)
                # slice off padding → [B,N]
                attn_dict['attention'] = full_attn[:, :N]

            # positional encoding for next layer
            h = self.pos_layer(h, Hs, Ws)
        wsi_feats = self.norm(h)[:, 0, :]    # [B, D]
        return wsi_feats, attn_dict

if __name__ == "__main__":
    # Example usage
    x = torch.randn(8, 100, 1024)
    model = TransMIL(in_dim=1024, embed_dim=512, num_fc_layers=1, dropout=0.1,
                     num_attention_layers=4, num_classes=10)
    out, attn = model.forward_features(x, return_attention=True)
    print(out.shape)
    print(attn['attention'].shape)