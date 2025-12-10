import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_modules import create_mlp, MIL

class StaticMIL(MIL):
    """
    Static MIL model that aggregates features using a fixed method (mean or max).
    
    Args:
        method (str): Aggregation method, either 'Mean' or 'Max'.
    """
    
    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            num_classes: int = 2,
            method: str = 'Mean'
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        if method not in ['Mean', 'Max']:
            raise ValueError(f"Unsupported aggregation method: {method}")
        self.method = method
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )
        
    def forward_features(self, h, return_attention = False):
        if len(h.shape) == 2:
            h = h.unsqueeze(0)

        h = self.patch_embed(h)

        if self.method == 'Mean':
            h = h.mean(dim=1)
        elif self.method == 'Max':
            h, _ = h.max(dim=1)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")
        return h, {'attention': None}
    
    def forward_head(self):
        pass
    
    def forward_attention(self) -> torch.Tensor:
        pass
    
    def forward(self) -> tuple:
        pass
