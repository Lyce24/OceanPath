"""
Shared building blocks for MIL models.

Contains reusable components that are composed by ABMIL, TransMIL, etc.
Kept separate from model classes to avoid circular imports and enable
independent testing.
"""

import torch
import torch.nn as nn

# ── MLP factory ───────────────────────────────────────────────────────────────


def create_mlp(
    in_dim: int = 768,
    hid_dims: list[int] | None = None,
    out_dim: int = 512,
    act: nn.Module | None = None,
    dropout: float = 0.0,
    end_with_fc: bool = True,
    bias: bool = True,
) -> nn.Module:
    """
    Build a sequential MLP with configurable depth, activation, and dropout.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dims : list[int]
        Hidden layer dimensions. Empty list = single linear layer.
    out_dim : int
        Output dimension.
    act : nn.Module
        Activation function (default: ReLU).
    dropout : float
        Dropout rate after each hidden layer.
    end_with_fc : bool
        If True, the last layer is a bare Linear (no activation).
        If False, activation is appended after the final Linear.
    bias : bool
        Whether Linear layers have bias.

    Returns
    -------
    nn.Sequential
    """
    if hid_dims is None:
        hid_dims = []
    if act is None:
        act = nn.ReLU()

    layers: list[nn.Module] = []
    current_dim = in_dim

    for hid_dim in hid_dims:
        layers.append(nn.Linear(current_dim, hid_dim, bias=bias))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hid_dim

    layers.append(nn.Linear(current_dim, out_dim, bias=bias))
    if not end_with_fc:
        layers.append(act)

    return nn.Sequential(*layers)


# ── Attention modules ─────────────────────────────────────────────────────────


class GlobalAttention(nn.Module):
    """
    Standard attention network (2 FC layers): Linear → Tanh → Dropout → Linear.

    Maps each instance embedding to a scalar attention logit.

    Parameters
    ----------
    L : int
        Input feature dimension.
    D : int
        Hidden attention dimension.
    dropout : float
        Dropout rate.
    num_classes : int
        Number of attention heads (typically 1).
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, L] → [B, N, num_classes]"""
        return self.module(x)


class GlobalGatedAttention(nn.Module):
    """
    Gated attention network (3 FC layers): two parallel branches (Tanh + Sigmoid)
    element-wise multiplied, then projected to attention logits.

    Parameters
    ----------
    L : int
        Input feature dimension.
    D : int
        Hidden attention dimension.
    dropout : float
        Dropout rate.
    num_classes : int
        Number of attention heads (typically 1).
    """

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, L] → [B, N, num_classes]"""
        a = self.attention_a(x)
        b = self.attention_b(x)
        return self.attention_c(a * b)
