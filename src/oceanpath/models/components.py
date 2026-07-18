"""Reusable neural-network components shared by the supported MIL models."""

from typing import cast

import torch
from torch import nn


def create_mlp(
    in_dim: int = 768,
    hid_dims: list[int] | None = None,
    out_dim: int = 512,
    act: nn.Module | None = None,
    dropout: float = 0.0,
    end_with_fc: bool = True,
    bias: bool = True,
) -> nn.Module:
    """Build a sequential MLP with configurable depth and dropout."""
    hid_dims = hid_dims or []
    act = act or nn.ReLU()
    layers: list[nn.Module] = []
    current_dim = in_dim

    for hid_dim in hid_dims:
        layers.extend((nn.Linear(current_dim, hid_dim, bias=bias), act))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hid_dim

    layers.append(nn.Linear(current_dim, out_dim, bias=bias))
    if not end_with_fc:
        layers.append(act)
    return nn.Sequential(*layers)


class GlobalAttention(nn.Module):
    """Map every instance embedding to an attention logit."""

    def __init__(self, L: int = 1024, D: int = 256, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``[B, N, L]`` embeddings into ``[B, N, K]`` logits."""
        return cast(torch.Tensor, self.module(x))


class GlobalGatedAttention(nn.Module):
    """Apply gated attention using parallel tanh and sigmoid branches."""

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
        """Transform ``[B, N, L]`` embeddings into ``[B, N, K]`` logits."""
        return cast(
            torch.Tensor,
            self.attention_c(self.attention_a(x) * self.attention_b(x)),
        )
