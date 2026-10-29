"""
Mean/Max pooling baseline MIL aggregator.

No learnable attention — just pools patch embeddings with mean or max.
Useful as a sanity-check baseline.
"""

import torch

from oceanpath.models.base import BaseMIL
from oceanpath.models.components import create_mlp


class StaticMIL(BaseMIL):
    """
    Simple pooling aggregator: MLP → mean/max pool → slide embedding.

    Parameters
    ----------
    in_dim : int
        Input patch feature dimension.
    embed_dim : int
        Embedding dimension after MLP.
    num_fc_layers : int
        FC layers in patch embedding MLP.
    dropout : float
        Dropout rate.
    pool_method : str
        'mean' or 'max'.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        pool_method: str = "mean",
        **kwargs,
    ):
        super().__init__(
            in_dim=in_dim,
            embed_dim=embed_dim,
            gradient_checkpointing=False,
        )
        assert pool_method in ("mean", "max"), (
            f"pool_method must be 'mean' or 'max', got '{pool_method}'"
        )
        self.pool_method = pool_method

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False,
        )

        self.initialize_weights()

    def forward_features(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Pool patch features into slide embedding.

        Returns
        -------
        slide_embedding : [B, E]
        extras : empty dict (no attention for pooling baselines)
        """
        h = self.patch_embed(h)  # [B, N, E]

        if mask is not None:
            # Zero out padded positions before pooling
            h = h * mask.unsqueeze(-1)

        if self.pool_method == "mean":
            if mask is not None:
                # Correct mean by dividing by actual valid count
                valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
                slide_embedding = h.sum(dim=1) / valid_counts  # [B, E]
            else:
                slide_embedding = h.mean(dim=1)  # [B, E]
        else:  # max
            if mask is not None:
                h = h.masked_fill((1 - mask).unsqueeze(-1).bool(), float("-inf"))
            slide_embedding = h.max(dim=1).values  # [B, E]

        return slide_embedding, {}
