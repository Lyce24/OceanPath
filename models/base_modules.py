import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch_geometric.nn.aggr import AttentionalAggregation as GeoGlobalAttention

def create_mlp(
        in_dim=768, 
        hid_dims=[512, 512], 
        out_dim=512, 
        act=nn.ReLU(),
        dropout=0.,
        end_with_fc=True, 
        end_with_dropout=False,
        bias=True
    ):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        if end_with_dropout:
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp

class GlobalAttention(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        num_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x)  # N x num_classes

class GlobalGatedAttention(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        num_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()

        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)
        ]

        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        ]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x num_classes
        return A

class MIL(ABC, nn.Module):
    """
    Abstract base class for MIL (Multiple Instance Learning) models.
    Defines the core forward pass methods that MIL implementations must provide.
    """

    def __init__(self,
                 in_dim: int,
                 embed_dim: int,
                 num_classes: int,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward_features(self, h: torch.Tensor, return_attention: bool=False) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patch features using attention into slide-level features.

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.
            return_attention: bool indicating whether to return attention scores in intermed dict.

        Returns:
            h: [B x D]-dim torch.Tensor, the aggregated bag-level feature.
            intermeds: dict containing intermediate results (optional, can be extended by concrete implementations).
        """
        pass

    @staticmethod
    def ensure_batched(tensor: torch.Tensor, return_was_unbatched: bool = False) -> torch.Tensor:
        """
        Ensure the tensor is batched (has a batch dimension).

        Args:
            tensor: A torch.Tensor that may or may not have a batch dimension.

        Returns:
            A batched torch.Tensor.
        """
        was_unbatched = False
        while len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
            was_unbatched = True
        if return_was_unbatched:
            return tensor, was_unbatched
        return tensor

    @staticmethod
    def ensure_unbatched(tensor: torch.Tensor, return_was_batched: bool = False) -> torch.Tensor:
        """
        Ensure the tensor is unbatched (removes the batch dimension if present).

        Args:
            tensor: A torch.Tensor that may or may not have a batch dimension.

        Returns:
            An unbatched torch.Tensor.
        """
        was_batched = True
        while len(tensor.shape) > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            was_batched = False
        if return_was_batched:
            return tensor, was_batched
        return tensor

    @staticmethod
    def compute_loss(loss_fn: nn.Module,
                     logits: torch.Tensor,
                     label: torch.LongTensor) -> torch.Tensor:
        """
        Compute the loss using the provided loss function.

        Args:
            loss_fn: A callable loss function.
            logits: The model's output logits.
            label: The ground truth labels.

        Returns:
            A scalar tensor representing the computed loss.
        """
        if loss_fn is None or logits is None:
            return None
        return loss_fn(logits, label)

    def initialize_weights(self):
        """
        Initialize the weights of the model with kaiming he for linear layers, and xavier for all others
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
            elif isinstance(layer, nn.Conv2d):
                layer.reset_parameters()
            elif isinstance(layer, nn.LayerNorm):
                layer.reset_parameters()
            elif isinstance(layer, nn.BatchNorm1d):
                layer.reset_parameters()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()
            # PyTorch's built‐in multi‐head attention
            elif isinstance(layer, nn.MultiheadAttention):
                layer._reset_parameters()  # Reset parameters for multi-head attention
            # GlobalAttention from torch_geometric
            elif isinstance(layer, GeoGlobalAttention):
                # GlobalAttention defines reset_parameters() internally
                layer.reset_parameters()
