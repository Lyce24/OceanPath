import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Union
from abc import ABC, abstractmethod
from nystrom_attention import NystromAttention

import torch.nn as nn
import math
from topk.svm import SmoothTop1SVM 

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
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

class ABMIL(MIL):
    """
    ABMIL (Attention-based Multiple Instance Learning) model.

    This class implements the core ABMIL architecture, which uses a patch embedding MLP,
    followed by a global attention or gated attention mechanism, and an optional classification head.

    Args:
        in_dim (int): Input feature dimension for each instance (default: 1024).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        gate (int): Whether to use gated attention (True) or standard attention (False) (default: True).
        num_classes (int): Number of output classes for the classification head (default: 2).
    """

    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            num_classes: int = 2,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        """
        Compute the attention scores (and optionally the embedded features) for the input instances.

        Args:
            h (torch.Tensor): Input tensor of shape [B, M, D], where B is the batch size,
                M is the number of instances (patches), and D is the input feature dimension.
            attn_mask (torch.Tensor, optional): Optional attention mask of shape [B, M], where 1 indicates
                valid positions and 0 indicates masked positions. If provided, masked positions are set to
                a very large negative value before softmax.
            attn_only (bool, optional): If True, return only the attention scores (A).
                If False, return a tuple (h, A) where h is the embedded features and A is the attention scores.

        Returns:
            torch.Tensor: If attn_only is True, returns the attention scores tensor of shape [B, K, M],
                where K is the number of attention heads (usually 1). If attn_only is False, returns a tuple
                (h, A) where h is the embedded features of shape [B, M, D'] and A is the attention scores.
        """
        h = self.patch_embed(h)
        A = self.global_attn(h)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base if return_attention else None}
        return h, log_dict

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
        x = x + self.attention(self.norm(x))
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

    def forward_attention(self, h: torch.Tensor) -> torch.Tensor:
        pass

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
        
        Hs = int(math.ceil(math.sqrt(N)))
        Ws = Hs
        pad = Hs*Ws - N
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)  # [B, Hs*Ws, D]
        cls = self.cls_token.expand(B, -1, -1).to(h.device)  # [B,1,D]
        h   = torch.cat([cls, h], dim=1)                      # [B, 1+Hs*Ws, D]

        # 4) apply trans layers + capture attn from first block
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

class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K, return_attention=False):
        """
        Args:
            Q: (B, S_Q, D_Q)
            K: (B, S_K, D_K)
        Returns:
            O: (B, S_Q, D_V) - output after attention
            A: (B, S_Q, S_K) - attention scores
        """

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)
        A, attention_weights = self.multihead_attention(Q, K, V,
                                                        need_weights=return_attention,
                                                        average_attn_weights=True)  # A is shaped S_Q, B, D_V
        attention_weights = attention_weights.transpose(0, 1) if attention_weights is not None else None

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O, attention_weights

class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H, _ = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat, _ = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat

class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, return_attention=False):
        global_embedding = self.S.repeat(X.size(0), 1, 1)  # expand to batch dim
        ret, attention = self.mha(global_embedding, X, return_attention=return_attention)  # cross attention scores
        if return_attention:
            attention = torch.sum(attention, dim=1)  # B x patches
        return ret, attention

class ILRA(MIL):
    def __init__(self, in_dim, embed_dim, num_heads, 
                 topk, num_attention_layers, num_classes, ln=True, mode='classification'):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.mode = mode
        topk = topk

        self.mlp = None

        gab_blocks = []
        for idx in range(num_attention_layers):
            block = GAB(dim_in=in_dim if idx == 0 else embed_dim,
                        dim_out=embed_dim,
                        num_heads=num_heads,
                        num_inds=topk,
                        ln=ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=embed_dim, num_heads=num_heads, ln=ln)

        self.initialize_weights()

    def forward_features(self, x, return_attention=False):
        for block in self.gab_blocks:
            x = block(x)
        slide_feat, attention = self.forward_attention(x, return_attention=return_attention)
        return slide_feat, {'attention': attention if return_attention else None}

    def forward_attention(self, x, return_attention):
        slide_feat, attention = self.pooling(x, return_attention)
        slide_feat = slide_feat.squeeze(1)
        return slide_feat, attention

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

    def __init__(self, in_dim: int, attn_dim: int = 384, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor, attn_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        device = h.device
        V = self.v(h)  # B x M x D
        Q = self.q(h)  # B x M x D_attn

        # Sort instances by class scores to find critical instances
        _, m_indices = torch.sort(c, dim=1, descending=True)

        # Select features of top instances for each class
        m_feats = torch.stack(
            [torch.index_select(h_i, dim=0, index=m_indices_i[0, :]) for h_i, m_indices_i in zip(h, m_indices)], 0
        )

        q_max = self.q(m_feats)  # B x C x D_attn
        # Attention mechanism: I think this could be the error?
        A = torch.bmm(Q, q_max.transpose(1, 2))  # B x M x C
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=2) * torch.finfo(A.dtype).min

        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=device)),
                      dim=1)  # Softmax over M # Shape: B x M x C

        # Aggregate features

        B = torch.bmm(A.transpose(1, 2), V)  # B x C x D

        B = self.norm(B)

        A = A.mean(dim=2)  # Aggregate attention scores over classes

        return B, A

class DSMIL(MIL):
    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            dropout_v: float = 0.0,
            num_classes: int = 2
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
        self.b_classifier = BClassifier(in_dim=embed_dim, attn_dim=attn_dim, dropout=dropout_v)
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

class CLAMSB(MIL):
    """
    Clustering-constrained Attention Multiple Instance Learning (CLAM) - Single-Branch version.
    This model integrates instance-level and bag-level learning through an attention mechanism
    and instance classifiers.
    """
    def __init__(self,
                 in_dim: int = 1024,  # Input dimension of each instance feature
                 embed_dim: int = 512,  # Dimension after instance embedding MLP
                 n_fc_layers: int = 1,  # Number of layers in the instance embedding MLP
                 dropout: float = 0.25,  # Dropout rate for MLP and attention
                 gate: bool = True,  # Whether to use gated attention
                 attention_dim: int = 384,  # Dimension for the attention network
                 num_classes: int = 2,  # Number of output classes for bag-level prediction
                 k_sample: int = 8,  # Number of top/bottom patches to sample for instance-level training
                 subtyping: bool = False,  # Whether this is a subtyping problem (affects inst_eval_out)
                 instance_loss_fn: str = 'svm',  # Loss function type for instance-level prediction ('ce' or 'svm')
                 bag_weight: float = 0.7):  # Weight for bag-level loss vs. instance-level loss

        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.k_sample = k_sample
        self.subtyping = subtyping
        self.bag_weight = bag_weight

        # Instance Embedding MLP: Transforms input features to 'embed_dim'
        self.patch_embed = create_mlp(in_dim=in_dim,
                              hid_dims=[embed_dim] * (n_fc_layers - 1),
                              dropout=dropout,
                              out_dim=embed_dim,
                              end_with_fc=False)

        # Attention Network: Calculates attention scores for instances
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(L=embed_dim,
                                       D=attention_dim,
                                       dropout=dropout,
                                       num_classes=1)  # Single attention head

        # Bag-level Classifier: Maps aggregated features to class logits
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Instance Classifiers: One classifier per class for instance-level prediction
        instance_classifiers = [nn.Linear(embed_dim, 2) for _ in range(num_classes)]  # Binary classifier per class
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.instance_loss_fn = SmoothTop1SVM(n_classes=2).cuda()

        self.initialize_weights()

    @staticmethod
    def create_positive_targets(length: int, device: torch.device) -> torch.LongTensor:
        """Helper to create a tensor of positive (1) labels."""
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length: int, device: torch.device) -> torch.LongTensor:
        """Helper to create a tensor of negative (0) labels."""
        return torch.full((length,), 0, device=device).long()

    def _check_inputs(self, features, loss_fn, label):
        if features.dim() == 3 and features.shape[0] > 1:
            raise ValueError(f'CLAM does not currently support batch size > 1')
        if label is None:
            raise ValueError(f'Label must be provided for CLAM')
        if loss_fn is None:
            raise ValueError("Loss function must be provided")

    def inst_eval(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Instance-level evaluation for in-the-class attention branch (positive instances).

        Args:
            A (torch.Tensor): Attention scores (N,).
            h (torch.Tensor): Instance features (N, D).
            classifier (nn.Module): Instance-level classifier.

        Returns:
            tuple: (instance_loss, predictions, targets)
        """
        # Ensure A is 2D for torch.topk consistency
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # Select top-k positive instances based on attention scores
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        # Select top-k negative instances (lowest attention)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        # Create targets for selected instances
        p_targets = self.create_positive_targets(self.k_sample, h.device)
        n_targets = self.create_negative_targets(self.k_sample, h.device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)

        # Concatenate selected instances and get logits
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)

        # Get predictions and compute loss
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Instance-level evaluation for out-of-the-class attention branch (negative instances).
        Used primarily in subtyping problems.

        Args:
            A (torch.Tensor): Attention scores (N,).
            h (torch.Tensor): Instance features (N, D).
            classifier (nn.Module): Instance-level classifier.

        Returns:
            tuple: (instance_loss, predictions, targets)
        """
        # Ensure A is 2D
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # Select top-k instances, which are treated as negative for out-of-class evaluation
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        # Targets are negative for these instances
        p_targets = self.create_negative_targets(self.k_sample, h.device)

        # Get logits and compute loss
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward_attention(self, h: torch.Tensor, attention_only: bool = False) -> Union[torch.Tensor, tuple[
        torch.Tensor, torch.Tensor]]:
        """
        Processes instance features through the instance embedding and attention network.

        Args:
            h (torch.Tensor): Input instance features of shape [B, M, D].
                              (Note: CLAM typically processes one bag at a time, so B=1 initially).
            attention_only (bool): If True, returns only attention scores.

        Returns:
            torch.Tensor: Attention scores [1, M] if attention_only=True.
                          Or (h_embedded [M, E], A [1, M]) if attention_only=False.
        """
        # Apply instance embedding
        # h should be [M, D] after squeezing batch dimension if input is [1, M, D]
        h = self.patch_embed(h.squeeze(0))  # Squeeze batch dimension for CLAM's attention logic if input is [1, M, D]

        # Compute attention scores
        A = self.global_attn(h)  # Output A: [M x 1] (or M x K if multiple heads)
        A = torch.transpose(A, 1, 0)  # Transpose to [1 x M] for consistency with CLAM's original logic

        if attention_only:
            return A
        else:
            return h, A  # h (embedded instances) [M, E], A (unnormalized attention) [1, M]

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Applies the bag-level classification head.

        Args:
            h (torch.Tensor): Aggregated bag-level features [B, E].

        Returns:
            logits (torch.Tensor): Raw classification logits [B, num_classes].
        """
        logits = self.classifier(h)
        return logits

    
    def forward_features(self, h: torch.Tensor, return_attention: bool = True) -> tuple[
        torch.Tensor, dict]:
        """
        Aggregates instance features into a single bag-level feature vector using attention.

        Args:
            h (torch.Tensor): Input instance features of shape [B, M, D].
            return_attention (bool): returns attention scores always
        Returns:
            tuple:
                - M (torch.Tensor): Aggregated bag-level features [B, E].
                - log_dict (dict): Dict storing 'attention' and 'instance_feats'.
                instance_feats (torch.Tensor): Embedded instance features [M, E] (if return_attention=True). attention (torch.Tensor): Unnormalized attention scores [1, M] (if return_attention=True).
        """
        # Get embedded instance features and unnormalized attention scores
        h_embedded, attention = self.forward_attention(h)  # h_embedded: [M, E], attention: [1, M]
        log_dict = {'instance_feats': h_embedded}
        log_dict['attention'] = attention

        # Apply softmax to attention scores for weighting
        attention_scaled = F.softmax(attention, dim=-1)  # Softmax over instances (M)

        M = torch.mm(attention_scaled, h_embedded)  # Aggregated bag feature [1, E]


        return M, log_dict  # M [B, E], h_embedded [M, E], attention [1, M]


    def forward_instance_heads(self, h: torch.Tensor, attention_scores: torch.Tensor,
                               label: torch.LongTensor = None) -> Union[torch.Tensor, None]:
        """
        Computes instance-level loss.

        Args:
            h (torch.Tensor): Embedded instance features [M, E].
            attention_scores (torch.Tensor): Unnormalized attention scores [1, M].
            label (torch.LongTensor, optional): True label for the bag [B].

        Returns:
            torch.Tensor | None: Total instance loss, or None if label is not provided.
        """
        if label is None:
            return None
        total_inst_loss = 0.0
        # Convert scalar label to one-hot for instance classifier selection
        # Ensure the one-hot tensor is on the same device as the input label
        inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze(0).to(label.device)

        for i in range(len(self.instance_classifiers)):
            inst_label_for_class = inst_labels[i].item()  # Get 0 or 1 for current class 'i'
            classifier = self.instance_classifiers[i]

            if inst_label_for_class == 1:  # If the bag belongs to class 'i'
                instance_loss, _, _ = self.inst_eval(attention_scores, h, classifier)
                total_inst_loss += instance_loss
            else:  # If the bag does not belong to class 'i'
                if self.subtyping:  # Only evaluate out-of-class instances for subtyping problems
                    instance_loss, _, _ = self.inst_eval_out(attention_scores, h, classifier)
                    total_inst_loss += instance_loss
                # Else: do nothing if not subtyping and bag is not of this class

        # Average loss only if subtyping or if total_inst_loss was incremented
        if self.subtyping and len(self.instance_classifiers) > 0:
            total_inst_loss /= len(self.instance_classifiers)
        elif not self.subtyping and inst_labels.sum().item() > 0:  # If there was at least one positive class
            total_inst_loss /= inst_labels.sum().item()  # Average over positive classes only
        elif total_inst_loss == 0 and inst_labels.sum().item() == 0:  # No positive classes, and not subtyping
            return None  # No instance loss to compute

        return total_inst_loss

    def compute_total_loss(self, logits: torch.Tensor, label: torch.LongTensor, loss_fn: nn.Module,
                           inst_loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined bag-level and instance-level loss.

        Args:
            logits (torch.Tensor): Bag-level raw output scores.
            label (torch.LongTensor): True bag-level label.
            loss_fn (nn.Module): Bag-level loss function.
            inst_loss (torch.Tensor): Computed instance-level loss.

        Returns:
            torch.Tensor: The total combined loss.
        """
        cls_loss = self.compute_loss(loss_fn, logits, label)
        if inst_loss is not None:
            # Combine bag-level and instance-level losses with a predefined weight
            loss = cls_loss * self.bag_weight + (1 - self.bag_weight) * inst_loss
        else:
            loss = cls_loss
        return loss

    def forward(self, h: torch.Tensor, label: torch.LongTensor = None,
                loss_fn: nn.Module = None, attn_mask = None, return_attention: bool = True,
                return_slide_feats: bool = None
                ) -> tuple[dict, dict]:
        """
        Full forward pass through the CLAMSB model.

        Args:
            h (torch.Tensor): Input instance features of shape [B, M, D].
            label (torch.LongTensor, optional): True label for the bag [B].
            loss_fn (nn.Module, optional): The bag-level loss function.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - results_dict (dict): Dictionary with 'logits' and 'loss'
                - log_dict (dict): Dictionary with 'instance_loss', 'cls_loss', 'loss'.
        """
        self._check_inputs(h, loss_fn, label)
        # Get bag-level features, embedded instance features, and unnormalized attention scores
        # slide_feats: [B, E], instance_feats: [M, E], attention_scores: [1, M] (assuming B=1)
        slide_feats, intermeds = self.forward_features(h, return_attention=return_attention)

        # Get bag-level classification logits
        logits = self.forward_head(slide_feats)

        # Compute instance-level loss
        inst_loss = self.forward_instance_heads(intermeds['instance_feats'], intermeds['attention'], label)

        # Compute total combined loss
        total_loss = self.compute_total_loss(logits, label, loss_fn, inst_loss)

        # Prepare log dictionary
        log_dict = {
            'instance_loss': inst_loss.item() if inst_loss is not None else -1,
            'cls_loss': loss_fn(logits, label).item() if loss_fn is not None and label is not None else -1,
            'loss': total_loss.item()
        }

        results_dict = {'logits': logits, 'loss': total_loss}
        if return_attention:
            log_dict['attention'] = intermeds['attention']
        if return_slide_feats:
            log_dict['slide_feats'] = slide_feats


        return results_dict, log_dict

class WIKGMIL(MIL):
    def __init__(self, 
                 in_dim=384, 
                 embed_dim=512, 
                 topk=6, 
                 num_classes=2, 
                 agg_type='bi-interaction', 
                 dropout=0.3, 
                 pool='attn'):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self._fc1 = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.LeakyReLU())

        self.W_head = nn.Linear(embed_dim, embed_dim)
        self.W_tail = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(embed_dim, embed_dim // 2)
        self.gate_V = nn.Linear(embed_dim, embed_dim // 2)
        self.gate_W = nn.Linear(embed_dim // 2, embed_dim)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(embed_dim, embed_dim)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(embed_dim * 2, embed_dim)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(embed_dim, embed_dim)
            self.linear2 = nn.Linear(embed_dim, embed_dim)
        else:
            raise NotImplementedError
        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embed_dim)

        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net=nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.LeakyReLU(), nn.Linear(embed_dim//2, 1))     
            self.readout = GeoGlobalAttention(att_net)


    def forward_features(self, x, return_attention: bool = False):
        x = self._fc1(x)    # [B,N,C]

        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # Create a full attention matrix for visualization/logging
        if return_attention:
            with torch.no_grad():
                full_attn = torch.full_like(attn_logit, float('-inf'))
                full_attn.scatter_(dim=-1, index=topk_index, src=topk_weight)
                full_attn = F.softmax(full_attn, dim=-1)  # [B, N, N]
                # mean over attn
                full_attn = full_attn.mean(dim=1)  # [B, N]

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        h = self.message_dropout(embedding)

        h = self.readout(h.squeeze(0))
        h = self.norm(h)
        
        return h, {'attention': full_attn if return_attention else None}

class StaticMIL(MIL):
    """
    Static MIL model that aggregates features using a fixed method (mean or max).
    
    Args:
        method (str): Aggregation method, either 'Mean' or 'Max'.
    """
    
    def __init__(self, method: str = 'Mean'):
        super().__init__(in_dim=0, embed_dim=0, num_classes=0)
        if method not in ['Mean', 'Max']:
            raise ValueError(f"Unsupported aggregation method: {method}")
        self.method = method
        
    def forward_features(self, h, return_attention = False):
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        
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

class WSIModel(nn.Module):
    def __init__(self, 
                 input_feature_dim,
                 encoder_type='ABMIL',
                 num_fc_layers=1, 
                 head_dim=512,
                 head_dropout=0.25, 
                 ds_dropout=0.2,
                 hidden_dim=128, 
                 simple_mlp=False, 
                 n_classes=1,
                 encoder_attrs = {},
                 freeze_encoder=False):
        super().__init__()

        self.encoder_type = encoder_type
        if encoder_type == 'ABMIL':
            self.feature_encoder = ABMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_fc_layers=num_fc_layers,
                dropout=head_dropout,
                attn_dim=encoder_attrs.get("attn_dim", 384),
                gate=encoder_attrs.get("gate", True),
                num_classes=0
            )
        elif encoder_type == 'TransMIL':
            self.feature_encoder = TransMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_fc_layers=num_fc_layers,
                dropout=head_dropout,
                num_attention_layers=encoder_attrs.get("num_attention_layers", 2),
                num_classes=0,
                num_heads=encoder_attrs.get("num_heads", 4)
            )
        elif encoder_type == 'ILRA':
            self.feature_encoder = ILRA(
                in_dim=input_feature_dim,
                embed_dim=encoder_attrs.get("embed_dim", head_dim),
                num_heads=encoder_attrs.get("num_heads", 8),
                topk=encoder_attrs.get("topk", 64),
                num_attention_layers=encoder_attrs.get("num_attention_layers", 2),
                num_classes=0,
                ln=encoder_attrs.get("ln", True),
                mode=encoder_attrs.get("mode", 'classification')
            )
        elif encoder_type == 'CLAM':
            self.feature_encoder = CLAMSB(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                n_fc_layers=num_fc_layers,
                dropout=head_dropout,
                gate=encoder_attrs.get("gate", True),
                attention_dim=encoder_attrs.get("attn_dim", 384),
                num_classes=n_classes,
                k_sample=encoder_attrs.get("k_sample", 8),
                subtyping=encoder_attrs.get("subtyping", False),
                instance_loss_fn=encoder_attrs.get("instance_loss_fn", 'svm'),
                bag_weight=encoder_attrs.get("bag_weight", 0.7)
            )
            # Instance Classifiers: One classifier per class for instance-level prediction
            instance_classifiers = [nn.Linear(head_dim, 2) for _ in range(n_classes)]  # Binary classifier per class
            self.instance_classifiers = nn.ModuleList(instance_classifiers)

        elif encoder_type == 'DSMIL':
            self.feature_encoder = DSMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_fc_layers=num_fc_layers,
                dropout=head_dropout,
                attn_dim=encoder_attrs.get("attn_dim", 384),
                dropout_v=encoder_attrs.get("dropout_v", 0.0),
                num_classes=n_classes
            )
        elif encoder_type == 'WIKGMIL':
            self.feature_encoder = WIKGMIL(
                in_dim=input_feature_dim,
                embed_dim=head_dim,
                num_classes=n_classes,
                agg_type=encoder_attrs.get("agg_type", 'bi-interaction'),
                pool=encoder_attrs.get("pool", 'attn'),
                dropout=head_dropout,
                topk=encoder_attrs.get("topk", 4)
            )
        elif encoder_type == 'Mean' or encoder_type == 'Max':
            self.feature_encoder = StaticMIL(
                method=encoder_type,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.feature_encoder.parameters():
                param.requires_grad = False

        # one head per task
        self.simple_mlp = simple_mlp
        self.n_classes = n_classes

        if self.simple_mlp:
            self.heads = create_mlp(
                in_dim=head_dim if encoder_type not in ['Mean', 'Max'] else input_feature_dim,
                hid_dims=[hidden_dim],
                dropout=ds_dropout,
                out_dim=n_classes,
                end_with_fc=True,
                end_with_dropout=False,
                bias=True
            )
        else:
            self.heads = nn.Linear(head_dim, n_classes) if encoder_type not in ['Mean', 'Max'] else nn.Linear(input_feature_dim, n_classes)
        
        if self.encoder_type == "DSMIL":
            self.heads = nn.Conv1d(n_classes, n_classes, kernel_size=head_dim)
        
        self.initialize_weights()
    
    def forward(self, x, return_raw_attention=False, labels=None):
        features, log_dict = self.feature_encoder.forward_features(x, return_attention=return_raw_attention)

        logits = self.heads(features)  # [B, n_classes] or [B, 1]
        
        if self.encoder_type == "DSMIL" and 'instance_classes' in log_dict:
            logits = logits.squeeze(dim=-1)  # [B, n_classes] if Conv1d head
            max_instance_logits, _ = torch.max(log_dict['instance_classes'], 1)
            logits = 0.5 * (logits + max_instance_logits)
            
        if self.encoder_type == "CLAM" and "instance_feats" in log_dict and labels is not None and "attention" in log_dict:
            instance_loss = self.feature_encoder.forward_instance_heads(log_dict["instance_feats"], log_dict['attention'], labels)
            log_dict['instance_loss'] = instance_loss if instance_loss is not None else -1

        if self.n_classes == 1:
            logits = logits.squeeze(dim=-1)

        return logits, log_dict

    def initialize_weights(self):
        if not self.freeze_encoder:
            self.feature_encoder.initialize_weights()
            
        for layer in self.heads.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
            elif isinstance(layer, nn.Conv1d):
                layer.reset_parameters()
                    
if __name__ == "__main__":
        # import dataset and dataloader for training testing
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast
    from torch.amp import GradScaler
    
    # Example usage
    for encoder_type in ['ABMIL', 'TransMIL', 'DSMIL', 'WIKGMIL', 'CLAM', 'ILRA', 'Mean', 'Max']:
        print(f"Testing encoder type: {encoder_type}")
        model = WSIModel(input_feature_dim=1024, encoder_type=encoder_type, n_classes=2)

        # Create dummy data
        x = torch.randn(100, 100, 1024)
        
        # convert to float16
        x = x.half()  # Convert to float16 for training

        y = torch.randint(0, 2, (100,))  # Binary classification
        
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        use_amp = True  # Use AMP if available
        scaler = GradScaler(enabled=use_amp)
        
        # Training loop
        model.train()
        for epoch in range(5):  # 5 epochs
            for i, batch in enumerate(dataloader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                opt.zero_grad()
                with autocast(device_type=device.type, enabled=use_amp):
                    if model.encoder_type == "CLAM":
                        outputs, log_dict = model(inputs, return_raw_attention=True, labels=labels)
                    else:
                        outputs, log_dict = model(inputs, return_raw_attention=True)
                        
                    if epoch == 0 and i == 0:
                        print(f"Outputs shape: {outputs.shape}, Attention shape: {log_dict['attention'].shape if log_dict['attention'] is not None else 'N/A'}")

            loss = loss_fn(outputs, labels)
            if model.encoder_type == "CLAM" and 'instance_loss' in log_dict and log_dict['instance_loss'] != -1:
                loss = 0.7 * loss + 0.3 * log_dict['instance_loss']
            
            scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            
        model.initialize_weights()  # Reinitialize weights after training