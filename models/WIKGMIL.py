import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_modules import create_mlp, MIL
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation as GeoGlobalAttention

def get_act(act: str):
    if act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    # Add other activations as needed
    else:
        raise NotImplementedError


# --- 2. Core WIKG-MIL Model ---
class WIKGMIL(MIL):
    def __init__(self, in_dim: int = 1024, embed_dim: int = 512, num_classes: int = 2, agg_type: str = 'bi-interaction',
                 pool: str = 'attn', dropout: float = 0.25, act: str = 'leaky_relu', topk: int = 6, **kwargs):
        """
        Initializes the WIKGMIL model.

        Args:
            in_dim (int): Input dimension of node features.
            embed_dim (int): Embedding dimension for node features.
            num_classes (int): Number of output classes.
            agg_type (str): Type of aggregation to use ('gcn', 'sage', or 'bi-interaction').
            pool (str): Type of pooling to use ('mean', 'max', or 'attn').
            dropout (float): Dropout rate.
            act (str): Activation function to use ('leaky_relu', 'relu', or 'tanh').
            topk (int): Number of top-k nodes to consider for attention.
            **kwargs: Additional keyword arguments.
        """
        self.agg_type = agg_type
        self.topk = topk
        self.pool = pool
        self.dropout = dropout
        self.act = act
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.agg_type = agg_type
        self.pool = pool
        self.dropout = dropout
        self.act = act

        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        for k, v in kwargs.items():
            setattr(self, k, v)

        dim_hidden = embed_dim

        # Renamed '_fc1' to 'patch_embed' for consistency
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[],
            dropout=dropout,
            out_dim=dim_hidden,
            end_with_fc=False
        )

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        # Attention mechanism layers

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)
        self.scale = dim_hidden ** -0.5

        # Aggregation layers
        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError(f"Aggregation type '{agg_type}' not supported.")

        self.activation = get_act(act)
        if self.dropout > 0:
            self.message_dropout = nn.Dropout(dropout)
        else:
            self.message_dropout = nn.Identity()

        # Pooling/Readout mechanism
        if self.pool == "mean":
            self.readout = global_mean_pool
        elif self.pool == "max":
            self.readout = global_max_pool
        elif self.pool == "attn":
            attn_net = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden // 2),
                nn.LeakyReLU(),
                nn.Linear(dim_hidden // 2, 1)
            )
            self.readout = GeoGlobalAttention(attn_net)
        else:
            raise NotImplementedError(f"Pooling type '{self.pool}' not supported.")

        self.norm = nn.LayerNorm(dim_hidden)
        self.classifier = nn.Linear(dim_hidden, self.num_classes)

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_only: bool = False, **kwargs):
        """
        Paper-style (#2) residual self-mixing + gated top-K,
        plus full-attention visualization (from #1).

        This version includes shape assertions and detailed comments for clarity.
        
        Args:
            h (torch.Tensor): Input tensor of shape [B, N, D_in]
            attn_only (bool): If True, returns only the full attention matrix for visualization.
        
        Returns:
            if attn_only: full_attn [B,N,N]
            else: (embedding [B,N,D_out], full_attn [B,N,N])
        """
        # Validate input tensor shape
        assert h.dim() == 3, f"Input tensor h must be 3-dimensional [B, N, D], but got shape {h.shape}"
        B, N, _ = h.shape

        h = self.patch_embed(h)  # Shape: [B, N, D]
        h = (h + h.mean(dim=1, keepdim=True)) * 0.5

        # Project to attention dimension (A)
        e_h = self.W_head(h)      # Shape: [B, N, A]
        e_t = self.W_tail(h)      # Shape: [B, N, A]

        # Calculate raw attention logits
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # Shape: [B, N, N]

        # Robustly select top-K, handling cases where N < self.topk
        K = min(self.topk, N)
        if K <= 0: # Handle edge case where there are no nodes to select from
            # Return a zero embedding of the correct shape to prevent downstream errors
            embedding_dim = self.linear.out_features if hasattr(self, 'linear') else h.size(-1)
            embedding = torch.zeros(B, N, embedding_dim, device=h.device)
            full_attn = torch.zeros(B, N, N, device=h.device)
            return embedding, full_attn
            
        topk_weight, topk_index = torch.topk(attn_logit, k=K, dim=-1)  # Shapes: [B, N, K]

        # Dense attention for logging/visualization (computationally isolated)
        with torch.no_grad():
            full_attn = torch.full_like(attn_logit, float("-inf"))
            full_attn.scatter_(dim=-1, index=topk_index, src=topk_weight)
            full_attn = F.softmax(full_attn, dim=-1)

        if attn_only:
            return full_attn

        # --- Start of Gated Knowledge Attention Mechanism ---

        # Gather neighbor features based on top-K indices
        batch_idx = torch.arange(B, device=h.device).view(B, 1, 1)
        # Advanced indexing to get the top-K tail embeddings for each head
        Nb_h = e_t[batch_idx, topk_index]  # Shape: [B, N, K, A]

        # Calculate edge embeddings (residual self-mix) based on Paper Eq. 4
        topk_prob = F.softmax(topk_weight, dim=-1)              # Shape: [B, N, K]
        e_h_exp = e_h.unsqueeze(2).expand(-1, -1, K, -1)     # Shape: [B, N, K, A]
        eh_r = topk_prob.unsqueeze(-1) * Nb_h + (1.0 - topk_prob).unsqueeze(-1) * e_h_exp # Shape: [B, N, K, A]

        # Gating mechanism based on Paper Eq. 6
        gate = torch.tanh(e_h_exp + eh_r)                    # Shape: [B, N, K, A]

        # Calculate knowledge-aware weights via batch dot product (Paper Eq. 6)
        ka_weight = torch.einsum('bnkd,bnkd->bnk', Nb_h, gate) # Shape: [B, N, K]
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(2)     # Shape: [B, N, 1, K]

        # Aggregate neighbor information based on new weights (Paper Eq. 5)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(2)           # Shape: [B, N, A]

        # --- Best Practice Note ---
        if e_Nh.size(-1) != h.size(-1):
            if not hasattr(self, "_tail_to_embed"):
                self._tail_to_embed = nn.Linear(e_Nh.size(-1), h.size(-1), bias=False).to(h.device)
            e_Nh = self._tail_to_embed(e_Nh)

        if e_h.size(-1) != h.size(-1):
            if not hasattr(self, "_head_to_embed"):
                self._head_to_embed = nn.Linear(e_h.size(-1), h.size(-1), bias=False).to(h.device)
            e_h_for_agg = self._head_to_embed(e_h)
        else:
            e_h_for_agg = e_h

        # Final aggregation of head and aggregated neighbor features (Paper Eq. 8)
        if self.agg_type == 'gcn':
            embedding = self.activation(self.linear(e_h_for_agg + e_Nh))
        elif self.agg_type == 'sage':
            embedding = self.activation(self.linear(torch.cat([e_h_for_agg, e_Nh], dim=-1)))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h_for_agg + e_Nh))
            bi_embedding  = self.activation(self.linear2(e_h_for_agg * e_Nh))
            embedding = sum_embedding + bi_embedding
        else:
            raise ValueError(f"Unknown agg_type: {self.agg_type}") # More informative error

        return embedding, full_attn # Shapes: [B, N, D_out], [B, N, N]

    
    def forward_features(self, h: torch.Tensor, return_attention=True, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Extracts slide-level features from input tensor using attention-based aggregation.

        Args:
            h (torch.Tensor): Input features of shape (batch, nodes, features).
            attn_mask (torch.Tensor, optional): Optional attention mask (not used in this implementation).

        Returns:
            Tuple[torch.Tensor, Dict]:
                - h_norm: Normalized slide-level feature tensor.
                - dict: Dictionary containing raw attention weights under key 'attention'.
        """
        h, A_raw = self.forward_attention(h, attn_only=False)
        h = self.message_dropout(h)

        # Squeeze batch dimension for torch_geometric pooling functions
        h_pool = self.readout(h.squeeze(0))

        h_norm = self.norm(h_pool)
        return h_norm, {'attention': A_raw}

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask: torch.Tensor = None,
                return_attention: bool = False,
                return_slide_feats: bool = False,
                ) -> torch.Tensor:
        """
        Forward pass for the WIKGMIL model.

        Args:
            h (torch.Tensor): Input features of shape (batch, nodes, features).
            loss_fn (nn.Module, optional): Loss function to compute classification loss.
            label (torch.LongTensor, optional): Ground truth labels.
            attn_mask (optional): Optional attention mask.
            return_attention (bool, optional): If True, return attention weights in log_dict.
            return_slide_feats (bool, optional): If True, return slide-level features in log_dict.

        Returns:
            Tuple[Dict, Dict]:
                - results_dict: Dictionary containing logits and loss.
                - log_dict: Dictionary containing attention weights, loss, and optionally slide features.
        """
        h, log_dict = self.forward_features(h, attn_mask=attn_mask)
        logits = self.forward_head(h)

        cls_loss = self.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1

        if not return_attention:
            del log_dict['attention']
        if return_slide_feats:
            log_dict['slide_feats'] = h

        return results_dict, log_dict

if __name__ == "__main__":
    x = torch.randn(1, 8, 1024)  # Example input tensor
    model = WIKGMIL(in_dim=1024, embed_dim=512, num_classes=2, agg_type='bi-interaction', pool='attn', dropout=0.1)
    output, ld = model.forward_features(x, return_attention=True)
    print(output.shape)
    print(ld['attention'].shape)