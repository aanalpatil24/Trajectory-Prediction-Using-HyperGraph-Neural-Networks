"""
HyperGraph Neural Network Convolution Layers.

Implements higher-order message passing where hyperedges can connect
arbitrary numbers of nodes, enabling group-level interaction modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean
from typing import Optional


class HypergraphConv(MessagePassing):
    """
    Hypergraph Convolutional Layer.

    Unlike standard GNNs where edges connect exactly 2 nodes, hyperedges
    connect arbitrary sets of nodes. This enables modeling of multi-agent
    group dynamics that pairwise graphs cannot capture.

    Message passing flow:
        1. Nodes -> Hyperedges (aggregation)
        2. Hyperedges -> Nodes (distribution)

    Reference:
        Feng et al. "Hypergraph Neural Networks" (AAAI 2019)
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_heads: int = 4
    ):
        """
        Initialize Hypergraph Convolution.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            attention_heads: Number of attention heads (if use_attention=True)
        """
        super().__init__(aggr="add", flow="source_to_target")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Feature transformation layers
        self.node_transform = nn.Linear(in_channels, out_channels)
        self.hyperedge_transform = nn.Linear(out_channels, out_channels)

        if use_attention:
            # Multi-head attention for weighted aggregation
            self.attention = nn.MultiheadAttention(
                out_channels, 
                num_heads=attention_heads, 
                dropout=dropout,
                batch_first=True
            )
            self.attention_proj = nn.Linear(out_channels * 2, 1)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.activation = nn.LeakyReLU(0.2)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.node_transform.weight)
        nn.init.xavier_uniform_(self.hyperedge_transform.weight)
        if self.use_attention:
            nn.init.xavier_uniform_(self.attention_proj.weight)

    def forward(
        self, 
        x: torch.Tensor, 
        hyperedge_index: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of hypergraph convolution.

        Args:
            x: Node features (num_nodes, in_channels)
            hyperedge_index: (2, num_connections) - [node_idx, hyperedge_idx]
            hyperedge_weight: (num_hyperedges,) - importance weights

        Returns:
            out: Updated node features (num_nodes, out_channels)
        """
        num_nodes = x.size(0)

        # Transform node features
        x_transformed = self.node_transform(x)

        # Step 1: Aggregate node features to hyperedges
        hyperedge_features = self._nodes_to_hyperedges(
            x_transformed, hyperedge_index, num_nodes
        )

        # Transform hyperedge features
        hyperedge_features = self.hyperedge_transform(hyperedge_features)

        # Apply hyperedge weights if provided
        if hyperedge_weight is not None:
            hyperedge_features = hyperedge_features * hyperedge_weight.unsqueeze(1)

        # Step 2: Distribute hyperedge features back to nodes
        out = self._hyperedges_to_nodes(
            hyperedge_features, hyperedge_index, num_nodes
        )

        # Residual connection and normalization
        out = self.layer_norm(out + x_transformed)
        out = self.activation(out)
        out = self.dropout(out)

        return out

    def _nodes_to_hyperedges(
        self, 
        x: torch.Tensor, 
        hyperedge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate node features to hyperedge centers.

        Args:
            x: (num_nodes, out_channels)
            hyperedge_index: (2, num_connections)
            num_nodes: Total number of nodes

        Returns:
            hyperedge_features: (num_hyperedges, out_channels)
        """
        node_idx, hyperedge_idx = hyperedge_index

        # Aggregate using scatter_add
        num_hyperedges = hyperedge_idx.max().item() + 1
        hyperedge_features = scatter_add(
            x[node_idx], 
            hyperedge_idx, 
            dim=0, 
            dim_size=num_hyperedges
        )

        # Average pooling (divide by hyperedge size)
        hyperedge_size = scatter_add(
            torch.ones_like(hyperedge_idx, dtype=torch.float),
            hyperedge_idx,
            dim=0,
            dim_size=num_hyperedges
        ).unsqueeze(1)

        hyperedge_features = hyperedge_features / (hyperedge_size + 1e-8)

        return hyperedge_features

    def _hyperedges_to_nodes(
        self, 
        hyperedge_features: torch.Tensor, 
        hyperedge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Distribute hyperedge features to member nodes.

        Args:
            hyperedge_features: (num_hyperedges, out_channels)
            hyperedge_index: (2, num_connections)
            num_nodes: Total number of nodes

        Returns:
            node_features: (num_nodes, out_channels)
        """
        node_idx, hyperedge_idx = hyperedge_index

        # Gather hyperedge features for each node
        node_features = hyperedge_features[hyperedge_idx]

        # Aggregate (mean) if node belongs to multiple hyperedges
        out = scatter_mean(
            node_features,
            node_idx,
            dim=0,
            dim_size=num_nodes
        )

        return out


class MultiLayerHGNN(nn.Module):
    """
    Multi-layer HyperGraph Neural Network stack.

    Stacks multiple HypergraphConv layers with residual connections
    for deep feature extraction.
    """

    def __init__(
        self, 
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialize Multi-layer HGNN.

        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of HGNN layers
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()

        self.num_layers = num_layers
        self.use_residual = use_residual

        self.layers = nn.ModuleList([
            HypergraphConv(
                hidden_dim,
                hidden_dim,
                dropout=dropout,
                use_attention=False
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(
        self, 
        x: torch.Tensor, 
        hyperedge_index: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multiple HGNN layers with residual connections.

        Args:
            x: (num_nodes, hidden_dim)
            hyperedge_index: (2, num_connections)
            hyperedge_weight: (num_hyperedges,)

        Returns:
            x: Updated features (num_nodes, hidden_dim)
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            residual = x

            x = layer(x, hyperedge_index, hyperedge_weight)
            x = norm(x)

            # Residual connection (except first layer)
            if self.use_residual and i > 0:
                x = x + residual

        return x
