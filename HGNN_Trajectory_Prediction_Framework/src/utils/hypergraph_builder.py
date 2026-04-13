"""
Hypergraph construction using DBSCAN clustering.

Transforms pairwise pedestrian interactions into higher-order group
relationships, enabling collective spatial reasoning.
"""
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class HypergraphConstructor:
    """
    Dynamic hypergraph construction for multi-agent systems.

    Uses DBSCAN to detect spatial clusters and converts them to hyperedges,
    allowing arbitrary-order interactions (not limited to pairs).
    """

    def __init__(
        self, 
        eps: float = 2.0, 
        min_samples: int = 2,
        metric: str = "euclidean"
    ):
        """
        Initialize Hypergraph Constructor.

        Args:
            eps: Maximum distance between agents to be considered neighbors (meters)
            min_samples: Minimum agents to form a group (hyperedge)
            metric: Distance metric for clustering
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )

    def construct_hypergraph(
        self, 
        positions: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        return_metadata: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Construct hypergraph from agent positions.

        The algorithm:
        1. Apply DBSCAN to detect spatial clusters (groups)
        2. Each cluster becomes a hyperedge connecting all members
        3. Isolated agents become singleton hyperedges
        4. Compute hyperedge weights based on group cohesion

        Args:
            positions: (num_agents, 2) - current (x, y) positions
            features: Optional (num_agents, feature_dim) for feature-based clustering
            return_metadata: Whether to return construction metadata

        Returns:
            hyperedge_index: (2, num_connections) - [node_idx, hyperedge_idx]
            hyperedge_weight: (num_hyperedges,) - importance weights
            metadata: Dictionary with clustering statistics
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Convert to numpy for DBSCAN
        pos_np = positions.cpu().numpy()

        # Optional: concatenate features if provided
        if features is not None:
            features_np = features.cpu().numpy()
            # Normalize features to match position scale
            features_np = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
            pos_np = np.concatenate([pos_np, features_np], axis=1)

        # Perform DBSCAN clustering
        clusters = self.dbscan.fit_predict(pos_np)

        # Build hyperedges from clusters
        hyperedges = []
        unique_clusters = set(clusters)

        # Remove noise points (-1) temporarily
        if -1 in unique_clusters:
            unique_clusters.remove(-1)

        # Create hyperedge for each valid cluster (group)
        for cluster_id in unique_clusters:
            members = np.where(clusters == cluster_id)[0]
            if len(members) >= self.min_samples:
                hyperedges.append(members.tolist())

        # Handle isolated agents (noise points)
        isolated = np.where(clusters == -1)[0]
        for agent_idx in isolated:
            hyperedges.append([agent_idx])

        # Convert to PyTorch Geometric hyperedge_index format
        # Format: [node_indices, hyperedge_indices]
        node_indices = []
        hyperedge_indices = []

        for he_idx, members in enumerate(hyperedges):
            for node_idx in members:
                node_indices.append(node_idx)
                hyperedge_indices.append(he_idx)

        hyperedge_index = torch.tensor(
            [node_indices, hyperedge_indices], 
            dtype=torch.long,
            device=device
        )

        # Compute hyperedge weights
        hyperedge_weight = self._compute_hyperedge_weights(
            positions, hyperedges, hyperedge_index, device
        )

        metadata = None
        if return_metadata:
            metadata = {
                "num_hyperedges": len(hyperedges),
                "num_agents": num_agents,
                "clusters": clusters,
                "hyperedges": hyperedges,
                "group_sizes": [len(he) for he in hyperedges],
                "avg_group_size": np.mean([len(he) for he in hyperedges]),
                "max_group_size": max([len(he) for he in hyperedges]) if hyperedges else 0,
                "num_isolated": len(isolated),
                "isolated_indices": isolated.tolist()
            }

        return hyperedge_index, hyperedge_weight, metadata

    def _compute_hyperedge_weights(
        self, 
        positions: torch.Tensor,
        hyperedges: List[List[int]],
        hyperedge_index: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute importance weights for each hyperedge based on spatial cohesion.

        Strategy:
        - Tighter groups (lower spatial variance) get higher weights
        - Singleton hyperedges get neutral weight (1.0)

        Args:
            positions: (num_agents, 2)
            hyperedges: List of hyperedge member lists
            hyperedge_index: (2, num_connections)
            device: torch device

        Returns:
            weights: (num_hyperedges,)
        """
        num_hyperedges = len(hyperedges)
        weights = torch.ones(num_hyperedges, device=device)

        for he_idx, members in enumerate(hyperedges):
            if len(members) > 1:
                member_positions = positions[members]
                # Inverse of spatial variance as weight
                variance = torch.var(member_positions, dim=0).mean()
                # Cohesion score: 1 / (1 + variance)
                weights[he_idx] = 1.0 / (1.0 + variance)

        # Normalize: sum of weights equals number of hyperedges
        weights = weights / (weights.sum() + 1e-8) * num_hyperedges

        return weights

    def construct_batch_hypergraphs(
        self,
        positions_batch: torch.Tensor,
        features_batch: Optional[torch.Tensor] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Construct hypergraphs for a batch of scenes.

        Args:
            positions_batch: (batch_size, num_agents, 2)
            features_batch: Optional (batch_size, num_agents, feature_dim)

        Returns:
            List of (hyperedge_index, hyperedge_weight, metadata) for each scene
        """
        batch_size = positions_batch.shape[0]
        results = []

        for b in range(batch_size):
            positions = positions_batch[b]
            features = features_batch[b] if features_batch is not None else None

            hyperedge_index, hyperedge_weight, metadata = self.construct_hypergraph(
                positions, features, return_metadata=True
            )
            results.append((hyperedge_index, hyperedge_weight, metadata))

        return results
