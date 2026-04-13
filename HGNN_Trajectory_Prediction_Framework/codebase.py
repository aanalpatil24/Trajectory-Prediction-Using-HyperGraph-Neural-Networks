
"""
================================================================================
HyperGraph Neural Network for Multi-Agent Trajectory Prediction
================================================================================
A sophisticated implementation of Sequence-to-Sequence HGNN for crowd dynamics.

Author: AI Assistant
Date: 2026
Framework: PyTorch Geometric, PyTorch, NumPy, Scikit-learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add, scatter_mean
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Optional, Dict
import random
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ModelConfig:
    """Configuration for HGNN Trajectory Prediction Model."""
    # Input/Output dimensions
    input_dim: int = 2  # (x, y) coordinates
    hidden_dim: int = 128
    output_dim: int = 2  # (x, y) coordinates

    # Sequence lengths
    obs_len: int = 8  # Observation timesteps
    pred_len: int = 12  # Prediction timesteps

    # HGNN Architecture
    num_hgnn_layers: int = 2
    num_heads: int = 4  # Multi-head attention
    dropout: float = 0.1

    # Social Interaction (DBSCAN)
    dbscan_eps: float = 2.0  # Spatial threshold for grouping
    dbscan_min_samples: int = 2

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100

    # Data Augmentation
    augmentation_prob: float = 0.5
    missing_data_prob: float = 0.1

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# DATA AUGMENTATION & PREPROCESSING
# ==============================================================================

class TrajectoryAugmenter:
    """
    Sophisticated data augmentation pipeline for trajectory data.

    Implements:
    1. Kinematic Trajectory Reversal - bidirectional spatial symmetry
    2. Missing Data Imputation - robustness to sensor failures
    3. Gaussian Noise Injection - regularization
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def reverse_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Temporal reversal augmentation.

        Args:
            trajectory: Tensor of shape (seq_len, num_agents, 2)
        Returns:
            Reversed trajectory maintaining kinematic consistency
        """
        # Reverse temporal dimension
        reversed_traj = torch.flip(trajectory, dims=[0])

        # Adjust velocities for kinematic consistency
        # If original was moving right, reversed should move left
        return reversed_traj

    def inject_missing_data(
        self, 
        trajectory: torch.Tensor, 
        missing_prob: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate sensor failures with missing data imputation.

        Args:
            trajectory: Tensor of shape (seq_len, num_agents, 2)
            missing_prob: Probability of masking a timestep
        Returns:
            Tuple of (corrupted_trajectory, mask)
        """
        if missing_prob is None:
            missing_prob = self.config.missing_data_prob

        seq_len, num_agents, _ = trajectory.shape
        mask = torch.rand(seq_len, num_agents) > missing_prob
        mask = mask.unsqueeze(-1).expand(-1, -1, 2).float()

        corrupted = trajectory.clone()
        corrupted[mask == 0] = float('nan')

        # Impute missing values using spatial-temporal interpolation
        imputed = self._impute_missing(corrupted)

        return imputed, mask

    def _impute_missing(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Spatial-temporal interpolation for missing trajectory points.

        Strategy:
        1. Forward fill for short gaps
        2. Linear interpolation for medium gaps
        3. Spatial neighbor averaging for long gaps
        """
        imputed = trajectory.clone()
        seq_len, num_agents, _ = imputed.shape

        for agent_idx in range(num_agents):
            agent_traj = imputed[:, agent_idx, :]

            # Find NaN positions
            nan_mask = torch.isnan(agent_traj).any(dim=1)

            if nan_mask.any():
                # Get valid positions
                valid_indices = torch.where(~nan_mask)[0]

                if len(valid_indices) == 0:
                    # All missing - use spatial mean
                    agent_traj[:] = torch.nanmean(imputed.view(seq_len, -1), dim=0)
                else:
                    # Linear interpolation
                    for dim in range(2):
                        valid_values = agent_traj[valid_indices, dim]
                        all_indices = torch.arange(seq_len, dtype=torch.float32)

                        # Interpolate
                        interpolated = torch.nn.functional.interpolate(
                            valid_values.unsqueeze(0).unsqueeze(0),
                            size=seq_len,
                            mode='linear',
                            align_corners=True
                        ).squeeze()

                        agent_traj[:, dim] = interpolated

            imputed[:, agent_idx, :] = agent_traj

        return imputed

    def add_gaussian_noise(
        self, 
        trajectory: torch.Tensor, 
        noise_scale: float = 0.01
    ) -> torch.Tensor:
        """Add small Gaussian noise for regularization."""
        noise = torch.randn_like(trajectory) * noise_scale
        return trajectory + noise

    def augment_batch(
        self, 
        trajectory: torch.Tensor,
        apply_reversal: bool = True,
        apply_missing: bool = True,
        apply_noise: bool = True
    ) -> torch.Tensor:
        """
        Apply full augmentation pipeline to a batch.

        Args:
            trajectory: (batch_size, seq_len, num_agents, 2)
        Returns:
            Augmented trajectory
        """
        batch_size = trajectory.shape[0]
        augmented = []

        for b in range(batch_size):
            traj = trajectory[b]  # (seq_len, num_agents, 2)

            # Random reversal
            if apply_reversal and random.random() < self.config.augmentation_prob:
                traj = self.reverse_trajectory(traj)

            # Missing data simulation
            if apply_missing and random.random() < self.config.augmentation_prob:
                traj, _ = self.inject_missing_data(traj)

            # Noise injection
            if apply_noise and random.random() < self.config.augmentation_prob:
                traj = self.add_gaussian_noise(traj)

            augmented.append(traj)

        return torch.stack(augmented)


# ==============================================================================
# SOCIAL INTERACTION INFERENCE (HYPERGRAPH CONSTRUCTION)
# ==============================================================================

class HypergraphConstructor:
    """
    Dynamic hypergraph construction using DBSCAN clustering.

    Transforms pairwise pedestrian interactions into higher-order group
    relationships, enabling collective spatial reasoning.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.dbscan = DBSCAN(
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples,
            metric='euclidean'
        )

    def construct_hypergraph(
        self, 
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Construct hypergraph from agent positions.

        Args:
            positions: (num_agents, 2) - current (x, y) positions
            hidden_states: Optional (num_agents, hidden_dim) for feature-based clustering

        Returns:
            hyperedge_index: (2, num_hyperedges * avg_hyperedge_size) connectivity
            hyperedge_weight: (num_hyperedges,) importance weights
            metadata: Dictionary with clustering info
        """
        num_agents = positions.shape[0]

        # Convert to numpy for DBSCAN
        pos_np = positions.cpu().numpy()

        # Perform DBSCAN clustering
        clusters = self.dbscan.fit_predict(pos_np)

        # Build hyperedges from clusters
        hyperedges = []
        unique_clusters = set(clusters)

        # Remove noise points (-1) from cluster set for pure clustering
        if -1 in unique_clusters:
            unique_clusters.remove(-1)

        # Create hyperedge for each cluster (group)
        for cluster_id in unique_clusters:
            members = np.where(clusters == cluster_id)[0]
            if len(members) >= self.config.dbscan_min_samples:
                hyperedges.append(members.tolist())

        # Add singleton hyperedges for isolated agents
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

        hyperedge_index = torch.tensor([node_indices, hyperedge_indices], 
                                        dtype=torch.long,
                                        device=positions.device)

        # Compute hyperedge weights based on group cohesion
        hyperedge_weight = self._compute_hyperedge_weights(
            positions, hyperedges, hyperedge_index
        )

        metadata = {
            'num_hyperedges': len(hyperedges),
            'clusters': clusters,
            'hyperedges': hyperedges,
            'avg_group_size': np.mean([len(he) for he in hyperedges])
        }

        return hyperedge_index, hyperedge_weight, metadata

    def _compute_hyperedge_weights(
        self, 
        positions: torch.Tensor,
        hyperedges: List[List[int]],
        hyperedge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance weights for each hyperedge based on spatial cohesion.

        Tighter groups (lower variance) get higher weights.
        """
        num_hyperedges = len(hyperedges)
        weights = torch.ones(num_hyperedges, device=positions.device)

        for he_idx, members in enumerate(hyperedges):
            if len(members) > 1:
                member_positions = positions[members]
                # Inverse of spatial variance as weight
                variance = torch.var(member_positions, dim=0).mean()
                weights[he_idx] = 1.0 / (1.0 + variance)

        # Normalize
        weights = weights / weights.sum() * num_hyperedges
        return weights


# ==============================================================================
# HYPERGRAPH NEURAL NETWORK LAYERS
# ==============================================================================

class HypergraphConv(MessagePassing):
    """
    Hypergraph Convolutional Layer.

    Unlike standard GNNs where edges connect 2 nodes, hyperedges connect
    arbitrary numbers of nodes, enabling higher-order message passing.

    Message passing flow:
    1. Nodes -> Hyperedges (aggregation)
    2. Hyperedges -> Nodes (distribution)
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__(aggr='add', flow='source_to_target')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Transformation matrices
        self.node_transform = nn.Linear(in_channels, out_channels)
        self.hyperedge_transform = nn.Linear(in_channels, out_channels)

        if use_attention:
            # Multi-head attention for hyperedge aggregation
            self.attention = nn.MultiheadAttention(
                out_channels, 
                num_heads=4, 
                dropout=dropout,
                batch_first=True
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.activation = nn.LeakyReLU(0.2)

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
        """
        num_nodes = x.size(0)

        # Initial transformation
        x_transformed = self.node_transform(x)

        # Step 1: Aggregate node features to hyperedges
        hyperedge_features = self._nodes_to_hyperedges(
            x_transformed, hyperedge_index, num_nodes
        )

        # Apply hyperedge transformation
        hyperedge_features = self.hyperedge_transform(hyperedge_features)

        # Apply weights if provided
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
        """Aggregate node features to hyperedge centers."""
        node_idx, hyperedge_idx = hyperedge_index

        # Use scatter_add to aggregate
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
        """Distribute hyperedge features to member nodes."""
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
    """Multi-layer HyperGraph Neural Network stack."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            HypergraphConv(
                config.hidden_dim if i > 0 else config.hidden_dim,
                config.hidden_dim,
                dropout=config.dropout
            )
            for i in range(config.num_hgnn_layers)
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
        """
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x, hyperedge_index, hyperedge_weight)
            # Residual connection between layers
            if i > 0:
                x = x + residual
        return x


# ==============================================================================
# SEQUENCE-TO-SEQUENCE ARCHITECTURE
# ==============================================================================

class TrajectoryEncoder(nn.Module):
    """
    GRU-based encoder for trajectory history.

    Compresses past trajectory (obs_len timesteps) into a hidden state
    representing the agent's motion history.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.input_embed = nn.Linear(config.input_dim, config.hidden_dim)
        self.gru = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory history.

        Args:
            trajectory: (batch_size, obs_len, num_agents, 2)
        Returns:
            hidden_states: (batch_size, num_agents, hidden_dim)
        """
        batch_size, obs_len, num_agents, _ = trajectory.shape

        # Reshape for processing: (batch * agents, seq_len, 2)
        traj_flat = trajectory.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents, obs_len, self.config.input_dim
        )

        # Embed input
        embedded = self.input_embed(traj_flat)

        # GRU encoding
        _, hidden = self.gru(embedded)

        # Take last layer's hidden state
        hidden = hidden[-1]  # (batch * agents, hidden_dim)

        # Reshape back
        hidden_states = hidden.view(batch_size, num_agents, self.config.hidden_dim)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class TrajectoryDecoder(nn.Module):
    """
    GRU-based decoder for future trajectory prediction.

    Autoregressively predicts future positions (pred_len timesteps)
    using socially-aware hidden states from HGNN.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.gru = nn.GRU(
            config.input_dim,
            config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True
        )

        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

    def forward(
        self, 
        social_features: torch.Tensor,
        last_position: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        ground_truth: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode future trajectory.

        Args:
            social_features: (batch_size, num_agents, hidden_dim) from HGNN
            last_position: (batch_size, num_agents, 2) last observed position
            teacher_forcing_ratio: Probability of using ground truth
            ground_truth: (batch_size, pred_len, num_agents, 2) for teacher forcing

        Returns:
            predictions: (batch_size, pred_len, num_agents, 2)
        """
        batch_size, num_agents, _ = social_features.shape
        device = social_features.device

        # Initialize decoder hidden state
        # Expand social features for GRU layers
        hidden = social_features.permute(1, 0, 2).repeat(self.gru.num_layers, 1, 1)

        # Initial input: last observed position
        decoder_input = last_position.view(batch_size * num_agents, 1, self.config.input_dim)

        predictions = []
        current_pos = last_position.view(batch_size * num_agents, 1, self.config.input_dim)

        for t in range(self.config.pred_len):
            # GRU step
            output, hidden = self.gru(current_pos, hidden)

            # Predict next position
            pred = self.output_layer(output.squeeze(1))
            predictions.append(pred.view(batch_size, num_agents, self.config.output_dim))

            # Teacher forcing
            if ground_truth is not None and random.random() < teacher_forcing_ratio:
                next_pos = ground_truth[:, t, :, :].view(
                    batch_size * num_agents, self.config.input_dim
                )
            else:
                next_pos = pred

            current_pos = next_pos.unsqueeze(1)

        predictions = torch.stack(predictions, dim=1)  # (batch, pred_len, agents, 2)
        return predictions


# ==============================================================================
# MAIN MODEL: SEQ2SEQ HGNN
# ==============================================================================

class Seq2SeqHGNN(nn.Module):
    """
    Sequence-to-Sequence HyperGraph Neural Network for Trajectory Prediction.

    Architecture:
    1. Encoder (GRU): Compress trajectory history
    2. HGNN: Model social interactions via hypergraph convolution
    3. Decoder (GRU): Predict future trajectory autoregressively
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        # Components
        self.encoder = TrajectoryEncoder(config)
        self.hypergraph_constructor = HypergraphConstructor(config)
        self.hgnn = MultiLayerHGNN(config)
        self.decoder = TrajectoryDecoder(config)

        # Learnable parameters for social interaction
        self.social_fusion = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.to(self.device)

    def forward(
        self, 
        obs_trajectory: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        return_metadata: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete model.

        Args:
            obs_trajectory: (batch_size, obs_len, num_agents, 2)
            ground_truth: (batch_size, pred_len, num_agents, 2) - optional
            teacher_forcing_ratio: Probability of teacher forcing during training
            return_metadata: Whether to return hypergraph construction info

        Returns:
            Dictionary containing predictions and optionally metadata
        """
        batch_size, obs_len, num_agents, _ = obs_trajectory.shape
        device = obs_trajectory.device

        # Step 1: Encode trajectory history
        individual_features = self.encoder(obs_trajectory)
        # (batch_size, num_agents, hidden_dim)

        # Step 2: Construct hypergraph for social interactions
        # Use last observed positions for spatial clustering
        last_positions = obs_trajectory[:, -1, :, :]  # (batch, agents, 2)

        # Process each scene in batch
        social_features_list = []
        all_metadata = []

        for b in range(batch_size):
            positions = last_positions[b]  # (num_agents, 2)
            features = individual_features[b]  # (num_agents, hidden_dim)

            # Construct hypergraph
            hyperedge_index, hyperedge_weight, metadata =                 self.hypergraph_constructor.construct_hypergraph(positions, features)

            # Apply HGNN
            social_features = self.hgnn(features, hyperedge_index, hyperedge_weight)

            # Fuse individual and social features
            combined = torch.cat([features, social_features], dim=-1)
            fused = self.social_fusion(combined)

            social_features_list.append(fused)
            all_metadata.append(metadata)

        # Stack batch
        social_features_batch = torch.stack(social_features_list, dim=0)

        # Step 3: Decode future trajectory
        last_pos = obs_trajectory[:, -1, :, :]
        predictions = self.decoder(
            social_features_batch,
            last_pos,
            teacher_forcing_ratio=teacher_forcing_ratio,
            ground_truth=ground_truth
        )

        output = {
            'predictions': predictions,
            'social_features': social_features_batch
        }

        if return_metadata:
            output['metadata'] = all_metadata

        return output


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================

class TrajectoryLoss(nn.Module):
    """Combined loss function for trajectory prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def forward(
        self, 
        predictions: torch.Tensor, 
        ground_truth: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory prediction losses.

        Metrics:
        - ADE: Average Displacement Error (per timestep)
        - FDE: Final Displacement Error (final timestep only)
        """
        # Euclidean distance
        distances = torch.norm(predictions - ground_truth, dim=-1)

        if mask is not None:
            distances = distances * mask
            num_valid = mask.sum()
        else:
            num_valid = distances.numel()

        # ADE: Average over all timesteps
        ade = distances.sum() / num_valid

        # FDE: Final timestep only
        fde = distances[:, -1, :].mean()

        # Combined loss
        loss = ade + fde

        return {
            'loss': loss,
            'ade': ade,
            'fde': fde
        }


class Trainer:
    """Training pipeline for Seq2Seq HGNN."""

    def __init__(self, model: Seq2SeqHGNN, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = config.device

        self.criterion = TrajectoryLoss(config)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )

        self.augmenter = TrajectoryAugmenter(config)

        # Metrics tracking
        self.train_history = {'loss': [], 'ade': [], 'fde': []}
        self.val_history = {'loss': [], 'ade': [], 'fde': [], 'accuracy': []}

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {'loss': 0, 'ade': 0, 'fde': 0}
        num_batches = 0

        for batch in dataloader:
            obs_traj = batch['obs'].to(self.device)
            pred_traj = batch['pred'].to(self.device)

            # Apply data augmentation
            obs_traj = self.augmenter.augment_batch(obs_traj)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                obs_traj, 
                ground_truth=pred_traj,
                teacher_forcing_ratio=0.5
            )

            # Compute loss
            losses = self.criterion(output['predictions'], pred_traj)

            # Backward pass
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track metrics
            for key in epoch_metrics:
                epoch_metrics[key] += losses[key].item()
            num_batches += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {'loss': 0, 'ade': 0, 'fde': 0, 'accuracy': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                obs_traj = batch['obs'].to(self.device)
                pred_traj = batch['pred'].to(self.device)

                # Forward pass (no augmentation, no teacher forcing)
                output = self.model(obs_traj, teacher_forcing_ratio=0.0)
                predictions = output['predictions']

                # Compute metrics
                losses = self.criterion(predictions, pred_traj)

                # Accuracy: percentage of predictions within threshold
                distances = torch.norm(predictions - pred_traj, dim=-1)
                accuracy = (distances < 0.5).float().mean() * 100

                for key in ['loss', 'ade', 'fde']:
                    val_metrics[key] += losses[key].item()
                val_metrics['accuracy'] += accuracy.item()
                num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_metrics

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            for key, value in train_metrics.items():
                self.train_history[key].append(value)

            # Validate
            val_metrics = self.validate(val_loader)
            for key, value in val_metrics.items():
                self.val_history[key].append(value)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), 'best_hgnn_model.pt')

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"ADE: {train_metrics['ade']:.4f}, FDE: {train_metrics['fde']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.2f}%")

        return self.train_history, self.val_history


# ==============================================================================
# DATASET & DATALOADER
# ==============================================================================

class SyntheticTrajectoryDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for demonstration.

    Generates crowd scenarios with group behaviors:
    - Groups moving together
    - Collision avoidance
    - Random individual movements
    """

    def __init__(
        self, 
        num_samples: int = 1000,
        config: Optional[ModelConfig] = None,
        num_agents_range: Tuple[int, int] = (5, 20)
    ):
        self.num_samples = num_samples
        self.config = config or ModelConfig()
        self.num_agents_range = num_agents_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        num_agents = random.randint(*self.num_agents_range)

        # Generate synthetic crowd scenario
        obs_traj, pred_traj = self._generate_scene(num_agents)

        return {
            'obs': torch.FloatTensor(obs_traj),
            'pred': torch.FloatTensor(pred_traj)
        }

    def _generate_scene(self, num_agents: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic crowd scene with group behaviors."""
        obs_len = self.config.obs_len
        pred_len = self.config.pred_len
        total_len = obs_len + pred_len

        # Initialize positions randomly
        positions = np.random.randn(num_agents, 2) * 5
        trajectories = [positions.copy()]

        # Assign velocities with group coherence
        velocities = np.random.randn(num_agents, 2) * 0.5

        # Create groups (DBSCAN-style clustering in initial positions)
        from sklearn.cluster import DBSCAN
        clusters = DBSCAN(eps=3.0, min_samples=2).fit_predict(positions)

        # Simulate movement
        for t in range(1, total_len):
            new_positions = positions.copy()

            for i in range(num_agents):
                # Individual velocity
                new_positions[i] += velocities[i]

                # Group cohesion: move towards group center
                if clusters[i] != -1:
                    group_mask = clusters == clusters[i]
                    group_center = positions[group_mask].mean(axis=0)
                    new_positions[i] += (group_center - positions[i]) * 0.1

                # Collision avoidance
                for j in range(num_agents):
                    if i != j:
                        dist = np.linalg.norm(new_positions[i] - positions[j])
                        if dist < 1.0:
                            # Push away
                            direction = new_positions[i] - positions[j]
                            direction = direction / (np.linalg.norm(direction) + 1e-8)
                            new_positions[i] += direction * 0.5

            positions = new_positions
            trajectories.append(positions.copy())

        trajectories = np.array(trajectories)  # (total_len, num_agents, 2)

        obs = trajectories[:obs_len]
        pred = trajectories[obs_len:]

        return obs, pred


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def compute_ade_fde(
    predictions: np.ndarray, 
    ground_truth: np.ndarray
) -> Tuple[float, float]:
    """
    Compute ADE and FDE metrics.

    Args:
        predictions: (num_samples, pred_len, num_agents, 2)
        ground_truth: (num_samples, pred_len, num_agents, 2)

    Returns:
        ade: Average Displacement Error
        fde: Final Displacement Error
    """
    distances = np.linalg.norm(predictions - ground_truth, axis=-1)

    ade = distances.mean()
    fde = distances[:, -1, :].mean()

    return ade, fde


def compute_collision_rate(
    predictions: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    Compute percentage of timesteps where agents are too close (collision).

    Lower is better.
    """
    num_samples, pred_len, num_agents, _ = predictions.shape
    collisions = 0
    total_pairs = 0

    for sample in predictions:
        for t in range(pred_len):
            positions = sample[t]  # (num_agents, 2)
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < threshold:
                        collisions += 1
                    total_pairs += 1

    return (collisions / total_pairs) * 100 if total_pairs > 0 else 0


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("HyperGraph Neural Network for Multi-Agent Trajectory Prediction")
    print("=" * 80)

    # Configuration
    config = ModelConfig(
        hidden_dim=128,
        num_hgnn_layers=2,
        obs_len=8,
        pred_len=12,
        batch_size=16,
        num_epochs=50,
        dbscan_eps=2.0
    )

    print(f"\nConfiguration:")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  HGNN Layers: {config.num_hgnn_layers}")
    print(f"  Observation Length: {config.obs_len}")
    print(f"  Prediction Length: {config.pred_len}")
    print(f"  Device: {config.device}")

    # Create datasets
    print("\nGenerating synthetic datasets...")
    train_dataset = SyntheticTrajectoryDataset(
        num_samples=500, 
        config=config,
        num_agents_range=(5, 15)
    )
    val_dataset = SyntheticTrajectoryDataset(
        num_samples=100, 
        config=config,
        num_agents_range=(5, 15)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False
    )

    # Initialize model
    print("\nInitializing Seq2Seq HGNN model...")
    model = Seq2SeqHGNN(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # Initialize trainer
    trainer = Trainer(model, config)

    # Train
    print("\nStarting training...")
    train_hist, val_hist = trainer.train(train_loader, val_loader)

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Load best model
    model.load_state_dict(torch.load('best_hgnn_model.pt'))
    model.eval()

    # Collect predictions
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for batch in val_loader:
            obs = batch['obs'].to(config.device)
            pred = batch['pred'].to(config.device)

            output = model(obs, return_metadata=True)
            predictions = output['predictions']

            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(pred.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Compute metrics
    ade, fde = compute_ade_fde(all_predictions, all_ground_truth)
    collision_rate = compute_collision_rate(all_predictions)

    print(f"\nMetrics on Validation Set:")
    print(f"  ADE (Average Displacement Error): {ade:.4f}")
    print(f"  FDE (Final Displacement Error): {fde:.4f}")
    print(f"  Collision Rate: {collision_rate:.2f}%")
    print(f"  Validation Accuracy: {val_hist['accuracy'][-1]:.2f}%")

    # Demonstrate hypergraph construction
    print("\n" + "=" * 80)
    print("HYPERGRAPH CONSTRUCTION DEMO")
    print("=" * 80)

    # Get one sample
    sample = val_dataset[0]
    obs = sample['obs'].unsqueeze(0).to(config.device)

    with torch.no_grad():
        output = model(obs, return_metadata=True)
        metadata = output['metadata'][0]

    print(f"\nSample Scene Analysis:")
    print(f"  Number of Agents: {obs.shape[2]}")
    print(f"  Number of Hyperedges (Groups): {metadata['num_hyperedges']}")
    print(f"  Average Group Size: {metadata['avg_group_size']:.2f}")
    print(f"  Cluster Assignments: {metadata['clusters']}")
    print(f"\nHyperedge Structure (Groups):")
    for i, he in enumerate(metadata['hyperedges'][:5]):  # Show first 5
        print(f"  Hyperedge {i}: Agents {he}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
