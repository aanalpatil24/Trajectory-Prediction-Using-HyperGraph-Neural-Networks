"""
Sequence-to-Sequence HyperGraph Neural Network for Trajectory Prediction.

Main model integrating:
- GRU Encoder for trajectory history
- HGNN for social interaction modeling
- GRU Decoder for future prediction
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .encoder_decoder import TrajectoryEncoder, TrajectoryDecoder
from .hypergraph_conv import MultiLayerHGNN
from ..utils.hypergraph_builder import HypergraphConstructor


class Seq2SeqHGNN(nn.Module):
    """
    Sequence-to-Sequence HyperGraph Neural Network.

    Architecture:
        1. Encoder (GRU): Compress trajectory history into individual features
        2. Hypergraph Constructor (DBSCAN): Infer social groups dynamically
        3. HGNN: Aggregate features within social groups
        4. Fusion: Combine individual and social features
        5. Decoder (GRU): Predict future trajectory autoregressively

    Key Innovation:
        Uses hypergraphs (not pairwise graphs) to model group dynamics,
        enabling collective reasoning about multi-agent interactions.
    """

    def __init__(self, config):
        """
        Initialize Seq2Seq HGNN.

        Args:
            config: ModelConfig object with hyperparameters
        """
        super().__init__()

        self.config = config
        self.device = config.device

        # Core components
        self.encoder = TrajectoryEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_gru_layers,
            dropout=config.dropout
        )

        self.hypergraph_constructor = HypergraphConstructor(
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples
        )

        self.hgnn = MultiLayerHGNN(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_hgnn_layers,
            dropout=config.dropout,
            use_residual=True
        )

        self.decoder = TrajectoryDecoder(
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_gru_layers,
            dropout=config.dropout,
            pred_len=config.pred_len
        )

        # Fusion layer: combine individual + social features
        self.social_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim)
        )

        # Optional: Learnable social attention
        self.use_social_attention = True
        if self.use_social_attention:
            self.social_attention = nn.MultiheadAttention(
                config.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True
            )

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
                Observed trajectory coordinates
            ground_truth: (batch_size, pred_len, num_agents, 2), optional
                Ground truth future trajectory (for teacher forcing during training)
            teacher_forcing_ratio: Probability of using ground truth (0.0 for inference)
            return_metadata: Whether to return hypergraph construction info

        Returns:
            Dictionary containing:
                - 'predictions': (batch_size, pred_len, num_agents, 2)
                - 'social_features': (batch_size, num_agents, hidden_dim)
                - 'metadata': List of hypergraph metadata (if return_metadata=True)
        """
        batch_size, obs_len, num_agents, _ = obs_trajectory.shape
        device = obs_trajectory.device

        # Step 1: Encode trajectory history
        # (batch_size, num_agents, hidden_dim)
        individual_features = self.encoder(obs_trajectory)

        # Step 2: Construct hypergraph and apply HGNN for each scene
        # Use last observed positions for spatial clustering
        last_positions = obs_trajectory[:, -1, :, :]  # (batch, agents, 2)

        social_features_list = []
        all_metadata = []

        for b in range(batch_size):
            positions = last_positions[b]  # (num_agents, 2)
            features = individual_features[b]  # (num_agents, hidden_dim)

            # Construct hypergraph based on spatial proximity
            hyperedge_index, hyperedge_weight, metadata =                 self.hypergraph_constructor.construct_hypergraph(
                    positions, features, return_metadata=True
                )

            # Apply HGNN for social feature aggregation
            social_features = self.hgnn(features, hyperedge_index, hyperedge_weight)

            # Optional: Social attention mechanism
            if self.use_social_attention:
                features_unsqueezed = features.unsqueeze(0)  # (1, agents, hidden)
                social_features_unsqueezed = social_features.unsqueeze(0)

                attended_social, _ = self.social_attention(
                    features_unsqueezed,
                    social_features_unsqueezed,
                    social_features_unsqueezed
                )
                social_features = attended_social.squeeze(0)

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

    def predict(
        self,
        obs_trajectory: torch.Tensor,
        return_metadata: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode prediction (no teacher forcing).

        Args:
            obs_trajectory: (batch_size, obs_len, num_agents, 2)
            return_metadata: Whether to return hypergraph info

        Returns:
            Same as forward() but with teacher_forcing_ratio=0
        """
        return self.forward(
            obs_trajectory,
            ground_truth=None,
            teacher_forcing_ratio=0.0,
            return_metadata=return_metadata
        )

    def get_social_groups(self, obs_trajectory: torch.Tensor) -> List[Dict]:
        """
        Extract social groups (hyperedges) for visualization/analysis.

        Args:
            obs_trajectory: (batch_size, obs_len, num_agents, 2)

        Returns:
            List of metadata dictionaries containing group information
        """
        with torch.no_grad():
            output = self.forward(obs_trajectory, return_metadata=True)
        return output['metadata']
