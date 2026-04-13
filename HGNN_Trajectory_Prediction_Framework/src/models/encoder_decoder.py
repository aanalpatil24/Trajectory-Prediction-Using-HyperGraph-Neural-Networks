"""
Sequence-to-Sequence Encoder-Decoder for Trajectory Prediction.

Encoder: GRU-based history encoder compressing past trajectories.
Decoder: GRU-based future decoder with autoregressive prediction.
"""
import torch
import torch.nn as nn
import random
from typing import Optional


class TrajectoryEncoder(nn.Module):
    """
    GRU-based encoder for trajectory history.

    Compresses observed trajectory (obs_len timesteps) into a hidden state
    vector representing the agent's motion history and current dynamics.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Trajectory Encoder.

        Args:
            input_dim: Input feature dimension (2 for x, y)
            hidden_dim: GRU hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability (applied between layers)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GRU encoder
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory history.

        Args:
            trajectory: (batch_size, obs_len, num_agents, 2)
                - batch_size: Number of scenes
                - obs_len: Observation timesteps
                - num_agents: Number of pedestrians
                - 2: (x, y) coordinates

        Returns:
            hidden_states: (batch_size, num_agents, hidden_dim)
                Encoded motion history for each agent
        """
        batch_size, obs_len, num_agents, _ = trajectory.shape

        # Reshape for efficient processing: (batch * agents, seq_len, 2)
        traj_flat = trajectory.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents, obs_len, self.input_dim
        )

        # Embed input coordinates
        embedded = self.input_embed(traj_flat)

        # GRU encoding
        # output: (batch * agents, seq_len, hidden_dim)
        # hidden: (num_layers, batch * agents, hidden_dim)
        output, hidden = self.gru(embedded)

        # Take last layer's final hidden state
        # hidden[-1]: (batch * agents, hidden_dim)
        final_hidden = hidden[-1]

        # Reshape back to batch format
        # (batch_size, num_agents, hidden_dim)
        hidden_states = final_hidden.view(batch_size, num_agents, self.hidden_dim)

        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class TrajectoryDecoder(nn.Module):
    """
    GRU-based decoder for future trajectory prediction.

    Autoregressively predicts future positions (pred_len timesteps) using
    socially-aware hidden states from the HGNN.
    """

    def __init__(
        self,
        output_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 12
    ):
        """
        Initialize Trajectory Decoder.

        Args:
            output_dim: Output dimension (2 for x, y)
            hidden_dim: GRU hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability
            pred_len: Number of future timesteps to predict
        """
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # GRU decoder
        self.gru = nn.GRU(
            output_dim,  # Input is previous position
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        social_features: torch.Tensor,
        last_position: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        ground_truth: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode future trajectory autoregressively.

        Args:
            social_features: (batch_size, num_agents, hidden_dim)
                Socially-aware features from HGNN
            last_position: (batch_size, num_agents, 2)
                Last observed position (starting point for prediction)
            teacher_forcing_ratio: Probability of using ground truth
                (0.0 for inference, >0.0 for training)
            ground_truth: (batch_size, pred_len, num_agents, 2)
                Ground truth future trajectory (for teacher forcing)

        Returns:
            predictions: (batch_size, pred_len, num_agents, 2)
                Predicted future trajectories
        """
        batch_size, num_agents, _ = social_features.shape
        device = social_features.device

        # Initialize decoder hidden state from social features
        # Expand to match GRU layer dimensions
        # social_features: (batch, agents, hidden) -> (agents, batch, hidden)
        hidden = social_features.permute(1, 0, 2)  # (agents, batch, hidden)
        hidden = hidden.repeat(self.num_layers, 1, 1)  # (layers, batch*agents, hidden)

        # Flatten batch and agents dimensions for GRU
        # hidden: (num_layers, batch * agents, hidden_dim)
        hidden = hidden.view(self.num_layers, batch_size * num_agents, self.hidden_dim)

        # Initial decoder input: last observed position
        # (batch_size, num_agents, 2) -> (batch_size * num_agents, 1, 2)
        decoder_input = last_position.view(batch_size * num_agents, 1, self.output_dim)

        predictions = []

        for t in range(self.pred_len):
            # GRU step
            # output: (batch*agents, 1, hidden_dim)
            output, hidden = self.gru(decoder_input, hidden)

            # Predict next position
            # pred: (batch*agents, output_dim)
            pred = self.output_layer(output.squeeze(1))

            # Reshape and store
            # (batch_size, num_agents, output_dim)
            pred_reshaped = pred.view(batch_size, num_agents, self.output_dim)
            predictions.append(pred_reshaped)

            # Teacher forcing: use ground truth or prediction as next input
            if ground_truth is not None and random.random() < teacher_forcing_ratio:
                # Use ground truth
                next_pos = ground_truth[:, t, :, :].view(
                    batch_size * num_agents, self.output_dim
                )
            else:
                # Use prediction
                next_pos = pred

            # Prepare next decoder input
            decoder_input = next_pos.unsqueeze(1)

        # Stack predictions along time dimension
        # (batch_size, pred_len, num_agents, output_dim)
        predictions = torch.stack(predictions, dim=1)

        return predictions
