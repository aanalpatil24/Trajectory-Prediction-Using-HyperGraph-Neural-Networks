"""
Loss functions for trajectory prediction.

Includes:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- Collision penalty
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class TrajectoryLoss(nn.Module):
    """
    Combined loss for trajectory prediction.

    Combines:
    - ADE: Average displacement over all timesteps
    - FDE: Final displacement at prediction horizon
    - Optional collision penalty
    """

    def __init__(
        self,
        ade_weight: float = 1.0,
        fde_weight: float = 1.0,
        collision_weight: float = 0.0,
        collision_threshold: float = 0.1
    ):
        """
        Initialize TrajectoryLoss.

        Args:
            ade_weight: Weight for ADE term
            fde_weight: Weight for FDE term
            collision_weight: Weight for collision penalty
            collision_threshold: Distance threshold for collision
        """
        super().__init__()

        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.collision_weight = collision_weight
        self.collision_threshold = collision_threshold

    def forward(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory prediction loss.

        Args:
            predictions: (batch_size, pred_len, num_agents, 2)
            ground_truth: (batch_size, pred_len, num_agents, 2)
            mask: (batch_size, num_agents) bool mask for valid agents

        Returns:
            Dictionary of loss components
        """
        # Compute per-timestep distances
        distances = torch.norm(predictions - ground_truth, dim=-1)  # (B, T, N)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, distances.shape[1], -1)
            distances = distances * mask_expanded
            num_valid = mask.sum() * distances.shape[1]
        else:
            num_valid = distances.numel()

        # ADE: Average Displacement Error
        ade = distances.sum() / num_valid

        # FDE: Final Displacement Error (last timestep)
        fde = distances[:, -1, :].sum() / (mask.sum() if mask is not None() else distances.shape[0] * distances.shape[2])

        # Total loss
        loss = self.ade_weight * ade + self.fde_weight * fde

        # Collision penalty (optional)
        collision_loss = torch.tensor(0.0, device=predictions.device)
        if self.collision_weight > 0:
            collision_loss = self._compute_collision_loss(predictions, mask)
            loss = loss + self.collision_weight * collision_loss

        return {
            "loss": loss,
            "ade": ade,
            "fde": fde,
            "collision_loss": collision_loss
        }

    def _compute_collision_loss(
        self,
        predictions: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute collision penalty (agents too close to each other).

        Args:
            predictions: (batch_size, pred_len, num_agents, 2)
            mask: (batch_size, num_agents)

        Returns:
            Collision penalty (scalar)
        """
        batch_size, pred_len, num_agents, _ = predictions.shape

        collision_penalty = 0.0
        count = 0

        for b in range(batch_size):
            for t in range(pred_len):
                positions = predictions[b, t, :, :]  # (num_agents, 2)

                # Compute pairwise distances
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        # Check if both agents are valid
                        if mask is not None and not (mask[b, i] and mask[b, j]):
                            continue

                        dist = torch.norm(positions[i] - positions[j])

                        # Penalize if too close
                        if dist < self.collision_threshold:
                            collision_penalty += (self.collision_threshold - dist)
                            count += 1

        return collision_penalty / (count + 1e-8)
