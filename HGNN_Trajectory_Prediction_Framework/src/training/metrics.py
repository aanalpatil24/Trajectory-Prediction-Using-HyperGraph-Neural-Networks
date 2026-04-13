"""
Evaluation metrics for trajectory prediction.
"""
import torch
import numpy as np
from typing import Tuple, Dict, List


def compute_ade_fde(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Compute ADE and FDE metrics.

    Args:
        predictions: (num_samples, pred_len, num_agents, 2) or (pred_len, num_agents, 2)
        ground_truth: Same shape as predictions
        mask: (num_samples, num_agents) or (num_agents,) bool mask

    Returns:
        ade: Average Displacement Error (meters)
        fde: Final Displacement Error (meters)
    """
    # Ensure numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Compute Euclidean distances
    distances = np.linalg.norm(predictions - ground_truth, axis=-1)  # (... , T, N)

    # Apply mask if provided
    if mask is not None:
        mask_expanded = np.expand_dims(mask, axis=-2)  # Add time dimension
        distances = distances * mask_expanded
        num_valid = mask.sum()
    else:
        num_valid = distances.shape[-1]

    # ADE: Average over time and agents
    ade = distances.mean(axis=(-2, -1)).mean()  # Average over all dimensions

    # FDE: Final timestep only
    fde = distances[..., -1, :].mean(axis=-1).mean()

    return ade, fde


def compute_collision_rate(
    predictions: np.ndarray,
    threshold: float = 0.1,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute percentage of timesteps where agents are too close.

    Args:
        predictions: (num_samples, pred_len, num_agents, 2)
        threshold: Distance threshold for collision (meters)
        mask: (num_samples, num_agents) bool mask

    Returns:
        collision_rate: Percentage of agent pairs in collision
    """
    predictions = np.array(predictions)

    if predictions.ndim == 3:
        predictions = np.expand_dims(predictions, axis=0)

    batch_size, pred_len, num_agents, _ = predictions.shape

    collisions = 0
    total_pairs = 0

    for b in range(batch_size):
        for t in range(pred_len):
            positions = predictions[b, t, :, :]

            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    # Check mask
                    if mask is not None and not (mask[b, i] and mask[b, j]):
                        continue

                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < threshold:
                        collisions += 1
                    total_pairs += 1

    return (collisions / total_pairs * 100) if total_pairs > 0 else 0.0


def compute_prediction_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute percentage of predictions within threshold distance.

    Args:
        predictions: (num_samples, pred_len, num_agents, 2)
        ground_truth: Same shape
        threshold: Success threshold (meters)
        mask: (num_samples, num_agents) bool mask

    Returns:
        accuracy: Percentage of predictions within threshold
    """
    distances = np.linalg.norm(predictions - ground_truth, axis=-1)

    if mask is not None:
        mask_expanded = np.expand_dims(mask, axis=1)
        valid_distances = distances[mask_expanded]
    else:
        valid_distances = distances.flatten()

    within_threshold = (valid_distances < threshold).sum()
    accuracy = within_threshold / len(valid_distances) * 100

    return accuracy


def compute_min_ade_fde(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_modes: int = 20
) -> Tuple[float, float]:
    """
    Compute minADE and minFDE (for multi-modal predictions).

    Args:
        predictions: (num_samples, num_modes, pred_len, num_agents, 2)
        ground_truth: (num_samples, pred_len, num_agents, 2)
        num_modes: Number of prediction modes

    Returns:
        min_ade: Minimum ADE across modes
        min_fde: Minimum FDE across modes
    """
    # Compute distance for each mode
    distances = np.linalg.norm(predictions - ground_truth[:, None, ...], axis=-1)

    # ADE for each mode
    ade_per_mode = distances.mean(axis=(-2, -1))  # (num_samples, num_modes)
    min_ade = ade_per_mode.min(axis=1).mean()

    # FDE for each mode (final timestep)
    fde_per_mode = distances[:, :, -1, :].mean(axis=-1)  # (num_samples, num_modes)
    min_fde = fde_per_mode.min(axis=1).mean()

    return min_ade, min_fde


def evaluate_all_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    collision_threshold: float = 0.1,
    success_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.

    Args:
        predictions: (num_samples, pred_len, num_agents, 2)
        ground_truth: (num_samples, pred_len, num_agents, 2)
        mask: (num_samples, num_agents) bool mask
        collision_threshold: Collision distance threshold
        success_threshold: Success distance threshold

    Returns:
        Dictionary with all metrics
    """
    ade, fde = compute_ade_fde(predictions, ground_truth, mask)
    collision_rate = compute_collision_rate(predictions, collision_threshold, mask)
    accuracy = compute_prediction_accuracy(predictions, ground_truth, success_threshold, mask)

    return {
        "ADE": ade,
        "FDE": fde,
        "Collision_Rate": collision_rate,
        "Accuracy": accuracy
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    print(f"\n{prefix}Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        if "Rate" in key or "Accuracy" in key:
            print(f"  {key:20s}: {value:6.2f}%")
        else:
            print(f"  {key:20s}: {value:6.4f}m")
    print("-" * 40)
