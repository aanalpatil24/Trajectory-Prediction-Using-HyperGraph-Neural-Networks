"""
Visualization utilities for trajectory prediction.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict
import seaborn as sns


def plot_trajectories(
    obs_trajectory: np.ndarray,
    pred_trajectory: Optional[np.ndarray] = None,
    gt_trajectory: Optional[np.ndarray] = None,
    title: str = "Trajectory Prediction",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot observed and predicted trajectories.

    Args:
        obs_trajectory: (obs_len, num_agents, 2) observed trajectory
        pred_trajectory: (pred_len, num_agents, 2) predicted trajectory
        gt_trajectory: (pred_len, num_agents, 2) ground truth trajectory
        title: Plot title
        save_path: Path to save figure
        show: Whether to show plot
    """
    plt.figure(figsize=(10, 8))

    num_agents = obs_trajectory.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    for agent_idx in range(num_agents):
        color = colors[agent_idx]

        # Plot observed trajectory
        obs = obs_trajectory[:, agent_idx, :]
        plt.plot(obs[:, 0], obs[:, 1], "o-", color=color, alpha=0.7, label=f"Agent {agent_idx} (Obs)")

        # Mark start and end of observation
        plt.scatter(obs[0, 0], obs[0, 1], s=100, c=[color], marker="o", edgecolors="black", zorder=5)
        plt.scatter(obs[-1, 0], obs[-1, 1], s=100, c=[color], marker="s", edgecolors="black", zorder=5)

        # Plot prediction
        if pred_trajectory is not None:
            pred = pred_trajectory[:, agent_idx, :]
            plt.plot(pred[:, 0], pred[:, 1], "--", color=color, alpha=0.5, label=f"Agent {agent_idx} (Pred)")

        # Plot ground truth
        if gt_trajectory is not None:
            gt = gt_trajectory[:, agent_idx, :]
            plt.plot(gt[:, 0], gt[:, 1], "-", color=color, alpha=0.9, linewidth=2, label=f"Agent {agent_idx} (GT)")

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_hypergraph(
    positions: np.ndarray,
    hyperedges: List[List[int]],
    title: str = "Hypergraph Structure",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize hypergraph structure with agents and hyperedges.

    Args:
        positions: (num_agents, 2) agent positions
        hyperedges: List of hyperedge (agent indices)
        title: Plot title
        save_path: Path to save figure
        show: Whether to show plot
    """
    plt.figure(figsize=(10, 8))

    num_agents = len(positions)

    # Draw hyperedges as convex hulls
    from matplotlib.patches import Polygon

    for he_idx, members in enumerate(hyperedges):
        if len(members) > 2:
            member_pos = positions[members]
            # Convex hull
            from scipy.spatial import ConvexHull
            if len(members) > 2:
                try:
                    hull = ConvexHull(member_pos)
                    hull_points = member_pos[hull.vertices]
                    polygon = Polygon(hull_points, alpha=0.2, facecolor=plt.cm.Set3(he_idx / len(hyperedges)))
                    plt.gca().add_patch(polygon)
                except:
                    pass

    # Draw agents
    plt.scatter(positions[:, 0], positions[:, 1], s=200, c="blue", zorder=5, edgecolors="black")

    # Label agents
    for i, pos in enumerate(positions):
        plt.text(pos[0], pos[1], str(i), ha="center", va="center", fontsize=10, fontweight="bold")

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with lists of metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ADE
    axes[0, 1].plot(epochs, history["val_ade"], label="ADE", color="orange")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("ADE (m)")
    axes[0, 1].set_title("Average Displacement Error")
    axes[0, 1].grid(True, alpha=0.3)

    # FDE
    axes[1, 0].plot(epochs, history["val_fde"], label="FDE", color="green")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("FDE (m)")
    axes[1, 0].set_title("Final Displacement Error")
    axes[1, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 1].plot(epochs, history["val_accuracy"], label="Accuracy", color="red")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Prediction Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_attention_heatmap(attention_weights: np.ndarray, save_path: Optional[str] = None):
    """
    Plot attention weights between agents.

    Args:
        attention_weights: (num_agents, num_agents) attention matrix
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Social Attention Weights")
    plt.xlabel("Agent")
    plt.ylabel("Agent")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
