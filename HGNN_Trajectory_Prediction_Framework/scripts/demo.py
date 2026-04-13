#!/usr/bin/env python
"""
Quick demo script for HGNN Trajectory Prediction.

Run this to see the model in action without training.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import numpy as np
from src.models.seq2seq_hgnn import Seq2SeqHGNN
from src.data.dataset import SyntheticTrajectoryDataset
from src.utils.config import ModelConfig
from src.utils.visualization import plot_trajectories, plot_hypergraph

def main():
    print("=" * 70)
    print("HGNN Trajectory Prediction - Quick Demo")
    print("=" * 70)

    # Configuration
    config = ModelConfig(
        hidden_dim=64,
        num_hgnn_layers=2,
        obs_len=8,
        pred_len=12,
        device='cpu'
    )

    print(f"\nConfiguration:")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  HGNN Layers: {config.num_hgnn_layers}")
    print(f"  Observation: {config.obs_len} timesteps")
    print(f"  Prediction: {config.pred_len} timesteps")

    # Create synthetic scene
    print(f"\nGenerating synthetic crowd scene...")
    dataset = SyntheticTrajectoryDataset(
        num_samples=1,
        obs_len=config.obs_len,
        pred_len=config.pred_len,
        num_agents_range=(8, 8),  # Exactly 8 agents
        group_behavior_prob=0.8,
        seed=42
    )

    sample = dataset[0]
    obs = sample['obs'].unsqueeze(0)  # Add batch dimension
    gt = sample['pred'].unsqueeze(0)

    print(f"  Scene generated: 8 agents, {config.obs_len + config.pred_len} timesteps")

    # Create model
    print(f"\nInitializing model...")
    model = Seq2SeqHGNN(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Forward pass
    print(f"\nRunning inference...")
    model.eval()
    with torch.no_grad():
        output = model(obs, return_metadata=True)
        predictions = output['predictions']
        metadata = output['metadata'][0]

    print(f"\nHypergraph Analysis:")
    print(f"  Number of groups (hyperedges): {metadata['num_hyperedges']}")
    print(f"  Average group size: {metadata['avg_group_size']:.2f}")
    print(f"  Isolated agents: {metadata['num_isolated']}")

    # Compute metrics
    ade = torch.norm(predictions - gt, dim=-1).mean().item()
    fde = torch.norm(predictions[:, -1, :, :] - gt[:, -1, :, :], dim=-1).mean().item()

    print(f"\nPrediction Metrics:")
    print(f"  ADE (Average Displacement Error): {ade:.4f}m")
    print(f"  FDE (Final Displacement Error): {fde:.4f}m")

    # Visualization paths
    output_dir = "outputs/demo"
    os.makedirs(output_dir, exist_ok=True)

    # Save visualizations
    traj_path = os.path.join(output_dir, "trajectory_prediction.png")
    plot_trajectories(
        obs_trajectory=obs[0].numpy(),
        pred_trajectory=predictions[0].numpy(),
        gt_trajectory=gt[0].numpy(),
        title="HGNN Trajectory Prediction",
        save_path=traj_path,
        show=False
    )
    print(f"\nTrajectory plot saved: {traj_path}")

    # Hypergraph visualization
    last_positions = obs[0, -1, :, :].numpy()
    hyper_path = os.path.join(output_dir, "hypergraph_structure.png")
    plot_hypergraph(
        positions=last_positions,
        hyperedges=metadata['hyperedges'],
        title=f"Social Groups (Hyperedges): {metadata['num_hyperedges']} groups",
        save_path=hyper_path,
        show=False
    )
    print(f"Hypergraph plot saved: {hyper_path}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. View visualizations in {output_dir}/")
    print(f"  2. Train model: python scripts/train.py --epochs 50")
    print(f"  3. Run tests: python -m pytest tests/")
    print(f"  4. Jupyter demo: jupyter notebook notebooks/demo.ipynb")

if __name__ == "__main__":
    main()
