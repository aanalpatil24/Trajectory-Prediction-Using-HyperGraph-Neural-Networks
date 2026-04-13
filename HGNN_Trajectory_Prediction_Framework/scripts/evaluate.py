#!/usr/bin/env python
"""
Evaluation script for HGNN Trajectory Prediction.

Example usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --dataset synthetic
"""
import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.seq2seq_hgnn import Seq2SeqHGNN
from src.data.dataset import SyntheticTrajectoryDataset, TrajectoryDataset
from src.data.dataloader import get_dataloader
from src.training.metrics import evaluate_all_metrics, print_metrics
from src.utils.config import ModelConfig
from src.utils.visualization import plot_trajectories, plot_hypergraph
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HGNN Trajectory Prediction")

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["synthetic", "trajectory"],
                       help="Dataset type")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                       help="Output directory for visualizations")

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = checkpoint["config"]

    # Create config from saved dict
    config = ModelConfig(**config_dict)

    print(f"\nLoaded config:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  HGNN layers: {config.num_hgnn_layers}")
    print(f"  DBSCAN eps: {config.dbscan_eps}")

    # Create test dataset
    if args.dataset == "synthetic":
        test_dataset = SyntheticTrajectoryDataset(
            num_samples=args.num_samples,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            num_agents_range=(config.min_agents, config.max_agents),
            seed=999
        )
    else:
        test_dataset = TrajectoryDataset(
            data_dir=args.data_dir,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            split="test"
        )

    test_loader = get_dataloader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create and load model
    model = Seq2SeqHGNN(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(config.device)

    # Evaluate
    print(f"\nEvaluating on {len(test_dataset)} samples...")

    all_predictions = []
    all_ground_truth = []
    all_obs = []
    all_metadata = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            obs = batch["obs"].to(config.device)
            pred = batch["pred"].to(config.device)

            # Forward pass
            output = model(obs, return_metadata=True)
            predictions = output["predictions"]

            # Store for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(pred.cpu().numpy())
            all_obs.append(obs.cpu().numpy())

            if "metadata" in output:
                all_metadata.extend(output["metadata"])

            # Visualize first few samples
            if args.visualize and i < 5:
                for b in range(min(2, obs.shape[0])):
                    save_path = os.path.join(
                        args.output_dir, 
                        f"trajectory_sample{i}_{b}.png"
                    )
                    plot_trajectories(
                        obs_trajectory=all_obs[-1][b],
                        pred_trajectory=all_predictions[-1][b],
                        gt_trajectory=all_ground_truth[-1][b],
                        title=f"Sample {i}-{b}",
                        save_path=save_path,
                        show=False
                    )

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Compute metrics
    metrics = evaluate_all_metrics(
        all_predictions,
        all_ground_truth,
        collision_threshold=config.collision_threshold,
        success_threshold=config.success_threshold
    )

    print_metrics(metrics, prefix="Test ")

    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"\nMetrics saved to {metrics_path}")

    # Visualize hypergraph structure for last batch
    if args.visualize and all_metadata:
        viz_path = os.path.join(args.output_dir, "hypergraph_structure.png")
        last_positions = all_obs[-1][:, -1, :, :]  # Last obs positions
        last_meta = all_metadata[-1]

        plot_hypergraph(
            positions=last_positions[0],
            hyperedges=last_meta["hyperedges"],
            title=f"Hypergraph Structure (Groups: {last_meta['num_hyperedges']})",
            save_path=viz_path,
            show=False
        )
        print(f"Hypergraph visualization saved to {viz_path}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
