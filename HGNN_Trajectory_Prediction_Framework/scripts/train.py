#!/usr/bin/env python
"""
Training script for HGNN Trajectory Prediction.

Example usage:
    python train.py --config configs/default.yaml
    python train.py --dataset synthetic --epochs 50 --batch_size 32
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.seq2seq_hgnn import Seq2SeqHGNN
from src.data.dataset import SyntheticTrajectoryDataset, TrajectoryDataset
from src.data.dataloader import get_dataloader
from src.training.trainer import Trainer
from src.utils.config import ModelConfig
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HGNN Trajectory Prediction")

    # Data
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["synthetic", "trajectory"],
                       help="Dataset type")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory for real datasets")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension size")
    parser.add_argument("--num_hgnn_layers", type=int, default=2,
                       help="Number of HGNN layers")
    parser.add_argument("--dbscan_eps", type=float, default=2.0,
                       help="DBSCAN epsilon parameter")

    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/cpu/cuda)")

    # Logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Create config
    config = ModelConfig()

    # Override config with args
    config.hidden_dim = args.hidden_dim
    config.num_hgnn_layers = args.num_hgnn_layers
    config.dbscan_eps = args.dbscan_eps
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.checkpoint_dir = args.checkpoint_dir

    if args.device != "auto":
        config.device = args.device

    print(config)

    # Create datasets
    if args.dataset == "synthetic":
        train_dataset = SyntheticTrajectoryDataset(
            num_samples=config.num_train_samples,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            num_agents_range=(config.min_agents, config.max_agents)
        )
        val_dataset = SyntheticTrajectoryDataset(
            num_samples=config.num_val_samples,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            num_agents_range=(config.min_agents, config.max_agents),
            seed=43  # Different seed for validation
        )
    else:
        train_dataset = TrajectoryDataset(
            data_dir=args.data_dir,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            split="train"
        )
        val_dataset = TrajectoryDataset(
            data_dir=args.data_dir,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            split="val"
        )

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Create model
    model = Seq2SeqHGNN(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader)

    # Train
    trainer.train()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
