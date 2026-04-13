"""
Training pipeline for Seq2Seq HGNN.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os
import time
from tqdm import tqdm

from .loss import TrajectoryLoss
from .metrics import evaluate_all_metrics, print_metrics
from ..data.augmentation import TrajectoryAugmenter


class Trainer:
    """
    Training manager for HGNN Trajectory Prediction.

    Handles:
    - Training loop with augmentation
    - Validation and early stopping
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Initialize Trainer.

        Args:
            model: Seq2SeqHGNN model
            config: ModelConfig
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # Loss and optimizer
        self.criterion = TrajectoryLoss(
            ade_weight=1.0,
            fde_weight=1.0
        )

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        if config.lr_scheduler == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.lr_factor,
                patience=config.lr_patience,
                verbose=True
            )
        elif config.lr_scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_factor
            )
        else:
            self.scheduler = None

        # Augmenter
        self.augmenter = TrajectoryAugmenter(config)

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_ade": [],
            "val_fde": [],
            "val_accuracy": []
        }

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            obs = batch["obs"].to(self.device)
            pred = batch["pred"].to(self.device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)

            # Apply augmentation
            obs = self.augmenter.augment_batch(obs, training=True)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                obs,
                ground_truth=pred,
                teacher_forcing_ratio=self.config.teacher_forcing_ratio
            )

            # Compute loss
            losses = self.criterion(output["predictions"], pred, mask)

            # Backward pass
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()

            # Track metrics
            batch_loss = losses["loss"].item()
            epoch_loss += batch_loss
            num_batches += 1

            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        avg_loss = epoch_loss / num_batches

        return {"loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        all_predictions = []
        all_ground_truth = []
        all_masks = []

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="[Validate]"):
                obs = batch["obs"].to(self.device)
                pred = batch["pred"].to(self.device)
                mask = batch.get("mask", None)

                # Forward pass (no augmentation, no teacher forcing)
                output = self.model(obs, teacher_forcing_ratio=0.0)
                predictions = output["predictions"]

                # Compute loss
                losses = self.criterion(predictions, pred, mask.to(self.device) if mask is not None else None)
                val_loss += losses["loss"].item()

                # Collect for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_ground_truth.append(pred.cpu().numpy())
                if mask is not None:
                    all_masks.append(mask.cpu().numpy())

                num_batches += 1

        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_ground_truth = np.concatenate(all_ground_truth, axis=0)
        all_masks = np.concatenate(all_masks, axis=0) if all_masks else None

        # Compute metrics
        metrics = evaluate_all_metrics(
            all_predictions,
            all_ground_truth,
            all_masks,
            collision_threshold=self.config.collision_threshold,
            success_threshold=self.config.success_threshold
        )

        metrics["loss"] = val_loss / num_batches

        return metrics

    def train(self, num_epochs: Optional[int] = None):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs (default: config.num_epochs)
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print(f"\n{'='*60}")
        print(f"Training HGNN Trajectory Prediction")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_ade"].append(val_metrics["ADE"])
            self.history["val_fde"].append(val_metrics["FDE"])
            self.history["val_accuracy"].append(val_metrics["Accuracy"])

            # Learning rate scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Logging
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs} - {elapsed:.1f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val ADE: {val_metrics['ADE']:.4f}m")
            print(f"  Val FDE: {val_metrics['FDE']:.4f}m")
            print(f"  Val Accuracy: {val_metrics['Accuracy']:.2f}%")

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "history": self.history
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)

        print(f"Checkpoint loaded: {path}")
