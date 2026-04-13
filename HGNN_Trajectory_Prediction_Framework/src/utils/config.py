"""
Configuration management for HGNN Trajectory Prediction.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import json
import os


@dataclass
class ModelConfig:
    """
    Configuration for HGNN Trajectory Prediction Model.

    All hyperparameters are centralized here for reproducibility.
    """
    # Model Architecture
    input_dim: int = 2                    # (x, y) coordinates
    hidden_dim: int = 128                 # GRU and HGNN hidden size
    output_dim: int = 2                   # (x, y) coordinates
    num_hgnn_layers: int = 2              # Number of HGNN convolution layers
    num_gru_layers: int = 2               # GRU depth for encoder/decoder
    num_heads: int = 4                    # Multi-head attention heads
    dropout: float = 0.1                  # Dropout rate

    # Sequence Configuration
    obs_len: int = 8                      # Observation timesteps (past)
    pred_len: int = 12                    # Prediction timesteps (future)

    # Social Interaction (DBSCAN Hypergraph Construction)
    dbscan_eps: float = 2.0               # Spatial threshold for grouping (meters)
    dbscan_min_samples: int = 2           # Minimum agents to form a group
    use_dynamic_hypergraph: bool = True     # Reconstruct hypergraph each timestep
    hypergraph_weighted: bool = True      # Use cohesion-based hyperedge weights

    # Training Configuration
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    teacher_forcing_ratio: float = 0.5    # Training only

    # Learning Rate Scheduling
    lr_scheduler: str = "reduce_on_plateau"  # Options: "step", "cosine", "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_step_size: int = 30

    # Data Augmentation
    augmentation_enabled: bool = True
    augmentation_prob: float = 0.5
    reverse_prob: float = 0.3
    missing_data_prob: float = 0.1
    noise_scale: float = 0.01
    max_missing_frames: int = 3

    # Dataset Configuration
    dataset_name: str = "synthetic"         # Options: "synthetic", "eth", "ucy", "nuscenes"
    num_train_samples: int = 5000
    num_val_samples: int = 1000
    num_test_samples: int = 500
    min_agents: int = 5
    max_agents: int = 20

    # System Configuration
    device: str = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    # Logging & Checkpointing
    log_dir: str = "outputs/logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10                  # Save checkpoint every N epochs
    log_every: int = 100                  # Log every N batches


    # Evaluation Metrics
    metrics: List[str] = field(default_factory=lambda: ["ade", "fde", "collision_rate"])
    collision_threshold: float = 0.1      # Meters
    success_threshold: float = 0.5        # Meters for accuracy calculation

    def __post_init__(self):
        """Validate configuration."""
        assert self.obs_len > 0, "obs_len must be positive"
        assert self.pred_len > 0, "pred_len must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith("_")
        }

    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["=" * 60, "Model Configuration", "=" * 60]
        for key, value in self.to_dict().items():
            lines.append(f"  {key:30s}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)
