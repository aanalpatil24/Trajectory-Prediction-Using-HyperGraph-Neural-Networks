import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Sequence lengths
    obs_len: int = 8      # Past steps we look at
    pred_len: int = 12    # Future steps we predict
    num_agents: int = 20  # Max people in a scene
    
    # Graph settings
    social_radius: float = 10.0  # Group people within 10 meters
    min_agents_per_hyperedge: int = 2
    max_agents_per_hyperedge: int = 5
    
    # Neural Network sizes
    input_dim: int = 2    # X and Y coordinates
    hidden_dim: int = 128 # Network memory size
    num_layers: int = 2   # Number of graph layers
    dropout: float = 0.1
    
    # Training rules
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    
    # Data Augmentation chances
    aug_prob: float = 0.3
    impute_prob: float = 0.1
    
    # Hardware setup
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 42
    
    # Folders
    data_dir: str = './data'
    save_dir: str = './checkpoints'
