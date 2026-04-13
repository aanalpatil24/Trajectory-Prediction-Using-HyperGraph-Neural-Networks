--------------------------------------------------------------------------------
FOLDER: hgnn_trajectory
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
FILE: config.py
--------------------------------------------------------------------------------
"""
Configuration parameters for Trajectory Prediction using HGNNs
"""

import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Config:
    # Data parameters
    obs_len: int = 8  # Observation length (timesteps)
    pred_len: int = 12  # Prediction length (timesteps)
    num_agents: int = 20  # Max agents per scene
    skip: int = 1  # Skip frames for downsampling
    
    # Hypergraph parameters
    social_radius: float = 10.0  # Meters for social grouping
    min_agents_per_hyperedge: int = 2
    max_agents_per_hyperedge: int = 5
    
    # Model architecture
    input_dim: int = 2  # x, y coordinates
    hidden_dim: int = 128
    hyperedge_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    teacher_forcing_ratio: float = 0.5
    
    # Augmentation
    aug_prob: float = 0.3
    impute_prob: float = 0.1
    
    # System
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers: int = 4
    seed: int = 42
    
    # Paths
    data_dir: str = './data'
    save_dir: str = './checkpoints'
    log_dir: str = './logs'


--------------------------------------------------------------------------------
FILE: data/augmentation.py
--------------------------------------------------------------------------------
"""
Data augmentation techniques: Trajectory reversal and imputation
"""

import numpy as np
import torch
from typing import Tuple, Optional


class TrajectoryAugmentation:
    def __init__(self, aug_prob: float = 0.3, impute_prob: float = 0.1):
        self.aug_prob = aug_prob
        self.impute_prob = impute_prob
    
    def reverse_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """
        Kinematic reversal: Reverses temporal order while preserving physical validity.
        traj shape: (T, N, 2) where T=timesteps, N=agents, 2=(x,y)
        """
        if np.random.random() > self.aug_prob:
            return traj
        
        # Reverse temporal dimension
        reversed_traj = np.flip(traj, axis=0).copy()
        
        # Adjust velocities for consistency (optional smoothing)
        return reversed_traj
    
    def impute_missing(self, traj: np.ndarray) -> np.ndarray:
        """
        Data imputation: Randomly mask and reconstruct trajectory points.
        Simulates real-world occlusion scenarios.
        """
        if np.random.random() > self.impute_prob:
            return traj
        
        T, N, D = traj.shape
        mask = np.random.random((T, N)) > 0.1  # 10% missing rate
        
        if not mask.all():
            # Linear interpolation for missing values per agent
            traj_imputed = traj.copy()
            for n in range(N):
                missing = ~mask[:, n]
                if missing.any():
                    valid_idx = np.where(~missing)[0]
                    if len(valid_idx) > 1:
                        # Interpolate between valid points
                        traj_imputed[:, n, :] = self._interpolate(
                            np.arange(T), 
                            valid_idx, 
                            traj[valid_idx, n, :]
                        )
            return traj_imputed
        return traj
    
    def _interpolate(self, x_new: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
        """Linear interpolation helper"""
        return np.array([
            np.interp(x_new, x_known, y_known[:, d]) 
            for d in range(y_known.shape[1])
        ]).T
    
    def add_noise(self, traj: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise for robustness"""
        noise = np.random.normal(0, noise_std, traj.shape)
        return traj + noise
    
    def __call__(self, traj: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline"""
        traj = self.reverse_trajectory(traj)
        traj = self.impute_missing(traj)
        traj = self.add_noise(traj)
        return traj


--------------------------------------------------------------------------------
FILE: utils/hypergraph_utils.py
--------------------------------------------------------------------------------
"""
Hypergraph construction from social interactions
"""

import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from typing import Tuple, List


class HypergraphConstructor:
    def __init__(self, radius: float = 10.0, min_size: int = 2, max_size: int = 5):
        self.radius = radius
        self.min_size = min_size
        self.max_size = max_size
    
    def construct_from_positions(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct hypergraph from agent positions.
        
        Args:
            positions: (N, 2) tensor of x,y coordinates
            
        Returns:
            node_features: (N, 2) 
            hyperedge_index: (2, num_connections) 
                           row 0: node indices, row 1: hyperedge indices
        """
        N = positions.shape[0]
        device = positions.device
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)  # (N, N)
        
        # Group agents within radius (greedy grouping algorithm)
        visited = torch.zeros(N, dtype=torch.bool, device=device)
        hyperedges = []
        hyperedge_idx = 0
        
        # Sort by degree centrality (most connected first)
        degrees = (dist_matrix < self.radius).sum(dim=1)
        node_order = torch.argsort(degrees, descending=True)
        
        for i in node_order:
            if visited[i]:
                continue
            
            # Find neighbors within radius
            neighbors = (dist_matrix[i] < self.radius).nonzero(as_tuple=True)[0]
            
            if len(neighbors) >= self.min_size:
                # Limit hyperedge size
                if len(neighbors) > self.max_size:
                    # Sort by distance and take closest
                    dists = dist_matrix[i, neighbors]
                    neighbors = neighbors[torch.argsort(dists)[:self.max_size]]
                
                hyperedges.append(neighbors)
                visited[neighbors] = True
                hyperedge_idx += 1
        
        # Handle isolated agents (singleton hyperedges)
        for i in range(N):
            if not visited[i]:
                hyperedges.append(torch.tensor([i], device=device))
                hyperedge_idx += 1
        
        # Build incidence matrix format [node_idx, hyperedge_idx]
        node_indices = []
        edge_indices = []
        for he_idx, nodes in enumerate(hyperedges):
            node_indices.extend(nodes.tolist())
            edge_indices.extend([he_idx] * len(nodes))
        
        hyperedge_index = torch.tensor([node_indices, edge_indices], device=device)
        
        return positions, hyperedge_index
    
    def batch_construct(self, positions_batch: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Construct hypergraphs for batch of scenes
        positions_batch: (B, N, 2)
        """
        batch_hypergraphs = []
        for b in range(positions_batch.shape[0]):
            node_feat, he_index = self.construct_from_positions(positions_batch[b])
            batch_hypergraphs.append((node_feat, he_index))
        return batch_hypergraphs


--------------------------------------------------------------------------------
FILE: models/layers.py
--------------------------------------------------------------------------------
"""
Custom HyperGraph Neural Network Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HypergraphConv(nn.Module):
    """
    Hypergraph Convolution Layer using star expansion method.
    Transforms hypergraph to bipartite graph (nodes <-> hyperedges)
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Learnable weights for node and hyperedge features
        self.lin_node = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor, 
                hyperedge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_channels)
            hyperedge_index: (2, E) where row 0: node idx, row 1: hyperedge idx
            hyperedge_attr: Optional hyperedge features
        """
        N = x.size(0)
        num_edges = hyperedge_index[1].max().item() + 1
        
        # Project features
        x_node = self.lin_node(x)  # (N, out_channels)
        
        if hyperedge_attr is None:
            # Aggregate node features to hyperedges
            hyperedge_attr = torch.zeros(num_edges, self.out_channels, device=x.device)
            hyperedge_attr.index_add_(0, hyperedge_index[1], x_node[hyperedge_index[0]])
            
            # Count nodes per hyperedge for averaging
            node_counts = torch.zeros(num_edges, device=x.device)
            node_counts.index_add_(0, hyperedge_index[1], 
                                  torch.ones(hyperedge_index.size(1), device=x.device))
            hyperedge_attr = hyperedge_attr / (node_counts.unsqueeze(1) + 1e-8)
        else:
            hyperedge_attr = self.lin_edge(hyperedge_attr)
        
        # Message passing: hyperedges -> nodes
        out = torch.zeros_like(x_node)
        out.index_add_(0, hyperedge_index[0], hyperedge_attr[hyperedge_index[1]])
        
        # Average by degree (how many hyperedges each node belongs to)
        degrees = torch.zeros(N, device=x.device)
        degrees.index_add_(0, hyperedge_index[0], 
                          torch.ones(hyperedge_index.size(1), device=x.device))
        out = out / (degrees.unsqueeze(1) + 1e-8)
        
        # Skip connection and bias
        out = out + x_node
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class TemporalEncoder(nn.Module):
    """LSTM-based temporal encoder for trajectory history"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F) - Batch, Time, Nodes, Features
        Returns:
            (B, N, Hidden) - Final hidden states per agent
        """
        B, T, N, F = x.shape
        # Process each agent independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, B*N, hidden)
        
        # Take last layer
        h_n = h_n[-1]  # (B*N, hidden)
        h_n = h_n.view(B, N, self.hidden_dim)
        return h_n


class TrajectoryDecoder(nn.Module):
    """MLP-based decoder for future trajectory prediction"""
    def __init__(self, hidden_dim: int, pred_len: int, output_dim: int = 2):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_len * output_dim)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, hidden_dim) - encoded features
        Returns:
            (B, T_pred, N, 2) - predicted trajectories
        """
        B, N, _ = h.shape
        out = self.mlp(h)  # (B, N, pred_len*2)
        out = out.view(B, N, self.pred_len, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # (B, T_pred, N, 2)
        return out


--------------------------------------------------------------------------------
FILE: models/hgnn.py
--------------------------------------------------------------------------------
"""
Complete HGNN Model for Trajectory Prediction
"""

import torch
import torch.nn as nn
from .layers import HypergraphConv, TemporalEncoder, TrajectoryDecoder
from typing import List, Tuple


class TrajectoryHGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.obs_len = config.obs_len
        self.pred_len = config.pred_len
        
        # Temporal encoding of observed trajectory
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=2
        )
        
        # Hypergraph convolution layers
        self.hgnn_layers = nn.ModuleList([
            HypergraphConv(
                in_channels=config.hidden_dim if i == 0 else config.hidden_dim,
                out_channels=config.hidden_dim
            )
            for i in range(config.num_layers)
        ])
        
        # Batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Decoder
        self.decoder = TrajectoryDecoder(
            hidden_dim=config.hidden_dim,
            pred_len=config.pred_len
        )
        
        # Learnable social embedding
        self.social_embedding = nn.Embedding(config.max_agents_per_hyperedge, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, obs_traj: torch.Tensor, 
                hyperedge_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_traj: (B, T_obs, N, 2) - observed trajectories
            hyperedge_indices: List of hyperedge indices for each batch item
        Returns:
            pred_traj: (B, T_pred, N, 2) - predicted future trajectories
        """
        B, T, N, _ = obs_traj.shape
        
        # 1. Temporal encoding per agent
        h_temp = self.temporal_encoder(obs_traj)  # (B, N, hidden)
        
        # 2. Hypergraph convolution for social interactions
        h_social = h_temp.clone()
        
        for i, (conv, bn) in enumerate(zip(self.hgnn_layers, self.batch_norms)):
            h_batch = []
            for b in range(B):
                he_idx = hyperedge_indices[b]  # (2, E)
                h_single = conv(h_social[b], he_idx)  # (N, hidden)
                
                # Batch norm requires (N, hidden) -> (hidden, N) -> (N, hidden)
                h_single = bn(h_single)
                h_batch.append(h_single)
            
            h_social = torch.stack(h_batch)  # (B, N, hidden)
            h_social = self.dropout(F.relu(h_social))
        
        # 3. Combine temporal and social features
        h_combined = h_temp + h_social  # Residual connection
        
        # 4. Decode future trajectory
        pred_traj = self.decoder(h_combined)  # (B, T_pred, N, 2)
        
        return pred_traj


--------------------------------------------------------------------------------
FILE: data/dataset.py
--------------------------------------------------------------------------------
"""
Dataset loader for trajectory prediction
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentation import TrajectoryAugmentation


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, obs_len: int = 8, pred_len: int = 12, 
                 split: str = 'train', aug_prob: float = 0.3):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.split = split
        
        # Load data (assuming numpy format: (num_scenes, T, N, 2))
        data_path = os.path.join(data_dir, f'{split}.npz')
        if os.path.exists(data_path):
            data = np.load(data_path)
            self.trajectories = data['trajectories']  # (num_scenes, T_total, N, 2)
        else:
            # Generate synthetic data for demonstration
            self.trajectories = self._generate_synthetic_data(1000)
        
        self.augmentor = TrajectoryAugmentation(aug_prob=aug_prob if split == 'train' else 0.0)
    
    def _generate_synthetic_data(self, num_scenes: int) -> np.ndarray:
        """Generate synthetic crowd movement for testing"""
        T_total = self.obs_len + self.pred_len
        N = 20
        
        trajectories = np.zeros((num_scenes, T_total, N, 2))
        
        for scene in range(num_scenes):
            # Random starting positions
            pos = np.random.randn(N, 2) * 10  # Random starts
            
            # Random velocities (social force-like)
            vel = np.random.randn(N, 2) * 0.5
            
            for t in range(T_total):
                trajectories[scene, t] = pos.copy()
                
                # Update with simple physics + social repulsion
                for i in range(N):
                    for j in range(i+1, N):
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff) + 1e-8
                        if dist < 2.0:  # Collision avoidance
                            repulsion = diff / (dist ** 2) * 0.1
                            vel[i] += repulsion
                            vel[j] -= repulsion
                
                pos += vel * 0.1  # dt = 0.1
        
        return trajectories
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx: int):
        traj = self.trajectories[idx]  # (T_total, N, 2)
        
        # Split observation and prediction
        obs = traj[:self.obs_len]  # (T_obs, N, 2)
        pred = traj[self.obs_len:self.obs_len + self.pred_len]  # (T_pred, N, 2)
        
        # Apply augmentation only to training
        if self.split == 'train':
            obs = self.augmentor(obs)
        
        return {
            'obs': torch.FloatTensor(obs),
            'pred': torch.FloatTensor(pred),
            'scene_id': idx
        }


def collate_fn(batch):
    """Custom collate for variable hypergraphs"""
    obs = torch.stack([b['obs'] for b in batch])  # (B, T, N, 2)
    pred = torch.stack([b['pred'] for b in batch])  # (B, T, N, 2)
    scene_ids = [b['scene_id'] for b in batch]
    
    return {'obs': obs, 'pred': pred, 'scene_ids': scene_ids}


--------------------------------------------------------------------------------
FILE: utils/metrics.py
--------------------------------------------------------------------------------
"""
Evaluation metrics for trajectory prediction
"""

import torch
import numpy as np
from typing import Optional


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> float:
    """
    Average Displacement Error (ADE)
    Args:
        pred_traj: (B, T, N, 2)
        gt_traj: (B, T, N, 2)
        mask: (B, N) - valid agents
    """
    error = torch.norm(pred_traj - gt_traj, dim=-1)  # (B, T, N)
    
    if mask is not None:
        mask = mask.unsqueeze(1)  # (B, 1, N)
        error = error * mask
        return error.sum() / (mask.sum() * pred_traj.size(1))
    
    return error.mean().item()


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> float:
    """
    Final Displacement Error (FDE) - error at final timestep
    """
    final_error = torch.norm(pred_traj[:, -1] - gt_traj[:, -1], dim=-1)  # (B, N)
    
    if mask is not None:
        final_error = final_error * mask
        return final_error.sum() / mask.sum()
    
    return final_error.mean().item()


def compute_mr(pred_traj: torch.Tensor, gt_traj: torch.Tensor, 
               threshold: float = 2.0) -> float:
    """
    Miss Rate: percentage of trajectories with final error > threshold
    """
    final_error = torch.norm(pred_traj[:, -1] - gt_traj[:, -1], dim=-1)
    miss = (final_error > threshold).float()
    return miss.mean().item()


--------------------------------------------------------------------------------
FILE: train.py
--------------------------------------------------------------------------------
"""
Training script for Trajectory HGNN
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config
from data.dataset import TrajectoryDataset, collate_fn
from models.hgnn import TrajectoryHGNN
from utils.hypergraph_utils import HypergraphConstructor
from utils.metrics import compute_ade, compute_fde


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        # Initialize model
        self.model = TrajectoryHGNN(config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        self.criterion = nn.MSELoss()
        
        # Hypergraph constructor
        self.hg_constructor = HypergraphConstructor(
            radius=config.social_radius,
            min_size=config.min_agents_per_hyperedge,
            max_size=config.max_agents_per_hyperedge
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            TrajectoryDataset(config.data_dir, config.obs_len, config.pred_len, 
                            split='train', aug_prob=config.aug_prob),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            TrajectoryDataset(config.data_dir, config.obs_len, config.pred_len, 
                            split='val', aug_prob=0.0),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn
        )
        
        # Logging
        os.makedirs(config.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')
    
    def construct_batch_hypergraphs(self, obs_traj: torch.Tensor):
        """
        Construct hypergraphs for batch of observations
        obs_traj: (B, T, N, 2)
        """
        B, T, N, _ = obs_traj.shape
        # Use last observed position for spatial grouping
        last_pos = obs_traj[:, -1, :, :]  # (B, N, 2)
        
        hyperedge_indices = []
        for b in range(B):
            _, he_idx = self.hg_constructor.construct_from_positions(last_pos[b])
            hyperedge_indices.append(he_idx)
        
        return hyperedge_indices
    
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            obs = batch['obs'].to(self.device)  # (B, T, N, 2)
            gt_pred = batch['pred'].to(self.device)  # (B, T, N, 2)
            
            # Construct dynamic hypergraphs based on current positions
            hyperedge_indices = self.construct_batch_hypergraphs(obs)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(obs, hyperedge_indices)
            
            # Compute loss
            loss = self.criterion(pred, gt_pred)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                ade = compute_ade(pred, gt_pred)
                fde = compute_fde(pred, gt_pred)
            
            total_loss += loss.item()
            total_ade += ade
            total_fde += fde
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ADE': f'{ade:.4f}',
                'FDE': f'{fde:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_ade = total_ade / len(self.train_loader)
        avg_fde = total_fde / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'ade': avg_ade,
            'fde': avg_fde
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            obs = batch['obs'].to(self.device)
            gt_pred = batch['pred'].to(self.device)
            
            hyperedge_indices = self.construct_batch_hypergraphs(obs)
            pred = self.model(obs, hyperedge_indices)
            
            loss = self.criterion(pred, gt_pred)
            
            total_loss += loss.item()
            total_ade += compute_ade(pred, gt_pred)
            total_fde += compute_fde(pred, gt_pred)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_ade = total_ade / len(self.val_loader)
        avg_fde = total_fde / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'ade': avg_ade,
            'fde': avg_fde
        }
    
    def train(self):
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"ADE: {train_metrics['ade']:.4f}, FDE: {train_metrics['fde']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': self.config
                }, os.path.join(self.config.save_dir, 'best_model.pth'))
                print(f"Saved best model (val_loss: {val_metrics['loss']:.4f})")
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pth'))


--------------------------------------------------------------------------------
FILE: evaluate.py
--------------------------------------------------------------------------------
"""
Evaluation and visualization script
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from models.hgnn import TrajectoryHGNN
from data.dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from utils.hypergraph_utils import HypergraphConstructor
from utils.metrics import compute_ade, compute_fde, compute_mr


def visualize_prediction(obs_traj, pred_traj, gt_traj, save_path=None):
    """
    Visualize trajectory prediction
    obs_traj: (T_obs, N, 2) numpy
    pred_traj: (T_pred, N, 2) numpy
    gt_traj: (T_pred, N, 2) numpy
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    N = obs_traj.shape[1]
    colors = plt.cm.tab20(np.linspace(0, 1, N))
    
    for i in range(N):
        # Past trajectory (solid)
        ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], 
                color=colors[i], linewidth=2, alpha=0.8)
        ax.scatter(obs_traj[-1, i, 0], obs_traj[-1, i, 1], 
                  color=colors[i], s=100, marker='o', label=f'Agent {i}')
        
        # Ground truth future (dashed)
        ax.plot(gt_traj[:, i, 0], gt_traj[:, i, 1], 
                color=colors[i], linewidth=2, linestyle='--', alpha=0.6)
        
        # Predicted future (solid)
        ax.plot(pred_traj[:, i, 0], pred_traj[:, i, 1], 
                color=colors[i], linewidth=2, linestyle='-', alpha=0.9)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Trajectory Prediction: Observation (solid), GT (dashed), Pred (solid)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def evaluate_model(model_path, config):
    """Load and evaluate model"""
    device = config.device
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = TrajectoryHGNN(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_dataset = TrajectoryDataset(config.data_dir, config.obs_len, 
                                    config.pred_len, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    hg_constructor = HypergraphConstructor(
        radius=config.social_radius,
        min_size=config.min_agents_per_hyperedge,
        max_size=config.max_agents_per_hyperedge
    )
    
    all_ade = []
    all_fde = []
    all_mr = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            obs = batch['obs'].to(device)
            gt = batch['pred'].to(device)
            
            # Construct hypergraph
            last_pos = obs[:, -1, :, :]
            _, he_idx = hg_constructor.construct_from_positions(last_pos[0])
            
            # Predict
            pred = model(obs, [he_idx])
            
            # Metrics
            ade = compute_ade(pred, gt)
            fde = compute_fde(pred, gt)
            mr = compute_mr(pred, gt, threshold=2.0)
            
            all_ade.append(ade)
            all_fde.append(fde)
            all_mr.append(mr)
            
            # Visualize first few
            if i < 5:
                visualize_prediction(
                    obs[0].cpu().numpy(),
                    pred[0].cpu().numpy(),
                    gt[0].cpu().numpy(),
                    save_path=f'results/prediction_{i}.png'
                )
    
    print(f"\nTest Results:")
    print(f"ADE: {np.mean(all_ade):.4f} ± {np.std(all_ade):.4f}")
    print(f"FDE: {np.mean(all_fde):.4f} ± {np.std(all_fde):.4f}")
    print(f"MR:  {np.mean(all_mr):.4f} ± {np.std(all_mr):.4f}")
    
    return {
        'ade': np.mean(all_ade),
        'fde': np.mean(all_fde),
        'mr': np.mean(all_mr)
    }


--------------------------------------------------------------------------------
FILE: main.py
--------------------------------------------------------------------------------
"""
Entry point for training/evaluation
"""

import argparse
import torch
import random
import numpy as np
from config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for eval')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    config = Config()
    set_seed(config.seed)
    
    if args.mode == 'train':
        from train import Trainer
        trainer = Trainer(config)
        trainer.train()
    else:
        from evaluate import evaluate_model
        if args.checkpoint is None:
            args.checkpoint = f'{config.save_dir}/best_model.pth'
        evaluate_model(args.checkpoint, config)