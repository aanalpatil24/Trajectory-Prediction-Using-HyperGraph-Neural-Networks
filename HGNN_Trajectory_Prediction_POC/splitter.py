import os

def create_project():
    base_dir = "hgnn_trajectory"
    
    # Create the folder structure
    dirs = [
        base_dir,
        f"{base_dir}/data",
        f"{base_dir}/models",
        f"{base_dir}/utils",
        f"{base_dir}/checkpoints",
        f"{base_dir}/results"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")
    
    # Define the files with simple, small comments
    files = {
        f"{base_dir}/config.py": '''import torch
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
''',

        f"{base_dir}/data/augmentation.py": '''import numpy as np
import torch

class TrajectoryAugmentation:
    def __init__(self, aug_prob=0.3, impute_prob=0.1):
        self.aug_prob = aug_prob
        self.impute_prob = impute_prob
    
    def reverse_trajectory(self, traj):
        # 30% chance to flip the trajectory backwards
        if np.random.random() > self.aug_prob:
            return traj
        return np.flip(traj, axis=0).copy()
    
    def impute_missing(self, traj):
        # 10% chance to simulate broken camera sensors
        if np.random.random() > self.impute_prob:
            return traj
        
        T, N, D = traj.shape
        mask = np.random.random((T, N)) > 0.1  # Drop 10% of points
        
        # If points are missing, connect the dots (linear interpolation)
        if not mask.all():
            traj_imputed = traj.copy()
            for n in range(N):
                missing = ~mask[:, n]
                if missing.any():
                    valid_idx = np.where(~missing)[0]
                    if len(valid_idx) > 1:
                        # Fill in the blanks
                        traj_imputed[:, n, :] = np.array([
                            np.interp(np.arange(T), valid_idx, traj[valid_idx, n, d]) 
                            for d in range(D)
                        ]).T
            return traj_imputed
        return traj
    
    def add_noise(self, traj, noise_std=0.01):
        # Add tiny jitters to make the model robust
        noise = np.random.normal(0, noise_std, traj.shape)
        return traj + noise
    
    def __call__(self, traj):
        # Run all augmentations
        traj = self.reverse_trajectory(traj)
        traj = self.impute_missing(traj)
        traj = self.add_noise(traj)
        return traj
''',

        f"{base_dir}/data/dataset.py": '''import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .augmentation import TrajectoryAugmentation

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, split='train', aug_prob=0.3):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.split = split
        
        # Load real data if it exists, otherwise generate fake data
        data_path = os.path.join(data_dir, f'{split}.npz')
        if os.path.exists(data_path):
            self.trajectories = np.load(data_path)['trajectories'] 
        else:
            self.trajectories = self._generate_synthetic_data(100)
        
        # Only augment during training
        self.augmentor = TrajectoryAugmentation(aug_prob=aug_prob if split == 'train' else 0.0)
    
    def _generate_synthetic_data(self, num_scenes):
        # Creates a dummy crowd with basic collision avoidance
        T_total = self.obs_len + self.pred_len
        N = 20
        trajectories = np.zeros((num_scenes, T_total, N, 2))
        
        for scene in range(num_scenes):
            pos = np.random.randn(N, 2) * 10
            vel = np.random.randn(N, 2) * 0.5
            for t in range(T_total):
                trajectories[scene, t] = pos.copy()
                
                # Push agents away from each other if they get too close
                for i in range(N):
                    for j in range(i+1, N):
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff) + 1e-8
                        if dist < 2.0:
                            repulsion = diff / (dist ** 2) * 0.1
                            vel[i] += repulsion
                            vel[j] -= repulsion
                
                # Step forward in time
                pos += vel * 0.1
        return trajectories
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Split into past (obs) and future (pred)
        obs = traj[:self.obs_len]
        pred = traj[self.obs_len:self.obs_len + self.pred_len]
        
        if self.split == 'train':
            obs = self.augmentor(obs)
            
        return {
            'obs': torch.FloatTensor(obs),
            'pred': torch.FloatTensor(pred),
            'scene_id': idx
        }

def collate_fn(batch):
    # Packages multiple scenes into one batch
    obs = torch.stack([b['obs'] for b in batch])
    pred = torch.stack([b['pred'] for b in batch])
    scene_ids = [b['scene_id'] for b in batch]
    return {'obs': obs, 'pred': pred, 'scene_ids': scene_ids}
''',

        f"{base_dir}/models/layers.py": '''import torch
import torch.nn as nn

class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Weights for updating people (nodes) and groups (edges)
        self.lin_node = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)
    
    def forward(self, x, hyperedge_index):
        N = x.size(0)
        num_edges = hyperedge_index[1].max().item() + 1
        
        # Step 1: Project node features
        x_node = self.lin_node(x)
        
        # Step 2: Pool people into groups (Nodes -> Edges)
        hyperedge_attr = torch.zeros(num_edges, x_node.size(1), device=x.device)
        hyperedge_attr.index_add_(0, hyperedge_index[1], x_node[hyperedge_index[0]])
        
        # Average the group features
        node_counts = torch.zeros(num_edges, device=x.device)
        node_counts.index_add_(0, hyperedge_index[1], torch.ones_like(hyperedge_index[1], dtype=torch.float))
        hyperedge_attr = hyperedge_attr / (node_counts.unsqueeze(1) + 1e-8)
        
        # Step 3: Broadcast group info back to individuals (Edges -> Nodes)
        out = torch.zeros_like(x_node)
        out.index_add_(0, hyperedge_index[0], hyperedge_attr[hyperedge_index[1]])
        
        # Average by how many groups a person is in
        degrees = torch.zeros(N, device=x.device)
        degrees.index_add_(0, hyperedge_index[0], torch.ones_like(hyperedge_index[0], dtype=torch.float))
        out = out / (degrees.unsqueeze(1) + 1e-8)
        
        # Add original features (Skip connection)
        return out + x_node + self.bias

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # LSTM reads the past trajectory steps
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        
        # Get the final memory state of the LSTM
        _, (h_n, _) = self.lstm(x)
        return h_n[-1].view(B, N, self.hidden_dim)

class TrajectoryDecoder(nn.Module):
    def __init__(self, hidden_dim, pred_len, output_dim=2):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # One-shot MLP: Predicts all future steps instantly
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )
    
    def forward(self, h):
        B, N, _ = h.shape
        out = self.mlp(h)
        
        # Reshape flat output into (Batch, Time, Nodes, XY)
        out = out.view(B, N, self.pred_len, self.output_dim)
        return out.permute(0, 2, 1, 3)
''',

        f"{base_dir}/models/hgnn.py": '''import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import HypergraphConv, TemporalEncoder, TrajectoryDecoder

class TrajectoryHGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Encodes the past
        self.encoder = TemporalEncoder(config.input_dim, config.hidden_dim)
        
        # 2. Shares info between groups
        self.hgnn_layers = nn.ModuleList([
            HypergraphConv(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
        # 3. Predicts the future
        self.decoder = TrajectoryDecoder(config.hidden_dim, config.pred_len)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, obs_traj, hyperedge_indices):
        B = obs_traj.size(0)
        
        # Compress past coordinates into hidden features
        h = self.encoder(obs_traj)
        h_social = h.clone()
        
        # Apply graph convolutions scene-by-scene
        for conv, bn in zip(self.hgnn_layers, self.batch_norms):
            h_batch = []
            for b in range(B):
                h_single = conv(h_social[b], hyperedge_indices[b])
                h_batch.append(bn(h_single))
            h_social = self.dropout(F.relu(torch.stack(h_batch)))
        
        # Combine personal history with crowd awareness, then predict
        h_combined = h + h_social
        return self.decoder(h_combined)
''',

        f"{base_dir}/utils/hypergraph_utils.py": '''import torch

class HypergraphConstructor:
    def __init__(self, radius=10.0, min_size=2, max_size=5):
        # Settings for grouping people
        self.radius = radius
        self.min_size = min_size
        self.max_size = max_size
    
    def construct_from_positions(self, positions):
        N = positions.size(0)
        
        # Fast GPU-native distance calculation (replaces Scikit-Learn DBSCAN)
        dist_matrix = torch.cdist(positions, positions)
        
        visited = torch.zeros(N, dtype=torch.bool, device=positions.device)
        hyperedges = []
        
        # Group people who have the most neighbors first
        degrees = (dist_matrix < self.radius).sum(dim=1)
        node_order = torch.argsort(degrees, descending=True)
        
        for i in node_order:
            if visited[i]: continue
            
            # Find everyone within radius
            neighbors = (dist_matrix[i] < self.radius).nonzero(as_tuple=True)[0]
            
            if len(neighbors) >= self.min_size:
                # Cap the group size to prevent massive edges
                if len(neighbors) > self.max_size:
                    dists = dist_matrix[i, neighbors]
                    neighbors = neighbors[torch.argsort(dists)[:self.max_size]]
                
                hyperedges.append(neighbors)
                visited[neighbors] = True
                
        # Give loners their own group so the network doesn't crash
        for i in range(N):
            if not visited[i]:
                hyperedges.append(torch.tensor([i], device=positions.device))
                
        # Format for PyTorch: Row 0 is Person ID, Row 1 is Group ID
        node_indices, edge_indices = [], []
        for he_idx, nodes in enumerate(hyperedges):
            node_indices.extend(nodes.tolist())
            edge_indices.extend([he_idx] * len(nodes))
            
        hyperedge_index = torch.tensor([node_indices, edge_indices], device=positions.device)
        return positions, hyperedge_index
''',

        f"{base_dir}/utils/metrics.py": '''import torch

def compute_ade(pred, gt):
    # Average Displacement Error: Average distance off across ALL future steps
    error = torch.norm(pred - gt, dim=-1)
    return error.mean().item()

def compute_fde(pred, gt):
    # Final Displacement Error: Distance off at the very LAST step
    final_error = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)
    return final_error.mean().item()

def compute_mr(pred, gt, threshold=2.0):
    # Miss Rate: Percent of predictions that missed the target by more than 2 meters
    final_error = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)
    return (final_error > threshold).float().mean().item()
''',

        f"{base_dir}/train.py": '''import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import TrajectoryDataset, collate_fn
from models.hgnn import TrajectoryHGNN
from utils.hypergraph_utils import HypergraphConstructor
from utils.metrics import compute_ade, compute_fde

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Setup model and optimizer
        self.model = TrajectoryHGNN(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.hg_constructor = HypergraphConstructor(radius=config.social_radius)
        
        # Load data
        self.train_loader = DataLoader(
            TrajectoryDataset(config.data_dir, split='train', aug_prob=config.aug_prob),
            batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            TrajectoryDataset(config.data_dir, split='val', aug_prob=0.0),
            batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
        )
        os.makedirs(config.save_dir, exist_ok=True)
        self.best_loss = float('inf')
        
    def build_graphs(self, obs):
        # Look at the most recent position to build social groups
        last_pos = obs[:, -1, :, :]
        return [self.hg_constructor.construct_from_positions(p)[1] for p in last_pos]
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            obs, gt = batch['obs'].to(self.device), batch['pred'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            graphs = self.build_graphs(obs)
            pred = self.model(obs, graphs)
            
            # Backprop
            loss = self.criterion(pred, gt)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        ade_total, fde_total, loss_total = 0, 0, 0
        
        for batch in self.val_loader:
            obs, gt = batch['obs'].to(self.device), batch['pred'].to(self.device)
            pred = self.model(obs, self.build_graphs(obs))
            
            loss_total += self.criterion(pred, gt).item()
            ade_total += compute_ade(pred, gt)
            fde_total += compute_fde(pred, gt)
            
        N = len(self.val_loader)
        return loss_total/N, ade_total/N, fde_total/N
        
    def train(self):
        print(f"Starting Training on {self.device}...")
        for epoch in range(1, 6): # Keep it short for PoC
            train_loss = self.train_epoch()
            val_loss, val_ade, val_fde = self.validate()
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | ADE: {val_ade:.3f} | FDE: {val_fde:.3f}")
            
            # Save the best weights
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({'model_state_dict': self.model.state_dict()}, f"{self.config.save_dir}/best_model.pth")
''',

        f"{base_dir}/evaluate.py": '''import torch
import matplotlib.pyplot as plt
import os
from config import Config
from models.hgnn import TrajectoryHGNN
from data.dataset import TrajectoryDataset, collate_fn
from torch.utils.data import DataLoader
from utils.hypergraph_utils import HypergraphConstructor

def plot_result(obs, pred, gt, save_path):
    # Draws the trajectories on a 2D plot
    plt.figure(figsize=(6, 6))
    for i in range(obs.shape[1]): # For each agent
        plt.plot(obs[:, i, 0], obs[:, i, 1], '-', alpha=0.5)            # Past = Faded Line
        plt.plot(gt[:, i, 0], gt[:, i, 1], '--', alpha=0.5)             # True Future = Dashed
        plt.plot(pred[:, i, 0], pred[:, i, 1], '-*', linewidth=2)       # Prediction = Starred Line
        plt.scatter(obs[-1, i, 0], obs[-1, i, 1], s=50)                 # Current position
        
    plt.title('Solid: Prediction | Dashed: Ground Truth')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, config):
    print("Running Evaluation & Saving Plots...")
    os.makedirs('results', exist_ok=True)
    
    # Load model
    model = TrajectoryHGNN(config).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device)['model_state_dict'])
    model.eval()
    
    # Load test data
    loader = DataLoader(TrajectoryDataset(config.data_dir, split='test'), batch_size=1, collate_fn=collate_fn)
    builder = HypergraphConstructor(radius=config.social_radius)
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3: break # Just plot 3 examples
            
            obs = batch['obs'].to(config.device)
            graphs = [builder.construct_from_positions(obs[0, -1])[1]]
            pred = model(obs, graphs)
            
            # Save plot
            plot_result(obs[0].cpu(), pred[0].cpu(), batch['pred'][0].cpu(), f'results/pred_{i}.png')
            print(f"Saved results/pred_{i}.png")
''',

        f"{base_dir}/main.py": '''import argparse
import torch
import numpy as np
import random
from config import Config

def set_seed(seed=42):
    # Lock randomness so results are repeatable
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    args = parser.parse_args()
    
    config = Config()
    set_seed(config.seed)
    
    # Launch standard training loop or visual evaluation
    if args.mode == 'train':
        from train import Trainer
        Trainer(config).train()
    else:
        from evaluate import evaluate_model
        evaluate_model(f'{config.save_dir}/best_model.pth', config)
'''
    }
    
    # Write everything to disk
    for file_path, content in files.items():
        with open(file_path, "w") as f:
            f.write(content.strip() + "\n")
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_project()
    print("\nProject generation complete! Navigate to 'hgnn_trajectory' to begin.")