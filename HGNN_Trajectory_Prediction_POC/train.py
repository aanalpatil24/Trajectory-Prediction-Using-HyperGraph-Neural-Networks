import os
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
