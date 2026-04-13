import torch
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
