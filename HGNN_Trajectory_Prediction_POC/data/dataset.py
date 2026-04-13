import os
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
