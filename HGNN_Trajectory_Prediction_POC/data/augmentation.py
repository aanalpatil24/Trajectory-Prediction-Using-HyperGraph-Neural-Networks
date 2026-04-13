import numpy as np
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
