import torch
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
