import argparse
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
