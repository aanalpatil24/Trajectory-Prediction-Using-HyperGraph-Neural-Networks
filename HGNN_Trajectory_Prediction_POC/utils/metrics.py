import torch

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
