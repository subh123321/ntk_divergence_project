import numpy as np
import torch

def generate_circle_dataset(n_points=6, device='cuda'):
    """
    Generate synthetic circle dataset with alternating labels.
    
    Args:
        n_points: Number of points on unit circle (default: 6)
        device: 'cuda' or 'cpu'
    
    Returns:
        X: Points on unit circle (n_points, 2)
        Y: Binary labels (n_points,)
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Alternating labels: 0, 1, 0, 1, 0, 1
    Y = np.array([i % 2 for i in range(n_points)])

    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.long, device=device)
    
    return X, Y

