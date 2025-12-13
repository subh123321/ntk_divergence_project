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

def generate_separable_circle(n_points=100, margin=0.1, noise_level=0.0, device='cuda'):
    """
    Generate circle dataset with controllable separability.
    
    Args:
        n_points: Number of points
        margin: Separation margin (higher = more separable)
        noise_level: Label noise probability
        device: 'cuda' or 'cpu'
    """
    # Inner circle (class 0)
    n_inner = n_points // 2
    theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    r_inner = np.random.uniform(0.3, 0.5 - margin, n_inner)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), 
                                r_inner * np.sin(theta_inner)])
    Y_inner = np.zeros(n_inner)
    
    # Outer circle (class 1)
    n_outer = n_points - n_inner
    theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
    r_outer = np.random.uniform(0.5 + margin, 0.8, n_outer)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer),
                                r_outer * np.sin(theta_outer)])
    Y_outer = np.ones(n_outer)
    
    X = np.vstack([X_inner, X_outer])
    Y = np.concatenate([Y_inner, Y_outer])
    
    # Add label noise
    if noise_level > 0:
        flip_mask = np.random.rand(n_points) < noise_level
        Y[flip_mask] = 1 - Y[flip_mask]
    
    # Shuffle
    perm = np.random.permutation(n_points)
    X, Y = X[perm], Y[perm]
    
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.long, device=device)
    
    return X, Y