import numpy as np
import torch

def generate_circle_dataset(
    n_points=6,
    device='cuda',
    dtype=torch.float64,
    mode="hard",
):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    X = np.stack([np.cos(theta), np.sin(theta)], axis=1)

 # Labels control separability
    if mode == "easy":
        # Linearly separable
        Y = np.array([1, 1, 1, 0, 0, 0])
    elif mode == "medium":
        # Partially separable
        Y = np.array([1, 1, 0, 0, 1, 0])
    elif mode == "hard":
        # Maximally non-separable (paper default)
        Y = np.array([i % 2 for i in range(n_points)])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    Y_signed = 2 * Y - 1

    X = torch.tensor(X, dtype=dtype, device=device)
    Y = torch.tensor(Y, dtype=dtype, device=device)
    Y_signed = torch.tensor(Y_signed, dtype=dtype, device=device)


    X=X.double()
    Y=Y.double()
    
    return X, Y, Y_signed


