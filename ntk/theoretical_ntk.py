# ntk/ntk_theoretical.py

import torch
import numpy as np


@torch.no_grad()
def _flatten_grads(model):
    return torch.cat([
        p.grad.flatten()
        for p in model.parameters()
        if p.grad is not None
    ])
def generate_circle_grid(n_points=400, device="cpu", dtype=torch.double):
    """
    Generate dense points on unit circle for NTK kernel plotting.
    """
    thetas = torch.linspace(-np.pi, np.pi, n_points)
    X = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)
    return thetas, X.to(device=device, dtype=dtype)


def compute_ntk_function(model, X, x0):
    """
    Compute NTK kernel function K(x, x0) for many x.

    Args:
        model: neural network
        X: (N, d) evaluation points on the circle
        x0: (1, d) fixed reference point

    Returns:
        K_vals: (N,) NTK values
    """
    model.eval()

    # Gradient at reference point
    model.zero_grad()
    model(x0).backward()
    g0 = _flatten_grads(model).detach()

    K_vals = []

    for x in X:
        model.zero_grad()
        model(x.unsqueeze(0)).backward()
        gx = _flatten_grads(model)
        K_vals.append(torch.dot(gx, g0).item())

    return np.array(K_vals)
