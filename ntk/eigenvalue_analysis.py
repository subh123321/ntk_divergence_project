import numpy as np
import torch


def compute_eigenvalues(K):
    if isinstance(K, torch.Tensor):
        K = K.cpu().double().numpy()

    if not np.isfinite(K).all():
        raise ValueError("NTK contains NaN or Inf; eigenvalues undefined")
    
    return np.sort(np.linalg.eigvalsh(K))[::-1]


def ntk_divergence(Kt, K0):
    if isinstance(Kt, torch.Tensor):
        Kt = Kt.detach().cpu().numpy()
    if isinstance(K0, torch.Tensor):
        K0 = K0.detach().cpu().numpy()

    return np.linalg.norm(Kt - K0, "fro") / (K0.shape[0] ** 2)

