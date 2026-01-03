"""
Empirical Neural Tangent Kernel (NTK) computation.

K(x_i, x_j) = <∇_θ f(x_i), ∇_θ f(x_j)>
"""

import torch

def compute_ntk_diagonal(model, x):
    """
    Compute K(x, x) = ||∇_θ f(x)||^2
    """
    model.zero_grad()
    output = model(x.unsqueeze(0))
    output.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())

    g = torch.cat(grads)
    return torch.dot(g, g).item()

def compute_empirical_ntk(model, X, device="cuda", max_samples=300):
    """
    Compute empirical NTK matrix.

    Args:
        model: neural network with scalar output
        X: input tensor (n, d)
        device: 'cuda' or 'cpu'
        max_samples: safety limit for n

    Returns:
        K: NTK matrix of shape (n, n)
    """

    model.eval()
    X = X.to(device)
    
    n = X.shape[0]
    if n > max_samples:
        raise RuntimeError(
            f"NTK computation aborted: n={n} too large (max={max_samples})"
        )

    num_params = sum(p.numel() for p in model.parameters())

    jacobians = []

    for i in range(n):
        model.zero_grad(set_to_none=True)

        x = X[i : i + 1]
        output = model(x)

        # Enforce scalar output
        if output.numel() != 1:
            raise ValueError(
                "Model output must be scalar for NTK computation."
            )

        output.backward()

        grads = []
        for p in model.parameters():
            if p.grad is None:
                grads.append(torch.zeros(p.numel(), device=device))
            else:
                grads.append(p.grad.detach().flatten())

        jacobians.append(torch.cat(grads))

    # Stack Jacobians
    J = torch.stack(jacobians)  # (n, num_params)

    # NTK matrix
    K = J @ J.T  # (n, n)
    K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    return K
