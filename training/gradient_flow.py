"""
Training module for reproducing NTK divergence experiments.

Implements FULL-BATCH gradient descent (SGD) with:
- Cross-entropy (BCE with logits)
- Epoch-based training
- NTK tracking at selected epochs

This matches the experimental setup of the ICLR 2025 paper.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ntk.ntk_computation import compute_empirical_ntk

def get_loss_function(loss_type):
    if loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type == "mse":
        return torch.nn.MSELoss()
    elif loss_type == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_type == "hinge":
        return None
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: shape (n,)
        targets: {0,1}
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return loss.mean()


def train_with_cross_entropy(
    model,
    X,
    Y,
    epochs=10000,
    lr=0.1,
    ntk_epochs=None,
    device="cuda",
):
    """
    Train network using full-batch SGD with cross-entropy loss.

    Args:
        model: neural network
        X: input data (n, d)
        Y: labels in {0,1}
        epochs: number of training epochs
        lr: learning rate (paper uses 0.1)
        ntk_epochs: list of epochs at which NTK is computed
        device: cpu or cuda

    Returns:
        history: dictionary with training statistics
    """

    model.to(device)
    X = X.to(device)
    Y = Y.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    if ntk_epochs is None:
        ntk_epochs = []

    history = {
        "loss": [],
        "outputs": [],        # shape: (epoch, n)
        "accuracy": [],
        "ntk": {}, 
        "ntk_epochs": [],             # epoch -> NTK matrix
    }

    print(f"Training for {epochs} epochs (full-batch SGD, lr={lr})")
    pbar = tqdm(range(1, epochs + 1))

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward
        outputs = model(X)  # shape (n,)
        loss = criterion(outputs, Y.float())

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            acc = (preds == Y).float().mean().item()

            history["loss"].append(loss.item())
            history["outputs"].append(outputs.detach().cpu().numpy())
            history["accuracy"].append(acc)

        # NTK computation (paper-style, at selected epochs)
        if epoch in ntk_epochs:
            print(f"\nComputing NTK at epoch {epoch}")
            K = compute_empirical_ntk(model, X, device=device)
            history["ntk"][epoch] = K.detach().cpu()
            history["ntk_epochs"].append(epoch)

        if epoch % 1000 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.3f}",
                max_out=f"{outputs.abs().max().item():.2f}",
            )

    print("Training complete.")
    return history


def train_with_loss(
    model,
    X,
    Y,
    Y_signed,
    loss_type="bce",
    epochs=10000,
    lr=0.1,
    ntk_epochs=(0, 10000),
    device="cuda",
):
    model.to(device)
    X = X.to(device)
    Y = Y.to(device)
    Y_signed = Y_signed.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {
        "ntk": {}
    }

    loss_fn = get_loss_function(loss_type)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(X).squeeze()

        if loss_type == "bce":
           loss = loss_fn(outputs, Y.float())

        elif loss_type == "mse":
           loss = loss_fn(outputs, Y_signed)

        elif loss_type == "focal":
           loss = loss_fn(outputs, Y)

        elif loss_type == "hinge":
           loss = torch.mean(torch.clamp(1 - Y_signed * outputs, min=0))

        loss.backward()
        optimizer.step()

        if epoch in ntk_epochs:
            K = compute_empirical_ntk(model, X, device=device)
            history["ntk"][epoch] = K.cpu()

    return history