"""
MNIST NTK divergence experiment (Section 6.2, ICLR 2025)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utils.device_utils import get_device

device = get_device()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.mnist_loader import generate_mnist_binary
from models.fcn import  NTKFullyConnectedNetwork
from models.utils import set_seed
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_ntk_diagonal

# --------------------------------------------------
def run_mnist_experiment(device="cuda"):
    set_seed(0)

    # Paper settings
    width = 500
    depth = 4
    lr = 0.5
    epochs = 100000
    ntk_interval = 200
    discard_first = 10

    # -----------------------
    # Data
    # -----------------------
    X, Y, _ = generate_mnist_binary(
        n_samples=2000,
        device=device,
        dtype=torch.float64,
    )

    # Choose 3 fixed samples
    sample_ids = [0, 1, 2]
    X_samples = X[sample_ids]

    # -----------------------
    # Model
    # -----------------------
    model = NTKFullyConnectedNetwork(
        input_dim=784,
        num_classes=1,
        depth=depth,
        width=width,
    ).to(device).double()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    ntk_history = {i: [] for i in sample_ids}
    epoch_list = []

    # -----------------------
    # Training loop
    # -----------------------
    output_norm_history = []
    epoch_history = []

    for epoch in range(1, epochs + 1):
       
        optimizer.zero_grad()

        outputs = model(X).squeeze()
        loss = criterion(outputs, Y)

        loss.backward()
        optimizer.step()

    # ---- NEW: track output divergence ----
        with torch.no_grad():
          max_output = torch.max(torch.abs(outputs)).item()
          output_norm_history.append(max_output)
          epoch_history.append(epoch)

        if epoch % ntk_interval == 0:
            epoch_list.append(epoch)
            for idx, x in zip(sample_ids, X_samples):
               ntk_val = compute_ntk_diagonal(model, x)
               ntk_history[idx].append(ntk_val)

        if epoch % 5000 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss = {loss.item():.4f}")


    # -----------------------
    # Plot (Figure 4)
    # -----------------------
    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(sample_ids):
        ntk_vals = np.array(ntk_history[idx])[discard_first:]
        epochs_used = np.array(epoch_list)[discard_first:]

        smooth_vals = gaussian_filter1d(ntk_vals, sigma=2)

        plt.subplot(1, 3, i + 1)
        plt.plot(epochs_used, ntk_vals, label="Original", alpha=0.7)
        plt.plot(epochs_used, smooth_vals, "r--", label="Smoothed")
        plt.xlabel("Epoch")
        plt.ylabel("NTK Value")
        plt.title(f"Sample {i+1}")
        plt.grid(alpha=0.3)
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig("results/plots/mnist_ntk_divergence.png", dpi=300)
    plt.show()

    print("✅ MNIST NTK divergence experiment completed.")

# --------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_mnist_experiment(device)
