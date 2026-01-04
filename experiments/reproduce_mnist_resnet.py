"""
Reproduce MNIST ResNet experiment from:

ICLR 2025 — Divergence of Empirical Neural Tangent Kernel

"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.mnist_loader import generate_mnist_binary
from models.resnet import FullyConnectedResNet
from ntk.ntk_computation import compute_ntk_diagonal

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_output_vs_log_epoch(epochs, outputs, save_path):
    log_epochs = np.log10(epochs)

    plt.figure(figsize=(7, 5))
    for i in range(outputs.shape[1]):
        plt.plot(log_epochs, outputs[:, i], linewidth=2, label=f"$x_{i+1}$")

    plt.xlabel(r"$\log_{10}(\mathrm{epoch})$")
    plt.ylabel(r"$f_t(x)$")
    plt.title("MNIST ResNet Output Divergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_ntk_diagonal(epochs, ntk_diag, save_path):
    plt.figure(figsize=(7, 5))
    for i in range(ntk_diag.shape[1]):
        plt.plot(epochs, ntk_diag[:, i], linewidth=2, label=f"$x_{i+1}$")

    plt.xlabel("Epoch")
    plt.ylabel(r"$K_t(x, x)$")
    plt.title("MNIST ResNet NTK Diagonal Divergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def run_mnist_resnet(
    width=500,
    depth=4,
    epochs=10000,
    lr=0.5,
    device="cuda",
):
    print("=" * 70)
    print("MNIST RESNET — NTK DIVERGENCE EXPERIMENT")
    print("=" * 70)
    print(f"Using device: {device}")

    # ---------------- Data ----------------
    X, Y, _ = generate_mnist_binary(
        n_samples=200,
        device=device
    )

    probe_ids = [0, 1, 2]
    X_probe = X[probe_ids]   # (3, 784)

    # ---------------- Model ----------------
    model = FullyConnectedResNet(
        input_dim=784,
        width=width,
        depth=depth,
        num_classes=1
    ).to(device).double()

    print(f"ResNet depth={depth}, width={width}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------------- Initial NTK diag ----------------
    K0_diag = [compute_ntk_diagonal(model, X_probe[i]) for i in range(3)]
    print("Initial NTK diagonal:", K0_diag)

    # ---------------- Training ----------------
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    output_history = []
    ntk_diag_history = []
    epoch_history = []

    print("\nTraining...")
    for epoch in tqdm(range(1, epochs + 1)):
        optimizer.zero_grad()
        logits = model(X).squeeze()
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        # Track after burn-in (paper discards early epochs)
        if epoch % 500 == 0 and epoch >= 2000:
            with torch.no_grad():
                out_vals = model(X_probe).cpu().numpy()
                output_history.append(out_vals)
                epoch_history.append(epoch)

            ntk_vals = [
                compute_ntk_diagonal(model, X_probe[i])
                for i in range(3)
            ]
            ntk_diag_history.append(ntk_vals)

    # ---------------- Convert to arrays ----------------
    output_history = np.array(output_history)       # (T, 3)
    ntk_diag_history = np.array(ntk_diag_history)   # (T, 3)
    epoch_history = np.array(epoch_history)

    # ---------------- Plots ----------------
    os.makedirs("results/plots", exist_ok=True)

    plot_output_vs_log_epoch(
        epoch_history,
        output_history,
        "results/plots/mnist_resnet_output_vs_log_epoch.png"
    )

    plot_ntk_diagonal(
        epoch_history,
        ntk_diag_history,
        "results/plots/mnist_resnet_ntk_diagonal.png"
    )

    print("\n✅ MNIST ResNet experiment completed correctly.")

    return {
        "epochs": epoch_history,
        "outputs": output_history,
        "ntk_diag": ntk_diag_history,
        "K0_diag": K0_diag,
    }


# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_mnist_resnet(device=device)
