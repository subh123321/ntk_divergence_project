"""
Reproduce the 6-point circle experiment from:

ICLR 2025 — Divergence of Empirical Neural Tangent Kernel
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.device_utils import get_device

device = get_device()

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from models.utils import set_seed
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import compute_eigenvalues, ntk_divergence
from ntk.theoretical_ntk import (
    generate_circle_grid,
    compute_ntk_function
)

# ------------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------------
def plot_output_vs_log_epoch(history, y_signed, save_path):
    """
    Figure 1 (paper):
    Network outputs f(x_i) vs log10(epoch)
    """
    outputs = np.array(history["outputs"])  # (epochs, n)
    epochs = np.arange(1, outputs.shape[0] + 1)
    log_epochs = np.log10(epochs)

    plt.figure(figsize=(7, 5))

    for i in range(outputs.shape[1]):
        # Display labels as 0/1 (paper convention)
        y01 = int((y_signed[i].item() + 1) / 2)
        plt.plot(
            log_epochs,
            outputs[:, i],
            linewidth=2,
            label=fr"$x_{i+1}$ (y={y01})",
        )

    plt.xlabel(r"$\log_{10}(\mathrm{epoch})$", fontsize=12)
    plt.ylabel(r"Network output $f(x_i)$", fontsize=12)
    plt.title("Divergence of Network Outputs", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_ntk_vs_polar_angle(K_theta, thetas, title, save_path):
    """
    Figure 2 / 3 (paper):
    NTK kernel function K(x(θ), x0) vs θ
    """
    plt.figure(figsize=(6, 4))
    plt.plot(
        thetas.cpu().numpy(),
        K_theta,
        linewidth=2
    )
    plt.xlabel(r"Polar angle $\theta$", fontsize=12)
    plt.ylabel(r"$K(x(\theta), x_0)$", fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------
def run_circle_experiment(
    width=2000,
    depth=3,
    epochs=10000,
    device="cuda",
):
    print("=" * 60)
    print("6-POINT CIRCLE EXPERIMENT (NTK DIVERGENCE)")
    print("=" * 60)

    set_seed(0)

    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------
    X, Y01, y_signed = generate_circle_dataset(device=device)
    n = X.shape[0]

    print(f"Dataset: {n} points on unit circle")
    print(f"Labels (0/1): {Y01.cpu().numpy()}")

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model = NTKFullyConnectedNetwork(
        input_dim=2,
        num_classes=1,
        depth=depth,
        width=width,
    ).to(device).double()

    print(f"Model: NTK-FCN, depth={depth}, width={width}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------------------
    # Initial NTK (empirical, training set)
    # --------------------------------------------------------------
    print("\nComputing initial empirical NTK")
    K0 = compute_empirical_ntk(model, X, device=device)
    eigs0 = compute_eigenvalues(K0)

    print(f"Initial λ_min = {eigs0[-1]:.6e}")
    print(f"Initial λ_max = {eigs0[0]:.6e}")

    # --------------------------------------------------------------
    # NTK kernel function at initialization (Figure 2)
    # --------------------------------------------------------------
    thetas, X_theta = generate_circle_grid(
        n_points=400,
        device=device,
        dtype=torch.double
    )
    x0 = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.double)

    K_theta_0 = compute_ntk_function(model, X_theta, x0)

    # --------------------------------------------------------------
    # Training (FULL-BATCH SGD, paper-faithful)
    # --------------------------------------------------------------
    history = train_with_cross_entropy(
        model,
        X,
        Y01,                    # 0/1 labels for BCE
        epochs=epochs,
        lr=0.1,
        ntk_epochs=[1, 2000, 5000, epochs],
        device=device,
    )

    # --------------------------------------------------------------
    # Final NTK (empirical, training set)
    # --------------------------------------------------------------
    Kt = history["ntk"][epochs]
    eigs_t = compute_eigenvalues(Kt)

    print("\nFinal NTK statistics:")
    print(f"Final λ_min = {eigs_t[-1]:.6e}")
    print(f"Final λ_max = {eigs_t[0]:.6e}")

    # --------------------------------------------------------------
    # NTK divergence
    # --------------------------------------------------------------
    div = ntk_divergence(Kt, K0)
    print(f"\nNTK Divergence: ||K_t − K_0||_F / n² = {div:.6e}")

    # --------------------------------------------------------------
    # NTK kernel function after training (Figure 3)
    # --------------------------------------------------------------
    K_theta_t = compute_ntk_function(model, X_theta, x0)

    # --------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------
    os.makedirs("results/plots", exist_ok=True)

    plot_output_vs_log_epoch(
        history,
        y_signed,
        save_path="results/plots/output_vs_log_epoch.png",
    )

    plot_ntk_vs_polar_angle(
        K_theta_0,
        thetas,
        title="NTK vs Polar Angle (t = 0)",
        save_path="results/plots/ntk_vs_angle_initial.png",
    )

    plot_ntk_vs_polar_angle(
        K_theta_t,
        thetas,
        title="NTK vs Polar Angle (final)",
        save_path="results/plots/ntk_vs_angle_final.png",
    )

    print("\n✅ Circle experiment completed successfully.")

    return {
        "K0": K0,
        "Kt": Kt,
        "eigs0": eigs0,
        "eigs_t": eigs_t,
        "divergence": div,
        "history": history,
    }

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_circle_experiment(
        width=2000,
        depth=3,
        epochs=10000,
        device=device,
    )
