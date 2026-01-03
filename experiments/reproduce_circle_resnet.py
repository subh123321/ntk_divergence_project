import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.generate_circle import generate_circle_dataset
from models.resnet import FullyConnectedResNet
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import compute_eigenvalues, ntk_divergence

import numpy as np
import matplotlib.pyplot as plt


def log_epoch_indices(T, num=120):
    """
    Generate log-spaced epoch indices safely.
    Returns VALID indices in [0, T-1].
    """
    idx = np.unique(
        np.logspace(0, np.log10(T), num=num)
        .astype(int)
        - 1  # epoch -> index
    )
    idx = idx[(idx >= 0) & (idx < T)]
    return idx


def plot_output_vs_log_epoch(history, save_path):
    """
    Plot network outputs f_t(x_i) vs log10(epoch)
    (Paper-style, stable, no artifacts)
    """
    # ----------------------------------
    # Load outputs
    # ----------------------------------
    outputs = np.array(history["outputs"])  # shape (T, n)
    T, n = outputs.shape

    # ----------------------------------
    # Log-spaced subsampling
    # ----------------------------------
    idx = log_epoch_indices(T, num=150)

    outputs = outputs[idx]          # (len(idx), n)
    epochs = idx + 1                # convert index -> epoch
    log_epochs = np.log10(epochs)

    # ----------------------------------
    # Plot
    # ----------------------------------
    plt.figure(figsize=(7.5, 5))

    for i in range(n):
        plt.plot(
            log_epochs,
            outputs[:, i],
            linewidth=2,
            label=fr"$x_{i+1}$"
        )

    plt.xlabel(r"$\log_{10}(\mathrm{epoch})$", fontsize=12)
    plt.ylabel(r"$f_t(x_i)$", fontsize=12)
    plt.title("Network Output Divergence", fontsize=13)

    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

    


def plot_ntk_eigenvalues(history, n_top=6, save_path=None):
    """
    Plot top NTK eigenvalues vs epoch
    """
    ntk_epochs = sorted(history["ntk"].keys())
    eigenvalue_list = []

    for epoch in ntk_epochs:
        K = history["ntk"][epoch]
        eigs = compute_eigenvalues(K)
        eigenvalue_list.append(eigs[:n_top])

    ntk_eigs = np.array(eigenvalue_list)

    plt.figure(figsize=(7, 5))

    for i in range(n_top):
        plt.plot(
            ntk_epochs,
            ntk_eigs[:, i],
            marker="o",
            linewidth=2,
            label=fr"$\lambda_{i+1}$"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Eigenvalue")
    plt.title("ResNet NTK Eigenvalue Evolution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()



def run_resnet_circle(
    width=2000,
    depth=6,
    epochs=10000,
    lr=0.1,
    device="cuda"
):
    print("=" * 60)
    print("RESNET CIRCLE EXPERIMENT (NTK DIVERGENCE)")
    print("=" * 60)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    X, Y, Y_signed = generate_circle_dataset(device=device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = FullyConnectedResNet(
        input_dim=2,
        width=width,
        depth=depth,
        num_classes=1
    ).to(device)

    model=model.double()

    print(f"ResNet depth={depth}, width={width}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------
    # Initial NTK
    # --------------------------------------------------
    print("\nComputing initial NTK...")
    K0 = compute_empirical_ntk(model, X, device=device)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    history = train_with_cross_entropy(
        model,
        X,
        Y,
        epochs=epochs,
        lr=lr,
        #compute_ntk_every=2000
        ntk_epochs=[1, 2000, 5000, epochs],
        device=device
    )

    # --------------------------------------------------
    # Final NTK
    # --------------------------------------------------
    # ---- Get final NTK safely ----
    ntk_epochs = sorted(history["ntk"].keys())
    last_epoch = ntk_epochs[-1]
    Kt = history["ntk"][last_epoch]

    print("History keys:", history.keys())
    print("NTK epochs:", history["ntk_epochs"])

    # --------------------------------------------------
    # NTK divergence
    # --------------------------------------------------
    div = ntk_divergence(Kt, K0)

    print("\nNTK divergence (relative):", div)

    # --------------------------------------------------
    # Eigenvalue reports
    # --------------------------------------------------
    eigs0 = compute_eigenvalues(K0)
    eigs_t = compute_eigenvalues(Kt)

    print("Initial NTK:")
    print(f"  λ_max = {eigs0[0]:.6e}")
    print(f"  λ_min = {eigs0[-1]:.6e}")

    print("Final NTK:")
    print(f"  λ_max = {eigs_t[0]:.6e}")
    print(f"  λ_min = {eigs_t[-1]:.6e}")

    
    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------
    os.makedirs("results/plots", exist_ok=True)

    plot_output_vs_log_epoch(
       history,
       save_path="results/plots/resnet_output_vs_log_epoch.png"
)

    plot_ntk_eigenvalues(
       history,
       n_top=6,
       save_path="results/plots/resnet_ntk_eigenvalues.png"
)

    return history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_resnet_circle(device=device)
