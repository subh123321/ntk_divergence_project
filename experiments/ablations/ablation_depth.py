import torch
import numpy as np
import matplotlib.pyplot as plt

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import compute_eigenvalues, ntk_divergence


def run_ablation_depth(device="cuda"):
    depths = [2, 3, 4, 5]
    width = 2000   # fixed width (paper-style)
    
    divergences = []
    top_eigs = []

    # Data
    X, Y, Y_signed = generate_circle_dataset(device=device)

    for depth in depths:
        print(f"\n🔹 Running depth = {depth}")

        model = NTKFullyConnectedNetwork(
            input_dim=2,
            num_classes=1,
            depth=depth,
            width=width,
        ).to(device).double()

        # Initial NTK
        K0 = compute_empirical_ntk(model, X, device=device)
        eigs0 = compute_eigenvalues(K0)

        # Train
        history = train_with_cross_entropy(
            model,
            X,
            Y,
            epochs=10000,
            lr=0.1,
            ntk_epochs=[10000],
            device=device,
        )

        Kt = history["ntk"][10000]
        eigs_t = compute_eigenvalues(Kt)

        # Metrics
        div = ntk_divergence(Kt, K0)
        divergences.append(div)

        top_eigs.append(eigs_t[:5])  # top-5 eigenvalues

        print(f"  NTK divergence = {div:.4e}")

    # Convert to arrays
    top_eigs = np.array(top_eigs)

    # -------------------------------
    # Plot 1: NTK Divergence vs Depth
    # -------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(depths, divergences, marker="o", linewidth=2)
    plt.xlabel("Network Depth")
    plt.ylabel(r"$\|K_T - K_0\|_F / n^2$")
    plt.title("Ablation 3: NTK Divergence vs Depth")
    plt.grid(alpha=0.3)
    plt.savefig("results/plots/ablation_depth_divergence.png", dpi=300)
    plt.show()

    # -----------------------------------
    # Plot 2: Top NTK Eigenvalues vs Depth
    # -----------------------------------
    plt.figure(figsize=(6, 4))
    for k in range(top_eigs.shape[1]):
        plt.plot(depths, top_eigs[:, k], marker="o", label=f"λ{k+1}")

    plt.xlabel("Network Depth")
    plt.ylabel("Eigenvalue")
    plt.title("Top NTK Eigenvalues vs Depth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("results/plots/ablation_depth_eigenvalues.png", dpi=300)
    plt.show()

    print("\n✅ Ablation 3 completed successfully.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_ablation_depth(device=device)