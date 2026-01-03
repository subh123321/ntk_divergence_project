import os
import torch
import matplotlib.pyplot as plt

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from models.utils import get_activation
from ntk.eigenvalue_analysis import ntk_divergence




def run_ablation_activation(device="cuda"):
    os.makedirs("results/plots", exist_ok=True)

    activations = ["relu", "gelu", "tanh"]
    divergences = []

    # Dataset
    X, Y, Y_signed = generate_circle_dataset(device=device)

    for act in activations:
        print(f"\nRunning FCN with activation = {act}")

        # FCN only
        model = NTKFullyConnectedNetwork(
            input_dim=2,
            num_classes=1,
            depth=3,
            width=2000,
        ).to(device).double()

        # Initial NTK
        K0 = compute_empirical_ntk(model, X, device=device)

        # Training
        history = train_with_cross_entropy(
            model,
            X,
            Y,
            epochs=10000,
            lr=0.1,
            ntk_epochs=[10000],
            device=device,
        )

        # Final NTK
        Kt = history["ntk"][10000]
        div = ntk_divergence(Kt, K0).item()
        divergences.append(div)

        print(f"NTK divergence ({act}) = {div:.4f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(activations, divergences)
    plt.ylabel(r"$\|K_T - K_0\|_F / n^2$")
    plt.title("Ablation 7: NTK Divergence vs Activation (FCN)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/ablation_activation.png", dpi=300)
    plt.show()

    return activations, divergences


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_ablation_activation(device=device)
