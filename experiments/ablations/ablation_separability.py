import torch
import matplotlib.pyplot as plt

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import ntk_divergence

def run_ablation_separability(device="cuda"):
    modes = ["easy", "medium", "hard"]
    divergences = []

    for mode in modes:
        print(f"\nRunning separability = {mode}")

        X, Y, Y_signed = generate_circle_dataset(device=device, mode=mode)

        model = NTKFullyConnectedNetwork(
            input_dim=2,
            num_classes=1,
            depth=3,
            width=2000
        ).to(device).double()

        # Initial NTK
        K0 = compute_empirical_ntk(model, X, device=device)

        history = train_with_cross_entropy(
            model,
            X,
            Y,
            epochs=10000,
            lr=0.1,
            ntk_epochs=[10000],
            device=device
        )

        Kt = history["ntk"][10000]
        div = ntk_divergence(Kt, K0).item()
        divergences.append(div)

        print(f"NTK divergence ({mode}) = {div:.4f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(modes, divergences)
    plt.ylabel(r"$\|K_T - K_0\|_F / n^2$")
    plt.title("Ablation 5: NTK Divergence vs Dataset Separability")
    plt.grid(alpha=0.3)
    plt.savefig("results/plots/ablation_separability.png", dpi=300)
    plt.show()

    return modes, divergences

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_ablation_separability(device=device)