import torch
import matplotlib.pyplot as plt

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import ntk_divergence

def run_ablation_duration(device="cuda"):
    epochs_list = [5000, 10000, 20000, 50000,100000]
    divergences = []

    # Data
    X, Y, Y_signed = generate_circle_dataset(device=device)

    # Model (fixed)
    model = NTKFullyConnectedNetwork(
        input_dim=2,
        num_classes=1,
        depth=3,
        width=2000
    ).to(device).double()

    # Initial NTK
    K0 = compute_empirical_ntk(model, X, device=device)

    for T in epochs_list:
        print(f"\nTraining for {T} epochs")

        history = train_with_cross_entropy(
            model,
            X,
            Y,
            epochs=T,
            lr=0.1,
            ntk_epochs=[T],
            device=device,
        )

        Kt = history["ntk"][T]
        div = ntk_divergence(Kt, K0).item()
        divergences.append(div)

        print(f"NTK divergence at {T} epochs = {div:.4f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, divergences, marker="o", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Training Epochs (log scale)")
    plt.ylabel(r"$\|K_T - K_0\|_F / n^2$")
    plt.title("Ablation 6: NTK Divergence vs Training Duration")
    plt.grid(alpha=0.3)
    plt.savefig("results/plots/ablation_duration.png", dpi=300)
    plt.show()

    return epochs_list, divergences


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_ablation_duration(device=device)