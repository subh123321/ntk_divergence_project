import torch
import numpy as np
import matplotlib.pyplot as plt

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from training.gradient_flow import train_with_loss
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import ntk_divergence


"""def ntk_divergence(Kt, K0):
    Kt = Kt.detach().cpu()
    K0 = K0.detach().cpu()
    return torch.norm(Kt - K0, p="fro") / (K0.shape[0] ** 2)"""

def run_ablation_loss(device="cuda"):
    losses = ["bce", "mse", "hinge", "focal"]
    divergences = []

    X, Y, Y_signed = generate_circle_dataset(device=device)

    for loss_type in losses:
        print(f"\nRunning loss = {loss_type}")

        model = NTKFullyConnectedNetwork(
            input_dim=2,
            num_classes=1,
            depth=3,
            width=2000
        ).to(device).double()
        
        K0 = compute_empirical_ntk(model, X, device=device)

        history = train_with_loss(
            model,
            X,
            Y,
            Y_signed,
            loss_type=loss_type,
            epochs=10000,
            lr=0.1,
            ntk_epochs=(10000,),
            device=device,
        )

        Kt = history["ntk"][10000]
        div = ntk_divergence(Kt, K0).item()
        divergences.append(div)

        print(f"NTK divergence ({loss_type}) = {div:.4f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(losses, divergences)
    plt.ylabel(r"$\|K_T - K_0\|_F / n^2$")
    plt.title("Ablation 2: NTK Divergence vs Loss Function")
    plt.grid(alpha=0.3)
    plt.savefig("results/plots/ablation_loss.png", dpi=300)
    plt.show()

    return losses, divergences

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_ablation_loss(device=device)