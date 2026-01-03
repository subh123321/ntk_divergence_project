import torch
import matplotlib.pyplot as plt

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import ntk_divergence


def scale_model_weights(model, scale):
    """
    Multiply all trainable parameters by a scalar.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)


"""def ntk_divergence(Kt, K0):
    Kt=Kt.cpu()
    K0=K0.cpu()
    return torch.norm(Kt - K0, p="fro") / (K0.shape[0] ** 2)"""


def run_ablation_init(device="cuda"):
    print("=" * 60)
    print("Ablation 4: Initialization Sensitivity")
    print("=" * 60)

    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    divergences = []

    # Dataset (fixed)
    X, Y, Y_signed = generate_circle_dataset(device=device)

    for scale in scales:
        print(f"\nRunning init scale = {scale}")

        # Model
        model = NTKFullyConnectedNetwork(
            input_dim=2,
            num_classes=1,
            depth=3,
            width=2000,
        ).to(device).double()

        # 🔑 SCALE INITIALIZATION
        scale_model_weights(model, scale)

        # Initial NTK
        K0 = compute_empirical_ntk(model, X, device=device)

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

        # Final NTK
        Kt = history["ntk"][10000]

        div = ntk_divergence(Kt, K0).item()
        divergences.append(div)

        print(f"NTK divergence (scale={scale}) = {div:.4f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(scales, divergences, marker="o", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Initialization Scale (log)")
    plt.ylabel(r"$\|K_T - K_0\|_F / n^2$")
    plt.title("Ablation 4: NTK Divergence vs Initialization Scale")
    plt.grid(alpha=0.3)

    plt.savefig("results/plots/ablation_init.png", dpi=300)
    plt.show()

    return scales, divergences


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_ablation_init(device=device)