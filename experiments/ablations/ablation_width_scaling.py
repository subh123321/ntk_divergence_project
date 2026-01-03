import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.generate_circle import generate_circle_dataset
from models.fcn import NTKFullyConnectedNetwork
from training.gradient_flow import train_with_cross_entropy
from ntk.ntk_computation import compute_empirical_ntk
from ntk.eigenvalue_analysis import ntk_divergence

# -----------------------------
# Config
# -----------------------------
WIDTHS = [64, 128, 256, 512, 1024, 2048]
DEPTH = 3
EPOCHS = 10000
LR = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs("results/plots", exist_ok=True)

# -----------------------------
# Data
# -----------------------------
X, Y , Y_signed= generate_circle_dataset(device=device)

# -----------------------------
# Ablation loop
# -----------------------------
results = []

for width in WIDTHS:
    print("\n" + "=" * 60)
    print(f"Ablation 1 — width = {width}")
    print("=" * 60)

    model = NTKFullyConnectedNetwork(
        input_dim=2,
       # num_classes=1,
        depth=DEPTH,
        width=width
    ).to(device).double()

    # Initial NTK
    K0 = compute_empirical_ntk(model, X, device=device)

    # Train
    history = train_with_cross_entropy(
        model,
        X,
        Y,
        epochs=EPOCHS,
        lr=LR,
        ntk_epochs=[EPOCHS],
        device=device,
    )

    # Final NTK
    KT = history["ntk"][EPOCHS]

    # Divergence
    div = ntk_divergence(KT, K0)

    print(f"NTK divergence = {div:.6e}")
    results.append((width, div))

# -----------------------------
# Save CSV
# -----------------------------
with open("results/ablation_width.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["width", "ntk_divergence"])
    for r in results:
        writer.writerow(r)

# -----------------------------
# Plot
# -----------------------------
widths, divergences = zip(*results)

plt.figure(figsize=(6, 4))
plt.plot(widths, divergences, marker="o", linewidth=2)
plt.xscale("log")
plt.xlabel("Width (log scale)")
plt.ylabel(r"$\|K_T - K_0\|_F / n^2$")
plt.title("Ablation 1: NTK Divergence vs Width")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/ablation_width.png", dpi=300)
plt.show()

print("\n✅ Ablation 1 complete.")
