import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.mnist_loader import generate_mnist_binary
from models.fcn import FullyConnectedNetwork
from ntk.ntk_computation import compute_empirical_ntk_efficient
from training.gradient_flow import train_with_cross_entropy

# Import eigenvalue analysis
from ntk.eigenvalue_analysis import (
    generate_eigenvalue_report,
    plot_eigenvalue_spectrum,
    plot_eigenvalue_evolution,
    compute_ntk_divergence,
    track_eigenvalue_evolution
)
def analyze_eigenvalues(history):
    """
    Analyze NTK eigenvalues from training history.
    Same logic as reproduce_circle.py, reused for MNIST.
    """
    print("\n" + "=" * 60)
    print("EIGENVALUE ANALYSIS")
    print("=" * 60)

    # Check if NTK data exists
    if 'ntk_matrices' not in history or len(history['ntk_matrices']) == 0:
        print("⚠️  No NTK matrices found in history!")
        print("   Make sure compute_ntk_every parameter is set in training.")
        return

    ntk_matrices = history['ntk_matrices']
    ntk_epochs = history['ntk_epochs']

    print(f"\n📊 Found {len(ntk_matrices)} NTK matrices at epochs: {ntk_epochs}")

    # ----- Initial NTK -----
    print("\n" + "─" * 60)
    print("INITIAL NTK (t = 0)")
    print("─" * 60)

    K_initial = ntk_matrices[0]
    report_initial = generate_eigenvalue_report(
        K_initial, label="Initial NTK (MNIST)"
    )

    # ----- Final NTK -----
    print("\n" + "─" * 60)
    print(f"FINAL NTK (t = {ntk_epochs[-1]})")
    print("─" * 60)

    K_final = ntk_matrices[-1]
    report_final = generate_eigenvalue_report(
        K_final, label="Final NTK (MNIST)"
    )

    # ----- NTK divergence metrics -----
    print("\n" + "─" * 60)
    print("NTK DIVERGENCE METRICS")
    print("─" * 60)

    div_frobenius = compute_ntk_divergence(
        K_final, K_initial, metric='frobenius'
    )
    div_spectral = compute_ntk_divergence(
        K_final, K_initial, metric='spectral'
    )
    div_relative = compute_ntk_divergence(
        K_final, K_initial, metric='relative'
    )

    print(f"Frobenius Norm: ||K_T − K_0||_F = {div_frobenius:.4f}")
    print(f"Spectral Norm:  ||K_T − K_0||_2 = {div_spectral:.4f}")
    print(f"Relative NTK Change: {div_relative:.4f} ({div_relative*100:.2f}%)")

    # ----- Verdict -----
    print("\n" + "─" * 60)
    print("VERDICT")
    print("─" * 60)

    if div_relative > 0.1:
        print("✅ SIGNIFICANT NTK DIVERGENCE DETECTED (>10%)")
        print("   Confirms breakdown of lazy training regime.")
    elif div_relative > 0.05:
        print("✓ Moderate NTK divergence detected (5–10%)")
        print("  Consistent with paper’s findings.")
    elif div_relative > 0.01:
        print("⚠️  Weak NTK divergence (1–5%)")
        print("   Consider longer training or larger learning rate.")
    else:
        print("❌ No significant NTK divergence (<1%)")
        print("   Training may be insufficient.")

    # ----- Track eigenvalue evolution -----
    print("\n" + "─" * 60)
    print("EIGEN EVALUATION STATISTICS")
    print("─" * 60)

    eigenvalue_history, statistics = track_eigenvalue_evolution(
        ntk_matrices, ntk_epochs
    )

    print("\nMinimum eigenvalue over time:")
    for epoch, min_eig in zip(ntk_epochs, statistics['min_eigenvalue_over_time']):
        print(f"  Epoch {epoch:5d}: λ_min = {min_eig:.6e}")

    print("\nCondition number over time:")
    for epoch, cond in zip(ntk_epochs, statistics['condition_number_over_time']):
        print(f"  Epoch {epoch:5d}: κ = {cond:.2e}")

    # ----- Generate plots -----
    print("\n" + "─" * 60)
    print("GENERATING EIGENVALUE PLOTS")
    print("─" * 60)

    plot_eigenvalue_spectrum(
        report_initial['eigenvalues'],
        title="Initial NTK Eigenvalue Spectrum (MNIST)",
        save_path='results/plots/eigenvalues_initial_mnist.png'
    )

    plot_eigenvalue_spectrum(
        report_final['eigenvalues'],
        title="Final NTK Eigenvalue Spectrum (MNIST)",
        save_path='results/plots/eigenvalues_final_mnist.png'
    )

    plot_eigenvalue_evolution(
        eigenvalue_history,
        ntk_epochs,
        n_top=6,
        save_path='results/plots/eigenvalue_evolution_mnist.png'
    )

    print("\n" + "=" * 60)
    print("EIGENVALUE ANALYSIS COMPLETE")
    print("=" * 60)

    return {
        'initial_report': report_initial,
        'final_report': report_final,
        'divergence': {
            'frobenius': div_frobenius,
            'spectral': div_spectral,
            'relative': div_relative
        },
        'eigenvalue_history': eigenvalue_history,
        'statistics': statistics
    }


def run_mnist_experiment(device='cuda'):
    print("=" * 60)
    print("MNIST EXPERIMENT")
    print("=" * 60)

    X, Y = generate_mnist_binary(
        digits=(0, 1),
        n_samples=200,
        device=device
    )

    model = FullyConnectedNetwork(
        input_dim=784,
        num_classes=1,
        depth=3,
        width=2000,
        activation='relu'
    ).to(device)

    # NTK at init
    K_init = compute_empirical_ntk_efficient(model, X, device=device)
    history_init = {
        'ntk_matrices': [K_init.cpu().numpy()],
        'ntk_epochs': [0],
        'ntk_eigenvalues': [
            torch.linalg.eigvalsh(K_init.cpu()).numpy()
        ]
    }

    # Training
    history_train = train_with_cross_entropy(
        model, X, Y,
        epochs=20000,
        lr=0.05,
        compute_ntk_every=5000,
        device=device
    )

    # Merge histories
    history = history_train
    history['ntk_matrices'] = history_init['ntk_matrices'] + history['ntk_matrices']
    history['ntk_epochs'] = history_init['ntk_epochs'] + history['ntk_epochs']
    history['ntk_eigenvalues'] = history_init['ntk_eigenvalues'] + history['ntk_eigenvalues']

    return history, Y



def plot_mnist_outputs(history, Y, n_tracked=10):
    outputs = np.array(history['outputs'])  # (epochs, N)
    epochs = np.arange(1, outputs.shape[0] + 1)
    log_epochs = np.log10(epochs)

    tracked = np.random.choice(outputs.shape[1], n_tracked, replace=False)

    plt.figure(figsize=(7, 5))
    for i in tracked:
        plt.plot(log_epochs, outputs[:, i], label=f"y={Y[i].item()}")

    plt.xlabel(r'$\log_{10}(\mathrm{epoch})$')
    plt.ylabel('Network output $f(x)$')
    plt.title('MNIST Output Divergence (Subset)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    history, Y = run_mnist_experiment(device)

    plot_mnist_outputs(history, Y)
    analyze_eigenvalues(history)
