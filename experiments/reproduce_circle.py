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

from data.generate_circle import generate_circle_dataset
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
def run_circle_experiment(device='cuda', width=2000, epochs=10000, lr=0.1):
    """
    Reproduce synthetic circle experiment from paper.
    """
    print("=" * 60)
    print("CIRCLE EXPERIMENT: Reproducing ICLR 2025 Paper")
    print("=" * 60)
    
    # Generate data
    X, Y = generate_circle_dataset(n_points=6, device=device)
    print(f"Generated {len(X)} points on unit circle")
    print(f"Labels: {Y.cpu().numpy()}")
    
 
    # Create model
    model = FullyConnectedNetwork(
        input_dim=2,
        num_classes=1,      # binary labels for the circle dataset
        depth=3,            # 3-layer FCN (input -> hidden -> hidden -> output)
        width=width,        # hidden layer width
        activation='relu'
    ).to(device)

    print(f"Model: 3-layer FCN with width {width}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nComputing NTK at initialization (t = 0)...")

    K_init = compute_empirical_ntk_efficient(model, X, device=device)

    history_init = {
        'ntk_matrices': [K_init.cpu().numpy()],
        'ntk_epochs': [0],
        'ntk_eigenvalues': [
        torch.linalg.eigvalsh(K_init.cpu()).numpy()
          ]
    }


    # Train model
    print(f"\nTraining for {epochs} epochs with lr={lr}...")
    history_train = train_with_cross_entropy(
        model, X, Y, 
        epochs=epochs, 
        lr=lr,
        log_interval=1000,
        compute_ntk_every=2000,
        device=device
    )
    history = history_train
    history['ntk_matrices'] = (
        history_init['ntk_matrices'] + history['ntk_matrices']
    )
    history['ntk_epochs'] = (
    history_init['ntk_epochs'] + history['ntk_epochs']
    )
    history['ntk_eigenvalues'] = (
      history_init['ntk_eigenvalues'] + history['ntk_eigenvalues']
    )
    
    print(history['ntk_epochs'])
    # Plot results
    plot_circle_results(history, X, Y)
    analyze_eigenvalues(history)
    
    print("\n" + "=" * 60)
    print("\nExperiment completed!")
    print("=" * 60)
    return history

def plot_circle_results(history, X, Y):
    """Plot network outputs and NTK evolution."""
   
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Plot 1: Network outputs over time
    
    outputs = np.array(history['outputs'])
    epochs = np.arange(1, outputs.shape[0] + 1)
    log_epochs = np.log10(epochs)
    
    ax = axes[0, 0]
    for i in range(outputs.shape[1]):
       label = f"x_{i+1} (y={Y[i].item()})"
       ax.plot(log_epochs, outputs[:, i], label=label, linewidth=2)
    
    ax.set_xlabel(r'$\log_{10}(\mathrm{epoch})$')
    ax.set_ylabel('Network Output $f(x_i)$')
    ax.set_title('Divergence of Network Outputs (Log Time)') 
    ax.legend()
    ax.grid(True, alpha=0.3)

    
    # Plot 2: Loss over time
    ax = axes[0, 1]
    ax.plot(history['losses'], linewidth=2, color='red')
    ax.set_xlabel('Epoch',fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Training Loss',fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: NTK eigenvalues
    ax=axes[1,0]
    if 'ntk_eigenvalues' in history and len(history['ntk_eigenvalues']) > 0:
        ntk_eigs = np.array(history['ntk_eigenvalues'])
        for i in range(min(6, ntk_eigs.shape[1])):
            ax.plot(history['ntk_epochs'], ntk_eigs[:, i], 
                   label=f'λ_{i+1}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('NTK Eigenvalue Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Data points
    ax = axes[1, 1]
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    colors = ['blue' if y == 0 else 'red' for y in Y_np]
    ax.scatter(X_np[:, 0], X_np[:, 1], c=colors, s=200, 
              edgecolors='black', linewidths=2)
    for i, (x, y) in enumerate(zip(X_np, Y_np)):
        ax.annotate(f'({i+1}, {y})', (x[0], x[1]), 
                   xytext=(5, 5), textcoords='offset points')
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle='--', 
                       color='gray', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Circle Dataset (6 points)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/circle_experiment.png', dpi=300, bbox_inches='tight')
    print("Saved plot to results/plots/circle_experiment.png")
    plt.show()
    

def analyze_eigenvalues(history):
    """
    Analyze NTK eigenvalues from training history.
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
    
    # Analyze initial NTK
    print("\n" + "─" * 60)
    print("INITIAL NTK (t=0)")
    print("─" * 60)
    K_initial = ntk_matrices[0]
    report_initial = generate_eigenvalue_report(K_initial, label="Initial NTK")
    
    # Analyze final NTK
    print("\n" + "─" * 60)
    print(f"FINAL NTK (t={ntk_epochs[-1]})")
    print("─" * 60)
    K_final = ntk_matrices[-1]
    report_final = generate_eigenvalue_report(K_final, label="Final NTK")
    
    # Compute divergence
    print("\n" + "─" * 60)
    print("NTK DIVERGENCE METRICS")
    print("─" * 60)
    
    div_frobenius = compute_ntk_divergence(K_final, K_initial, metric='frobenius')
    div_spectral = compute_ntk_divergence(K_final, K_initial, metric='spectral')
    div_relative = compute_ntk_divergence(K_final, K_initial, metric='relative')
    
    print(f"Frobenius Norm: ||K_final - K_init||_F = {div_frobenius:.4f}")
    print(f"Spectral Norm:  ||K_final - K_init||_2 = {div_spectral:.4f}")
    print(f"Relative Div:   {div_relative:.4f} ({div_relative*100:.2f}%)")
    
    # Verdict
    print("\n" + "─" * 60)
    print("VERDICT")
    print("─" * 60)
    if div_relative > 0.1:
        print("✅ SIGNIFICANT NTK DIVERGENCE DETECTED (>10%)")
        print("   This confirms the paper's main result!")
        print("   The 'lazy training regime' has broken down.")
    elif div_relative > 0.05:
        print("✓ Moderate NTK divergence detected (5-10%)")
        print("  Results consistent with paper's findings")
    elif div_relative > 0.01:
        print("⚠️  Weak divergence (1-5%)")
        print("   Consider longer training or check hyperparameters")
    else:
        print("❌ No significant divergence (<1%)")
        print("   Training may be too short or learning rate too small")
    
    # Track evolution
    print("\n" + "─" * 60)
    print("EIGENVALUE EVOLUTION STATISTICS")
    print("─" * 60)
    
    eigenvalue_history, statistics = track_eigenvalue_evolution(
        ntk_matrices, ntk_epochs
    )
    
    print(f"\nMin eigenvalue over time:")
    for epoch, min_eig in zip(ntk_epochs, statistics['min_eigenvalue_over_time']):
        print(f"  Epoch {epoch:5d}: λ_min = {min_eig:.6e}")
    
    print(f"\nCondition number over time:")
    for epoch, cond in zip(ntk_epochs, statistics['condition_number_over_time']):
        print(f"  Epoch {epoch:5d}: κ = {cond:.2e}")
    
    # Generate plots
    print("\n" + "─" * 60)
    print("GENERATING EIGENVALUE PLOTS")
    print("─" * 60)
    
    plot_eigenvalue_spectrum(
        report_initial['eigenvalues'],
        title="Initial NTK Eigenvalue Spectrum",
        save_path='results/plots/eigenvalues_initial.png'
    )
    
    plot_eigenvalue_spectrum(
        report_final['eigenvalues'],
        title="Final NTK Eigenvalue Spectrum",
        save_path='results/plots/eigenvalues_final.png'
    )
    
    plot_eigenvalue_evolution(
        eigenvalue_history,
        ntk_epochs,
        n_top=6,
        save_path='results/plots/eigenvalue_evolution.png'
    )
    
    print("\n" + "=" * 60)
    print("EIGENVALUE ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  ✓ results/plots/circle_experiment.png")
    print("  ✓ results/plots/eigenvalues_initial.png")
    print("  ✓ results/plots/eigenvalues_final.png")
    print("  ✓ results/plots/eigenvalue_evolution.png")
    
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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    history = run_circle_experiment(
        device=device,
        width=2000,
        epochs=10000,
        lr=0.1
    )