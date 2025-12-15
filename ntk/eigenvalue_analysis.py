"""
Eigenvalue analysis for Neural Tangent Kernel matrices.
Provides spectral analysis and stability checks for NTK studies.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import warnings


def compute_eigenvalues(K, use_cpu=True, use_double=True):
    """
    Compute eigenvalues of NTK matrix.
    
    Args:
        K: NTK matrix (n, n) - can be torch tensor or numpy array
        use_cpu: Move to CPU for computation (more stable)
        use_double: Use double precision (more accurate)
    
    Returns:
        eigenvalues: Sorted eigenvalues (descending)
    """
    # Convert to appropriate format
    if isinstance(K, torch.Tensor):
        if use_cpu:
            K = K.cpu()
        if use_double:
            K = K.double()
        
        # Compute eigenvalues using PyTorch
        eigenvalues = torch.linalg.eigvalsh(K)
        eigenvalues = eigenvalues.cpu().numpy()
    else:
        # Already numpy array
        if use_double:
            K = K.astype(np.float64)
        
        # Compute eigenvalues using NumPy/SciPy
        eigenvalues = np.linalg.eigvalsh(K)
    
    # Sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    return eigenvalues


def compute_eigenvalues_and_vectors(K, use_cpu=True, use_double=True):
    """
    Compute eigenvalues and eigenvectors of NTK matrix.
    
    Args:
        K: NTK matrix (n, n)
        use_cpu: Move to CPU for computation
        use_double: Use double precision
    
    Returns:
        eigenvalues: Eigenvalues (sorted descending)
        eigenvectors: Corresponding eigenvectors
    """
    # Convert to appropriate format
    if isinstance(K, torch.Tensor):
        if use_cpu:
            K = K.cpu()
        if use_double:
            K = K.double()
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        eigenvalues = eigenvalues.cpu().numpy()
        eigenvectors = eigenvectors.cpu().numpy()
    else:
        if use_double:
            K = K.astype(np.float64)
        
        eigenvalues, eigenvectors = np.linalg.eigh(K)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def compute_condition_number(K):
    """
    Compute condition number of NTK matrix.
    
    Condition number = λ_max / λ_min
    High condition number indicates numerical instability.
    
    Args:
        K: NTK matrix (n, n)
    
    Returns:
        condition_number: Condition number
    """
    eigenvalues = compute_eigenvalues(K)
    
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[-1]
    
    if lambda_min < 1e-10:
        warnings.warn(f"Very small minimum eigenvalue: {lambda_min:.2e}")
        condition_number = np.inf
    else:
        condition_number = lambda_max / lambda_min
    
    return condition_number


def check_positive_definiteness(K, threshold=1e-8):
    """
    Check if NTK matrix is positive definite.
    
    Args:
        K: NTK matrix (n, n)
        threshold: Minimum eigenvalue threshold
    
    Returns:
        is_positive_definite: Boolean
        min_eigenvalue: Minimum eigenvalue
    """
    eigenvalues = compute_eigenvalues(K)
    min_eigenvalue = eigenvalues[-1]
    
    is_positive_definite = min_eigenvalue > threshold
    
    return is_positive_definite, min_eigenvalue


def compute_effective_rank(K, threshold=0.01):
    """
    Compute effective rank of NTK matrix.
    
    Effective rank = number of eigenvalues above threshold * λ_max
    
    Args:
        K: NTK matrix (n, n)
        threshold: Relative threshold (default: 1% of max eigenvalue)
    
    Returns:
        effective_rank: Effective rank
        eigenvalues: All eigenvalues
    """
    eigenvalues = compute_eigenvalues(K)
    lambda_max = eigenvalues[0]
    
    effective_rank = np.sum(eigenvalues > threshold * lambda_max)
    
    return effective_rank, eigenvalues


def analyze_spectral_decay(eigenvalues):
    """
    Analyze how fast eigenvalues decay.
    
    Args:
        eigenvalues: Sorted eigenvalues (descending)
    
    Returns:
        decay_rate: Approximate decay rate
        analysis: Dictionary with decay statistics
    """
    n = len(eigenvalues)
    
    # Normalize eigenvalues
    eigenvalues_norm = eigenvalues / eigenvalues[0]
    
    # Fit exponential decay: λ_i ≈ λ_0 * exp(-α * i)
    # log(λ_i) ≈ log(λ_0) - α * i
    indices = np.arange(1, n)
    log_eigs = np.log(eigenvalues_norm[1:] + 1e-10)
    
    # Linear fit
    decay_rate = -np.polyfit(indices, log_eigs, 1)[0]
    
    analysis = {
        'decay_rate': decay_rate,
        'eigenvalue_ratio': eigenvalues[-1] / eigenvalues[0],
        'top_5_sum': np.sum(eigenvalues[:5]) / np.sum(eigenvalues),
        'top_10_sum': np.sum(eigenvalues[:10]) / np.sum(eigenvalues),
        'effective_dimension': np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    }
    
    return decay_rate, analysis


def compute_ntk_divergence(K_current, K_initial, metric='frobenius'):
    """
    Compute divergence between current and initial NTK.
    
    This measures how much NTK has changed during training.
    
    Args:
        K_current: Current NTK matrix (n, n)
        K_initial: Initial NTK matrix (n, n)
        metric: 'frobenius', 'spectral', or 'relative'
    
    Returns:
        divergence: Divergence measure
    """
    if isinstance(K_current, torch.Tensor):
        K_current = K_current.cpu().numpy()
    if isinstance(K_initial, torch.Tensor):
        K_initial = K_initial.cpu().numpy()
    
    diff = K_current - K_initial
    
    if metric == 'frobenius':
        # ||K_t - K_0||_F
        divergence = np.linalg.norm(diff, 'fro')
    
    elif metric == 'spectral':
        # ||K_t - K_0||_2 (largest singular value)
        divergence = np.linalg.norm(diff, 2)
    
    elif metric == 'relative':
        # ||K_t - K_0||_F / ||K_0||_F
        divergence = np.linalg.norm(diff, 'fro') / np.linalg.norm(K_initial, 'fro')
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return divergence


def track_eigenvalue_evolution(ntk_matrices, ntk_epochs):
    """
    Track how eigenvalues evolve over training.
    
    Args:
        ntk_matrices: List of NTK matrices at different epochs
        ntk_epochs: Corresponding epochs
    
    Returns:
        eigenvalue_history: Array (num_epochs, n) of eigenvalues
        statistics: Dictionary with evolution statistics
    """
    num_matrices = len(ntk_matrices)
    n = ntk_matrices[0].shape[0]
    
    eigenvalue_history = np.zeros((num_matrices, n))
    
    for i, K in enumerate(ntk_matrices):
        eigenvalue_history[i] = compute_eigenvalues(K)
    
    # Compute statistics
    statistics = {
        'epochs': ntk_epochs,
        'min_eigenvalue_over_time': eigenvalue_history[:, -1],
        'max_eigenvalue_over_time': eigenvalue_history[:, 0],
        'condition_number_over_time': eigenvalue_history[:, 0] / (eigenvalue_history[:, -1] + 1e-10),
        'trace_over_time': np.sum(eigenvalue_history, axis=1),
        'effective_rank_over_time': []
    }
    
    for i in range(num_matrices):
        eff_rank, _ = compute_effective_rank(ntk_matrices[i])
        statistics['effective_rank_over_time'].append(eff_rank)
    
    return eigenvalue_history, statistics


def plot_eigenvalue_spectrum(eigenvalues, title="NTK Eigenvalue Spectrum", 
                             save_path=None):
    """
    Plot eigenvalue spectrum.
    
    Args:
        eigenvalues: Array of eigenvalues
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Linear scale
    ax = axes[0]
    ax.plot(eigenvalues, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(f'{title} (Linear Scale)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    ax = axes[1]
    ax.semilogy(eigenvalues, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax.set_title(f'{title} (Log Scale)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Eigenvalue spectrum saved to {save_path}")
    
    plt.show()


def plot_eigenvalue_evolution(eigenvalue_history, epochs, n_top=5, 
                              save_path=None):
    """
    Plot how top eigenvalues evolve over training.
    
    Args:
        eigenvalue_history: Array (num_epochs, n) of eigenvalues
        epochs: Epoch numbers
        n_top: Number of top eigenvalues to plot
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(n_top):
        ax.plot(epochs, eigenvalue_history[:, i], 
               linewidth=2, marker='o', label=f'λ_{i+1}')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(f'Evolution of Top {n_top} Eigenvalues', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Eigenvalue evolution saved to {save_path}")
    
    plt.show()


def generate_eigenvalue_report(K, label="NTK"):
    """
    Generate comprehensive eigenvalue analysis report.
    
    Args:
        K: NTK matrix (n, n)
        label: Label for the matrix
    
    Returns:
        report: Dictionary with analysis results
    """
    print("=" * 70)
    print(f"EIGENVALUE ANALYSIS REPORT: {label}")
    print("=" * 70)
    
    # Compute eigenvalues
    eigenvalues = compute_eigenvalues(K)
    n = len(eigenvalues)
    
    print(f"\nMatrix Dimension: {n} × {n}")
    
    # Basic statistics
    print(f"\nEigenvalue Statistics:")
    print(f"  Maximum: {eigenvalues[0]:.6f}")
    print(f"  Minimum: {eigenvalues[-1]:.6e}")
    print(f"  Mean: {np.mean(eigenvalues):.6f}")
    print(f"  Median: {np.median(eigenvalues):.6f}")
    print(f"  Std Dev: {np.std(eigenvalues):.6f}")
    
    # Condition number
    cond_num = compute_condition_number(K)
    print(f"\nCondition Number: {cond_num:.2e}")
    if cond_num > 1e10:
        print("  ⚠️  Very high condition number - matrix is ill-conditioned!")
    elif cond_num > 1e6:
        print("  ⚠️  High condition number - numerical issues possible")
    else:
        print("  ✓ Condition number is acceptable")
    
    # Positive definiteness
    is_pd, min_eig = check_positive_definiteness(K)
    print(f"\nPositive Definiteness:")
    print(f"  Minimum eigenvalue: {min_eig:.6e}")
    if is_pd:
        print("  ✓ Matrix is positive definite")
    else:
        print("  ❌ Matrix is NOT positive definite")
    
    # Effective rank
    eff_rank, _ = compute_effective_rank(K)
    print(f"\nEffective Rank: {eff_rank} / {n}")
    print(f"  Rank Ratio: {eff_rank/n:.2%}")
    
    # Spectral decay
    decay_rate, analysis = analyze_spectral_decay(eigenvalues)
    print(f"\nSpectral Decay:")
    print(f"  Decay Rate: {decay_rate:.4f}")
    print(f"  Top 5 eigenvalues: {analysis['top_5_sum']:.2%} of total")
    print(f"  Top 10 eigenvalues: {analysis['top_10_sum']:.2%} of total")
    print(f"  Effective Dimension: {analysis['effective_dimension']:.2f}")
    
    print("=" * 70)
    
    # Create report dictionary
    report = {
        'eigenvalues': eigenvalues,
        'n': n,
        'lambda_max': eigenvalues[0],
        'lambda_min': eigenvalues[-1],
        'condition_number': cond_num,
        'is_positive_definite': is_pd,
        'effective_rank': eff_rank,
        'decay_rate': decay_rate,
        'spectral_analysis': analysis
    }
    
    return report


# Test function
def test_eigenvalue_analysis():
    """Test eigenvalue analysis functions."""
    print("Testing eigenvalue analysis module...")
    
    # Create a simple positive definite matrix
    n = 10
    A = np.random.randn(n, n)
    K = A @ A.T + np.eye(n)  # Guaranteed positive definite
    
    print("\n1. Computing eigenvalues...")
    eigenvalues = compute_eigenvalues(K)
    print(f"   ✓ Shape: {eigenvalues.shape}")
    print(f"   ✓ Range: [{eigenvalues[-1]:.4f}, {eigenvalues[0]:.4f}]")
    
    print("\n2. Checking positive definiteness...")
    is_pd, min_eig = check_positive_definiteness(K)
    print(f"   ✓ Positive definite: {is_pd}")
    print(f"   ✓ Min eigenvalue: {min_eig:.6f}")
    
    print("\n3. Computing condition number...")
    cond_num = compute_condition_number(K)
    print(f"   ✓ Condition number: {cond_num:.2e}")
    
    print("\n4. Computing effective rank...")
    eff_rank, _ = compute_effective_rank(K)
    print(f"   ✓ Effective rank: {eff_rank} / {n}")
    
    print("\n5. Generating full report...")
    report = generate_eigenvalue_report(K, label="Test Matrix")
    
    print("\n✓ All tests passed!")
    
    return report


if __name__ == '__main__':
    test_eigenvalue_analysis()