"""
Training module for neural networks with gradient flow optimization.
Implements cross-entropy loss with NTK tracking for reproducibility study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')

from ntk.ntk_computation import compute_empirical_ntk_efficient


def train_with_cross_entropy(model, X, Y, epochs=10000, lr=0.1, 
                            log_interval=1000, compute_ntk_every=None,
                            device='cuda', save_checkpoints=False):
    """
    Train neural network using gradient flow with cross-entropy loss.
    
    This implements the training procedure from the ICLR 2025 paper:
    - Cross-entropy loss for binary classification
    - Gradient flow (continuous-time gradient descent approximation)
    - Track network outputs and NTK evolution
    
    Args:
        model: Neural network (FullyConnectedNetwork or ResNet)
        X: Input data (n, d)
        Y: Binary labels (n,) with values in {0, 1}
        epochs: Number of training epochs
        lr: Learning rate
        log_interval: Log progress every N epochs
        compute_ntk_every: Compute NTK every N epochs (None = don't compute)
        device: 'cuda' or 'cpu'
        save_checkpoints: Save model checkpoints during training
    
    Returns:
        history: Dictionary containing training history
    """
    model.to(device)
    X, Y = X.to(device), Y.to(device)
    
    # Optimizer (gradient flow approximation)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Loss function
    
    criterion =nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'losses': [],
        'accuracies': [],
        'outputs': [],  # Network outputs at each epoch
        'ntk_matrices': [],  # NTK matrices (if computed)
        'ntk_eigenvalues': [],  # NTK eigenvalues
        'ntk_epochs': [],  # Epochs where NTK was computed
        'u_values': []  # Residual values: u_i = |o_i - y_i|
    }
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    pbar = tqdm(range(epochs), desc="Training")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X).squeeze(-1)  # (n,)
        
        # Compute loss
        loss = criterion(outputs, Y.float())
        
        # Backward pass
        loss.backward()
        
        # Gradient descent step
        optimizer.step()
        
        
           # Compute metrics
        with torch.no_grad():
    # outputs: [batch, 2] logits
            probs = torch.sigmoid(outputs)  # P(y=1 | x), shape [batch]
            preds = (probs >= 0.5).long()                  # 0/1 predictions, [batch]
            accuracy = (preds == Y).float().mean().item()

    # Residual values: u_i = |sigmoid(f(x_i)) - y_i|
            u_values = torch.abs(probs - Y.float())

            
            # Store history
            history['losses'].append(loss.item())
            history['accuracies'].append(accuracy)
            history['outputs'].append(outputs.cpu().numpy().copy())
            history['u_values'].append(u_values.cpu().numpy().copy())
        
        # Compute NTK if requested
        if compute_ntk_every is not None and epoch % compute_ntk_every == 0:
          
                try:
                    # Compute empirical NTK
                    K = compute_empirical_ntk_efficient(model, X, device=device)
                    
                    # Compute eigenvalues (on CPU for stability)
                    K_cpu = K.cpu().double()
                    eigenvalues = torch.linalg.eigvalsh(K_cpu)
                    
                    history['ntk_matrices'].append(K.cpu().numpy().copy())
                    history['ntk_eigenvalues'].append(eigenvalues.numpy().copy())
                    history['ntk_epochs'].append(epoch)
                    
                    min_eig = eigenvalues.min().item()
                    max_eig = eigenvalues.max().item()
                    
                except Exception as e:
                    print(f"\nWarning: NTK computation failed at epoch {epoch}: {e}")
        
        # Logging
        if epoch % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'max_output': f'{outputs.abs().max().item():.2f}'
            })
        
        # Save checkpoint
        if save_checkpoints and epoch % 5000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
    
    print(f"\nTraining completed!")
    print(f"Final loss: {history['losses'][-1]:.6f}")
    print(f"Final accuracy: {history['accuracies'][-1]:.4f}")
    print(f"Max output magnitude: {np.abs(history['outputs'][-1]).max():.2f}")
    
    return history


def train_with_mse_loss(model, X, Y, epochs=10000, lr=0.01, 
                       log_interval=1000, compute_ntk_every=None,
                       device='cuda'):
    """
    Train neural network with MSE loss for comparison.
    
    This is used in ablation studies to compare behavior:
    - MSE loss should show NTK convergence (lazy training regime)
    - Cross-entropy shows NTK divergence
    
    Args:
        model: Neural network
        X: Input data (n, d)
        Y: Binary labels (n,) with values in {0, 1}
        epochs: Number of training epochs
        lr: Learning rate
        log_interval: Log progress every N epochs
        compute_ntk_every: Compute NTK every N epochs
        device: 'cuda' or 'cpu'
    
    Returns:
        history: Dictionary containing training history
    """
    model.to(device)
    X, Y = X.to(device), Y.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'losses': [],
        'accuracies': [],
        'outputs': [],
        'ntk_matrices': [],
        'ntk_eigenvalues': [],
        'ntk_epochs': []
    }
    
    print(f"Training with MSE loss for {epochs} epochs...")
    pbar = tqdm(range(epochs), desc="Training (MSE)")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        
        # MSE loss (outputs vs. labels directly)
        loss = criterion(outputs, Y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            preds = (outputs > 0.5).long()
            accuracy = (preds == Y).float().mean().item()
            
            history['losses'].append(loss.item())
            history['accuracies'].append(accuracy)
            history['outputs'].append(outputs.cpu().numpy().copy())
        
        # Compute NTK
        if compute_ntk_every is not None and epoch % compute_ntk_every == 0:
            with torch.no_grad():
                try:
                    K = compute_empirical_ntk_efficient(model, X, device=device)
                    K_cpu = K.cpu().double()
                    eigenvalues = torch.linalg.eigvalsh(K_cpu)
                    
                    history['ntk_matrices'].append(K.cpu().numpy().copy())
                    history['ntk_eigenvalues'].append(eigenvalues.numpy().copy())
                    history['ntk_epochs'].append(epoch)
                except Exception as e:
                    print(f"\nWarning: NTK computation failed: {e}")
        
        if epoch % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    return history


def train_with_hinge_loss(model, X, Y, epochs=10000, lr=0.01,
                         log_interval=1000, compute_ntk_every=None,
                         device='cuda'):
    """
    Train with hinge loss for ablation study.
    
    Hinge loss: max(0, 1 - y*f(x)) where y ∈ {-1, 1}
    """
    model.to(device)
    X, Y = X.to(device), Y.to(device)
    
    # Convert labels to {-1, 1}
    Y_hinge = 2 * Y.float() - 1
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    history = {
        'losses': [],
        'accuracies': [],
        'outputs': [],
        'ntk_matrices': [],
        'ntk_eigenvalues': [],
        'ntk_epochs': []
    }
    
    print(f"Training with Hinge loss for {epochs} epochs...")
    pbar = tqdm(range(epochs), desc="Training (Hinge)")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X)
        
        # Hinge loss
        loss = torch.mean(torch.clamp(1 - Y_hinge * outputs, min=0))
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = (outputs > 0).long()
            accuracy = (preds == Y).float().mean().item()
            
            history['losses'].append(loss.item())
            history['accuracies'].append(accuracy)
            history['outputs'].append(outputs.cpu().numpy().copy())
        
        if compute_ntk_every is not None and epoch % compute_ntk_every == 0:
            with torch.no_grad():
                try:
                    K = compute_empirical_ntk_efficient(model, X, device=device)
                    K_cpu = K.cpu().double()
                    eigenvalues = torch.linalg.eigvalsh(K_cpu)
                    
                    history['ntk_matrices'].append(K.cpu().numpy().copy())
                    history['ntk_eigenvalues'].append(eigenvalues.numpy().copy())
                    history['ntk_epochs'].append(epoch)
                except:
                    pass
        
        if epoch % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    return history


def evaluate_model(model, X, Y, device='cuda'):
    """
    Evaluate model on given data.
    
    Returns:
        metrics: Dictionary with loss, accuracy, predictions
    """
    model.eval()
    model.to(device)
    X, Y = X.to(device), Y.to(device)
    
    with torch.no_grad():
        outputs = model(X)
        
        # Cross-entropy loss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, Y.float()).item()
        
        # Predictions
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        accuracy = (preds == Y).float().mean().item()
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'outputs': outputs.cpu().numpy(),
            'predictions': preds.cpu().numpy()
        }
    
    return metrics