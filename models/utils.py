"""
Utility functions for neural network models.
Includes initialization schemes, activation functions, and helper methods.
"""

import torch
import torch.nn as nn
import numpy as np


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name ('relu', 'gelu', 'tanh', 'sigmoid')
    
    Returns:
        PyTorch activation module
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def init_weights_he(module: nn.Module) -> None:
    """
    He initialization suitable for ReLU-like activations.
    Also known as Kaiming initialization.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_xavier(module: nn.Module) -> None:
    """
    Xavier/Glorot initialization suitable for tanh/sigmoid activations.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_ntk(module: nn.Module, scale: float = 1.0) -> None:
    """
    NTK-style initialization: N(0, scale^2) for both weights and biases.
    This is the initialization used in the ICLR 2025 paper.
    
    Args:
        module: PyTorch module to initialize
        scale: Standard deviation for initialization (default: 1.0)
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=scale)
        if module.bias is not None:
            nn.init.normal_(module.bias, mean=0.0, std=scale)
    elif isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=scale)
        if module.bias is not None:
            nn.init.normal_(module.bias, mean=0.0, std=scale)


def init_weights_custom_scale(module: nn.Module, weight_scale: float = 1.0, 
                               bias_scale: float = 0.0) -> None:
    """
    Custom initialization with different scales for weights and biases.
    Useful for initialization sensitivity ablation study.
    
    Args:
        module: PyTorch module to initialize
        weight_scale: Scale for weight initialization
        bias_scale: Scale for bias initialization (0 = zero init)
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=weight_scale)
        if module.bias is not None:
            if bias_scale == 0.0:
                nn.init.zeros_(module.bias)
            else:
                nn.init.normal_(module.bias, mean=0.0, std=bias_scale)


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_device(model: nn.Module) -> torch.device:
    """
    Get the device where model parameters are stored.
    
    Args:
        model: PyTorch model
    
    Returns:
        Device (cuda or cpu)
    """
    return next(model.parameters()).device


def freeze_layers(model: nn.Module, layer_names: list = None) -> None:
    """
    Freeze specific layers or all layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze (None = freeze all)
    """
    if layer_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: list = None) -> None:
    """
    Unfreeze specific layers or all layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze (None = unfreeze all)
    """
    if layer_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Unfreeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm across all parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_ntk_scaling_factor(width: int, layer_idx: int, num_layers: int) -> float:
    """
    Compute NTK scaling factor for a specific layer.
    According to NTK theory: scale by sqrt(2/width) at each layer.
    
    Args:
        width: Layer width
        layer_idx: Current layer index
        num_layers: Total number of layers
    
    Returns:
        Scaling factor
    """
    return np.sqrt(2.0 / width)


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                          epoch: int, loss: float, filepath: str) -> None:
    """
    Save model checkpoint with optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                          filepath: str, device: str = 'cuda') -> tuple:
    """
    Load model checkpoint with optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.6f})")
    return epoch, loss


def print_model_summary(model: nn.Module, input_size: tuple = None) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (optional)
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    # Count parameters
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    
    print("\nModel Architecture:")
    print("-" * 70)
    print(model)
    print("=" * 70)
    
    # If input size is provided, show layer-wise output shapes
    if input_size is not None:
        try:
            device = get_model_device(model)
            dummy_input = torch.randn(1, *input_size).to(device)
            print("\nLayer-wise Output Shapes:")
            print("-" * 70)
            
            # Hook to capture layer outputs
            layer_outputs = []
            
            def hook_fn(module, input, output):
                layer_outputs.append((module.__class__.__name__, output.shape))
            
            # Register hooks
            hooks = []
            for layer in model.modules():
                if len(list(layer.children())) == 0:  # Leaf modules only
                    hooks.append(layer.register_forward_hook(hook_fn))
            
            # Forward pass
            with torch.no_grad():
                model(dummy_input)
            
            # Print outputs
            for i, (name, shape) in enumerate(layer_outputs):
                print(f"Layer {i+1} ({name}): {tuple(shape)}")
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            print("=" * 70)
        except Exception as e:
            print(f"Could not compute layer shapes: {e}")


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def adjust_learning_rate(optimizer: torch.optim.Optimizer, 
                         new_lr: float) -> None:
    """
    Adjust learning rate of optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        new_lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(f"Learning rate adjusted to {new_lr}")


# For ablation study: Initialization scale variants
INIT_SCALES = {
    'very_small': 0.1,
    'small': 0.5,
    'standard': 1.0,
    'large': 2.0,
    'very_large': 5.0
}


def initialize_with_scale(model: nn.Module, scale_name: str = 'standard') -> None:
    """
    Initialize model with predefined scale for ablation study.
    
    Args:
        model: PyTorch model
        scale_name: One of 'very_small', 'small', 'standard', 'large', 'very_large'
    """
    if scale_name not in INIT_SCALES:
        raise ValueError(f"Unknown scale: {scale_name}. Choose from {list(INIT_SCALES.keys())}")
    
    scale = INIT_SCALES[scale_name]
    model.apply(lambda m: init_weights_ntk(m, scale=scale))
    print(f"Model initialized with scale: {scale_name} (σ={scale})")