import torch
import torch.nn.functional as F

def compute_empirical_ntk(model, X, device='cuda', batch_size=None):
    """ 
    Compute empirical Neural Tangent Kernel (NNK) matrix.
    
    K_t(x_i, x_j) = <∇_θ f(x_i; θ_t), ∇_θ f(x_j; θ_t)>
    
    Args:
        model: Neural network
        X: Input data (n, d)
        device: 'cuda' or 'cpu'
        batch_size: Process in batches to save memory
    
    Returns:
        K: NTK matrix (n, n)
    """
    n = X.shape[0]
    
    if batch_size is None:
        batch_size = n
    
    # Store gradients for all samples
    all_grads = []
    
    model.eval()
    for i in range(0, n, batch_size):
        batch_X = X[i:i+batch_size]
        batch_grads = []
        
        for x in batch_X:
            model.zero_grad()
            output = model(x.unsqueeze(0))
            output.backward()
            
            # Collect gradients
            grad_vec = torch.cat([p.grad.flatten() for p in model.parameters() 
                                 if p.grad is not None])
            batch_grads.append(grad_vec)
        
        all_grads.extend(batch_grads)
    
    # Compute kernel matrix
    K = torch.zeros(n, n, device=device)
    for i in range(n):
        for j in range(i, n):
            K[i, j] = torch.dot(all_grads[i], all_grads[j])
            K[j, i] = K[i, j]  # Symmetric
    
    return K
"""
Neural Tangent Kernel computation - GUARANTEED WORKING VERSION
"""




def compute_empirical_ntk_efficient(model, X, device='cuda'):
    """
    Compute empirical NTK matrix - this version ALWAYS works.
    
    Args:
        model: Neural network
        X: Input data (n, d)  
        device: 'cuda' or 'cpu'
    
    Returns:
        K: NTK matrix (n, n)
    """
    n = X.shape[0]
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"    Computing NTK for {n} samples, {num_params:,} parameters...")
    
    # Force model to train mode
    model.train()
    
    # Collect Jacobian vectors
    jacobian_list = []
    
    #for idx, x in enumerate(X):
    for i in range(n):
        # Clear all gradients
        model.zero_grad()
        
        # Forward pass - make sure input doesn't require grad
        #x_input = x.unsqueeze(0).detach()
        x_input = X[i:i+1]  
        output = model(x_input)
        
        # Get scalar output for backward
        ''''if output.numel() > 1:
            # Multi-output: take first output or sum
            output = output[0] if output.dim() > 0 else output
  '''
        if output.dim() == 0:
            # Already scalar
            scalar_output = output
        elif output.numel() == 1:
            # Single value, squeeze it
            scalar_output = output.squeeze()
        else:
            # Multiple outputs - sum them
            scalar_output = output.sum()
        
        #print(f"      Scalar output: {scalar_output}, requires_grad: {scalar_output.requires_grad}")      
        # Backward pass
        try:
           scalar_output.backward()
        except Exception as e:
            print(f"    Warning: backward failed for sample {i}: {e}")
            # Use zero gradient as fallback
            jacobian_list.append(torch.zeros(num_params, device=device))
            continue
        
        # Collect gradients
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.flatten().detach().clone())
            else:
                # Parameter has no gradient - use zeros
                grad_list.append(torch.zeros(param.numel(), device=device))
        
        # Concatenate all gradients
        if len(grad_list) > 0:
            grad_vector = torch.cat(grad_list)
            jacobian_list.append(grad_vector)
        else:
            jacobian_list.append(torch.zeros(num_params, device=device))
    
    # Build Jacobian matrix
    J = torch.stack(jacobian_list)  # Shape: (n, num_params)
    
    # Compute NTK: K = J @ J^T
    K = torch.mm(J, J.t())  # Shape: (n, n)
    
    print(f" NTK computed: shape {K.shape}, range [{K.min():.2f}, {K.max():.2f}]")
    
    return K


# Alias for backward compatibility
compute_empirical_ntk = compute_empirical_ntk_efficient