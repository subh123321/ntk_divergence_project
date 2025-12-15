from typing import List
import torch
from torch import nn

from .utils import get_activation, init_weights_he


class FullyConnectedNetwork(nn.Module):
    """
    Generic fully connected network:
    - input_dim: flattened input size (e.g., 2 for circle, 28*28 for MNIST)
    - depth: total number of layers including output
      * depth >= 2: (input -> hidden -> ... -> hidden -> output)
    - width: hidden layer width
    - num_classes: output dimension
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        depth: int = 3,
        width: int = 2000,
        activation: str = "relu",
    ) -> None:

        super().__init__()

        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.num_classes = num_classes  # Store for forward pass
        act = get_activation(activation)
        
        layers: List[nn.Module] = []

        # input -> first hidden
        layers.append(nn.Linear(input_dim, width))
        layers.append(act)

        # hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(act)

        # FIXED: last hidden -> output (single output for binary classification)
        if num_classes == 2:
            layers.append(nn.Linear(width, 1))  # Binary: single output
        else:
            layers.append(nn.Linear(width, num_classes))  # Multi-class

        self.net = nn.Sequential(*layers)
        self.apply(init_weights_he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - FIXED for NTK computation.
        
        Args:
            x: Input tensor
               - circle: (batch_size, 2)
               - MNIST: (batch_size, 784) or (batch_size, 1, 28, 28)
        
        Returns:
            output: (batch_size,) for binary, (batch_size, num_classes) for multi-class
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through network
        out = self.net(x)
        
        # Handle binary classification - return scalar per sample
        if self.num_classes == 2:
            # Output is already (batch_size, 1), squeeze to (batch_size,)
            return out.squeeze(-1)
        else:
            # Multi-class: keep as (batch_size, num_classes)
            return out