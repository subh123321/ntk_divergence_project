# models/fcn.py
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
    def __init__( self,input_dim: int,num_classes: int,depth: int = 3,width: int = 2000,activation: str = "relu", ) -> None:

        super().__init__()

        if depth < 2:
            raise ValueError("depth must be >= 2")

        act = get_activation(activation)
        
        layers: List[nn.Module] = []

        # input -> first hidden
        layers.append(nn.Linear(input_dim, width))
        layers.append(act)

        # hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act)

        # last hidden -> output
        layers.append(nn.Linear(width, num_classes))

        self.net = nn.Sequential(*layers)
        self.apply(init_weights_he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape:
        # - circle: (batch_size, 2)
        # - MNIST: (batch_size, 1, 28, 28) or (batch_size, 28, 28)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
