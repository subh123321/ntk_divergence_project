import torch
import math
from torch import nn
from .utils import init_weights_ntk, get_activation


class NTKFullyConnectedNetwork(nn.Module):
    """
    Fully Connected Network with NTK parameterization.
    """

    def __init__(self, input_dim, num_classes,depth, width):
        super().__init__()
        self.num_classes=num_classes
        self.width = width
        self.activation = get_activation()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))

        for _ in range(depth - 2):
            self.layers.append(nn.Linear(width, width))

        self.layers.append(nn.Linear(width, 1))

        # NTK initialization
        self.apply(init_weights_ntk)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        h = x
        for layer in self.layers[:-1]:
            h = layer(h)
            
            h = math.sqrt(2.0 / self.width) * self.activation(h)


        return self.layers[-1](h).squeeze(-1)
