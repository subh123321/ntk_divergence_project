# models/resnet.py
import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, width, activation="relu"):
        super().__init__()

        self.fc = nn.Linear(width, width, bias=False)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # NTK-consistent residual scaling
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, x):
        return x + self.scale * self.activation(self.fc(x))


class FullyConnectedResNet(nn.Module):
    """
    Fully-connected ResNet used in NTK divergence experiments
    """
    def __init__(self, input_dim, width, depth, num_classes=1, activation="relu"):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, width)

        self.blocks = nn.ModuleList(
            [ResidualBlock(width, activation=activation) for _ in range(depth)]
        )

        self.output_layer = nn.Linear(width, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)

        out = self.output_layer(h)
        return out.squeeze(-1)
