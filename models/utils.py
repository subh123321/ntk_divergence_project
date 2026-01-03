import torch
import torch.nn as nn
import numpy as np


def get_activation(name="relu"):
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("Only ReLU/Tanh/GeLU supported for NTK experiments")


def init_weights_ntk(module: nn.Module, scale: float = 1.0):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=scale)
        nn.init.zeros_(module.bias)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


INIT_SCALES = {
    "small": 0.5,
    "standard": 1.0,
    "large": 2.0
}


def initialize_with_scale(model, scale_name="standard"):
    scale = INIT_SCALES[scale_name]
    model.apply(lambda m: init_weights_ntk(m, scale))
