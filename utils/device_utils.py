import torch

def get_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("⚠️ CUDA not available, using CPU")

    return device
