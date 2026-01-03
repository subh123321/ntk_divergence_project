import torch
from torchvision import datasets, transforms

def generate_mnist_binary(
    n_samples=2000,
    device='cuda',
    dtype=torch.float64
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    X_list, Y_list = [], []

    for x, y in dataset:
        X_list.append(x.view(-1))
        Y_list.append(y % 2)   # odd = 1, even = 0

        if len(X_list) >= n_samples:
            break

    X = torch.stack(X_list).to(device=device, dtype=dtype)
    Y = torch.tensor(Y_list, dtype=dtype, device=device)
    Y_signed = 2 * Y - 1

    return X, Y, Y_signed

