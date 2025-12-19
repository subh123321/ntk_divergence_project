import torch
from torchvision import datasets, transforms

def generate_mnist_binary(
    digits=(0, 1),
    n_samples=30000,
    device='cuda'
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
        if y in digits:
            X_list.append(x.view(-1))
            Y_list.append(y % 2)

        if len(X_list) >= n_samples:
            break

    X = torch.stack(X_list).to(device)
    Y = torch.tensor(Y_list,dtype=torch.long, device=device)

    return X, Y
