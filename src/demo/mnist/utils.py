import torch
from model import Net
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchviz import make_dot


def get_mnist_loaders(batch_size=256):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize():
    model = Net()

    x = torch.randn(256, 1, 28, 28)

    output = model(x)

    dot = make_dot(output, params=dict(model.named_parameters()))

    dot.format = "pdf"
    dot.render("network_graph")
