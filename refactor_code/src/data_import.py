import torch

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms

def load_fashion_mnist(batch_size=64, shuffle=True, class_subset=None):
    """Load FashionMNIST dataset with optional class subset filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        class_subset (list, optional): List of class labels to include. If None, includes all classes.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    if class_subset is not None:
        # Filter indices where label is in the specified class subset
        train_idx = [i for i in range(len(train_set)) if train_set[i][1] in class_subset]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] in class_subset]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def load_mnist_digit(batch_size=64, shuffle=True, digit=None):
    """Load MNIST dataset with optional digit filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        digit (int, optional): Digit label to include. If None, includes all digits.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    if digit is not None:
        # Filter indices where label is the specified digit
        train_idx = [i for i in range(len(train_set)) if train_set[i][1] == digit]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] == digit]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

