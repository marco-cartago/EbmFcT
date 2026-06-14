import torch
import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from ebm import *
from sampler import *


def load_fashion_mnist(batch_size=64, shuffle=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
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


def load_mnist_digit_9(batch_size=64, shuffle=True):
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

    # Filter indices where label is 9
    train_idx = [i for i in range(len(train_set)) if train_set[i][1] == 0]
    test_idx = [i for i in range(len(test_set)) if test_set[i][1] == 0]

    train_set = Subset(train_set, train_idx)
    test_set = Subset(test_set, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def loss(
    model: EBM,
    x: torch.Tensor,
    m: float = 1.0,
    n_samples: int = 4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:

    tru_energy = model(x)

    # head_outputs = model.head_outputs
    # head_outputs = head_outputs.squeeze(-1)
    # X_centered = head_outputs - head_outputs.mean(dim=0, keepdim=True)
    # cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
    # std = X_centered.std(dim=0, unbiased=True, keepdim=True)
    # corr = cov / (std.T @ std)
    # corr_norm = torch.norm(corr, p="fro") - model.n_heads

    samples = []

    for i in range(x.size(0)):
        sample_start = x[i]
        sample_start = sample_start.unsqueeze(0)
        sample, _ = langevin_sample_from(model, sample_start, n_step=100)
        samples.append(sample)

    gen_energy = torch.cat([model(s) for s in samples])

    l = torch.mean(tru_energy - gen_energy) #+ corr_norm
    return l


def train_one_epoch(model, loader, optimizer, criterion, device, s=20):
    model.train()
    running_loss = 0.0

    for inputs, _ in tqdm.tqdm(loader):
        inputs = inputs.to(device)
        inputs += torch.rand_like(inputs) / s
        optimizer.zero_grad()
        loss = criterion(model, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for inputs, _ in tqdm.tqdm(loader):
        inputs = inputs.to(device)
        loss = criterion(model, inputs)
        running_loss += loss.item() * inputs.size(0)

    running_loss /= len(loader.dataset)

    return running_loss


def show_image(x, energy, epoch_idx):
    image_np = x.squeeze().to('cpu').detach().numpy()
    image_np = np.clip(image_np, 0, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title(f"Energy: {energy.item()}")
    plt.axis("off")
    plt.savefig(f"./images/sample_{epoch_idx:.4f}.png")


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    batch_size,
    optimizer,
    criterion,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):

    train_losses = []
    val_losses = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss = 0
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | " +
            f"Train Loss: {train_loss:.4f} | " +
            f"Val Loss: {val_loss:.4f} | "
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 1 == 0:
            img, energy = langevin_sample_from(
                model,
                torch.rand((1,1,28,28)).to(device),
                n_step=1000
            )
            show_image(img, energy, epoch)

    return train_losses, val_losses


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sz = 32
    epochs = 2000
    torch.manual_seed(0) # Random state

    # Data loading
    train_loader, test_loader = load_mnist_digit_9(batch_size=batch_sz)

    # Model init
    model = EBM(28 * 28, 150, n_heads=1, batch_size=batch_sz)
    model.to(device)
    for h in model.heads:
        h.to(device)

    # Optim
    optim = torch.optim.Adam(model.parameters(), 1e-3)

    # Model training
    train_losses, val_losses = train(
        model,
        train_loader,
        test_loader,
        epochs,
        batch_sz,
        optim,
        loss
    )

    print(train_losses)
    print(val_losses)


if __name__ == "__main__":
    main()
