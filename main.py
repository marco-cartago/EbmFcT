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

def load_mnist(batch_size=64, shuffle=True):
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
    sampler: Sampler,
    m: float = 1.0,
    n_samples: int = 4,
    alpha: float = 0.1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:

    tru_energy = model(x)
    samples = sampler.sample_new_exmps()
    gen_energy = torch.cat(
        [model(samples[i].unsqueeze(0)) for i in range(samples.size(0))]
    )
    l = torch.mean(tru_energy) - torch.mean(gen_energy)
    r = alpha * (torch.mean(tru_energy**2) + torch.mean(gen_energy**2))

    return l + r


def train_one_epoch(model, loader, optimizer, criterion, sampler, device, s=20):
    model.train()
    running_loss = 0.0

    for inputs, _ in tqdm.tqdm(loader):
        inputs = inputs.to(device)
        inputs += torch.rand_like(inputs) / s
        optimizer.zero_grad()
        loss = criterion(model, inputs, sampler)
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


def save_image(x, energy, epoch_idx):
    image_np = x.squeeze().to('cpu').detach().numpy()
    image_np = np.clip(image_np, 0, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title(f"Energy: {energy.item()}")
    plt.axis("off")
    plt.savefig(f"./images/sample_{epoch_idx:.4f}.png")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    batch_size: int,
    optimizer,
    criterion,
    img_shape: tuple,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):

    train_losses = []
    val_losses = []
    model.to(device)

    sampler = Sampler(model, img_shape, batch_size, device)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, sampler, device
        )
        val_loss = 0
        # val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | " +
            f"Train Loss: {train_loss:.4f} | " +
            f"Val Loss: {val_loss:.4f} | "
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 1 == 0:
            img = sampler.examples[0]
            save_image(img, model(img), epoch)

    return train_losses, val_losses


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sz = 32
    epochs = 2000
    torch.manual_seed(0) # Random state
    img_shape = (1, 28, 28)

    # Data loading
    train_loader, test_loader = load_mnist(batch_size=batch_sz)

    # Model init
    model = EBM(28 * 28, 150, n_heads=1, batch_size=batch_sz)
    model.to(device)

    # Optim
    # It seems EBMs have issue with moment.
    optim = torch.optim.Adam(model.parameters(), 1e-3, betas=(0.0, 0.999))

    # Model training
    train_losses, val_losses = train(
        model,
        train_loader,
        test_loader,
        epochs,
        batch_sz,
        optim,
        loss,
        img_shape,
    )

    print(train_losses)
    print(val_losses)


if __name__ == "__main__":
    main()
