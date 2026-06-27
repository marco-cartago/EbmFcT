import torch
import numpy as np
import matplotlib.pyplot as plt
from src.sampler import ReplaySampler
from copy import deepcopy

def show_single_sample(x: torch.Tensor, title: str = None, cmap: str = "Gray"):
    """
    Visualizza una singola immagine PyTorch [C,H,W] o [H,W].
    """
    x = x.detach().cpu()

    # se batch, prendi primo elemento
    if x.ndim == 4:
        x = x[0]

    # CHW -> HWC
    if x.ndim == 3:
        x = x.permute(1, 2, 0)

    x = x.clamp(0, 1)

    plt.imshow(x, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

def compare_real_fake(real: torch.Tensor, fake: torch.Tensor):
    """
    Mostra fianco a fianco un esempio reale e uno generato.
    """
    real = real.detach().cpu()
    fake = fake.detach().cpu()

    if real.ndim == 4:
        real = real[0]
    if fake.ndim == 4:
        fake = fake[0]

    real = real.permute(1, 2, 0)
    fake = fake.permute(1, 2, 0)

    real = real.clamp(0, 1)
    fake = fake.clamp(0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(real)
    ax[0].set_title("Real")
    ax[0].axis("off")

    ax[1].imshow(fake)
    ax[1].set_title("Fake / EBM sample")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

def show_grid(batch, n=8):
    batch = batch[:n].detach().cpu()
    batch = (batch + 1) / 2  # se usi [-1,1]

    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))

    for i in range(n):
        img = batch[i].permute(1,2,0).clamp(0,1)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()


def visualize_heads(model, sampler_buffer, img_shape, device, k=4, cmap="hot"):
    model.eval()
    n_heads = len(model.heads)
    
    fig, axes = plt.subplots(k, n_heads + 1)
    model_sampler = ReplaySampler(model, img_shape=img_shape, buffer_size=50, noise_fraction=0.005, device=device)
    model_sampler.buffer = deepcopy(sampler_buffer)
    model_samples = model_sampler.sample(batch_size=k, steps=80, step_size=10, noise_std=0.05)

    axes[0, 0].set_title("Model")
    for s in range(k):
        axes[s, 0].imshow(model_samples[s].detach().squeeze(0).cpu().numpy(), cmap=cmap)
        axes[s, 0].axis("off")

    for i, head in enumerate(model.heads):
        sampler = model_sampler
        sampler.model = head
        sampler.buffer = deepcopy(sampler_buffer)
        head_samples = sampler.sample(batch_size=k, steps=200, step_size=10, noise_std=0.05)
        
        axes[0, i + 1].set_title(f"Head {i}")
        for s in range(k):
            axes[s, i + 1].imshow(head_samples[s].detach().squeeze(0).cpu().numpy(), cmap=cmap)
            axes[s, i + 1].axis("off")

    fig.suptitle("Sampled images", fontsize=12)
    fig.tight_layout()
    return fig
