import torch
import matplotlib.pyplot as plt

def show_single_sample(x: torch.Tensor, title: str = None):
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

    plt.imshow(x)
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