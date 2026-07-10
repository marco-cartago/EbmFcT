import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def evaluate_sampling_fid(real_images: torch.Tensor, fake_images: torch.Tensor):
    """
    Evaluates the Frechet Inception Distance on the generated images.
    """

    fid = FrechetInceptionDistance(feature=2048)

    # The fid model needs RGB so we repat 3 times the same channel (MNIST is grayscale)
    real_images = real_images.repeat(1, 3, 1, 1)
    fake_images = fake_images.repeat(1, 3, 1, 1)

    # The fid model needs uint8 images in [0, 255]
    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
    fake_images = ((fake_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)

    batch_size = 64

    with torch.inference_mode():
        for i in range(0, len(real_images), batch_size):
            fid.update(real_images[i : i + batch_size], real=True)

        for i in range(0, len(fake_images), batch_size):
            fid.update(fake_images[i : i + batch_size], real=False)
            fid.update(fake_images[i:i+batch_size], real=False)

    score = fid.compute()
    print("FID:", score.item())
