import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from src.data_import import load_fashion_mnist
from src.model import EnergyHead, EBM
from src.sampler import ReplaySampler
from src.train import train_one_epoch
from src.diagnostic import diagnose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO]: device = {device}")

img_shape = (1, 28, 28)
classes = [4]
heads = 4
batch_size = 4


train_loader, test_loader = load_fashion_mnist(batch_size=batch_size, class_subset=classes)
print(f"[INFO]: dataset imported FMNIST(classes = {classes}) train_len = {len(train_loader.dataset)} | test_len = {len(test_loader.dataset)}")

model = EBM(image_shape = img_shape, n_heads=heads, batch_size=32)
print(f"[INFO]: model instanciated {model}  | {heads} heads")

model.to(device)
for h in model.heads:
    h.to(device)

sampler = ReplaySampler(model, img_shape = img_shape, buffer_size=200, noise_fraction=0.05, device=device)
print(f"[INFO]: sampler instanciated {sampler}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.001, 0.999))


epoch_loss = train_one_epoch(model, sampler, train_loader, optimizer,
                                sample_steps=50,
                                sample_step_size=10.0,
                                sample_noise_std=0.005,
                                energy_reg=0.3,
                                corr_param=0.1,
                                device=device)

print(f"Epoch {0}, Loss: {epoch_loss:.4f}")
    # if (epoch + 1) % 5 == 0:
    #     torch.save(model.state_dict(), f"checkpoint{epoch+1}_ebm_fmnist_3heads_150mid_64bsz.pth")

diagnose(model = model, sampler = sampler, img_shape = img_shape, train_loader = train_loader, test_loader = test_loader, device=device, n_samples=100)
print(f"[INFO]: End of diagnosis")

torch.save(model.state_dict(), "ebm_fmnist_4heads_150mid_4bsz.pth")
