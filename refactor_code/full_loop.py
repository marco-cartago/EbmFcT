import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.data_import import load_fashion_mnist
from src.model import EnergyHead, EBM
from src.sampler import ReplaySampler
from src.train import train_one_epoch
from src.diagnostic import diagnose


train_loader, test_loader = load_fashion_mnist(32, shuffle=True)


model = EBM(in_dim=28*28, mid_dim=150, n_heads=3, batch_size=32)

model.to(device)
for h in model.heads:
    h.to(device)


sampler = ReplaySampler(model, img_shape=(1, 28, 28), buffer_size=200, noise_fraction=0.05, device=device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))



for epoch in range(15):
    epoch_loss = train_one_epoch(model, sampler, train_loader, optimizer,
                                 sample_steps=60,
                                 sample_step_size=10.0,
                                 sample_noise_std=0.005,
                                 energy_reg=0.3,
                                 corr_param=0.7,
                                 device=device)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoint{epoch+1}_ebm_fmnist_3heads_150mid_32bsz.pth")

diagnose(model, sampler, train_loader, test_loader, device=device, n_samples=16)

torch.save(model.state_dict(), "ebm_fmnist_3heads_150mid_32bsz.pth")
