"""
Allena VAE e Diffusion baseline su:
  - Olivetti faces (64x64, 1 canale)
  - FashionMNIST, solo classe 9 (28x28, 1 canale)

Salva:
  - checkpoint dei modelli in ./checkpoints/
  - un tensore di sample generati per ciascun modello in ./generated/
    (in [-1,1], compatibile con evaluate_sampling_fid di evaluation.py)

Uso:
    python train_baselines.py
"""

import os
import torch

from src.data_import import load_olivetti, load_fashion_mnist
from baseline_models import VAE, DiffusionModel, train_vae, train_diffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "./checkpoints"
GEN_DIR = "./generated"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(GEN_DIR, exist_ok=True)

N_SAMPLES_FOR_FID = 500  # quanti sample generare per il confronto FID successivo


# =====================================================================
# Config dei due dataset
# =====================================================================
# normalize_to_pm1=True per Olivetti perché load_olivetti ritorna [0,1]
# (nessuna Normalize nel transform), mentre load_fashion_mnist ritorna
# già [-1,1] (Normalize(0.5, 0.5)).

DATASETS = {
    "olivetti": dict(
        loader_fn=lambda: load_olivetti(batch_size=32, shuffle=True),
        img_size=64,
        img_channels=1,
        normalize_to_pm1=True,
        vae_epochs=60,
        diff_epochs=60,
    ),
    "fashionmnist_class9": dict(
        loader_fn=lambda: load_fashion_mnist(batch_size=64, shuffle=True, class_subset=[9]),
        img_size=28,
        img_channels=1,
        normalize_to_pm1=False,
        vae_epochs=30,
        diff_epochs=30,
    ),
}


def run_for_dataset(name, cfg):
    print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")

    train_loader, test_loader = cfg["loader_fn"]()
    img_size = cfg["img_size"]
    img_channels = cfg["img_channels"]
    normalize_to_pm1 = cfg["normalize_to_pm1"]

    # --- VAE ---
    print("\n--- Training VAE ---")
    vae = VAE(img_channels=img_channels, img_size=img_size, latent_dim=64)
    vae = train_vae(
        vae, train_loader, epochs=cfg["vae_epochs"], lr=1e-3,
        device=DEVICE, normalize_to_pm1=normalize_to_pm1,
    )
    vae_ckpt = os.path.join(CKPT_DIR, f"vae_{name}.pt")
    torch.save(vae.state_dict(), vae_ckpt)
    print(f"VAE salvato in {vae_ckpt}")

    vae.eval()
    with torch.no_grad():
        vae_samples = vae.sample(N_SAMPLES_FOR_FID, device=DEVICE).cpu()
    torch.save(vae_samples, os.path.join(GEN_DIR, f"vae_{name}_samples.pt"))

    # --- Diffusion ---
    print("\n--- Training Diffusion ---")
    diffusion = DiffusionModel(
        img_channels=img_channels, img_size=img_size, base_ch=32,
        timesteps=500, device=DEVICE,
    )
    diffusion = train_diffusion(
        diffusion, train_loader, epochs=cfg["diff_epochs"], lr=2e-4,
        device=DEVICE, normalize_to_pm1=normalize_to_pm1,
    )
    diff_ckpt = os.path.join(CKPT_DIR, f"diffusion_{name}.pt")
    torch.save(diffusion.unet.state_dict(), diff_ckpt)
    print(f"Diffusion (UNet) salvato in {diff_ckpt}")

    diffusion.eval_mode()
    # generazione a batch per non saturare la memoria (DDPM sampling è lento: timesteps step per batch)
    gen_batches = []
    remaining = N_SAMPLES_FOR_FID
    batch = 64
    while remaining > 0:
        n = min(batch, remaining)
        gen_batches.append(diffusion.sample(n, device=DEVICE).cpu())
        remaining -= n
    diff_samples = torch.cat(gen_batches, dim=0)
    torch.save(diff_samples, os.path.join(GEN_DIR, f"diffusion_{name}_samples.pt"))

    print(f"\nSample salvati in {GEN_DIR}/vae_{name}_samples.pt e diffusion_{name}_samples.pt")


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        run_for_dataset(name, cfg)

    print("\nFatto. Per confrontare con l'EBM via FID:")
    print("""
from evaluation import evaluate_sampling_fid
import torch

real_images = ...        # batch di immagini reali in [0,1], shape [N,1,H,W]
vae_samples = torch.load("generated/vae_olivetti_samples.pt")       # [-1,1]
diff_samples = torch.load("generated/diffusion_olivetti_samples.pt") # [-1,1]
ebm_samples = ...         # tuoi sample EBM, [-1,1]

evaluate_sampling_fid(real_images, vae_samples)
evaluate_sampling_fid(real_images, diff_samples)
evaluate_sampling_fid(real_images, ebm_samples)
""")