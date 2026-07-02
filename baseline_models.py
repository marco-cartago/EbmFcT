"""
Baseline models per confronto con l'EBM: VAE e Diffusion (DDPM).

Entrambi i modelli lavorano in [-1, 1] (stessa convenzione del sampler EBM),
così i sample generati sono direttamente compatibili con evaluate_sampling_fid
in evaluation.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# VAE
# =====================================================================

class VAE(nn.Module):
    """
    VAE convoluzionale generico, funziona con qualsiasi img_size che sia
    multiplo di 4 (28 -> con padding gestito, 64 -> ok nativamente).

    Decoder termina con Tanh -> output in [-1, 1].
    """

    def __init__(self, img_channels=1, img_size=28, latent_dim=32, base_ch=32):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.enc = nn.Sequential(
            nn.Conv2d(img_channels, base_ch, 4, stride=2, padding=1),      # /2
            nn.BatchNorm2d(base_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),        # /4
            nn.BatchNorm2d(base_ch * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),    # /8
            nn.BatchNorm2d(base_ch * 4), nn.LeakyReLU(0.2, inplace=True),
        )

        # Determina dinamicamente la shape flattenata con un forward fittizio
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, img_size, img_size)
            enc_out = self.enc(dummy)
            self._enc_shape = enc_out.shape[1:]  # (C, H, W)
            flat_dim = enc_out.numel()

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # --- Decoder ---
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, img_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, *self._enc_shape)
        out = self.dec(h)
        # Se per arrotondamenti la size non torna esatta, forza il resize
        if out.shape[-1] != self.img_size:
            out = F.interpolate(out, size=(self.img_size, self.img_size),
                                 mode="bilinear", align_corners=False)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon, x, mu, logvar, kld_weight=1e-3):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_weight * kld, recon_loss, kld

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def train_vae(model, loader, epochs=20, lr=1e-3, device="cpu",
              kld_weight=1e-3, normalize_to_pm1=False, log_every=1):
    """
    normalize_to_pm1: True se il loader ritorna immagini in [0,1]
    (es. load_olivetti) e vanno rimappate in [-1,1] per coerenza col resto.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            if normalize_to_pm1:
                x = x * 2 - 1

            recon, mu, logvar = model(x)
            loss, recon_l, kld_l = model.loss_function(recon, x, mu, logvar, kld_weight)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % log_every == 0:
            print(f"[VAE] epoch {epoch+1}/{epochs} - loss: {total_loss/n_batches:.4f}")

    return model


# =====================================================================
# Diffusion (DDPM) con piccola UNet
# =====================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """
    UNet compatta per DDPM su immagini piccole (28x28 / 64x64).
    Predice il rumore epsilon aggiunto a x_t.
    """

    def __init__(self, img_channels=1, base_ch=32, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, time_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)   # /2

        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # /4

        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.up_block2 = ResBlock(base_ch * 4, base_ch, time_dim)  # cat con skip down2

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.up_block1 = ResBlock(base_ch * 2, base_ch, time_dim)  # cat con skip down1

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, img_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h1p = self.pool1(h1)

        h2 = self.down2(h1p, t_emb)
        h2p = self.pool2(h2)

        hm = self.mid(h2p, t_emb)

        u2 = self.up2(hm)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.up_block2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up_block1(u1, t_emb)

        return self.out_conv(u1)


class DiffusionModel:
    """
    Wrapper DDPM: beta schedule lineare, loss su epsilon, sampling ancestrale.
    """

    def __init__(self, img_channels=1, img_size=28, base_ch=32,
                 timesteps=500, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.img_channels = img_channels
        self.img_size = img_size
        self.timesteps = timesteps
        self.device = device

        self.unet = SimpleUNet(img_channels=img_channels, base_ch=base_ch).to(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def parameters(self):
        return self.unet.parameters()

    def to(self, device):
        self.device = device
        self.unet.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def train_mode(self):
        self.unet.train()

    def eval_mode(self):
        self.unet.eval()

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise, noise

    def loss(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        x_t, noise = self.q_sample(x0, t)
        pred_noise = self.unet(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or self.device
        x = torch.randn(n, self.img_channels, self.img_size, self.img_size, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]

            pred_noise = self.unet(x, t)
            coef = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = (1 / torch.sqrt(alpha_t)) * (x - coef * pred_noise)

            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x.clamp(-1.0, 1.0)


def train_diffusion(diffusion, loader, epochs=20, lr=2e-4, device="cpu",
                     normalize_to_pm1=False, log_every=1):
    diffusion.to(device)
    opt = torch.optim.Adam(diffusion.parameters(), lr=lr)

    for epoch in range(epochs):
        diffusion.train_mode()
        total_loss, n_batches = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            if normalize_to_pm1:
                x = x * 2 - 1

            loss = diffusion.loss(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % log_every == 0:
            print(f"[Diffusion] epoch {epoch+1}/{epochs} - loss: {total_loss/n_batches:.4f}")

    return diffusion