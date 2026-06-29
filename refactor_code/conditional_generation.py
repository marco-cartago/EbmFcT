import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np

from src.data_import import load_fashion_mnist
from src.sampler import ReplaySampler
from src.model import EBM_Old, EnergyHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO]: device = {device}")


class EnergyHead(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        c, w, h = image_shape if len(image_shape) == 3 else image_shape[1:]

        self.conv_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c, 16, 3, padding=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        flat_dim = 64 * (w // 4) * (h // 4)

        self.linear = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(flat_dim, 7 * 7)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(7 * 7, 1))
        )

    def forward(self, x):
        return self.linear(self.conv_net(x))  # [B, 1]
    
    
class EBM(nn.Module):

    def __init__(self, image_shape: tuple, mid_dim: int, n_classes: int,
                 n_heads: int = 4, batch_size: int = 32) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_classes = n_classes

        self.heads = nn.ModuleList(
            [EnergyHead(image_shape) for _ in range(n_heads)]
        )
        self.head_outputs = torch.empty(batch_size, n_heads, requires_grad=False)

        self.combiner = nn.Linear(n_heads + n_classes, 1, bias=True)

    def forward(self, x, class_idx=None):

        h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)  # [B, K]
        self.head_outputs = h_o.detach()

        if class_idx is not None:
            c = F.one_hot(class_idx, num_classes=self.n_classes).float()  # [B, n_classes]
        else:
            c = torch.zeros(x.size(0), self.n_classes, device=x.device)

        combined = torch.cat([h_o, c], dim=1)           # [B, K + n_classes]
        energy   = self.combiner(combined).squeeze(-1)   # [B]

        return energy, h_o

class ConditionalReplaySampler(ReplaySampler):

    def __init__(self, model, img_shape, n_classes, buffer_size=8192, noise_fraction=0.05, device=None):
        super().__init__(model, img_shape, buffer_size, noise_fraction, device)
        # Sostituisce il buffer flat con uno per classe
        self.n_classes = n_classes
        self.buffer = {c: [] for c in range(n_classes)}

    def initialize_chains(self, batch_size, class_idx):
        # class_idx è un tensor [B] — campiona per ogni classe separatamente
        if isinstance(class_idx, int):
            class_idx = torch.full((batch_size,), class_idx, dtype=torch.long)

        samples = []
        for c_val in class_idx:
            c = c_val.item()
            buf = self.buffer[c]

            if len(buf) > 0 and np.random.random() > self.noise_fraction:
                i = np.random.randint(len(buf))
                samples.append(buf[i])
            else:
                samples.append(torch.rand(*self.img_shape) * 2 - 1)

        return torch.stack(samples).to(self.device)

    def update_buffer(self, samples, class_idx):
        if isinstance(class_idx, int):
            class_idx = torch.full((samples.size(0),), class_idx, dtype=torch.long)

        for i, c_val in enumerate(class_idx):
            c = c_val.item()
            self.buffer[c].append(samples[i].detach().cpu())
            if len(self.buffer[c]) > self.buffer_size:
                self.buffer[c] = self.buffer[c][-self.buffer_size:]

    def langevin_step(self, x, class_idx, step_size=10.0, noise_std=0.005, grad_clip=0.03):
        x = x.detach()
        x = (x + torch.randn_like(x) * noise_std).clamp(-1, 1)

        with torch.enable_grad():
            x.requires_grad_(True)
            energy = self.model(x, class_idx=class_idx)[0].sum()
            grad = torch.autograd.grad(energy, x, only_inputs=True)[0]

        grad = grad.detach().clamp(-grad_clip, grad_clip)
        return (x.detach() - step_size * grad).clamp(-1, 1)

    def run_langevin(self, x, class_idx, steps=60, step_size=10.0, noise_std=0.005):
        was_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        for _ in range(steps):
            x = self.langevin_step(x, class_idx, step_size=step_size, noise_std=noise_std)

        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train(was_training)
        return x

    def sample(self, batch_size, class_idx, steps=60, step_size=10.0, noise_std=0.005):
        if isinstance(class_idx, int):
            class_tensor = torch.full((batch_size,), class_idx, dtype=torch.long, device=self.device)
        else:
            class_tensor = class_idx.to(self.device)

        x = self.initialize_chains(batch_size, class_idx)
        x = self.run_langevin(x, class_tensor, steps=steps, step_size=step_size, noise_std=noise_std)
        self.update_buffer(x, class_idx)
        return x

    def generate(self, n_samples, class_idx, steps=60, step_size=10.0, noise_std=0.005, batch_size=128):
        generated = []
        while len(generated) * batch_size < n_samples:
            x = self.sample(batch_size=batch_size, class_idx=class_idx,
                            steps=steps, step_size=step_size, noise_std=noise_std)
            generated.append(x.detach().cpu())
        return torch.cat(generated)[:n_samples] 
   
def cd_loss(model, x_real, x_fake, class_idx, energy_regularization=0.05, grad_div_param=0.1, return_components=False):
    e_real, h_real = model(x_real, class_idx=class_idx)
    e_fake, h_fake = model(x_fake, class_idx=class_idx)

    cd  = e_real.mean() - e_fake.mean()
    reg = energy_regularization * (e_real.pow(2).mean() + e_fake.pow(2).mean())
    div = grad_div_param * head_gradient_diversity_penalty(model, x_real, class_idx)

    total = cd + reg + div

    if return_components:
        return total, cd, reg, div
    return total

def head_gradient_diversity_penalty(model, x, class_idx):
    x = x.detach().requires_grad_(True)

    grads = []
    for head in model.heads:
        e_i = head(x).sum()  # class_idx rimosso
        g_i = torch.autograd.grad(e_i, x, create_graph=True)[0]
        grads.append(g_i.flatten(start_dim=1))

    penalty = torch.tensor(0.0, device=x.device)
    n_pairs = 0

    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            gi = grads[i] / (grads[i].norm(dim=1, keepdim=True) + 1e-8)
            gj = grads[j] / (grads[j].norm(dim=1, keepdim=True) + 1e-8)
            penalty = penalty + (gi * gj).sum(dim=1).pow(2).mean()
            n_pairs += 1

    return penalty / n_pairs

def train_one_epoch(model, sampler, train_loader, optimizer, sample_steps, sample_step_size, sample_noise_std, energy_reg, grad_div_param, device="cpu"):

    model.train()
    running_loss   = 0.0
    running_cd     = 0.0
    running_reg    = 0.0
    running_div    = 0.0
    running_e_real = 0.0
    running_e_fake = 0.0

    for x_real, y in tqdm.tqdm(train_loader):

        x_real = x_real.to(device)
        y      = y.to(device)

        x_neg = sampler.sample(batch_size=x_real.size(0), class_idx=y,
                               steps=sample_steps, step_size=sample_step_size,
                               noise_std=sample_noise_std)

        e_fake, _ = model(x_neg, class_idx=y)
        e_real, _ = model(x_real, class_idx=y)

        loss, cd, reg, div = cd_loss(
            model=model,
            x_fake=x_neg,
            x_real=x_real,
            class_idx=y,
            energy_regularization=energy_reg,
            grad_div_param=grad_div_param,
            return_components=True
        )

        running_loss   += loss.item()
        running_cd     += cd.item()
        running_reg    += reg.item()
        running_div    += div.item()
        running_e_real += e_real.mean().item()
        running_e_fake += e_fake.mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    n = len(train_loader)
    print(f"  CD:     {running_cd    / n:.4f}")
    print(f"  Reg:    {running_reg   / n:.4f}")
    print(f"  GradDiv:{running_div   / n:.4f}")
    print(f"  E_real: {running_e_real / n:.4f}")
    print(f"  E_fake: {running_e_fake / n:.4f}")
    print(f"  Gap:    {(running_e_real - running_e_fake) / n:.4f}")

    return running_loss / n


train_loader, test_loader = load_fashion_mnist(batch_size=32, class_subset=[0, 2, 9])
print(f"[INFO]: dataset loaded: train_len = {len(train_loader.dataset)} | test_len = {len(test_loader.dataset)}")

image_shape = (1, 28, 28)
n_classes   = 10

model = EBM(image_shape=image_shape, mid_dim=150, n_classes=n_classes, n_heads=4, batch_size=32)
model.to(device)

sampler = ConditionalReplaySampler(model, img_shape=image_shape, n_classes=n_classes,
                                   buffer_size=200, noise_fraction=0.05, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

for epoch in range(15):
    epoch_loss = train_one_epoch(model, sampler, train_loader, optimizer,
                                 sample_steps=60,
                                 sample_step_size=10.0,
                                 sample_noise_std=0.005,
                                 energy_reg=0.3,
                                 grad_div_param=0.1,
                                 device=device)
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"conditional/checkpoint{epoch+1}_ebm_fmnist_4heads_150mid_conditional.pth")

torch.save(model.state_dict(), "conditional/ebm_fmnist_4heads_150mid_conditional.pth")