import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

class EnergyHead(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        c, h, w = image_shape

        # first three conv blocks
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(c, 16, 3, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.GELU()
        )

        # compute output spatial size
        h_out = math.floor((h + 2*1 - 3) / 1 + 1)
        w_out = math.floor((w + 2*1 - 3) / 1 + 1)

        h_out = math.floor((h_out + 2*1 - 4) / 2 + 1)      
        w_out = math.floor((w_out + 2*1 - 4) / 2 + 1)

        h_out = math.floor((h_out + 2*1 - 4) / 2 + 1)
        w_out = math.floor((w_out + 2*1 - 4) / 2 + 1)

        self.flat_dim = 64 * h_out * w_out

        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.flat_dim, 7 * 7)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(7 * 7, 1))
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)



class SmallEnergyHead(nn.Module):
    def __init__(self, image_shape) -> None:
        super().__init__()
        self.image_shape = image_shape
        b, w, h = image_shape 
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(b, 4, 3, stride=1)), # (B, 1, 12, 12)
            nn.Flatten(),
            nn.GELU(),
            nn.utils.spectral_norm((nn.Linear(4 * (w-2) * (w-2), 32))),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(32, 1))
        )
    
    def forward(self, x):
        return self.net(x)
    


class EBM(nn.Module):

    def __init__(self, image_shape: tuple, mid_dim: int, n_heads: int = 4, batch_size: int = 32) -> None:
        super(EBM, self).__init__()
        self.in_dim = image_shape
        self.out_dim = 1
        self.mid_dim = mid_dim
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList(
            [EnergyHead(image_shape) for _ in range(n_heads)]
        )
        self.head_outputs = torch.empty(batch_size, n_heads, requires_grad=False)

    def forward(self, x, head_idx=None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)
            self.head_outputs = h_o.detach() # Used to estimate the TC

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o


class SmallEBM(nn.Module):

    def __init__(self, image_shape: tuple, mid_dim: int, n_heads: int = 4, batch_size: int = 32) -> None:
        super(SmallEBM, self).__init__()
        self.in_dim = image_shape
        self.out_dim = 1
        self.mid_dim = mid_dim
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList(
            [SmallEnergyHead(image_shape) for _ in range(n_heads)]
        )
        self.head_outputs = torch.empty(batch_size, n_heads, requires_grad=False)

    def forward(self, x, head_idx=None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)
            self.head_outputs = h_o.detach() # Used to estimate the TC

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o

class EnergyHeadOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 16, 3, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.GELU(),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(64 * 7 * 7, 1))  # 3136 → 1
        )

    def forward(self, x):
        return self.net(x)

class EBM_Old(nn.Module):

    def __init__(self, in_dim: int, mid_dim: int, n_heads: int = 4, batch_size: int = 32) -> None:
        super(EBM_Old, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.mid_dim = mid_dim
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList(
            [EnergyHeadOld() for _ in range(n_heads)]
        )
        self.head_outputs = torch.empty(batch_size, n_heads, requires_grad=False)

    def forward(self, x, head_idx=None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)
            self.head_outputs = h_o.detach() # Used to estimate the TC

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o
