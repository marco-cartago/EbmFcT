import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 16, 3, padding=1)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)), # 14x14
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)), # 7x7
            nn.GELU(),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(64 * 7 * 7, 1))
        )
    
    def forward(self, x):
        return self.net(x)  # (B, 1)



class EBM(nn.Module):

    def __init__(self, in_dim: int, mid_dim: int, n_heads: int = 4, batch_size: int = 32) -> None:
        super(EBM, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.mid_dim = mid_dim
        self.n_heads = n_heads
        self.device = None

        self.heads = nn.ModuleList(
            [EnergyHead() for _ in range(n_heads)]
        )
        self.head_outputs = torch.empty(batch_size, n_heads)

    def forward(self, x, head_idx=None):

        if head_idx is not None:
            h_o = self.heads[head_idx](x).squeeze(-1)

        else:
            h_o = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=1)
            self.head_outputs = h_o # Used to estimate the TC

        if h_o.dim() == 1:
            energy = h_o
        else:
            energy = h_o.sum(dim=1)

        return energy, h_o
