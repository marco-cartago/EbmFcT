import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnergyHead(nn.Module):

    def __init__(self, in_dim: int, mid_dim: int) -> None:
        super(EnergyHead, self).__init__()
        self.C1 = nn.Conv2d(1, 5, 3)
        self.MP1 = nn.MaxPool2d(2)
        self.C2 = nn.Conv2d(5, 5, 3)
        self.MP2 = nn.MaxPool2d(2)
        self.L3 = nn.Linear(125, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, single=False):
        x = self.C1(x)
        x = self.act(x)
        x = self.MP1(x)

        x = self.C2(x)
        x = self.act(x)
        x = self.MP2(x)
        x = x.flatten(start_dim=1)

        x = self.L3(x)

        if single:
            x = x.flatten(start_dim=0)

        return x


class EBM(nn.Module):

    def __init__(self, in_dim: int, mid_dim: int, n_heads: int = 4, batch_size: int = 32) -> None:
        super(EBM, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.mid_dim = mid_dim
        self.n_heads = n_heads

        self.heads = nn.ModuleList(
            [EnergyHead(in_dim, mid_dim) for _ in range(n_heads)]
        )
        self.head_outputs = torch.empty(batch_size, n_heads)
        # self.head_weights = nn.Parameter(torch.randn(n_heads, 1))

    def forward(self, x: torch.Tensor, single=False) -> torch.Tensor:
        h_o = torch.stack([head(x, single=False)
                          for head in self.heads], dim=1).squeeze(-1)
        self.head_outputs = h_o
        out_pb = torch.sum(h_o, dim=1)
        return out_pb
