import torch
import torch.nn as nn
from information import TotalCorrelationEstimator

def cd_loss(model, x_real, x_fake, energy_regularization=0.05, corr_param=0.1, return_components=False):
    e_real, h_real = model(x_real)
    e_fake, h_fake = model(x_fake)

    cd   = e_real.mean() - e_fake.mean()
    reg  = energy_regularization * (e_real.pow(2).mean() + e_fake.pow(2).mean())
    corr = corr_param * (head_correlation_penalty(h_real) + head_correlation_penalty(h_fake))

    total = cd + reg + corr

    if return_components:
        return total, cd, reg, corr
    return total


def head_correlation_penalty(H):
    if H.shape[0] < 2:
        return torch.tensor(0.0, device=H.device)
    
    H = H - H.mean(dim=0, keepdim=True)
    H = H / (H.std(dim=0, keepdim=True) + 1e-8)
    corr = H.T @ H / (H.shape[0] - 1)
    corr = corr - torch.eye(corr.size(0), device=corr.device)
    return corr.pow(2).mean()


def cd_loss_with_tc(
        model: nn.Module, 
        tc_estimator: TotalCorrelationEstimator, 
        x_real, x_fake,
        energy_regularization=0.05, 
        tc_regularizations=0.1, 
        return_components=False
    ):
    e_fake, h_fake = model(x_fake)
    e_real, h_real = model(x_real)
    head_outputs = model.head_outputs

    cd   = e_real.mean() - e_fake.mean()
    reg  = energy_regularization * (e_real.pow(2).mean() + e_fake.pow(2).mean())
    tc = tc_regularizations * tc_estimator.total_correlation(head_outputs)

    total = cd + reg + tc

    if return_components:
        return total, cd, reg, tc
    return total