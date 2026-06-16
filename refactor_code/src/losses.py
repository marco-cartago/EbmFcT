import torch
import torch.nn as nn

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

    H = H - H.mean(dim=0, keepdim=True)

    H = H / (
        H.std(dim=0, keepdim=True)
        + 1e-8
    )

    corr = H.T @ H / (H.shape[0] - 1)

    corr = corr - torch.eye(
        corr.size(0),
        device=corr.device
    )

    return corr.pow(2).mean()