
from typing import Tuple

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


# Langevin dynamics with preconditioning and momentum
# -------------------------------------------------------------------------------------- #


def langevin_step_precond(
    x: torch.Tensor,
    M: nn.Module,
    precond: torch.Tensor,
    eta: float = 1e-3,
    beta: float = 1.0,
    momentum: float = 0.9
):

    x.requires_grad_(True)
    U = M(x)
    U.backward()
    grad = x.grad
    grad = grad.clamp(-10, 10)

    # Update preconditioner
    precond.mul_(momentum).add_(grad.pow(2), alpha=1-momentum)

    # Preconditioned step
    step = eta * grad / (precond.sqrt() + 1e-8)
    noise = torch.randn_like(x) * torch.sqrt(2 * eta /
                                             (beta * precond.sqrt() + 1e-8))
    x.data = x - step + noise

    x.grad.zero_()
    return x, precond


def langevin_sample_from(
    M: nn.Module,
    x0: torch.Tensor,
    n_step: int = 1_000,
    eta: float = 1e-3,
    beta: float = 1.0,
    momentum: float = 0.9
):
    M.eval()
    x = x0
    precond = torch.ones_like(x)

    for _ in range(n_step):
        x, precond = langevin_step_precond(
            x, M, precond, eta=eta, beta=beta, momentum=momentum)

    return (x, M(x))


# Hamiltonian Monte Carlo (HMC) sampling
# -------------------------------------------------------------------------------------- #


def leapfrog_step(
    x: torch.Tensor,
    p: torch.Tensor,
    M: nn.Module,
    epsilon: float,
):
    """Perform a single leapfrog step."""
    x.requires_grad_(True)
    U = M(x)
    U.backward()
    grad = x.grad
    grad = grad.clamp(-10, 10)

    # Half step for momentum
    p.data -= epsilon * grad / 2

    # Full step for position
    x.data += epsilon * p

    x.grad.zero_()
    # Full step for momentum
    U = M(x, single=True)
    U.backward()
    grad = x.grad
    grad = grad.clamp(-10, 10)
    p.data -= epsilon * grad / 2

    x.grad.zero_()
    return x, p


def hmc_step(
    x: torch.Tensor,
    M: nn.Module,
    epsilon: float = 1e-3,
    L: int = 10,
):
    """Perform a single HMC step with L leapfrog steps."""
    # Sample momentum from Gaussian
    p = torch.randn_like(x)

    # Copy initial state
    x_init = x.clone()
    p_init = p.clone()

    # Leapfrog integration
    x = x.clone().detach().requires_grad_(True)
    for _ in range(L):
        x, p = leapfrog_step(x, p, M, epsilon)

    # Compute energy at start and end
    x_init.requires_grad_(True)
    U_init = M(x_init)
    K_init = torch.sum(p_init**2) / 2
    H_init = U_init + K_init

    x.requires_grad_(True)
    U_proposed = M(x)
    K_proposed = torch.sum(p**2) / 2
    H_proposed = U_proposed + K_proposed

    # Metropolis acceptance
    delta_H = H_proposed - H_init
    if torch.rand(1).to(delta_H.device) < torch.exp(-delta_H):
        return x, p
    else:
        return x_init, p_init


def hmc_sample_from(
    M: nn.Module,
    x0: torch.Tensor,
    n_step: int = 1_000,
    epsilon: float = 1e-3,
    L: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:

    M.eval()
    x = torch.rand_like(x0)

    for _ in range(n_step):
        x, _ = hmc_step(x, M, epsilon=epsilon, L=L)

    return (x, M(x))
