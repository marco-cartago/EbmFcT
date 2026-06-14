
from typing import Tuple

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


# Langevin dynamics
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
    grad = grad.clamp(-3, 3)

    # Update preconditioner
    precond.mul_(momentum).add_(grad.pow(2), alpha=1-momentum)

    # Preconditioned step
    step = eta * grad / (precond.sqrt() + 1e-8)
    noise = torch.randn_like(x) * torch.sqrt(2 * eta /(beta * precond.sqrt() + 1e-8))
    x.data = x - step + noise

    x.grad.zero_()
    return x, precond


def langevin_step(
    x: torch.Tensor,
    M: nn.Module,
    eta: float = 1e-3,
    beta: float = 1.0,
):


    return x


def langevin_sample_from(
    M: nn.Module,
    x0: torch.Tensor,
    n_step: int = 1_000,
    eta: float = 1e-3,
    beta: float = 1.0,
    momentum: float = 0.9,
    use_precond: bool = False
):
    M.eval()
    x = x0
    x0.requires_grad_(True)

    if use_precond:
        precond = torch.ones_like(x)
        for _ in range(n_step):
            x, precond = langevin_step_precond(
                x, M, precond, eta=eta, beta=beta, momentum=momentum)

    else:
        x = x0
        noise = torch.empty_like(x0)

        for _ in range(n_step):
            U = M(x)
            U.backward()
            grad = x.grad
            grad.clamp_(-0.03, 0.03)

            if grad is None:
                raise ValueError("None gradient")

            etaT = torch.tensor((eta,))
            betaT = torch.tensor((beta,))

            # Preconditioned step
            noise.normal_(1)
            noise *= torch.sqrt(2 * etaT /(betaT + 1e-8))
            step = etaT * grad
            
            x.data = x - step + noise
            x.data.clamp_(0, 1) # Keep pixels in [0,1]
            
            x.grad.zero_()


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

# Replay sample buffer -> really interesting
# Idea found on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
# -------------------------------------------------------------------------------------- #

class Sampler:

    def __init__(self, model, img_shape, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(self.model.device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs