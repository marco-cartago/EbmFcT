import torch

def synthetize_image(
    model,
    n_images=1,
    steps=500,
    step_size=10.0,
    noise_std=0.005,          # metti 0.0 per gradient descent puro
    init="uniform",           # "uniform" | "gaussian" | tensor iniziale
    clamp=(-1.0, 1.0),
    device="cpu",
    img_dimension = 28):
    """
    Finds imgs with low energy from a modeled distribution.
    
    Return:
        images  : tensor [n_images, 1, img_dimension, img_dimension] with generated imgs
        energies: mean energies for every step
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


    if isinstance(init, torch.Tensor):
        x = init.clone().to(device)
    elif init == "uniform":
        x = torch.zeros(n_images, 1, img_dimension, img_dimension, device=device).uniform_(*clamp)
    elif init == "gaussian":
        x = torch.randn(n_images, 1, img_dimension, img_dimension, device=device).clamp(*clamp)
    else:
        raise ValueError(f"init non valido: {init}")

    energies = []

    for step in range(steps):
        x = x.detach().requires_grad_(True)

        energy, _ = model(x)               # [n_images, 1]
        energy_sum = energy.sum()
        energy_sum.backward()

        with torch.no_grad():
            grad = x.grad
            x = x - step_size * grad      # GD
            if noise_std > 0:
                x = x + noise_std * torch.randn_like(x)
            x = x.clamp(*clamp)

        energies.append(energy.mean().item())

    for p in model.parameters():
        p.requires_grad_(True)

    return x.detach(), energies

def synthetize_image_from_head(
    model_head,
    n_images=1,
    steps=500,
    step_size=10.0,
    noise_std=0.005,          # 0.0 for GD pure
    init="uniform",           # "uniform" | "gaussian" | tensor iniziale
    clamp=(-1.0, 1.0),
    device="cpu",
    img_dimension = 28):
    """
    Finds images with low energy from a modeled distribution by a head
    
    Return:
        images  : tensor [n_images, 1, img_dimension, img_dimension] with generated imgs
        energies: mean energies for every step
    """
    model_head.eval()
    for p in model_head.parameters():
        p.requires_grad_(False)

    # Inizializzazione
    if isinstance(init, torch.Tensor):
        x = init.clone().to(device)
    elif init == "uniform":
        x = torch.zeros(n_images, 1, img_dimension, img_dimension, device=device).uniform_(*clamp)
    elif init == "gaussian":
        x = torch.randn(n_images, 1, img_dimension, img_dimension, device=device).clamp(*clamp)
    else:
        raise ValueError(f"init non valido: {init}")

    energies = []

    for step in range(steps):
        x = x.detach().requires_grad_(True)

        energy = model_head(x)               # [n_images, 1]
        energy_sum = energy.sum()
        energy_sum.backward()

        with torch.no_grad():
            grad = x.grad
            x = x - step_size * grad
            if noise_std > 0:
                x = x + noise_std * torch.randn_like(x)
            x = x.clamp(*clamp)

        energies.append(energy.mean().item())

    for p in model_head.parameters():
        p.requires_grad_(True)

    return x.detach(), energies