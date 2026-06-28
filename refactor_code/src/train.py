from src.losses import cd_loss, cd_loss_with_tc
import torch
import tqdm

def train_one_epoch(model, sampler, train_loader, optimizer, sample_steps, sample_step_size, sample_noise_std, energy_reg, corr_param, device="cpu"):

    model.train()
    running_loss = 0.0
    running_cd = 0.0
    running_reg = 0.0
    running_corr = 0.0
    running_e_real = 0.0
    running_e_fake = 0.0

    for x_real, _ in tqdm.tqdm(train_loader):

        x_real = x_real.to(device)

        x_neg = sampler.sample(batch_size=x_real.size(0), steps=sample_steps, step_size=sample_step_size, noise_std=sample_noise_std)

        e_fake, _ = model(x_neg)
        e_real, _ = model(x_real)

        loss, cd, reg, corr = cd_loss(
            model=model,
            x_fake=x_neg,
            x_real=x_real,
            energy_regularization=energy_reg,
            corr_param=corr_param,
            return_components=True
        )

        running_loss  += loss.item()
        running_cd    += cd.item()
        running_reg   += reg.item()
        running_corr  += corr.item()
        running_e_real += e_real.mean().item()
        running_e_fake += e_fake.mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    n = len(train_loader)
    print(f"  CD:     {running_cd    / n:.4f}")
    print(f"  Reg:    {running_reg   / n:.4f}")
    print(f"  Corr:   {running_corr  / n:.4f}")
    print(f"  E_real: {running_e_real / n:.4f}")
    print(f"  E_fake: {running_e_fake / n:.4f}")
    print(f"  Gap:    {(running_e_real - running_e_fake) / n:.4f}")

    return running_loss / n


def train_one_epoch_TC(
        model, 
        tc_estimator, 
        sampler, 
        train_loader, 
        optimizer, 
        sample_steps, 
        sample_step_size, 
        sample_noise_std, 
        energy_reg, 
        tc_reg, 
        device="cpu"
    ):

    model.train()
    running_loss = 0.0
    running_cd = 0.0
    running_reg = 0.0
    running_tc = 0.0
    running_e_real = 0.0
    running_e_fake = 0.0

    for x_real, _ in tqdm.tqdm(train_loader):

        x_real = x_real.to(device)

        x_neg = sampler.sample(batch_size=x_real.size(0), steps=sample_steps, step_size=sample_step_size, noise_std=sample_noise_std)

        e_fake, _ = model(x_neg)
        e_real, _ = model(x_real)
        mod_heads = model.head_outputs

        loss, cd, reg, tc = cd_loss_with_tc(
            model=model,
            tc_estimator=tc_estimator,
            x_fake=x_neg,
            x_real=x_real,
            energy_regularization=energy_reg,
            tc_regularizations=tc_reg,
            return_components=True
        )

        running_loss   += loss.item()
        running_cd     += cd.item()
        running_reg    += reg.item()
        running_tc     += tc.item()
        running_e_real += e_real.mean().item()
        running_e_fake += e_fake.mean().item()

        # Update the model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the total correlation estimator
        tc_estimator.train_step(mod_heads)

    n = len(train_loader)
    print(f"  CD:     {running_cd    / n:.4f}")
    print(f"  Reg:    {running_reg   / n:.4f}")
    print(f"  TC:     {running_tc  / n:.4f}")
    print(f"  E_real: {running_e_real / n:.4f}")
    print(f"  E_fake: {running_e_fake / n:.4f}")
    print(f"  Gap:    {(running_e_real - running_e_fake) / n:.4f}")

    return running_loss / n


def train_one_epoch_dSprites(model, sampler, train_loader, optimizer, sample_steps, sample_step_size, sample_noise_std, energy_reg, corr_param, device="cpu"):

    model.train()
    running_loss = 0.0
    running_cd = 0.0
    running_reg = 0.0
    running_corr = 0.0
    running_e_real = 0.0
    running_e_fake = 0.0

    for batch in tqdm.tqdm(train_loader):
        x_real = batch["image"]
        x_real = x_real.to(device)

        x_neg = sampler.sample(batch_size=x_real.size(0), steps=sample_steps, step_size=sample_step_size, noise_std=sample_noise_std)

        e_fake, _ = model(x_neg)
        e_real, _ = model(x_real)

        loss, cd, reg, corr = cd_loss(
            model=model,
            x_fake=x_neg,
            x_real=x_real,
            energy_regularization=energy_reg,
            corr_param=corr_param,
            return_components=True
        )

        running_loss  += loss.item()
        running_cd    += cd.item()
        running_reg   += reg.item()
        running_corr  += corr.item()
        running_e_real += e_real.mean().item()
        running_e_fake += e_fake.mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    n = len(train_loader)
    print(f"  CD:     {running_cd    / n:.4f}")
    print(f"  Reg:    {running_reg   / n:.4f}")
    print(f"  Corr:   {running_corr  / n:.4f}")
    print(f"  E_real: {running_e_real / n:.4f}")
    print(f"  E_fake: {running_e_fake / n:.4f}")
    print(f"  Gap:    {(running_e_real - running_e_fake) / n:.4f}")

    return running_loss / n
