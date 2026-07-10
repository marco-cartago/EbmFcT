# Energy Based Density Factorization

## Project Description

A project in which we try to work our way to a "factorization" of a probability density over a set of signals, in this setting, images. ($x \in \mathbb{R}^n$)

$$p(x) = \frac{1}{Z} \Psi _0(x) \Psi _1 (x) \cdots \Psi_k(x)$$

Where heach density assumes the boltzmann form: $\frac{1}{Z_i} e^{\Psi_i(x)}$

Each ${\Psi_k}$ rapresnts an unnomalized probaiblity density, and each component is pushed, ideally, trough training to model different features of the input. The goal of this project would consequently be to reach trough training a factorization given by what we call "Energy Heads" (EH) each one modelling a different aspect of the data generating distribution.

![Generation](https://github.com/marco-cartago/EbmFcT/blob/main/images/FMNIST/Generation_new_model_boot.gif)

## Structure of the repo

- `/src` All the machinery needed for the model to run. This includes samplers model definition and data imports...
    - `data_import.py`: Import and preprocessing functions for all the dataset used **MNIST**, **LFW** and **FMNIST**.
    - `diagnostic.py`: Functions for measuring the quality of the generation and of the "energy landscape" produced by the model.
    - `evaluation.py`: Calculation of the Freched Inception Distance (FID).
    - `gradient_inspect.py`: Functions from sampling with langevin dynamics from the EBM
    - `information.py`: Initial implementation of a MINE-like tecnique for estimating the total correlation of the heads.
    - `losses.py`: Loss functions, later used in `train`.
    - `MINE.py`: Definitive implementation of the MINE inspired head regularizer for the whole model: Functions for computing lower bounds to the total correlation.
    - `model.py`: Definition of modules and submodules used to implement the EBM.
    - `plot.py`: Mainly functions for inspecting the gradients of the heads and for plotting grids of samples either from the `ReplaySampler` or freshly generated from the model itself.
    - `sampler.py`: Home to the `ReplaySampler` class, massively important for the training dynamics of presistent contrastive divergence. 
    - `train.py`: Functions for performing a single epoch of training.

- `model_train.ipynb`: Setup for a general train run for all of the three main datasets with checpoints in the `/models/` folder.
- `other_models_benchmark.ipynb`: Execution of the benchmarks for the other models used for comparison, a variation auto encoder and u-net diffusion model.
- `train_other_models.py`: File for training the benchmark models (VAE & U-net diffusion).
- `test_total_correlation.ipynb`: Training of the model with the total correlation regularization defined in `MINE.py`.
- `toy_example.ipynb`: Experimentation of the TC regularization on a 1d density.

## Build instructions (Linux)

For building the code of the project first build a local enviroment and install the *requirements*:
```bash
python3 -m venv venv 
source ./venv/bin/activate
pip3 install -r requirements.txt
```


## Architecture:
At this point we are using a Deep Convolutional Network for every EH and the output of the entire model it's just a sum over those results.
Things to investigate:
 - possibility of logsum(output) to make the energy non-negative. This can improve the convergence of the model. In any case in the literature about EBM it's shown how this doesn't improve performances and leads to a collapse around zero value
 - possibility to differentiate the architecture for every EH to implicitly bias the distribution towards different levels of details or different kind of aspects of data.

## Loss:
At the moment we are working on a contrastive learning framework, pushing the model to learn high (not sure) energy for good samples and low energy for the ones OOD.
The current loss for the model is formalized over a batch $\mathcal{B}_S$ of generated samples and a batch of training data $\mathcal{B}_D$ as:

```math
\mathcal{L}(\theta) = \mathcal{L}_{CD} + \lambda\,\mathcal{L}_{reg} + \gamma\,\mathcal{L}_{corr}
```

where

```math
\mathcal{L}_{CD} = \frac{1}{|\mathcal{B}_D|} \sum_{x\in\mathcal{B}_D} f_\theta(x) - \frac{1}{|\mathcal{B}_S|} \sum_{x\in\mathcal{B}_S} f_\theta(x)
```

```math
\mathcal{L}_{reg}= \frac{1}{|\mathcal{B}_D|} \sum_{x\in\mathcal{B}_D} f_\theta(x)^2 + \frac{1}{|\mathcal{B}_S|} \sum_{x\in\mathcal{B}_S} f_\theta(x)^2
```

```math
\mathcal{L}_{corr} = \mathcal{C}(h_D) + \mathcal{C}(h_S)
```

If we treat $h$ as a vector function outputting a vector with the head outputs the penalization on the correlation. The approaches we tried were respectively:
- Pearson Correlation
- Pixel-to-pixel linear correlation of the gradients
- Total correlation (what we sticked with)

### Total Correlation
A possible generalization of mutual information, the difference between the joint entropy of the vector:  $\mathrm{TC}(x) = -\mathrm{H}(x) + \sum_{i=0}^k \mathrm{H}(x_i)$. Where $\mathrm{H}(x)$ is the joint entropy of the vector $x$ and $\mathrm{H}(x_i)$ is the single marginal entropy for the i-th vector component. The way we estimate it starts from the following equality:

$$\mathrm{TC}(x) = \mathrm{D_{KL}}[ p(x_1, x_2, \ldots x_n) \| p(x_1)p(x_2) \cdots p(x_n) ]$$

being a KL divergence it can be expressed using the Donsker-Vardan theorem, in an alternative form:

$$ \mathrm{TC}(x) = \sup_{f:X\rightarrow\mathbb{R}} \left[ \mathbb{E}_{x \sim p(x_1, \ldots x_n)}[f(x)] - \mathbb{E}_{x \sim p(x_1) \cdots p(x_n)}  \left[e^{f(x)}\right] \right] $$

We then replace the supremum with a maximization over a set of parametrized functions $f_\theta$ (${\theta \in \Theta}$) that is performed with gradient ascent together with the training of the EBM.

## Sampling:
The sampling strategy for this kind of project can is a combination of Persistent Contrastive Divergence an Stochastic Gradient Langevin Sampling, a MCMC method that exploit the gradient of the enrgy function to move the current sample towards one with higher energy and adding a gaussian noise factor to favor exploration. The PCD influence is on the fact that we still maintain a persistent chain so we don't initialize at random for every single point but we start from the last point of the previous chain and producing a sample by performing gradient ascent.

Key problems with this approach:
 - There is the possibility that the sampler becomes "too good" in finding the lowest points in the energy landscape and produce samples in that direction instead of the one truly modeled. In that case the model is learning to approximate a wrong lanscape.
 - Computational cost given by MCMC

## Interesting future directions
- Try with logsum output
- Scheduler on langevin steps
- Langevin sampling on subset of pixel

