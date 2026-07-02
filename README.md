# EbmFcT

## Project Description

A little project (a test) in which we try to factorize a probability density over a set of signals, in this setting, images.

$$p(\underline{x}) = \frac{1}{Z} \Psi _0(\underline{x}) \Psi _1 (\underline{x}) \cdots \Psi_k({\underline{x}})$$

where each ${\Psi_k}$ is an unnomalized probaiblity density, and each component is pushed, trough training to be as indipendent as possibile from the others. The goal of this kind of project is to have a factorization given by what we call "Energy Heads" (EH) each one modelling a different aspect of the data generating distribution.


![Generation](https://github.com/marco-cartago/EbmFcT/blob/main/images/FMNIST/Generation_new_model_boot.gif)
## Build instructions (Linux)

For building the code of the project first build a local enviroment and install the *requirements*:
```bash
python3 -m venv venv 
source ./venv/bin/activate
pip3 install -r requirements.txt
```

Then just run the script:

```bash
python3 main.py
```

At this point in time the project is more a QR code generator than an image generator, even so, its interesting to work on.

---
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
Things to investigate:
- pass
  
## Sampling:
The sampling strategy for this kind of project can is a combination of Persistent Contrastive Divergence an Stochastic Gradient Langevin Sampling, a MCMC method that exploit the gradient of the enrgy function tu move the current sample towards one with higher energy and adding a gaussian noise factor to favor exploration. The PCD influence is on the fact that we still maintain a persistent chain so we don't initialize at random for every single point but we start from the last point of the previous chain and producing a sample by performing gradient ascent.
Key problems with this approach:
 - There is the possibility that the sampler becomes "too good" in finding the highest points in the energy landscape and produce samples in that direction instead of the one truly modeled. In that case the model is learning to approximate a wrong lanscape.
 - Computational cost given by MCMC

## Benchmarks:
- Human looking at picture and saying *"mmmmmmmmmmmmmmmmmmmmmm, for me impossible"* (Ref: https://youtu.be/ilfmlWVRAgQ?si=9I_bAYY09utl8eJa)
- Image generation from the FashionMnist dataset

## Correlation problem
We need to make the different EHs to learn more of SEMANTIC uncorrelations instead of numerical ones.
Tried:
- Penalizing the correlation between outputs of heads -> the model learn to change just magnitudes or sign of the energy
- Penalizing the gradient of the heads on the imput -> the heads look at the same area of the image but with different pixels

To try:
- Penalize KL divergence using the energies given by every head of the batch as distributions (not sure will work because it's always a numerical than Semantic)
- For me it mus be something about the gradients of every EH


# TODO:
- Try with logsum output
- Scheduler on langevin steps
- Langevin sampling on subset of pixel

