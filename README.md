# EbmFcT

## Project Description

A little project (a test) in which we try to factorize a probability density over a set of signals, in this setting, images.

$$p(\underline{x}) = \frac{1}{Z} \Psi _0(\underline{x}) \Psi _1 (\underline{x}) \cdots \Psi_k({\underline{x}})$$

where each ${\Psi_k}$ is an unnomalized probaiblity density, and each component is pushed, trough training to be as indipendent as possibile fom the others.

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

TODO:
 -[] Try with logsum output
 -[] Scheduler on langevin steps
 -[] Langevin sampling on subset of pixel

