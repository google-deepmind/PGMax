[![continuous-integration](https://github.com/deepmind/PGMax/actions/workflows/ci.yaml/badge.svg)](https://github.com/deepmind/PGMax/actions/workflows/ci.yaml)
[![PyPI version](https://badge.fury.io/py/pgmax.svg)](https://badge.fury.io/py/pgmax)
[![Documentation Status](https://readthedocs.org/projects/pgmax/badge/?version=latest)](https://pgmax.readthedocs.io/en/latest/?badge=latest)

# PGMax

PGMax implements general [factor graphs](https://en.wikipedia.org/wiki/Factor_graph)
for discrete probabilistic graphical models (PGMs), and
hardware-accelerated differentiable [loopy belief propagation (LBP)](https://en.wikipedia.org/wiki/Belief_propagation)
in [JAX](https://jax.readthedocs.io/en/latest/).

- **General factor graphs**: PGMax supports easy specification of general
factor graphs with potentially complicated topology, factor definitions,
and discrete variables with a varying number of states.
- **LBP in JAX**: PGMax generates pure JAX functions implementing LBP for a
given factor graph. The generated pure JAX functions run on modern accelerators
(GPU/TPU), work with JAX transformations
(e.g. `vmap` for processing batches of models/samples,
`grad` for differentiating through the LBP iterative process),
and can be easily used as part of a larger end-to-end differentiable system.

See our [companion paper](https://arxiv.org/abs/2202.04110) for more details.

PGMax is under active development. APIs may change without notice,
and expect rough edges!

[**Installation**](#installation)
| [**Getting started**](#getting-started)

## Installation

### Install from PyPI
```
pip install pgmax
```

### Install latest version from GitHub
```
pip install git+https://github.com/deepmind/PGMax.git
```

### Developer
While you can install PGMax in your standard python environment,
we *strongly* recommend using a
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
to manage your dependencies. This should help to avoid version conflicts and
just generally make the installation process easier.

```
git clone https://github.com/deepmind/PGMax.git
cd PGMax
python3 -m venv pgmax_env
source pgmax_env/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
python3 setup.py develop
```

### Install on GPU

By default the above commands install JAX for CPU. If you have access to a GPU, 
follow the official instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda)
to install JAX for GPU.

## Getting Started


Here are a few self-contained Colab notebooks to help you get started on using PGMax:

- [Tutorial on basic PGMax usage](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/rbm.ipynb)
- [LBP inference on Ising model](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/ising_model.ipynb)
- [Implementing max-product LBP](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/rcn.ipynb)
for [Recursive Cortical Networks](https://www.science.org/doi/10.1126/science.aag2612)
- [End-to-end differentiable LBP for gradient-based PGM training](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/gmrf.ipynb)
- [2D binary deconvolution](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/pmp_binary_deconvolution.ipynb)
- [Alternative inference with Smooth Dual LP-MAP](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/sdlp_examples.ipynb)

## Citing PGMax

PGMax is part of the DeepMind JAX ecosystem. If you use PGMax in your work, please consider citing our [companion paper](https://arxiv.org/abs/2202.04110)
```
@article{zhou2022pgmax,
  author = {Zhou, Guangyao and Dedieu, Antoine and Kumar, Nishanth and L{\'a}zaro-Gredilla, Miguel and Kushagra, Shrinu and George, Dileep},
  title = {{PGMax: Factor Graphs for Discrete Probabilistic Graphical Models and Loopy Belief Propagation in JAX}},
  journal = {arXiv preprint arXiv:2202.04110},
  year={2022}
}
```
and using the DeepMind JAX Ecosystem citation
```
bibtex
@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/deepmind},
  year = {2020},
}
```


## Note

This is not an officially supported Google product.
