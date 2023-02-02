[![continuous-integration](https://github.com/deepmind/PGMax/actions/workflows/ci.yaml/badge.svg)](https://github.com/deepmind/PGMax/actions/workflows/ci.yaml)
[![PyPI version](https://badge.fury.io/py/pgmax.svg)](https://badge.fury.io/py/pgmax)
[![codecov](https://codecov.io/gh/deepmind/PGMax/branch/master/graph/badge.svg?token=FrRlTDCFjk)](https://codecov.io/gh/deepmind/PGMax)
[![Documentation Status](https://readthedocs.org/projects/PGMax/badge/?version=latest)](https://pgmax.readthedocs.io/en/latest/?badge=latest)

# PGMax

PGMax implements general [factor graphs](https://en.wikipedia.org/wiki/Factor_graph)
for discrete probabilistic graphical models (PGMs), and
hardware-accelerated differentiable [loopy belief propagation (LBP)]
(https://en.wikipedia.org/wiki/Belief_propagation)
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
- [Implementing max-product LBP](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/rcn.ipynb)
for [Recursive Cortical Networks](https://www.science.org/doi/10.1126/science.aag2612)
- [End-to-end differentiable LBP for gradient-based PGM training](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/gmrf.ipynb)
- [2D binary deconvolution](https://colab.research.google.com/github/deepmind/PGMax/blob/master/examples/pmp_binary_deconvolution.ipynb)

## Citing PGMax

Please consider citing our [companion paper](https://arxiv.org/abs/2202.04110) if you use PGMax in your work:

```
@article{zhou2022pgmax,
  author = {Zhou, Guangyao and Dedieu, Antoine and Kumar, Nishanth and L{\'a}zaro-Gredilla, Miguel and Kushagra, Shrinu and George, Dileep},
  title = {{PGMax: Factor Graphs for Discrete Probabilistic Graphical Models and Loopy Belief Propagation in JAX}},
  journal = {arXiv preprint arXiv:2202.04110},
  year={2022}
}
```

## Note

This is not an officially supported Google product.
