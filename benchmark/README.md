# Benchmarking PGMax

This directory contains code to reproduce the results on benchmarking PGMax against [pomegranate](https://github.com/jmschrei/pomegranate) and [pgmpy](https://github.com/pgmpy/pgmpy) in Section 5.1 of the [PGMax companion paper](https://arxiv.org/abs/2202.04110).

## Running all the experiments

Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), and use

```
mamba env create -f environment.yml
mamba activate pgmax_jmlr
pip install jax==0.4.25 jaxlib==0.4.25+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

to install the necessary dependencies.

Run 

```
XLA_PYTHON_CLIENT_ALLOCATOR=platform python rbm.py
```

to generate all inference results (excluding PGMax GPU results).

Update `jax_platform_name = 'cpu'` in `rbm.py` to `jax_platform_name = 'gpu'` and re-run

```
XLA_PYTHON_CLIENT_ALLOCATOR=platform python rbm.py
```

to additionally generate PGMax GPU inference results.

Use `results_analysis.ipynb` to reproduce the analysis in Section 5.1.

The code is tested on Debian 11.0 with CUDA 11.8. For other CUDA versions please update the JAX and PyTorch dependencies accordingly. The results were generated using a machine with a single V100.

## Reproducing paper results using pre-computed inference results

For convenience, we included the inference results used to obtain the numbers and figures in the paper under the `precomputed_results` directory. Use `results_analysis.ipynb` to reproduce the numbers and figures reported in the paper.
