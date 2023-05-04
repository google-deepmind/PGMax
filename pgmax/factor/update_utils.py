# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils function for message updates."""

import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from pgmax.utils import NEG_INF


@functools.partial(jax.jit, static_argnames="num_labels")
def get_maxes_and_argmaxes(
    data: jnp.array,
    labels: jnp.array,
    num_labels: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Given a flattened sequence of elements and their corresponding labels, returns the maxes and argmaxes of each label.

  Args:
    data: Array of shape (a_len,) where a_len is an arbitrary integer.
    labels: Label array of shape (a_len,), assigning a label to each entry.
      Labels must be 0,..., num_labels - 1.
    num_labels: Number of different labels.

  Returns:
    Maxes and argmaxes arrays

  Note:
    To leverage the restricted list of primitives supported by JAX ndarray.at
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    we represent the argmax of an array x of length N and max x^* by
    argmax x = max_k {k - N * 1(x[k] = x^*)}
  """
  num_obs = data.shape[0]

  maxes = jnp.full(shape=(num_labels,), fill_value=NEG_INF).at[labels].max(data)
  only_maxes_pos = jnp.arange(num_obs) - num_obs * jnp.where(
      data != maxes[labels], 1, 0
  )

  argmaxes = (
      jnp.full(
          shape=(num_labels,),
          fill_value=jnp.iinfo(jnp.int32).min,
          dtype=jnp.int32,
      )
      .at[labels]
      .max(only_maxes_pos)
  )
  return maxes, argmaxes


@functools.partial(jax.jit, static_argnames=("num_labels"))
def logsumexps_with_temp(
    data: jnp.array,
    labels: jnp.array,
    num_labels: int,
    temperature: float,
    maxes: Optional[jnp.array] = None,
) -> jnp.ndarray:
  """Given a flattened sequence of elements and their corresponding labels, returns the stable logsumexp for each label at the given temperature.

  Args:
    data: Array of shape (a_len,) where a_len is an arbitrary integer.
    labels: Label array of shape (a_len,), assigning a label to each entry.
      Labels must be 0,..., num_labels - 1.
    num_labels: Number of different labels.
    temperature: Temperature for loopy belief propagation.
    maxes: Optional array of precomputed maxes per label

  Returns:
    The array of logsumexp for each label
  """
  if maxes is None:
    maxes = (
        jnp.full(shape=(num_labels,), fill_value=NEG_INF).at[labels].max(data)
    )

  logsumexps_wo_maxes = temperature * jnp.log(
      jnp.zeros((num_labels,))
      .at[labels]
      .add(jnp.exp((data - maxes[labels]) / temperature))
  )
  return logsumexps_wo_maxes + maxes


@functools.partial(jax.jit, static_argnames=("num_labels"))
def softmax_and_logsumexps_with_temp(
    data: jnp.array,
    labels: jnp.array,
    num_labels: int,
    temperature: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Given a flattened sequence of elements and their corresponding labels, returns for each label the stable (1) softmax and (2) logsumexp at the given temperature.

  Args:
    data: Array of shape (a_len,) where a_len is an arbitrary integer.
    labels: Label array of shape (a_len,), assigning a label to each entry.
      Labels must be 0,..., num_labels - 1.
    num_labels: Number of different labels.
    temperature: Temperature for loopy belief propagation.

  Returns:
    softmax_data: Stable softmax with per-label normalization at the given
      temperature.
    logsumexp_data: Stable logsumexp for each label at the given temperature.
  """
  maxes = (
      jnp.full(shape=(num_labels,), fill_value=NEG_INF).at[labels].max(data)
  )
  exp_data = temperature * jnp.exp((data - maxes[labels]) / temperature)
  sumexp_data = (
      jnp.full(shape=(num_labels,), fill_value=0.0).at[labels].add(exp_data)
  )
  logsumexp_data = maxes + temperature * jnp.log(sumexp_data / temperature)
  softmax_data = exp_data / sumexp_data[labels]
  return softmax_data, logsumexp_data


@jax.jit
def log1mexp(x):
  """Returns a stable implementation of f(x) = log(1 - exp(-x)) for x >= 0.

  See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

  Args:
    x: input
  """
  return jnp.where(
      x <= jnp.log(2),
      jnp.log(-jnp.expm1(-x)),
      jnp.log1p(-jnp.exp(-x))
  )


@jax.jit
def logaddexp_with_temp(
    data1: jnp.array,
    data2: jnp.array,
    temperature: float,
) -> jnp.ndarray:
  """Returns the stable logaddexp of two arrays of same length at a given temperature, ie T * log(exp(data1/T) + exp(data2/T)).

  Args:
    data1: Array of shape (a_len,) where a_len is an arbitrary integer.
    data2: Array of same shape (a_len,)
    temperature: Temperature for loopy belief propagation.
  """
  maxes = jnp.maximum(data1, data2)
  mins = jnp.minimum(data1, data2)
  logsumexps_wo_maxes = jnp.log1p(jnp.exp((mins - maxes) / temperature))
  return temperature * logsumexps_wo_maxes + maxes


@jax.jit
def logminusexp_with_temp(
    data1: jnp.array,
    data2: jnp.array,
    temperature: float,
    eps=1e-30
) -> jnp.ndarray:
  """Returns a stable version of T * log(exp(data1/T) - exp(data2/T)) where data1 >= data2 entry-wise.

  Args:
    data1: Array of shape (a_len,) where a_len is an arbitrary integer.
    data2: Array of same shape (a_len,)
    temperature: Temperature for loopy belief propagation.
    eps: If the difference between the entries is lower than this threshold,
      we set the result to NEG_INF
  """
  return jnp.where(
      data1 >= data2 + eps,
      temperature * log1mexp((data1 - data2) / temperature)
      + data1,
      NEG_INF,
  )
