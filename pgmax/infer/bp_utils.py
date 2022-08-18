# pyformat: mode=midnight
# ==============================================================================
# Copyright 2022 Intrinsic Innovation LLC.
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
# ==============================================================================
"""A module containing helper functions used for belief propagation."""

import functools

import jax
import jax.numpy as jnp

from pgmax.utils import NEG_INF


@functools.partial(jax.jit, static_argnames="max_segment_length")
def segment_max_opt(
    data: jnp.ndarray, segments_lengths: jnp.ndarray, max_segment_length: int
) -> jnp.ndarray:
  """Computes the max of every segment of data, where segments_lengths specifies the segments.

  Args:
    data: Array of shape (a_len,) where a_len is an arbitrary integer.
    segments_lengths: Array of shape (num_segments,) where 0 < num_segments <=
      a_len. segments_lengths.sum() should yield a_len, and all elements must be
      > 0.
    max_segment_length: The maximum value in segments_lengths.

  Returns:
    An array of shape (num_segments,) that contains the maximum value
      from data of every segment specified by segments_lengths
  """

  @functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
  def get_max(data, start_index, segment_length):
    return jnp.max(
        jnp.where(
            jnp.arange(max_segment_length) < segment_length,
            jax.lax.dynamic_slice(
                data, jnp.array([start_index]), [max_segment_length]
            ),
            NEG_INF,
        )
    )

  start_indices = jnp.concatenate(
      [
          jnp.full(shape=(1,), fill_value=int(NEG_INF), dtype=int),
          jnp.cumsum(segments_lengths),
      ]
  )[:-1]
  expanded_data = jnp.concatenate([data, jnp.zeros(max_segment_length)])
  return get_max(expanded_data, start_indices, segments_lengths)
