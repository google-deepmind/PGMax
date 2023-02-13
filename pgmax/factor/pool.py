# Copyright 2022 DeepMind Technologies Limited.
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

"""Defines a pool factor."""

import dataclasses
import functools
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from pgmax.factor import factor
from pgmax.factor import logical
from pgmax.utils import NEG_INF


# pylint: disable=unexpected-keyword-arg
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class PoolWiring(factor.Wiring):
  """Wiring for PoolFactors.

  Attributes:
    pool_choices_edge_states: Array of shape (num_pool_choices, 2)
      pool_choices_edge_states[ii, 0] contains the global PoolFactor index
      pool_choices_edge_states[ii, 1] contains the message index of the pool
      choice variable's state 0. The message index of the pool choice variable's
      state 1 is pool_choices_edge_states[ii, 1] + 1

    pool_indicators_edge_states: Array of shape (num_pool_factors,)
      pool_indicators_edge_states[ii] contains the message index of the pool
      indicator variable's state 0, which takes into account all the PoolFactors
      of the FactorGraph. The message index of the pool indicator variable's
      state 1 is pool_indicators_edge_states[ii, 1] + 1

  Raises:
    ValueError: If:
      (1) The are no num_pool_factors different factor indices
      (2) There is a factor index higher than num_pool_factors - 1
  """

  pool_choices_edge_states: Union[np.ndarray, jnp.ndarray]
  pool_indicators_edge_states: Union[np.ndarray, jnp.ndarray]

  def __post_init__(self):
    super().__post_init__()

    if self.pool_choices_edge_states.shape[0] > 0:
      pool_factor_indices = self.pool_choices_edge_states[:, 0]
      num_pool_factors = self.pool_indicators_edge_states.shape[0]

      if np.unique(pool_factor_indices).shape[0] != num_pool_factors:
        raise ValueError(
            f"The PoolWiring must have {num_pool_factors} different"
            " PoolFactor indices"
        )

      if pool_factor_indices.max() >= num_pool_factors:
        raise ValueError(
            f"The highest PoolFactor index must be {num_pool_factors - 1}"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PoolFactor(factor.Factor):
  """A Pool factor of the form (pc1, ...,pcn, pi) where (pc1,...,pcn) are the pool choices and pi is the pool indicator.

  A Pool factor is defined as:
  F(pc1, ...,pcn, pi) = 0 <=> (pc1=...=pcn=pi=0) OR (pi=1 AND pc1 +...+ pcn=1)
  F(pc1, ...,pcn, pi) = -inf o.w.

  i.e. either (a) all the variables are set to 0, or (b) the pool indicator
  variable is set to 1 and exactly one of the pool choices variables is set to 1

  Note: placing the pool indicator at the end allows us to reuse our
  existing infrastucture for wiring logical factors
  """

  log_potentials: np.ndarray = dataclasses.field(
      init=False,
      default=np.empty((0,)),
  )

  def __post_init__(self):
    if len(self.variables) < 2:
      raise ValueError(
          "A PoolFactor requires at least one pool choice and one pool "
          "indicator."
      )

    if not np.all([variable[1] == 2 for variable in self.variables]):
      raise ValueError("All the variables in a PoolFactor should all be binary")

  @staticmethod
  def concatenate_wirings(wirings: Sequence[PoolWiring]) -> PoolWiring:
    """Concatenate a list of PoolWirings.

    Args:
      wirings: A list of PoolWirings

    Returns:
      Concatenated PoolWiring
    """
    if not wirings:
      return PoolWiring(
          edges_num_states=np.empty((0,), dtype=int),
          var_states_for_edges=np.empty((0,), dtype=int),
          pool_choices_edge_states=np.empty((0, 2), dtype=int),
          pool_indicators_edge_states=np.empty((0,), dtype=int),
      )

    logical_wirings = []
    for wiring in wirings:
      logical_wiring = logical.LogicalWiring(
          edges_num_states=wiring.edges_num_states,
          var_states_for_edges=wiring.var_states_for_edges,
          parents_edge_states=wiring.pool_choices_edge_states,
          children_edge_states=wiring.pool_indicators_edge_states,
          edge_states_offset=1,
      )
      logical_wirings.append(logical_wiring)

    logical_wiring = logical.LogicalFactor.concatenate_wirings(logical_wirings)

    return PoolWiring(
        edges_num_states=logical_wiring.edges_num_states,
        var_states_for_edges=logical_wiring.var_states_for_edges,
        pool_choices_edge_states=logical_wiring.parents_edge_states,
        pool_indicators_edge_states=logical_wiring.children_edge_states,
    )

  # pylint: disable=g-doc-args
  @staticmethod
  def compile_wiring(
      factor_edges_num_states: np.ndarray,
      variables_for_factors: Sequence[List[Tuple[int, int]]],
      factor_sizes: np.ndarray,
      vars_to_starts: Mapping[Tuple[int, int], int],
  ) -> PoolWiring:
    """Compile a PoolWiring for a PoolFactor or for a FactorGroup with PoolFactors.

    Internally uses the logical factor compile_wiring.

    Args: See LogicalFactor.compile_wiring docstring.

    Returns:
      The PoolWiring
    """
    logical_wiring = logical.LogicalFactor.compile_wiring(
        factor_edges_num_states=factor_edges_num_states,
        variables_for_factors=variables_for_factors,
        factor_sizes=factor_sizes,
        vars_to_starts=vars_to_starts,
        edge_states_offset=1,
    )

    return PoolWiring(
        edges_num_states=logical_wiring.edges_num_states,
        var_states_for_edges=logical_wiring.var_states_for_edges,
        pool_choices_edge_states=logical_wiring.parents_edge_states,
        pool_indicators_edge_states=logical_wiring.children_edge_states,
    )


# pylint: disable=unused-argument
@functools.partial(jax.jit, static_argnames="temperature")
def pass_pool_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    pool_choices_edge_states: jnp.ndarray,
    pool_indicators_edge_states: jnp.ndarray,
    temperature: float,
    log_potentials: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Passes messages from PoolFactors to Variables.

  Args:
    vtof_msgs: Array of shape (num_edge_state,). This holds all the flattened
      variable to all the PoolFactors messages.

    pool_choices_edge_states: Array of shape (num_pool_choices, 2)
      pool_choices_edge_states[ii, 0] contains the global PoolFactor index
      pool_choices_edge_states[ii, 1] contains the message index of the pool
      choice variable's state 0. The message index of the pool choice variable's
      state 1 is pool_choices_edge_states[ii, 1] + 1

    pool_indicators_edge_states: Array of shape (num_pool_factors,)
      pool_indicators_edge_states[ii] contains the message index of the pool
      indicator variable's state 0, which takes into account all the PoolFactors
      of the FactorGraph. The message index of the pool indicator variable's
      state 1 is pool_indicators_edge_states[ii, 1] + 1

    temperature: Temperature for loopy belief propagation. 1.0 corresponds to
      sum-product, 0.0 corresponds to max-product.

    log_potentials: Optional array of log potentials

  Returns:
      Array of shape (num_edge_state,). This holds all the flattened PoolFactors
      to variable messages.
  """
  num_factors = pool_indicators_edge_states.shape[0]
  factor_indices = pool_choices_edge_states[..., 0]

  pool_choices_tof_msgs = (
      vtof_msgs[pool_choices_edge_states[..., 1] + 1]
      - vtof_msgs[pool_choices_edge_states[..., 1]]
  )
  pool_indicators_tof_msgs = (
      vtof_msgs[pool_indicators_edge_states + 1]
      - vtof_msgs[pool_indicators_edge_states]
  )

  pool_choices_maxes, pool_choices_argmaxes = logical.get_maxes_and_argmaxes(
      pool_choices_tof_msgs, factor_indices, num_factors
  )

  # Consider the max-product case separately.
  if temperature == 0.0:
    # Get the first and second pool choice argmaxes per factor
    pool_choices_wo_maxes = pool_choices_tof_msgs.at[pool_choices_argmaxes].set(
        NEG_INF
    )

    pool_choices_second_maxes = (
        jnp.full(shape=(num_factors,), fill_value=NEG_INF)
        .at[factor_indices]
        .max(pool_choices_wo_maxes)
    )

    # Compute the maximum of the incoming messages without self
    pool_choices_maxes_wo_self = pool_choices_maxes[factor_indices]
    pool_choices_maxes_wo_self = pool_choices_maxes_wo_self.at[
        pool_choices_argmaxes
    ].set(pool_choices_second_maxes)

    # Get the outgoing messages
    pool_choices_msgs = jnp.minimum(
        pool_indicators_tof_msgs[factor_indices], -pool_choices_maxes_wo_self
    )
    pool_indicators_msgs = pool_choices_maxes

  else:
    exp_pool_choices_wo_maxes = jnp.exp(
        (pool_choices_tof_msgs - pool_choices_maxes[factor_indices])
        / temperature
    )
    exp_minus_parents_wo_maxes = jnp.exp(
        -(pool_indicators_tof_msgs + pool_choices_maxes) / temperature
    )

    sum_exp_pool_choices_wo_maxes = (
        jnp.zeros((num_factors,))
        .at[factor_indices]
        .add(exp_pool_choices_wo_maxes)
    )
    sum_exp_pool_choices_wo_maxes_wo_self = (
        sum_exp_pool_choices_wo_maxes[factor_indices]
        - exp_pool_choices_wo_maxes
    )

    # Get the outgoing messages
    pool_choices_msgs = -pool_choices_maxes[
        factor_indices
    ] - temperature * jnp.log(
        sum_exp_pool_choices_wo_maxes_wo_self
        + exp_minus_parents_wo_maxes[factor_indices]
    )
    pool_indicators_msgs = pool_choices_maxes + temperature * jnp.log(
        sum_exp_pool_choices_wo_maxes
    )

  # Special case: factors with a single parent
  num_pool_choices = jnp.bincount(factor_indices, length=num_factors)
  first_pool_choices = jnp.concatenate(
      [jnp.zeros(1, dtype=int), jnp.cumsum(num_pool_choices)]
  )[:-1]
  pool_choices_msgs = pool_choices_msgs.at[first_pool_choices].set(
      jnp.where(
          num_pool_choices == 1,
          pool_indicators_tof_msgs,
          pool_choices_msgs[first_pool_choices],
      ),
  )

  ftov_msgs = jnp.zeros_like(vtof_msgs)
  ftov_msgs = ftov_msgs.at[pool_choices_edge_states[..., 1] + 1].set(
      pool_choices_msgs
  )
  ftov_msgs = ftov_msgs.at[pool_indicators_edge_states + 1].set(
      pool_indicators_msgs
  )
  return ftov_msgs
