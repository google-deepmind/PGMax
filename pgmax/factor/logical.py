# Copyright 2022 Intrinsic Innovation LLC.
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

"""Defines a logical factor."""

import dataclasses
import functools
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from pgmax.factor import factor
from pgmax.factor import update_utils
from pgmax.utils import NEG_INF

# Temperature threshold, under which we estimate that stability problems may
# arise and we add some extra compute to fix them
TEMPERATURE_STABILITY_THRE = 0.5


# pylint: disable=unexpected-keyword-arg
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class LogicalWiring(factor.Wiring):
  """Wiring for LogicalFactors.

  Attributes:
    parents_edge_states: Array of shape (num_parents, 2)
      parents_edge_states[ii, 0] contains the global LogicalFactor index
      parents_edge_states[ii, 1] contains the message index of the parent
      variable's relevant state
      The message index of the parent variable's other state is
      parents_edge_states[ii, 1] + edge_states_offset
      Both indices only take into account the LogicalFactors of the same subtype
      (OR/AND) of the FactorGraph

    children_edge_states: Array of shape (num_factors,)
      children_edge_states[ii] contains the message index of the child
      variable's relevant state
      The message index of the child variable's other state is
      children_edge_states[ii] + edge_states_offset
      Only takes into account the LogicalFactors of the same subtype (OR/AND)
      of the FactorGraph

    edge_states_offset: Offset to go from a variable's relevant state to its
      other state
      For ORFactors the edge_states_offset is 1
      For ANDFactors the edge_states_offset is -1

  Raises:
    ValueError: If:
      (1) The are no num_logical_factors different factor indices
      (2) There is a factor index higher than num_logical_factors - 1
      (3) The edge_states_offset is not 1 or -1
  """

  parents_edge_states: Union[np.ndarray, jnp.ndarray]
  children_edge_states: Union[np.ndarray, jnp.ndarray]
  edge_states_offset: int

  def __post_init__(self):
    super().__post_init__()

    if self.children_edge_states.shape[0] > 0:
      logical_factor_indices = self.parents_edge_states[:, 0]
      num_logical_factors = self.children_edge_states.shape[0]

      if np.unique(logical_factor_indices).shape[0] != num_logical_factors:
        raise ValueError(
            f"The LogicalWiring must have {num_logical_factors} different"
            " LogicalFactor indices"
        )

      if logical_factor_indices.max() >= num_logical_factors:
        raise ValueError(
            f"The highest LogicalFactor index must be {num_logical_factors - 1}"
        )

      if self.edge_states_offset != 1 and self.edge_states_offset != -1:
        raise ValueError(
            "The LogicalWiring's edge_states_offset must be 1 (for OR) and -1"
            f" (for AND), but is {self.edge_states_offset}"
        )

  def get_inference_arguments(self) -> Dict[str, Any]:
    """Return the list of arguments to run BP with LogicalWirings."""
    return {
        "parents_factor_indices": self.parents_edge_states[..., 0],
        "parents_msg_indices": self.parents_edge_states[..., 1],
        "children_edge_states": self.children_edge_states,
        "edge_states_offset": self.edge_states_offset,
    }


@dataclasses.dataclass(frozen=True, eq=False)
class LogicalFactor(factor.Factor):
  """A logical OR/AND factor of the form (p1,...,pn, c) where p1,...,pn are the parents variables and c is the child variable.

  Attributes:
    edge_states_offset: Offset to go from a variable's relevant state to its
      other state
      For ORFactors the edge_states_offset is 1
      For ANDFactors the edge_states_offset is -1

  Raises:
    ValueError: If:
      (1) There are less than 2 variables
      (2) The variables are not all binary
  """

  log_potentials: np.ndarray = dataclasses.field(
      init=False,
      default_factory=lambda: np.empty((0,)),
  )
  edge_states_offset: int = dataclasses.field(init=False)

  def __post_init__(self):
    if len(self.variables) < 2:
      raise ValueError(
          "A LogicalFactor requires at least one parent variable and one child"
          " variable"
      )

    if not np.all([variable[1] == 2 for variable in self.variables]):
      raise ValueError("All the variables in a LogicalFactor should be binary")

  @staticmethod
  def concatenate_wirings(wirings: Sequence[LogicalWiring]) -> LogicalWiring:
    """Concatenate a list of LogicalWirings.

    Args:
      wirings: A list of LogicalWirings

    Returns:
      Concatenated LogicalWiring
    """
    if not wirings:
      return LogicalWiring(
          var_states_for_edges=np.empty((0, 3), dtype=int),
          parents_edge_states=np.empty((0, 2), dtype=int),
          children_edge_states=np.empty((0,), dtype=int),
          edge_states_offset=1,
      )

    # Factors indices offsets
    num_factors_cumsum = np.insert(
        np.cumsum(
            [wiring.parents_edge_states[-1, 0] + 1 for wiring in wirings]
        ),
        0,
        0,
    )[:-1]

    # Messages offsets
    concatenated_var_states_for_edges = factor.concatenate_var_states_for_edges(
        [wiring.var_states_for_edges for wiring in wirings]
    )

    # Factors indices offsets
    num_factors_cumsum = np.insert(
        np.cumsum(
            [wiring.parents_edge_states[-1, 0] + 1 for wiring in wirings]
        ),
        0,
        0,
    )[:-1]

    # Messages offsets
    # Note: this is all the factor_to_msgs_starts for the LogicalFactors
    num_edge_states_cumsum = np.insert(
        np.cumsum(
            [wiring.var_states_for_edges.shape[0] for wiring in wirings]
        ),
        0,
        0,
    )[:-1]
    parents_edge_states = []
    children_edge_states = []

    for ww, or_wiring in enumerate(wirings):
      offsets = np.array(
          [[num_factors_cumsum[ww], num_edge_states_cumsum[ww]]], dtype=int
      )
      parents_edge_states.append(or_wiring.parents_edge_states + offsets)
      children_edge_states.append(
          or_wiring.children_edge_states + offsets[:, 1]
      )

    return LogicalWiring(
        var_states_for_edges=concatenated_var_states_for_edges,
        parents_edge_states=np.concatenate(parents_edge_states, axis=0),
        children_edge_states=np.concatenate(children_edge_states, axis=0),
        edge_states_offset=wirings[0].edge_states_offset,
    )

  @staticmethod
  def compile_wiring(
      factor_edges_num_states: np.ndarray,
      variables_for_factors: Sequence[List[Tuple[int, int]]],
      factor_sizes: np.ndarray,
      vars_to_starts: Mapping[Tuple[int, int], int],
      edge_states_offset: int,
  ) -> LogicalWiring:
    """Compile a LogicalWiring for a LogicalFactor or a FactorGroup with LogicalFactors.

    Internally calls _compile_var_states_numba and
    _compile_logical_wiring_numba for speed.

    Args:
      factor_edges_num_states: An array concatenating the number of states for
        the variables connected to each Factor of the FactorGroup. Each variable
        will appear once for each Factor it connects to.
      variables_for_factors: A tuple of tuples containing variables connected to
        each Factor of the FactorGroup. Each variable will appear once for each
        Factor it connects to.
      factor_sizes: An array containing the different factor sizes.
      vars_to_starts: A dictionary that maps variables to their global starting
        indices For an n-state variable, a global start index of m means the
        global indices of its n variable states are m, m + 1, ..., m + n - 1
      edge_states_offset: Offset to go from a variable's relevant state to its
        other state
        For ORFactors the edge_states_offset is 1
        For ANDFactors the edge_states_offset is -1

    Returns:
      The LogicalWiring
    """
    # Step 1: compute var_states_for_edges
    first_var_state_by_edges = []
    factor_indices = []
    for factor_idx, variables_for_factor in enumerate(variables_for_factors):
      for variable in variables_for_factor:
        first_var_state_by_edges.append(vars_to_starts[variable])
        factor_indices.append(factor_idx)
    first_var_state_by_edges = np.array(first_var_state_by_edges)
    factor_indices = np.array(factor_indices)

    # All the variables in a LogicalFactorGroup are binary
    num_edges_states_cumsum = np.arange(
        0, 2 * first_var_state_by_edges.shape[0] + 2, 2
    )
    var_states_for_edges = np.empty(
        shape=(2 * first_var_state_by_edges.shape[0], 3), dtype=int
    )
    factor.compile_var_states_for_edges_numba(
        var_states_for_edges,
        num_edges_states_cumsum,
        first_var_state_by_edges,
        factor_indices,
    )

    # Step 2: compute parents_edge_states and children_edge_states
    num_parents = factor_sizes - 1
    num_parents_cumsum = np.insert(np.cumsum(num_parents), 0, 0)
    parents_edge_states = np.empty(shape=(num_parents_cumsum[-1], 2), dtype=int)
    children_edge_states = np.empty(shape=(factor_sizes.shape[0],), dtype=int)

    # Relevant state differs for ANDFactors and ORFactors
    relevant_state = (-edge_states_offset + 1) // 2

    # Note: edges_num_states_cumsum corresponds to the factor_to_msgs_start
    factor_num_edges_states_cumsum = np.insert(
        np.cumsum(2 * factor_sizes), 0, 0
    )
    _compile_logical_wiring_numba(
        parents_edge_states=parents_edge_states,
        children_edge_states=children_edge_states,
        num_parents=num_parents,
        num_parents_cumsum=num_parents_cumsum,
        factor_num_edges_states_cumsum=factor_num_edges_states_cumsum,
        relevant_state=relevant_state,
    )

    return LogicalWiring(
        var_states_for_edges=var_states_for_edges,
        parents_edge_states=parents_edge_states,
        children_edge_states=children_edge_states,
        edge_states_offset=edge_states_offset,
    )

  @staticmethod
  @jax.jit
  def compute_energy(
      edge_states_one_hot_decoding: jnp.ndarray,
      parents_factor_indices: jnp.ndarray,
      parents_msg_indices: jnp.ndarray,
      children_edge_states: jnp.ndarray,
      edge_states_offset: int,
      log_potentials: Optional[jnp.ndarray] = None,
  ) -> float:
    """Returns the contribution to the energy of several LogicalFactors.

    Args:
      edge_states_one_hot_decoding: Array of shape (num_edge_states,)
        Flattened array of one-hot decoding of the edge states connected to the
        LogicalFactors

      parents_factor_indices: Array of shape (num_parents,)
        parents_factor_indices[ii] contains the global LogicalFactor index of
        the parent variable's relevant state.
        Only takes into account the LogicalFactors of the same subtype (OR/AND)
        of the FactorGraph

      parents_msg_indices: Array of shape (num_parents,)
        parents_msg_indices[ii] contains the message index of the parent
        variable's relevant state.
        The message index of the parent variable's other state is
        parents_msg_indices[ii] + edge_states_offset.
        Only takes into account the LogicalFactors of the same subtype (OR/AND)
        of the FactorGraph

      children_edge_states: Array of shape (num_factors,)
        children_edge_states[ii] contains the message index of the child
        variable's relevant state
        The message index of the child variable's other state is
        children_edge_states[ii] + edge_states_offset
        Only takes into account the LogicalFactors of the same subtype (OR/AND)
        of the FactorGraph

      edge_states_offset: Offset to go from a variable's relevant state to its
        other state
        For ORFactors the edge_states_offset is 1
        For ANDFactors the edge_states_offset is -1

      log_potentials: Optional array of log potentials
    """
    num_factors = children_edge_states.shape[0]

    # parents_edge_states[..., 1] indicates the relevant state,
    # which is 0 for ORFactors and 1 for ANDFactors
    # If all the parents variables of a factor are in the factor type's relevant
    # state, then the child variable must be in the relevant state too
    logical_parents_decoded = (
        jnp.ones(shape=(num_factors,))
        .at[parents_factor_indices]
        .multiply(edge_states_one_hot_decoding[parents_msg_indices])
    )
    logical_children_decoded = edge_states_one_hot_decoding[
        children_edge_states
    ]
    energy = jnp.where(
        jnp.any(logical_parents_decoded != logical_children_decoded),
        jnp.inf,  # invalid decoding
        0.0
    )
    return energy


@dataclasses.dataclass(frozen=True, eq=False)
class ORFactor(LogicalFactor):
  """An OR factor of the form (p1,...,pn, c) where p1,...,pn are the parents variables and c is the child variable.

  An OR factor is defined as:
  F(p1, p2, ..., pn, c) = 0 <=> c = OR(p1, p2, ..., pn)
  F(p1, p2, ..., pn, c) = -inf o.w.

  Attributes:
    edge_states_offset: Offset to go from a variable's relevant state to its
      other state
      For ORFactors the edge_states_offset is 1.
  """

  edge_states_offset: int = dataclasses.field(init=False, default=1)

  @staticmethod
  def compute_factor_energy(
      variables: List[Hashable],
      vars_to_map_states: Dict[Hashable, Any],
      **kwargs,
  ) -> float:
    """Returns the contribution to the energy of a single ORFactor.

    Args:
      variables: List of variables connected by the ORFactor
      vars_to_map_states: A dictionary mapping each individual variable to
        its MAP state.
      **kwargs: Other parameters, not used
    """
    vars_decoded_states = np.array(
        [vars_to_map_states[var] for var in variables]
    )
    parents_decoded_states = vars_decoded_states[:-1]
    child_decoded_states = vars_decoded_states[-1]

    if np.any(parents_decoded_states) != child_decoded_states:
      warnings.warn(
          f"Invalid decoding for OR factor {variables} "
          f"with parents set to {parents_decoded_states} "
          f"and child set to {child_decoded_states}!"
      )
      factor_energy = np.inf
    else:
      factor_energy = 0.0
    return factor_energy


@dataclasses.dataclass(frozen=True, eq=False)
class ANDFactor(LogicalFactor):
  """An AND factor of the form (p1,...,pn, c) where p1,...,pn are the parents variables and c is the child variable.

  An AND factor is defined as:
    F(p1, p2, ..., pn, c) = 0 <=> c = AND(p1, p2, ..., pn)
    F(p1, p2, ..., pn, c) = -inf o.w.

  Attributes:
    edge_states_offset: Offset to go from a variable's relevant state to its
      other state
      For ANDFactors the edge_states_offset is -1.
  """

  edge_states_offset: int = dataclasses.field(init=False, default=-1)

  @staticmethod
  def compute_factor_energy(
      variables: List[Hashable],
      vars_to_map_states: Dict[Hashable, Any],
      **kwargs,
  ) -> float:
    """Returns the contribution to the energy of a single ANDFactor.

    Args:
      variables: List of variables connected by the ANDFactor
      vars_to_map_states: A dictionary mapping each individual variable to
        its MAP state.
      **kwargs: Other parameters, not used
    """
    vars_decoded_states = np.array(
        [vars_to_map_states[var] for var in variables]
    )
    parents_decoded_states = vars_decoded_states[:-1]
    child_decoded_states = vars_decoded_states[-1]

    if np.all(parents_decoded_states) != child_decoded_states:
      warnings.warn(
          f"Invalid decoding for AND factor {variables} "
          f"with parents set to {parents_decoded_states} "
          f"and child set to {child_decoded_states}!"
      )
      factor_energy = np.inf
    else:
      factor_energy = 0.0
    return factor_energy


# pylint: disable=g-doc-args
@nb.jit(parallel=False, cache=True, fastmath=True, nopython=True)
def _compile_logical_wiring_numba(
    parents_edge_states: np.ndarray,
    children_edge_states: np.ndarray,
    num_parents: np.ndarray,
    num_parents_cumsum: np.ndarray,
    factor_num_edges_states_cumsum: np.ndarray,
    relevant_state: int,
):
  """Fast numba computation of the parents_edge_states and children_edge_states of a LogicalWiring.

  parents_edge_states and children_edge_states are updated in-place.
  """

  for factor_idx in nb.prange(num_parents.shape[0]):
    start_parents, end_parents = (
        num_parents_cumsum[factor_idx],
        num_parents_cumsum[factor_idx + 1],
    )
    parents_edge_states[start_parents:end_parents, 0] = factor_idx
    parents_edge_states[start_parents:end_parents, 1] = np.arange(
        factor_num_edges_states_cumsum[factor_idx] + relevant_state,
        factor_num_edges_states_cumsum[factor_idx]
        + 2 * num_parents[factor_idx],
        2,
    )
    children_edge_states[factor_idx] = (
        factor_num_edges_states_cumsum[factor_idx]
        + 2 * num_parents[factor_idx]
        + relevant_state
    )


# pylint: disable=unused-argument
@functools.partial(jax.jit, static_argnames=("temperature", "normalize"))
def pass_logical_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    parents_factor_indices: jnp.ndarray,
    parents_msg_indices: jnp.ndarray,
    children_edge_states: jnp.ndarray,
    edge_states_offset: int,
    temperature: float,
    normalize: bool,
    log_potentials: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Passes messages from LogicalFactors to Variables.

  Args:
    vtof_msgs: Array of shape (num_edge_states,). This holds all the
      flattened variable to all the LogicalFactors messages.

    parents_factor_indices: Array of shape (num_parents,)
      parents_factor_indices[ii] contains the global LogicalFactor index of the
      parent variable's relevant state.
      Only takes into account the LogicalFactors of the same subtype (OR/AND)
      of the FactorGraph

    parents_msg_indices: Array of shape (num_parents,)
      parents_msg_indices[ii] contains the message index of the parent
      variable's relevant state.
      The message index of the parent variable's other state is
      parents_msg_indices[ii] + edge_states_offset.
      Only takes into account the LogicalFactors of the same subtype (OR/AND)
      of the FactorGraph

    children_edge_states: Array of shape (num_factors,)
      children_edge_states[ii] contains the message index of the child
      variable's relevant state
      The message index of the child variable's other state is
      children_edge_states[ii] + edge_states_offset
      Only takes into account the LogicalFactors of the same subtype (OR/AND)
      of the FactorGraph

    edge_states_offset: Offset to go from a variable's relevant state to its
      other state
      For ORFactors the edge_states_offset is 1
      For ANDFactors the edge_states_offset is -1

    temperature: Temperature for loopy belief propagation. 1.0 corresponds
      to sum-product, 0.0 corresponds to max-product.

    normalize: Whether we normalize the outgoing messages. Set to True for BP
      and to False for Smooth Dual LP.

    log_potentials: Optional array of log potentials

  Returns:
    Array of shape (num_edge_states,). This holds all the flattened
    LogicalFactors of the same subtype (OR/AND) to variable messages.

  Note: The updates below are derived to be mathematically stable at low
    temperatures in (0, 0.1].
    logaddexp_with_temp and logminusexp_with_temp are also derived to be
    numerically stable at these low temperatures.

    However, we observed that deriving efficient message updates for OR/AND
    factors -with linear complexity- comes at the cost of a decrease in
    numerical stability for two regimes:
    (1) Larger factors have higher but rarer errors
    (2) Temperatures around 0.1 have higher errors than lower (and higher)
    temperatures
    More compute could be spent to improve the stability of these two cases.
  """
  num_factors = children_edge_states.shape[0]

  parents_tof_msgs_relevant = vtof_msgs[
      parents_msg_indices + edge_states_offset
  ]
  parents_tof_msgs_other = vtof_msgs[parents_msg_indices]
  parents_tof_msgs_diffs = parents_tof_msgs_relevant - parents_tof_msgs_other

  children_tof_msgs_relevant = vtof_msgs[
      children_edge_states + edge_states_offset
  ]
  children_tof_msgs_other = vtof_msgs[children_edge_states]

  # Get the easier outgoing messages to children variables
  factorsum_parents_other = (
      jnp.zeros((num_factors,))
      .at[parents_factor_indices]
      .add(parents_tof_msgs_other)
  )
  children_msgs_other = factorsum_parents_other

  # Get first maxes and argmaxes for incoming parents messages of each factor
  (
      first_parents_diffs_maxes,
      first_parents_diffs_argmaxes,
  ) = update_utils.get_maxes_and_argmaxes(
      parents_tof_msgs_diffs, parents_factor_indices, num_factors
  )
  # Get second maxes
  second_parents_diffs_maxes = (
      jnp.full(shape=(num_factors,), fill_value=NEG_INF)
      .at[parents_factor_indices]
      .max(parents_tof_msgs_diffs.at[first_parents_diffs_argmaxes].set(NEG_INF))
  )

  # Consider the max-product case separately.
  # See https://arxiv.org/pdf/2111.02458.pdf, Appendix C.3
  if temperature == 0.0:
    # First, get the easier outgoing message to parents variables
    maxes_parents_by_edges = jnp.maximum(
        parents_tof_msgs_other, parents_tof_msgs_relevant
    )
    factorsum_maxes_parents_by_edges = (
        jnp.full(shape=(num_factors,), fill_value=0.0)
        .at[parents_factor_indices]
        .add(maxes_parents_by_edges)
    )
    parents_msgs_relevant = (
        factorsum_maxes_parents_by_edges[parents_factor_indices]
        + children_tof_msgs_relevant[parents_factor_indices]
        - maxes_parents_by_edges
    )

    # Outgoing messages to children variables
    min_w_zeros_first_parents_diffs_maxes = jnp.minimum(
        0.0, first_parents_diffs_maxes
    )
    children_msgs_relevant = (
        factorsum_maxes_parents_by_edges + min_w_zeros_first_parents_diffs_maxes
    )

    # Finally, derive the harder outgoing messages to parents variables
    parents_msgs_other_option1 = (
        children_tof_msgs_other[parents_factor_indices]
        + factorsum_parents_other[parents_factor_indices]
        - parents_tof_msgs_other
    )
    parents_msgs_other_option2 = (
        parents_msgs_relevant
        + min_w_zeros_first_parents_diffs_maxes[parents_factor_indices]
    )

    # Consider the first_parents_diffs_argmaxes separately
    parents_msgs_other_option2_for_argmax = parents_msgs_relevant[
        first_parents_diffs_argmaxes
    ] + jnp.minimum(0.0, second_parents_diffs_maxes)
    parents_msgs_other_option2 = parents_msgs_other_option2.at[
        first_parents_diffs_argmaxes
    ].set(parents_msgs_other_option2_for_argmax)

    parents_msgs_other = jnp.maximum(
        parents_msgs_other_option1, parents_msgs_other_option2
    )

  else:
    # Get the easier outgoing messages to parents variables
    # The steps are similar to temperature=0
    logsumexp_parents_by_edges = update_utils.logaddexp_with_temp(
        parents_tof_msgs_relevant, parents_tof_msgs_other, temperature
    )
    factorsum_logsumexp_parents_by_edges = (
        jnp.full(shape=(num_factors,), fill_value=0.0)
        .at[parents_factor_indices]
        .add(logsumexp_parents_by_edges)
    )
    factorsum_logsumexp_parents_by_edges_wo_self = (
        factorsum_logsumexp_parents_by_edges[parents_factor_indices]
        - logsumexp_parents_by_edges
    )
    factorsum_parents_other_wo_self = (
        factorsum_parents_other[parents_factor_indices] - parents_tof_msgs_other
    )
    parents_msgs_relevant = (
        children_tof_msgs_relevant[parents_factor_indices]
        + factorsum_logsumexp_parents_by_edges_wo_self
    )

    # As before, the harder outgoing messages to children variables follow
    children_msgs_relevant = update_utils.logminusexp_with_temp(
        factorsum_logsumexp_parents_by_edges,
        factorsum_parents_other,
        temperature,
        eps=1e-4
    )

    if temperature < TEMPERATURE_STABILITY_THRE:
      # If factorsum_logsumexp_parents_by_edges and factorsum_parents_other
      # are closer than the high threshold eps, logminusexp_with_temp above sets
      # the outgoing messages to NEG_INF, which we replace by a lower bound.
      children_msgs_relevant = jnp.maximum(
          children_msgs_relevant,
          update_utils.logaddexp_with_temp(
              factorsum_parents_other + first_parents_diffs_maxes,
              factorsum_parents_other + second_parents_diffs_maxes,
              temperature
          )
      )

    # Finally, derive the harder outgoing messages to parents variables
    parents_msgs_other_option1 = (
        children_tof_msgs_other[parents_factor_indices]
        + factorsum_parents_other_wo_self
    )
    parents_msgs_other_option2 = (
        children_tof_msgs_relevant[parents_factor_indices]
        + factorsum_logsumexp_parents_by_edges_wo_self
    )
    parents_msgs_other_option3 = (
        children_tof_msgs_relevant[parents_factor_indices]
        + factorsum_parents_other_wo_self
    )
    logaddexp_parents_msgs_other_option1_2 = update_utils.logaddexp_with_temp(
        parents_msgs_other_option1, parents_msgs_other_option2, temperature
    )
    parents_msgs_other = update_utils.logminusexp_with_temp(
        logaddexp_parents_msgs_other_option1_2,
        parents_msgs_other_option3,
        temperature,
        eps=1e-4
    )

    if temperature < TEMPERATURE_STABILITY_THRE:
      # If logaddexp_parents_msgs_other_option1_2 and parents_msgs_other_option3
      # are closer than the high threshold eps, logminusexp_with_temp above sets
      # the outgoing messages to NEG_INF, which we replace by a lower bound.
      factorsum_parents_other_plus_max_diff_wo_self = (
          factorsum_parents_other_wo_self
          + first_parents_diffs_maxes[parents_factor_indices]
      )
      factorsum_parents_other_plus_max_diff_wo_self = (
          factorsum_parents_other_plus_max_diff_wo_self.at[
              first_parents_diffs_argmaxes
          ].set(
              factorsum_parents_other_wo_self[first_parents_diffs_argmaxes]
              + second_parents_diffs_maxes
          )
      )
      # Note: we can always spend more compute to refine the lower bound
      parents_msgs_other_lower_bound = update_utils.logaddexp_with_temp(
          parents_msgs_other_option1,
          children_tof_msgs_relevant[parents_factor_indices]
          + factorsum_parents_other_plus_max_diff_wo_self,
          temperature,
      )
      parents_msgs_other = jnp.maximum(
          parents_msgs_other, parents_msgs_other_lower_bound,
      )

  # Special case: factors with a single parent
  num_parents = jnp.bincount(parents_factor_indices, length=num_factors)
  first_elements = jnp.concatenate(
      [jnp.zeros(1, dtype=int), jnp.cumsum(num_parents)]
  )[:-1]
  parents_msgs_relevant = parents_msgs_relevant.at[first_elements].set(
      jnp.where(
          num_parents == 1,
          children_tof_msgs_relevant,
          parents_msgs_relevant[first_elements],
      ),
  )
  parents_msgs_other = parents_msgs_other.at[first_elements].set(
      jnp.where(
          num_parents == 1,
          children_tof_msgs_other,
          parents_msgs_other[first_elements],
      ),
  )

  # Outgoing messages
  ftov_msgs = jnp.zeros_like(vtof_msgs)
  if normalize:
    ftov_msgs = ftov_msgs.at[
        parents_msg_indices + edge_states_offset
    ].set(parents_msgs_relevant - parents_msgs_other)

    ftov_msgs = ftov_msgs.at[children_edge_states + edge_states_offset].set(
        children_msgs_relevant - children_msgs_other
    )
  else:
    ftov_msgs = ftov_msgs.at[
        parents_msg_indices + edge_states_offset
    ].set(parents_msgs_relevant)
    ftov_msgs = ftov_msgs.at[parents_msg_indices].set(parents_msgs_other)

    ftov_msgs = ftov_msgs.at[children_edge_states + edge_states_offset].set(
        children_msgs_relevant
    )
    ftov_msgs = ftov_msgs.at[children_edge_states].set(children_msgs_other)
  return ftov_msgs
