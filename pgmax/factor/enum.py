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

"""Defines an enumeration factor."""

import dataclasses
import functools
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from pgmax.factor import factor
from pgmax.factor import update_utils
from pgmax.utils import NEG_INF


# pylint: disable=unexpected-keyword-arg
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class EnumWiring(factor.Wiring):
  """Wiring for EnumFactors.

  Attributes:
    factor_configs_edge_states: Array of shape (num_factor_configs, 2)
      factor_configs_edge_states[ii] contains a pair of global enumeration
      factor_config and global edge_state indices
      factor_configs_edge_states[ii, 0] contains the global EnumFactor
      config index
      factor_configs_edge_states[ii, 1] contains the corresponding global
      edge_state index
      Both indices only take into account the EnumFactors of the FactorGraph

    num_val_configs: Number of valid configurations for this wiring
    num_factors: Number of factors covered by this wiring
  """

  factor_configs_edge_states: Union[np.ndarray, jnp.ndarray]

  def __post_init__(self):
    super().__post_init__()

    if self.factor_configs_edge_states.shape[0] == 0:
      num_val_configs = 0
    else:
      num_val_configs = int(self.factor_configs_edge_states[-1, 0]) + 1
    object.__setattr__(self, "num_val_configs", num_val_configs)

    if self.var_states_for_edges.shape[0] == 0:
      num_factors = 0
    else:
      num_factors = int(self.var_states_for_edges[-1, 2]) + 1
    object.__setattr__(self, "num_factors", num_factors)

  def get_inference_arguments(self) -> Dict[str, Any]:
    """Return the list of arguments to run BP with EnumWirings."""
    assert hasattr(self, "num_val_configs")
    return {
        "factor_configs_indices": self.factor_configs_edge_states[..., 0],
        "factor_configs_edge_states": self.factor_configs_edge_states[..., 1],
        "num_val_configs": self.num_val_configs,
        "num_factors": self.num_factors
    }


@dataclasses.dataclass(frozen=True, eq=False)
class EnumFactor(factor.Factor):
  """An enumeration factor.

  Attributes:
    factor_configs: Array of shape (num_val_configs, num_variables)
      An array containing an explicit enumeration of all valid configurations
    log_potentials: Array of shape (num_val_configs,)
      An array containing the log of the potential value for each valid
      configuration

  Raises:
    ValueError: If:
      (1) The dtype of the factor_configs array is not int
      (2) The dtype of the potential array is not float
      (3) factor_configs does not have the correct shape
      (4) The potential array does not have the correct shape
      (5) The factor_configs array contains invalid values
  """

  factor_configs: np.ndarray
  log_potentials: np.ndarray

  def __post_init__(self):
    self.factor_configs.flags.writeable = False
    if not np.issubdtype(self.factor_configs.dtype, np.integer):
      raise ValueError(
          f"Configurations should be integers. Got {self.factor_configs.dtype}."
      )

    if not np.issubdtype(self.log_potentials.dtype, np.floating):
      raise ValueError(
          f"Potential should be floats. Got {self.log_potentials.dtype}."
      )

    if self.factor_configs.ndim != 2:
      raise ValueError(
          "factor_configs should be a 2D array containing a list of valid"
          " configurations for EnumFactor. Got a factor_configs array of shape"
          f" {self.factor_configs.shape}."
      )

    if len(self.variables) != self.factor_configs.shape[1]:
      raise ValueError(
          f"Number of variables {len(self.variables)} doesn't match given"
          f" configurations {self.factor_configs.shape}"
      )

    if self.log_potentials.shape != (self.factor_configs.shape[0],):
      raise ValueError(
          "Expected log potentials of shape"
          f" {(self.factor_configs.shape[0],)} for"
          f" ({self.factor_configs.shape[0]}) valid configurations. Got log"
          f" potentials of shape {self.log_potentials.shape}."
      )

    vars_num_states = np.array([variable[1] for variable in self.variables])
    if not np.logical_and(
        self.factor_configs >= 0, self.factor_configs < vars_num_states[None]
    ).all():
      raise ValueError("Invalid configurations for given variables")

  @staticmethod
  def concatenate_wirings(wirings: Sequence[EnumWiring]) -> EnumWiring:
    """Concatenate a list of EnumWirings.

    Args:
      wirings: A list of EnumWirings

    Returns:
      Concatenated EnumWiring
    """
    if not wirings:
      return EnumWiring(
          var_states_for_edges=np.empty((0, 3), dtype=int),
          factor_configs_edge_states=np.empty((0, 2), dtype=int),
      )

    concatenated_var_states_for_edges = factor.concatenate_var_states_for_edges(
        [wiring.var_states_for_edges for wiring in wirings]
    )

    factor_configs_cumsum = np.insert(
        np.cumsum(
            [wiring.factor_configs_edge_states[-1, 0] + 1 for wiring in wirings]
        ),
        0,
        0,
    )[:-1]

    # Note: this correspomds to all the factor_to_msgs_starts of the EnumFactors
    num_edge_states_cumsum = np.insert(
        np.cumsum(
            [wiring.var_states_for_edges.shape[0] for wiring in wirings]
        ),
        0,
        0,
    )[:-1]

    factor_configs_edge_states = []
    for ww, wiring in enumerate(wirings):
      factor_configs_edge_states.append(
          wiring.factor_configs_edge_states
          + np.array(
              [[factor_configs_cumsum[ww], num_edge_states_cumsum[ww]]],
              dtype=int,
          )
      )

    return EnumWiring(
        var_states_for_edges=concatenated_var_states_for_edges,
        factor_configs_edge_states=np.concatenate(
            factor_configs_edge_states, axis=0
        ),
    )

  @staticmethod
  def compile_wiring(
      factor_edges_num_states: np.ndarray,
      variables_for_factors: Sequence[List[Tuple[int, int]]],
      factor_configs: np.ndarray,
      vars_to_starts: Mapping[Tuple[int, int], int],
      num_factors: int,
  ) -> EnumWiring:
    """Compile an EnumWiring for an EnumFactor or a FactorGroup with EnumFactors.

    Internally calls _compile_var_states_numba and
    _compile_enumeration_wiring_numba for speed.

    Args:
      factor_edges_num_states: An array concatenating the number of states
        for the variables connected to each Factor of the FactorGroup.
        Each variable will appear once for each Factor it connects to.
      variables_for_factors: A list of list of variables. Each list within
        the outer list contains the variables connected to a Factor. The
        same variable can be connected to multiple Factors.
      factor_configs: Array of shape (num_val_configs, num_variables)
        containing an explicit enumeration of all valid configurations.
      vars_to_starts: A dictionary that maps variables to their global
        starting indices For an n-state variable, a global start index of
        m means the global indices of its n variable states are m, m + 1,
        ..., m + n - 1
      num_factors: Number of Factors in the FactorGroup.

    Raises: ValueError if factor_edges_num_states is not of shape
      (num_factors * num_variables, )

    Returns:
      The EnumWiring
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

    num_edges_states_cumsum = np.insert(
        np.cumsum(factor_edges_num_states), 0, 0
    )
    var_states_for_edges = np.empty(
        shape=(num_edges_states_cumsum[-1], 3), dtype=int
    )
    factor.compile_var_states_for_edges_numba(
        var_states_for_edges,
        num_edges_states_cumsum,
        first_var_state_by_edges,
        factor_indices,
    )

    # Step 2: compute factor_configs_edge_states
    num_configs, num_variables = factor_configs.shape
    if factor_edges_num_states.shape != (num_factors * num_variables,):
      raise ValueError(
          "Expected factor_edges_num_states shape is"
          f" {(num_factors * num_variables,)}. Got"
          f" {factor_edges_num_states.shape}."
      )
    factor_configs_edge_states = np.empty(
        (num_factors * num_configs * num_variables, 2), dtype=int
    )
    factor_edges_starts = np.insert(np.cumsum(factor_edges_num_states), 0, 0)
    _compile_enumeration_wiring_numba(
        factor_configs_edge_states,
        factor_configs,
        factor_edges_starts,
        num_factors,
    )

    return EnumWiring(
        var_states_for_edges=var_states_for_edges,
        factor_configs_edge_states=factor_configs_edge_states,
    )

  @staticmethod
  @functools.partial(
      jax.jit, static_argnames=("num_val_configs", "num_factors")
  )
  def compute_energy(
      edge_states_one_hot_decoding: jnp.ndarray,
      log_potentials: jnp.ndarray,
      factor_configs_indices: jnp.ndarray,
      factor_configs_edge_states: jnp.ndarray,
      num_val_configs: int,
      num_factors: int
    ) -> float:
    """Returns the contribution to the energy of several EnumFactors.

    Args:
      edge_states_one_hot_decoding: Array of shape (num_edge_states,)
        Flattened array of one-hot decoding of the edge states connected to the
        EnumFactors
      log_potentials: Array of shape (num_val_configs, ). An entry at index i
        is the log potential function value for the configuration with global
        EnumFactor config index i.
      factor_configs_indices: Array of shape (num_factor_configs,) containing
        the global EnumFactor config indices.
        Only takes into account the EnumFactors of the FactorGraph
      factor_configs_edge_states: Array of shape (num_factor_configs,)
        containingthe global edge_state index associated with each EnumFactor
        config index.
        Only takes into account the EnumFactors of the FactorGraph
      num_val_configs: the total number of valid configurations for all the
        EnumFactors in the factor graph.
      num_factors: the total number of EnumFactors in the factor graph.
    """
    # One-hot decoding of all the factors configs
    fac_config_decoded = (
        jnp.ones(shape=(num_val_configs,), dtype=bool)
        .at[factor_configs_indices]
        .multiply(edge_states_one_hot_decoding[factor_configs_edge_states])
    )

    # Replace infinite log potentials
    clipped_nan_log_potentials = jnp.where(
        jnp.logical_and(jnp.isinf(log_potentials), fac_config_decoded != 1),
        -NEG_INF * jnp.sign(log_potentials),
        log_potentials,
    )

    energy = jnp.where(
        jnp.sum(fac_config_decoded) != num_factors,
        jnp.inf,  # invalid decoding
        -jnp.sum(clipped_nan_log_potentials, where=fac_config_decoded),
    )
    return energy

  @staticmethod
  def compute_factor_energy(
      variables: List[Hashable],
      vars_to_map_states: Dict[Hashable, Any],
      factor_configs: jnp.ndarray,
      log_potentials: jnp.ndarray,
  ) -> float:
    """Returns the contribution to the energy of a single EnumFactor.

    Args:
      variables: List of variables connected by the EnumFactor
      vars_to_map_states: A dictionary mapping each individual variable to
        its MAP state.
      factor_configs: Array of shape (num_val_configs, num_variables)
        An array containing an explicit enumeration of all valid configurations
      log_potentials: Array of shape (num_val_configs,)
        An array containing the log of the potential value for each valid
        configuration
    """
    vars_states_to_configs_indices = {}
    for factor_config_idx, vars_states in enumerate(factor_configs):
      vars_states_to_configs_indices[tuple(vars_states)] = factor_config_idx

    vars_decoded_states = tuple([vars_to_map_states[var] for var in variables])
    if vars_decoded_states not in vars_states_to_configs_indices:
      warnings.warn(
          f"Invalid decoding for Enum factor {variables} "
          f"with variables set to {vars_decoded_states}!"
      )
      factor_energy = np.inf
    else:
      factor_config_idx = vars_states_to_configs_indices[
          vars_decoded_states
      ]
      factor_energy = -log_potentials[factor_config_idx]
    return float(factor_energy)


# pylint: disable=g-doc-args
@nb.jit(parallel=False, cache=True, fastmath=True, nopython=True)
def _compile_enumeration_wiring_numba(
    factor_configs_edge_states: np.ndarray,
    factor_configs: np.ndarray,
    factor_edges_starts: np.ndarray,
    num_factors: int,
):
  """Fast numba computation of the factor_configs_edge_states of an EnumWiring.

  factor_edges_starts is updated in-place.
  """

  num_configs, num_variables = factor_configs.shape

  for factor_idx in nb.prange(num_factors):
    for config_idx in range(num_configs):
      factor_config_idx = num_configs * factor_idx + config_idx
      factor_configs_edge_states[
          num_variables
          * factor_config_idx : num_variables
          * (factor_config_idx + 1),
          0,
      ] = factor_config_idx

      for var_idx in range(num_variables):
        factor_configs_edge_states[
            num_variables * factor_config_idx + var_idx, 1
        ] = (
            factor_edges_starts[num_variables * factor_idx + var_idx]
            + factor_configs[config_idx, var_idx]
        )


# pylint: disable=unused-argument
@functools.partial(
    jax.jit, static_argnames=("num_val_configs", "num_factors", "temperature")
)
def pass_enum_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    factor_configs_indices: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    log_potentials: jnp.ndarray,
    num_val_configs: int,
    num_factors: int,
    temperature: float,
    normalize: bool
) -> jnp.ndarray:
  """Passes messages from EnumFactors to Variables.

  The update is performed in two steps.
  (1) First, a "summary" array is generated that has an entry for every valid
  configuration for every EnumFactor. The elements of this array are simply
  the sums of messages across each valid config.
  (2) Then, the info from factor_configs_edge_states is used to apply the
  scattering operation and generate a flat set of output messages.

  Args:
    vtof_msgs: Array of shape (num_edge_states,)
      This holds all the flattened variable to all the EnumFactors messages

    factor_configs_indices: Array of shape (num_factor_configs,) containing the
      global EnumFactor config indices.
      Only takes into account the EnumFactors of the FactorGraph

    factor_configs_edge_states: Array of shape (num_factor_configs,) containing
      the global edge_state index associated with each EnumFactor config index.
      Only takes into account the EnumFactors of the FactorGraph

    log_potentials: Array of shape (num_val_configs, ). An entry at index i
      is the log potential function value for the configuration with global
      EnumFactor config index i.

    num_val_configs: the total number of valid configurations for all the
      EnumFactors in the factor graph.

    num_factors: total number of EnumFactors in the factor graph.

    temperature: Temperature for loopy belief propagation. 1.0 corresponds
      to sum-product, 0.0 corresponds to max-product.

    normalize: Whether we normalize the outgoing messages. Not used for
      EnumFactors.

  Returns:
    Array of shape (num_edge_states,). This holds all the flattened
    EnumFactors to variable messages.
  """
  fac_config_summary_sum = (
      jnp.zeros(shape=(num_val_configs,))
      .at[factor_configs_indices]
      .add(vtof_msgs[factor_configs_edge_states])
  ) + log_potentials
  max_factor_config_summary_for_edge_states = (
      jnp.full(shape=(vtof_msgs.shape[0],), fill_value=-jnp.inf)
      .at[factor_configs_edge_states]
      .max(fac_config_summary_sum[factor_configs_indices])
  )

  if temperature == 0.0:
    ftov_msgs = max_factor_config_summary_for_edge_states
  else:
    ftov_msgs = update_utils.logsumexps_with_temp(
        data=fac_config_summary_sum[factor_configs_indices],
        labels=factor_configs_edge_states,
        num_labels=vtof_msgs.shape[0],
        temperature=temperature,
        maxes=max_factor_config_summary_for_edge_states
    )

  # Remove incoming messages
  ftov_msgs -= vtof_msgs
  return ftov_msgs
