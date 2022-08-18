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
"""Defines an enumeration factor."""

import dataclasses
import functools
from typing import List, Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from pgmax.factor import factor
from pgmax.utils import NEG_INF


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class EnumWiring(factor.Wiring):
  """Wiring for EnumFactors.

  Attributes:
    factor_configs_edge_states: Array of shape (num_factor_configs, 2)
      factor_configs_edge_states[ii] contains a pair of global enumeration
      factor_config and global edge_state indices
        - factor_configs_edge_states[ii, 0] contains the global EnumFactor
          config index
        - factor_configs_edge_states[ii, 1] contains the corresponding global
          edge_state index
      Both indices only take into account the EnumFactors of the FactorGraph

    num_val_configs: Number of valid configurations for this wiring
  """

  factor_configs_edge_states: Union[np.ndarray, jnp.ndarray]

  def __post_init__(self):
    super().__post_init__()

    if self.factor_configs_edge_states.shape[0] == 0:
      num_val_configs = 0
    else:
      num_val_configs = int(self.factor_configs_edge_states[-1, 0]) + 1
    object.__setattr__(self, "num_val_configs", num_val_configs)


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
          edges_num_states=np.empty((0,), dtype=int),
          var_states_for_edges=np.empty((0,), dtype=int),
          factor_configs_edge_states=np.empty((0, 2), dtype=int),
      )

    factor_configs_cumsum = np.insert(
        np.array(
            [wiring.factor_configs_edge_states[-1, 0] + 1 for wiring in wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]

    # Note: this correspomds to all the factor_to_msgs_starts of the EnumFactors
    num_edge_states_cumsum = np.insert(
        np.array(
            [wiring.edges_num_states.sum() for wiring in wirings]
        ).cumsum(),
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
        edges_num_states=np.concatenate(
            [wiring.edges_num_states for wiring in wirings]
        ),
        var_states_for_edges=np.concatenate(
            [wiring.var_states_for_edges for wiring in wirings]
        ),
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
    var_states = []
    for variables_for_factor in variables_for_factors:
      for variable in variables_for_factor:
        var_states.append(vars_to_starts[variable])
    var_states = np.array(var_states)

    num_states_cumsum = np.insert(np.cumsum(factor_edges_num_states), 0, 0)
    var_states_for_edges = np.empty(shape=(num_states_cumsum[-1],), dtype=int)
    factor.compile_var_states_numba(
        var_states_for_edges, num_states_cumsum, var_states
    )

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
        edges_num_states=factor_edges_num_states,
        var_states_for_edges=var_states_for_edges,
        factor_configs_edge_states=factor_configs_edge_states,
    )


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


@functools.partial(jax.jit, static_argnames=("num_val_configs", "temperature"))
def pass_enum_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    log_potentials: jnp.ndarray,
    num_val_configs: int,
    temperature: float,
) -> jnp.ndarray:
  """Passes messages from EnumFactors to Variables.

  The update is performed in two steps.
  (1) First, a "summary" array is generated that has an entry for every valid
  configuration for every EnumFactor. The elements of this array are simply
  the sums of messages across each valid config.
  (2) Then, the info from factor_configs_edge_states is used to apply the
  scattering operation and generate a flat set of output messages.

  Args:
    vtof_msgs: Array of shape (num_edge_state,)
      This holds all theflattened variable to all the EnumFactors messages

    factor_configs_edge_states: Array of shape (num_factor_configs, 2)
      factor_configs_edge_states[ii] contains a pair of global enumeration
      factor_config and global edge_state indices
        - factor_configs_edge_states[ii, 0] contains the global EnumFactor
          config index
        - factor_configs_edge_states[ii, 1] contains the corresponding global
          edge_state index
      Both indices only take into account the EnumFactors of the FactorGraph

    log_potentials: Array of shape (num_val_configs, ). An entry at index i
      is the log potential function value for the configuration with global
      EnumFactor config index i.

    num_val_configs: the total number of valid configurations for all the
      EnumFactors in the factor graph.

    temperature: Temperature for loopy belief propagation. 1.0 corresponds
      to sum-product, 0.0 corresponds to max-product.

  Returns:
      Array of shape (num_edge_state,). This holds all the flattened
      EnumFactors to variable messages.
  """
  fac_config_summary_sum = (
      jnp.zeros(shape=(num_val_configs,))
      .at[factor_configs_edge_states[..., 0]]
      .add(vtof_msgs[factor_configs_edge_states[..., 1]])
  ) + log_potentials
  max_factor_config_summary_for_edge_states = (
      jnp.full(shape=(vtof_msgs.shape[0],), fill_value=NEG_INF)
      .at[factor_configs_edge_states[..., 1]]
      .max(fac_config_summary_sum[factor_configs_edge_states[..., 0]])
  )
  ftov_msgs = max_factor_config_summary_for_edge_states - vtof_msgs
  if temperature != 0.0:
    ftov_msgs = ftov_msgs + (
        temperature
        * jnp.log(
            jnp.full(shape=(vtof_msgs.shape[0],), fill_value=jnp.exp(NEG_INF))
            .at[factor_configs_edge_states[..., 1]]
            .add(
                jnp.exp(
                    (
                        fac_config_summary_sum[
                            factor_configs_edge_states[..., 0]
                        ]
                        - max_factor_config_summary_for_edge_states[
                            factor_configs_edge_states[..., 1]
                        ]
                    )
                    / temperature
                )
            )
        )
    )
  return ftov_msgs
