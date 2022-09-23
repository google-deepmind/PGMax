# pyformat style:midnight
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
"""Defines EnumFactorGroup and PairwiseFactorGroup."""

import collections
import dataclasses
from typing import Any, FrozenSet, Optional, OrderedDict, Type, Union

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from pgmax import factor
from pgmax.factor import enum
from pgmax.fgroup import fgroup


@dataclasses.dataclass(frozen=True, eq=False)
class EnumFactorGroup(fgroup.FactorGroup):
  """Class to represent a group of EnumFactors.

  All factors in the group are assumed to have the same set of valid
  configurations.
  The associated log potentials can however be different across factors.

  Attributes:
    factor_configs: Array of shape (num_val_configs, num_variables) containing
      explicit enumeration of all valid configurations
    log_potentials: Optional 1D array of shape (num_val_configs,) or 2D array of
      shape (num_factors, num_val_configs).
      If 1D, the log potentials are copied for each factor of the group.
      If 2D, it specifices the log potentials of each factor.
      If None, the log potential are initialized to uniform 0.
    factor_type: Factor type shared by all the Factors in the FactorGroup.

  Raises:
    ValueError if:
      (1) The specified log_potentials is not of the expected shape.
      (2) The dtype of the potential array is not float
  """

  factor_configs: np.ndarray
  log_potentials: Optional[np.ndarray] = None  #: :meta private:
  factor_type: Type[factor.Factor] = dataclasses.field(
      init=False,
      default=enum.EnumFactor,
  )  #: :meta private:

  def __post_init__(self):
    super().__post_init__()

    num_val_configs = self.factor_configs.shape[0]
    if self.log_potentials is None:
      log_potentials = np.zeros(
          (self.num_factors, num_val_configs),
          dtype=float,
      )
    else:
      if self.log_potentials.shape != (
          num_val_configs,) and self.log_potentials.shape != (self.num_factors,
                                                              num_val_configs):
        raise ValueError(
            f"Expected log potentials shape: {(num_val_configs,)} or"
            f" {(self.num_factors, num_val_configs)}. Got"
            f" {self.log_potentials.shape}.")
      log_potentials = np.broadcast_to(
          self.log_potentials,
          (self.num_factors, num_val_configs),
      )

    if not np.issubdtype(log_potentials.dtype, np.floating):
      raise ValueError(
          f"Potentials should be floats. Got {log_potentials.dtype}.")
    object.__setattr__(self, "log_potentials", log_potentials)

  # pylint: disable=g-complex-comprehension
  def _get_variables_to_factors(
      self,) -> OrderedDict[FrozenSet[Any], enum.EnumFactor]:
    """Function that generates a dictionary mapping set of connected variables to factors.

    This function is only called on demand when the user requires it.

    Returns:
      A dictionary mapping all possible set of connected variables to different
      factors.
    """
    variables_to_factors = collections.OrderedDict([(
        frozenset(variables_for_factor),
        enum.EnumFactor(
            variables=variables_for_factor,
            factor_configs=self.factor_configs,
            log_potentials=np.array(self.log_potentials)[ii],
        ),
    ) for ii, variables_for_factor in enumerate(self.variables_for_factors)])
    return variables_to_factors

  def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Function that turns meaningful structured data into a flat data array for internal use.

    Args:
      data: Meaningful structured data. Array of shape (num_val_configs,) (for
        shared log potentials) or (num_factors, num_val_configs) (for log
        potentials) or (num_factors, num_edge_states) (for ftov messages).

    Returns:
      A flat jnp.array for internal use

    Raises:
      ValueError: if data is not of the right shape.
    """
    num_factors = len(self.factors)
    factor_edges_num_states = sum(
        [variable[1] for variable in self.variables_for_factors[0]])
    if (data.shape != (num_factors, self.factor_configs.shape[0]) and
        data.shape != (num_factors, factor_edges_num_states) and data.shape !=
        (self.factor_configs.shape[0],)):
      raise ValueError("data should be of shape"
                       f" {(num_factors, self.factor_configs.shape[0])} or"
                       f" {(num_factors, factor_edges_num_states)} or"
                       f" {(self.factor_configs.shape[0],)}. Got {data.shape}.")

    if data.shape == (self.factor_configs.shape[0],):
      flat_data = jnp.tile(data, num_factors)
    else:
      flat_data = jax.device_put(data).flatten()

    return flat_data

  def unflatten(
      self,
      flat_data: Union[np.ndarray, jnp.ndarray],
  ) -> Union[np.ndarray, jnp.ndarray]:
    """Function that recovers meaningful structured data from internal flat data array.

    Args:
      flat_data: Internal flat data array.

    Returns:
      Meaningful structured data.
      Array of shape (num_val_configs,) (for shared log potentials)
      or (num_factors, num_val_configs) (for log potentials)
      or (num_factors, num_edge_states) (for ftov messages).

    Raises:
      ValueError if:
        (1) flat_data is not a 1D array
        (2) flat_data is not of the right shape
    """
    if flat_data.ndim != 1:
      raise ValueError(
          f"Can only unflatten 1D array. Got a {flat_data.ndim}D array.")

    num_factors = len(self.factors)
    factor_edges_num_states = sum(
        [variable[1] for variable in self.variables_for_factors[0]])
    if flat_data.size == num_factors * self.factor_configs.shape[0]:
      data = flat_data.reshape((num_factors, self.factor_configs.shape[0]),)
    elif flat_data.size == num_factors * np.sum(factor_edges_num_states):
      data = flat_data.reshape((num_factors, np.sum(factor_edges_num_states)))
    else:
      raise ValueError("flat_data should be compatible with shape"
                       f" {(num_factors, self.factor_configs.shape[0])} or"
                       f" {(num_factors, np.sum(factor_edges_num_states))}. Got"
                       f" {flat_data.shape}.")
    return data


@dataclasses.dataclass(frozen=True, eq=False)
class PairwiseFactorGroup(fgroup.FactorGroup):
  """Class to represent a group of EnumFactors where each factor connects to two different variables.

  All factors in the group are assumed to be such that all possible
  configurations of the two variables are valid.
  The associated log potentials can however be different across factors.

  Attributes:
    log_potential_matrix: Optional 2D array of shape (num_states1, num_states2)
      or 3D array of shape (num_factors, num_states1, num_states2) where
      num_states1 and num_states2 are the number of states of the first and
      second variables involved in each factor.
      If 2D, the log potentials are copied for each factor of the group.
      If 3D, it specifies the log potentials of each factor.
      If None, the log potential are initialized to uniform 0.
    factor_type: Factor type shared by all the Factors in the FactorGroup.

  Raises:
    ValueError if:
      (1) The specified log_potential_matrix is not a 2D or 3D array.
      (2) The dtype of the potential array is not float
      (3) Some pairwise factors connect to less or more than 2 variables.
      (4) The specified log_potential_matrix does not match the number of
      factors.
      (5) The specified log_potential_matrix does not match the number of
      variable states of the
          variables in the factors.
  """

  log_potential_matrix: Optional[np.ndarray] = None  #: :meta private:
  factor_type: Type[factor.Factor] = dataclasses.field(
      init=False,
      default=enum.EnumFactor,
  )  #: :meta private:

  def __post_init__(self):
    super().__post_init__()

    if self.log_potential_matrix is None:
      log_potential_matrix = np.zeros((
          self.variables_for_factors[0][0][1],
          self.variables_for_factors[0][1][1],
      ))
    else:
      log_potential_matrix = self.log_potential_matrix

    if not (log_potential_matrix.ndim == 2 or log_potential_matrix.ndim == 3):
      raise ValueError(
          "log_potential_matrix should be either a 2D array, specifying shared"
          " parameters for all pairwise factors, or 3D array, specifying"
          " parameters for individual pairwise factors. Got a"
          f" {log_potential_matrix.ndim}D log_potential_matrix array.")

    if not np.issubdtype(log_potential_matrix.dtype, np.floating):
      raise ValueError("Potential matrix should be floats. Got"
                       f" {self.log_potential_matrix.dtype}.")

    if log_potential_matrix.ndim == 3 and log_potential_matrix.shape[0] != len(
        self.variables_for_factors):
      raise ValueError(
          "Expected log_potential_matrix for"
          f" {len(self.variables_for_factors)} factors. Got"
          f" log_potential_matrix for {log_potential_matrix.shape[0]} factors.")

    log_potential_shape = log_potential_matrix.shape[-2:]
    for variables_for_factor in self.variables_for_factors:
      if len(variables_for_factor) != 2:
        raise ValueError(
            "All pairwise factors should connect to exactly 2 variables. Got a"
            f" factor connecting to {len(variables_for_factor)} variables"
            f" ({variables_for_factor}).")

      factor_num_configs = (
          variables_for_factor[0][1],
          variables_for_factor[1][1],
      )
      if log_potential_shape != factor_num_configs:
        raise ValueError(
            f"The specified pairwise factor {variables_for_factor} (with"
            f" {factor_num_configs}configurations) does not match the specified"
            " log_potential_matrix (with"
            f" {log_potential_shape} configurations).")
    object.__setattr__(self, "log_potential_matrix", log_potential_matrix)

    factor_configs = (
        np.mgrid[:log_potential_matrix.shape[-2], :log_potential_matrix
                 .shape[-1]].transpose((1, 2, 0)).reshape((-1, 2)))
    object.__setattr__(self, "factor_configs", factor_configs)

    log_potential_matrix = np.broadcast_to(
        log_potential_matrix,
        (len(self.variables_for_factors),) + log_potential_matrix.shape[-2:],
    )
    log_potentials = np.empty((self.num_factors, self.factor_configs.shape[0]))
    _compute_log_potentials(
        log_potentials,
        log_potential_matrix,
        self.factor_configs,
    )
    object.__setattr__(self, "log_potentials", log_potentials)

  # pylint: disable=g-complex-comprehension
  def _get_variables_to_factors(
      self,) -> OrderedDict[FrozenSet[Any], enum.EnumFactor]:
    """Function that generates a dictionary mapping set of connected variables to factors.

    This function is only called on demand when the user requires it.

    Returns:
      A dictionary mapping all possible set of connected variables to factors.
    """
    variables_to_factors = collections.OrderedDict([(
        frozenset(variable_for_factor),
        enum.EnumFactor(
            variables=variable_for_factor,
            factor_configs=self.factor_configs,
            log_potentials=self.log_potentials[ii],
        ),
    ) for ii, variable_for_factor in enumerate(self.variables_for_factors)])
    return variables_to_factors

  def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Function that turns meaningful structured data into a flat data array for internal use.

    Args:
      data: Meaningful structured data. Should be an array of shape
        (num_factors, var0_num_states, var1_num_states) (for log potential
        matrices) or (num_factors, var0_num_states + var1_num_states) (for ftov
        messages) or (var0_num_states, var1_num_states) (for shared log
        potential matrix).

    Returns:
      A flat jnp.array for internal use
    """
    assert isinstance(self.log_potential_matrix, np.ndarray)
    num_factors = len(self.factors)
    if (data.shape != (num_factors,) + self.log_potential_matrix.shape[-2:] and
        data.shape !=
        (num_factors, np.sum(self.log_potential_matrix.shape[-2:])) and
        data.shape != self.log_potential_matrix.shape[-2:]):
      raise ValueError(
          "data should be of shape"
          f" {(num_factors,) + self.log_potential_matrix.shape[-2:]} or"
          f" {(num_factors, np.sum(self.log_potential_matrix.shape[-2:]))} or"
          f" {self.log_potential_matrix.shape[-2:]}. Got {data.shape}.")

    if data.shape == self.log_potential_matrix.shape[-2:]:
      flat_data = jnp.tile(jax.device_put(data).flatten(), num_factors)
    else:
      flat_data = jax.device_put(data).flatten()

    return flat_data

  def unflatten(
      self, flat_data: Union[np.ndarray,
                             jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    """Function that recovers meaningful structured data from internal flat data array.

    Args:
      flat_data: Internal flat data array.

    Returns:
      Meaningful structured data. Should be an array of shape
      (num_factors, var0_num_states, var1_num_states) (for log potential
      matrices)
      or (num_factors, var0_num_states + var1_num_states) (for ftov messages)
      or (var0_num_states, var1_num_states) (for shared log potential matrix).

    Raises:
        ValueError if:
            (1) flat_data is not a 1D array
            (2) flat_data is not of the right shape
    """
    if flat_data.ndim != 1:
      raise ValueError(
          f"Can only unflatten 1D array. Got a {flat_data.ndim}D array.")

    assert isinstance(self.log_potential_matrix, np.ndarray)
    num_factors = len(self.factors)
    if flat_data.size == num_factors * np.product(
        self.log_potential_matrix.shape[-2:]):
      data = flat_data.reshape((num_factors,) +
                               self.log_potential_matrix.shape[-2:])
    elif flat_data.size == num_factors * np.sum(
        self.log_potential_matrix.shape[-2:]):
      data = flat_data.reshape(
          (num_factors, np.sum(self.log_potential_matrix.shape[-2:])))
    else:
      raise ValueError(
          "flat_data should be compatible with shape"
          f" {(num_factors,) + self.log_potential_matrix.shape[-2:]} or"
          f" {(num_factors, np.sum(self.log_potential_matrix.shape[-2:]))}. Got"
          f" {flat_data.shape}.")

    return data


# pylint: disable=g-doc-args
@nb.jit(parallel=False, cache=True, fastmath=True, nopython=True)
def _compute_log_potentials(
    log_potentials: np.ndarray,
    log_potential_matrix: np.ndarray,
    factor_configs: np.ndarray,
):
  """Fast numba computation of the log_potentials of a PairwiseFactorGroup.

  log_potentials is updated in-place.
  """

  for config_idx in range(factor_configs.shape[0]):
    aux = log_potential_matrix[:, factor_configs[config_idx, 0],
                               factor_configs[config_idx, 1]]
    log_potentials[:, config_idx] = aux
