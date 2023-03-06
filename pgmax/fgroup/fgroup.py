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

"""A module containing the base classes for factor groups in a factor graph."""

import dataclasses
import functools
import inspect
from typing import Any, FrozenSet, List, Mapping, OrderedDict, Sequence, Tuple, Type, Union

import jax.numpy as jnp
import numpy as np
from pgmax import factor
from pgmax.utils import cached_property


@functools.total_ordering
@dataclasses.dataclass(frozen=True, eq=False)
class FactorGroup:
  """Class to represent a group of Factors.

  Attributes:
    variables_for_factors: A list of list of variables. Each list within the
      outer list contains the variables connected to a Factor. The same variable
      can be connected to multiple Factors.
    factor_configs: Optional array containing an explicit enumeration of all
      valid configurations
    log_potentials: Array of log potentials.
    factor_type: Factor type shared by all the Factors in the FactorGroup.

  Raises:
      ValueError: if the FactorGroup does not contain a Factor
  """

  variables_for_factors: Sequence[List[Tuple[int, int]]]
  factor_configs: np.ndarray = dataclasses.field(init=False)
  log_potentials: np.ndarray = dataclasses.field(
      init=False,
      default=np.empty((0,)),
  )
  factor_type: Type[Any] = dataclasses.field(init=False)

  def __post_init__(self):
    if not self.variables_for_factors:
      raise ValueError("Cannot create a FactorGroup with no Factor.")

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __lt__(self, other):
    return hash(self) < hash(other)

  def __getitem__(self, variables: Sequence[Tuple[int, int]]) -> Any:
    """Function to query individual factors in the factor group.

    Args:
      variables: a set of variables, used to query an individual factor in the
        factor group involving this set of variables

    Returns:
      A queried individual factor

    Raises:
      ValueError: if the queried factor is not present in the factor group
    """
    variables = frozenset(variables)
    if variables not in self._variables_to_factors:
      raise ValueError(
          f"The queried factor connected to the set of variables {variables} is"
          " not present in the factor group."
      )
    return self._variables_to_factors[variables]

  @cached_property
  def factor_sizes(self) -> np.ndarray:
    """Computes the factor sizes.

    Returns:
        Array of the different factor sizes.
    """
    return np.array(
        [
            len(variables_for_factor)
            for variables_for_factor in self.variables_for_factors
        ]
    )

  @cached_property
  def factor_edges_num_states(self) -> np.ndarray:
    """Computes an array concatenating the number of states for the variables connected to each Factor of the FactorGroup.

    Each variable will appear once for each Factor it connects to.

    Returns:
      Array with the the number of states for the variables connected to
      each Factor.
    """
    factor_edges_num_states = np.empty(
        shape=(self.factor_sizes.sum(),),
        dtype=int,
    )

    idx = 0
    for variables_for_factor in self.variables_for_factors:
      for variable in variables_for_factor:
        factor_edges_num_states[idx] = variable[1]
        idx += 1
    return factor_edges_num_states

  @cached_property
  def _variables_to_factors(
      self,
  ) -> Mapping[FrozenSet[Sequence[List[Tuple[int, int]]]], factor.Factor]:
    """Function to compile potential array for the factor group.

    This function is only called on demand when the user requires it.

    Returns:
      A dictionnary mapping set of connected variables to the factors
    """
    return self._get_variables_to_factors()

  @cached_property
  def factor_group_log_potentials(self) -> np.ndarray:
    """Flattened array of log potentials."""
    return self.log_potentials.flatten()

  @cached_property
  def factors(self) -> Tuple[factor.Factor, ...]:
    """Returns all factors in the factor group.

    This function is only called on demand when the user requires it.
    """
    return tuple(self._variables_to_factors.values())

  @cached_property
  def num_factors(self) -> int:
    """Returns the number of factors in the FactorGroup."""
    return len(self.variables_for_factors)

  def _get_variables_to_factors(
      self,
  ) -> OrderedDict[FrozenSet[Sequence[List[Tuple[int, int]]]], Any]:
    """Function that generates a dictionary mapping names to factors.

    This function is only called on demand when the user requires it.

    Returns:
      A dictionary mapping all possible names to different factors.
    """
    raise NotImplementedError(
        "Please subclass the FactorGroup class and override this method"
    )

  def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Function that turns meaningful structured data into a flat data array for internal use.

    Args:
      data: Meaningful structured data.

    Returns:
      A flat jnp.array for internal use
    """
    raise NotImplementedError(
        "Please subclass the FactorGroup class and override this method"
    )

  def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
    """Function that recovers meaningful structured data from internal flat data array.

    Args:
      flat_data: Internal flat data array.

    Returns:
      Meaningful structured data.
    """
    raise NotImplementedError(
        "Please subclass the FactorGroup class and override this method"
    )

  def compile_wiring(
      self,
      vars_to_starts: Mapping[Tuple[int, int], int],
  ) -> Any:
    """Compile an efficient wiring for the FactorGroup.

    Args:
      vars_to_starts: A dictionary that maps variables to their global starting
        indices For an n-state variable, a global start index of m means the
        global indices of its n variable states are m, m + 1, ..., m + n - 1

    Returns:
      Wiring for the FactorGroup
    """
    compile_wiring_arguments = inspect.getfullargspec(
        self.factor_type.compile_wiring
    ).args
    compile_wiring_arguments.remove("vars_to_starts")
    compile_wiring_arguments = {
        key: getattr(self, key) for key in compile_wiring_arguments
    }

    wiring = self.factor_type.compile_wiring(
        vars_to_starts=vars_to_starts, **compile_wiring_arguments
    )
    return wiring


@dataclasses.dataclass(frozen=True, eq=False)
class SingleFactorGroup(FactorGroup):
  """Class to represent a FactorGroup with a single factor.

  For internal use only. Should not be directly used to add FactorGroups to a
  factor graph.

  Attributes:
    single_factor: the single factor in the SingleFactorGroup
  """

  single_factor: factor.Factor

  def __post_init__(self):
    super().__post_init__()

    if len(self.variables_for_factors) != 1:
      raise ValueError(
          "SingleFactorGroup should only contain one factor. Got"
          f" {len(self.variables_for_factors)}"
      )

    object.__setattr__(self, "factor_type", type(self.single_factor))
    compile_wiring_arguments = inspect.getfullargspec(
        self.factor_type.compile_wiring
    ).args
    compile_wiring_arguments.remove("vars_to_starts")
    for key in compile_wiring_arguments:
      if not hasattr(self, key):
        object.__setattr__(self, key, getattr(self.single_factor, key))

    single_factor_log_potentials = getattr(self.single_factor, "log_potentials")
    if single_factor_log_potentials.shape[0] > 0:
      # Fgroup log potentials is of shape (num_factors, num_configs)
      single_factor_log_potentials = single_factor_log_potentials[None]
    object.__setattr__(self, "log_potentials", single_factor_log_potentials)

  def _get_variables_to_factors(
      self,
  ) -> OrderedDict[FrozenSet[Sequence[List[Tuple[int, int]]]], factor.Factor]:
    """Function that generates a dictionary mapping names to factors.

    Returns:
      A dictionary mapping all possible names to different factors.
    """
    return OrderedDict(
        [(frozenset(self.variables_for_factors[0]), self.single_factor)]
    )

  def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Function that turns meaningful structured data into a flat data array for internal use.

    Args:
      data: Meaningful structured data.

    Returns:
      A flat jnp.array for internal use
    """
    raise NotImplementedError(
        "SingleFactorGroup does not support vectorized factor operations."
    )

  def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
    """Function that recovers meaningful structured data from internal flat data array.

    Args:
      flat_data: Internal flat data array.

    Returns:
      Meaningful structured data.
    """
    raise NotImplementedError(
        "SingleFactorGroup does not support vectorized factor operations."
    )
