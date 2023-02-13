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

"""A module containing the base classes for factors in a factor graph."""

import dataclasses
from typing import List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Wiring:
  """Wiring for factors.

  Attributes:
    edges_num_states: Array of shape (num_edges,) Number of states for the
      variables connected to each edge
    var_states_for_edges: Array of shape (num_edge_states,) Global variable
      state indices for each edge state
  """

  edges_num_states: Union[np.ndarray, jnp.ndarray]
  var_states_for_edges: Union[np.ndarray, jnp.ndarray]

  def __post_init__(self):
    for field in self.__dataclass_fields__:
      if isinstance(getattr(self, field), np.ndarray):
        getattr(self, field).flags.writeable = False

  def tree_flatten(self):
    return jax.tree_util.tree_flatten(dataclasses.asdict(self))

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data.unflatten(children))


@dataclasses.dataclass(frozen=True, eq=False)
class Factor:
  """A factor.

  Attributes:
    variables: List of variables connected by the Factor. Each variable is
      represented by a tuple (variable hash, variable num_states)
    log_potentials: Array of log potentials

  Raises:
    NotImplementedError: If compile_wiring is not implemented
  """

  variables: List[Tuple[int, int]]
  log_potentials: np.ndarray

  def __post_init__(self):
    if not hasattr(self, "compile_wiring"):
      raise NotImplementedError(
          "Please implement compile_wiring in for your factor"
      )

  @staticmethod
  def concatenate_wirings(wirings: Sequence[Wiring]) -> Wiring:
    """Concatenate a list of Wirings.

    Args:
        wirings: A list of Wirings

    Returns:
        Concatenated Wiring
    """
    raise NotImplementedError(
        "Please subclass the Wiring class and override this method."
    )


# pylint: disable=g-doc-args
@nb.jit(parallel=False, cache=True, fastmath=True, nopython=True)
def compile_var_states_numba(
    var_states_for_edges: np.ndarray,
    num_states_cumsum: np.ndarray,
    var_states: np.ndarray,
):
  """Fast numba computation of the var_states_for_edges of a Wiring.

  var_states_for_edges is updated in-place
  """
  for variable_idx in nb.prange(num_states_cumsum.shape[0] - 1):
    start_variable, end_variable = (
        num_states_cumsum[variable_idx],
        num_states_cumsum[variable_idx + 1],
    )
    var_states_for_edges[start_variable:end_variable] = var_states[
        variable_idx
    ] + np.arange(end_variable - start_variable)
