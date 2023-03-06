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

"""A module containing the base classes for factors in a factor graph."""

import dataclasses
from typing import Any, Dict, Hashable, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Wiring:
  """Wiring for factors.

  Attributes:
    var_states_for_edges: Array of shape (num_edge_states, 3)
      For each edge state:
      var_states_for_edges[ii, 0] contains its global variable state index
      var_states_for_edges[ii, 1] contains its global edge index
      var_states_for_edges[ii, 2] contains its global factor index
  """
  var_states_for_edges: Union[np.ndarray, jnp.ndarray]

  def __post_init__(self):
    for field in self.__dataclass_fields__:
      if isinstance(getattr(self, field), np.ndarray):
        getattr(self, field).flags.writeable = False

  def get_inference_arguments(self):
    """Return the list of arguments to run BP for this Wiring."""
    raise NotImplementedError(
        "Please subclass the Wiring class and override this method."
    )

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

  @staticmethod
  def compute_energy(
      wiring: Wiring,
      edge_states_one_hot_decoding: jnp.ndarray,
      log_potentials: jnp.ndarray,
  ) -> float:
    """Returns the contribution to the energy of several Factors of same type.

    Args:
      wiring: Wiring
      edge_states_one_hot_decoding: Array of shape (num_edge_states,)
        Flattened array of one-hot decoding of the edge states connected to the
        Factors
      log_potentials: Array of log potentials
    """
    raise NotImplementedError(
        "Please subclass the Factor class and override this method."
    )

  @staticmethod
  def compute_factor_energy(
      variables: List[Hashable],
      vars_to_map_states: Dict[Hashable, Any],
  ) -> Tuple[float, Dict[Hashable, Any]]:
    """Returns the contribution to the energy of a single Factor.

    Args:
      variables: List of variables connected by the Factor
      vars_to_map_states: A dictionary mapping each individual variable to
        its MAP state.
    """
    raise NotImplementedError(
        "Please subclass the Factor class and override this method"
    )


def concatenate_var_states_for_edges(
    list_var_states_for_edges: Sequence[np.ndarray],
) -> np.ndarray:
  """Concatenate a list of var_states_for_edges.

  Args:
    list_var_states_for_edges: A list of var_states_for_edges

  Returns:
    Concatenated var_states_for_edges

  Raises: ValueError if
    (1) list_var_states_for_edges is None
    (2) one of the list_var_states_for_edges entry is None
  """
  if list_var_states_for_edges is None:
    raise ValueError("list_var_states_for_edges cannot be None")

  # Remove empty
  list_var_states_for_edges_wo_empty = []
  for var_states_for_edges in list_var_states_for_edges:
    # var_states_for_edges can be empty but cannot be None
    if var_states_for_edges is None:
      raise ValueError("var_states_for_edges cannot be None")
    if var_states_for_edges.shape[0] > 0:
      list_var_states_for_edges_wo_empty.append(var_states_for_edges)

  num_edges_cumsum = np.insert(
      np.cumsum(
          [
              var_states_for_edges[-1, 1] + 1
              for var_states_for_edges in list_var_states_for_edges_wo_empty
          ]
      ),
      0,
      0,
  )[:-1]

  num_factors_cumsum = np.insert(
      np.cumsum(
          [
              var_states_for_edges[-1, 2] + 1
              for var_states_for_edges in list_var_states_for_edges_wo_empty
          ]
      ),
      0,
      0,
  )[:-1]

  var_states_for_edges_w_offsets = []
  for idx, var_states_for_edges in enumerate(
      list_var_states_for_edges_wo_empty
  ):
    var_states_for_edges_w_offsets.append(
        var_states_for_edges
        + np.array(
            [[0, num_edges_cumsum[idx], num_factors_cumsum[idx]]],
            dtype=int,
        )
    )
  return np.concatenate(var_states_for_edges_w_offsets, axis=0)


# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
@nb.jit(parallel=False, cache=True, fastmath=True, nopython=True)
def compile_var_states_for_edges_numba(
    var_states_for_edges: np.ndarray,
    num_edges_states_cumsum: np.ndarray,
    first_var_state_by_edges: np.ndarray,
    factor_indices: np.ndarray,
):
  """Fast numba computation of the var_states_for_edges of a Wiring.

  var_states_for_edges is updated in-place
  """
  for edge_idx in nb.prange(num_edges_states_cumsum.shape[0] - 1):
    start_edge_state_idx, end_edge_state_idx = (
        num_edges_states_cumsum[edge_idx],
        num_edges_states_cumsum[edge_idx + 1],
    )
    # Variable states for each edge state
    var_states_for_edges[
        start_edge_state_idx:end_edge_state_idx, 0
    ] = first_var_state_by_edges[edge_idx] + np.arange(
        end_edge_state_idx - start_edge_state_idx
    )
    # Edge index for each edge state
    var_states_for_edges[start_edge_state_idx:end_edge_state_idx, 1] = edge_idx
    # Factor index for each edge state
    var_states_for_edges[
        start_edge_state_idx:end_edge_state_idx, 2
    ] = factor_indices[edge_idx]
  return var_states_for_edges
