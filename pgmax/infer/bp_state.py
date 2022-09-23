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
"""A module defining container classes for belief propagation states."""

import dataclasses
import functools
from typing import Any, Dict, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from pgmax import fgraph
from pgmax import fgroup
from pgmax.utils import NEG_INF


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class BPArrays:
  """Container for the relevant flat arrays used in belief propagation.

  Attributes:
    log_potentials: Flat log potentials array.
    ftov_msgs: Flat factor to variable messages array.
    evidence: Flat evidence array.
  """

  log_potentials: Union[np.ndarray, jnp.ndarray]
  ftov_msgs: Union[np.ndarray, jnp.ndarray]
  evidence: Union[np.ndarray, jnp.ndarray]

  def __post_init__(self):
    for field in self.__dataclass_fields__:
      if isinstance(getattr(self, field), np.ndarray):
        getattr(self, field).flags.writeable = False

  def tree_flatten(self):
    return jax.tree_util.tree_flatten(dataclasses.asdict(self))

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data.unflatten(children))


@functools.partial(jax.jit, static_argnames="fg_state")
def update_log_potentials(
    log_potentials: jnp.ndarray,
    updates: Dict[Any, jnp.ndarray],
    fg_state: fgraph.FactorGraphState,
) -> jnp.ndarray:
  """Function to update log_potentials.

  Args:
    log_potentials: A flat jnp array containing log_potentials.
    updates: A dictionary containing updates for log_potentials
    fg_state: Factor graph state

  Returns:
    A flat jnp array containing updated log_potentials.

  Raises: ValueError if
    (1) Provided log_potentials shape does not match the expected
        log_potentials shape.
    (2) Provided name is not valid for log_potentials updates.
  """
  # Clip updates to not have infinite values
  updates = jax.tree_util.tree_map(
      lambda x: jnp.clip(x, NEG_INF, -NEG_INF), updates
  )

  for factor_group, data in updates.items():
    if factor_group in fg_state.factor_group_to_potentials_starts:
      flat_data = factor_group.flatten(data)
      if flat_data.shape != factor_group.factor_group_log_potentials.shape:
        raise ValueError(
            "Expected log potentials shape"
            f" {factor_group.factor_group_log_potentials.shape} for factor"
            f" group. Got incompatible data shape {data.shape}."
        )

      start = fg_state.factor_group_to_potentials_starts[factor_group]
      log_potentials = log_potentials.at[
          start : start + flat_data.shape[0]
      ].set(flat_data)
    else:
      raise ValueError("Invalid FactorGroup for log potentials updates.")

  return log_potentials


@dataclasses.dataclass(frozen=True, eq=False)
class LogPotentials:
  """Class for storing and manipulating log potentials.

  Attributes:
    fg_state: Factor graph state
    value: Optionally specify an initial value
  Raises: ValueError if provided value shape does not match the expected
    log_potentials shape.
  """

  fg_state: fgraph.FactorGraphState
  value: Optional[np.ndarray] = None

  def __post_init__(self):
    if self.value is None:
      object.__setattr__(self, "value", self.fg_state.log_potentials)
    else:
      if self.value.shape != self.fg_state.log_potentials.shape:
        raise ValueError(
            "Expected log potentials shape"
            f" {self.fg_state.log_potentials.shape}. Got {self.value.shape}."
        )

      object.__setattr__(self, "value", self.value)

  def __getitem__(self, factor_group: fgroup.FactorGroup) -> np.ndarray:
    """Function to query log potentials for a FactorGroup.

    Args:
      factor_group: Queried FactorGroup

    Returns:
      The queried log potentials.
    """
    value = cast(np.ndarray, self.value)
    if factor_group in self.fg_state.factor_group_to_potentials_starts:
      start = self.fg_state.factor_group_to_potentials_starts[factor_group]
      log_potentials = value[
          start : start + factor_group.factor_group_log_potentials.shape[0]
      ]
    else:
      raise ValueError("Invalid FactorGroup queried to access log potentials.")
    return log_potentials

  def __setitem__(
      self,
      factor_group: fgroup.FactorGroup,
      data: Union[np.ndarray, jnp.ndarray],
  ):
    """Set the log potentials for a FactorGroup.

    Args:
      factor_group: FactorGroup
      data: Array containing the log potentials for the FactorGroup
    """
    object.__setattr__(
        self,
        "value",
        np.asarray(
            update_log_potentials(
                jax.device_put(self.value),
                {factor_group: jax.device_put(data)},
                self.fg_state,
            )
        ),
    )


@functools.partial(jax.jit, static_argnames="fg_state")
def update_ftov_msgs(
    ftov_msgs: jnp.ndarray,
    updates: Dict[Any, jnp.ndarray],
    fg_state: fgraph.FactorGraphState,
) -> jnp.ndarray:
  """Function to update ftov_msgs.

  Args:
    ftov_msgs: A flat jnp array containing ftov_msgs.
    updates: A dictionary containing updates for ftov_msgs
    fg_state: Factor graph state

  Returns:
    A flat jnp array containing updated ftov_msgs.

  Raises: ValueError if:
    (1) provided ftov_msgs shape does not match the expected ftov_msgs shape.
    (2) provided variable is not in the FactorGraph.
  """
  # Clip updates to not have infinite values
  updates = jax.tree_util.tree_map(
      lambda x: jnp.clip(x, NEG_INF, -NEG_INF), updates
  )

  for variable, data in updates.items():
    if variable in fg_state.vars_to_starts:
      if data.shape != (variable[1],):
        raise ValueError(
            f"Given belief shape {data.shape} does not match expected "
            f"shape {(variable[1],)} for variable {variable}."
        )

      var_states_for_edges = np.concatenate(
          [
              wiring_by_type.var_states_for_edges
              for wiring_by_type in fg_state.wiring.values()
          ]
      )

      starts = np.nonzero(
          var_states_for_edges == fg_state.vars_to_starts[variable]
      )[0]
      for start in starts:
        ftov_msgs = ftov_msgs.at[start : start + variable[1]].set(
            data / starts.shape[0]
        )
    else:
      raise ValueError("Provided variable is not in the FactorGraph")
  return ftov_msgs


@dataclasses.dataclass(frozen=True, eq=False)
class FToVMessages:
  """Class for storing and manipulating factor to variable messages.

  Attributes:
    fg_state: Factor graph state
    value: Optionally specify initial value for ftov messages
  Raises: ValueError if provided value does not match expected ftov messages
    shape.
  """

  fg_state: fgraph.FactorGraphState
  value: Optional[np.ndarray] = None

  def __post_init__(self):
    if self.value is None:
      object.__setattr__(
          self, "value", np.zeros(self.fg_state.total_factor_num_states)
      )
    else:
      if self.value.shape != (self.fg_state.total_factor_num_states,):
        raise ValueError(
            "Expected messages shape"
            f" {(self.fg_state.total_factor_num_states,)}. Got"
            f" {self.value.shape}."
        )

      object.__setattr__(self, "value", self.value)

  def __setitem__(
      self,
      variable: Tuple[int, int],
      data: Union[np.ndarray, jnp.ndarray],
  ) -> None:
    """Spreading beliefs at a variable to all connected Factors.

    Args:
      variable: Variable queried
      data: An array containing the beliefs to be spread uniformly across all
        factors to variable messages involving this variable.
    """

    object.__setattr__(
        self,
        "value",
        np.asarray(
            update_ftov_msgs(
                jax.device_put(self.value),
                {variable: jax.device_put(data)},
                self.fg_state,
            )
        ),
    )


@functools.partial(jax.jit, static_argnames="fg_state")
def update_evidence(
    evidence: jnp.ndarray,
    updates: Dict[Any, jnp.ndarray],
    fg_state: fgraph.FactorGraphState,
) -> jnp.ndarray:
  """Function to update evidence.

  Args:
    evidence: A flat jnp array containing evidence.
    updates: A dictionary containing updates for evidence
    fg_state: Factor graph state

  Returns:
    A flat jnp array containing updated evidence.
  """
  # Clip updates to not have infinite values
  updates = jax.tree_util.tree_map(
      lambda x: jnp.clip(x, NEG_INF, -NEG_INF), updates
  )

  for name, data in updates.items():
    # Name is a variable_group or a variable
    if name in fg_state.variable_groups:
      first_variable = name.variables[0]
      start_index = fg_state.vars_to_starts[first_variable]
      flat_data = name.flatten(data)
      evidence = evidence.at[
          start_index : start_index + flat_data.shape[0]
      ].set(flat_data)
    elif name in fg_state.vars_to_starts:
      start_index = fg_state.vars_to_starts[name]
      evidence = evidence.at[start_index : start_index + name[1]].set(data)
    else:
      raise ValueError(
          "Got evidence for a variable or a VarGroup not in the FactorGraph!"
      )
  return evidence


@dataclasses.dataclass(frozen=True, eq=False)
class Evidence:
  """Class for storing and manipulating evidence.

  Attributes:
    fg_state: Factor graph state
    value: Optionally specify initial value for evidence
  Raises: ValueError if provided value does not match expected evidence shape.
  """

  fg_state: fgraph.FactorGraphState
  value: Optional[np.ndarray] = None

  def __post_init__(self):
    if self.value is None:
      object.__setattr__(self, "value", np.zeros(self.fg_state.num_var_states))
    else:
      if self.value.shape != (self.fg_state.num_var_states,):
        raise ValueError(
            f"Expected evidence shape {(self.fg_state.num_var_states,)}. "
            f"Got {self.value.shape}."
        )

      object.__setattr__(self, "value", self.value)

  def __getitem__(self, variable: Tuple[int, int]) -> np.ndarray:
    """Function to query evidence for a variable.

    Args:
      variable: Variable queried

    Returns:
      Evidence for the queried variable
    """
    value = cast(np.ndarray, self.value)
    start = self.fg_state.vars_to_starts[variable]
    evidence = value[start : start + variable[1]]
    return evidence

  def __setitem__(
      self,
      name: Any,
      data: np.ndarray,
  ) -> None:
    """Function to update the evidence for variables.

    Args:
      name: The name of a variable group or a single variable. If name is the
        name of a variable group, updates are derived by using the variable
        group to flatten the data. If name is the name of a variable, data
        should be of an array shape (num_states,) If name is None, updates are
        derived by using self.fg_state.variable_groups to flatten the data.
      data: Array containing the evidence updates.
    """
    object.__setattr__(
        self,
        "value",
        np.asarray(
            update_evidence(
                jax.device_put(self.value),
                {name: jax.device_put(data)},
                self.fg_state,
            ),
        ),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class BPState:
  """Container class for belief propagation states, including log potentials, ftov messages and evidence (unary log potentials).

  Attributes:
    log_potentials: log potentials of the model
    ftov_msgs: factor to variable messages
    evidence: evidence (unary log potentials) for variables.
    fg_state: associated factor graph state
  Raises: ValueError if log_potentials, ftov_msgs or evidence are not derived
    from the same Factor graph state.
  """

  log_potentials: LogPotentials
  ftov_msgs: FToVMessages
  evidence: Evidence

  def __post_init__(self):
    if (self.log_potentials.fg_state != self.ftov_msgs.fg_state) or (
        self.ftov_msgs.fg_state != self.evidence.fg_state
    ):
      raise ValueError(
          "log_potentials, ftov_msgs and evidence should be derived from the"
          " same fg_state."
      )

  @property
  def fg_state(self) -> fgraph.FactorGraphState:
    return self.log_potentials.fg_state
