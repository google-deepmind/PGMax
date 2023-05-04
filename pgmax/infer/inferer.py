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

"""Shared context classes for the inference methods."""

import dataclasses
import functools
from typing import Any, Callable, Dict, Hashable, Optional, Sequence
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from pgmax import factor
from pgmax import vgroup
from pgmax.factor import FAC_TO_VAR_UPDATES
from pgmax.infer import bp_state as bpstate
from pgmax.infer.bp_state import BPArrays
from pgmax.infer.bp_state import BPState
from pgmax.infer.bp_state import Evidence
from pgmax.infer.bp_state import FToVMessages
from pgmax.infer.bp_state import LogPotentials


@dataclasses.dataclass(frozen=True, eq=False)
class Inferer:
  """Inferer pure functions.

  Attributes:
    init: Function to create log_potentials, ftov_msgs and evidence.
    update: Function to update log_potentials, ftov_msgs and evidence.
    to_bp_state: Function to reconstruct the BPState from a BPArrays.
    get_beliefs: Function to calculate beliefs from a BPArrays.
    run: Function to run inference.
  """

  init: Callable[..., BPArrays]
  update: Callable[..., BPArrays]
  to_bp_state: Callable[..., BPArrays]
  get_beliefs: Callable[..., Dict[Hashable, Any]]
  run: Callable[..., BPArrays]


@dataclasses.dataclass(frozen=True, eq=False)
class InfererContext:
  """Shared inference context for the different inferers.

  Attributes:
    bp_state: Belief propagation state.
  """

  bp_state: BPState

  def __post_init__(self):
    if jax.lib.xla_bridge.get_backend().platform == "tpu":  # pragma: no cover
      warnings.warn(
          "PGMax is not optimized for the TPU backend. Please consider using"
          " GPUs!"
      )
    object.__setattr__(self, "wiring", self.bp_state.fg_state.wiring)
    object.__setattr__(
        self, "evidence_to_vars", self.bp_state.fg_state.evidence_to_vars
    )

    # Add offsets to the edges and factors indices of var_states_for_edges
    var_states_for_edges = factor.concatenate_var_states_for_edges(
        [
            self.wiring[factor_type].var_states_for_edges
            for factor_type in FAC_TO_VAR_UPDATES
        ]
    )
    object.__setattr__(
        self, "var_states_for_edge_states", var_states_for_edges[..., 0]
    )
    object.__setattr__(
        self, "edge_indices_for_edge_states", var_states_for_edges[..., 1]
    )
    object.__setattr__(
        self, "factor_indices_for_edge_states", var_states_for_edges[..., 2]
    )
    # Useful static quantities
    num_variables = int(self.bp_state.fg_state.evidence_to_vars[-1]) + 1
    num_edges = int(var_states_for_edges[-1, 1]) + 1
    num_factors = int(var_states_for_edges[-1, 2]) + 1
    object.__setattr__(self, "num_variables", num_variables)
    object.__setattr__(self, "num_edges", num_edges)
    object.__setattr__(self, "num_factors", num_factors)

    # Inference arguments per factor type
    inference_arguments = {}
    for factor_type in FAC_TO_VAR_UPDATES:
      this_inference_arguments = self.wiring[
          factor_type
      ].get_inference_arguments()
      inference_arguments[factor_type] = this_inference_arguments
    object.__setattr__(self, "inference_arguments", inference_arguments)

    object.__setattr__(
        self,
        "factor_type_to_msgs_range",
        self.bp_state.fg_state.factor_type_to_msgs_range,
    )
    object.__setattr__(
        self,
        "factor_type_to_potentials_range",
        self.bp_state.fg_state.factor_type_to_potentials_range,
    )

  def update(
      self,
      bp_arrays: Optional[BPArrays] = None,
      log_potentials_updates: Optional[Dict[Any, jnp.ndarray]] = None,
      ftov_msgs_updates: Optional[Dict[Any, jnp.ndarray]] = None,
      evidence_updates: Optional[Dict[Any, jnp.ndarray]] = None,
  ) -> BPArrays:
    """Returns a BPArrays with the updated log_potentials, ftov_msgs and evidence.

    Args:
      bp_arrays: Optional arrays of log_potentials, ftov_msgs, evidence.
      log_potentials_updates: Optional dictionary containing log_potentials
        updates.
      ftov_msgs_updates: Optional dictionary containing ftov_msgs updates.
      evidence_updates: Optional dictionary containing evidence updates.
    """
    if bp_arrays is not None:
      log_potentials = bp_arrays.log_potentials
      evidence = bp_arrays.evidence
      ftov_msgs = bp_arrays.ftov_msgs
    else:
      log_potentials = jax.device_put(self.bp_state.log_potentials.value)
      ftov_msgs = self.bp_state.ftov_msgs.value
      evidence = self.bp_state.evidence.value

    if log_potentials_updates is not None:
      log_potentials = bpstate.update_log_potentials(
          log_potentials,
          log_potentials_updates,
          self.bp_state.fg_state,
      )

    if ftov_msgs_updates is not None:
      ftov_msgs = bpstate.update_ftov_msgs(
          ftov_msgs,
          ftov_msgs_updates,
          self.bp_state.fg_state,
      )

    if evidence_updates is not None:
      evidence = bpstate.update_evidence(
          evidence, evidence_updates, self.bp_state.fg_state
      )

    return BPArrays(
        log_potentials=log_potentials,
        ftov_msgs=ftov_msgs,
        evidence=evidence,
    )

  def init(
      self,
      log_potentials_updates: Optional[Dict[Any, jnp.ndarray]] = None,
      ftov_msgs_updates: Optional[Dict[Any, jnp.ndarray]] = None,
      evidence_updates: Optional[Dict[Any, jnp.ndarray]] = None,
  ) -> BPArrays:
    """Returns a BPArrays with the initialized log_potentials, ftov_msgs and evidence.

    Args:
      log_potentials_updates: Optional dictionary containing log_potentials
        updates.
      ftov_msgs_updates: Optional dictionary containing ftov_msgs updates.
      evidence_updates: Optional dictionary containing evidence updates.
    """
    return self.update(
        bp_arrays=None,
        log_potentials_updates=log_potentials_updates,
        ftov_msgs_updates=ftov_msgs_updates,
        evidence_updates=evidence_updates,
    )

  def to_bp_state(self, bp_arrays: BPArrays) -> BPState:
    """Returns a BPState reconstructed from a BPArrays.

    Args:
      bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.
    """
    return BPState(
        log_potentials=LogPotentials(
            fg_state=self.bp_state.fg_state, value=bp_arrays.log_potentials
        ),
        ftov_msgs=FToVMessages(
            fg_state=self.bp_state.fg_state,
            value=bp_arrays.ftov_msgs,
        ),
        evidence=Evidence(
            fg_state=self.bp_state.fg_state,
            value=bp_arrays.evidence,
        ),
    )

  @functools.partial(jax.jit, static_argnames="self")
  def get_beliefs(self, bp_arrays: BPArrays) -> Dict[Hashable, Any]:
    """Returns the beliefs derived from a BPArrays.

    Args:
      bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.
    """
    flat_beliefs = (
        jax.device_put(bp_arrays.evidence)
        .at[jax.device_put(self.var_states_for_edge_states)]
        .add(bp_arrays.ftov_msgs)
    )
    return unflatten_beliefs(
        flat_beliefs, self.bp_state.fg_state.variable_groups
    )


def unflatten_beliefs(
    flat_beliefs: jnp.array, variable_groups: Sequence[vgroup.VarGroup]
) -> Dict[Hashable, Any]:
  """Returns unflattened beliefs from flat beliefs.

  Args:
    flat_beliefs: Flattened array of beliefs
    variable_groups: All the variable groups in the FactorGraph.
  """
  beliefs = {}
  start = 0
  for variable_group in variable_groups:
    num_states = variable_group.num_states
    assert isinstance(num_states, np.ndarray)
    length = num_states.sum()

    beliefs[variable_group] = variable_group.unflatten(
        flat_beliefs[start : start + length], True
    )
    start += length
  return beliefs


@jax.jit
def decode_map_states(beliefs: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
  """Returns the MAP states of several VarGroups given their beliefs.

  Args:
    beliefs: A dictionary containing the beliefs of the VarGroups.
  """
  # pylint: disable=g-long-lambda
  return jax.tree_util.tree_map(
      lambda x: jnp.argmax(x, axis=-1)
      if x.size > 0  # Deal with MAP state of zero-sized array
      else jnp.zeros(x.shape[:-1]),
      beliefs,
  )
