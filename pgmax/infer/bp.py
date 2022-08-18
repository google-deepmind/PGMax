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
"""A module containing the core message-passing functions for belief propagation."""

import dataclasses
import functools
import inspect
from typing import Any, Callable, Dict, Hashable, Optional, Tuple
import warnings

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from pgmax.factor import FAC_TO_VAR_UPDATES
from pgmax.infer import bp_state as bpstate
from pgmax.infer import bp_utils
from pgmax.infer.bp_state import BPArrays
from pgmax.infer.bp_state import BPState
from pgmax.infer.bp_state import Evidence
from pgmax.infer.bp_state import FToVMessages
from pgmax.infer.bp_state import LogPotentials


@dataclasses.dataclass(frozen=True, eq=False)
class BeliefPropagation:
  """Belief propagation functions.

  Attributes:
    init: Function to create log_potentials, ftov_msgs and evidence.
      Args:
        log_potentials_updates: Optional dictionary of log_potentials updates.
        ftov_msgs_updates: Optional dictionary of ftov_msgs updates.
        evidence_updates: Optional dictionary of evidence updates.

      Returns:
        A BPArrays with the log_potentials, ftov_msgs and evidence.

    update: Function to update log_potentials, ftov_msgs and evidence.
      Args:
        bp_arrays: Optional arrays of log_potentials, ftov_msgs, evidence.
        log_potentials_updates: Optional dictionary of log_potentials updates.
        ftov_msgs_updates: Optional dictionary of ftov_msgs updates.
        evidence_updates: Optional dictionary of evidence updates.

      Returns:
        A BPArrays with the updated log_potentials, ftov_msgs and evidence.

    run_bp: Function to run belief propagation for num_iters with damping.
      Args:
        bp_arrays: Initial arrays of log_potentials, ftov_msgs, evidence.
        num_iters: Number of belief propagation iterations.
        damping: The damping factor to use for message updates between one
          timestep and the next.

      Returns:
        A BPArrays containing the updated ftov_msgs.

    to_bp_state: Function to reconstruct the BPState from a BPArrays.
      Args:
        bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.

      Returns:
        The reconstructed BPState

    get_beliefs: Function to calculate beliefs from a BPArrays.
      Args:
        bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.

      Returns:
        beliefs: Beliefs returned by belief propagation.
  """

  init: Callable[..., BPArrays]
  update: Callable[..., BPArrays]
  run_bp: Callable[..., BPArrays]
  to_bp_state: Callable[..., BPArrays]
  get_beliefs: Callable[..., Dict[Hashable, Any]]


# pylint: disable=invalid-name
def BP(bp_state: BPState, temperature: float = 0.0) -> BeliefPropagation:
  """Function for generating belief propagation functions.

  Args:
    bp_state: Belief propagation state.
    temperature: Temperature for loopy belief propagation. 1.0 corresponds to
      sum-product, 0.0 corresponds to max-product.

  Returns:
    Belief propagation functions.
  """
  if jax.lib.xla_bridge.get_backend().platform == "tpu":
    warnings.warn(
        "PGMax is not optimized for the TPU backend. Please consider using"
        " GPUs!"
    )

  wiring = bp_state.fg_state.wiring
  edges_num_states = np.concatenate(
      [
          wiring[factor_type].edges_num_states
          for factor_type in FAC_TO_VAR_UPDATES
      ]
  )
  max_msg_size = int(np.max(edges_num_states))

  var_states_for_edges = np.concatenate(
      [
          wiring[factor_type].var_states_for_edges
          for factor_type in FAC_TO_VAR_UPDATES
      ]
  )

  # Inference argumnets per factor type
  inference_arguments = {}
  for factor_type in FAC_TO_VAR_UPDATES:
    this_inference_arguments = inspect.getfullargspec(
        FAC_TO_VAR_UPDATES[factor_type]
    ).args
    this_inference_arguments.remove("vtof_msgs")
    this_inference_arguments.remove("log_potentials")
    this_inference_arguments.remove("temperature")
    this_inference_arguments = {
        key: getattr(wiring[factor_type], key)
        for key in this_inference_arguments
    }
    inference_arguments[factor_type] = this_inference_arguments

  factor_type_to_msgs_range = bp_state.fg_state.factor_type_to_msgs_range
  factor_type_to_potentials_range = (
      bp_state.fg_state.factor_type_to_potentials_range
  )

  def update(
      bp_arrays: Optional[BPArrays] = None,
      log_potentials_updates: Optional[Dict[Any, jnp.ndarray]] = None,
      ftov_msgs_updates: Optional[Dict[Any, jnp.ndarray]] = None,
      evidence_updates: Optional[Dict[Any, jnp.ndarray]] = None,
  ) -> BPArrays:
    """Function to update belief propagation log_potentials, ftov_msgs, evidence.

    Args:
      bp_arrays: Optional arrays of log_potentials, ftov_msgs, evidence.
      log_potentials_updates: Optional dictionary containing log_potentials
        updates.
      ftov_msgs_updates: Optional dictionary containing ftov_msgs updates.
      evidence_updates: Optional dictionary containing evidence updates.

    Returns:
      A BPArrays with the updated log_potentials, ftov_msgs and evidence.
    """
    if bp_arrays is not None:
      log_potentials = bp_arrays.log_potentials
      evidence = bp_arrays.evidence
      ftov_msgs = bp_arrays.ftov_msgs
    else:
      log_potentials = jax.device_put(bp_state.log_potentials.value)
      ftov_msgs = bp_state.ftov_msgs.value
      evidence = bp_state.evidence.value

    if log_potentials_updates is not None:
      log_potentials = bpstate.update_log_potentials(
          log_potentials,
          log_potentials_updates,
          bp_state.fg_state,
      )

    if ftov_msgs_updates is not None:
      ftov_msgs = bpstate.update_ftov_msgs(
          ftov_msgs,
          ftov_msgs_updates,
          bp_state.fg_state,
      )

    if evidence_updates is not None:
      evidence = bpstate.update_evidence(
          evidence, evidence_updates, bp_state.fg_state
      )

    return BPArrays(
        log_potentials=log_potentials,
        ftov_msgs=ftov_msgs,
        evidence=evidence,
    )

  def run_bp(
      bp_arrays: BPArrays,
      num_iters: int,
      damping: float = 0.5,
  ) -> BPArrays:
    """Function to run belief propagation for num_iters with a damping_factor.

    Args:
      bp_arrays: Initial arrays of log_potentials, ftov_msgs, evidence.
      num_iters: Number of belief propagation iterations.
      damping: The damping factor to use for message updates between one
        timestep and the next.

    Returns:
      A BPArrays containing the updated ftov_msgs.
    """
    log_potentials = bp_arrays.log_potentials
    evidence = bp_arrays.evidence
    ftov_msgs = bp_arrays.ftov_msgs

    # Normalize the messages to ensure the maximum value is 0.
    ftov_msgs = normalize_and_clip_msgs(
        ftov_msgs, edges_num_states, max_msg_size
    )

    @jax.checkpoint
    def update(msgs: jnp.ndarray, _) -> Tuple[jnp.ndarray, None]:
      # Compute new variable to factor messages by message passing
      vtof_msgs = pass_var_to_fac_messages(
          msgs,
          evidence,
          var_states_for_edges,
      )
      ftov_msgs = jnp.zeros_like(vtof_msgs)
      for factor_type in FAC_TO_VAR_UPDATES:
        msgs_start, msgs_end = factor_type_to_msgs_range[factor_type]
        potentials_start, potentials_end = factor_type_to_potentials_range[
            factor_type
        ]
        ftov_msgs_type = FAC_TO_VAR_UPDATES[factor_type](
            vtof_msgs=vtof_msgs[msgs_start:msgs_end],
            log_potentials=log_potentials[potentials_start:potentials_end],
            temperature=temperature,
            **inference_arguments[factor_type],
        )
        ftov_msgs = ftov_msgs.at[msgs_start:msgs_end].set(ftov_msgs_type)

      # Use the results of message passing to perform damping and
      # update the factor to variable messages
      delta_msgs = ftov_msgs - msgs
      msgs = msgs + (1 - damping) * delta_msgs
      # Normalize and clip these damped, updated messages before returning them.
      msgs = normalize_and_clip_msgs(msgs, edges_num_states, max_msg_size)
      return msgs, None

    ftov_msgs, _ = jax.lax.scan(update, ftov_msgs, None, num_iters)

    return BPArrays(
        log_potentials=log_potentials, ftov_msgs=ftov_msgs, evidence=evidence
    )

  def to_bp_state(bp_arrays: BPArrays) -> BPState:
    """Function to reconstruct the BPState from a BPArrays.

    Args:
      bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.

    Returns:
      The reconstructed BPState
    """
    return BPState(
        log_potentials=LogPotentials(
            fg_state=bp_state.fg_state, value=bp_arrays.log_potentials
        ),
        ftov_msgs=FToVMessages(
            fg_state=bp_state.fg_state,
            value=bp_arrays.ftov_msgs,
        ),
        evidence=Evidence(fg_state=bp_state.fg_state, value=bp_arrays.evidence),
    )

  def unflatten_beliefs(flat_beliefs, variable_groups) -> Dict[Hashable, Any]:
    """Function that returns unflattened beliefs from the flat beliefs.

    Args:
      flat_beliefs: Flattened array of beliefs
      variable_groups: All the variable groups in the FactorGraph.

    Returns:
      Unflattened beliefs
    """
    beliefs = {}
    start = 0
    for variable_group in variable_groups:
      num_states = variable_group.num_states
      assert isinstance(num_states, np.ndarray)
      length = num_states.sum()

      beliefs[variable_group] = variable_group.unflatten(
          flat_beliefs[start : start + length]
      )
      start += length
    return beliefs

  @jax.jit
  def get_beliefs(bp_arrays: BPArrays) -> Dict[Hashable, Any]:
    """Function to calculate beliefs from a BPArrays.

    Args:
      bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.

    Returns:
      beliefs: Beliefs returned by belief propagation.
    """

    flat_beliefs = (
        jax.device_put(bp_arrays.evidence)
        .at[jax.device_put(var_states_for_edges)]
        .add(bp_arrays.ftov_msgs)
    )
    return unflatten_beliefs(flat_beliefs, bp_state.fg_state.variable_groups)

  bp = BeliefPropagation(
      init=functools.partial(update, None),
      update=update,
      run_bp=run_bp,
      to_bp_state=to_bp_state,
      get_beliefs=get_beliefs,
  )
  return bp


@jax.jit
def pass_var_to_fac_messages(
    ftov_msgs: jnp.array,
    evidence: jnp.array,
    var_states_for_edges: jnp.array,
) -> jnp.array:
  """Passes messages from Variables to Factors.

  The update works by first summing the evidence and neighboring factor to
  variable messages for each variable.
  Next, it subtracts messages from the correct elements of this
  sum to yield the correct updated messages.

  Args:
    ftov_msgs: Array of shape (num_edge_state,). This holds all the flattened
      factor to variable messages.
    evidence: Array of shape (num_var_states,) representing the flattened
      evidence for each variable
    var_states_for_edges: Array of shape (num_edge_states,) This holds the
      global variable state indices for each edge state

  Returns:
      Array of shape (num_edge_state,). This holds all the flattened variable
      to factor messages.
  """
  var_sums_arr = evidence.at[var_states_for_edges].add(ftov_msgs)
  vtof_msgs = var_sums_arr[var_states_for_edges] - ftov_msgs
  return vtof_msgs


@functools.partial(jax.jit, static_argnames="max_msg_size")
def normalize_and_clip_msgs(
    msgs: jnp.ndarray,
    edges_num_states: jnp.ndarray,
    max_msg_size: int,
) -> jnp.ndarray:
  """Performs normalization and clipping of flattened messages.

  Normalization is done by subtracting the maximum value of every message from
  every element of every message,
  clipping is done to keep every message value in the range [-1000, 0].

  Args:
    msgs: Array of shape (num_edge_state,). This holds all the flattened factor
      to variable messages.
    edges_num_states: Array of shape (num_edges,). Number of states for the
      variables connected to each edge
    max_msg_size: the max of edges_num_states

  Returns:
    Array of shape (num_edge_state,). This holds all the flattened factor to
    variable messages after normalization and clipping
  """
  msgs = msgs - jnp.repeat(
      bp_utils.segment_max_opt(msgs, edges_num_states, max_msg_size),
      edges_num_states,
      total_repeat_length=msgs.shape[0],
  )
  # Clip message values to be always greater than -1000
  msgs = jnp.clip(msgs, -1000, None)
  return msgs


@jax.jit
def decode_map_states(beliefs: Dict[Hashable, Any]) -> Any:
  """Function to decode MAP states given the calculated beliefs.

  Args:
    beliefs: An array or a PyTree container containing beliefs for different
      variables.

  Returns:
    An array or a PyTree container containing the MAP states for different
    variables.
  """
  return jax.tree_util.tree_map(lambda x: jnp.argmax(x, axis=-1), beliefs)


@jax.jit
def get_marginals(beliefs: Dict[Hashable, Any]) -> Any:
  """Function to get marginal probabilities given the calculated beliefs.

  Args:
    beliefs: An array or a PyTree container containing beliefs for different
      variables.

  Returns:
    An array or a PyTree container containing the marginal probabilities
    different variables.
  """
  return jax.tree_util.tree_map(
      lambda x: jnp.exp(x - logsumexp(x, axis=-1, keepdims=True)), beliefs
  )
