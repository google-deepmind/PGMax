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

"""A module containing the core message-passing functions for belief propagation."""

import dataclasses
import functools
from typing import Any, Callable, Dict, Hashable, Optional, Tuple
import warnings

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from pgmax.factor import FAC_TO_VAR_UPDATES
from pgmax.infer.bp_state import BPArrays
from pgmax.infer.bp_state import BPState
from pgmax.infer.inferer import Inferer
from pgmax.infer.inferer import InfererContext
from pgmax.utils import LOG_POTENTIAL_MAX_ABS
from pgmax.utils import MSG_NEG_INF
from pgmax.utils import NEG_INF


@dataclasses.dataclass(frozen=True, eq=False)
class BeliefPropagation(Inferer):
  """Belief propagation functions.

  Attributes:
    run_bp: Backward compatible version of run.
    run_with_diffs: Run inference while monitoring convergence
  """

  run_bp: Callable[..., BPArrays]
  run_with_diffs: Callable[..., Tuple[BPArrays, jnp.ndarray]]


# pylint: disable=invalid-name
def BP(
    bp_state: BPState, temperature: Optional[float] = 0.0
) -> BeliefPropagation:
  """Returns the generated belief propagation functions.

  Args:
    bp_state: Belief propagation state.
    temperature: Temperature for loopy belief propagation. 1.0 corresponds to
      sum-product, 0.0 corresponds to max-product. Used for backward
      compatibility
  """
  inferer_context = InfererContext(bp_state)

  def run_with_diffs(
      bp_arrays: BPArrays,
      num_iters: int,
      damping: float = 0.5,
      temperature: float = temperature,
  ) -> BPArrays:
    """Run belief propagation for num_iters with a damping_factor and a temperature.

    Args:
      bp_arrays: Initial arrays of log_potentials, ftov_msgs, evidence.
      num_iters: Number of belief propagation iterations.
      damping: The damping factor to use for message updates between one
        timestep and the next.
      temperature: Temperature for loopy belief propagation. 1.0 corresponds to
        sum-product, 0.0 corresponds to max-product.

    Returns:
      bp_arrays: A BPArrays containing the updated ftov_msgs.
      msgs_deltas: The maximum absolute difference between the current and the
        updated messages, at each BP iteration.
    """
    # Clip the log potentials for numeric stability.
    log_potentials = jnp.clip(
        bp_arrays.log_potentials, -LOG_POTENTIAL_MAX_ABS, LOG_POTENTIAL_MAX_ABS
    )
    evidence = bp_arrays.evidence
    ftov_msgs = bp_arrays.ftov_msgs

    # Normalize and clip the messages
    ftov_msgs = normalize_and_clip_msgs(
        ftov_msgs,
        inferer_context.edge_indices_for_edge_states,
        inferer_context.num_edges,
    )

    @jax.checkpoint
    def update(msgs: jnp.ndarray, _) -> Tuple[jnp.ndarray, None]:
      # Compute new variable to factor messages by message passing
      vtof_msgs = pass_var_to_fac_messages(
          msgs,
          evidence,
          inferer_context.var_states_for_edge_states,
      )
      ftov_msgs = jnp.zeros_like(vtof_msgs)
      for factor_type in FAC_TO_VAR_UPDATES:
        msgs_start, msgs_end = inferer_context.factor_type_to_msgs_range[
            factor_type
        ]
        potentials_start, potentials_end = (
            inferer_context.factor_type_to_potentials_range[factor_type]
        )
        # Do not update the messages for factor types not present in the graph
        if msgs_start != msgs_end:
          ftov_msgs_type = FAC_TO_VAR_UPDATES[factor_type](
              vtof_msgs=vtof_msgs[msgs_start:msgs_end],
              log_potentials=log_potentials[potentials_start:potentials_end],
              temperature=temperature,
              normalize=True,  # BP normalizes the messages
              **inferer_context.inference_arguments[factor_type],
          )
          ftov_msgs = ftov_msgs.at[msgs_start:msgs_end].set(ftov_msgs_type)

      # Use the results of message passing to perform damping and
      # update the factor to variable messages
      new_msgs = damping * msgs + (1 - damping) * ftov_msgs
      # Normalize and clip these damped, updated messages before returning them.
      new_msgs = normalize_and_clip_msgs(
          new_msgs,
          inferer_context.edge_indices_for_edge_states,
          inferer_context.num_edges,
      )
      # Monitor message convergence via the maximum absolute difference between
      # the current and the updated messages
      msgs_delta = jnp.max(jnp.abs(new_msgs - msgs))
      return new_msgs, msgs_delta

    # Scan can have significant overhead for a small number of iterations
    # if not JITed.  Running one it at a time is a common use-case
    # for checking convergence, so specialize that case.
    if num_iters > 1:
      ftov_msgs, msgs_deltas = jax.lax.scan(update, ftov_msgs, None, num_iters)
    else:
      ftov_msgs, msgs_delta = update(ftov_msgs, None)
      msgs_deltas = msgs_delta[None]

    return (
        BPArrays(
            log_potentials=bp_arrays.log_potentials,
            ftov_msgs=ftov_msgs,
            evidence=bp_arrays.evidence,
        ),
        msgs_deltas,
    )

  def run(
      bp_arrays: BPArrays,
      num_iters: int,
      damping: float = 0.5,
      temperature: float = temperature,
  ) -> BPArrays:
    """A wrapper around run_with_diffs to only return the updated BPArrays."""
    bp_arrays, _ = run_with_diffs(bp_arrays, num_iters, damping, temperature)
    return bp_arrays

  def run_bp(
      bp_arrays: BPArrays,
      num_iters: int,
      damping: float = 0.5,
  ) -> BPArrays:
    """Backward compatible version of run."""
    warnings.warn(
        "BP.run_bp is deprecated. Please consider using BP.run instead."
    )
    return run(bp_arrays, num_iters, damping, temperature)

  # pylint: disable=unexpected-keyword-arg
  bp = BeliefPropagation(
      init=inferer_context.init,
      update=inferer_context.update,
      to_bp_state=inferer_context.to_bp_state,
      get_beliefs=inferer_context.get_beliefs,
      run=run,
      run_bp=run_bp,
      run_with_diffs=run_with_diffs,
  )
  return bp


@jax.jit
def pass_var_to_fac_messages(
    ftov_msgs: jnp.ndarray,
    evidence: jnp.ndarray,
    var_states_for_edge_states: jnp.ndarray,
) -> jnp.ndarray:
  """Pass messages from Variables to Factors.

  The update works by first summing the evidence and neighboring factor to
  variable messages for each variable.
  Next, it subtracts messages from the correct elements of this
  sum to yield the correct updated messages.

  Args:
    ftov_msgs: Array of shape (num_edge_states,). This holds all the flattened
      factor to variable messages.
    evidence: Array of shape (num_var_states,) representing the flattened
      evidence for each variable
    var_states_for_edge_states: Array of shape (num_edge_states,).
      var_states_for_edges[ii] contains the global variable state index
      associated to each edge state

  Returns:
      Array of shape (num_edge_state,). This holds all the flattened variable
      to factor messages.
  """
  var_sums_arr = evidence.at[var_states_for_edge_states].add(ftov_msgs)
  vtof_msgs = var_sums_arr[var_states_for_edge_states] - ftov_msgs
  return vtof_msgs


@functools.partial(jax.jit, static_argnames="num_edges")
def normalize_and_clip_msgs(
    msgs: jnp.ndarray, edge_indices_for_edge_states: jnp.ndarray, num_edges: int
) -> jnp.ndarray:
  """Perform normalization and clipping of flattened messages.

  Normalization is done by subtracting the maximum value of every message from
  every element of every message,

  Clipping keeps every message value in the range [MSG_NEG_INF, 0],
  which is equivalent to a noisy channel between each factor and variable
  in the specified graph, with a log-odds of flipping equal to MSG_NEG_INF.
  Otherwise, in loopy graphs with unsatisfied constraints, messages can tend to
  -inf which results in NaNs.

  Args:
    msgs: Array of shape (num_edge_states,). This holds all the flattened factor
      to variable messages.
    edge_indices_for_edge_states: Array of shape (num_edge_states,)
      edge_indices_for_edge_states[ii] contains the global edge index associated
      with the edge state
    num_edges: Total number of edges in the factor graph

  Returns:
    Array of shape (num_edge_states,). This holds all the flattened factor to
    variable messages after normalization and clipping
  """
  max_by_edges = (
      jnp.full(shape=(num_edges,), fill_value=NEG_INF)
      .at[edge_indices_for_edge_states]
      .max(msgs)
  )
  norm_msgs = msgs - max_by_edges[edge_indices_for_edge_states]

  # Clip message values to be always greater than MSG_NEG_INF
  # This is equvialent to a noisy channel between each factor and variable
  # in the specified graph.
  new_msgs = jnp.clip(norm_msgs, MSG_NEG_INF, None)
  return new_msgs


@jax.jit
def get_marginals(beliefs: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
  """Returns the normalized beliefs of several VarGroups, so that they form a valid probability distribution.

  When the temperature is equal to 1.0, get_marginals returns the sum-product
  estimate of the marginal probabilities.

  When the temperature is equal to 0.0, get_marginals returns the max-product
  estimate of the normalized max-marginal probabilities, defined as:
  norm_max_marginals(x_i^*) ∝ max_{x: x_i = x_i^*} p(x)

  When the temperature is strictly between 0.0 and 1.0, get_marginals returns
  the belief propagation estimate of the normalized soft max-marginal
  probabilities, defined as:
  norm_soft_max_marginals(x_i^*) ∝ (sum_{x: x_i = x_i^*} p(x)^{1 /Temp})^Temp

  Args:
    beliefs: A dictionary containing the beliefs of the VarGroups.
  """
  # pylint: disable=g-long-lambda
  return jax.tree_util.tree_map(
      lambda x: jnp.exp(x - logsumexp(x, axis=-1, keepdims=True))
      if x.size > 0  # Don't need to normalize empty variables
      else x,
      beliefs,
  )
