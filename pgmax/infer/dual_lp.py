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

"""A module solving the smoothed dual of the LP relaxation of the MAP problem."""

import dataclasses
import functools
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import jax
import jax.numpy as jnp
from pgmax import infer
from pgmax.factor import FAC_TO_VAR_UPDATES
from pgmax.factor import update_utils
from pgmax.infer.bp_state import BPArrays
from pgmax.infer.bp_state import BPState
from pgmax.infer.inferer import Inferer
from pgmax.infer.inferer import InfererContext
from pgmax.utils import NEG_INF


@dataclasses.dataclass(frozen=True, eq=False)
class SmoothDualLP(Inferer):
  """Smooth Dual LP-MAP solver functions.

  Attributes:
    run_with_objvals: Solves the Smooth Dual LP-MAP problem via accelerated
      gradient descent (or subgradient descent) and returns the objective value
      at each step.
    decode_primal_unaries: Decodes the primal LP-MAP unaries and returns a state
      assignment for each variable of the FactorGraph.
    get_primal_upper_bound: Returns an upper bound of the optimal objective
      value of the (non smooth) LP-MAP problem.
    get_map_lower_bound: Returns a lower bound of the optimal objective value of
      the (Integer Programming) MAP problem.
    get_bp_updates: Used for unit test. Get the BP updates involved in the SDLP
      solver.
  """
  run_with_objvals: Callable[..., Tuple[BPArrays, float]]
  decode_primal_unaries: Callable[
      ..., Tuple[Dict[Hashable, Any], Dict[Hashable, Any]]
  ]
  get_primal_upper_bound: Callable[..., float]
  get_map_lower_bound: Callable[..., float]
  get_bp_updates: Callable[..., Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]


# pylint: disable=invalid-name
def SDLP(bp_state: BPState) -> SmoothDualLP:
  """Returns the generated Smooth Dual LP-MAP functions.

  Args:
    bp_state: Belief propagation state.
  """
  inferer_context = InfererContext(bp_state)

  def smooth_dual_objval_and_grad(
      ftov_msgs: jnp.ndarray,
      log_potentials: jnp.ndarray,
      evidence: jnp.ndarray,
      logsumexp_temp: float,
  ) -> Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the Smooth Dual LP-MAP objective value and its closed-form gradient with respect to the dual messages.

    Note 1: We reuse the Belief Propagation message updates at the temperature
    equal to logsumexp_temp.

    Note 2: Here, ftov_msgs correspond to the dual LP-MAP "messages"
    (cf (1.2) https://people.csail.mit.edu/dsontag/papers/SonGloJaa_optbook.pdf)
    which have a different semantic than the Belief Propagation "messages".

    Args:
      ftov_msgs: Flat array of dual LP-MAP messages from factors to variables
      log_potentials: Flat array of log potentials
      evidence: Array of shape (num_var_states,) containing the flattened
        evidence for each variable
      logsumexp_temp: Temperature for the log sum-exp function, which controls
        how well the objective value of the Smooth Dual LP-MAP approximates the
        Dual LP-MAP one. When logsumexp_temp=0.0, it computes the subgradient of
        the objective value.

    Returns:
      objval: Smooth Dual LP-MAP objective value
      grad_ftov_msgs: Closed-form gradient of the Smooth Dual objective value
        with respect to the dual messages.
        When logsumexp_temp=0.0, it is the subgradient of the objective value.
      bp_ftov_msgs_updates: Belief Propagation message updates, used to compute
        the gradient of the Smooth Dual objective value.
        This quantity is only used for unit tests
      logsumexp_fac_configs_w_reps_at_edges: At each edge, this contains the log
        sum-exp of the outgoing messages over all the factor configs for the
        (unique) factor that this edge connects to.
        This quantity is only used for unit tests
    """
    var_sums_arr = evidence.at[inferer_context.var_states_for_edge_states].add(
        ftov_msgs
    )

    # BP message updates
    bp_ftov_msgs_updates = jnp.zeros_like(ftov_msgs)
    for factor_type in FAC_TO_VAR_UPDATES:
      msgs_start, msgs_end = inferer_context.factor_type_to_msgs_range[
          factor_type
      ]
      potentials_start, potentials_end = (
          inferer_context.factor_type_to_potentials_range[factor_type]
      )
      # Do not update the messages for factor types not present in the graph
      if msgs_start != msgs_end:
        this_bp_ftov_msgs_updates = FAC_TO_VAR_UPDATES[factor_type](
            vtof_msgs=-ftov_msgs[msgs_start:msgs_end],
            log_potentials=log_potentials[potentials_start:potentials_end],
            temperature=logsumexp_temp,
            normalize=False,  # SDLP does not normalize the messages
            **inferer_context.inference_arguments[factor_type],
        )
        bp_ftov_msgs_updates = bp_ftov_msgs_updates.at[msgs_start:msgs_end].set(
            this_bp_ftov_msgs_updates
        )

    if logsumexp_temp == 0.0:
      # Compute the subgradient of the loss w.r.t. the dual messages
      # First compute the contribution of the variables to the subgradient
      maxes_var_states, argmaxes_var_states = (
          update_utils.get_maxes_and_argmaxes(
              var_sums_arr,
              inferer_context.evidence_to_vars,
              inferer_context.num_variables,
          )
      )
      num_var_states = evidence.shape[0]
      subgrad_ftov_msgs_plus = jnp.zeros(shape=(num_var_states,))
      subgrad_ftov_msgs_plus = subgrad_ftov_msgs_plus.at[
          argmaxes_var_states
      ].set(1.0)

      # Second, compute the contribution of the factors to the subgradient
      # maxes_fac_configs_w_reps_at_edges are the same at each edge of a factor
      (
          maxes_fac_configs_w_reps_at_edges,
          argmaxes_fac_configs_by_edges,
      ) = update_utils.get_maxes_and_argmaxes(
          bp_ftov_msgs_updates - ftov_msgs,
          inferer_context.edge_indices_for_edge_states,
          inferer_context.num_edges,
      )
      num_edge_states = inferer_context.var_states_for_edge_states.shape[0]
      subgrad_ftov_msgs_minus = jnp.zeros(shape=(num_edge_states,))
      subgrad_ftov_msgs_minus = subgrad_ftov_msgs_minus.at[
          argmaxes_fac_configs_by_edges
      ].set(-1.0)

      subgrad_ftov_msgs = subgrad_ftov_msgs_plus[
          inferer_context.var_states_for_edge_states
      ] + subgrad_ftov_msgs_minus

      # Finally, compute the Dual LP-MAP objective value
      maxes_fac_configs = (
          jnp.full(shape=(inferer_context.num_factors,), fill_value=NEG_INF)
          .at[inferer_context.factor_indices_for_edge_states]
          .max(
              maxes_fac_configs_w_reps_at_edges[
                  inferer_context.edge_indices_for_edge_states
              ]
          )
      )
      objval = jnp.sum(maxes_var_states) + jnp.sum(maxes_fac_configs)

      return (
          objval,
          subgrad_ftov_msgs,
          bp_ftov_msgs_updates,
          maxes_fac_configs_w_reps_at_edges,
      )

    else:
      # Get the stable softmax and logsumexp of the evidence for each variable
      (
          softmax_var_states,
          logsumexp_var_states
      ) = update_utils.softmax_and_logsumexps_with_temp(
          data=var_sums_arr,
          labels=inferer_context.evidence_to_vars,
          num_labels=inferer_context.num_variables,
          temperature=logsumexp_temp,
      )

      # Get the stable softmax and logsumexp of the messages for each factor
      # logsumexp_fac_configs_w_reps_at_edges are equal at each edge of a factor
      (
          sum_by_edge_state_softmax_fac_configs,
          logsumexp_fac_configs_w_reps_at_edges,
      ) = update_utils.softmax_and_logsumexps_with_temp(
          data=bp_ftov_msgs_updates - ftov_msgs,
          labels=inferer_context.edge_indices_for_edge_states,
          num_labels=inferer_context.num_edges,
          temperature=logsumexp_temp,
      )
      # Compute the closed-form gradient of the loss w.r.t the dual messages
      grad_ftov_msgs = (
          softmax_var_states[inferer_context.var_states_for_edge_states]
          - sum_by_edge_state_softmax_fac_configs
      )

      # Finally, compute the Smooth Dual LP-MAP objective value
      logsumexp_fac_configs = (
          jnp.full(shape=(inferer_context.num_factors,), fill_value=NEG_INF)
          .at[inferer_context.factor_indices_for_edge_states]
          .max(
              logsumexp_fac_configs_w_reps_at_edges[
                  inferer_context.edge_indices_for_edge_states
              ]
          )
      )
      objval = jnp.sum(logsumexp_var_states) + jnp.sum(logsumexp_fac_configs)

      return (
          objval,
          grad_ftov_msgs,
          bp_ftov_msgs_updates,
          logsumexp_fac_configs_w_reps_at_edges,
      )

  @functools.partial(
      jax.jit, static_argnames=("logsumexp_temp", "num_iters", "lr")
  )
  def run_with_objvals(
      sdlp_arrays: BPArrays,
      logsumexp_temp: float,
      num_iters: int,
      lr: Optional[float] = None,
  ) -> Tuple[BPArrays, float]:
    """Solves the Smooth Dual LP-MAP problem via accelerated gradient descent (or subgradient descent) and returns the objective value at each step.

    Args:
      sdlp_arrays: A BPArrays containing log_potentials, LP-MAP ftov_msgs,
        evidence
      logsumexp_temp: Temperature for the log sum-exp function, which controls
        how well the objective value of the Smooth Dual LP-MAP approximates the
        Dual LP-MAP one
      num_iters: Number of (sub)gradient descent iterations
      lr: Optional learning rate. If None, default learning rate is set to
        logsumexp_temp for gradient descent and 0.01 for subgradient descent.

    Returns:
      sdlp_arrays: A BPArrays containing the updated LP-MAP ftov_msgs
      objvals: The list of Smooth Dual LP-MAP objective values at each step

    Raises: ValueError if
      (1) logsumexp_temp is not between 0.0 and 1.0
      (2) logsumexp_temp > 0.0 and lr > logsumexp_temp
    """
    if logsumexp_temp < 0.0 or logsumexp_temp > 1.0:
      raise ValueError(
          "The log sum-exp temperature of the Dual LP-MAP solver has to be"
          " between 0.0 and 1.0"
      )

    if logsumexp_temp != 0.0 and lr is not None and lr > logsumexp_temp:
      raise ValueError(
          "For gradient descent, the learning rate must be smaller than the"
          " log sum-exp temperature."
      )

    if lr is None:
      if logsumexp_temp != 0.0:
        lr = logsumexp_temp
      else:
        lr = 0.01

    log_potentials = sdlp_arrays.log_potentials
    evidence = sdlp_arrays.evidence
    ftov_msgs = sdlp_arrays.ftov_msgs
    eta = jnp.copy(ftov_msgs)

    def gradient_descent_update(msgs_eta, it):
      """Runs one step of accelerated gradient descent and updates the dual messages."""
      ftov_msgs, eta = msgs_eta
      # Compute the objective value and the gradient w.r.t the dual messages
      objval, msgs_grads, _, _ = smooth_dual_objval_and_grad(
          ftov_msgs,
          log_potentials,
          evidence,
          logsumexp_temp=logsumexp_temp,
      )
      # Accelerated gradient descent
      if logsumexp_temp > 0:
        # Log-sum-exp has a Lipschitz gradient with constant 1 / logsumexp_temp
        # https://arxiv.org/pdf/1704.00805.pdf, Prop. 4
        new_eta = ftov_msgs - lr * msgs_grads
        # Nesterov's acceleration
        new_ftov_msgs = new_eta + (it + 1.0) / (it + 4.0) * (new_eta - eta)

      # Subgradient descent
      else:
        new_eta = ftov_msgs - lr / jnp.sqrt(it + 1.0) * msgs_grads
        # Empirically found to accelerate convergence
        new_ftov_msgs = new_eta + (it + 1.0) / (it + 4.0) * (new_eta - eta)

      return (new_ftov_msgs, new_eta), objval

    (ftov_msgs, eta), objvals = jax.lax.scan(
        gradient_descent_update, (ftov_msgs, eta), jnp.arange(num_iters)
    )
    sdlp_arrays = BPArrays(
        log_potentials=log_potentials, ftov_msgs=ftov_msgs, evidence=evidence
    )
    return sdlp_arrays, objvals

  def run(
      sdlp_arrays: BPArrays,
      logsumexp_temp: float,
      num_iters: int,
      lr: Optional[float] = None,
  ) -> BPArrays:
    """A wrapper around run_with_objvals to only return the updated BPArrays."""
    sdlp_arrays, _ = run_with_objvals(
        sdlp_arrays=sdlp_arrays,
        logsumexp_temp=logsumexp_temp,
        num_iters=num_iters,
        lr=lr,
    )
    return sdlp_arrays

  def decode_primal_unaries(
      sdlp_arrays: BPArrays,
  ) -> Tuple[Dict[Hashable, Any], Dict[Hashable, Any]]:
    """Decodes the primal unaries and returns a state assignment for each variable of the FactorGraph.

    This implements the local decoding suggested in Section 1.7 of
    https://people.csail.mit.edu/dsontag/papers/SonGloJaa_optbook.pdf

    Note: We do not decode the primal factor binary variables. Also note that
    the decoded factor binaries would not be guaranteed to be consistent with
    the decoded unaries when we are not at a minimum of the Dual LP-MAP problem.

    Args:
      sdlp_arrays: A BPArrays containing log_potentials, LP-MAP ftov_msgs,
        evidence

    Returns:
      decoded_primal_unaries: The primal unaries decoded from the SDLP solution
      dual_beliefs: The associated dual beliefs
    """
    dual_beliefs = inferer_context.get_beliefs(sdlp_arrays)
    decoded_primal_unaries = infer.decode_map_states(dual_beliefs)
    return decoded_primal_unaries, dual_beliefs

  def get_primal_upper_bound(sdlp_arrays: BPArrays) -> float:
    """Returns an upper bound of the optimal objective value of the (non smooth) LP-MAP problem.

    Args:
      sdlp_arrays: A BPArrays containing log_potentials, LP-MAP ftov_msgs,
        evidence
    """
    return smooth_dual_objval_and_grad(
        sdlp_arrays.ftov_msgs,
        sdlp_arrays.log_potentials,
        sdlp_arrays.evidence,
        logsumexp_temp=0.0,
    )[0]

  def get_map_lower_bound(
      sdlp_arrays: BPArrays,
      decoded_primal_unaries: Dict[Hashable, Any],
      debug_mode=False,
  ) -> float:
    """Given decoded primal unaries, returns a lower bound of the optimal objective value of the (Integer Programming) MAP problem.

    Note 1: This lower bound is the opposite of the energy of the decoding.

    Note 2: This lower bound is also a lower bound of the optimal objective
    value of the relaxed LP-MAP problem.
    A tighter lower bound could be derived for the LP-MAP problem, but it would
    first require to map the dual messages to a feasible point for the primal
    of the LP-MAP problem.

    Note 3: If the MAP lower bound and the LP-MAP upper bound returned by
    get_primal_upper_bound are equal, then strong duality theorem guarantees
    that the LP relaxation is tight and that we are at the optimal MAP solution.

    Args:
      sdlp_arrays: A BPArrays containing log_potentials, LP-MAP ftov_msgs,
        evidence
      decoded_primal_unaries: The primal unaries decoded from the smooth dual
        solution.
      debug_mode: Debug mode give access to the individual contributions of each
        variable and factor

    Returns:
      A lower bound of the optimal objective value of the MAP problem
    """
    return -infer.compute_energy(
        bp_state=bp_state,
        bp_arrays=sdlp_arrays,
        map_states=decoded_primal_unaries,
        debug_mode=debug_mode,
    )[0]

  def get_bp_updates(
      sdlp_arrays: BPArrays, logsumexp_temp: float
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the BP updates involved in the Dual solver, for unit tests.

    Args:
      sdlp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.
      logsumexp_temp: Temperature for the log sum-exp function, which controls
        how well the Smooth Dual LP-MAP objective value approximates the
        Dual LP-MAP objective value

    Returns:
      bp_ftov_msgs_updates: Belief Propagation message updates used to compute
        the Smooth Dual gradients
      logsumexp_fac_configs_w_reps_at_edges: At each edge, this contains the log
        sum exp of the outgoing messages over all the factor configs for the
        (unique) factor that this edge connects to.
    """
    (
        _,
        _,
        bp_ftov_msgs_updates,
        logsumexp_fac_configs_w_reps_at_edges,
    ) = smooth_dual_objval_and_grad(
        sdlp_arrays.ftov_msgs,
        sdlp_arrays.log_potentials,
        sdlp_arrays.evidence,
        logsumexp_temp=logsumexp_temp,
    )

    return (
        bp_ftov_msgs_updates,
        logsumexp_fac_configs_w_reps_at_edges,
    )

  bp = SmoothDualLP(
      init=inferer_context.init,
      update=inferer_context.update,
      to_bp_state=inferer_context.to_bp_state,
      get_beliefs=inferer_context.get_beliefs,
      run=run,
      run_with_objvals=run_with_objvals,
      decode_primal_unaries=decode_primal_unaries,
      get_primal_upper_bound=get_primal_upper_bound,
      get_map_lower_bound=get_map_lower_bound,
      get_bp_updates=get_bp_updates,
  )
  return bp
