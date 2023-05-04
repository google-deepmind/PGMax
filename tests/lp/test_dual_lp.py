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

"""Test the Smooth Dual LP-MAP solver for different factor types."""


import numpy as np

from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup
from pgmax.utils import primal_lp
import pytest

RTOL = 5e-3


def test_lp_primal_vs_dual_ising():
  """Test the LP-MAP primal and dual solvers on a fully connected Ising model with 3 states, when LP relaxation is tight.

  Both the accelerated gradient descent (with log sum-exp temperature > 0)
  and subgradient descent (with log sum-exp temperature = 0) are tested.

  At each iteration, compare the optimal objective value returned by the
  primal solver, with the upper and lower bounds returned by the dual solver
  """
  grid_size = 4
  num_states = 3
  n_iters = 10

  # Create a fully connected categorical Ising model
  variables = vgroup.NDVarArray(
      num_states=num_states, shape=(grid_size, grid_size)
  )
  fg = fgraph.FactorGraph(variable_groups=variables)

  variables_for_factors = []
  for ii in range(grid_size):
    for jj in range(grid_size):
      for kk in range(ii + 1, grid_size):
        for ll in range(jj + 1, grid_size):
          variables_for_factors.append([variables[ii, jj], variables[kk, ll]])

  factor_group = fgroup.PairwiseFactorGroup(
      variables_for_factors=variables_for_factors,
      log_potential_matrix=0.01  # rescaling makes the test more robust
      * np.random.normal(
          size=(len(variables_for_factors), num_states, num_states)
      ),
  )
  fg.add_factors(factor_group)

  # Create the dual LP-MAP solver
  sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

  sdlp_arrays = sdlp.init()
  with pytest.raises(
      ValueError,
      match=(
          "The log sum-exp temperature of the Dual LP-MAP solver has to be"
          " between 0.0 and 1.0"
      ),
  ):
    sdlp_arrays = sdlp.run(
        sdlp_arrays,
        logsumexp_temp=1.01,
        num_iters=1,
    )
  with pytest.raises(
      ValueError,
      match=(
          "For gradient descent, the learning rate must be smaller than the"
          " log sum-exp temperature."
      ),
  ):
    sdlp_arrays = sdlp.run(
        sdlp_arrays, logsumexp_temp=0.01, num_iters=1, lr=0.1
    )

  for it in range(n_iters):
    np.random.seed(it)
    if it % 2 == 0:
      logsumexp_temp = 1e-3
    else:
      logsumexp_temp = 0.0

    # Evidence updates
    evidence_updates = {
        variables: np.random.gumbel(size=(grid_size, grid_size, num_states))
    }

    # Solve the primal LP-MAP
    _, cvxpy_lp_objval = primal_lp.primal_lp_solver(fg, evidence_updates)

    # Solve the Smooth Dual LP-MAP
    sdlp_arrays = sdlp.init(evidence_updates=evidence_updates)
    sdlp_arrays = sdlp.run(
        sdlp_arrays,
        logsumexp_temp=logsumexp_temp,
        lr=None,
        num_iters=5_000,
    )
    sdlp_unaries_decoded, _ = sdlp.decode_primal_unaries(sdlp_arrays)

    # Compare the upper and lower bounds of the primal problem (obtained
    # from the dual LP-MAP solution) to the optimal objective value
    primal_upper_bound = sdlp.get_primal_upper_bound(sdlp_arrays)
    primal_lower_bound = sdlp.get_map_lower_bound(
        sdlp_arrays, sdlp_unaries_decoded
    )
    assert np.isclose(cvxpy_lp_objval, primal_upper_bound, rtol=RTOL)
    assert np.isclose(primal_lower_bound, primal_upper_bound, rtol=RTOL)

    # Also test that both standard and debug modes return the same lower bound
    primal_lower_bound_debug = sdlp.get_map_lower_bound(
        sdlp_arrays, sdlp_unaries_decoded, debug_mode=True
    )
    assert np.isclose(primal_lower_bound, primal_lower_bound_debug)


# pylint: disable=invalid-name
def test_lp_primal_vs_dual_line_sparsification():
  """Test the LP-MAP primal and dual solvers to sparsify a line, when the LP relaxation is tight.

  The FactorGraph uses ORFactors to sparsify a line: the line is represented
  by the bottom variables, while the sparse representation is represented by
  the top variables. Each bottom variable is active and can be explained
  by each one of its 3 closest top variables.

  At each iteration, compare
  (1) The decodings returned by both solvers
  (2) The optimal objective value returned by the primal solver, with the upper
  and lower bounds returned by the dual solver
  """
  line_length = 20
  n_iters = 10

  # Create the FactorGraph
  top_variables = vgroup.NDVarArray(num_states=2, shape=(line_length,))
  bottom_variables = vgroup.NDVarArray(num_states=2, shape=(line_length,))
  fg = fgraph.FactorGraph(variable_groups=[top_variables, bottom_variables])

  # Add ORFactors to the graph
  variables_for_OR_factors = []
  for factor_idx in range(line_length):
    variables_for_OR_factor = [top_variables[factor_idx]]
    if factor_idx >= 1:
      variables_for_OR_factor.append(top_variables[factor_idx - 1])
    if factor_idx <= line_length - 2:
      variables_for_OR_factor.append(top_variables[factor_idx + 1])
    # Add child variable at the last position
    variables_for_OR_factor.append(bottom_variables[factor_idx])
    variables_for_OR_factors.append(variables_for_OR_factor)

  factor_group = fgroup.ORFactorGroup(variables_for_OR_factors)
  fg.add_factors(factor_group)

  # Evidence update: bottom variables are all turned ON
  bottom_variables_evidence = np.zeros((line_length, 2))
  bottom_variables_evidence[..., 0] = -10_000
  # Top variables are more likely to be turned OFF
  top_variables_evidence = np.zeros((line_length, 2))
  top_variables_evidence[..., 1] = -100

  # Create the dual LP-MAP solver
  sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

  for it in range(n_iters):
    np.random.seed(it)

    logsumexp_temp = 1e-3
    lr = None

    # Add Gumbel noise to the updates
    top_variables_add = np.random.gumbel(size=top_variables_evidence.shape)
    evidence_updates = {
        top_variables: top_variables_evidence + top_variables_add,
        bottom_variables: bottom_variables_evidence,
    }

    # Solve the primal LP-MAP
    cvxpy_lp_vgroups_solution, cvxpy_lp_objval = primal_lp.primal_lp_solver(
        fg, evidence_updates
    )
    cvxpy_map_states = infer.decode_map_states(cvxpy_lp_vgroups_solution)

    # Solve the smooth dual LP-MAP
    sdlp_arrays = sdlp.init(evidence_updates=evidence_updates)
    sdlp_arrays = sdlp.run(
        sdlp_arrays,
        logsumexp_temp=logsumexp_temp,
        lr=lr,
        num_iters=5_000,
    )
    sdlp_unaries_decoded, _ = sdlp.decode_primal_unaries(sdlp_arrays)

    # First compare the unaries decoded from the dual to the optimal LP solution
    assert np.allclose(
        sdlp_unaries_decoded[top_variables], cvxpy_map_states[top_variables]
    )
    assert sdlp_unaries_decoded[top_variables].sum() == (line_length + 3) // 3

    # Second compare the upper and lower bounds of the primal problem (obtained
    # from the dual LP-MAP solution) to the optimal objective value
    primal_upper_bound = sdlp.get_primal_upper_bound(sdlp_arrays)
    primal_lower_bound = sdlp.get_map_lower_bound(
        sdlp_arrays, sdlp_unaries_decoded
    )
    assert np.isclose(cvxpy_lp_objval, primal_upper_bound, rtol=RTOL)
    assert np.isclose(primal_lower_bound, primal_upper_bound, rtol=RTOL)

    # Also test that both standard and debug modes return the same lower bound
    primal_lower_bound_debug = sdlp.get_map_lower_bound(
        sdlp_arrays, sdlp_unaries_decoded, debug_mode=True
    )
    assert np.isclose(primal_lower_bound, primal_lower_bound_debug)


def test_lp_primal_vs_dual_and_factors():
  """Test the LP-MAP primal and dual solvers on a model with ANDFactors, when the LP relaxation is tight.

  The FactorGraph uses ANDFactors to finds the positions at which all columns
  of a matrix are filled with 1s

  At each iteration, compare
  (1) The decodings returned by both solvers
  (2) The optimal objective value returned by the primal solver, with the upper
  and lower bounds returned by the dual solver
  """
  num_rows = 10
  num_cols = 5
  p_on = 0.8
  n_iters = 5

  # Create the FactorGraph
  matrix = vgroup.NDVarArray(num_states=2, shape=(num_rows, num_cols))
  all_ones = vgroup.NDVarArray(num_states=2, shape=(num_rows,))
  fg = fgraph.FactorGraph(variable_groups=[matrix, all_ones])

  # Add ANDFactors to the graph
  variables_for_AND_factors = []
  for factor_idx in range(num_rows):
    variables_for_AND_factor = matrix[factor_idx]
    variables_for_AND_factor.append(all_ones[factor_idx])
    variables_for_AND_factors.append(variables_for_AND_factor)

  factor_group = fgroup.ANDFactorGroup(variables_for_AND_factors)
  fg.add_factors(factor_group)

  # Create the dual LP-MAP solver
  sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

  for it in range(n_iters):
    np.random.seed(it)
    logsumexp_temp = 1e-3

    # Evidence update: bottom variables are all turned ON
    matrix_obs = np.random.binomial(1, p_on, (num_rows, num_cols))
    matrix_evidence = np.zeros((num_rows, num_cols, 2))
    matrix_evidence[..., 1] = 1_000 * (2 * matrix_obs - 1)
    evidence_updates = {matrix: matrix_evidence}

    # Ground truth solution
    gt_all_ones = np.all(matrix_obs, axis=1).astype(int)

    # Solve the primal LP-MAP
    cvxpy_lp_vgroups_solution, cvxpy_lp_objval = primal_lp.primal_lp_solver(
        fg, evidence_updates
    )
    cvxpy_map_states = infer.decode_map_states(cvxpy_lp_vgroups_solution)

    # Solve the smooth dual LP-MAP
    sdlp_arrays = sdlp.init(evidence_updates=evidence_updates)
    sdlp_arrays = sdlp.run(
        sdlp_arrays,
        logsumexp_temp=logsumexp_temp,
        lr=None,
        num_iters=5_000,
    )
    sdlp_unaries_decoded, _ = sdlp.decode_primal_unaries(sdlp_arrays)

    # First compare the unaries decoded from the dual to the optimal LP solution
    assert np.allclose(cvxpy_map_states[all_ones], gt_all_ones)
    assert np.allclose(sdlp_unaries_decoded[all_ones], gt_all_ones)

    # Second compare the upper and lower bounds of the primal problem (obtained
    # from the dual LP-MAP solution) to the optimal objective value
    primal_upper_bound = sdlp.get_primal_upper_bound(sdlp_arrays)
    primal_lower_bound = sdlp.get_map_lower_bound(
        sdlp_arrays, sdlp_unaries_decoded
    )
    assert np.isclose(cvxpy_lp_objval, primal_upper_bound, rtol=RTOL)
    assert np.isclose(primal_lower_bound, primal_upper_bound, rtol=RTOL)
    print(primal_lower_bound, primal_upper_bound)


def test_lp_primal_vs_dual_pool_factors():
  """Test the LP-MAP primal and dual solvers on a model with PoolFactors, when the LP relaxation is tight.

  The factor graph uses a hierarchy of pool factors of depth n_layers, where
  the n_th layer contains 2^n pool variables.
  The unique pool variable at level 1 is forced to be ON.
  Each pool variable at level n > 1 is
  (1) a pool choice in a single pool involving a pool indicator at level
  n - 1 and another pool choice at level n
  (2) a pool indicator in a single pool involving 2 pools choices at level n

  At each iteration, compare
  (1) The decodings returned by both solvers
  (2) The optimal objective value returned by the primal solver, with the upper
  and lower bounds returned by the dual solver
  """
  n_layers = 4
  n_pool_choices_by_pool = 2
  n_iters = 10

  n_pool_choices_by_layers = [
      n_pool_choices_by_pool**idx_layer for idx_layer in range(n_layers)
  ]
  cumsum_variables = np.insert(np.cumsum(n_pool_choices_by_layers), 0, 0)
  variables = vgroup.NDVarArray(num_states=2, shape=(cumsum_variables[-1],))
  fg = fgraph.FactorGraph(variable_groups=[variables])

  variables_for_PoolFactors = []
  for idx_pool_layer in range(n_layers - 1):
    pool_choices_indices_start = cumsum_variables[idx_pool_layer + 1]

    for pool_indicator_idx in range(
        cumsum_variables[idx_pool_layer], cumsum_variables[idx_pool_layer + 1]
    ):
      variables_for_PoolFactor = [
          variables[pool_choices_indices_start + pool_choice_idx]
          for pool_choice_idx in range(n_pool_choices_by_pool)
      ] + [variables[pool_indicator_idx]]
      pool_choices_indices_start += n_pool_choices_by_pool
      variables_for_PoolFactors.append(variables_for_PoolFactor)

  factor_group = fgroup.PoolFactorGroup(variables_for_PoolFactors)
  fg.add_factors(factor_group)

  # Create the dual LP-MAP solver
  sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

  for it in range(n_iters):
    np.random.seed(it)

    logsumexp_temp = 1e-3
    lr = None

    # Evidence updates
    updates = np.random.gumbel(size=(variables.shape[0], 2))
    updates[0, 1] = 1_000
    evidence_updates = {variables: updates}

    # Solve the primal LP-MAP
    cvxpy_lp_vgroups_solution, cvxpy_lp_objval = primal_lp.primal_lp_solver(
        fg, evidence_updates
    )
    cvxpy_map_states = infer.decode_map_states(cvxpy_lp_vgroups_solution)

    # Solve the smooth dual LP-MAP
    sdlp_arrays = sdlp.init(evidence_updates=evidence_updates)
    sdlp_arrays = sdlp.run(
        sdlp_arrays,
        logsumexp_temp=logsumexp_temp,
        lr=lr,
        num_iters=5_000,
    )
    sdlp_unaries_decoded, _ = sdlp.decode_primal_unaries(sdlp_arrays)

    # First compare the unaries decoded from the dual to the optimal LP solution
    assert np.allclose(
        sdlp_unaries_decoded[variables], cvxpy_map_states[variables]
    )
    assert sdlp_unaries_decoded[variables].sum() == n_layers

    # Second compare the upper and lower bounds of the primal problem (obtained
    # from the dual LP-MAP solution) to the optimal objective value
    primal_upper_bound = sdlp.get_primal_upper_bound(sdlp_arrays)
    primal_lower_bound = sdlp.get_map_lower_bound(
        sdlp_arrays, sdlp_unaries_decoded
    )
    assert np.isclose(cvxpy_lp_objval, primal_upper_bound, rtol=RTOL)
    assert np.isclose(primal_lower_bound, primal_upper_bound, rtol=RTOL)

    # Also test that both standard and debug modes return the same lower bound
    primal_lower_bound_debug = sdlp.get_map_lower_bound(
        sdlp_arrays, sdlp_unaries_decoded, debug_mode=True
    )
    assert np.isclose(primal_lower_bound, primal_lower_bound_debug, rtol=RTOL)
