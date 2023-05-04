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

"""Test the primal LP-MAP solver."""

import numpy as np

from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup
from pgmax.utils import primal_lp


def test_lp_primal_ising(grid_size=4, n_iters=5):
  """Test the primal LP-MAP solver on a fully connected Ising model."""
  np.random.seed(0)

  gt_visible = np.array([
      [
          0.0,
          1.0,
          0.0,
          0.0,
          1.0,
          1.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0,
          0.0,
          0.0,
          0.0,
          1.0,
          0.0,
      ],
      [
          0.0,
          0.0,
          0.0,
          1.0,
          0.0,
          1.0,
          0.0,
          1.0,
          0.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
      ],
      [
          0.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          0.0,
          0.0,
          1.0,
          1.0,
          0.0,
          0.0,
          0.0,
          1.0,
      ],
      [
          1.0,
          1.0,
          1.0,
          0.0,
          0.0,
          1.0,
          0.0,
          0.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          0.0,
          0.0,
      ],
      [
          1.0,
          1.0,
          0.0,
          0.0,
          0.0,
          1.0,
          1.0,
          0.0,
          1.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0,
          1.0,
      ],
  ])
  gt_objvals = np.array(
      [14.64071274, 21.893808, 25.80109529, 20.0094238, 16.02391594]
  )

  # Create a fully connected Ising model
  variables = vgroup.NDVarArray(num_states=2, shape=(grid_size, grid_size))
  fg = fgraph.FactorGraph(variable_groups=variables)

  variables_for_factors = []
  for ii in range(grid_size):
    for jj in range(grid_size):
      for kk in range(ii + 1, grid_size):
        for ll in range(jj + 1, grid_size):
          variables_for_factors.append([variables[ii, jj], variables[kk, ll]])

  factor_group = fgroup.PairwiseFactorGroup(
      variables_for_factors=variables_for_factors,
  )
  fg.add_factors(factor_group)

  for it in range(n_iters):
    # Evidence array
    evidence_updates = {
        variables: np.random.gumbel(size=(grid_size, grid_size, 2))
    }
    # Solve with cvxpy
    cvxpy_lp_vgroups_solution, cvxpy_lp_objval = primal_lp.primal_lp_solver(
        fg, evidence_updates
    )
    cvxpy_map_states = infer.decode_map_states(cvxpy_lp_vgroups_solution)
    assert np.allclose(
        cvxpy_map_states[variables].flatten(), gt_visible[it], atol=1e-6
    )
    assert np.allclose(cvxpy_lp_objval, gt_objvals[it], atol=1e-6)


def test_lp_primal_rbm(n_hidden=8, n_visible=12, n_iters=5):
  """Test the primal LP-MAP solver on a RBM."""
  np.random.seed(0)

  gt_hidden = np.array([
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
      [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
      [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
  ])
  gt_visible = np.array([
      [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
      [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
      [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
  ])
  gt_objvals = np.array(
      [31.16162399, 26.15338596, 27.35837594, 23.34839416, 32.82615209]
  )

  # Rescaling helps making the LP relaxation tight
  bh = 0.1 * np.random.normal(size=(n_hidden,))
  bv = 0.1 * np.random.normal(size=(n_visible,))
  # pylint: disable=invalid-name
  W = 0.1 * np.random.normal(size=(n_hidden, n_visible))

  # Initialize factor graph
  hidden_variables = vgroup.NDVarArray(num_states=2, shape=bh.shape)
  visible_variables = vgroup.NDVarArray(num_states=2, shape=bv.shape)
  fg = fgraph.FactorGraph(variable_groups=[hidden_variables, visible_variables])

  # Create unary factors
  hidden_unaries = fgroup.EnumFactorGroup(
      variables_for_factors=[
          [hidden_variables[ii]] for ii in range(bh.shape[0])
      ],
      factor_configs=np.arange(2)[:, None],
      log_potentials=np.stack([np.zeros_like(bh), bh], axis=1),
  )
  visible_unaries = fgroup.EnumFactorGroup(
      variables_for_factors=[
          [visible_variables[jj]] for jj in range(bv.shape[0])
      ],
      factor_configs=np.arange(2)[:, None],
      log_potentials=np.stack([np.zeros_like(bv), bv], axis=1),
  )

  # Create pairwise factors
  log_potential_matrix = np.zeros(W.shape + (2, 2)).reshape((-1, 2, 2))
  log_potential_matrix[:, 1, 1] = W.ravel()

  # pylint: disable=g-complex-comprehension
  variables_for_factors = [
      [hidden_variables[ii], visible_variables[jj]]
      for ii in range(bh.shape[0])
      for jj in range(bv.shape[0])
  ]
  pairwise_factors = fgroup.PairwiseFactorGroup(
      variables_for_factors=variables_for_factors,
      log_potential_matrix=log_potential_matrix,
  )

  # Add factors to the FactorGraph
  fg.add_factors([hidden_unaries, visible_unaries, pairwise_factors])

  # Evidence array
  for it in range(n_iters):
    evidence_updates = {
        hidden_variables: np.random.gumbel(size=(bh.shape[0], 2)),
        visible_variables: np.random.gumbel(size=(bv.shape[0], 2)),
    }

    # Solve with cvxpy
    cvxpy_lp_vgroups_solution, cvxpy_lp_objval = primal_lp.primal_lp_solver(
        fg, evidence_updates
    )
    cvxpy_map_states = infer.decode_map_states(cvxpy_lp_vgroups_solution)
    cvxpy_visible = cvxpy_map_states[visible_variables].flatten()
    cvxpy_hidden = cvxpy_map_states[hidden_variables].flatten()

    assert np.allclose(cvxpy_hidden, gt_hidden[it], atol=1e-6)
    assert np.allclose(cvxpy_visible, gt_visible[it], atol=1e-6)
    assert np.allclose(cvxpy_lp_objval, gt_objvals[it], atol=1e-6)
