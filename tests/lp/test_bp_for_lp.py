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

"""Test the convergence of the BP updates at very low temperature to the max-product updates, as well as their consistency across edges."""

import numpy as np

from pgmax import factor
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup

ATOL = 1e-5


def test_convergence_consistency_enum_factors():
  """Test the convergence and consistency of the BP updates for EnumFactors, for an Ising model with categorical variables.

  Simultaneously test that
  (1) the BP updates at very low temperature are close to the max-product
  updates for T=0
  (2) for each factor, the logsumexp of the outgoing messages messages over all
  its configs is constant, when evaluated at each edge connected to this factor
  """
  num_states = 3
  grid_size = 4
  n_iters = 10

  # Create an Ising model
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
      log_potential_matrix=np.random.normal(
          size=(len(variables_for_factors), num_states, num_states)
      ),
  )
  fg.add_factors(factor_group)

  for idx in range(n_iters):
    np.random.seed(idx)

    # Evidence updates
    evidence_updates = {
        variables: np.random.gumbel(size=(grid_size, grid_size, num_states))
    }

    # Test for different very low temperature
    if idx % 2 == 0:
      temperature = 1e-3
    else:
      temperature = 0.01

    # Create the dual solver
    sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

    # Messages are randomly initialized
    ftov_msgs_updates = {
        factor.EnumFactor: np.random.normal(
            size=fg.bp_state.ftov_msgs.value.shape
        )
    }
    sdlp_arrays = sdlp.init(
        evidence_updates=evidence_updates, ftov_msgs_updates=ftov_msgs_updates
    )

    # Max-product updates
    (
        bp_updates_t0,
        maxes_fac_configs_t0,
    ) = sdlp.get_bp_updates(sdlp_arrays=sdlp_arrays, logsumexp_temp=0.0)

    # BP with low temperature updates
    (
        bp_updates_t_small,
        logsumexp_fac_configs_t_small,
    ) = sdlp.get_bp_updates(sdlp_arrays=sdlp_arrays, logsumexp_temp=temperature)

    # Check that the updates are close
    assert np.allclose(bp_updates_t0, bp_updates_t_small, atol=temperature)

    # Map factor indices to edge indices
    inferer_context = infer.inferer.InfererContext(fg.bp_state)
    factor_to_edges = {}
    for edge_idx, factor_idx in zip(
        inferer_context.edge_indices_for_edge_states,
        inferer_context.factor_indices_for_edge_states,
    ):
      if factor_idx not in factor_to_edges:
        factor_to_edges[factor_idx] = [edge_idx]
      else:
        if edge_idx not in factor_to_edges[factor_idx]:
          factor_to_edges[factor_idx].append(edge_idx)

    # Check that the maxes and logsumexp of the outgoing messages over all the
    # factor configs is the same, when evaluated at each edge connected to
    # the same factor
    for edge_indices in factor_to_edges.values():
      edge_indices = np.array(edge_indices)

      # Check that maxes_fac_configs are the same at each edge of a factor
      maxes_fac_configs_min = maxes_fac_configs_t0[edge_indices].min()
      maxes_fac_configs_max = maxes_fac_configs_t0[edge_indices].max()
      assert np.allclose(
          maxes_fac_configs_min, maxes_fac_configs_max, atol=ATOL
      )

      # Check that logsumexp_fac_configs are the same at each edge of a factor
      logsumexp_fac_configs_min = logsumexp_fac_configs_t_small[
          edge_indices
      ].min()
      logsumexp_fac_configs_max = logsumexp_fac_configs_t_small[
          edge_indices
      ].max()
      assert np.allclose(
          logsumexp_fac_configs_min, logsumexp_fac_configs_max, atol=ATOL
      )


def test_convergence_consistency_or_factors():
  """Test the convergence and consistency of the BP updates for ORFactors.

  The FactorGraph uses ORFactors to sparsify a line: the line is represented
  by the bottom variables, while the sparse representation is represented by
  the top variables. Each bottom variable is active and can be explained
  by each one of its 3 closest top variables.

  Simultaneously test that
  (1) the BP updates at very low temperature are close to the max-product
  updates for T=0
  (2) for each factor, the logsumexp of the outgoing messages messages over all
  its configs is constant, when evaluated at each edge connected to this factor
  """
  line_length = 20
  n_iters = 10

  # Create the FactorGraph
  top_variables = vgroup.NDVarArray(num_states=2, shape=(line_length,))
  bottom_variables = vgroup.NDVarArray(num_states=2, shape=(line_length,))
  fg = fgraph.FactorGraph(variable_groups=[top_variables, bottom_variables])

  # Add ORFactors to the graph
  variables_for_or_factors = []
  for factor_idx in range(line_length):
    variables_for_or_factor = [top_variables[factor_idx]]
    if factor_idx >= 1:
      variables_for_or_factor.append(top_variables[factor_idx - 1])
    if factor_idx <= line_length - 2:
      variables_for_or_factor.append(top_variables[factor_idx + 1])
    # Add child variable at the last position
    variables_for_or_factor.append(bottom_variables[factor_idx])
    variables_for_or_factors.append(variables_for_or_factor)

  factor_group = fgroup.ORFactorGroup(variables_for_or_factors)
  fg.add_factors(factor_group)

  # Evidence update: bottom variables are all turned ON
  bottom_variables_evidence = np.zeros((line_length, 2))
  bottom_variables_evidence[..., 0] = -10_000
  # Top variables are more likely to be turned OFF
  top_variables_evidence = np.zeros((line_length, 2))
  top_variables_evidence[..., 1] = -100

  # Create the dual solver
  sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

  for idx in range(n_iters):
    np.random.seed(idx)

    # Test for different very low temperature
    if idx % 2 == 0:
      temperature = 1e-3
    else:
      temperature = 0.01

    # Add Gumbel noise to the evidence updates
    top_variables_add = np.random.gumbel(size=top_variables_evidence.shape)
    evidence_updates = {
        top_variables: top_variables_evidence + top_variables_add,
        bottom_variables: bottom_variables_evidence,
    }

    # Create the dual solver
    sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

    # Messages are randomly initialized
    ftov_msgs_updates = {
        factor.ORFactor: np.random.normal(
            size=fg.bp_state.ftov_msgs.value.shape
        )
    }
    sdlp_arrays = sdlp.init(
        evidence_updates=evidence_updates, ftov_msgs_updates=ftov_msgs_updates
    )

    # Max-product updates
    (
        bp_updates_t0,
        maxes_fac_configs_t0,
    ) = sdlp.get_bp_updates(sdlp_arrays=sdlp_arrays, logsumexp_temp=0.0)

    # BP with low temperature updates
    (
        bp_updates_t_small,
        logsumexp_fac_configs_t_small,
    ) = sdlp.get_bp_updates(sdlp_arrays=sdlp_arrays, logsumexp_temp=temperature)

    # Check that the updates are close
    assert np.allclose(bp_updates_t0, bp_updates_t_small, atol=temperature)

    # Map factor indices to edge indices
    inferer_context = infer.inferer.InfererContext(fg.bp_state)
    factor_to_edges = {}
    for edge_idx, factor_idx in zip(
        inferer_context.edge_indices_for_edge_states,
        inferer_context.factor_indices_for_edge_states,
    ):
      if factor_idx not in factor_to_edges:
        factor_to_edges[factor_idx] = [edge_idx]
      else:
        if edge_idx not in factor_to_edges[factor_idx]:
          factor_to_edges[factor_idx].append(edge_idx)

    # Check that the maxes logsumexp of the outgoing messages over all the
    # factor configs is the same, when evaluated at each edge connected to
    # a same factor
    for edge_indices in factor_to_edges.values():
      edge_indices = np.array(edge_indices)

      # Check that maxes_fac_configs are the same at each edge of a factor
      maxes_fac_configs_min = maxes_fac_configs_t0[edge_indices].min()
      maxes_fac_configs_max = maxes_fac_configs_t0[edge_indices].max()
      assert np.allclose(
          maxes_fac_configs_min, maxes_fac_configs_max, atol=ATOL
      )

      # Check that logsumexp_fac_configs are the same at each edge of a factor
      logsumexp_fac_configs_min = logsumexp_fac_configs_t_small[
          edge_indices
      ].min()
      logsumexp_fac_configs_max = logsumexp_fac_configs_t_small[
          edge_indices
      ].max()
      assert np.allclose(
          logsumexp_fac_configs_min, logsumexp_fac_configs_max, atol=ATOL
      )


def test_convergence_consistency_pool_factors():
  """Test the convergence and consistency of the BP updates for PoolFactors.

  The factor graph uses a hierarchy of pool factors of depth n_layers, where
  the n_th layer contains 2^n pool variable.
  The unique pool variable at level 1 is forced to be ON.
  Each pool variable at level n > 1 is
  (1) a pool choice in one pool also involving a pool indicator at level n - 1
  and another pool choice at level n
  (2) a pool indicator in one pool also involving 2 pools choices at level n

  Simultaneously test that
  (1) the BP updates at very low temperature are close to the max-product
  updates for T=0
  (2) for each factor, the logsumexp of the outgoing messages messages over all
  its configs is constant, when evaluated at each edge connected to this factor
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

  variables_for_pool_factors = []
  for idx_pool_layer in range(n_layers - 1):
    pool_choices_indices_start = cumsum_variables[idx_pool_layer + 1]

    for pool_indicator_idx in range(
        cumsum_variables[idx_pool_layer], cumsum_variables[idx_pool_layer + 1]
    ):
      variables_for_pool_factor = [
          variables[pool_choices_indices_start + pool_choice_idx]
          for pool_choice_idx in range(n_pool_choices_by_pool)
      ] + [variables[pool_indicator_idx]]
      pool_choices_indices_start += n_pool_choices_by_pool
      variables_for_pool_factors.append(variables_for_pool_factor)

  factor_group = fgroup.PoolFactorGroup(variables_for_pool_factors)
  fg.add_factors(factor_group)

  for idx in range(n_iters):
    np.random.seed(idx)
    if idx % 2 == 0:
      temperature = 1e-3
    else:
      temperature = 0.01

    # Evidence update
    updates = np.random.gumbel(size=(variables.shape[0], 2))
    updates[0, 1] = 10
    evidence_updates = {variables: updates}

    # Create the dual solver
    sdlp = infer.build_inferer(fg.bp_state, backend="sdlp")

    # Messages are randomly initialized
    ftov_msgs_updates = {
        factor.PoolFactor: np.random.normal(
            size=fg.bp_state.ftov_msgs.value.shape
        )
    }
    sdlp_arrays = sdlp.init(
        evidence_updates=evidence_updates, ftov_msgs_updates=ftov_msgs_updates
    )

    # Max-product updates
    (
        bp_updates_t0,
        maxes_fac_configs_t0,
    ) = sdlp.get_bp_updates(sdlp_arrays=sdlp_arrays, logsumexp_temp=0.0)

    # BP with low temperature updates
    (
        bp_updates_t_small,
        logsumexp_fac_configs_t_small,
    ) = sdlp.get_bp_updates(sdlp_arrays=sdlp_arrays, logsumexp_temp=temperature)

    # Check that the updates are close
    assert np.allclose(bp_updates_t0, bp_updates_t_small, atol=temperature)

    # Map factor indices to edge indices
    inferer_context = infer.inferer.InfererContext(fg.bp_state)
    factor_to_edges = {}
    for edge_idx, factor_idx in zip(
        inferer_context.edge_indices_for_edge_states,
        inferer_context.factor_indices_for_edge_states,
    ):
      if factor_idx not in factor_to_edges:
        factor_to_edges[factor_idx] = [edge_idx]
      else:
        if edge_idx not in factor_to_edges[factor_idx]:
          factor_to_edges[factor_idx].append(edge_idx)

    # Check that the maxes and logsumexp of the outgoing messages over all the
    # factor configs is the same, when evaluated at each edge connected to
    # a same factor
    for edge_indices in factor_to_edges.values():
      edge_indices = np.array(edge_indices)

      # Check that maxes_fac_configs are the same at each edge of a factor
      maxes_fac_configs_min = maxes_fac_configs_t0[edge_indices].min()
      maxes_fac_configs_max = maxes_fac_configs_t0[edge_indices].max()
      assert np.allclose(
          maxes_fac_configs_min, maxes_fac_configs_max, atol=ATOL
      )

      # Check that logsumexp_fac_configs are the same at each edge of a factor
      logsumexp_fac_configs_min = logsumexp_fac_configs_t_small[
          edge_indices
      ].min()
      logsumexp_fac_configs_max = logsumexp_fac_configs_t_small[
          edge_indices
      ].max()
      assert np.allclose(
          logsumexp_fac_configs_min, logsumexp_fac_configs_max, atol=ATOL
      )
