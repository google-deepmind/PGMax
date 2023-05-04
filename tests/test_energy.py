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

"""Test inference in two models with hardcoded set of messages."""

import jax.numpy as jnp
import numpy as np
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup


def test_energy_empty_vgroup():
  """Compute energy with an empty vgroup."""

  variables = vgroup.NDVarArray(num_states=2, shape=(2, 2))

  variables_empty = vgroup.NDVarArray(num_states=2, shape=(0, 2))
  fg = fgraph.FactorGraph(variable_groups=[variables, variables_empty])
  factor_group = fgroup.EnumFactorGroup(
      variables_for_factors=[[variables[0, 0], variables[0, 1]]],
      factor_configs=np.zeros((1, 2), int),
  )
  fg.add_factors(factor_group)

  bp = infer.build_inferer(fg.bp_state, backend="bp")
  bp_arrays = bp.init()
  beliefs = bp.get_beliefs(bp_arrays)
  map_states = infer.decode_map_states(beliefs)
  init_energy = infer.compute_energy(fg.bp_state, bp_arrays, map_states)[0]
  assert init_energy == 0
  init_energy2, _, _ = infer.compute_energy(
      fg.bp_state, bp_arrays, map_states, debug_mode=True
  )
  assert init_energy == init_energy2


def test_energy_single_state():
  """Compute energy with a single state."""
  variables = vgroup.NDVarArray(num_states=2, shape=(2, 2))

  variables_single_state = vgroup.NDVarArray(num_states=1, shape=(1, 2))
  fg = fgraph.FactorGraph(variable_groups=[variables, variables_single_state])
  factor_group = fgroup.EnumFactorGroup(
      variables_for_factors=[[variables[0, 0], variables[0, 1]]],
      factor_configs=np.zeros((1, 2), int),
  )
  fg.add_factors(factor_group)

  bp = infer.build_inferer(fg.bp_state, backend="bp")
  bp_arrays = bp.init()
  beliefs = bp.get_beliefs(bp_arrays)
  assert beliefs[variables_single_state].shape == (1, 2, 1)
  map_states = infer.decode_map_states(beliefs)
  init_energy = infer.compute_energy(fg.bp_state, bp_arrays, map_states)[0]
  assert init_energy == 0
  init_energy2, _, _ = infer.compute_energy(
      fg.bp_state, bp_arrays, map_states, debug_mode=True
  )
  assert init_energy == init_energy2


def test_energy_all_infinite_but_one_log_potentials():
  """Compute energy with all but one log potentials are infinite."""
  variables = vgroup.NDVarArray(num_states=2, shape=(2,))

  fg = fgraph.FactorGraph(variable_groups=[variables])
  factor_group = fgroup.PairwiseFactorGroup(
      variables_for_factors=[[variables[0], variables[1]]],
      log_potential_matrix=np.array([[-np.inf, -np.inf], [-np.inf, 0]]),
  )
  fg.add_factors(factor_group)

  bp = infer.build_inferer(fg.bp_state, backend="bp")
  bp_arrays = bp.init()
  bp_arrays = bp.run(bp_arrays, num_iters=1, temperature=0)
  beliefs = bp.get_beliefs(bp_arrays)
  assert beliefs[variables].shape == (2, 2)

  map_states = infer.decode_map_states(beliefs)
  assert np.all(map_states[variables] == np.array([1, 1]))

  init_energy = infer.compute_energy(fg.bp_state, bp_arrays, map_states)[0]
  assert init_energy == 0
  init_energy2, _, _ = infer.compute_energy(
      fg.bp_state, bp_arrays, map_states, debug_mode=True
  )
  assert init_energy == init_energy2


def test_energy_all_infinite_log_potentials():
  """Compute energy with all infinite log potentials."""
  variables = vgroup.NDVarArray(num_states=2, shape=(2,))

  fg = fgraph.FactorGraph(variable_groups=[variables])
  factor_group = fgroup.PairwiseFactorGroup(
      variables_for_factors=[[variables[0], variables[1]]],
      log_potential_matrix=jnp.array([[-np.inf, -np.inf], [-np.inf, -np.inf]]),
  )
  fg.add_factors(factor_group)

  bp = infer.build_inferer(fg.bp_state, backend="bp")
  bp_arrays = bp.init()
  bp_arrays = bp.run(bp_arrays, num_iters=1, temperature=0)
  beliefs = bp.get_beliefs(bp_arrays)
  assert beliefs[variables].shape == (2, 2)

  map_states = infer.decode_map_states(beliefs)
  init_energy = infer.compute_energy(fg.bp_state, bp_arrays, map_states)[0]
  assert init_energy == np.inf
  init_energy2, _, _ = infer.compute_energy(
      fg.bp_state, bp_arrays, map_states, debug_mode=True
  )
  assert init_energy == init_energy2
