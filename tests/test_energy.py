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

  bp = infer.BP(fg.bp_state, temperature=0)
  bp_arrays = bp.init()
  beliefs = bp.get_beliefs(bp_arrays)
  map_states = infer.decode_map_states(beliefs)
  init_energy = infer.compute_energy(fg.bp_state, bp_arrays, map_states)[0]
  assert init_energy == 0
