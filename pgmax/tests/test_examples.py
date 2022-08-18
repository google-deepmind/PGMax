# pyformat: mode=midnight
# ==============================================================================
# Copyright 2022 Intrinsic Innovation LLC.c
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
"""Test running inference with PGMax in some simple models."""

import jax
import numpy as np
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup


def test_ising_model():
  """Runs inference in a small Ising model."""
  variables = vgroup.NDVarArray(num_states=2, shape=(50, 50))
  fg = fgraph.FactorGraph(variable_groups=variables)

  variables_for_factors = []
  for ii in range(50):
    for jj in range(50):
      kk = (ii + 1) % 50
      ll = (jj + 1) % 50
      variables_for_factors.append([variables[ii, jj], variables[kk, jj]])
      variables_for_factors.append([variables[ii, jj], variables[ii, ll]])

  factor_group = fgroup.PairwiseFactorGroup(
      variables_for_factors=variables_for_factors,
      log_potential_matrix=0.8 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
  )
  fg.add_factors(factor_group)

  # Run inference
  bp = infer.BP(fg.bp_state, temperature=0)

  bp_arrays = bp.init(evidence_updates={
      variables: jax.device_put(np.random.gumbel(size=(50, 50, 2)))
  })
  bp_arrays = bp.run_bp(bp_arrays, num_iters=3000)
  beliefs = bp.get_beliefs(bp_arrays)
  img = infer.decode_map_states(beliefs)[variables]
  assert img.shape == (50, 50)
