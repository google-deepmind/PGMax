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

"""Test clipping behavior doesn't explode."""

import numpy as np
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup


def test_divergent_system_decodes_correctly():
  """Test that we can still sanely decode variables once we hit MSG_NEG_INF.

  Fails before the MSG_NEG_INF and NEG_INF split as it decodes to all 0,
  but it should be all 1.

  if NEG_INF is not sufficient negative compared to MSG_NEG_INF, then when the
  messages from variables with states 0 to factors M_INC becomes lower than
  NEG_INF, the outgoing messages to states 0 will be equal to NEG_INF - M_INC
  which is a very large value. However these messages should be equal to M_INC.
  """

  pg_vars = vgroup.NDVarArray(num_states=2, shape=(10,))
  variables_for_factors = []
  for i in range(10):
    for j in range(i + 1, 10):
      variables_for_factors.append([pg_vars[i], pg_vars[j]])

  f = fgroup.EnumFactorGroup(
      variables_for_factors=variables_for_factors,
      factor_configs=np.array([[0, 0], [1, 1]]),
  )

  fg = fgraph.FactorGraph(pg_vars)
  fg.add_factors(f)

  bp = infer.BP(fg.bp_state)
  bp_arrays = bp.init(
      evidence_updates={pg_vars: np.transpose([np.zeros(10), np.ones(10)])}
  )
  bp_arrays2, _ = bp.run_with_diffs(bp_arrays, 100, damping=0)

  beliefs = bp.get_beliefs(bp_arrays2)[pg_vars]
  decoded = infer.decode_map_states(beliefs)
  assert np.all(decoded == 1)
