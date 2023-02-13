# Copyright 2022 DeepMind Technologies Limited.
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

"""Test building factor graphs and running inference with Pool factors."""

import jax
import numpy as np
from pgmax import factor
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup


# pylint: disable=invalid-name
def test_run_bp_with_PoolFactors():
  """Test building factor graphs and running inference with pool factors.

  Simultaneously test
  (1) the support of PoolFactors in a FactorGraph and their specialized
  inference for different temperatures
  (2) the support of several factor types in a FactorGraph and during
  inference

  To do so, observe that a PoolFactor can be defined as an equivalent
  EnumFactor (which list all the valid Pool configurations) and define two
  equivalent FactorGraphs:
  FG1: first half of factors are defined as EnumFactors, second half are
  defined as PoolFactors
  FG2: first half of factors are defined as PoolFactors, second half are defined
  as EnumFactors

  Inference for the EnumFactors is run with pass_enum_fac_to_var_messages
  while inference for the PoolFactors is run with pass_pool_fac_to_var_messages.

  Note: for the first seed, add all the EnumFactors to FG1 and all the
    PoolFactors to FG2
  """
  for idx in range(10):
    np.random.seed(idx)

    # Parameters
    num_factors = np.random.randint(3, 8)
    num_pool_choices = np.random.randint(1, 6, num_factors)
    num_pool_choices_cumsum = np.insert(np.cumsum(num_pool_choices), 0, 0)

    # Setting the temperature
    if idx % 2 == 0:
      # Max-product
      temperature = 0.0
    else:
      temperature = np.random.uniform(low=0.5, high=1.0)

    # Graph 1
    pool_indicators_variables1 = vgroup.NDVarArray(
        num_states=2, shape=(num_factors,)
    )
    pool_choices_variables1 = vgroup.NDVarArray(
        num_states=2, shape=(num_pool_choices.sum(),)
    )
    fg1 = fgraph.FactorGraph(
        variable_groups=[pool_indicators_variables1, pool_choices_variables1]
    )

    # Graph 2
    pool_indicators_variables2 = vgroup.NDVarArray(
        num_states=2, shape=(num_factors,)
    )
    pool_choices_variables2 = vgroup.NDVarArray(
        num_states=2, shape=(num_pool_choices.sum(),)
    )
    fg2 = fgraph.FactorGraph(
        variable_groups=[pool_indicators_variables2, pool_choices_variables2]
    )

    # Variable names for factors
    variables_for_factors1 = []
    variables_for_factors2 = []
    for factor_idx in range(num_factors):
      variables1 = []
      # Pool choices variables must be added first
      for idx1 in range(
          num_pool_choices_cumsum[factor_idx],
          num_pool_choices_cumsum[factor_idx + 1],
      ):
        variables1.append(pool_choices_variables1[idx1])
      variables1.append(pool_indicators_variables1[factor_idx])
      variables_for_factors1.append(variables1)

      variables2 = []
      for idx2 in range(
          num_pool_choices_cumsum[factor_idx],
          num_pool_choices_cumsum[factor_idx + 1],
      ):
        variables2.append(pool_choices_variables2[idx2])
      variables2.append(pool_indicators_variables2[factor_idx])
      variables_for_factors2.append(variables2)

    # Option 1: Define EnumFactors equivalent to the PoolFactors
    for factor_idx in range(num_factors):
      this_num_pool_choices = num_pool_choices[factor_idx]

      valid_configs = np.zeros(
          (this_num_pool_choices + 1, this_num_pool_choices + 1), dtype=int
      )
      valid_configs[1:, -1] = 1
      valid_configs[1:, :-1] = np.eye(this_num_pool_choices)

      if factor_idx < num_factors // 2:
        # Add the first half of factors to FactorGraph1
        enum_factor = factor.EnumFactor(
            variables=variables_for_factors1[factor_idx],
            factor_configs=valid_configs,
            log_potentials=np.zeros(valid_configs.shape[0]),
        )
        fg1.add_factors(enum_factor)
      else:
        if idx != 0:
          # Add the second half of factors to FactorGraph2
          enum_factor = factor.EnumFactor(
              variables=variables_for_factors2[factor_idx],
              factor_configs=valid_configs,
              log_potentials=np.zeros(valid_configs.shape[0]),
          )
          fg2.add_factors(enum_factor)
        else:
          # Add all the EnumFactors to FactorGraph1 for the first iter
          enum_factor = factor.EnumFactor(
              variables=variables_for_factors1[factor_idx],
              factor_configs=valid_configs,
              log_potentials=np.zeros(valid_configs.shape[0]),
          )
          fg1.add_factors(enum_factor)

    # Option 2: Define the PoolFactors
    variables_for_PoolFactors_fg1 = []
    variables_for_PoolFactors_fg2 = []

    for factor_idx in range(num_factors):
      if factor_idx < num_factors // 2:
        # Add the first half of factors to FactorGraph2
        variables_for_PoolFactors_fg2.append(variables_for_factors2[factor_idx])
      else:
        if idx != 0:
          # Add the second half of factors to FactorGraph1
          variables_for_PoolFactors_fg1.append(
              variables_for_factors1[factor_idx]
          )
        else:
          # Add all the PoolFactors to FactorGraph2 for the first iter
          variables_for_PoolFactors_fg2.append(
              variables_for_factors2[factor_idx]
          )
    if idx != 0:
      factor_group = fgroup.PoolFactorGroup(variables_for_PoolFactors_fg1)
      fg1.add_factors(factor_group)

    factor_group = fgroup.PoolFactorGroup(variables_for_PoolFactors_fg2)
    fg2.add_factors(factor_group)

    # Run inference
    bp1 = infer.BP(fg1.bp_state, temperature=temperature)
    bp2 = infer.BP(fg2.bp_state, temperature=temperature)

    evidence_pool_indicators = jax.device_put(
        np.random.gumbel(size=(num_factors, 2))
    )
    evidence_pool_choices = jax.device_put(
        np.random.gumbel(size=(sum(num_pool_choices), 2))
    )

    evidence_updates1 = {
        pool_indicators_variables1: evidence_pool_indicators,
        pool_choices_variables1: evidence_pool_choices,
    }
    evidence_updates2 = {
        pool_indicators_variables2: evidence_pool_indicators,
        pool_choices_variables2: evidence_pool_choices,
    }

    bp_arrays1 = bp1.init(evidence_updates=evidence_updates1)
    bp_arrays1 = bp1.run_bp(bp_arrays1, num_iters=5)
    bp_arrays2 = bp2.init(evidence_updates=evidence_updates2)
    bp_arrays2 = bp2.run_bp(bp_arrays2, num_iters=5)

    # Get beliefs
    beliefs1 = bp1.get_beliefs(bp_arrays1)
    beliefs2 = bp2.get_beliefs(bp_arrays2)

    assert np.allclose(
        beliefs1[pool_choices_variables1],
        beliefs2[pool_choices_variables2],
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.allclose(
        beliefs1[pool_indicators_variables1],
        beliefs2[pool_indicators_variables2],
        atol=1e-4,
        rtol=1e-4,
    )
