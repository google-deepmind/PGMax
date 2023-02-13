# Copyright 2022 Intrinsic Innovation LLC.
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

"""Test building factor graphs and running inference with OR factors."""

import itertools

import jax
import numpy as np
from pgmax import factor
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup


# pylint: disable=invalid-name
def test_run_bp_with_ORFactors():
  """Test building factor graphs and running inference with OR factors.

  Simultaneously test
  (1) the support of ORFactors in a FactorGraph and their specialized
  inference for different temperatures
  (2) the support of several factor types in a FactorGraph and during
  inference

  To do so, observe that an ORFactor can be defined as an equivalent
  EnumFactor (which list all the valid OR configurations) and define two
  equivalent FactorGraphs:
  FG1: first half of factors are defined as EnumFactors, second half are
  defined as ORFactors
  FG2: first half of factors are defined as ORFactors, second half are defined
  as EnumFactors

  Inference for the EnumFactors is run with pass_enum_fac_to_var_messages
  while inference for the ORFactors is run with pass_logical_fac_to_var_messages

  Note: for the first seed, add all the EnumFactors to FG1 and all the
    ORFactors to FG2
  """
  for idx in range(10):
    np.random.seed(idx)

    # Parameters
    num_factors = np.random.randint(3, 8)
    num_parents = np.random.randint(1, 6, num_factors)
    num_parents_cumsum = np.insert(np.cumsum(num_parents), 0, 0)

    # Setting the temperature
    if idx % 2 == 0:
      # Max-product
      temperature = 0.0
    else:
      temperature = np.random.uniform(low=0.5, high=1.0)

    # Graph 1
    parents_variables1 = vgroup.NDVarArray(
        num_states=2, shape=(num_parents.sum(),)
    )
    children_variables1 = vgroup.NDVarArray(num_states=2, shape=(num_factors,))
    fg1 = fgraph.FactorGraph(
        variable_groups=[parents_variables1, children_variables1]
    )

    # Graph 2
    parents_variables2 = vgroup.NDVarArray(
        num_states=2, shape=(num_parents.sum(),)
    )
    children_variables2 = vgroup.NDVarArray(num_states=2, shape=(num_factors,))
    fg2 = fgraph.FactorGraph(
        variable_groups=[parents_variables2, children_variables2]
    )

    # Variable names for factors
    variables_for_factors1 = []
    variables_for_factors2 = []
    for factor_idx in range(num_factors):
      variables1 = []
      for idx1 in range(
          num_parents_cumsum[factor_idx], num_parents_cumsum[factor_idx + 1]
      ):
        variables1.append(parents_variables1[idx1])
      variables1 += [children_variables1[factor_idx]]
      variables_for_factors1.append(variables1)

      variables2 = []
      for idx2 in range(
          num_parents_cumsum[factor_idx], num_parents_cumsum[factor_idx + 1]
      ):
        variables2.append(parents_variables2[idx2])
      variables2 += [children_variables2[factor_idx]]
      variables_for_factors2.append(variables2)

    # Option 1: Define EnumFactors equivalent to the ORFactors
    for factor_idx in range(num_factors):
      this_num_parents = num_parents[factor_idx]

      configs = np.array(
          list(itertools.product([0, 1], repeat=this_num_parents + 1))
      )
      # Children state is last
      valid_ON_configs = configs[
          np.logical_and(configs[:, :-1].sum(axis=1) >= 1, configs[:, -1] == 1)
      ]
      valid_configs = np.concatenate(
          [np.zeros((1, this_num_parents + 1), dtype=int), valid_ON_configs],
          axis=0,
      )
      assert valid_configs.shape[0] == 2**this_num_parents

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

    # Option 2: Define the ORFactors
    variables_for_ORFactors_fg1 = []
    variables_for_ORFactors_fg2 = []

    for factor_idx in range(num_factors):
      if factor_idx < num_factors // 2:
        # Add the first half of factors to FactorGraph2
        variables_for_ORFactors_fg2.append(variables_for_factors2[factor_idx])
      else:
        if idx != 0:
          # Add the second half of factors to FactorGraph1
          variables_for_ORFactors_fg1.append(variables_for_factors1[factor_idx])
        else:
          # Add all the ORFactors to FactorGraph2 for the first iter
          variables_for_ORFactors_fg2.append(variables_for_factors2[factor_idx])
    if idx != 0:
      factor_group = fgroup.ORFactorGroup(variables_for_ORFactors_fg1)
      fg1.add_factors(factor_group)

    factor_group = fgroup.ORFactorGroup(variables_for_ORFactors_fg2)
    fg2.add_factors(factor_group)

    # Run inference
    bp1 = infer.BP(fg1.bp_state, temperature=temperature)
    bp2 = infer.BP(fg2.bp_state, temperature=temperature)

    evidence_parents = jax.device_put(
        np.random.gumbel(size=(sum(num_parents), 2))
    )
    evidence_children = jax.device_put(np.random.gumbel(size=(num_factors, 2)))

    evidence_updates1 = {
        parents_variables1: evidence_parents,
        children_variables1: evidence_children,
    }
    evidence_updates2 = {
        parents_variables2: evidence_parents,
        children_variables2: evidence_children,
    }

    bp_arrays1 = bp1.init(evidence_updates=evidence_updates1)
    bp_arrays1 = bp1.run_bp(bp_arrays1, num_iters=5)
    bp_arrays2 = bp2.init(evidence_updates=evidence_updates2)
    bp_arrays2 = bp2.run_bp(bp_arrays2, num_iters=5)

    # Get beliefs
    beliefs1 = bp1.get_beliefs(bp_arrays1)
    beliefs2 = bp2.get_beliefs(bp_arrays2)

    assert np.allclose(
        beliefs1[children_variables1], beliefs2[children_variables2], atol=1e-4
    )
    assert np.allclose(
        beliefs1[parents_variables1], beliefs2[parents_variables2], atol=1e-4
    )
