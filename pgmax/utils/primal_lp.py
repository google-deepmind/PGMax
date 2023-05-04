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

"""A solver for the primal of the LP relaxation of the MAP problem using cvxpy with ECOS."""

import time
from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

from absl import logging
import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from pgmax import factor
from pgmax import fgraph
from pgmax import infer
from pgmax import vgroup
from pgmax.infer import bp_state as bpstate


def primal_lp_solver(
    fg: fgraph.FactorGraph,
    evidence_updates: Optional[Dict[Any, jnp.ndarray]] = None,
) -> Tuple[Dict[Hashable, Any], float]:
  """Solves the primal of the LP-MAP problem with the Ecos solver.
  
  Note: There is no need to compute the wiring

  Args:
    fg: FactorGraph with FactorGroups
    evidence_updates: Optional dictionary containing evidence updates.

  Returns:
    lp_vgroups_solution: Dictionary mapping each VarGroup of the FactorGraph
      to its assignment in the LP-MAP solution
    obj_val: Associated objective value
  """
  # Update the evidence
  evidence_value = bpstate.update_evidence(
      fg.bp_state.evidence.value, evidence_updates, fg.bp_state.fg_state
  )
  evidence = infer.Evidence(fg.fg_state, evidence_value)

  start = time.time()
  lp_vars = {}
  lp_factor_vars = {}
  lp_constraints = []
  lp_obj_val = 0

  # Build the LP objective value and add constraints in two steps
  # Step 1: loop through the variables
  for variable_group in fg.variable_groups:
    # VarGroups can have different sizes, which is not supported by cp.Variable
    for var in variable_group.variables:
      lp_vars[var] = cp.Variable(var[1])
      # Add constraints
      lp_constraints.append(cp.sum(lp_vars[var]) == 1)
      lp_constraints.append(lp_vars[var] >= 0)
      # Update the objective value
      lp_obj_val += cp.sum(cp.multiply(lp_vars[var], evidence[var]))

  # Step 2: loop through the different factor types
  for factor_type, factor_type_groups in fg.factor_groups.items():
    assert factor_type in [
        factor.EnumFactor, factor.ORFactor, factor.ANDFactor, factor.PoolFactor
    ]

    # Enumeration factors
    if factor_type == factor.EnumFactor:
      for factor_group in factor_type_groups:
        assert (
            len(factor_group.variables_for_factors)
            == factor_group.log_potentials.shape[0]
        )
        for variables_for_factor, factor_log_potentials in zip(
            factor_group.variables_for_factors, factor_group.log_potentials
        ):
          assert (
              factor_log_potentials.shape[0]
              == factor_group.factor_configs.shape[0]
          )
          # Factor variables
          lp_this_factor_vars = cp.Variable(
              factor_group.factor_configs.shape[0]
          )
          lp_factor_vars[tuple(variables_for_factor)] = lp_this_factor_vars
          # Add constraints
          lp_constraints.append(lp_this_factor_vars >= 0)
          # Update objective value
          lp_obj_val += cp.sum(
              cp.multiply(lp_this_factor_vars, factor_log_potentials)
          )

          # Consistency constraint
          for var_idx, variable in enumerate(variables_for_factor):
            for var_state in range(variable[1]):
              sum_indices = np.where(
                  factor_group.factor_configs[:, var_idx] == var_state
              )[0]
              lp_constraints.append(
                  cp.sum(
                      [lp_this_factor_vars[sum_idx] for sum_idx in sum_indices]
                  )
                  == lp_vars[variable][var_state]
              )

    # OR factors
    elif factor_type == factor.ORFactor:
      for factor_group in factor_type_groups:
        for variables_for_factor in factor_group.variables_for_factors:
          parents_variables = variables_for_factor[:-1]
          child_variable = variables_for_factor[-1]

          # Add OR constraints
          lp_constraints.append(
              lp_vars[child_variable][1]
              <= cp.sum(
                  [
                      lp_vars[parent_variable][1]
                      for parent_variable in parents_variables
                  ]
              )
          )
          for parent_variable in parents_variables:
            lp_constraints.append(
                lp_vars[parent_variable][1] <= lp_vars[child_variable][1]
            )

    # AND factors
    elif factor_type == factor.ANDFactor:
      for factor_group in factor_type_groups:
        for variables_for_factor in factor_group.variables_for_factors:
          parents_variables = variables_for_factor[:-1]
          child_variable = variables_for_factor[-1]

          # Add AND constraints
          lp_constraints.append(
              cp.sum(
                  [
                      lp_vars[parent_variable][1]
                      for parent_variable in parents_variables
                  ]
              )
              <= lp_vars[child_variable][1] + len(parents_variables) - 1
          )
          for parent_variable in parents_variables:
            lp_constraints.append(
                lp_vars[child_variable][1] <= lp_vars[parent_variable][1]
            )

    # Pool factors
    elif factor_type == factor.PoolFactor:
      for factor_group in factor_type_groups:
        for variables_for_factor in factor_group.variables_for_factors:
          pool_choices = variables_for_factor[:-1]
          pool_indicator = variables_for_factor[-1]

          # Add Pool constraints
          lp_constraints.append(
              cp.sum([lp_vars[pool_choice][1] for pool_choice in pool_choices])
              == lp_vars[pool_indicator][1]
          )

  # Call the LP solver
  prob = cp.Problem(cp.Maximize(lp_obj_val), lp_constraints)
  logging.info("Building the cvxpy model took %.3f s", (time.time() - start))

  start = time.time()
  prob.solve(solver=cp.ECOS)
  logging.info("Solving the model with ECOS took %.3f s", (time.time() - start))
  lp_vars_solution = {k: v.value for k, v in lp_vars.items()}
  lp_vgroups_solution = unflatten_lp_vars(lp_vars_solution, fg.variable_groups)
  obj_val = prob.value
  return lp_vgroups_solution, obj_val


def unflatten_lp_vars(
    lp_vars_solution: Dict[Tuple[int, int], np.ndarray],
    variable_groups: Sequence[vgroup.VarGroup],
) -> Dict[Hashable, Any]:
  """Returns a mapping from variable groups to their LP solutions.

  Args:
    lp_vars_solution: Mapping from the variables to their LP solutions
    variable_groups: All the variable groups in the FactorGraph.
  """
  lp_vgroups_solution = {}
  for variable_group in variable_groups:
    flat_lp_vars_solution = np.array(
        [lp_vars_solution[var] for var in variable_group.variables]
    ).flatten()

    lp_vgroups_solution[variable_group] = variable_group.unflatten(
        flat_lp_vars_solution, per_state=True
    )
  return lp_vgroups_solution
