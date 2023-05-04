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

"""Compute the energy of a MAP decoding."""

from typing import Any, Dict, Hashable, Tuple

import jax.numpy as jnp
import numpy as np
from pgmax import factor
from pgmax import vgroup
from pgmax.factor import FAC_TO_VAR_UPDATES
from pgmax.infer import bp_state as bpstate
from pgmax.infer.bp_state import BPArrays
from pgmax.infer.bp_state import BPState
from pgmax.infer.bp_state import Evidence


def get_vars_to_map_states(
    map_states: Dict[Hashable, Any]
) -> Dict[Hashable, Any]:
  """Maps each variable of a FactorGraph to its MAP state.

  Args:
    map_states: A dictionary mapping each VarGroup to the MAP states of all its
      variables.

  Returns:
    A dictionary mapping each individual variable to its MAP state.
  """
  vars_to_map_states = {}
  for variable_group, vg_map_states in map_states.items():
    if np.prod(variable_group.shape) == 0:  # Skip empty variable groups
      continue
    vg_map_states_flat = variable_group.flatten(vg_map_states)
    vars_to_map_states.update(
        zip(variable_group.variables, list(np.array(vg_map_states_flat)))
    )
  return vars_to_map_states


def compute_energy(
    bp_state: BPState,
    bp_arrays: BPArrays,
    map_states: Dict[Hashable, Any],
    debug_mode=False,
) -> Tuple[float, Any, Any]:
  """Return the energy of a decoding, expressed by its MAP states.

  Args:
    bp_state: Belief propagation state
    bp_arrays: Arrays of log_potentials, ftov_msgs, evidence
    map_states: A dictionary mapping the VarGroups of the FactorGraph to their
      MAP states
    debug_mode: Debug mode returns the individual energies of each variable and
      factor in the FactorGraph

  Returns:
    energy: The energy of the decoding
    vars_energies: The energy of each individual variable (only in debug mode)
    factors_energies: The energy of each individual factor (only in debug mode)

  Note: Remember that the lower the energy, the better the decoding!
  """
  if debug_mode:
    return _compute_energy_debug_mode(bp_state, bp_arrays, map_states)

  wiring = bp_state.fg_state.wiring
  factor_type_to_potentials_range = (
      bp_state.fg_state.factor_type_to_potentials_range
  )
  factor_type_to_msgs_range = bp_state.fg_state.factor_type_to_msgs_range

  # Add offsets to the edges and factors indices of var_states_for_edges
  var_states_for_edges = factor.concatenate_var_states_for_edges(
      [
          wiring[factor_type].var_states_for_edges
          for factor_type in FAC_TO_VAR_UPDATES
      ]
  )
  log_potentials = bp_arrays.log_potentials
  evidence = bp_arrays.evidence

  # Inference argumnets per factor type
  inference_arguments = {}
  for factor_type in FAC_TO_VAR_UPDATES:
    this_inference_arguments = wiring[factor_type].get_inference_arguments()
    inference_arguments[factor_type] = this_inference_arguments

  # Step 1: compute the contribution of all the variables to the energy
  # Represent the decoding of each variable groups via a one-hot vector
  vgroups_one_hot_decoding = {}
  for variable_group in bp_state.fg_state.variable_groups:
    if variable_group.num_states.size == 0:
      continue
    # VarDict will soon inherit from NDVarArray
    assert isinstance(variable_group, vgroup.NDVarArray)

    vgroup_decoded = map_states[variable_group]
    vgroup_one_hot_decoding = jnp.zeros(
        shape=vgroup_decoded.shape + (variable_group.num_states.max(),)
    )
    dims = [np.arange(dim) for dim in variable_group.shape]
    meshgrid = jnp.meshgrid(*dims, indexing="ij")
    vgroup_one_hot_decoding = vgroup_one_hot_decoding.at[
        tuple(meshgrid) + (vgroup_decoded,)
    ].set(1.0)
    vgroups_one_hot_decoding[variable_group] = vgroup_one_hot_decoding

  # Flatten the one-hot decoding
  var_states_one_hot_decoding = bpstate.update_evidence(
      jnp.zeros_like(evidence),
      vgroups_one_hot_decoding,
      bp_state.fg_state,
  )
  energy = -jnp.sum(var_states_one_hot_decoding * evidence)

  # Step 2: compute the contribution of each factor type to the energy
  # Extract the one-hot decoding of all the edge states
  edge_states_one_hot_decoding = var_states_one_hot_decoding[
      var_states_for_edges[..., 0]
  ]
  for factor_type in FAC_TO_VAR_UPDATES:
    msgs_start, msgs_end = factor_type_to_msgs_range[factor_type]
    potentials_start, potentials_end = factor_type_to_potentials_range[
        factor_type
    ]
    # Do not compute the energy for factor types not present in the graph
    if msgs_start != msgs_end:
      energy += factor_type.compute_energy(
          edge_states_one_hot_decoding=edge_states_one_hot_decoding[
              msgs_start:msgs_end
          ],
          log_potentials=log_potentials[potentials_start:potentials_end],
          **inference_arguments[factor_type],
      )
  return energy, None, None


def _compute_energy_debug_mode(
    bp_state: BPState,
    bp_arrays: BPArrays,
    map_states: Dict[Hashable, Any],
) -> Tuple[float, Any, Any]:
  """Return the energy of a decoding, expressed by its MAP states, as well as the energies of each variable and factor.

  Args:
    bp_state: Belief propagation state
    bp_arrays: Arrays of log_potentials, ftov_msgs, evidence
    map_states: A dictionary mapping the VarGroups of the FactorGraph to their
      MAP states

  Returns:
    energy: The energy of the decoding
    vars_energies: The energy of each individual variable (only in debug mode)
    factors_energies: The energy of each individual factor (only in debug mode)
  """
  print("Computing the energy of a decoding in debug mode is slow...")

  # Outputs
  energy = 0.0
  vars_energies = {}
  factors_energies = {}

  # Map each variable to its MAP state
  vars_to_map_states = get_vars_to_map_states(map_states)

  # Step 1: compute the contribution of each variable to the energy
  evidence = Evidence(bp_state.fg_state, value=np.array(bp_arrays.evidence))
  for variable_group in bp_state.fg_state.variable_groups:
    for var in variable_group.variables:
      var_decoded_state = int(vars_to_map_states[var])
      var_energy = -float(evidence[var][var_decoded_state])
      vars_energies[var] = var_energy
      energy += var_energy

  # Step 2: compute the contribution of each factor to the energy
  for factor_group in bp_state.fg_state.factor_group_to_potentials_starts:
    # All the factors in a FactorGroup share the same configurations
    factor_configs = factor_group.factor_configs

    for this_factor in factor_group.factors:
      this_factor_variables = this_factor.variables
      factor_energy = factor_group.factor_type.compute_factor_energy(
          variables=this_factor_variables,
          vars_to_map_states=vars_to_map_states,
          factor_configs=factor_configs,
          log_potentials=this_factor.log_potentials,
      )
      energy += factor_energy
      factors_energies[frozenset(this_factor_variables)] = factor_energy

  return energy, vars_energies, factors_energies
