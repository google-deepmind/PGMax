# Copyright 2022 Intrinsic Innovation LLC.
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

"""Test the correct implementation of the different factors."""

import re

import numpy as np
from pgmax import factor
from pgmax import vgroup
import pytest


def test_enumeration_factor():
  """Test the correct implementation of enumeration factors."""
  variables = vgroup.NDVarArray(num_states=3, shape=(1,))

  with pytest.raises(
      NotImplementedError,
      match="Please implement compile_wiring in for your factor",
  ):
    factor.Factor(
        variables=[variables[0]],
        log_potentials=np.array([0.0]),
    )

  with pytest.raises(
      ValueError, match="Configurations should be integers. Got"
  ):
    factor.EnumFactor(
        variables=[variables[0]],
        factor_configs=np.array([[1.0]]),
        log_potentials=np.array([0.0]),
    )

  with pytest.raises(ValueError, match="Potential should be floats. Got"):
    factor.EnumFactor(
        variables=[variables[0]],
        factor_configs=np.array([[1]]),
        log_potentials=np.array([0]),
    )

  with pytest.raises(ValueError, match="factor_configs should be a 2D array"):
    factor.EnumFactor(
        variables=[variables[0]],
        factor_configs=np.array([1]),
        log_potentials=np.array([0.0]),
    )

  with pytest.raises(
      ValueError,
      match=re.escape(
          "Number of variables 1 doesn't match given configurations (1, 2)"
      ),
  ):
    factor.EnumFactor(
        variables=[variables[0]],
        factor_configs=np.array([[1, 2]]),
        log_potentials=np.array([0.0]),
    )

  with pytest.raises(
      ValueError, match=re.escape("Expected log potentials of shape (1,)")
  ):
    factor.EnumFactor(
        variables=[variables[0]],
        factor_configs=np.array([[1]]),
        log_potentials=np.array([0.0, 1.0]),
    )

  with pytest.raises(
      ValueError, match="Invalid configurations for given variables"
  ):
    factor.EnumFactor(
        variables=[variables[0]],
        factor_configs=np.array([[10]]),
        log_potentials=np.array([0.0]),
    )

  with pytest.raises(
      ValueError, match="list_var_states_for_edges cannot be None"
  ):
    factor.concatenate_var_states_for_edges(None)

  with pytest.raises(
      ValueError, match="var_states_for_edges cannot be None"
  ):
    factor.concatenate_var_states_for_edges([None])


def test_logical_factor():
  """Test the correct implementation of the logical factors."""
  child = vgroup.NDVarArray(num_states=2, shape=(1,))[0]
  wrong_parent = vgroup.NDVarArray(num_states=3, shape=(1,))[0]
  parent = vgroup.NDVarArray(num_states=2, shape=(1,))[0]

  with pytest.raises(
      ValueError,
      match=(
          "A LogicalFactor requires at least one parent variable and one child"
          " variable"
      ),
  ):
    factor.logical.LogicalFactor(variables=(child,))

  with pytest.raises(
      ValueError, match="All the variables in a LogicalFactor should be binary"
  ):
    factor.logical.LogicalFactor(variables=(wrong_parent, child))

  logical_factor = factor.logical.LogicalFactor(variables=(parent, child))
  num_parents = len(logical_factor.variables) - 1
  parents_edge_states = np.vstack(
      [
          np.zeros(num_parents, dtype=int),
          np.arange(0, 2 * num_parents, 2, dtype=int),
      ],
  ).T
  child_edge_state = np.array([2 * num_parents], dtype=int)

  with pytest.raises(
      ValueError, match="The highest LogicalFactor index must be 0"
  ):
    wiring = factor.logical.LogicalWiring(
        var_states_for_edges=None,
        parents_edge_states=parents_edge_states + np.array([[1, 0]]),
        children_edge_states=child_edge_state,
        edge_states_offset=1,
    )
    wiring.get_inference_arguments()

  with pytest.raises(
      ValueError,
      match="The LogicalWiring must have 1 different LogicalFactor indices",
  ):
    wiring = factor.logical.LogicalWiring(
        var_states_for_edges=None,
        parents_edge_states=parents_edge_states + np.array([[0], [1]]),
        children_edge_states=child_edge_state,
        edge_states_offset=1,
    )
    wiring.get_inference_arguments()

  with pytest.raises(
      ValueError,
      match=re.escape(
          "The LogicalWiring's edge_states_offset must be 1 (for OR) and -1"
          " (for AND), but is 0"
      ),
  ):
    wiring = factor.logical.LogicalWiring(
        var_states_for_edges=None,
        parents_edge_states=parents_edge_states,
        children_edge_states=child_edge_state,
        edge_states_offset=0,
    )
    wiring.get_inference_arguments()


def test_pool_factor():
  """Test the correct implementation of the pool factors."""
  pool_choice = vgroup.NDVarArray(num_states=2, shape=(1,))[0]
  wrong_pool_indicator = vgroup.NDVarArray(num_states=3, shape=(1,))[0]
  pool_indicator = vgroup.NDVarArray(num_states=2, shape=(1,))[0]

  with pytest.raises(
      ValueError,
      match=(
          "A PoolFactor requires at least one pool choice and one pool "
          "indicator."
      ),
  ):
    factor.pool.PoolFactor(variables=(pool_choice,))

  with pytest.raises(
      ValueError, match="All the variables in a PoolFactor should all be binary"
  ):
    factor.pool.PoolFactor(variables=(wrong_pool_indicator, pool_choice))

  pool_factor = factor.pool.PoolFactor(variables=(pool_indicator, pool_choice))
  num_children = len(pool_factor.variables) - 1
  pool_choices_edge_states = np.vstack(
      [
          np.zeros(num_children, dtype=int),
          np.arange(0, 2 * num_children, 2, dtype=int),
      ],
  ).T
  pool_indicators_edge_state = np.array([2 * num_children], dtype=int)

  with pytest.raises(
      ValueError, match="The highest PoolFactor index must be 0"
  ):
    wiring = factor.pool.PoolWiring(
        var_states_for_edges=None,
        pool_choices_edge_states=pool_choices_edge_states + np.array([[1, 0]]),
        pool_indicators_edge_states=pool_indicators_edge_state,
    )
    wiring.get_inference_arguments()

  with pytest.raises(
      ValueError,
      match="The PoolWiring must have 1 different PoolFactor indices",
  ):
    wiring = factor.pool.PoolWiring(
        var_states_for_edges=None,
        pool_indicators_edge_states=pool_indicators_edge_state,
        pool_choices_edge_states=pool_choices_edge_states
        + np.array([[0], [1]]),
    )
    wiring.get_inference_arguments()
