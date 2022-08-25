# pyformat: mode=midnight
# ==============================================================================
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
# ==============================================================================
"""Test the correct implementation of a factor graph."""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
from pgmax import factor
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup
import pytest


def test_factor_graph():
  """Test the correct implementation of a factor graph."""
  vg = vgroup.VarDict(variable_names=(0,), num_states=15)
  fg = fgraph.FactorGraph(vg)

  enum_factor = factor.EnumFactor(
      variables=[vg[0]],
      factor_configs=np.arange(15)[:, None],
      log_potentials=np.zeros(15),
  )
  fg.add_factors(enum_factor)

  factor_group = fgroup.EnumFactorGroup(
      variables_for_factors=[[vg[0]]],
      factor_configs=np.arange(15)[:, None],
      log_potentials=np.zeros(15),
  )
  with pytest.raises(
      ValueError,
      match=re.escape(
          f"A Factor of type {factor.EnumFactor} involving variables"
          f" {frozenset([(vg.__hash__(), 15)])} already exists."
      ),
  ):
    fg.add_factors(factor_group)


def test_bp_state():
  """Test the correct implementation of a Belief Propagation state."""
  vg = vgroup.VarDict(variable_names=(0,), num_states=15)
  fg0 = fgraph.FactorGraph(vg)
  enum_factor = factor.EnumFactor(
      variables=[vg[0]],
      factor_configs=np.arange(15)[:, None],
      log_potentials=np.zeros(15),
  )
  fg0.add_factors(enum_factor)

  fg1 = fgraph.FactorGraph(vg)
  fg1.add_factors(enum_factor)

  with pytest.raises(
      ValueError,
      match=(
          "log_potentials, ftov_msgs and evidence should be derived from the"
          " same fg_state"
      ),
  ):
    infer.BPState(
        log_potentials=fg0.bp_state.log_potentials,
        ftov_msgs=fg1.bp_state.ftov_msgs,
        evidence=fg1.bp_state.evidence,
    )


def test_log_potentials():
  """Test the correct implementation of log potentials."""
  vg = vgroup.VarDict(variable_names=(0,), num_states=15)
  fg = fgraph.FactorGraph(vg)
  factor_group = fgroup.EnumFactorGroup(
      variables_for_factors=[[vg[0]]],
      factor_configs=np.arange(10)[:, None],
  )
  fg.add_factors(factor_group)

  with pytest.raises(
      ValueError,
      match=re.escape("Expected log potentials shape (10,) for factor group."),
  ):
    fg.bp_state.log_potentials[factor_group] = jnp.zeros((1, 15))

  with pytest.raises(
      ValueError,
      match=re.escape("Invalid FactorGroup for log potentials updates."),
  ):
    factor_group2 = fgroup.EnumFactorGroup(
        variables_for_factors=[[vg[0]]],
        factor_configs=np.arange(10)[:, None],
    )
    fg.bp_state.log_potentials[factor_group2] = jnp.zeros((1, 15))

  with pytest.raises(
      ValueError,
      match=re.escape("Invalid FactorGroup queried to access log potentials."),
  ):
    _ = fg.bp_state.log_potentials[vg[0]]

  with pytest.raises(
      ValueError,
      match=re.escape("Expected log potentials shape (10,). Got (15,)"),
  ):
    infer.LogPotentials(fg_state=fg.fg_state, value=np.zeros(15))

  log_potentials = infer.LogPotentials(fg_state=fg.fg_state, value=np.zeros(10))
  assert jnp.all(log_potentials[factor_group] == jnp.zeros(10))


def test_ftov_msgs():
  """Test the correct implementation of ftov messages."""
  vg = vgroup.VarDict(variable_names=(0,), num_states=15)
  fg = fgraph.FactorGraph(vg)
  factor_group = fgroup.EnumFactorGroup(
      variables_for_factors=[[vg[0]]],
      factor_configs=np.arange(10)[:, None],
  )
  fg.add_factors(factor_group)

  with pytest.raises(
      ValueError,
      match=re.escape("Provided variable is not in the FactorGraph"),
  ):
    fg.bp_state.ftov_msgs[0] = np.ones(10)

  with pytest.raises(
      ValueError,
      match=re.escape(
          "Given belief shape (10,) does not match expected shape (15,) for"
          f" variable ({vg.__hash__()}, 15)."
      ),
  ):
    fg.bp_state.ftov_msgs[vg[0]] = np.ones(10)

  with pytest.raises(
      ValueError, match=re.escape("Expected messages shape (15,). Got (10,)")
  ):
    infer.FToVMessages(fg_state=fg.fg_state, value=np.zeros(10))

  ftov_msgs = infer.FToVMessages(fg_state=fg.fg_state, value=np.zeros(15))
  with pytest.raises(
      TypeError, match=re.escape("'FToVMessages' object is not subscriptable")
  ):
    _ = ftov_msgs[(10,)]


def test_evidence():
  """Test the correct implementation of evidence."""
  vg = vgroup.VarDict(variable_names=(0,), num_states=15)
  fg = fgraph.FactorGraph(vg)
  factor_group = fgroup.EnumFactorGroup(
      variables_for_factors=[[vg[0]]],
      factor_configs=np.arange(10)[:, None],
  )
  fg.add_factors(factor_group)

  with pytest.raises(
      ValueError, match=re.escape("Expected evidence shape (15,). Got (10,).")
  ):
    infer.Evidence(fg_state=fg.fg_state, value=np.zeros(10))

  evidence = infer.Evidence(fg_state=fg.fg_state, value=np.zeros(15))
  assert jnp.all(evidence.value == jnp.zeros(15))

  vg2 = vgroup.VarDict(variable_names=(0,), num_states=15)
  with pytest.raises(
      ValueError,
      match=re.escape(
          "Got evidence for a variable or a VarGroup not in the FactorGraph!"
      ),
  ):
    infer.bp_state.update_evidence(
        jax.device_put(evidence.value),
        {vg2[0]: jax.device_put(np.zeros(15))},
        fg.fg_state,
    )


def test_bp():
  """Test running belief propagation."""
  vg = vgroup.VarDict(variable_names=(0,), num_states=15)
  fg = fgraph.FactorGraph(vg)
  factor_group = fgroup.EnumFactorGroup(
      variables_for_factors=[[vg[0]]],
      factor_configs=np.arange(10)[:, None],
  )
  fg.add_factors(factor_group)

  bp = infer.BP(fg.bp_state, temperature=0)
  bp_arrays = bp.update()
  bp_arrays = bp.update(
      bp_arrays=bp_arrays,
      ftov_msgs_updates={vg[0]: np.zeros(15)},
      log_potentials_updates={factor_group: np.ones(10)},
  )
  bp_arrays = bp.run_bp(bp_arrays, num_iters=1)
  bp_arrays = dataclasses.replace(bp_arrays, log_potentials=jnp.zeros((10)))
  bp_state = bp.to_bp_state(bp_arrays)
  assert bp_state.fg_state == fg.fg_state


def test_bp_different_num_states():
  """Test belief propagation when variables have different number of states."""
  # FactorFraph where VarDict and NDVarArray both have distinct number of states
  num_states = np.array([2, 3, 4])
  vdict = vgroup.VarDict(
      variable_names=tuple(["a", "b", "c"]), num_states=num_states
  )
  varray = vgroup.NDVarArray(shape=(3,), num_states=num_states)
  fg = fgraph.FactorGraph([vdict, varray])

  # Add factors
  # We enforce the variables with same number of states to be in the same state
  for var_dict, var_arr, num_state in zip(
      ["a", "b", "c"], [0, 1, 2], num_states
  ):
    enum_factor = factor.EnumFactor(
        variables=[vdict[var_dict], varray[var_arr]],
        factor_configs=np.array([[idx, idx] for idx in range(num_state)]),
        log_potentials=np.zeros(num_state),
    )
    fg.add_factors(enum_factor)

  # BP functions
  bp = infer.BP(fg.bp_state, temperature=0)

  # Evidence for both VarDict and NDVarArray
  vdict_evidence = {
      var: np.random.gumbel(size=(var[1],)) for var in vdict.variables
  }
  bp_arrays = bp.init(evidence_updates=vdict_evidence)

  varray_evidence = {
      varray: np.random.gumbel(size=(num_states.shape[0], num_states.max()))
  }
  bp_arrays = bp.update(bp_arrays=bp_arrays, evidence_updates=varray_evidence)

  assert np.all(bp_arrays.evidence != 0)

  # Run BP
  bp_arrays = bp.run_bp(bp_arrays, num_iters=50)
  beliefs = bp.get_beliefs(bp_arrays)
  map_states = infer.decode_map_states(beliefs)

  vdict_states = map_states[vdict]
  varray_states = map_states[varray]

  # Verify that variables with same number of states are in the same state
  for var_dict, var_arr in zip(["a", "b", "c"], [0, 1, 2]):
    assert vdict_states[var_dict] == varray_states[var_arr]
