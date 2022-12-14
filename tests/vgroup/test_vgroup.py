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
"""Test the correct implementation of the different variables groups."""

import re

import jax
import jax.numpy as jnp
import numpy as np
from pgmax import vgroup
import pytest


def test_variable_dict():
  """Test the correct implementation of variable dict."""
  num_states = np.full((4,), fill_value=2)
  with pytest.raises(
      ValueError, match=re.escape("Expected num_states shape (3,). Got (4,).")
  ):
    vgroup.VarDict(variable_names=tuple([0, 1, 2]), num_states=num_states)

  num_states = np.full((3,), fill_value=2, dtype=np.float32)
  with pytest.raises(
      ValueError,
      match=re.escape(
          "num_states should be an integer or a NumPy array of dtype int"
      ),
  ):
    vgroup.VarDict(variable_names=tuple([0, 1, 2]), num_states=num_states)

  variable_dict = vgroup.VarDict(variable_names=tuple([0, 1, 2]), num_states=15)
  with pytest.raises(
      ValueError, match="data is referring to a non-existent variable 3"
  ):
    variable_dict.flatten({3: np.zeros(10)})

  with pytest.raises(
      ValueError,
      match=re.escape(
          "Variable 2 expects a data array of shape (15,) or (1,). Got (10,)."
      ),
  ):
    variable_dict.flatten({2: np.zeros(10)})

  with pytest.raises(
      ValueError, match="Can only unflatten 1D array. Got a 2D array."
  ):
    variable_dict.unflatten(jnp.zeros((10, 20)))

  assert jnp.all(
      jnp.array(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(
                  lambda x, y: jnp.all(x == y),
                  variable_dict.unflatten(jnp.zeros(3)),
                  {name: np.zeros(1) for name in range(3)},
              )
          )
      )
  )
  with pytest.raises(
      ValueError,
      match=re.escape(
          "flat_data should be either of shape (num_variables(=3),), or"
          " (num_variable_states(=45),)"
      ),
  ):
    variable_dict.unflatten(jnp.zeros((100)))


def test_nd_variable_array():
  """Test the correct implementation of variable array."""
  max_size = int(vgroup.vgroup.MAX_SIZE)
  with pytest.raises(
      ValueError,
      match=re.escape(
          f"Currently only support NDVarArray of size smaller than {max_size}."
          f" Got {max_size + 1}"
      ),
  ):
    vgroup.NDVarArray(shape=(max_size + 1,), num_states=2)

  num_states = np.full((2, 3), fill_value=2)
  with pytest.raises(
      ValueError,
      match=re.escape("Expected num_states shape (2, 2). Got (2, 3)."),
  ):
    vgroup.NDVarArray(shape=(2, 2), num_states=num_states)

  num_states = np.full((2, 3), fill_value=2, dtype=np.float32)
  with pytest.raises(
      ValueError,
      match=re.escape(
          "num_states should be an integer or a NumPy array of dtype int"
      ),
  ):
    vgroup.NDVarArray(shape=(2, 2), num_states=num_states)

  variable_group0 = vgroup.NDVarArray(shape=(5, 5), num_states=2)
  assert len(variable_group0[:3, :3]) == 9

  variable_group = vgroup.NDVarArray(
      shape=(2, 2), num_states=np.array([[1, 2], [3, 4]])
  )

  if variable_group0 < variable_group:
    pass

  with pytest.raises(
      ValueError,
      match=re.escape(
          "data should be of shape (2, 2) or (2, 2, 4). Got (3, 3)."
      ),
  ):
    variable_group.flatten(np.zeros((3, 3)))

  assert jnp.all(
      variable_group.flatten(np.array([[1, 2], [3, 4]]))
      == jnp.array([1, 2, 3, 4])
  )
  assert jnp.all(
      variable_group.flatten(np.zeros((2, 2, 4))) == jnp.zeros((10,))
  )

  with pytest.raises(
      ValueError, match="Can only unflatten 1D array. Got a 2D array."
  ):
    variable_group.unflatten(np.zeros((10, 20)))

  with pytest.raises(
      ValueError,
      match=re.escape(
          "flat_data size should be equal to 4 or to 10. Got size 12."
      ),
  ):
    variable_group.unflatten(np.zeros((12,)))

  assert jnp.all(variable_group.unflatten(np.zeros(4)) == jnp.zeros((2, 2)))
  unflattened = jnp.full((2, 2, 4), fill_value=jnp.nan)
  unflattened = unflattened.at[0, 0, 0].set(0)
  unflattened = unflattened.at[0, 1, :1].set(0)
  unflattened = unflattened.at[1, 0, :2].set(0)
  unflattened = unflattened.at[1, 1].set(0)
  mask = ~jnp.isnan(unflattened)
  assert jnp.all(
      variable_group.unflatten(np.zeros(10))[mask] == unflattened[mask]
  )
