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

"""A module containing helper functions."""

import functools
from typing import Any, Callable

import jax.numpy as jnp

# A large negative value used to clip message assignments.
# It can be useful to limit this for gradient/numerical stability
# in small models but it can cause max product to fail to find
# a globally optimal solution.
MSG_NEG_INF = -1e32

# A large absolute value used to clip log potentials at runtime, as inf
# log potentials can result in NaN results during message normalization
# You probably want LOG_POTENTIAL_MAX_ABS * MSG_NEG_INF to be finite.
# Doesn't apply in computing the energy of a decoding by default.
LOG_POTENTIAL_MAX_ABS = 1e6

# What value to use as a "base" value for, e.g., log potentials of specific
# configs.  If NEG_INF is too close to MSG_NEG_INF, it'll blow up as unlikely
# configs become extremely likely.
NEG_INF = jnp.NINF


def cached_property(func: Callable[..., Any]) -> property:
  """Customized cached property decorator.

  Args:
    func: Member function to be decorated

  Returns:
    Decorated cached property
  """
  return property(functools.lru_cache(None)(func))
