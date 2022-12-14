# pyformat style:midnight
# ==============================================================================
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
# ==============================================================================
"""Defines PoolFactorGroup."""

import collections
import dataclasses
from typing import Any, FrozenSet, Optional, OrderedDict, Type

import numpy as np
from pgmax import factor
from pgmax.factor import pool
from pgmax.fgroup import fgroup


@dataclasses.dataclass(frozen=True, eq=False)
class PoolFactorGroup(fgroup.FactorGroup):
  """Class to represent a group of PoolFactors."""
  factor_configs: Optional[np.ndarray] = dataclasses.field(
      init=False,
      default=None,
  )
  factor_type: Type[factor.Factor] = dataclasses.field(
      init=False,
      default=pool.PoolFactor,
  )

  # pylint: disable=g-complex-comprehension
  def _get_variables_to_factors(
      self,) -> OrderedDict[FrozenSet[Any], pool.PoolFactor]:
    """Function that generates a dictionary mapping set of connected variables to factors.

    This function is only called on demand when the user requires it.

    Returns:
      A dictionary mapping all possible set of connected variables to different
      factors.
    """
    variables_to_factors = collections.OrderedDict([(
        frozenset(variables_for_factor),
        pool.PoolFactor(variables=variables_for_factor)
    ) for variables_for_factor in self.variables_for_factors])
    return variables_to_factors
