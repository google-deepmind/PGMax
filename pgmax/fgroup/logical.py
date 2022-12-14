# pyformat style:midnight
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
"""Defines LogicalFactorGroup and its two children, ORFactorGroup and ANDFactorGroup.
"""

import collections
import dataclasses
from typing import Any, FrozenSet, Optional, OrderedDict, Type

import numpy as np
from pgmax import factor
from pgmax.factor import logical
from pgmax.fgroup import fgroup


@dataclasses.dataclass(frozen=True, eq=False)
class LogicalFactorGroup(fgroup.FactorGroup):
  """Class to represent a group of LogicalFactors.

  All factors in the group are assumed to have the same edge_states_offset.
  Consequently, the factors are all ORFactors or ANDFactors.

  Attributes:
    edge_states_offset: Offset to go from a variable's relevant state to its
      other state. For ORFactors the edge_states_offset is 1 For ANDFactors the
      edge_states_offset is -1
  """

  factor_configs: Optional[np.ndarray] = dataclasses.field(
      init=False,
      default=None,
  )
  edge_states_offset: int = dataclasses.field(init=False)

  # pylint: disable=g-complex-comprehension
  def _get_variables_to_factors(
      self,) -> OrderedDict[FrozenSet[Any], logical.LogicalFactor]:
    """Function that generates a dictionary mapping set of connected variables to factors.

    This function is only called on demand when the user requires it.

    Returns:
      A dictionary mapping all possible set of connected variables to different
      factors.
    """
    variables_to_factors = collections.OrderedDict([
        (frozenset(variables_for_factor),
         self.factor_type(variables=variables_for_factor))
        for variables_for_factor in self.variables_for_factors
    ])
    return variables_to_factors


@dataclasses.dataclass(frozen=True, eq=False)
class ORFactorGroup(LogicalFactorGroup):
  """Class to represent a group of ORFactors.

  Attributes:
    edge_states_offset: Offset to go from a variable's relevant state to its
      other state. For ORFactors the edge_states_offset is 1.
  """

  edge_states_offset: int = dataclasses.field(init=False, default=1)
  factor_type: Type[factor.Factor] = dataclasses.field(
      init=False,
      default=logical.ORFactor,
  )


@dataclasses.dataclass(frozen=True, eq=False)
class ANDFactorGroup(LogicalFactorGroup):
  """Class to represent a group of ANDFactors.

  Attributes:
    edge_states_offset: Offset to go from a variable's relevant state to its
      other state. For ANDFactors the edge_states_offset is -1.
  """

  edge_states_offset: int = dataclasses.field(init=False, default=-1)
  factor_type: Type[factor.Factor] = dataclasses.field(
      init=False,
      default=logical.ANDFactor,
  )
