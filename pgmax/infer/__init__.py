# Copyright 2022 Intrinsic Innovation LLC.
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

"""A sub-package containing functions to perform belief propagation."""

from pgmax.infer.bp import BeliefPropagation
from pgmax.infer.bp import BP
from pgmax.infer.bp import get_marginals
from pgmax.infer.bp_state import BPArrays
from pgmax.infer.bp_state import BPState
from pgmax.infer.bp_state import Evidence
from pgmax.infer.bp_state import FToVMessages
from pgmax.infer.bp_state import LogPotentials
from pgmax.infer.dual_lp import SDLP
from pgmax.infer.dual_lp import SmoothDualLP
from pgmax.infer.energy import compute_energy
from pgmax.infer.inferer import decode_map_states
from pgmax.infer.inferer import Inferer
from pgmax.infer.inferer import InfererContext


def build_inferer(bp_state: BPState, backend: str) -> Inferer:
  """Build a supported inferer."""
  if backend == "bp":
    return BP(bp_state)
  elif backend == "sdlp":
    return SDLP(bp_state)
  else:
    raise NotImplementedError(f"Inferer {backend} is not supported.")
