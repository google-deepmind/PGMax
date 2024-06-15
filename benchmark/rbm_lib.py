# Copyright 2024 Intrinsic Innovation LLC.
# Copyright 2024 DeepMind Technologies Limited.
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
"""Module with functions for RBM inference."""
import itertools
from timeit import default_timer as timer

import numpy as np
import torch
from pgmax import fgraph, fgroup, infer, vgroup
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.models import MarkovNetwork
from pomegranate.distributions import Categorical, JointCategorical
from pomegranate.factor_graph import FactorGraph
from scipy.special import softmax
from tqdm import tqdm


def calc_energies(hidden, visible, W, bh, bv):
    """Calculate the energy of RBM configuration."""
    energies = (
        -np.einsum("h,...h->...", bh, hidden)
        - np.einsum("v,...v->...", bv, visible)
        - np.einsum("hv,...h,...v->...", W, hidden, visible)
    )
    return energies


def enumeration_infer(W, bh, bv):
    """Inference based on brute-force enumeration."""
    nh = bh.shape[0]
    nv = bv.shape[0]
    all_states = np.array(list(itertools.product([0, 1], repeat=nh + nv)))
    all_states = np.stack(
        np.meshgrid(*[np.array([0, 1]) for _ in range(nh + nv)]), axis=-1
    )
    all_energies = calc_energies(
        hidden=all_states[..., :nh], visible=all_states[..., -nv:], W=W, bh=bh, bv=bv
    )
    optimal_state = np.unravel_index(np.argmin(all_energies), all_energies.shape)
    optimal_hidden = optimal_state[:nh]
    optimal_visible = optimal_state[-nv:]
    energy = calc_energies(
        hidden=optimal_hidden, visible=optimal_visible, W=W, bh=bh, bv=bv
    )
    return dict(hidden=optimal_hidden, visible=optimal_visible, energy=energy)


def pomegranate_infer(W, bh, bv, optimal_state=None, backend='cuda'):
    """Inference with pomegranate."""
    nh = bh.shape[0]
    nv = bv.shape[0]
    # Start by creating evidence for the hidden and visible variables
    hidden_evidence_vals = np.stack([np.zeros(nh), bh], axis=-1)
    visible_evidence_vals = np.stack([np.zeros(nv), bv], axis=-1)
    hidden_unary_probs = softmax(hidden_evidence_vals, axis=-1)
    visible_unary_probs = softmax(visible_evidence_vals, axis=-1)
    # Construct factor graph
    model = FactorGraph()
    hidden_variables = []
    for ii in tqdm(range(nh)):
        hidden_var = Categorical(hidden_unary_probs[[ii]])
        model.add_marginal(hidden_var)
        hidden_variables.append(hidden_var)

    visible_variables = []
    for jj in tqdm(range(nv)):
        visible_var = Categorical(visible_unary_probs[[jj]])
        model.add_marginal(visible_var)
        visible_variables.append(visible_var)

    for ii in tqdm(range(nh)):
        for jj in range(nv):
            joint_probs = softmax(np.array([[0.0, 0.0], [0.0, W[ii, jj]]]))
            factor = JointCategorical(joint_probs)
            model.add_factor(factor)
            model.add_edge(hidden_variables[ii], factor)
            model.add_edge(visible_variables[jj], factor)

    # Do inference
    X_masked = torch.masked.MaskedTensor(
        torch.zeros((1, nh + nv), dtype=torch.int32),
        mask=torch.zeros((1, nh + nv), dtype=torch.bool),
    )
    device = torch.device(backend)
    model = model.to(device)
    X_masked = X_masked.to(device)
    start = timer()
    pred_state = model.predict(X_masked)
    inference_time = timer() - start
    pred_hidden = pred_state[0, :nh].cpu().numpy()
    pred_visible = pred_state[0, -nv:].cpu().numpy()
    energy = calc_energies(hidden=pred_hidden, visible=pred_visible, W=W, bh=bh, bv=bv)
    if optimal_state is not None:
        optimal_energy = calc_energies(
            hidden=optimal_state[..., :nh],
            visible=optimal_state[..., -nv:],
            W=W,
            bh=bh,
            bv=bv,
        )
        pred_state = np.concatenate([pred_hidden, pred_visible], axis=-1)
        error = np.sum(optimal_state != pred_state)
    else:
        optimal_energy = None
        error = None

    return dict(
        hidden=pred_hidden,
        visible=pred_visible,
        inference_time=inference_time,
        energy=energy,
        optimal_energy=optimal_energy,
        error=error,
    )


def pgmax_infer(W, bh, bv, optimal_state=None):
    """Inference with PGMax."""
    # Initialize factor graph
    hidden_variables = vgroup.NDVarArray(num_states=2, shape=bh.shape)
    visible_variables = vgroup.NDVarArray(num_states=2, shape=bv.shape)
    fg = fgraph.FactorGraph(variable_groups=[hidden_variables, visible_variables])

    # Create unary factors
    hidden_unaries = fgroup.EnumFactorGroup(
        variables_for_factors=[[hidden_variables[ii]] for ii in range(bh.shape[0])],
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.stack([np.zeros_like(bh), bh], axis=1),
    )
    visible_unaries = fgroup.EnumFactorGroup(
        variables_for_factors=[[visible_variables[jj]] for jj in range(bv.shape[0])],
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.stack([np.zeros_like(bv), bv], axis=1),
    )

    # Create pairwise factors
    log_potential_matrix = np.zeros(W.shape + (2, 2)).reshape((-1, 2, 2))
    log_potential_matrix[:, 1, 1] = W.ravel()

    variables_for_factors = [
        [hidden_variables[ii], visible_variables[jj]]
        for ii in range(bh.shape[0])
        for jj in range(bv.shape[0])
    ]
    pairwise_factors = fgroup.PairwiseFactorGroup(
        variables_for_factors=variables_for_factors,
        log_potential_matrix=log_potential_matrix,
    )

    # Add factors to the FactorGraph
    fg.add_factors([hidden_unaries, visible_unaries, pairwise_factors])

    # Do inference
    bp = infer.build_inferer(fg.bp_state, backend="bp")
    bp_arrays = bp.init(
        evidence_updates={
            hidden_variables: 0 * np.random.gumbel(size=(bh.shape[0], 2)),
            visible_variables: 0 * np.random.gumbel(size=(bv.shape[0], 2)),
        }
    )
    # Time inference time
    start = timer()
    bp_arrays = bp.run(bp_arrays, num_iters=200, damping=0.5, temperature=0.0)
    beliefs = bp.get_beliefs(bp_arrays)
    map_states = infer.decode_map_states(beliefs)
    map_states[hidden_variables].block_until_ready()
    inference_time = timer() - start
    pred_hidden = map_states[hidden_variables]
    pred_visible = map_states[visible_variables]
    nh = bh.shape[0]
    nv = bv.shape[0]
    energy = calc_energies(hidden=pred_hidden, visible=pred_visible, W=W, bh=bh, bv=bv)
    if optimal_state is not None:
        optimal_energy = calc_energies(
            hidden=optimal_state[..., :nh],
            visible=optimal_state[..., -nv:],
            W=W,
            bh=bh,
            bv=bv,
        )
        pred_state = np.concatenate([pred_hidden, pred_visible], axis=-1)
        error = np.sum(optimal_state != pred_state)
    else:
        optimal_energy = None
        error = None

    return dict(
        hidden=pred_hidden,
        visible=pred_visible,
        inference_time=inference_time,
        energy=energy,
        optimal_energy=optimal_energy,
        error=error,
    )


def pgmpy_infer(W, bh, bv, optimal_state=None):
    """Inference with pgmpy."""
    nh = bh.shape[0]
    nv = bv.shape[0]
    # Initialize a MarkovNetwork object
    graph = MarkovNetwork()

    # Start by creating evidence for the hidden and visible variables
    hidden_evidence_vals = np.stack(
        [np.zeros(nh), np.random.logistic(size=(nh,))], axis=-1
    )
    visible_evidence_vals = np.stack(
        [np.zeros(nv), np.random.logistic(size=(nv,))], axis=-1
    )

    # Add all hidden variable nodes
    hidden_variable_nodes = []
    for var_i in range(nh):
        hidden_variable_nodes.append(
            DiscreteFactor(
                ["hidden" + str(var_i)],
                cardinality=[2],
                values=hidden_evidence_vals[var_i],
            )
        )

    # Add all visible variable nodes
    visible_variable_nodes = []
    visible_variable_names = []
    for var_j in range(nv):
        visible_variable_nodes.append(
            DiscreteFactor(
                ["visible" + str(var_j)],
                cardinality=[2],
                values=visible_evidence_vals[var_j],
            )
        )
        visible_variable_names.append("visible" + str(var_j))

    # Next, create Factors and make sure they're connected to the variables
    factor_nodes = []
    for ii in range(nh):
        for jj in range(nv):
            factor_nodes.append(
                DiscreteFactor(
                    ["hidden" + str(ii), "visible" + str(jj)],
                    cardinality=[2, 2],
                    values=np.array([0, 0, 0, W[ii, jj]]),
                )
            )
            graph.add_edge("hidden" + str(ii), "visible" + str(jj))

    # Add all relevant factors to the graph
    for node in visible_variable_nodes + hidden_variable_nodes + factor_nodes:
        graph.add_factors(node)

    # Perform MAP estimation using MPLP
    start = timer()
    mplp = Mplp(graph)
    mplp_result = mplp.map_query()
    inference_time = timer() - start
    pred_hidden = np.array([mplp_result[f'hidden{ii}'] for ii in range(nh)])
    pred_visible = np.array([mplp_result[f'visible{ii}'] for ii in range(nv)])
    energy = calc_energies(hidden=pred_hidden, visible=pred_visible, W=W, bh=bh, bv=bv)
    if optimal_state is not None:
        optimal_energy = calc_energies(
            hidden=optimal_state[..., :nh],
            visible=optimal_state[..., -nv:],
            W=W,
            bh=bh,
            bv=bv,
        )
        pred_state = np.concatenate([pred_hidden, pred_visible], axis=-1)
        error = np.sum(optimal_state != pred_state)
    else:
        optimal_energy = None
        error = None

    return dict(
        hidden=pred_hidden,
        visible=pred_visible,
        inference_time=inference_time,
        energy=energy,
        optimal_energy=optimal_energy,
        error=error,
    )
