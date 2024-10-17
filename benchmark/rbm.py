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
"""Script to benchmark PGMax against pgmpy and pomegranate."""
import jax

# Test JAX on 'cpu' or 'gpu'
jax_platform_name = 'cpu'

jax.config.update('jax_platform_name', jax_platform_name)
print('Running JAX on platform', jax.numpy.ones(3).device())

import os

import joblib
import numpy as np

import rbm_lib

np.random.seed(0)
np.set_printoptions(suppress=True)

# With ground-truth
n_rbms = 50
n_units = 24
nh = nv = n_units // 2
for rbm_idx in range(n_rbms):
    # Generate random RBM and save weights
    rbm_weights_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_weights.joblib'
    if os.path.exists(rbm_weights_fname):
        W, bh, bv = joblib.load(rbm_weights_fname)
    else:
        W = np.random.randn(nh, nv)
        bh = np.random.logistic(size=(nh,))
        bv = np.random.logistic(size=(nv,))
        joblib.dump((W, bh, bv), rbm_weights_fname)

    # Optimal inference with brute-force enumeration
    results_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_optimal.joblib'
    if os.path.exists(results_fname):
        optimal_results = joblib.load(results_fname)
    else:
        optimal_results = rbm_lib.enumeration_infer(W=W, bh=bh, bv=bv)
        joblib.dump(optimal_results, results_fname)

    print('Optimal', optimal_results)
    optimal_state = np.concatenate(
        [optimal_results['hidden'], optimal_results['visible']], axis=-1
    )

    for num_iters in [20, 200]:
        for batch_size in [1, 100, 1000]:
            # Pomegranate on CPUs
            results_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_pomegranate_cpu_num_iters_{num_iters}_batch_size_{batch_size}.joblib'
            if os.path.exists(results_fname):
                pomegranate_results = joblib.load(results_fname)
            else:
                pomegranate_results = rbm_lib.pomegranate_infer(
                    W=W,
                    bh=bh,
                    bv=bv,
                    num_iters=num_iters,
                    batch_size=batch_size,
                    optimal_state=optimal_state,
                    backend='cpu',
                )
                joblib.dump(pomegranate_results, results_fname)

            print('Pomegranate cpu', pomegranate_results)

            # Pomegranate on GPUs
            results_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_pomegranate_gpu_num_iters_{num_iters}_batch_size_{batch_size}.joblib'
            if os.path.exists(results_fname):
                pomegranate_results = joblib.load(results_fname)
            else:
                pomegranate_results = rbm_lib.pomegranate_infer(
                    W=W,
                    bh=bh,
                    bv=bv,
                    num_iters=num_iters,
                    batch_size=batch_size,
                    optimal_state=optimal_state,
                    backend='cuda',
                )
                joblib.dump(pomegranate_results, results_fname)

            print('Pomegranate gpu', pomegranate_results)

            # PGMax
            pgmax_results = rbm_lib.pgmax_infer(
                W=W,
                bh=bh,
                bv=bv,
                num_iters=num_iters,
                batch_size=batch_size,
                optimal_state=optimal_state,
            )
            print(f'PGMax {jax_platform_name}', pgmax_results)
            joblib.dump(
                pgmax_results,
                f'n_units_{n_units}_rbm_idx_{rbm_idx}_pgmax_{jax_platform_name}_num_iters_{num_iters}_batch_size_{batch_size}.joblib',
            )

    # pgmpy
    results_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_pgmpy.joblib'
    if os.path.exists(results_fname):
        pgmpy_results = joblib.load(results_fname)
    else:
        pgmpy_results = rbm_lib.pgmpy_infer(
            W=W, bh=bh, bv=bv, optimal_state=optimal_state
        )
        joblib.dump(pgmpy_results, f'n_units_{n_units}_rbm_idx_{rbm_idx}_pgmpy.joblib')

    print('pgmpy', pgmpy_results)

n_rbms = 20
n_units_list = [40, 60, 80, 100, 120, 140, 160, 180, 200]
# Without ground-truth, pomegranate_results and pgmax
for n_units in n_units_list:
    nh = nv = n_units // 2
    for rbm_idx in range(n_rbms):
        # Generate random RBM and save weights
        rbm_weights_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_weights.joblib'
        if os.path.exists(rbm_weights_fname):
            W, bh, bv = joblib.load(rbm_weights_fname)
        else:
            W = np.random.randn(nh, nv)
            bh = np.random.logistic(size=(nh,))
            bv = np.random.logistic(size=(nv,))
            joblib.dump((W, bh, bv), rbm_weights_fname)

        for num_iters in [20, 200]:
            for batch_size in [1, 100, 1000]:
                # Pomegranate on CPUs
                results_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_pomegranate_cpu_num_iters_{num_iters}_batch_size_{batch_size}.joblib'
                if os.path.exists(results_fname):
                    pomegranate_results = joblib.load(results_fname)
                else:
                    pomegranate_results = rbm_lib.pomegranate_infer(
                        W=W,
                        bh=bh,
                        bv=bv,
                        num_iters=num_iters,
                        batch_size=batch_size,
                        backend='cpu',
                    )
                    joblib.dump(pomegranate_results, results_fname)

                print('Pomegranate cpu', pomegranate_results)

                # Pomegranate on GPUs
                results_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_pomegranate_gpu_num_iters_{num_iters}_batch_size_{batch_size}.joblib'
                if os.path.exists(results_fname):
                    pomegranate_results = joblib.load(results_fname)
                else:
                    pomegranate_results = rbm_lib.pomegranate_infer(
                        W=W,
                        bh=bh,
                        bv=bv,
                        num_iters=num_iters,
                        batch_size=batch_size,
                        backend='cuda',
                    )
                    joblib.dump(pomegranate_results, results_fname)

                print('Pomegranate gpu', pomegranate_results)

                # PGMax
                pgmax_results = rbm_lib.pgmax_infer(
                    W=W,
                    bh=bh,
                    bv=bv,
                    num_iters=num_iters,
                    batch_size=batch_size,
                )
                print(f'PGMax {jax_platform_name}', pgmax_results)
                joblib.dump(
                    pgmax_results,
                    f'n_units_{n_units}_rbm_idx_{rbm_idx}_pgmax_{jax_platform_name}_num_iters_{num_iters}_batch_size_{batch_size}.joblib',
                )

# Without ground-truth, pgmpy. Slow, so saved for the last.
for n_units in n_units_list:
    nh = nv = n_units // 2
    for rbm_idx in range(n_rbms):
        rbm_weights_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_weights.joblib'
        assert os.path.exists(rbm_weights_fname)
        W, bh, bv = joblib.load(rbm_weights_fname)
        # pgmpy
        results_fname = f'n_units_{n_units}_rbm_idx_{rbm_idx}_pgmpy.joblib'
        if os.path.exists(results_fname):
            pgmpy_results = joblib.load(results_fname)
        else:
            pgmpy_results = rbm_lib.pgmpy_infer(W=W, bh=bh, bv=bv)
            joblib.dump(
                pgmpy_results, f'n_units_{n_units}_rbm_idx_{rbm_idx}_pgmpy.joblib'
            )

        print('pgmpy', pgmpy_results)
