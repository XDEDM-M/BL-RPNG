"""Check physical tangent amplification on the already-opened R2 example."""
from __future__ import annotations

import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, ensure_artifact_path, sha256_file, write_json
from blrpng.solvers.sfepy_gray_scott_worker import _laplace_matrix, _newton_step


def main():
    path = 'artifacts/data/gray_scott_r2_seed42/blind_query/task12_query_256/pair_9001200.pt'
    payload = torch.load(resolve_in_project(path), map_location='cpu', weights_only=False)
    n = 256; h = 1 / (n - 1)
    lap = _laplace_matrix(n, h); identity = sp.eye((n - 2)**2, format='csr')
    feed, kill = payload['meta']['feed'], payload['meta']['kill']
    axis = np.linspace(0, 1, n)[1:-1]; x, y = np.meshgrid(axis, axis, indexing='ij')
    zero = np.zeros((n - 2)**2)
    records = []
    with workspace_audit('diagnose_gray_scott_tangent'):
        for t in (5, 20, 50):
            u, v = payload['trajectory'][t, :, 1:-1, 1:-1].numpy().astype(float).reshape(2, -1)
            jacobian = sp.bmat((
                (identity - 2e-5 * lap + sp.diags(v*v + feed), sp.diags(2*u*v)),
                (sp.diags(-v*v), identity - 1e-5 * lap + sp.diags(-2*u*v + feed + kill)),
            ), format='csc')
            lu = spla.splu(jacobian)
            diagonal = np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(2*np.pi*(3*x+y))
            modes = {'diagonal_zero_trace': diagonal,
                     'background_only': (v.reshape(n-2, n-2) < .01) * diagonal,
                     'spot_amplitude': v.reshape(n-2, n-2) * np.sin(np.pi*x) * np.sin(np.pi*y)}
            for name, mode in modes.items():
                direction = np.concatenate((zero, mode.ravel()))
                direction /= np.linalg.norm(direction)
                tangent = lu.solve(direction)  # dt=1: derivative of implicit step is J^{-1}.
                result = {'time': t, 'direction': name, 'true_solver_derivative_gain': float(np.linalg.norm(tangent))}
                if t == 5 and name in ('background_only', 'spot_amplitude'):
                    epsilon = 1e-3
                    previous = payload['trajectory'][t-1].numpy().astype(float)
                    trace = payload['boundary_values'].numpy().astype(float)
                    perturbed = []
                    for sign in (-1, 1):
                        shifted = previous.copy()
                        shifted[:, 1:-1, 1:-1] += sign * epsilon * direction.reshape(2, n-2, n-2)
                        uf, vf, _, _, ok = _newton_step(
                            shifted[0], shifted[1], trace[0], trace[1], dt=1.,
                            diffusion_u=2e-5, diffusion_v=1e-5, feed=feed, kill=kill,
                            max_iters=20, tolerance=1e-8)
                        if not ok: raise RuntimeError('Finite-difference solver failed')
                        perturbed.append(np.stack((uf, vf))[:, 1:-1, 1:-1].reshape(-1))
                    fd = (perturbed[1] - perturbed[0]) / (2 * epsilon)
                    discrepancy = float(np.linalg.norm(fd - tangent) / np.linalg.norm(tangent))
                    if discrepancy > 1e-3: raise RuntimeError(f'JVP finite difference mismatch: {discrepancy}')
                    result['centered_fd_relative_discrepancy'] = discrepancy
                records.append(result)
                print(json.dumps(result), flush=True)
        report = {'source': path, 'source_sha256': sha256_file(path), 'records': records,
                  'query_scope': 'Already-opened R2 case; no R3 query accessed',
                  'interpretation': 'Directional derivatives at the true solution; not global contraction and not the network finite-error secant.',
                  'norm': 'unweighted joint Euclidean norm of u,v interior', 'dt': 1.0}
        write_json('artifacts/reports/gray_scott_r3/physical_tangent_diagnostic.json', report)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        labels = {'diagonal_zero_trace': 'Diagonal zero-trace perturbation',
                  'background_only': 'Background perturbation', 'spot_amplitude': 'Spot amplitude perturbation'}
        for name in labels:
            rs = [r for r in records if r['direction'] == name]
            ax.plot([r['time'] for r in rs], [r['true_solver_derivative_gain'] for r in rs], 'o-', label=labels[name])
        ax.axhline(1, color='black', linestyle=':'); ax.set_xlabel('Rollout step')
        ax.set_ylabel('True implicit-solver directional derivative gain')
        ax.set_title('Gray–Scott damps background errors but can amplify spot perturbations')
        ax.legend(frameon=False); ax.grid(alpha=.2)
        output = ensure_artifact_path('artifacts/figures/gray_scott_r3/09_physical_tangent_response.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=240); plt.close(fig)
        write_json(output.with_suffix('.json'), {**report, 'png_sha256': sha256_file(output)})


if __name__ == '__main__': main()
