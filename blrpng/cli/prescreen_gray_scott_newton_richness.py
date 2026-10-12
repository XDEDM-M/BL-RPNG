"""Cheap solver-only gate for a Gray-Scott regime with nonredundant Newton labels."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.cli.gray_scott_r3 import read
from blrpng.data.generate import workspace_audit
from blrpng.paths import ensure_artifact_path, sha256_file, write_json
from blrpng.solvers.gray_scott import generate_gray_scott_rollout

OUT = 'artifacts/reports/gray_scott_r6/newton_richness_prescreen.json'
DTS = (1.0, 2.0, 4.0, 8.0, 12.0)
TASKS = (0, 4, 8, 12, 16)


def main():
    torch.set_num_threads(1)
    r4_path = 'artifacts/manifests/gray_scott_r4_protocol.json'
    r4 = read(r4_path)
    records = []
    with workspace_audit('prescreen_gray_scott_newton_richness'):
        for dt in DTS:
            for task in TASKS:
                spec = r4['trials']['w03']['tasks'][str(task)]
                rollout = generate_gray_scott_rollout(
                    resolution=64, steps=5, seed=15_000_000 + task,
                    case_id=15_000_000 + task, initial_seed=20260920 + task,
                    boundary_seed=spec['boundary_seed'], boundary_id=spec['boundary_id'],
                    target_task_id=task, initial_condition_id=15_000_000 + task,
                    pair_id=15_000_000 + task, role='newton_richness_prescreen',
                    correlation=spec['correlation'], amplitude=spec['amplitude'], dt=dt,
                    diffusion_u=2e-5, diffusion_v=1e-5, feed=spec['feed'], kill=spec['kill'],
                    fidelity=0, progress_threshold=.25,
                    boundary_regime=spec['boundary_regime'],
                    frequency_low=spec['frequency_low'], frequency_high=spec['frequency_high'])
                scales = torch.tensor((1., .25), dtype=torch.float64)[None, :, None, None]
                correction = rollout.newton_correction.double() / scales
                endpoint = rollout.trajectory.double().diff(dim=0) / scales
                a, b = correction.flatten(1), endpoint.flatten(1)
                an, bn = a.norm(dim=1), b.norm(dim=1)
                cosine = (a * b).sum(1) / (an * bn).clamp_min(1e-30)
                discrepancy = (a - b).norm(dim=1) / bn.clamp_min(1e-30)
                for step in range(5):
                    records.append({
                        'dt': dt, 'task': task, 'step': step,
                        'converged': bool(rollout.meta['solver_converged']),
                        'final_residual': float(rollout.meta['final_residual_norms'][step]),
                        'newton_iterations': int(rollout.meta['nonlinear_iterations'][step]),
                        'valid': bool(rollout.valid_correction_mask[step].all()),
                        'selected_progress': float(rollout.residual_progress[step, 0]),
                        'cosine': float(cosine[step]),
                        'relative_vector_discrepancy': float(discrepancy[step]),
                        'endpoint_scaled_norm': float(bn[step]),
                        'state_min_u': float(rollout.trajectory[step + 1, 0].min()),
                        'state_max_u': float(rollout.trajectory[step + 1, 0].max()),
                        'state_min_v': float(rollout.trajectory[step + 1, 1].min()),
                        'state_max_v': float(rollout.trajectory[step + 1, 1].max()),
                    })
                print(json.dumps({'dt': dt, 'task': task,
                                  'median_discrepancy_percent': 100 * float(np.median(discrepancy.numpy())),
                                  'iterations': rollout.meta['nonlinear_iterations'],
                                  'max_residual': max(rollout.meta['final_residual_norms'])}), flush=True)
        summaries, eligible = {}, []
        for dt in DTS:
            rows = [r for r in records if r['dt'] == dt and r['valid']]
            discrepancy = np.asarray([r['relative_vector_discrepancy'] for r in rows])
            iterations = np.asarray([r['newton_iterations'] for r in rows])
            residuals = np.asarray([r['final_residual'] for r in rows])
            row = {
                'steps': len(rows), 'all_converged': all(r['converged'] for r in rows),
                'valid_fraction': len(rows) / 25,
                'max_final_residual': float(residuals.max()),
                'discrepancy_mean': float(discrepancy.mean()),
                'discrepancy_median': float(np.median(discrepancy)),
                'discrepancy_p25': float(np.quantile(discrepancy, .25)),
                'discrepancy_p95': float(np.quantile(discrepancy, .95)),
                'cosine_mean': float(np.mean([r['cosine'] for r in rows])),
                'iterations_mean': float(iterations.mean()),
                'iterations_median': float(np.median(iterations)),
                'physical_range': {
                    'u_min': min(r['state_min_u'] for r in rows), 'u_max': max(r['state_max_u'] for r in rows),
                    'v_min': min(r['state_min_v'] for r in rows), 'v_max': max(r['state_max_v'] for r in rows)},
            }
            row['richness_gate_passed'] = bool(
                row['all_converged'] and row['valid_fraction'] >= .95 and
                row['max_final_residual'] <= 1e-8 and
                row['discrepancy_median'] >= .05 and row['discrepancy_p25'] >= .01 and
                row['iterations_median'] >= 3)
            summaries[str(dt)] = row
            if row['richness_gate_passed']:
                eligible.append(dt)
        report = {
            'schema': 'gray_scott_newton_richness_prescreen_v1',
            'resolution': 64, 'steps_per_task': 5, 'tasks': list(TASKS), 'dts': list(DTS),
            'source_protocol': r4_path, 'source_protocol_sha256': sha256_file(r4_path),
            'selection_uses_model_or_hr_query': False,
            'gate': {'all_converged': True, 'valid_fraction_min': .95,
                     'max_final_residual': 1e-8, 'median_discrepancy_min': .05,
                     'p25_discrepancy_min': .01, 'median_newton_iterations_min': 3},
            'summary': summaries, 'eligible_dts': eligible,
            'selected_dt': min(eligible) if eligible else None,
            'decision': ('pilot_matched_endpoint_vs_newton_training' if eligible else
                         'do_not_expand_gray_scott_path_training; proceed_to_hyperelastic'),
            'records': records,
        }
        write_json(OUT, report)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        x = np.asarray(DTS)
        axes[0].plot(x, [100 * summaries[str(dt)]['discrepancy_median'] for dt in DTS], 'o-', label='median')
        axes[0].plot(x, [100 * summaries[str(dt)]['discrepancy_p25'] for dt in DTS], 's--', label='p25')
        axes[0].axhline(5, color='#D55E00', linestyle=':', label='median gate')
        axes[0].set_ylabel('Newton target vs endpoint discrepancy (%)'); axes[0].legend(frameon=False)
        axes[1].plot(x, [summaries[str(dt)]['iterations_median'] for dt in DTS], 'o-')
        axes[1].axhline(3, color='#D55E00', linestyle=':'); axes[1].set_ylabel('Median accepted Newton updates')
        axes[2].semilogy(x, [summaries[str(dt)]['max_final_residual'] for dt in DTS], 'o-')
        axes[2].axhline(1e-8, color='#D55E00', linestyle=':'); axes[2].set_ylabel('Maximum final residual')
        for ax in axes: ax.set_xlabel('Implicit step size dt'); ax.grid(alpha=.2)
        fig.suptitle('Gray–Scott solver-only Newton-information prescreen (64²)')
        path = ensure_artifact_path('artifacts/figures/gray_scott_r6/01_newton_richness_prescreen.png')
        path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=240); plt.close(fig)
        write_json(path.with_suffix('.json'), {'source': OUT, 'source_sha256': sha256_file(OUT),
                   'png_sha256': sha256_file(path), 'selection_performed': True,
                   'selection_does_not_use_model_or_hr_query': True})
        print(json.dumps({'eligible_dts': eligible, 'selected_dt': report['selected_dt'],
                          'decision': report['decision']}), flush=True)


if __name__ == '__main__':
    main()
