"""Paired grid-discrepancy diagnostic, without changing any selected model."""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from blrpng.cli.gray_scott_r3 import protocol, query, read
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, ensure_artifact_path, sha256_file, write_json
from blrpng.training.gray_scott_r3 import load_shards, field_errors, evaluate
from blrpng.evaluation.checkpoint import load_checkpoint_model


def evaluate_models(p):
    selection = read('artifacts/manifests/gray_scott_r3b_selection.json')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    records = []
    for task in p['tasks']:
        payloads = load_shards(f'artifacts/data/gray_scott_r3/task{int(task):02d}_query_64.index.json')
        for source in ('final', 'path'):
            trial = selection['trials'][task][source]
            if sha256_file(trial['checkpoint']) != trial['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            model, _ = load_checkpoint_model(trial['checkpoint'], device=device)
            result = evaluate(model, payloads, device)
            records.extend([{**r, 'method': f'r3b_{source}', 'resolution': 64}
                            for r in result['records']])
            del model
            if device.type == 'cuda': torch.cuda.empty_cache()
    for resolution in (128, 256):
        records.extend([r for r in read(f'artifacts/reports/gray_scott_r3/fresh_query_{resolution}.json')['records']
                        if r['method'] in ('r3b_final', 'r3b_path')])
    means = {}
    for resolution in (64, 128, 256):
        for method in ('r3b_final', 'r3b_path'):
            rows = [r for r in records if r['resolution'] == resolution and r['method'] == method]
            fe = np.asarray([r['field_errors'] for r in rows])
            means[f'{resolution}/{method}'] = {'u': float(fe[:,:,0].mean()), 'v': float(fe[:,:,1].mean()),
                'balanced': float(fe.mean()), 'joint': float(np.asarray([r['joint_errors'] for r in rows]).mean())}
    target = 'artifacts/reports/gray_scott_r3/resolution_transfer_diagnostic.json'
    write_json(target, {'post_query_diagnostic': True, 'selection_changed': False, 'records': records,
                       'selection_sha256': sha256_file('artifacts/manifests/gray_scott_r3b_selection.json'),
                       'means_50_steps_all5_tasks': means})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, field in zip(axes, ('joint', 'v')):
        for method in ('r3b_final', 'r3b_path'):
            ax.plot([64,128,256], [100*means[f'{r}/{method}'][field] for r in (64,128,256)],
                    'o-', label=method.replace('r3b_', 'R3B '))
        ax.set_xticks([64,128,256], ['64²','128²','256²'])
        ax.set_ylabel(f'50-step {field} relL2 (%)'); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle('Same frozen models, boundaries and fresh initial-function seeds across resolutions')
    output = ensure_artifact_path('artifacts/figures/gray_scott_r3/14_paired_resolution_transfer.png')
    fig.savefig(output, dpi=240); plt.close(fig)
    write_json(output.with_suffix('.json'), {'source': target, 'source_sha256': sha256_file(target),
                'png_sha256': sha256_file(output), 'selection_performed': False})
    print(json.dumps(means), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-models', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    p = protocol()
    with workspace_audit('gray_scott_resolution_diagnostic'):
        if not all(resolve_in_project(f'artifacts/data/gray_scott_r3/task{int(t):02d}_query_64.index.json').exists()
                   for t in p['tasks']):
            query(p, 64)
        records = []
        for task in p['tasks']:
            coarse = load_shards(f'artifacts/data/gray_scott_r3/task{int(task):02d}_query_64.index.json')
            fine = load_shards(f'artifacts/data/gray_scott_r3/task{int(task):02d}_query_256.index.json')
            for c, f in zip(coarse, fine):
                if c['ids']['pair_id'] != f['ids']['pair_id']:
                    raise ValueError('Unpaired discretization diagnostic')
                truth = f['trajectory'][1:]
                interpolated = F.interpolate(c['trajectory'][1:], size=(256, 256),
                                             mode='bilinear', align_corners=True)
                projected = torch.where(f['constraint_mask'][None], f['boundary_values'][None], interpolated)
                for name, estimate in [('raw_interpolation', interpolated), ('exact_trace_projection', projected)]:
                    fields = field_errors(estimate, truth)
                    joint = (estimate-truth).flatten(1).norm(dim=1) / truth.flatten(1).norm(dim=1)
                    records.append({'task': int(task), 'pair_id': c['ids']['pair_id'], 'method': name,
                                    'field_errors': fields.tolist(), 'joint_errors': joint.tolist()})
        stats = {}
        for method in ('raw_interpolation', 'exact_trace_projection'):
            rows = [r for r in records if r['method'] == method]
            for horizon in (20, 50):
                fe = np.asarray([r['field_errors'][:horizon] for r in rows])
                stats[f'{method}/{horizon}'] = {'u': float(fe[:,:,0].mean()), 'v': float(fe[:,:,1].mean()),
                    'balanced': float(fe.mean()),
                    'joint': float(np.asarray([r['joint_errors'][:horizon] for r in rows]).mean())}
        report = {'protocol_sha256': p['protocol_sha256'], 'post_query_diagnostic': True,
                  'checkpoint_selection_changed': False, 'records': records, 'means': stats,
                  'interpretation': 'Coarse numerical solution interpolated to fine grid is a reference, not a rigorous surrogate-error lower bound.',
                  'note': 'Same physical parameters and initial-function seeds; coarse and fine sampling of the initial field differ.'}
        target = 'artifacts/reports/gray_scott_r3/discretization_diagnostic.json'
        write_json(target, report)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, field in zip(axes, (0, 1)):
            for task in (0, 4, 8, 12, 16):
                rs = [r for r in records if r['task'] == task and r['method'] == 'exact_trace_projection']
                curves = np.asarray([r['field_errors'] for r in rs])
                ax.plot(np.arange(1,51), 100*curves[:,:,field].mean(0), label=f'Task {task}')
            ax.set_xlabel('Rollout step'); ax.set_ylabel(f'{"uv"[field]} relative L2 (%)')
            ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
        fig.suptitle('64² numerical solution interpolated to 256² vs fine truth (exact boundary projected)')
        output = ensure_artifact_path('artifacts/figures/gray_scott_r3/13_discretization_reference.png')
        fig.savefig(output, dpi=240); plt.close(fig)
        write_json(output.with_suffix('.json'), {'source': target, 'source_sha256': sha256_file(target),
                    'png_sha256': sha256_file(output), 'selection_performed': False})
        print(json.dumps(stats), flush=True)
        if args.evaluate_models:
            evaluate_models(p)


if __name__ == '__main__': main()
