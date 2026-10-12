"""R3 freeze / low-resolution adaptation / fresh query / evaluation entry point."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, write_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import (ensure_artifact_path, resolve_in_project, sha256_file,
                         sha256_json, write_json, exclusive_artifact_lock)
from blrpng.training.gray_scott_r3 import load_shards, train_adapter, evaluate
from blrpng.utils import atomic_torch_save

PROTOCOL = 'artifacts/manifests/gray_scott_r3_protocol.json'
SELECTION = 'artifacts/manifests/gray_scott_r3_selection.json'
ROOT = 'artifacts/data/gray_scott_r3'


def read(path):
    return json.loads(resolve_in_project(path, must_exist=True).read_text())


def protocol():
    value = read(PROTOCOL)
    if sha256_json({k: v for k, v in value.items() if k != 'protocol_sha256'}) != value['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    return value


def generate_rows(task, samples, resolution, steps, role):
    # Sequential single-threaded sparse solves avoid the previous nested CPU overload.
    from blrpng.cli.generate_gray_scott_formal import _worker
    rows = []
    for i, sample in enumerate(samples):
        job = {k: task[k] for k in (
            'boundary_seed', 'boundary_id', 'boundary_regime', 'amplitude',
            'correlation', 'feed', 'kill', 'frequency_low', 'frequency_high', 'target_task_id')}
        job.update(sample)
        job.update(resolution=resolution, steps=steps, seed=sample['case_id'], role=role,
                   dt=1.0, diffusion_u=2e-5, diffusion_v=1e-5, progress_threshold=0.25,
                   fidelity=int(resolution > 64), order=i,
                   output=f"{ROOT}/task{task['target_task_id']:02d}_{role}_{resolution}/pair_{sample['pair_id']}.pt")
        row = _worker(job)
        row.pop('order')
        if not row['solver_converged'] or row['final_residual_max'] > 1e-8:
            raise RuntimeError('New query solver did not meet the frozen tolerance')
        rows.append(row)
        print(json.dumps({'generated': row['path'], 'residual': row['final_residual_max']}), flush=True)
    return rows


def freeze():
    if resolve_in_project(PROTOCOL).exists():
        return protocol()
    require_capacity(3 * 2**30)
    old = read('artifacts/manifests/gray_scott_seed42_protocol.json')
    rng = np.random.default_rng(20260917)
    sources = {}
    for label, name in [('final', 'final_only'), ('path', 'blrpng_safe_annealed_r005')]:
        path = f'artifacts/checkpoints/gray_scott_r2/gray_scott_r2_source_s42_{name}.inference.pt'
        sources[label] = {'path': path, 'sha256': sha256_file(path)}
    tasks = {}
    used_initial_seeds = set()
    # Use all prior task ICs for the seed collision audit, including earlier blind queries.
    for p in resolve_in_project('artifacts/manifests').glob('gray_scott*.json'):
        def scan(obj):
            if isinstance(obj, dict):
                if 'initial_seed' in obj: used_initial_seeds.add(int(obj['initial_seed']))
                for v in obj.values(): scan(v)
            elif isinstance(obj, list):
                for v in obj: scan(v)
        scan(json.loads(p.read_text()))
    def samples(task, count, offset):
        result = []
        for i in range(count):
            seed = int(rng.integers(0, 2**31 - 1))
            while seed in used_initial_seeds: seed = int(rng.integers(0, 2**31 - 1))
            used_initial_seeds.add(seed)
            case = 12_000_000 + task * 1000 + offset + i
            result.append({'case_id': case, 'pair_id': case,
                           'initial_condition_id': case, 'initial_seed': seed})
        return result
    for task_id in (0, 4, 8, 12, 16):
        original = old['targets'][task_id if task_id != 16 else 12]
        task = {k: original[k] for k in ('target_task_id', 'boundary_id', 'boundary_seed',
                'boundary_regime', 'correlation', 'amplitude', 'feed', 'kill')}
        task['frequency_low'], task['frequency_high'] = ((5, 8) if task_id in (0, 12) else (1, 4))
        task['label'] = {0: 'High-frequency BC', 4: 'Amplitude OOD', 8: 'Correlation shift',
                         12: 'Compound OOD (F/K + frequency + correlation)',
                         16: 'F/K-only OOD'}[task_id]
        if task_id == 16:
            task.update(target_task_id=16, boundary_id=10016, correlation=0.5, amplitude=0.10)
            task['support_samples'], task['validation_samples'] = samples(16, 4, 0), samples(16, 4, 100)
            task['support_index'] = f'{ROOT}/task16_support_64.index.json'
            task['validation_index'] = f'{ROOT}/task16_validation_64.index.json'
        else:
            task['support_index'] = f'artifacts/data/gray_scott_seed42/task{task_id:02d}_support_64_m4.index.json'
            task['validation_index'] = f'artifacts/data/gray_scott_r2_seed42/task{task_id:02d}_validation_long_64.index.json'
        task['query_samples'] = samples(task_id, 2, 200)
        tasks[str(task_id)] = task
    value = {
        'schema': 'blrpng_gray_scott_r3_v1', 'frozen': True, 'sources': sources, 'tasks': tasks,
        'training': {'seed': 42, 'epochs': 80, 'batch_size': 4, 'lr': 1e-3,
                     'validation_interval': 5, 'horizons': [1, 3]},
        'loss': 'equal-field relL2, per-field RMS denominator floor 1e-3',
        'selection': 'one common horizon for both sources; minimum mean LR50 balanced error over both sources and five tasks',
        'checkpoint_rule': 'minimum LR50 balanced error, raw adapter weights, evaluated every five epochs',
        'query': {'resolutions': [128, 256], 'steps': 50, 'per_task': 2, 'seed_count': 1,
                  'role': 'fresh IC confirmation on development target boundaries; not unseen-boundary confirmation'},
        'resource': {'single_gpu': True, 'cpu_threads': 2, 'solver_workers': 1,
                     'estimated_new_bytes': 3 * 2**30, 'total_limit_gb': 300},
        'interpretation': 'A256 sources unchanged; isolates target-loss/closed-loop correction, not balanced source pretraining',
    }
    value['protocol_sha256'] = sha256_json(value)
    write_json(PROTOCOL, value)
    return value


def prepare(p):
    task = p['tasks']['16']
    for role, steps in [('support', 10), ('validation', 50)]:
        rows = generate_rows(task, task[f'{role}_samples'], 64, steps, role)
        write_index(task[f'{role}_index'], rows, metadata={
            'equation': 'gray_scott', 'resolution': 64, 'steps': steps,
            'protocol_sha256': p['protocol_sha256'], 'role': role})


def select(p):
    trials = {}
    for task in p['tasks']:
        trials[task] = {}
        for source in p['sources']:
            for h in p['training']['horizons']:
                trials[task][f'{source}_h{h}'] = read(
                    f'artifacts/reports/gray_scott_r3/task{int(task):02d}_{source}_h{h}.json')
    means = {str(h): float(np.mean([trial[f'{s}_h{h}']['best_validation_balanced']
                                   for trial in trials.values() for s in p['sources']]))
             for h in p['training']['horizons']}
    selected = min(means, key=means.get)
    result = {'frozen': True, 'protocol_sha256': p['protocol_sha256'],
              'shared_horizon': int(selected), 'mean_validation_by_horizon': means,
              'selection_used_hr_query': False, 'trials': trials,
              'training_code_sha256': sha256_file('blrpng/training/gray_scott_r3.py'),
              'cli_code_sha256': sha256_file('blrpng/cli/gray_scott_r3.py')}
    if resolve_in_project(SELECTION).exists() and read(SELECTION) != result:
        raise RuntimeError('Cannot overwrite an existing frozen selection')
    write_json(SELECTION, result)
    return result


def query(p, resolution):
    sel = read(SELECTION)
    if sel['protocol_sha256'] != p['protocol_sha256']: raise ValueError('Selection mismatch')
    require_capacity(2 * 2**30)
    for task in p['tasks'].values():
        rows = generate_rows(task, task['query_samples'], resolution, 50, 'query')
        write_index(f"{ROOT}/task{task['target_task_id']:02d}_query_{resolution}.index.json", rows,
                    metadata={'equation': 'gray_scott', 'resolution': resolution, 'steps': 50,
                              'protocol_sha256': p['protocol_sha256'], 'selection_sha256': sha256_file(SELECTION)})


def run_evaluation(p, resolution):
    sel = read(SELECTION)
    h = sel['shared_horizon']
    old = read('artifacts/manifests/gray_scott_r2_target_selection.json')
    scratch = read('artifacts/manifests/gray_scott_seed42_saturated_scratch_freeze.json')
    scratch_paths = {str(row['task']): row['selected_path'] for row in scratch['tasks']}
    # R3B was preregistered before any fresh-query evaluation was opened.
    supplemental_selection = 'artifacts/manifests/gray_scott_r3b_selection.json'
    supplement = read(supplemental_selection) if resolve_in_project(supplemental_selection).exists() else None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    records, snapshots = [], {}
    for task in p['tasks']:
        index = f'{ROOT}/task{int(task):02d}_query_{resolution}.index.json'
        payloads = load_shards(index)
        methods = {f'r3_{s}': sel['trials'][task][f'{s}_h{h}']['checkpoint'] for s in p['sources']}
        if supplement is not None:
            methods.update({f'r3b_{s}': supplement['trials'][task][s]['checkpoint'] for s in p['sources']})
        methods.update({f'r3_{s}_h{other}': sel['trials'][task][f'{s}_h{other}']['checkpoint']
                        for s in p['sources'] for other in p['training']['horizons'] if other != h})
        if task in old['checkpoints']:
            methods.update({f'r2_{s}': old['checkpoints'][task][f'{m}_nlt']['path']
                            for s, m in [('final', 'final'), ('path', 'path')]})
            methods['r2_scratch'] = scratch_paths[task]
        snapshots[task] = {'truth': payloads[0]['trajectory'][[0, 1, 5, 10, 20, 35, 50]].clone()}
        for method, checkpoint in methods.items():
            model, _ = load_checkpoint_model(checkpoint, device=device)
            if device.type == 'cuda': torch.cuda.synchronize()
            started = time.perf_counter()
            result = evaluate(model, payloads, device, save_fields=True, teacher=True)
            if device.type == 'cuda': torch.cuda.synchronize()
            diagnostic_seconds = time.perf_counter() - started
            snapshots[task][method] = result.pop('representatives')[0]
            for row in result['records']:
                row.update(method=method, seed=42, resolution=resolution,
                           diagnostic_evaluation_seconds=diagnostic_seconds,
                           diagnostic_timing_includes_teacher_forcing_and_fft=True,
                           checkpoint_sha256=sha256_file(checkpoint), index_sha256=sha256_file(index))
                records.append(row)
            del model
            torch.cuda.empty_cache()
            print(json.dumps({'task': task, 'method': method, 'resolution': resolution,
                              'balanced': result['balanced']}), flush=True)
    report = {'frozen': True, 'protocol_sha256': p['protocol_sha256'],
              'selection_sha256': sha256_file(SELECTION), 'resolution': resolution,
              'supplemental_selection_sha256': sha256_file(supplemental_selection) if supplement else None,
              'records': records, 'selection_used_query': False}
    write_json(f'artifacts/reports/gray_scott_r3/fresh_query_{resolution}.json', report)
    atomic_torch_save(snapshots, ensure_artifact_path(f'artifacts/reports/gray_scott_r3/fields_{resolution}.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'prepare', 'train', 'select', 'query', 'evaluate'])
    parser.add_argument('--task', type=int)
    parser.add_argument('--source', choices=['final', 'path'])
    parser.add_argument('--horizon', type=int, choices=[1, 3])
    parser.add_argument('--resolution', type=int, choices=[128, 256], default=256)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with workspace_audit(f'gray_scott_r3_{args.stage}'):
        if args.stage == 'freeze': result = freeze()
        else:
            p = protocol()
            if args.stage == 'prepare': result = prepare(p)
            elif args.stage == 'train':
                if args.task is None or args.source is None or args.horizon is None:
                    parser.error('train requires --task, --source, --horizon')
                with exclusive_artifact_lock(f'r3_{args.task}_{args.source}_{args.horizon}'):
                    result = train_adapter(p, args.task, args.source, args.horizon)
            elif args.stage == 'select': result = select(p)
            elif args.stage == 'query': result = query(p, args.resolution)
            elif args.stage == 'evaluate': result = run_evaluation(p, args.resolution)
        print(json.dumps({'stage': args.stage, 'completed': True}), flush=True)


if __name__ == '__main__':
    main()
