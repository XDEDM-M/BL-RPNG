"""Control source-field weighting before opening any R3 fresh query evaluation."""
from __future__ import annotations

import argparse
import copy
import json

import torch

from blrpng.cli.gray_scott_r3 import protocol, read
from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, sha256_json, write_json
from blrpng.training.adapters import compact_full_checkpoint_for_inference
from blrpng.training.gray_scott_r3 import train_adapter
from blrpng.training.engine import train

FREEZE = 'artifacts/manifests/gray_scott_r3b_source_protocol.json'
ADAPT = 'artifacts/manifests/gray_scott_r3b_adaptation_protocol.json'
SELECT = 'artifacts/manifests/gray_scott_r3b_selection.json'


def freeze():
    if resolve_in_project(FREEZE).exists(): return read(FREEZE)
    for res in (128, 256):
        if resolve_in_project(f'artifacts/reports/gray_scott_r3/fresh_query_{res}.json').exists():
            raise RuntimeError('R3B must be frozen before opening R3 query results')
    require_capacity(14 * 2**30)
    configs = {}
    for source in ('final', 'path'):
        config = load_config('configs/gray_scott_r2_development.yaml')
        config = {k: copy.deepcopy(v) for k, v in config.items() if not k.startswith('_')}
        config['experiment']['name'] = f'gray_scott_r3b_source_{source}'
        config['training'].update(final_loss='field_balanced_relative_l2',
                                  validation_metric='field_balanced_relative_l2',
                                  validation_batch_size=1, path_gradient_ratio=.05,
                                  objective='final_only' if source == 'final' else 'blrpng_safe_annealed')
        config['training']['adapter']['strategy'] = 'full'
        path = f'artifacts/manifests/gray_scott_r3b_{source}_config.json'
        write_json(path, config)
        configs[source] = {'path': path, 'sha256': sha256_file(path)}
    result = {'frozen': True, 'configs': configs, 'seed': 42,
              'source_train': 'artifacts/data/gray_scott_r2_seed42/source_train_256.index.json',
              'source_validation': 'artifacts/data/gray_scott_r2_seed42/source_validation_256.index.json',
              'epochs': 20, 'adaptation_horizon': 3, 'adaptation_epochs': 80,
              'source_from_scratch_matched_initialization': True,
              'query_evaluation_opened': False,
              'rationale': 'R3 LR-only trials found no clear path advantage; remove final-loss field-weight confound in A as well as B.',
              'claim_scope': 'One source seed and development boundaries; not a formal multi-seed stage gate'}
    result['protocol_sha256'] = sha256_json(result)
    write_json(FREEZE, result)
    return result


def train_source(source):
    spec = read(FREEZE)
    cfg_spec = spec['configs'][source]
    if sha256_file(cfg_spec['path']) != cfg_spec['sha256']: raise ValueError('Source config changed')
    config = load_config(cfg_spec['path'])
    output = f'artifacts/checkpoints/gray_scott_r3b/source_{source}.pt'
    recovery = resolve_in_project(f'artifacts/checkpoints/source_{source}_recovery.pt')
    result = train(config, seed=42, objective=config['training']['objective'],
                   train_data_path=spec['source_train'], validation_data_paths=[spec['source_validation']],
                   output_checkpoint=output, resume_checkpoint=str(recovery) if recovery.exists() else None)
    checkpoint = compact_full_checkpoint_for_inference(result['checkpoint'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    write_json(f'artifacts/manifests/gray_scott_r3b_source_{source}.json', {
        'checkpoint': str(checkpoint), 'sha256': sha256_file(checkpoint),
        'initialization_sha256': payload['initialization_sha256'],
        'best_validation_metric': payload['best_validation_metric'],
        'best_metric_name': payload['best_metric_name'],
        'elapsed_seconds': payload['elapsed_seconds'], 'optimizer_steps': payload['optimizer_steps'],
        'peak_gpu_bytes': payload['peak_gpu_memory_bytes'], 'loss_config': payload['loss_config'],
        'query_used': False})


def adapt():
    p = copy.deepcopy(protocol())
    p['namespace'] = 'gray_scott_r3b'
    source_specs = {s: read(f'artifacts/manifests/gray_scott_r3b_source_{s}.json') for s in ('final', 'path')}
    if source_specs['final']['initialization_sha256'] != source_specs['path']['initialization_sha256']:
        raise ValueError('Unmatched source initialization')
    p['sources'] = {s: {'path': source_specs[s]['checkpoint'], 'sha256': source_specs[s]['sha256']}
                    for s in source_specs}
    p['training']['horizons'] = [3]
    p['source_protocol_sha256'] = sha256_file(FREEZE)
    p.pop('protocol_sha256')
    p['protocol_sha256'] = sha256_json(p)
    if resolve_in_project(ADAPT).exists() and read(ADAPT) != p: raise RuntimeError('Adapt protocol changed')
    write_json(ADAPT, p)
    trials = {}
    for task in (0, 4, 8, 12, 16):
        trials[str(task)] = {}
        for source in ('final', 'path'):
            existing = resolve_in_project(f'artifacts/reports/gray_scott_r3b/task{task:02d}_{source}_h3.json')
            result = read(str(existing)) if existing.exists() else train_adapter(p, task, source, 3)
            if result['protocol_sha256'] != p['protocol_sha256']: raise ValueError('Trial protocol changed')
            trials[str(task)][source] = result
    write_json(SELECT, {'frozen': True, 'protocol_sha256': p['protocol_sha256'],
                       'trials': trials, 'query_used_for_selection': False, 'horizon': 3})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'source', 'adapt'))
    parser.add_argument('--source', choices=('final', 'path'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with workspace_audit(f'gray_scott_r3b_{args.stage}'):
        if args.stage == 'freeze': freeze()
        elif args.stage == 'source':
            if args.source is None: parser.error('--source required')
            train_source(args.source)
        else: adapt()
    print(json.dumps({'stage': args.stage, 'completed': True}), flush=True)


if __name__ == '__main__': main()
