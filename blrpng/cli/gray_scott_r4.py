"""LR-only preregistered screen of symmetric lifted commutation."""
from __future__ import annotations

import argparse
import copy
import json
import numpy as np
import torch

from blrpng.cli.gray_scott_r3 import read
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, sha256_json, write_json, exclusive_artifact_lock
from blrpng.training.gray_scott_r3 import train_adapter

FREEZE = 'artifacts/manifests/gray_scott_r4_protocol.json'
SELECT = 'artifacts/manifests/gray_scott_r4_selection.json'


def freeze():
    if resolve_in_project(FREEZE).exists(): return read(FREEZE)
    require_capacity(4 * 2**30)
    base = read('artifacts/manifests/gray_scott_r3b_adaptation_protocol.json')
    baseline = read('artifacts/manifests/gray_scott_r3b_selection.json')
    trials = {}
    for label, weight in [('w01', .1), ('w03', .3)]:
        p = copy.deepcopy(base)
        p['namespace'] = f'gray_scott_r4_{label}'
        p['training'].update(commutator_weight=weight, commutator_resolution=128)
        p.pop('protocol_sha256')
        p['protocol_sha256'] = sha256_json(p)
        trials[label] = p
    # Independent confirmation ICs are fixed now but no fine solutions are generated.
    rng = np.random.default_rng(20260918)
    used = set()
    def visit(obj):
        if isinstance(obj, dict):
            if 'initial_seed' in obj: used.add(int(obj['initial_seed']))
            for v in obj.values(): visit(v)
        elif isinstance(obj, list):
            for v in obj: visit(v)
    for path in resolve_in_project('artifacts/manifests').glob('gray_scott*.json'):
        visit(json.loads(path.read_text()))
    confirmation = {}
    for task in (0,4,8,12,16):
        cases = []
        for offset in range(2):
            seed = int(rng.integers(2**31-1))
            while seed in used: seed = int(rng.integers(2**31-1))
            used.add(seed)
            case = 13_000_000 + task*1000 + offset
            cases.append(dict(case_id=case, pair_id=case, initial_condition_id=case, initial_seed=seed))
        confirmation[str(task)] = cases
    result = {'frozen': True, 'stage': 'R4 LR-only commutator screen', 'trials': trials,
              'baseline_selection': 'artifacts/manifests/gray_scott_r3b_selection.json',
              'baseline_selection_sha256': sha256_file('artifacts/manifests/gray_scott_r3b_selection.json'),
              'new_query_samples': confirmation,
              'selection': 'One common weight for both sources and all five tasks, including zero baseline; minimum mean LR50 equal-field error.',
              'fine_query_gate': {'mean_lr_improvement_at_least': .05, 'path_per_task_balanced_max_ratio': 1.1,
                                  'path_per_task_joint_max_ratio': 1.1},
              'consistency': 'Symmetric 3-step 64-to128 restriction loss, first shuffled batch member; fixed LR-label denominator, analytic fine boundary, no HR solution.',
              'fine_solutions_read_for_training_or_selection': False,
              'new_query_generated': False,
              'code_sha256': {f: sha256_file(f) for f in ('blrpng/training/gray_scott_r3.py',
                                  'blrpng/training/gray_scott_commutator.py', 'blrpng/cli/gray_scott_r4.py')}}
    result['sha256'] = sha256_json(result)
    write_json(FREEZE, result)
    return result


def verify_protocol():
    p = read(FREEZE)
    if sha256_json({k:v for k,v in p.items() if k != 'sha256'}) != p['sha256']:
        raise ValueError('R4 protocol changed')
    for f, digest in p['code_sha256'].items():
        if sha256_file(f) != digest: raise ValueError(f'Frozen R4 code changed: {f}')
    if sha256_file(p['baseline_selection']) != p['baseline_selection_sha256']:
        raise ValueError('Baseline changed')
    return p


def train_all():
    p = verify_protocol()
    for label, spec in p['trials'].items():
        for task in (0,4,8,12,16):
            for source in ('final','path'):
                path = f"artifacts/reports/{spec['namespace']}/task{task:02d}_{source}_h3.json"
                if resolve_in_project(path).exists():
                    r = read(path)
                    if r['protocol_sha256'] != spec['protocol_sha256']:
                        raise ValueError('R4 existing trial protocol changed')
                else:
                    with exclusive_artifact_lock(f'r4_{label}_{task}_{source}'):
                        r = train_adapter(spec, task, source, 3)
                print(json.dumps({'completed': r['trial'], 'variant': label,
                                  'best_lr_balanced': r['best_validation_balanced']}), flush=True)


def select():
    p = verify_protocol()
    trials = {'w0': read(p['baseline_selection'])['trials']}
    for label, spec in p['trials'].items():
        trials[label] = {str(t): {s: read(f"artifacts/reports/{spec['namespace']}/task{t:02d}_{s}_h3.json")
                                for s in ('final','path')} for t in (0,4,8,12,16)}
    means = {k: float(np.mean([r['best_validation_balanced'] for t in tasks.values() for r in t.values()]))
             for k,tasks in trials.items()}
    chosen = min(means, key=means.get)
    checks = {'nonzero_weight': chosen != 'w0', 'mean_improvement': means[chosen] <= .95*means['w0']}
    for t in (0,4,8,12,16):
        def best_row(label):
            r = trials[label][str(t)]['path']
            return next(v for v in r['history'] if v['epoch'] == r['best_epoch'])
        candidate, baseline = best_row(chosen), best_row('w0')
        for m in ('balanced','joint'):
            checks[f'task{t}_{m}_safety'] = candidate[f'validation_{m}'] <= 1.1*baseline[f'validation_{m}']
    result = {'frozen': True, 'protocol_sha256': p['sha256'], 'chosen': chosen,
              'means': means, 'checks': checks, 'may_generate_fresh_hr_query': all(checks.values()),
              'selection_used_hr_query': False, 'trials': trials}
    if resolve_in_project(SELECT).exists() and read(SELECT) != result:
        raise RuntimeError('Cannot replace frozen R4 selection')
    write_json(SELECT, result)
    print(json.dumps({k: result[k] for k in ('chosen','means','checks','may_generate_fresh_hr_query')}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze','train','select'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with workspace_audit(f'gray_scott_r4_{args.stage}'):
        if args.stage == 'freeze': freeze()
        elif args.stage == 'train': train_all()
        else: select()


if __name__ == '__main__': main()
