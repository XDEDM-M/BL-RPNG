"""Audit completed R3 artifacts and preserve the mixed stage conclusion."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from blrpng.cli.gray_scott_r3 import read
from blrpng.data.capacity import artifact_size_bytes
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.paths import resolve_in_project, sha256_file, write_json, git_status_outside_project


def main():
    with workspace_audit('finalize_gray_scott_r3'):
        source = {s: read(f'artifacts/manifests/gray_scott_r3b_source_{s}.json') for s in ('final', 'path')}
        assert source['final']['initialization_sha256'] == source['path']['initialization_sha256']
        checks = []
        for s in source.values():
            assert sha256_file(s['checkpoint']) == s['sha256']
            checks.append(s['checkpoint'])
        selection = read('artifacts/manifests/gray_scott_r3b_selection.json')
        trials = [r for t in selection['trials'].values() for r in t.values()]
        for r in trials:
            assert r['reload_verified'] and not r['selection_used_hr_query']
            assert sha256_file(r['checkpoint']) == r['checkpoint_sha256']
            checks.append(r['checkpoint'])
        query_index_hashes = {}
        for task in (0,4,8,12,16):
            pairings = []
            for resolution in (64,128,256):
                path = f'artifacts/data/gray_scott_r3/task{task:02d}_query_{resolution}.index.json'
                index = read_index(path, verify_hashes=True)
                pairings.append(sorted((r['pair_id'], r['initial_condition_id'], r['boundary_id']) for r in index['shards']))
                assert all(r['solver_converged'] and r['final_residual_max'] <= 1e-8 for r in index['shards'])
                query_index_hashes[path] = sha256_file(path)
            assert pairings[0] == pairings[1] == pairings[2]
        figures = read('artifacts/figures/gray_scott_r3/manifest.json')
        for fig in figures['figures']:
            assert sha256_file(fig['png']) == fig['png_sha256']
        assert sha256_file(figures['csv']) == figures['csv_sha256']
        outside = git_status_outside_project()
        if outside:
            raise RuntimeError(f'Unexpected outside-project worktree changes: {outside}')
        size = artifact_size_bytes()
        if size > 300_000_000_000:
            raise RuntimeError('Artifact storage exceeds 300 GB')
        summary = read('artifacts/reports/gray_scott_r3/summary.json')
        code_paths = ['blrpng/training/gray_scott_r3.py', 'blrpng/training/engine.py',
                      'blrpng/training/losses.py', 'blrpng/evaluation/rollout.py']
        code_paths += [str(p) for p in Path('blrpng/cli').glob('*gray_scott*r3*.py')]
        code_paths += [str(p) for p in Path('blrpng/cli').glob('diagnose_gray_scott_*.py')]
        report = {
            'stage_completed': True, 'formal_gate_passed': False, 'promote_r3b_to_main_method': False,
            'reason': 'Short-horizon and weak-field improvements coexist with long-horizon joint regression and task12 spot loss; matched path benefit CI crosses zero.',
            'seed': 42, 'scope': 'Five development boundaries with two fresh query ICs each; four overlap historical models',
            'matched_path_50step_balanced_improvement': summary['paired_comparisons']['256/r3b_path_vs_r3b_final/50/balanced'],
            'adaptation_seconds_mean': float(np.mean([r['training_seconds'] for r in trials])),
            'adaptation_trainable_parameters': sorted(set(r['trainable_parameters'] for r in trials)),
            'source': source, 'checked_checkpoints': checks,
            'query_index_hashes': query_index_hashes, 'outside_project_status': outside,
            'artifact_bytes': size, 'png_count': len(figures['figures']),
            'summary_sha256': sha256_file('artifacts/reports/gray_scott_r3/summary.json'),
            'code_sha256': {p: sha256_file(p) for p in sorted(set(code_paths))},
            'next_priority': [
                'Extend source trajectory coverage to late-time states; keep boundary splits and dt fixed.',
                'Use label-free cross-resolution commutation and physically consistent directional response controls, not global contraction.',
                'Match Newton auxiliary against endpoint-increment auxiliary under identical weighting/budget.',
                'Freeze future choices using LR validation and reserve new query ICs/boundaries for confirmation.',
            ],
        }
        write_json('artifacts/reports/gray_scott_r3/stage_conclusion.json', report)
        print(json.dumps({'completed': True, 'promoted': False, 'pngs': report['png_count'],
                          'artifact_gb': size/1e9, 'mean_adaptation_seconds': report['adaptation_seconds_mean']}))


if __name__ == '__main__': main()
