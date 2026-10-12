"""Audit whether cumulative Newton targets differ from endpoint increments.

This is a descriptive source-training-data diagnostic, not a causal test of
the information content of Newton supervision.
"""
from __future__ import annotations

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, ensure_artifact_path, sha256_file, write_json


def main():
    torch.set_num_threads(1)
    index_path = 'artifacts/data/gray_scott_r2_seed42/source_train_256.index.json'
    index = json.loads(resolve_in_project(index_path).read_text())
    records = []
    with workspace_audit('gray_scott_path_information'):
        for shard in index['shards']:
            if sha256_file(shard['path']) != shard['sha256']:
                raise ValueError('Source shard hash changed')
            p = torch.load(resolve_in_project(shard['path']), map_location='cpu', weights_only=False)
            scales = torch.tensor(p['meta']['field_scales'], dtype=torch.float64)[None, :, None, None]
            correction = p['newton_correction'].double() / scales
            increment = p['trajectory'].double().diff(dim=0) / scales
            a, b = correction.flatten(1), increment.flatten(1)
            an, bn = a.norm(dim=1), b.norm(dim=1)
            cosine = (a * b).sum(1) / (an * bn).clamp_min(1e-30)
            discrepancy = (a - b).norm(dim=1) / bn.clamp_min(1e-30)
            valid = p['valid_correction_mask'].reshape(len(a), -1).all(1)
            for step in range(len(a)):
                records.append({'pair_id': shard['pair_id'], 'step': step,
                                'valid': bool(valid[step]), 'cosine': float(cosine[step]),
                                'relative_vector_discrepancy': float(discrepancy[step]),
                                'selected_progress': float(p['residual_progress'][step].flatten()[0])})
        selected = [r for r in records if r['valid']]
        stats = {}
        for key in ('cosine', 'relative_vector_discrepancy', 'selected_progress'):
            v = np.asarray([r[key] for r in selected])
            stats[key] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()),
                          'median': float(np.median(v)), 'p95': float(np.quantile(v, .95))}
        report = {'index': index_path, 'index_sha256': sha256_file(index_path),
                  'rollouts': len(index['shards']), 'valid_steps': len(selected),
                  'precision': 'float64 arithmetic on stored float32 labels',
                  'target_definition': 'cumulative displacement from initial Newton iterate to selected accepted iterate',
                  'norm': 'joint L2 after field scaling [1, 0.25]', 'statistics': stats, 'records': records,
                  'interpretation': 'Near-collinearity suggests an endpoint-increment auxiliary control is necessary; it does not establish causal redundancy or absence of optimization benefits.'}
        write_json('artifacts/reports/gray_scott_r3/path_information_diagnostic.json', report)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].hist([100*r['relative_vector_discrepancy'] for r in selected], bins=35)
        axes[0].set_xlabel('Newton target vs endpoint increment discrepancy (%)')
        axes[0].set_ylabel('Source training steps')
        axes[1].hist([r['selected_progress'] for r in selected], bins=30)
        axes[1].set_xlabel('Selected normalized log-residual progress')
        axes[1].set_ylabel('Source training steps')
        fig.suptitle('Does the selected Newton target contain a distinct intermediate displacement?')
        output = ensure_artifact_path('artifacts/figures/gray_scott_r3/10_path_target_information.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=240); plt.close(fig)
        write_json(output.with_suffix('.json'), {'source': 'artifacts/reports/gray_scott_r3/path_information_diagnostic.json',
                   'statistics': stats, 'png_sha256': sha256_file(output)})
        print(json.dumps({'rollouts': len(index['shards']), 'valid_steps': len(selected), 'statistics': stats}), flush=True)


if __name__ == '__main__': main()
