"""Describe state-support shift using only A training and B LR validation."""
from __future__ import annotations
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blrpng.cli.gray_scott_r3 import read
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.paths import resolve_in_project, ensure_artifact_path, sha256_file, write_json


def describe(index_path):
    rows = []
    for record in read_index(index_path, verify_hashes=True)['shards']:
        payload = torch.load(resolve_in_project(record['path']), weights_only=False, map_location='cpu')
        u, v = payload['trajectory'][:,0,1:-1,1:-1], payload['trajectory'][:,1,1:-1,1:-1]
        rows.append(torch.stack((u.amin((-1,-2)), v.amax((-1,-2)),
                    v.square().mean((-1,-2)).sqrt(), (u*v*v).mean((-1,-2))),dim=-1).numpy())
    return np.stack(rows)


def main():
    torch.set_num_threads(1)
    with workspace_audit('gray_scott_lr_coverage_diagnostic'):
        source_path = 'artifacts/data/gray_scott_r2_seed42/source_train_256.index.json'
        source = describe(source_path)
        lo, hi = source.min((0,1)), source.max((0,1))
        protocol = read('artifacts/manifests/gray_scott_r3b_adaptation_protocol.json')
        targets = {t: describe(s['validation_index']) for t,s in protocol['tasks'].items()}
        summary = {}
        for task, values in targets.items():
            out = (values < lo) | (values > hi)
            summary[task] = {'early_1_to10_fraction_outside': out[:,1:11].mean((0,1)).tolist(),
                             'late_11_to50_fraction_outside': out[:,11:51].mean((0,1)).tolist()}
        keys = ['interior min u','interior max v','interior RMS v','mean u*v²']
        report = {'metrics': keys, 'source_min': lo.tolist(), 'source_max': hi.tolist(),
                  'source_index': source_path, 'source_index_sha256': sha256_file(source_path),
                  'target_index_sha256': {t: sha256_file(s['validation_index']) for t,s in protocol['tasks'].items()},
                  'targets': {t:v.tolist() for t,v in targets.items()}, 'summary': summary,
                  'hr_query_read': False,
                  'interpretation': 'Descriptive extrema/aggregate coverage; not proof that time alone causes failure. Boundary/parameter shifts and grid differences also contribute.'}
        path = 'artifacts/reports/gray_scott_r4/state_coverage.json'
        write_json(path, report)
        fig, axes = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        for i, ax in enumerate(axes.flat):
            ax.axhspan(lo[i],hi[i],color='gray',alpha=.2,label='A256 train: full observed range')
            for task in ('0','4','8','12','16'):
                values = targets[task][:,:,i]
                ax.plot(np.arange(values.shape[1]), values.mean(0), label=f'B64 validation task {task}')
            ax.axvline(10,color='black',linestyle=':')
            ax.set_xlabel('Rollout time step'); ax.set_ylabel(keys[i]); ax.grid(alpha=.2)
        axes[0,0].legend(frameon=False,fontsize=7)
        fig.suptitle('State coverage: source training ends at step 10; target validation continues to 50')
        output = ensure_artifact_path('artifacts/figures/gray_scott_r4/03_state_coverage.png')
        fig.savefig(output,dpi=240);plt.close(fig)
        write_json(output.with_suffix('.json'), {'source':path, 'source_sha256':sha256_file(path),
                   'png_sha256':sha256_file(output), 'selection_performed':False})
        print(json.dumps({'source_min':lo.tolist(),'source_max':hi.tolist(),'metrics':keys,'summary':summary}),flush=True)


if __name__ == '__main__': main()
