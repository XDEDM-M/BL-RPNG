"""Freeze LR-screen figures and measure consistency without HR solution labels."""
from __future__ import annotations
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from blrpng.cli.gray_scott_r3 import read
from blrpng.cli.gray_scott_r4 import verify_protocol, SELECT
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import ensure_artifact_path, sha256_file, write_json
from blrpng.training.gray_scott_r3 import load_shards, geometry, advance
from blrpng.training.gray_scott_commutator import refined_geometry, prolong_lifted, field_commutator


@torch.no_grad()
def measure(trial, payloads, device):
    if sha256_file(trial['checkpoint']) != trial['checkpoint_sha256']:
        raise ValueError('Selected adapter changed')
    model, _ = load_checkpoint_model(trial['checkpoint'], device=device)
    model.eval()
    values = []
    for p in payloads:
        geo = geometry([p], device)
        fine_geo = refined_geometry(p,128,device)
        for step in (0,9,19,29,39):
            state = p['trajectory'][step:step+1].to(device)
            target = p['trajectory'][step+1:step+2].to(device)
            low = advance(model,state,geo)
            high = advance(model,prolong_lifted(state,geo[0],fine_geo),fine_geo)
            values.append(float(field_commutator(high,low,target,geo[1])))
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return float(np.mean(values))


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with workspace_audit('analyze_gray_scott_r4'):
        p, selection = verify_protocol(), read(SELECT)
        rows = []
        for task in (0,4,8,12,16):
            spec = p['trials']['w01']['tasks'][str(task)]
            payloads = load_shards(spec['validation_index'])
            for source in ('final','path'):
                for label in ('w0','w01','w03'):
                    trial = selection['trials'][label][str(task)][source]
                    row = next(r for r in trial['history'] if r['epoch'] == trial['best_epoch'])
                    result = {'task': task, 'source': source, 'weight': {'w0':0.,'w01':.1,'w03':.3}[label],
                              'label':label, 'validation_balanced': trial['best_validation_balanced'],
                              'validation_joint': row['validation_joint'], 'best_epoch': trial['best_epoch'],
                              'validation_commutator': measure(trial,payloads,device),
                              'adapter_sha256': trial['checkpoint_sha256'],
                              'validation_index_sha256': sha256_file(spec['validation_index']),
                              'adaptation_seconds': trial['training_seconds']}
                    rows.append(result); print(json.dumps(result),flush=True)
        output = 'artifacts/reports/gray_scott_r4/lr_screen_analysis.json'
        write_json(output, {'selection_sha256':sha256_file(SELECT),'rows':rows,
                           'hr_solution_read':False,'checkpoint_reselected':False,
                           'gate_passed':selection['may_generate_fresh_hr_query'],
                           'diagnostic': 'One-step 64/128 consistency on five fixed LR validation times; diagnostic does not affect selection.'})
        root = 'artifacts/figures/gray_scott_r4'
        figures = []
        def save(fig,name,note):
            path = ensure_artifact_path(f'{root}/{name}.png')
            fig.savefig(path,dpi=240);plt.close(fig)
            meta = {'png':str(path),'png_sha256':sha256_file(path),'source':output,
                    'source_sha256':sha256_file(output),'note':note,'selection_performed':False}
            write_json(path.with_suffix('.json'),meta);figures.append(meta)
        fig, axes = plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
        for ax,source in zip(axes,('final','path')):
            values = np.array([[next(r['validation_balanced'] for r in rows if r['task']==t and r['source']==source and r['label']==w)
                                for w in ('w0','w01','w03')] for t in (0,4,8,12,16)])*100
            im=ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=max(50,float(values.max())),aspect='auto')
            for i in range(5):
                for j in range(3): ax.text(j,i,f'{values[i,j]:.2f}%',ha='center',va='center')
            ax.set_xticks(range(3),['weight 0','weight 0.1','weight 0.3'])
            ax.set_yticks(range(5),[f'Task {t}' for t in (0,4,8,12,16)])
            ax.set_title(f'{source} A256 pretraining');fig.colorbar(im,ax=ax,label='LR50 equal-field error (%)')
        fig.suptitle('R4 validation screen: no new HR query used')
        save(fig,'01_lr_validation_screen','Best adapters selected only by LR validation; development results, not blind-test gains.')
        fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for source in ('final','path'):
            for ax,key in zip(axes,('validation_commutator','validation_balanced')):
                values=[100*np.mean([r[key] for r in rows if r['source']==source and r['weight']==w]) for w in (0,.1,.3)]
                ax.plot([0,.1,.3],values,'o-',label=source)
        axes[0].set_ylabel('LR-validation-input 64/128 commutator (%)')
        axes[1].set_ylabel('LR50 equal-field error (%)')
        for ax in axes: ax.set_xlabel('Consistency loss weight');ax.legend(frameon=False);ax.grid(alpha=.2)
        save(fig,'02_consistency_vs_accuracy','Reducing grid disagreement need not improve physical trajectory accuracy; both quantities shown.')
        csv_path=ensure_artifact_path(f'{root}/metrics.csv')
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_json(f'{root}/manifest.json',{'figures':figures,'csv':str(csv_path),'csv_sha256':sha256_file(csv_path),
                   'selection_sha256':sha256_file(SELECT),'hr_query_opened':False})
        print(json.dumps({'completed':True,'gate_passed':selection['may_generate_fresh_hr_query']}),flush=True)


if __name__ == '__main__': main()
