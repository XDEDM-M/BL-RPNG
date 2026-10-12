"""Label-free 64/128/256 consistency diagnostic on LR validation states."""
from __future__ import annotations
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blrpng.cli.gray_scott_r3 import read
from blrpng.data.generate import workspace_audit
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import ensure_artifact_path, sha256_file, write_json
from blrpng.training.gray_scott_r3 import load_shards, geometry, advance
from blrpng.training.gray_scott_commutator import refined_geometry, prolong_lifted, field_commutator


@torch.no_grad()
def measure(model,payloads,device):
    results=[]
    for p in payloads:
        g64=geometry([p],device);g128=refined_geometry(p,128,device);g256=refined_geometry(p,256,device)
        for step in (0,9,19,29,39):
            s64=p['trajectory'][step:step+1].to(device);target=p['trajectory'][step+1:step+2].to(device)
            s128=prolong_lifted(s64,g64[0],g128);s256=prolong_lifted(s64,g64[0],g256)
            y64=advance(model,s64,g64);y128=advance(model,s128,g128);y256=advance(model,s256,g256)
            target128=prolong_lifted(target,g64[0],g128)
            results.append({'step':step,'comm64_128':float(field_commutator(y128,y64,target,g64[1])),
                            'comm128_256':float(field_commutator(y256,y128,target128,g128[1]))})
    return results


@torch.no_grad()
def measure_free_rollout(model,payloads,device,steps=20):
    results=[]
    for p in payloads:
        g64=geometry([p],device);g128=refined_geometry(p,128,device);g256=refined_geometry(p,256,device)
        states={64:p['trajectory'][:1].to(device)}
        states[128]=prolong_lifted(states[64],g64[0],g128)
        states[256]=prolong_lifted(states[64],g64[0],g256)
        geos={64:g64,128:g128,256:g256}
        for step in range(steps):
            states={r:advance(model,states[r],geos[r]) for r in states}
            target=p['trajectory'][step+1:step+2].to(device)
            target128=prolong_lifted(target,g64[0],g128)
            results.append({'step':step+1,
                'comm64_128':float(field_commutator(states[128],states[64],target,g64[1])),
                'comm128_256':float(field_commutator(states[256],states[128],target128,g128[1]))})
    return results


def main():
    torch.set_num_threads(2);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    selection=read('artifacts/manifests/gray_scott_r4_selection.json')
    p=read('artifacts/manifests/gray_scott_r4_protocol.json')
    rows=[]
    with workspace_audit('gray_scott_multires_consistency'):
        for task in (0,4,8,12,16):
            payloads=load_shards(p['trials']['w03']['tasks'][str(task)]['validation_index'])
            for variant in ('w0','w03'):
                for source in ('final','path'):
                    trial=selection['trials'][variant][str(task)][source]
                    if sha256_file(trial['checkpoint'])!=trial['checkpoint_sha256']:raise ValueError('Adapter changed')
                    model,_=load_checkpoint_model(trial['checkpoint'],device=device)
                    values=measure(model,payloads,device)
                    free=measure_free_rollout(model,payloads,device)
                    rows.append({'task':task,'variant':variant,'source':source,
                                 'comm64_128':float(np.mean([x['comm64_128'] for x in values])),
                                 'comm128_256':float(np.mean([x['comm128_256'] for x in values])),
                                 'free20_comm64_128':float(np.mean([x['comm64_128'] for x in free])),
                                 'free20_comm128_256':float(np.mean([x['comm128_256'] for x in free])),
                                 'records':values,'free_rollout_records':free,
                                 'checkpoint_sha256':trial['checkpoint_sha256']})
                    del model
                    if device.type=='cuda':torch.cuda.empty_cache()
                    print(json.dumps({k:rows[-1][k] for k in ('task','variant','source','comm64_128','comm128_256','free20_comm64_128','free20_comm128_256')}),flush=True)
        path='artifacts/reports/gray_scott_r4/multires_consistency.json'
        write_json(path,{'rows':rows,'selection_sha256':sha256_file('artifacts/manifests/gray_scott_r4_selection.json'),
                         'hr_solution_read':False,'selection_changed':False,
                         'note':'One-step teacher-state and 20-step free-rollout predictions at 64/128/256 from LR validation states; consistency is not physical accuracy.'})
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for ax,key in zip(axes,('comm64_128','comm128_256')):
            x=np.arange(5);width=.18
            for i,(variant,source) in enumerate((('w0','final'),('w0','path'),('w03','final'),('w03','path'))):
                vals=[100*next(r[key] for r in rows if r['task']==t and r['variant']==variant and r['source']==source)
                      for t in (0,4,8,12,16)]
                ax.bar(x+(i-1.5)*width,vals,width,label=f'{variant} {source}')
            ax.set_xticks(x,[f'Task {t}' for t in (0,4,8,12,16)]);ax.set_ylabel(f'{key} (%)');ax.grid(axis='y',alpha=.2)
        axes[0].legend(frameon=False,fontsize=7);fig.suptitle('Label-free cross-resolution disagreement on LR validation states')
        output=ensure_artifact_path('artifacts/figures/gray_scott_r4/08_multires_consistency.png')
        fig.savefig(output,dpi=240);plt.close(fig)
        write_json(output.with_suffix('.json'),{'source':path,'source_sha256':sha256_file(path),
                   'png_sha256':sha256_file(output),'selection_performed':False})
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for ax,key in zip(axes,('free20_comm64_128','free20_comm128_256')):
            x=np.arange(5);width=.18
            for i,(variant,source) in enumerate((('w0','final'),('w0','path'),('w03','final'),('w03','path'))):
                vals=[100*next(r[key] for r in rows if r['task']==t and r['variant']==variant and r['source']==source)
                      for t in (0,4,8,12,16)]
                ax.bar(x+(i-1.5)*width,vals,width,label=f'{variant} {source}')
            ax.set_xticks(x,[f'Task {t}' for t in (0,4,8,12,16)]);ax.set_ylabel(f'{key} (%)');ax.grid(axis='y',alpha=.2)
        axes[0].legend(frameon=False,fontsize=7);fig.suptitle('Label-free 20-step cross-resolution disagreement on LR validation rollouts')
        output=ensure_artifact_path('artifacts/figures/gray_scott_r4/09_free_rollout_multires_consistency.png')
        fig.savefig(output,dpi=240);plt.close(fig)
        write_json(output.with_suffix('.json'),{'source':path,'source_sha256':sha256_file(path),
                   'png_sha256':sha256_file(output),'selection_performed':False})


if __name__=='__main__':main()
