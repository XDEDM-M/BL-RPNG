"""Generate/evaluate fresh confirmation cases only after the frozen LR gate."""
from __future__ import annotations
import argparse
import json
import torch
from blrpng.cli.gray_scott_r3 import read
from blrpng.cli.gray_scott_r4 import verify_protocol, SELECT
from blrpng.cli.generate_gray_scott_formal import _worker
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import write_index, read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import resolve_in_project, ensure_artifact_path, sha256_file, write_json
from blrpng.training.gray_scott_r3 import load_shards, evaluate
from blrpng.utils import atomic_torch_save

ROOT = 'artifacts/data/gray_scott_r4'
MODELS = 'artifacts/manifests/gray_scott_r4_confirmation_models.json'


def frozen_models(p, selection):
    if resolve_in_project(MODELS).exists(): return read(MODELS)
    old = read('artifacts/manifests/gray_scott_r2_target_selection.json')
    models = {}
    for task in p['new_query_samples']:
        current = {}
        for label, variant in [('base','w0'),('comm',selection['chosen'])]:
            for source in ('final','path'):
                r = selection['trials'][variant][task][source]
                current[f'{label}_{source}'] = {'path':r['checkpoint'],'sha256':r['checkpoint_sha256']}
        if task in old['checkpoints']:
            for source in ('final','path'):
                r = old['checkpoints'][task][f'{source}_nlt']
                current[f'r2_{source}'] = {'path':r['path'],'sha256':sha256_file(r['path'])}
        models[task] = current
    result = {'frozen':True,'selection_sha256':sha256_file(SELECT),'models':models,
              'historical_models_are_references_not_candidates':True,'query_evaluation_opened':False}
    write_json(MODELS,result)
    return result


def generate(p, selection, resolution):
    frozen_models(p,selection)
    require_capacity(2 * 2**30)
    for task, samples in p['new_query_samples'].items():
        path = f'{ROOT}/task{int(task):02d}_query_{resolution}.index.json'
        if resolve_in_project(path).exists():
            index=read_index(path,verify_hashes=True)
            if index['metadata']['selection_sha256'] != sha256_file(SELECT):
                raise ValueError('Existing query belongs to a different selection')
            continue
        spec = p['trials']['w01']['tasks'][task]
        rows=[]
        for order,sample in enumerate(samples):
            job={k:spec[k] for k in ('boundary_seed','boundary_id','boundary_regime','amplitude',
                 'correlation','feed','kill','frequency_low','frequency_high','target_task_id')}
            job.update(sample)
            job.update(resolution=resolution,steps=50,seed=sample['case_id'],role='r4_fresh_query',
                dt=1.,diffusion_u=2e-5,diffusion_v=1e-5,progress_threshold=.25,fidelity=1,order=order,
                output=f"{ROOT}/task{int(task):02d}_query_{resolution}/pair_{sample['pair_id']}.pt")
            row=_worker(job);row.pop('order')
            if not row['solver_converged'] or row['final_residual_max']>1e-8:
                raise RuntimeError('Fresh query solver gate failed')
            rows.append(row)
            print(json.dumps({'generated':row['path'],'residual':row['final_residual_max']}),flush=True)
        write_index(path,rows,metadata={'resolution':resolution,'steps':50,'equation':'gray_scott',
                    'selection_sha256':sha256_file(SELECT),'protocol_sha256':p['sha256']})


def test(p,selection,resolution):
    models=frozen_models(p,selection)
    if models['selection_sha256'] != sha256_file(SELECT): raise ValueError('Selection changed')
    for task in models['models']:
        index = read_index(f'{ROOT}/task{int(task):02d}_query_{resolution}.index.json', verify_hashes=True)
        if index['metadata']['selection_sha256'] != sha256_file(SELECT):
            raise ValueError('Query index does not match frozen selection')
        if resolution == 256:
            paired = read_index(f'{ROOT}/task{int(task):02d}_query_128.index.json', verify_hashes=True)
            keys = lambda data: sorted((r['pair_id'],r['initial_condition_id'],r['boundary_id']) for r in data['shards'])
            if keys(index) != keys(paired): raise ValueError('Unpaired confirmation resolutions')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    records=[];snapshots={}
    for task, methods in models['models'].items():
        index=f'{ROOT}/task{int(task):02d}_query_{resolution}.index.json'
        payloads=load_shards(index)
        snapshots[task]={'truth':payloads[0]['trajectory'][[0,1,5,10,20,35,50]].clone()}
        for method,spec in methods.items():
            if sha256_file(spec['path']) != spec['sha256']: raise ValueError('Frozen model changed')
            model,_=load_checkpoint_model(spec['path'],device=device)
            result=evaluate(model,payloads,device,save_fields=True,teacher=True)
            snapshots[task][method]=result.pop('representatives')[0]
            records.extend([{**r,'method':method,'resolution':resolution,'seed':42,
                             'checkpoint_sha256':spec['sha256'],'index_sha256':sha256_file(index)}
                            for r in result['records']])
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
            print(json.dumps({'task':task,'method':method,'resolution':resolution,
                              'balanced':result['balanced']}),flush=True)
    write_json(f'artifacts/reports/gray_scott_r4/fresh_query_{resolution}.json',{
        'selection_sha256':sha256_file(SELECT),'models_manifest_sha256':sha256_file(MODELS),
        'records':records,'selection_used_query':False,'frozen':True})
    atomic_torch_save(snapshots,ensure_artifact_path(f'artifacts/reports/gray_scott_r4/fields_{resolution}.pt'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('generate','evaluate'))
    parser.add_argument('--resolution',type=int,choices=(128,256),required=True);args=parser.parse_args()
    torch.set_num_threads(1 if args.stage=='generate' else 2)
    p,selection=verify_protocol(),read(SELECT)
    if not selection['may_generate_fresh_hr_query']:
        raise PermissionError('R4 LR gate failed: fresh HR query must remain unopened')
    with workspace_audit(f'gray_scott_r4_{args.stage}_{args.resolution}'):
        if args.stage=='generate':generate(p,selection,args.resolution)
        else:test(p,selection,args.resolution)


if __name__=='__main__':main()
