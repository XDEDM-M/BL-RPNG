"""LR-label + label-free multires stability router, confirmed on fresh ICs."""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from blrpng.cli.gray_scott_r3 import read
from blrpng.cli.generate_gray_scott_formal import _worker
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import write_index,read_index
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import resolve_in_project,ensure_artifact_path,sha256_file,sha256_json,write_json
from blrpng.training.gray_scott_r3 import load_shards,evaluate
from blrpng.utils import atomic_torch_save

PROTOCOL='artifacts/manifests/gray_scott_r5_safe_router_protocol.json'
ROOT='artifacts/data/gray_scott_r5'
THRESHOLD=.06


def choose_route(candidates, threshold=THRESHOLD):
    """Choose using only LR accuracy after rejecting multires-unstable models."""
    eligible = [name for name, row in candidates.items()
                if row['free20_comm128_256'] <= threshold]
    if eligible:
        return min(eligible, key=lambda name: candidates[name]['lr_validation_balanced'])
    return min(candidates, key=lambda name: candidates[name]['free20_comm128_256'])


def freeze():
    if resolve_in_project(PROTOCOL).exists():return read(PROTOCOL)
    require_capacity(4*2**30)
    r4=read('artifacts/manifests/gray_scott_r4_protocol.json')
    sel=read('artifacts/manifests/gray_scott_r4_selection.json')
    diag=read('artifacts/reports/gray_scott_r4/multires_consistency.json')
    rng=np.random.default_rng(20260919);used=set()
    def scan(obj):
        if isinstance(obj,dict):
            if 'initial_seed' in obj:used.add(int(obj['initial_seed']))
            for v in obj.values():scan(v)
        elif isinstance(obj,list):
            for v in obj:scan(v)
    for path in resolve_in_project('artifacts/manifests').glob('gray_scott*.json'):scan(json.loads(path.read_text()))
    tasks={}
    for task in (0,4,8,12,16):
        candidates={}
        for source in ('final','path'):
            trial=sel['trials']['w03'][str(task)][source]
            row=next(x for x in diag['rows'] if x['task']==task and x['variant']=='w03' and x['source']==source)
            candidates[source]={'checkpoint':trial['checkpoint'],'sha256':trial['checkpoint_sha256'],
                'lr_validation_balanced':trial['best_validation_balanced'],
                'free20_comm128_256':row['free20_comm128_256'],
                'stability_passed':row['free20_comm128_256']<=THRESHOLD}
        chosen=choose_route(candidates)
        spec=r4['trials']['w03']['tasks'][str(task)]
        samples=[]
        for offset in range(2):
            seed=int(rng.integers(2**31-1))
            while seed in used:seed=int(rng.integers(2**31-1))
            used.add(seed);case=14_000_000+task*1000+offset
            samples.append({'case_id':case,'pair_id':case,'initial_condition_id':case,'initial_seed':seed})
        tasks[str(task)]={'chosen':chosen,'candidates':candidates,'samples':samples,
                         **{k:spec[k] for k in ('boundary_seed','boundary_id','boundary_regime','amplitude',
                            'correlation','feed','kill','frequency_low','frequency_high','target_task_id')}}
    value={'frozen':True,'schema':'gray_scott_r5_safe_router_v1','threshold':THRESHOLD,'tasks':tasks,
           'selection_rule':'Among candidates with LR-validation-state free20 128-to256 commutator <=0.06, choose lower LR50 balanced error; if none pass, choose lowest commutator.',
           'selection_inputs':'B64 labeled validation plus analytic-boundary unlabeled 128/256 model consistency; no HR solution',
           'confirmation':{'resolutions':[128,256],'steps':50,'new_ics_per_task':2,'same_development_boundaries':True},
           'r4_selection_sha256':sha256_file('artifacts/manifests/gray_scott_r4_selection.json'),
           'diagnostic_sha256':sha256_file('artifacts/reports/gray_scott_r4/multires_consistency.json'),
           'query_generated_or_read':False,'claim_scope':'Independent new-IC confirmation of a post-R4 development rule; not new-boundary or multi-seed confirmation'}
    value['sha256']=sha256_json(value);write_json(PROTOCOL,value);return value


def protocol():
    p=read(PROTOCOL)
    if sha256_json({k:v for k,v in p.items() if k!='sha256'})!=p['sha256']:raise ValueError('R5 protocol changed')
    if sha256_file('artifacts/manifests/gray_scott_r4_selection.json')!=p['r4_selection_sha256']:raise ValueError('R4 selection changed')
    if sha256_file('artifacts/reports/gray_scott_r4/multires_consistency.json')!=p['diagnostic_sha256']:raise ValueError('R5 diagnostic changed')
    return p


def generate(p,resolution):
    require_capacity(2*2**30)
    for task,spec in p['tasks'].items():
        output=f'{ROOT}/task{int(task):02d}_query_{resolution}.index.json'
        if resolve_in_project(output).exists():read_index(output,verify_hashes=True);continue
        rows=[]
        for order,sample in enumerate(spec['samples']):
            job={k:spec[k] for k in ('boundary_seed','boundary_id','boundary_regime','amplitude','correlation','feed','kill','frequency_low','frequency_high','target_task_id')}
            job.update(sample);job.update(resolution=resolution,steps=50,seed=sample['case_id'],role='r5_fresh_query',
                dt=1.,diffusion_u=2e-5,diffusion_v=1e-5,progress_threshold=.25,fidelity=1,order=order,
                output=f"{ROOT}/task{int(task):02d}_query_{resolution}/pair_{sample['pair_id']}.pt")
            row=_worker(job);row.pop('order')
            if not row['solver_converged'] or row['final_residual_max']>1e-8:raise RuntimeError('R5 solver gate failed')
            rows.append(row);print(json.dumps({'generated':row['path'],'residual':row['final_residual_max']}),flush=True)
        write_index(output,rows,metadata={'equation':'gray_scott','resolution':resolution,'steps':50,
                                         'protocol_sha256':p['sha256'],'router_frozen':True})


def run_evaluation(p,resolution):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');records=[];snapshots={}
    for task,spec in p['tasks'].items():
        index=f'{ROOT}/task{int(task):02d}_query_{resolution}.index.json';payloads=load_shards(index)
        snapshots[task]={'truth':payloads[0]['trajectory'][[0,1,5,10,20,35,50]].clone()}
        method_results={}
        for source in ('final','path'):
            candidate=spec['candidates'][source]
            if sha256_file(candidate['checkpoint'])!=candidate['sha256']:raise ValueError('R5 candidate changed')
            model,_=load_checkpoint_model(candidate['checkpoint'],device=device)
            result=evaluate(model,payloads,device,save_fields=True,teacher=True)
            snapshots[task][source]=result.pop('representatives')[0];method_results[source]=result
            for row in result['records']:
                records.append({**row,'method':source,'resolution':resolution,'seed':42,
                                'checkpoint_sha256':candidate['sha256'],
                                'query_index_sha256':sha256_file(index)})
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        selected=spec['chosen']
        for row in method_results[selected]['records']:
            records.append({**row,'method':'safe_router','routed_source':selected,
                            'resolution':resolution,'seed':42,
                            'checkpoint_sha256':spec['candidates'][selected]['sha256'],
                            'query_index_sha256':sha256_file(index)})
        snapshots[task]['safe_router']=snapshots[task][selected]
        print(json.dumps({'task':task,'routed':selected,'resolution':resolution,
                          'balanced':method_results[selected]['balanced']}),flush=True)
    write_json(f'artifacts/reports/gray_scott_r5/fresh_query_{resolution}.json',{
        'records':records,'protocol_sha256':p['sha256'],'selection_used_query':False,'frozen':True})
    atomic_torch_save(snapshots,ensure_artifact_path(f'artifacts/reports/gray_scott_r5/fields_{resolution}.pt'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','generate','evaluate'))
    parser.add_argument('--resolution',type=int,choices=(128,256));args=parser.parse_args();torch.set_num_threads(2)
    with workspace_audit(f'gray_scott_r5_{args.stage}'):
        if args.stage=='freeze':result=freeze()
        else:
            if args.resolution is None:parser.error('resolution required')
            p=protocol()
            if args.stage=='generate':generate(p,args.resolution)
            else:run_evaluation(p,args.resolution)


if __name__=='__main__':main()
