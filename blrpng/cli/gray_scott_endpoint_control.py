"""Matched endpoint-increment auxiliary control; no target query in selection."""
from __future__ import annotations
import argparse
import copy
import json
import torch
from blrpng.cli.gray_scott_r3 import read
from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, sha256_json, write_json
from blrpng.training.engine import train
from blrpng.training.adapters import compact_full_checkpoint_for_inference
from blrpng.training.gray_scott_r3 import train_adapter, load_shards, evaluate
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.utils import atomic_torch_save

FREEZE='artifacts/manifests/gray_scott_endpoint_control_protocol.json'
ADAPT='artifacts/manifests/gray_scott_endpoint_control_adaptation.json'
SELECT='artifacts/manifests/gray_scott_endpoint_control_selection.json'


def freeze():
    if resolve_in_project(FREEZE).exists():return read(FREEZE)
    require_capacity(8*2**30)
    config=copy.deepcopy(read('artifacts/manifests/gray_scott_r3b_path_config.json'))
    config['experiment']['name']='gray_scott_endpoint_control'
    config['training']['auxiliary_target']='endpoint_increment'
    config_path='artifacts/manifests/gray_scott_endpoint_control_config.json'
    write_json(config_path,config)
    p={'frozen':True,'config':config_path,'config_sha256':sha256_file(config_path),
       'reference':'artifacts/manifests/gray_scott_r3b_source_path.json',
       'source_train':'artifacts/data/gray_scott_r2_seed42/source_train_256.index.json',
       'source_validation':'artifacts/data/gray_scott_r2_seed42/source_validation_256.index.json',
       'code_sha256':sha256_file('blrpng/training/engine.py'),
       'seed':42,'epochs':20,'optimizer_steps':10000,
       'scope':'Mechanism-development control motivated by source-label near-collinearity; not a new blind-query claim.',
       'label_definition':'endpoint y minus physical x; original Newton shards remain unmodified',
       'all_other_training_choices_matched_to_newton_reference':True,
       'target_hr_labels_used_for_training_or_selection':False}
    p['sha256']=sha256_json(p);write_json(FREEZE,p);return p


def run():
    p=read(FREEZE)
    if sha256_file(p['config'])!=p['config_sha256'] or sha256_file('blrpng/training/engine.py')!=p['code_sha256']:
        raise ValueError('Frozen endpoint-control code/config changed')
    report='artifacts/manifests/gray_scott_endpoint_control_source.json'
    if resolve_in_project(report).exists():
        r=read(report)
        if sha256_file(r['checkpoint'])!=r['sha256']:raise ValueError('Completed checkpoint changed')
        return
    cfg=load_config(p['config'])
    recovery=resolve_in_project('artifacts/checkpoints/source_endpoint_recovery.pt')
    result=train(cfg,seed=42,objective=cfg['training']['objective'],
                 train_data_path=p['source_train'],validation_data_paths=[p['source_validation']],
                 output_checkpoint='artifacts/checkpoints/gray_scott_endpoint_control/source_endpoint.pt',
                 resume_checkpoint=str(recovery) if recovery.exists() else None)
    checkpoint=compact_full_checkpoint_for_inference(result['checkpoint'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    reference=read(p['reference'])
    if state['initialization_sha256']!=reference['initialization_sha256']:
        raise ValueError('Source initializations differ')
    write_json(report,{'checkpoint':str(checkpoint),'sha256':sha256_file(checkpoint),
                      'initialization_sha256':state['initialization_sha256'],
                      'best_validation_metric':state['best_validation_metric'],
                      'optimizer_steps':state['optimizer_steps'],'elapsed_seconds':state['elapsed_seconds'],
                      'loss_config':state['loss_config'],'reference':p['reference'],
                      'protocol_sha256':p['sha256'],'target_query_evaluated':False})
    print(json.dumps({'completed':True,'source_validation':state['best_validation_metric']}),flush=True)


def adaptation_protocol():
    source=read('artifacts/manifests/gray_scott_endpoint_control_source.json')
    base=copy.deepcopy(read('artifacts/manifests/gray_scott_r3b_adaptation_protocol.json'))
    base['namespace']='gray_scott_endpoint_control'
    base['sources']={'endpoint':{'path':source['checkpoint'],'sha256':source['sha256']}}
    base['training']['horizons']=[3]
    base['source_protocol_sha256']=source['protocol_sha256']
    base.pop('protocol_sha256')
    base['protocol_sha256']=sha256_json(base)
    return base


def adapt():
    p=adaptation_protocol()
    if resolve_in_project(ADAPT).exists() and read(ADAPT)!=p:
        raise RuntimeError('Endpoint adaptation protocol changed')
    write_json(ADAPT,p)
    trials={}
    for task in (0,4,8,12,16):
        path=f'artifacts/reports/gray_scott_endpoint_control/task{task:02d}_endpoint_h3.json'
        r=read(path) if resolve_in_project(path).exists() else train_adapter(p,task,'endpoint',3)
        if r['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Endpoint trial mismatch')
        trials[str(task)]=r
    result={'frozen':True,'protocol_sha256':p['protocol_sha256'],'trials':trials,
            'selection_used_query':False,'horizon':3,
            'selection':'minimum per-task LR50 balanced validation during fixed 80-epoch run'}
    if resolve_in_project(SELECT).exists() and read(SELECT)!=result:
        raise RuntimeError('Endpoint selection changed')
    write_json(SELECT,result)


def evaluate_endpoint(resolution):
    selection=read(SELECT);p=read(ADAPT)
    if selection['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Endpoint selection mismatch')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    records=[];snapshots={}
    for task,trial in selection['trials'].items():
        if sha256_file(trial['checkpoint'])!=trial['checkpoint_sha256']:raise ValueError('Endpoint adapter changed')
        index=f'artifacts/data/gray_scott_r4/task{int(task):02d}_query_{resolution}.index.json'
        payloads=load_shards(index)
        model,_=load_checkpoint_model(trial['checkpoint'],device=device)
        result=evaluate(model,payloads,device,save_fields=True,teacher=True)
        snapshots[task]={'truth':payloads[0]['trajectory'][[0,1,5,10,20,35,50]].clone(),
                         'endpoint':result.pop('representatives')[0]}
        records.extend([{**r,'method':'endpoint','resolution':resolution,'seed':42,
                         'checkpoint_sha256':trial['checkpoint_sha256'],'index_sha256':sha256_file(index)}
                        for r in result['records']])
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
        print(json.dumps({'task':task,'resolution':resolution,'balanced':result['balanced']}),flush=True)
    write_json(f'artifacts/reports/gray_scott_endpoint_control/fresh_query_{resolution}.json',{
        'records':records,'selection_sha256':sha256_file(SELECT),'selection_used_query':False,
        'interpretation':'Post-hoc mechanism comparison on already-opened R4 queries; not an independent blind test.'})
    atomic_torch_save(snapshots,resolve_in_project(
        f'artifacts/reports/gray_scott_endpoint_control/fields_{resolution}.pt'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','train','adapt','evaluate'))
    parser.add_argument('--resolution',type=int,choices=(128,256));args=parser.parse_args()
    torch.set_num_threads(2)
    with workspace_audit(f'gray_scott_endpoint_control_{args.stage}'):
        if args.stage=='freeze':freeze()
        elif args.stage=='train':run()
        elif args.stage=='adapt':adapt()
        else:
            if args.resolution is None:parser.error('evaluate requires --resolution')
            evaluate_endpoint(args.resolution)


if __name__=='__main__':main()
