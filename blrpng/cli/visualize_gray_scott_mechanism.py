"""Mechanism control: Newton cumulative target versus endpoint increment."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blrpng.cli.gray_scott_r3 import read
from blrpng.data.generate import workspace_audit
from blrpng.paths import ensure_artifact_path, sha256_file, write_json


def aggregate(rows,tasks,horizon):
    values=[]
    for r in rows:
        if r['target_task_id'] in tasks:
            fields=np.asarray(r['field_errors'][:horizon]);values.append(
                [np.mean(r['joint_errors'][:horizon]),fields[:,0].mean(),fields[:,1].mean(),fields.mean()])
    return np.mean(values,axis=0)


def main():
    source={
        'ordinary':read('artifacts/manifests/gray_scott_r3b_source_final.json')['best_validation_metric'],
        'Newton cumulative':read('artifacts/manifests/gray_scott_r3b_source_path.json')['best_validation_metric'],
        'Endpoint increment':read('artifacts/manifests/gray_scott_endpoint_control_source.json')['best_validation_metric']}
    endpoint_selection=read('artifacts/manifests/gray_scott_endpoint_control_selection.json')
    newton_selection=read('artifacts/manifests/gray_scott_r3b_selection.json')
    lr={'Newton cumulative':np.mean([r['path']['best_validation_balanced'] for r in newton_selection['trials'].values()]),
        'Endpoint increment':np.mean([r['best_validation_balanced'] for r in endpoint_selection['trials'].values()])}
    endpoint=read('artifacts/reports/gray_scott_endpoint_control/fresh_query_256.json')['records']
    r4=read('artifacts/reports/gray_scott_r4/fresh_query_256.json')['records']
    comparisons={}
    for scope,tasks in [('common4',(0,4,8,12)),('all5',(0,4,8,12,16))]:
        comparisons[scope]={
            'ordinary':aggregate([r for r in r4 if r['method']=='base_final'],tasks,50),
            'Newton cumulative':aggregate([r for r in r4 if r['method']=='base_path'],tasks,50),
            'Endpoint increment':aggregate(endpoint,tasks,50)}
    diagnostic=read('artifacts/reports/gray_scott_r3/path_information_diagnostic.json')
    report={'source_validation':source,'target_lr_validation':lr,
            'fresh_256_h50':{s:{m:v.tolist() for m,v in d.items()} for s,d in comparisons.items()},
            'metrics':['joint','u','v','balanced'],'path_target_statistics':diagnostic['statistics'],
            'interpretation':'Endpoint and cumulative-Newton controls have the same initialization/data/budget. Target-query comparison is post-hoc on already-opened R4 queries.',
            'sources':{p:sha256_file(p) for p in (
                'artifacts/manifests/gray_scott_r3b_source_final.json','artifacts/manifests/gray_scott_r3b_source_path.json',
                'artifacts/manifests/gray_scott_endpoint_control_source.json','artifacts/manifests/gray_scott_endpoint_control_selection.json',
                'artifacts/reports/gray_scott_endpoint_control/fresh_query_256.json','artifacts/reports/gray_scott_r4/fresh_query_256.json',
                'artifacts/reports/gray_scott_r3/path_information_diagnostic.json')}}
    path='artifacts/reports/gray_scott_endpoint_control/mechanism_summary.json';write_json(path,report)
    with workspace_audit('visualize_gray_scott_mechanism'):
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
        names=list(source);colors=['#777777','#0072B2','#D55E00']
        bars=axes[0].bar(names,[100*source[n] for n in names],color=colors);axes[0].bar_label(bars,fmt='%.2f%%')
        axes[0].set_title('A256 source validation');axes[0].set_ylabel('Equal-field rollout relL2 (%)')
        names2=list(lr);bars=axes[1].bar(names2,[100*lr[n] for n in names2],color=colors[1:]);axes[1].bar_label(bars,fmt='%.2f%%')
        axes[1].set_title('B64 LR50 validation, mean over 5 tasks')
        x=np.arange(4);width=.25
        for i,name in enumerate(names):
            vals=100*comparisons['common4'][name]
            axes[2].bar(x+(i-1)*width,vals,width,label=name,color=colors[i])
        axes[2].set_xticks(x,report['metrics']);axes[2].set_title('Fresh 256², 50-step mean, 4 common tasks')
        axes[2].set_ylabel('relL2 (%)');axes[2].legend(frameon=False,fontsize=7)
        for ax in axes:ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelrotation=15)
        fig.suptitle('Gray–Scott mechanism control: cumulative Newton label is nearly the endpoint increment')
        output=ensure_artifact_path('artifacts/figures/gray_scott_r4/10_newton_vs_endpoint_control.png')
        fig.savefig(output,dpi=240);plt.close(fig)
        write_json(output.with_suffix('.json'),{'source':path,'source_sha256':sha256_file(path),
                   'png_sha256':sha256_file(output),'selection_performed':False})
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
