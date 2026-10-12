"""Frozen fresh-query R4 comparison; PNG-only graphics with source hashes."""
from __future__ import annotations
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blrpng.cli.gray_scott_r3 import read
from blrpng.cli.gray_scott_r4 import SELECT
from blrpng.cli.visualize_gray_scott_r3 import metric, paired_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, ensure_artifact_path, sha256_file, write_json

ROOT='artifacts/figures/gray_scott_r4'
LABELS={'comm_path':'R4 path + NLT + consistency','comm_final':'R4 ordinary + NLT + consistency',
        'base_path':'R3B path + NLT','base_final':'R3B ordinary + NLT',
        'r2_path':'R2 path + NLT','r2_final':'R2 ordinary + NLT'}
COLORS=dict(zip(LABELS,('#0072B2','#E69F00','#56B4E9','#777777','#009E73','#CC79A7')))


def main():
    reports={r:read(f'artifacts/reports/gray_scott_r4/fresh_query_{r}.json') for r in (128,256)}
    snapshots=torch.load(resolve_in_project('artifacts/reports/gray_scott_r4/fields_256.pt'),weights_only=False,map_location='cpu')
    selection=read(SELECT)
    summary={'seed':42,'boundary_count':5,'fresh_ics_per_boundary':2,'selected':selection['chosen'],
             'selection_sha256':sha256_file(SELECT),'means':{},'paired':{},
             'limitations':['Development boundaries; new ICs only','One training seed; conditional bootstrap',
                            'No saturated HR oracle or deployment cost claim','No checkpoint reselection using these results']}
    table=[]
    for resolution,report in reports.items():
        for method in LABELS:
            for scope,tasks in [('common4',(0,4,8,12)),('all5',(0,4,8,12,16))]:
                if scope=='all5' and method.startswith('r2'):continue
                rows=[r for r in report['records'] if r['method']==method and r['target_task_id'] in tasks]
                for h in (20,50):
                    for name in ('joint','u','v','balanced'):
                        value=float(np.mean([metric(r,name,h) for r in rows]))
                        summary['means'][f'{resolution}/{scope}/{method}/{h}/{name}']=value
                        table.append(dict(resolution=resolution,scope=scope,method=method,horizon=h,metric=name,value=value))
        for base in ('base_path','comm_final','base_final','r2_path','r2_final'):
            for h in (20,50):
                for name in ('joint','v','balanced'):
                    summary['paired'][f'{resolution}/comm_path_vs_{base}/{h}/{name}']=paired_bootstrap(
                        report['records'],'comm_path',base,name,h)
    sources=[{'path':f'artifacts/reports/gray_scott_r4/fresh_query_{r}.json',
              'sha256':sha256_file(f'artifacts/reports/gray_scott_r4/fresh_query_{r}.json')} for r in reports]
    figures=[]
    def save(fig,name,note):
        path=ensure_artifact_path(f'{ROOT}/{name}.png');fig.savefig(path,dpi=240);plt.close(fig)
        meta={'png':str(path),'png_sha256':sha256_file(path),'sources':sources,'note':note,'selection_performed':False}
        write_json(path.with_suffix('.json'),meta);figures.append(meta)
    with workspace_audit('visualize_gray_scott_r4'):
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
        methods=list(LABELS);x=np.arange(len(methods))
        for ax,name in zip(axes.flat,('joint','u','v','balanced')):
            for offset,h in [(-.18,20),(.18,50)]:
                vals=[100*summary['means'][f'256/common4/{m}/{h}/{name}'] for m in methods]
                bars=ax.bar(x+offset,vals,.34,color=[COLORS[m] for m in methods],alpha=1 if h==20 else .5,
                            hatch='' if h==20 else '//',label=f'{h} steps')
                ax.bar_label(bars,fmt='%.1f',fontsize=7)
            ax.set_xticks(x,[LABELS[m].replace(' + ','\n+') for m in methods],fontsize=7)
            ax.set_ylabel('Time-averaged relL2 (%)');ax.set_title(name);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
        fig.suptitle('R4 fresh 256² query: four common boundaries × two new ICs; seed42')
        save(fig,'04_fresh_query_performance','Same new initial conditions for all models. h20/h50 are time means, not final-time errors.')
        fig,axes=plt.subplots(2,5,figsize=(18,7),constrained_layout=True)
        for col,task in enumerate((0,4,8,12,16)):
            for method in ('comm_path','comm_final','base_path','r2_path'):
                rows=[r for r in reports[256]['records'] if r['target_task_id']==task and r['method']==method]
                if not rows:continue
                values=np.asarray([r['field_errors'] for r in rows])
                for field in (0,1):
                    axes[field,col].plot(np.arange(1,51),100*values[:,:,field].mean(0),label=LABELS[method],color=COLORS[method])
            axes[0,col].set_title(f'Task {task}')
            for field in (0,1):
                axes[field,col].set_xlabel('Time step');axes[field,col].set_ylabel(f'{"uv"[field]} relL2 (%)');axes[field,col].grid(alpha=.2)
        axes[0,0].legend(fontsize=6,frameon=False)
        save(fig,'05_fresh_error_growth','Per-field average over two frozen new ICs, all five development boundaries.')
        times=[0,5,10,20,35,50];lookup=[0,1,5,10,20,35,50]
        for task,row in snapshots.items():
            truth=row['truth'].numpy();models=['comm_final','base_path','comm_path']
            names=['Truth']+[LABELS[m] for m in models]+['|R4 path - truth|']
            fig,axes=plt.subplots(10,6,figsize=(13,20),constrained_layout=True)
            for field in (0,1):
                vmax=max(float(truth[:,field].max()),max(float(row[m][t][field].max()) for m in models for t in times))
                vmin=min(float(truth[:,field].min()),min(float(row[m][t][field].min()) for m in models for t in times))
                emax=max(float((row['comm_path'][t][field]-row['truth'][lookup.index(t),field]).abs().max()) for t in times)
                for r,name in enumerate(names):
                    for col,t in enumerate(times):
                        ax=axes[field*5+r,col]
                        value=(truth[lookup.index(t),field] if r==0 else
                               row[models[r-1]][t][field].numpy() if r<4 else
                               abs(row['comm_path'][t][field].numpy()-truth[lookup.index(t),field]))
                        im=ax.imshow(value,origin='lower',cmap='magma' if r==4 else 'viridis',
                                     vmin=0 if r==4 else vmin,vmax=emax if r==4 else vmax)
                        ax.set_xticks([]);ax.set_yticks([])
                        if col==0:ax.set_ylabel(f'{"uv"[field]}: {name}',fontsize=8)
                        if r==0:ax.set_title(f't = {t}')
                    if r in (0,4):fig.colorbar(im,ax=list(axes[field*5+r]),shrink=.65,pad=.01)
            fig.suptitle(f'Task {task}: first frozen R4 query IC, 256², same field color scales')
            save(fig,f'06_evolution_task{int(task):02d}','First prespecified IC; common field scales over all methods/times, error separately scaled.')
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for ax,name in zip(axes,('joint','v')):
            for method in ('comm_path','comm_final','base_path','base_final'):
                ax.plot([128,256],[100*summary['means'][f'{r}/all5/{method}/50/{name}'] for r in (128,256)],
                        'o-',label=LABELS[method],color=COLORS[method])
            ax.set_xticks([128,256],['128²','256²']);ax.set_ylabel(f'50-step {name} relL2 (%)');ax.grid(alpha=.2)
        axes[0].legend(fontsize=7,frameon=False)
        save(fig,'07_fresh_resolution_transfer','Paired new ICs across resolutions; all adapters trained on LR64 labels only.')
        csv_path=ensure_artifact_path(f'{ROOT}/fresh_metrics.csv')
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        write_json('artifacts/reports/gray_scott_r4/fresh_summary.json',summary)
        write_json(f'{ROOT}/fresh_manifest.json',{'figures':figures,'csv':str(csv_path),'csv_sha256':sha256_file(csv_path),
                   'sources':sources,'selection_performed':False})
        print(json.dumps({'fresh_figures':len(figures),'summary':'artifacts/reports/gray_scott_r4/fresh_summary.json'}),flush=True)


if __name__=='__main__':main()
