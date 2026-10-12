"""Freeze statistics and PNG figures for the prespecified R5 safe-router confirmation."""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.cli.gray_scott_r3 import read
from blrpng.cli.gray_scott_r5_safe_router import PROTOCOL, protocol
from blrpng.cli.visualize_gray_scott_r3 import metric, paired_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file, write_json

REPORT_ROOT = 'artifacts/reports/gray_scott_r5'
FIGURE_ROOT = 'artifacts/figures/gray_scott_r5'
METHODS = ('final', 'path', 'safe_router')
LABELS = {'final': 'Ordinary pretrain + NLT + consistency',
          'path': 'Newton-path pretrain + NLT + consistency',
          'safe_router': 'Prespecified safe router'}
COLORS = {'final': '#E69F00', 'path': '#0072B2', 'safe_router': '#009E73'}


def _validate_pairing(p):
    expected = {(int(task), int(s['pair_id'])) for task, spec in p['tasks'].items()
                for s in spec['samples']}
    hashes = {}
    for resolution in (128, 256):
        observed = set()
        for task in p['tasks']:
            path = f'artifacts/data/gray_scott_r5/task{int(task):02d}_query_{resolution}.index.json'
            index = read_index(path, verify_hashes=True)
            if index['metadata']['protocol_sha256'] != p['sha256']:
                raise ValueError('Query index protocol mismatch')
            for row in index['shards']:
                observed.add((int(task), int(row['pair_id'])))
            hashes[path] = sha256_file(path)
        if observed != expected:
            raise ValueError(f'R5 query pairing mismatch at {resolution}')
    return hashes


def _means(report, method, horizon, name, tasks=None):
    rows = [r for r in report['records'] if r['method'] == method and
            (tasks is None or r['target_task_id'] in tasks)]
    return float(np.mean([metric(r, name, horizon) for r in rows]))


def main():
    p = protocol()
    index_hashes = _validate_pairing(p)
    reports = {r: read(f'{REPORT_ROOT}/fresh_query_{r}.json') for r in (128, 256)}
    fields = torch.load(resolve_in_project(f'{REPORT_ROOT}/fields_256.pt'),
                        map_location='cpu', weights_only=False)
    for resolution, report in reports.items():
        if report['protocol_sha256'] != p['sha256'] or report['selection_used_query']:
            raise ValueError(f'Invalid R5 evaluation provenance at {resolution}')
    summary = {
        'schema': 'gray_scott_r5_safe_router_summary_v1', 'seed': 42,
        'protocol_sha256': p['sha256'], 'protocol_file_sha256': sha256_file(PROTOCOL),
        'query_index_sha256': index_hashes, 'selection_used_query': False,
        'route': {task: spec['chosen'] for task, spec in p['tasks'].items()},
        'scope': 'Five reused development boundaries; two entirely new frozen ICs per boundary.',
        'limitations': ['One training seed', 'Boundaries are not new; only initial conditions are new',
                        'No saturated HR specialist comparison',
                        'Conditional boundary-to-IC bootstrap is descriptive, not a multi-seed paper claim'],
        'means': {}, 'per_task': {}, 'paired': {},
    }
    table = []
    for resolution, report in reports.items():
        for method in METHODS:
            for horizon in (20, 50):
                for name in ('joint', 'u', 'v', 'balanced'):
                    value = _means(report, method, horizon, name)
                    summary['means'][f'{resolution}/{method}/{horizon}/{name}'] = value
                    table.append({'resolution': resolution, 'task': 'all', 'method': method,
                                  'horizon': horizon, 'metric': name, 'value': value})
        for task in map(int, p['tasks']):
            for method in METHODS:
                for horizon in (20, 50):
                    for name in ('joint', 'u', 'v', 'balanced'):
                        value = _means(report, method, horizon, name, (task,))
                        key = f'{resolution}/task{task:02d}/{method}/{horizon}/{name}'
                        summary['per_task'][key] = value
                        table.append({'resolution': resolution, 'task': task, 'method': method,
                                      'horizon': horizon, 'metric': name, 'value': value})
        for reference in ('final', 'path'):
            for horizon in (20, 50):
                for name in ('joint', 'v', 'balanced'):
                    key = f'{resolution}/safe_router_vs_{reference}/{horizon}/{name}'
                    summary['paired'][key] = paired_bootstrap(
                        report['records'], 'safe_router', reference, name, horizon)

    router50 = summary['means']['256/safe_router/50/balanced']
    final50 = summary['means']['256/final/50/balanced']
    path50 = summary['means']['256/path/50/balanced']
    versus_final = summary['paired']['256/safe_router_vs_final/50/balanced']
    versus_path = summary['paired']['256/safe_router_vs_path/50/balanced']
    summary['stage_assessment'] = {
        'safe_router_h50_balanced': router50,
        'always_final_h50_balanced': final50,
        'always_path_h50_balanced': path50,
        'router_improvement_vs_always_final': 1 - router50 / final50,
        'router_improvement_vs_always_path': 1 - router50 / path50,
        'practical_stability_gate_passed': bool(router50 <= min(final50, path50)),
        'practical_stability_gate_definition': 'Point-estimate mean no worse than both fixed choices on the prespecified 256-square h50 balanced endpoint.',
        'router_vs_final_conditional_ci95': versus_final['ci95'],
        'router_vs_path_conditional_ci95': versus_path['ci95'],
        'statistical_superiority_confirmed': bool(versus_final['ci95'][0] > 0 and versus_path['ci95'][0] > 0),
        'mechanism_claim_passed': False,
        'mechanism_reason': 'In this dt=1 Gray-Scott regime, accepted Newton corrections are numerically almost identical to endpoint increments; endpoint control matched path control.',
    }

    sources = [{'path': f'{REPORT_ROOT}/fresh_query_{r}.json',
                'sha256': sha256_file(f'{REPORT_ROOT}/fresh_query_{r}.json')} for r in (128, 256)]
    figures = []

    def save(fig, name, note):
        path = ensure_artifact_path(f'{FIGURE_ROOT}/{name}.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        meta = {'png': str(path), 'png_sha256': sha256_file(path), 'sources': sources,
                'protocol_sha256': p['sha256'], 'selection_performed': False, 'note': note}
        write_json(path.with_suffix('.json'), meta); figures.append(meta)

    with workspace_audit('finalize_gray_scott_r5'):
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                             'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        x = np.arange(3)
        for ax, name in zip(axes.flat, ('joint', 'u', 'v', 'balanced')):
            for offset, horizon in ((-.18, 20), (.18, 50)):
                values = [100 * summary['means'][f'256/{m}/{horizon}/{name}'] for m in METHODS]
                bars = ax.bar(x + offset, values, .34, color=[COLORS[m] for m in METHODS],
                              alpha=1 if horizon == 20 else .55,
                              hatch='' if horizon == 20 else '//', label=f'{horizon} steps')
                ax.bar_label(bars, fmt='%.1f', fontsize=8)
            ax.set_xticks(x, [LABELS[m].replace(' + ', '\n+ ') for m in METHODS], fontsize=7)
            ax.set_ylabel('Time-averaged relL2 (%)'); ax.set_title(name)
            ax.grid(axis='y', alpha=.2); ax.legend(frameon=False)
        fig.suptitle('Gray–Scott R5: independent new-IC confirmation at 256² (seed42)')
        save(fig, '01_performance', 'Same frozen query for all methods; path/final route was frozen without HR labels.')

        tasks = sorted(map(int, p['tasks']))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        for ax, name in zip(axes, ('joint', 'v', 'balanced')):
            values = np.array([[100 * summary['per_task'][f'256/task{t:02d}/{m}/50/{name}']
                                for m in METHODS] for t in tasks])
            im = ax.imshow(values, cmap='YlOrRd', aspect='auto', vmin=0)
            for i in range(len(tasks)):
                for j in range(3): ax.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center', fontsize=8)
            ax.set_xticks(range(3), ['final', 'path', 'router'])
            ax.set_yticks(range(len(tasks)), [f'Task {t}' for t in tasks])
            ax.set_title(f'50-step {name} error (%)'); fig.colorbar(im, ax=ax)
        save(fig, '02_taskwise_heatmap', 'Taskwise means over the two new ICs expose failures hidden by overall means.')

        fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), constrained_layout=True)
        for col, task in enumerate(tasks):
            for method in METHODS:
                rows = [r for r in reports[256]['records'] if r['method'] == method and r['target_task_id'] == task]
                values = np.asarray([r['field_errors'] for r in rows])
                for field in (0, 1):
                    axes[field, col].plot(np.arange(1, 51), 100 * values[:, :, field].mean(0),
                                          color=COLORS[method], label=LABELS[method])
            axes[0, col].set_title(f'Task {task} ({p["tasks"][str(task)]["chosen"]} routed)')
            for field in (0, 1):
                axes[field, col].set_xlabel('Rollout step')
                axes[field, col].set_ylabel(f'{"uv"[field]} relL2 (%)'); axes[field, col].grid(alpha=.2)
        axes[0, 0].legend(frameon=False, fontsize=7)
        save(fig, '03_error_growth', 'Per-field error evolution, averaged over two prespecified fresh ICs per boundary.')

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for task, spec in p['tasks'].items():
            for source, marker in (('final', 'o'), ('path', 's')):
                row = spec['candidates'][source]
                selected = spec['chosen'] == source
                ax.scatter(100 * row['free20_comm128_256'], 100 * row['lr_validation_balanced'],
                           s=100 if selected else 50, marker=marker,
                           color=COLORS[source], edgecolor='black' if selected else 'none')
                ax.annotate(f'T{task}', (100 * row['free20_comm128_256'],
                                        100 * row['lr_validation_balanced']), fontsize=7)
        ax.axvline(100 * p['threshold'], color='#D55E00', linestyle='--', label='frozen stability threshold')
        ax.set_xlabel('Unlabeled free20 128→256 disagreement (%)')
        ax.set_ylabel('Labeled LR50 validation balanced error (%)')
        ax.grid(alpha=.2); ax.legend(frameon=False)
        save(fig, '04_router_decision', 'Black outline denotes the frozen routed candidate; task16 path is rejected for instability.')

        times = (0, 5, 10, 20, 35, 50); lookup = (0, 1, 5, 10, 20, 35, 50)
        for task in tasks:
            row = fields[str(task)]
            truth = row['truth']
            display = ('truth', 'final', 'path', 'safe_router')
            fig, axes = plt.subplots(8, 6, figsize=(13, 16), constrained_layout=True)
            for field in (0, 1):
                all_values = [truth[lookup.index(t), field] for t in times]
                all_values += [row[m][t][field] for m in ('final', 'path') for t in times]
                vmin = min(float(v.min()) for v in all_values); vmax = max(float(v.max()) for v in all_values)
                for ridx, method in enumerate(display):
                    for col, t in enumerate(times):
                        value = truth[lookup.index(t), field] if method == 'truth' else row[method][t][field]
                        ax = axes[field * 4 + ridx, col]
                        im = ax.imshow(value, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                        ax.set_xticks([]); ax.set_yticks([])
                        if col == 0: ax.set_ylabel(f'{"uv"[field]}: {method}', fontsize=8)
                        if ridx == 0: ax.set_title(f't={t}')
                    if ridx == 0: fig.colorbar(im, ax=list(axes[field * 4 + ridx]), shrink=.65, pad=.01)
            fig.suptitle(f'R5 task {task}: first new IC at 256²; routed {p["tasks"][str(task)]["chosen"]}')
            save(fig, f'05_evolution_task{task:02d}', 'Common per-field color scale across truth and both candidates; router is an exact selected candidate.')

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, name in zip(axes, ('joint', 'balanced')):
            for method in METHODS:
                ax.plot((128, 256), [100 * summary['means'][f'{r}/{method}/50/{name}'] for r in (128, 256)],
                        'o-', color=COLORS[method], label=LABELS[method])
            ax.set_xticks((128, 256), ('128²', '256²')); ax.set_ylabel(f'50-step {name} relL2 (%)'); ax.grid(alpha=.2)
        axes[0].legend(frameon=False, fontsize=7)
        save(fig, '06_resolution_transfer', 'All adapters use only LR64 target labels; query ICs are paired across resolutions.')

        csv_path = ensure_artifact_path(f'{FIGURE_ROOT}/metrics.csv')
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        write_json(f'{REPORT_ROOT}/summary.json', summary)
        write_json(f'{FIGURE_ROOT}/manifest.json', {
            'figures': figures, 'csv': str(csv_path), 'csv_sha256': sha256_file(csv_path),
            'sources': sources, 'protocol_sha256': p['sha256'], 'selection_performed': False})
        print(json.dumps(summary['stage_assessment'], allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
