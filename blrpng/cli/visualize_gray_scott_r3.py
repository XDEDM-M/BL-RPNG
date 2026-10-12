"""Read-only frozen-result analysis and PNG export; no checkpoint selection."""
from __future__ import annotations

import csv
import io
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.cli.gray_scott_r3 import protocol, read, SELECTION
from blrpng.data.generate import workspace_audit
from blrpng.paths import ensure_artifact_path, sha256_file, write_json, resolve_in_project

OUT = 'artifacts/figures/gray_scott_r3'
LABELS = {'r3b_path': 'R3B Newton-path + NLT', 'r3b_final': 'R3B ordinary + NLT',
          'r3_path': 'R3 Newton-path + NLT', 'r3_final': 'R3 ordinary + NLT',
          'r2_path': 'R2 Newton-path + NLT', 'r2_final': 'R2 ordinary + NLT',
          'r2_scratch': 'R2 LR scratch'}
COLORS = {'r3b_path': '#0072B2', 'r3b_final': '#E69F00',
          'r3_path': '#56B4E9', 'r3_final': '#777777', 'r2_path': '#009E73',
          'r2_final': '#CC79A7', 'r2_scratch': '#D55E00'}
METRICS = ['joint', 'u', 'v', 'balanced']


def metric(row, name, horizon):
    if name == 'joint': return float(np.mean(row['joint_errors'][:horizon]))
    fields = np.asarray(row['field_errors'][:horizon])
    if name == 'balanced': return float(fields.mean())
    return float(fields[:, 0 if name == 'u' else 1].mean())


def paired_bootstrap(rows, candidate, reference, metric_name, horizon):
    """Paired boundary -> IC bootstrap, conditional on the one observed seed."""
    groups = {}
    for r in rows:
        if r['method'] in (candidate, reference):
            groups.setdefault(r['target_task_id'], {}).setdefault(r['initial_condition_id'], {})[r['method']] = metric(r, metric_name, horizon)
    values = []
    for task, initials in sorted(groups.items()):
        pairs = [list((v[candidate], v[reference])) for _, v in sorted(initials.items())
                 if candidate in v and reference in v]
        if pairs: values.append(pairs)
    x = np.asarray(values)
    rng = np.random.default_rng(20260917)
    b = rng.integers(x.shape[0], size=(50000, x.shape[0]))
    ic = rng.integers(x.shape[1], size=(50000, x.shape[0], x.shape[1]))
    resampled = x[b[:, :, None], ic].mean(axis=(1, 2))
    gains = 1 - resampled[:, 0] / resampled[:, 1].clip(1e-12)
    return {'relative_improvement': float(1 - x[:, :, 0].mean() / x[:, :, 1].mean()),
            'ci95': np.quantile(gains, [.025, .975]).tolist(),
            'paired_rollouts': int(x.shape[0] * x.shape[1]), 'boundaries': x.shape[0],
            'replicates': 50000, 'seed': 42, 'conditional_on_one_seed': True}


def main():
    p, selection = protocol(), read(SELECTION)
    reports = {r: read(f'artifacts/reports/gray_scott_r3/fresh_query_{r}.json') for r in (128, 256)}
    snaps = torch.load(resolve_in_project('artifacts/reports/gray_scott_r3/fields_256.pt'),
                       map_location='cpu', weights_only=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 240})
    sources = [{'path': f'artifacts/reports/gray_scott_r3/fresh_query_{r}.json',
                'sha256': sha256_file(f'artifacts/reports/gray_scott_r3/fresh_query_{r}.json')}
               for r in (128, 256)]
    summary = {'seed': 42, 'fresh_ics_per_task': 2, 'target_tasks': 5,
               'selected_shared_horizon': selection['shared_horizon'],
               'primary_candidate': 'r3b_path',
               'r3b_definition': 'Both A256 and B64 use field-balanced final loss; B uses 3-step NLT. Same-init A path-vs-final matched control.',
               'limitations': ['Development boundaries reused; only ICs fresh',
                               'One seed; no saturated HR oracle',
                               'R2 scratch has old joint loss and raw coefficient inputs; not a matched R3 scratch',
                               'R2/R3 differ in checkpoint metric, EMA/raw selection and schedule as well as target loss'],
               'means': {}, 'paired_comparisons': {}}
    tabular = []
    for resolution, report in reports.items():
        for method in LABELS:
            for horizon in (20, 50):
                for name in METRICS:
                    for scope, tasks in [('common4', [0, 4, 8, 12]), ('all5', [0, 4, 8, 12, 16])]:
                        if scope == 'all5' and method.startswith('r2'): continue
                        rows = [r for r in report['records'] if r['method'] == method and r['target_task_id'] in tasks]
                        value = float(np.mean([metric(r, name, horizon) for r in rows]))
                        key = f'{resolution}/{scope}/{method}/{horizon}/{name}'
                        summary['means'][key] = value
                        tabular.append({'resolution': resolution, 'scope': scope, 'method': method,
                                        'horizon': horizon, 'metric': name, 'value': value})
        for base in ('r3b_final', 'r3_path', 'r3_final', 'r2_path', 'r2_final', 'r2_scratch'):
            for name in ('joint', 'balanced', 'v'):
                for horizon in (20, 50):
                    summary['paired_comparisons'][f'{resolution}/r3b_path_vs_{base}/{horizon}/{name}'] = paired_bootstrap(
                        report['records'], 'r3b_path', base, name, horizon)
    figures = []

    def save(fig, name, note):
        path = ensure_artifact_path(f'{OUT}/{name}.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        meta = {'png': str(path), 'png_sha256': sha256_file(path), 'note': note,
                'sources': sources, 'selection_performed': False, 'seed': 42}
        write_json(f'{OUT}/{name}.json', meta)
        figures.append(meta)

    with workspace_audit('visualize_gray_scott_r3'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        methods = list(LABELS)
        x = np.arange(len(methods))
        for ax, name in zip(axes.flat, METRICS):
            for offset, h in [(-.18, 20), (.18, 50)]:
                vals = [100 * summary['means'][f'256/common4/{m}/{h}/{name}'] for m in methods]
                bars = ax.bar(x + offset, vals, .34, color=[COLORS[m] for m in methods],
                              alpha=1 if h == 20 else .5, hatch='' if h == 20 else '//', label=f'{h} steps')
                ax.bar_label(bars, fmt='%.1f', fontsize=7)
            ax.set_xticks(x, [LABELS[m].replace(' + ', '\n+').replace('Newton-path','path') for m in methods], fontsize=7)
            ax.set_ylabel('Mean relative L2 (%)'); ax.set_title(name if name != 'balanced' else 'Equal-field mean (u + v)/2')
            ax.legend(frameon=False); ax.grid(axis='y', alpha=.2)
        fig.suptitle('Fresh 256² query | 4 matched development boundaries × 2 new ICs | seed 42')
        save(fig, '01_fieldwise_performance', 'Matched 4 original tasks. R2 scratch uses old objective; separate u/v expose hidden errors.')

        fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), sharex=True, constrained_layout=True)
        for col, task in enumerate((0, 4, 8, 12, 16)):
            for method in ('r2_final', 'r2_path', 'r3b_final', 'r3b_path'):
                rows = [r for r in reports[256]['records'] if r['target_task_id'] == task and r['method'] == method]
                if not rows: continue
                errors = np.asarray([r['field_errors'] for r in rows])
                for field in (0, 1):
                    axes[field, col].plot(np.arange(1, 51), 100 * errors[:, :, field].mean(0), color=COLORS[method], label=LABELS[method])
            axes[0, col].set_title(f'Task {task}\n' + p['tasks'][str(task)]['label'].replace(' (F/K + frequency + correlation)',''))
            for field in (0, 1):
                axes[field, col].set_ylabel(f'{"uv"[field]} error (%)'); axes[field, col].grid(alpha=.2)
                axes[field, col].set_xlabel('Rollout step')
        axes[0, 0].legend(fontsize=7, frameon=False)
        save(fig, '02_fieldwise_error_growth', 'Averages of two fresh ICs per task; not confidence bands.')

        times = [0, 5, 10, 20, 35, 50]; lookup = [0, 1, 5, 10, 20, 35, 50]
        for task in p['tasks']:
            row = snaps[task]; truth = row['truth'].numpy()
            models = (['r2_final'] if 'r2_final' in row else []) + ['r3b_final', 'r3b_path']
            labels = ['Truth'] + [LABELS[m] for m in models] + ['|R3B path - truth|']
            fig, axes = plt.subplots(2 * len(labels), len(times), figsize=(13, 2.1 * 2 * len(labels)), constrained_layout=True)
            for field in (0, 1):
                maximum = max(float(truth[:, field].max()), max(float(row[m][t][field].max()) for m in models for t in times))
                minimum = min(float(truth[:, field].min()), min(float(row[m][t][field].min()) for m in models for t in times))
                errmax = max(float((row['r3b_path'][t][field] - row['truth'][lookup.index(t), field]).abs().max()) for t in times)
                for r, label in enumerate(labels):
                    for c, t in enumerate(times):
                        ax = axes[field * len(labels) + r, c]
                        if r == 0: value = truth[lookup.index(t), field]
                        elif r <= len(models): value = row[models[r - 1]][t][field].numpy()
                        else: value = abs(row['r3b_path'][t][field].numpy() - truth[lookup.index(t), field])
                        im = ax.imshow(value, origin='lower', cmap='magma' if r == len(labels)-1 else 'viridis',
                                       vmin=0 if r == len(labels)-1 else minimum, vmax=errmax if r == len(labels)-1 else maximum,
                                       extent=(0, 1, 0, 1))
                        ax.set_xticks([]); ax.set_yticks([])
                        if c == 0: ax.set_ylabel(f'{"uv"[field]}: {label}', fontsize=8)
                        if r == 0: ax.set_title(f't = {t}')
                    if r == len(labels)-1 or r == 0:
                        fig.colorbar(im, ax=list(axes[field * len(labels) + r]), shrink=.65, pad=.01)
            fig.suptitle(f'Task {task}: {p["tasks"][task]["label"]} | first frozen fresh IC | 256²')
            save(fig, f'03_evolution_task{int(task):02d}', 'Fixed first fresh IC; per-field common color scale includes all methods/time; absolute error separately scaled.')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for ax, source in zip(axes, ('final', 'path')):
            values = np.array([[selection['trials'][str(t)][f'{source}_h{h}']['best_validation_balanced']
                                for h in (1, 3)] for t in (0, 4, 8, 12, 16)])
            im = ax.imshow(values * 100, cmap='YlOrRd', aspect='auto', vmin=0)
            for i in range(5):
                for j in range(2): ax.text(j, i, f'{values[i,j]*100:.1f}%', ha='center', va='center')
            ax.set_xticks([0, 1], ['Balanced single-step', 'Balanced 3-step'])
            ax.set_yticks(range(5), [f'Task {t}' for t in (0, 4, 8, 12, 16)])
            ax.set_title(f'{source} A256 pretraining; LR50 validation'); fig.colorbar(im, ax=ax)
        save(fig, '04_training_ablation', 'LR-only checkpoint and common horizon selection. All trials 80 epochs / 640 optimizer steps.')

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, name in zip(axes, ('joint', 'v')):
            for method in ('r2_final', 'r2_path', 'r3b_final', 'r3b_path'):
                values = [summary['means'][f'{res}/common4/{method}/50/{name}'] * 100 for res in (128, 256)]
                ax.plot([128, 256], values, 'o-', color=COLORS[method], label=LABELS[method])
            ax.set_xticks([128, 256], ['128²', '256²']); ax.set_ylabel(f'50-step {name} relL2 (%)'); ax.grid(alpha=.2)
        axes[0].legend(frameon=False, fontsize=8)
        save(fig, '05_resolution_transfer', 'Paired fresh ICs on 128/256; shared LR64 adapters.')

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for method in ('r2_final', 'r2_path', 'r3b_final', 'r3b_path'):
            rows = [r for r in reports[256]['records'] if r['target_task_id'] == 12 and r['method'] == method]
            for row in rows[:1]:
                tf = np.asarray(row['teacher_field_errors'])[:, 1] * 100
                free = np.asarray(row['field_errors'])[:, 1] * 100
                axes[0].plot(np.arange(1,51), free, color=COLORS[method], label=LABELS[method])
                axes[0].plot(np.arange(1,51), tf, '--', color=COLORS[method], alpha=.6)
                axes[1].plot(np.arange(2,51), row['finite_error_secant_gain'][1:], color=COLORS[method])
        axes[0].set_ylabel('v relative error (%)'); axes[0].set_title('Task12: free rollout (solid) / teacher forcing (dashed)')
        axes[0].legend(frameon=False, fontsize=7)
        axes[1].axhline(1, color='black', linestyle=':'); axes[1].set_ylabel('Finite-error secant gain')
        axes[1].set_title('Observed error direction; not a Jacobian spectral radius')
        for ax in axes: ax.set_xlabel('Rollout step'); ax.grid(alpha=.2)
        save(fig, '06_task12_feedback_diagnostic', 'Fixed first fresh IC; finite-amplitude directional gain has no global contraction interpretation.')

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for source in ('final', 'path'):
            for h in (1, 3):
                trial = selection['trials']['12'][f'{source}_h{h}']
                epochs = [r['epoch'] for r in trial['history']]
                for ax, name in zip(axes, ('train_loss', 'validation_balanced')):
                    ax.plot(epochs, [r[name] for r in trial['history']], label=f'{source}, {h}-step')
        axes[0].set_title('Task12 training loss'); axes[1].set_title('Task12 LR50 balanced validation error')
        for ax in axes: ax.set_xlabel('Epoch'); ax.legend(frameon=False); ax.grid(alpha=.2)
        save(fig, '07_task12_optimization', 'Training/validation curves reveal overfit and horizon mismatch; only LR used for selection.')

        fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        row = snaps['12']
        truth = row['truth'][-1, 1].numpy()
        methods = ['r2_final', 'r2_path', 'r3b_final', 'r3b_path']
        for col, method in enumerate(methods):
            field = row[method][50][1].numpy()
            error = field - truth
            power = np.abs(np.fft.fftshift(np.fft.fft2(error - error.mean())))**2
            power /= max(power.sum(), 1e-20)
            n = len(field); c = n // 2
            axes[0, col].imshow(field, origin='lower', vmin=0, vmax=max(float(truth.max()), .4), cmap='viridis')
            axes[0, col].set_title(LABELS[method]); axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
            im = axes[1, col].imshow(np.log10(power[c-12:c+13, c-12:c+13] + 1e-12),
                                       origin='lower', extent=(-12.5,12.5,-12.5,12.5), cmap='magma', vmin=-7, vmax=-.5)
            axes[1, col].set_xlabel('k_y'); axes[1, col].set_ylabel('k_x')
        fig.colorbar(im, ax=list(axes[1]), label='log10 fraction of v-error spectral energy', shrink=.75)
        fig.suptitle('Task12 first fresh IC, t=50 | physical v field and its error spectrum')
        save(fig, '08_task12_stripe_spectrum', 'Error spectrum removes DC, normalized by its own total power; absolute error shown in separate field plots.')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        tasks = (0, 4, 8, 12, 16)
        for ax, name in zip(axes, ('balanced', 'v')):
            values = []
            for task in tasks:
                means = {}
                for method in ('r3b_final', 'r3b_path'):
                    rows = [r for r in reports[256]['records']
                            if r['target_task_id'] == task and r['method'] == method]
                    means[method] = np.mean([metric(r, name, 50) for r in rows])
                values.append(100 * (1 - means['r3b_path'] / means['r3b_final']))
            bars = ax.bar(np.arange(len(tasks)), values,
                          color=['#0072B2' if v >= 0 else '#D55E00' for v in values])
            ax.bar_label(bars, fmt='%.1f%%', padding=3)
            ax.axhline(0, color='black', linewidth=.8)
            ax.set_xticks(np.arange(len(tasks)), [f'Task {t}' for t in tasks])
            ax.set_ylabel('Path improvement over matched ordinary pretraining (%)')
            ax.set_title(f'50-step {name} error; positive favors Newton path')
            ax.margins(y=.2); ax.grid(axis='y', alpha=.2)
        save(fig, '11_matched_path_mechanism', 'Same initialization, source data, balanced loss, training budget and target adaptation; one seed and two fresh ICs per development boundary.')

        histories = {s: read(f'artifacts/reports/source_{s}_training.json')['history']
                     for s in ('final', 'path')}
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for source, rows in histories.items():
            epochs = [r['epoch'] + 1 for r in rows]
            axes[0, 0].plot(epochs, [100*r['validation']['rollout_rel_l2'] for r in rows],
                            'o-', label=source)
        axes[0, 0].set_ylabel('Source balanced validation error (%)')
        axes[0, 0].legend(frameon=False)
        rows = histories['path']; epochs = [r['epoch'] + 1 for r in rows]
        axes[0, 1].plot(epochs, [r['train']['gradient_cosine'] for r in rows], 'o-')
        axes[0, 1].axhline(0, color='black', linestyle=':')
        axes[0, 1].set_ylabel('Mean final/path gradient cosine')
        axes[1, 0].plot(epochs, [r['train']['path_target_ratio'] for r in rows], label='Target ratio')
        axes[1, 0].plot(epochs, [r['train']['gradient_ratio'] for r in rows], label='Actual mean ratio')
        axes[1, 0].plot(epochs, [r['train']['path_weight'] for r in rows], label='Actual mean loss weight')
        axes[1, 0].set_ylabel('Ratio / weight'); axes[1, 0].legend(frameon=False)
        axes[1, 1].plot(epochs, [r['train']['path_gate_active'] for r in rows], 'o-')
        axes[1, 1].set_ylabel('Fraction of batches with active path gate')
        axes[1, 1].set_ylim(0, 1)
        for ax in axes.flat: ax.set_xlabel('Source pretraining epoch'); ax.grid(alpha=.2)
        fig.suptitle('Matched A256 retraining: path gradients are active, then intentionally annealed')
        save(fig, '12_source_training_and_path_gate', 'Source histories; path weight is intentionally zero in final epoch, not a disconnected-gradient failure.')
        figures[-1]['sources'] = [
            {'path': f'artifacts/reports/source_{s}_training.json',
             'sha256': sha256_file(f'artifacts/reports/source_{s}_training.json')}
            for s in ('final', 'path')]
        write_json(f'{OUT}/12_source_training_and_path_gate.json', figures[-1])

        # Diagnostics have distinct source datasets; retain their own provenance.
        for name in ('09_physical_tangent_response', '10_path_target_information',
                     '13_discretization_reference', '14_paired_resolution_transfer'):
            path = resolve_in_project(f'{OUT}/{name}.png')
            if path.exists():
                metadata = read(f'{OUT}/{name}.json')
                if metadata['png_sha256'] != sha256_file(path):
                    raise ValueError(f'Diagnostic figure changed: {name}')
                figures.append({'png': str(path), **metadata})

        csv_path = ensure_artifact_path(f'{OUT}/metrics.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tabular[0])); writer.writeheader(); writer.writerows(tabular)
        write_json('artifacts/reports/gray_scott_r3/summary.json', summary)
        write_json(f'{OUT}/manifest.json', {'figures': figures, 'sources': sources,
                   'csv': str(csv_path), 'csv_sha256': sha256_file(csv_path), 'selection_performed': False})
    print(json.dumps({'figures': len(figures),
                      'summary': 'artifacts/reports/gray_scott_r3/summary.json'}, indent=2))


if __name__ == '__main__':
    main()
