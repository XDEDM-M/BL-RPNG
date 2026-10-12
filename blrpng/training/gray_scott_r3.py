"""Small, resumable R3 experiments; original R2 training remains reproducible."""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch

from blrpng.data.shards import read_index, validate_shard
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file, write_json
from blrpng.training.adapters import save_adapter_delta, trainable_state
from blrpng.training.losses import per_sample_field_relative_l2
from blrpng.utils import atomic_torch_save, set_seed


def field_errors(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-example, per-field relL2 with an explicit RMS floor for near-zero fields."""
    numerator = (prediction - target).square().mean(dim=(-2, -1)).sqrt()
    denominator = target.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-3)
    return numerator / denominator


def balanced_field_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return per_sample_field_relative_l2(prediction, target).mean()


def load_shards(path: str) -> list[dict[str, Any]]:
    index = read_index(path, verify_hashes=True)
    output = []
    for row in index['shards']:
        payload = torch.load(resolve_in_project(row['path'], must_exist=True),
                             map_location='cpu', weights_only=False)
        validate_shard(payload)
        if not payload['meta']['solver_converged']:
            raise ValueError('Unconverged ground truth')
        output.append(payload)
    return output


def geometry(payloads: list[dict], device: torch.device):
    n = payloads[0]['trajectory'].shape[-1]
    axis = torch.linspace(0, 1, n, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing='ij'))[None]
    return (
        torch.stack([p['boundary_lift'] for p in payloads]).to(device),
        torch.stack([p['constraint_mask'] for p in payloads]).to(device),
        torch.stack([p['condition_fields'] for p in payloads]).to(device),
        grid.expand(len(payloads), -1, -1, -1),
    )


def advance(model, state, geo):
    lift, mask, conditions, grid = geo
    return model(torch.cat((state, grid, conditions), dim=1), lift, mask)[0]


@torch.no_grad()
def evaluate(model, payloads, device, *, save_fields=False, teacher=False):
    model.eval()
    records, representatives = [], []
    # Single-rollout inference keeps peak VRAM/RAM predictable after system crashes.
    for p in payloads:
        geo = geometry([p], device)
        truth = p['trajectory'].to(device)
        current = truth[:1]
        per_field, joint, boundary, tf, gains, stripe_energy = [], [], [], [], [], []
        fields = {0: current[0].cpu()}
        for step in range(len(truth) - 1):
            target = truth[step + 1:step + 2]
            output = advance(model, current, geo)
            per_field.append(field_errors(output, target)[0].cpu().tolist())
            joint.append(float((output - target).norm() / target.norm().clamp_min(1e-12)))
            boundary.append(float((output - target).masked_select(geo[1]).abs().max()))
            if teacher:
                forced = advance(model, truth[step:step + 1], geo)
                tf.append(field_errors(forced, target)[0].cpu().tolist())
                gains.append(float((output - forced).norm() /
                                   (current - truth[step:step + 1]).norm().clamp_min(1e-12)))
                error_v = output[0, 1] - target[0, 1]
                spectrum = torch.fft.fft2(error_v - error_v.mean()).abs().square()
                stripe_energy.append(float((spectrum[3, 1] + spectrum[-3, -1]) /
                                           spectrum.sum().clamp_min(1e-20)))
            if save_fields and step + 1 in (1, 5, 10, 20, 35, 50):
                fields[step + 1] = output[0].cpu()
            current = output
        per_field = np.asarray(per_field)
        records.append({
            **p['ids'], 'field_errors': per_field.tolist(), 'joint_errors': joint,
            'balanced': float(per_field.mean()), 'u': float(per_field[:, 0].mean()),
            'v': float(per_field[:, 1].mean()), 'joint': float(np.mean(joint)),
            'boundary_max_abs': max(boundary), 'teacher_field_errors': tf,
            'finite_error_secant_gain': gains,
            'v_error_mode_3_1_pair_energy_fraction': stripe_energy,
        })
        if save_fields:
            representatives.append(fields)
    return {'records': records, 'balanced': float(np.mean([r['balanced'] for r in records])),
            'representatives': representatives}


def train_adapter(protocol: dict, task: int, source: str, horizon: int) -> dict:
    """Fixed-step matched NLT trials; only adapters/Adam state are serialized."""
    settings = protocol['training']
    namespace = protocol.get('namespace', 'gray_scott_r3')
    key = f'task{task:02d}_{source}_h{horizon}'
    report_path = f'artifacts/reports/{namespace}/{key}.json'
    checkpoint_path = f'artifacts/checkpoints/{namespace}/{key}.adapter.pt'
    recovery = ensure_artifact_path(f'artifacts/checkpoints/{namespace}/{key}.recovery.pt')
    base = protocol['sources'][source]
    if sha256_file(base['path']) != base['sha256']:
        raise ValueError('Source checkpoint hash mismatch')
    spec = protocol['tasks'][str(task)]
    support = load_shards(spec['support_index'])
    validation = load_shards(spec['validation_index'])
    if {p['ids']['initial_condition_id'] for p in support} & {
        p['ids']['initial_condition_id'] for p in validation
    }:
        raise ValueError('Support/validation leakage')
    set_seed(settings['seed'])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    commutator_weight = float(settings.get('commutator_weight', 0.0))
    if commutator_weight < 0:
        raise ValueError('Negative commutator weight')
    refined = None
    if commutator_weight:
        from blrpng.training.gray_scott_commutator import refined_geometry, prolong_lifted, field_commutator
        refined = [refined_geometry(p, int(settings.get('commutator_resolution', 128)), device)
                   for p in support]
    model, _ = load_checkpoint_model(base['path'], device=device)
    adapter = {'strategy': 'nlt', 'rank': 4, 'alpha': 4.0, 'train_last_blocks': 0}
    counts = model.configure_adaptation(**adapter)
    # Final-only adapters leave path probes unchanged, so do not allocate their optimizer state.
    for probe in model.fno.probes:
        for param in probe.parameters():
            param.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=settings['lr'], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings['epochs'], eta_min=1e-5)
    history, start, best, best_state, best_epoch = [], 0, float('inf'), None, -1
    elapsed_before = 0.0
    if recovery.exists():
        restored = torch.load(recovery, map_location='cpu', weights_only=False)
        if restored['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(restored['adapter_state'], strict=False)
        optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler'])
        history, start = restored['history'], restored['epoch'] + 1
        best, best_state = restored['best'], restored['best_state']
        best_epoch, elapsed_before = restored['best_epoch'], restored['elapsed_seconds']
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # Identical 32 starting states and batch order for h1/h3 and final/path sources.
    samples = [(i, t) for i in range(len(support)) for t in range(8)]
    for epoch in range(start, settings['epochs']):
        model.train()
        order = np.random.default_rng(settings['seed'] + epoch).permutation(len(samples))
        losses, commutators = [], []
        for first in range(0, len(samples), settings['batch_size']):
            pairs = [samples[int(i)] for i in order[first:first + settings['batch_size']]]
            ps = [support[i] for i, t in pairs]
            geo = geometry(ps, device)
            state = torch.stack([support[i]['trajectory'][t] for i, t in pairs]).to(device)
            if refined is not None:
                fine_geo = refined[pairs[0][0]]
                fine_state = prolong_lifted(state[:1], geo[0][:1], fine_geo)
            optimizer.zero_grad(set_to_none=True)
            loss, commutator = 0.0, 0.0
            for step in range(horizon):
                state = advance(model, state, geo)
                target = torch.stack([support[i]['trajectory'][t + step + 1]
                                      for i, t in pairs]).to(device)
                loss = loss + balanced_field_loss(state, target) / horizon
                if refined is not None:
                    fine_state = advance(model, fine_state, fine_geo)
                    commutator = commutator + field_commutator(fine_state, state[:1], target[:1], geo[1][:1]) / horizon
            if refined is not None:
                loss = loss + commutator_weight * commutator
                commutators.append(float(commutator.detach()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss in {key}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if (epoch + 1) % settings['validation_interval'] == 0 or epoch == 0:
            val = evaluate(model, validation, device)
            row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                   'validation_balanced': val['balanced'],
                   'validation_u': float(np.mean([r['u'] for r in val['records']])),
                   'validation_v': float(np.mean([r['v'] for r in val['records']])),
                   'validation_joint': float(np.mean([r['joint'] for r in val['records']]))}
            if commutators:
                row['train_commutator'] = float(np.mean(commutators))
                row['commutator_weight'] = commutator_weight
            history.append(row)
            if val['balanced'] < best:
                best, best_state, best_epoch = val['balanced'], trainable_state(model), epoch + 1
            elapsed = elapsed_before + time.perf_counter() - started
            atomic_torch_save({
                'protocol_sha256': protocol['protocol_sha256'], 'epoch': epoch,
                'adapter_state': trainable_state(model), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'history': history,
                'best': best, 'best_state': best_state, 'best_epoch': best_epoch,
                'elapsed_seconds': elapsed,
            }, recovery)
            write_json(f'artifacts/reports/progress/{namespace}_{key}.json',
                       {**row, 'best': best, 'elapsed_seconds': elapsed})
            print(json_line({'trial': key, **row}), flush=True)
    model.load_state_dict(best_state, strict=False)
    result = {
        'trial': key, 'task': task, 'source': source, 'horizon': horizon,
        'checkpoint': checkpoint_path, 'protocol_sha256': protocol['protocol_sha256'],
        'best_validation_balanced': best, 'best_epoch': best_epoch, 'history': history,
        'training_seconds': elapsed_before + time.perf_counter() - started,
        'optimizer_steps': settings['epochs'] * len(samples) // settings['batch_size'],
        'trainable_parameters': sum(p.numel() for p in parameters),
        'peak_gpu_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
        'support_sha256': sha256_file(spec['support_index']),
        'validation_sha256': sha256_file(spec['validation_index']),
        'selection_used_hr_query': False,
        'commutator_weight': commutator_weight,
        'commutator_resolution': settings.get('commutator_resolution') if commutator_weight else None,
    }
    save_adapter_delta(checkpoint_path, model, base_checkpoint=base['path'],
                       adapter_config=adapter, metadata=result)
    reloaded, _ = load_checkpoint_model(checkpoint_path, device=device)
    expected = evaluate(model, validation[:1], device)
    actual = evaluate(reloaded, validation[:1], device)
    if expected['records'] != actual['records']:
        raise RuntimeError('Adapter checkpoint reload changes rollout')
    result['reload_verified'] = True
    result['checkpoint_sha256'] = sha256_file(checkpoint_path)
    write_json(report_path, result)
    return result


def json_line(value):
    import json
    return json.dumps(value, allow_nan=False)
