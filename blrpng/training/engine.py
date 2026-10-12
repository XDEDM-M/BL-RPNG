from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from blrpng.data.lifting import harmonic_lift
from blrpng.data.schema import DatasetBundle, load_dataset
from blrpng.data.shards import RolloutShardDataset, materialize_index
from blrpng.evaluation.rollout import projected_rollout, projected_rollout_index
from blrpng.models.fno import DirichletOperator, build_model
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_file, write_json
from blrpng.training.ema import ExponentialMovingAverage
from blrpng.training.losses import (
    balanced_blrpng_loss,
    correction_direction_magnitude_loss,
    per_sample_relative_l2,
    per_sample_field_relative_l2,
)
from blrpng.training.physics import relative_defect_loss, restriction_commutator_loss
from blrpng.utils import atomic_torch_save, set_seed, state_dict_sha256


_REFINED_LIFT_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _exact_refined_lift(coarse_lift: torch.Tensor, resolution: int) -> torch.Tensor:
    """Recompute a discrete harmonic lift after interpolating only its trace."""
    outputs = []
    for sample in coarse_lift.detach().cpu():
        edge = torch.cat((sample[:, 0, :].flatten(), sample[:, -1, :].flatten(),
                          sample[:, 1:-1, 0].flatten(), sample[:, 1:-1, -1].flatten()))
        digest = hashlib.sha256(edge.contiguous().numpy().tobytes()).hexdigest()
        key = (digest, int(resolution))
        cached = _REFINED_LIFT_CACHE.get(key)
        if cached is None:
            interpolated = nn.functional.interpolate(
                sample.unsqueeze(0), size=(resolution, resolution),
                mode="bilinear", align_corners=True,
            )[0].numpy()
            fields = []
            for channel in interpolated:
                boundary = np.zeros_like(channel)
                boundary[0, :], boundary[-1, :] = channel[0, :], channel[-1, :]
                boundary[:, 0], boundary[:, -1] = channel[:, 0], channel[:, -1]
                fields.append(harmonic_lift(boundary))
            cached = torch.from_numpy(np.stack(fields).astype(np.float32))
            _REFINED_LIFT_CACHE[key] = cached
        outputs.append(cached)
    return torch.stack(outputs).to(device=coarse_lift.device, dtype=coarse_lift.dtype)


class BundleDataset(Dataset):
    def __init__(self, bundle: DatasetBundle):
        self.tensors = bundle.tensors

    def __len__(self) -> int:
        return int(self.tensors["x"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


class RolloutBatchSampler(Sampler[list[int]]):
    """Deterministic cache-friendly batches spanning distinct rollout shards."""

    def __init__(
        self,
        dataset: RolloutShardDataset,
        *,
        batch_size: int,
        seed: int,
        start_epoch: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(start_epoch)
        self.by_shard: dict[int, list[int]] = {}
        for sample_index, (shard_index, _step) in enumerate(dataset.samples):
            self.by_shard.setdefault(int(shard_index), []).append(sample_index)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shards = list(self.by_shard)
        order = torch.randperm(len(shards), generator=generator).tolist()
        shards = [shards[index] for index in order]
        for start in range(0, len(shards), self.batch_size):
            selected = shards[start : start + self.batch_size]
            shuffled_steps = []
            for shard in selected:
                values = self.by_shard[shard]
                permutation = torch.randperm(len(values), generator=generator).tolist()
                shuffled_steps.append([values[index] for index in permutation])
            for local_step in range(max(map(len, shuffled_steps))):
                batch = [values[local_step] for values in shuffled_steps if local_step < len(values)]
                if batch:
                    yield batch


def _gpu_power_watts() -> float | None:
    """Best-effort power sample without adding a package dependency."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    command = ["nvidia-smi"]
    if visible and visible.isdigit():
        command.extend(("--id", visible))
    command.extend(("--query-gpu=power.draw", "--format=csv,noheader,nounits"))
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=2)
        return float(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def _boundary_budget(bundle: DatasetBundle, budget: int | None) -> DatasetBundle:
    if budget is None:
        return bundle
    boundary_ids = sorted(set(int(value) for value in bundle.tensors["boundary_id"].tolist()))
    if budget > len(boundary_ids):
        raise ValueError(f"Boundary budget {budget} exceeds available {len(boundary_ids)}")
    selected_ids = torch.tensor(boundary_ids[:budget], dtype=torch.long)
    selected = torch.isin(bundle.tensors["boundary_id"].long(), selected_ids)
    tensors = {key: value[selected] for key, value in bundle.tensors.items()}
    meta = dict(bundle.meta)
    meta["boundary_budget"] = budget
    meta["selected_boundary_ids"] = boundary_ids[:budget]
    return DatasetBundle(tensors=tensors, meta=meta, validation=dict(bundle.validation))


def _load_initialization(
    model: DirichletOperator,
    checkpoint_path: str | Path | None,
    *,
    representation: str,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"kind": "scratch", "checkpoint": None}
    path = resolve_in_project(checkpoint_path, must_exist=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["representation"] != representation:
        raise ValueError(
            f"Initialization representation mismatch: {checkpoint['representation']} != {representation}"
        )
    expected_config = model.model_config_dict()
    checkpoint_config = dict(checkpoint["model_config"])
    # v1 checkpoints omit identity field scales and an empty gauge list.
    checkpoint_config.setdefault("condition_channels", 0)
    checkpoint_config.pop("field_scales", None) if checkpoint_config.get("field_scales") == [1.0] * model.config.out_channels else None
    checkpoint_config.pop("gauge_channels", None) if checkpoint_config.get("gauge_channels") == [] else None
    if checkpoint_config != expected_config:
        raise ValueError("Initialization model configuration mismatch.")
    state = checkpoint.get("ema_weights", checkpoint["raw_weights"])
    model.load_state_dict(state, strict=True)
    return {
        "kind": "checkpoint",
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
    }


def _load_adapter_initialization(
    model: DirichletOperator,
    adapter_path: str | Path | None,
    *,
    base_checkpoint: str | Path | None,
) -> dict[str, Any] | None:
    if adapter_path is None:
        return None
    if base_checkpoint is None:
        raise ValueError("Adapter initialization requires its base checkpoint.")
    path = resolve_in_project(adapter_path, must_exist=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "blrpng_adapter_delta_v1":
        raise ValueError("Adapter initialization is not a BL-RPNG delta checkpoint.")
    base = resolve_in_project(base_checkpoint, must_exist=True)
    if payload["base_checkpoint_sha256"] != sha256_file(base):
        raise ValueError("Adapter initialization was built against a different base checkpoint.")
    if payload["representation"] != model.representation:
        raise ValueError("Adapter initialization representation mismatch.")
    unexpected = model.load_state_dict(payload["adapter_state"], strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"Unexpected adapter initialization keys: {unexpected}")
    return {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path)}


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _train_batch(
    *,
    model: DirichletOperator,
    batch: dict[str, torch.Tensor],
    objective: str,
    training_config: dict[str, Any],
    epoch_progress: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, probes = model(
        batch["x"],
        batch["boundary_lift"],
        batch["boundary_mask"],
    )
    final_metric = str(training_config.get('final_loss', 'joint_relative_l2'))
    if final_metric == 'field_weighted_relative_l2':
        field_errors = per_sample_field_relative_l2(prediction, batch['y'])
        weights = prediction.new_tensor(training_config['field_loss_weights'])
        if weights.numel() != prediction.shape[1] or bool(torch.any(weights < 0)):
            raise ValueError('field_loss_weights must be non-negative with one entry per field.')
        weights = weights / weights.sum().clamp_min(1e-12)
        final_loss = (field_errors * weights[None]).sum(dim=1).mean()
    elif final_metric == 'field_balanced_relative_l2':
        final_loss = per_sample_field_relative_l2(prediction, batch['y']).mean()
    elif final_metric == 'joint_relative_l2':
        final_loss = per_sample_relative_l2(prediction, batch["y"]).mean()
    else:
        raise ValueError(f'Unknown final loss: {final_metric}')
    metrics = {"final_loss": float(final_loss.detach())}
    if prediction.shape[1] == 3 and (
        float(training_config.get("vorticity_weight", 0.0)) > 0.0
        or float(training_config.get("divergence_weight", 0.0)) > 0.0
    ):
        spacing = 1.0 / float(prediction.shape[-1] - 1)

        def derivatives(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            u, v = value[:, 0:1], value[:, 1:2]
            du_dy = (u[..., 1:-1, 2:] - u[..., 1:-1, :-2]) / (2.0 * spacing)
            dv_dx = (v[..., 2:, 1:-1] - v[..., :-2, 1:-1]) / (2.0 * spacing)
            du_dx = (u[..., 2:, 1:-1] - u[..., :-2, 1:-1]) / (2.0 * spacing)
            dv_dy = (v[..., 1:-1, 2:] - v[..., 1:-1, :-2]) / (2.0 * spacing)
            return dv_dx - du_dy, du_dx + dv_dy

        predicted_vorticity, predicted_divergence = derivatives(prediction)
        target_vorticity, _target_divergence = derivatives(batch["y"])
        vorticity_loss = per_sample_relative_l2(
            predicted_vorticity, target_vorticity
        ).mean()
        divergence_loss = predicted_divergence.square().mean().sqrt()
        final_loss = (
            final_loss
            + float(training_config.get("vorticity_weight", 0.0)) * vorticity_loss
            + float(training_config.get("divergence_weight", 0.0)) * divergence_loss
        )
        metrics.update(
            {
                "vorticity_loss": float(vorticity_loss.detach()),
                "divergence_loss": float(divergence_loss.detach()),
                "final_loss_with_flow_regularizers": float(final_loss.detach()),
            }
        )
    fine_prediction = None
    fine_previous = None
    fine_conditions = None
    if objective in {"blrpng_v2", "blrpng_plus", "commutator_only", "pino"}:
        fields = model.config.out_channels
        coarse_resolution = int(batch["x"].shape[-1])
        configured = training_config.get("calibration_resolutions")
        resolutions = (
            [int(value) for value in configured]
            if configured is not None else
            [int(training_config.get("calibration_resolution", coarse_resolution * 2))]
        )
        resolutions = sorted({value for value in resolutions if value > coarse_resolution})
        if not resolutions:
            raise ValueError("At least one calibration resolution must exceed the labeled grid.")
        constrained_channels = batch["boundary_mask"].bool().flatten(2).any(dim=2)
        conditions = batch["x"][:, fields + 2 :]
        predictions = {coarse_resolution: prediction}
        previous_states = {coarse_resolution: batch["x"][:, :fields]}
        geometry: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {
            coarse_resolution: (
                batch["boundary_lift"], batch["boundary_mask"].bool(),
                batch["x"][:, fields : fields + 2], conditions,
            )
        }
        commutators = []
        scales = torch.tensor(model.field_scales, device=prediction.device)
        previous_resolution = coarse_resolution
        for resolution in resolutions:
            fine_previous = nn.functional.interpolate(
                batch["x"][:, :fields], size=(resolution, resolution),
                mode="bilinear", align_corners=True,
            )
            fine_lift = _exact_refined_lift(batch["boundary_lift"], resolution)
            fine_mask = torch.zeros_like(fine_lift, dtype=torch.bool)
            for channel in range(fields):
                active = torch.nonzero(constrained_channels[:, channel], as_tuple=False).flatten()
                fine_mask[active, channel, 0, :] = True
                fine_mask[active, channel, -1, :] = True
                fine_mask[active, channel, :, 0] = True
                fine_mask[active, channel, :, -1] = True
            axis = torch.linspace(
                0.0, 1.0, resolution, device=fine_lift.device, dtype=fine_lift.dtype
            )
            xx, yy = torch.meshgrid(axis, axis, indexing="ij")
            grid = torch.stack((xx, yy))[None].expand(fine_lift.shape[0], -1, -1, -1)
            fine_conditions = nn.functional.interpolate(
                conditions, size=(resolution, resolution), mode="bilinear", align_corners=True
            ) if conditions.shape[1] else conditions.new_empty(
                conditions.shape[0], 0, resolution, resolution
            )
            fine_prediction, _ = model(
                torch.cat((fine_previous, grid, fine_conditions), dim=1),
                fine_lift, fine_mask,
            )
            predictions[resolution] = fine_prediction
            previous_states[resolution] = fine_previous
            geometry[resolution] = (fine_lift, fine_mask, grid, fine_conditions)
            value = restriction_commutator_loss(
                fine_prediction, predictions[previous_resolution], field_scales=scales
            )
            commutators.append(value)
            metrics[f"commutator_{previous_resolution}_to_{resolution}"] = float(value.detach())
            previous_resolution = resolution

        rollout_steps = int(training_config.get("commutator_rollout_steps", 1))
        if rollout_steps < 1:
            raise ValueError("commutator_rollout_steps must be positive.")
        for rollout_step in range(1, rollout_steps):
            next_predictions = {}
            previous_resolution = coarse_resolution
            for resolution in (coarse_resolution, *resolutions):
                lift, mask, grid, raw_conditions = geometry[resolution]
                state = predictions[resolution]
                advanced, _ = model(
                    torch.cat((state, grid, raw_conditions), dim=1), lift, mask
                )
                next_predictions[resolution] = advanced
                if resolution != coarse_resolution:
                    value = restriction_commutator_loss(
                        advanced, next_predictions[previous_resolution], field_scales=scales
                    )
                    commutators.append(value)
                    metrics[
                        f"commutator_step{rollout_step + 1}_{previous_resolution}_to_{resolution}"
                    ] = float(value.detach())
                previous_resolution = resolution
            previous_states = predictions
            predictions = next_predictions
        fine_prediction = predictions[resolutions[-1]]
        fine_previous = previous_states[resolutions[-1]]
        commutator = torch.stack(commutators).mean()
        final_loss = final_loss + float(training_config.get("commutator_weight", 0.05)) * commutator
        metrics["commutator_loss"] = float(commutator.detach())
    defect_loss = final_loss.detach().new_zeros(())
    if objective in {"blrpng_plus", "pino"}:
        equation_parameters = dict(training_config["equation_parameters"])
        if fine_conditions is not None and fine_conditions.shape[1]:
            equation_parameters["condition_fields"] = fine_conditions
        defect_loss = relative_defect_loss(
            str(training_config["equation"]),
            fine_previous,
            fine_prediction,
            equation_parameters,
        )
        final_loss = final_loss + float(training_config.get("defect_weight", 1e-4)) * defect_loss
        metrics["defect_loss"] = float(defect_loss.detach())
    if objective in {"final_only", "commutator_only", "pino"}:
        return final_loss, {**metrics, "path_loss": 0.0, "gradient_ratio": 0.0}

    layer = int(training_config["path_layer"])
    probe = probes[f"layer{layer}:hook0"]
    if objective == "legacy_path":
        path_loss = nn.functional.mse_loss(probe, batch["trajectory"])
        total = final_loss + 0.01 * path_loss
        return total, {
            **metrics,
            "path_loss": float(path_loss.detach()),
            "path_weight": 0.01,
            "gradient_ratio": float("nan"),
        }
    if objective == "dual_stage":
        if model.representation != "lifted":
            raise ValueError("Dual-stage correction supervision requires lifted representation.")
        loss_025, details_025 = correction_direction_magnitude_loss(
            probes["layer1:hook0"],
            batch["newton_correction"],
            batch["valid_correction_mask"],
            log_magnitude_weight=float(training_config["path_log_magnitude_weight"]),
            epsilon=float(training_config["path_epsilon"]),
        )
        loss_075, details_075 = correction_direction_magnitude_loss(
            probes["layer3:hook0"],
            batch["newton_correction_p075"],
            batch["valid_correction_mask_p075"],
            log_magnitude_weight=float(training_config["path_log_magnitude_weight"]),
            epsilon=float(training_config["path_epsilon"]),
        )
        path_loss = 0.5 * (loss_025 + loss_075)
        balanced = balanced_blrpng_loss(
            final_loss=final_loss,
            path_loss=path_loss,
            path_parameters=model.path_active_parameters(3),
            target_ratio=float(training_config["path_gradient_ratio"]),
        )
        return balanced.total, {
            **metrics,
            "path_loss": float(path_loss.detach()),
            "path_weight": balanced.path_weight,
            "gradient_ratio": balanced.gradient_ratio,
            "gradient_cosine": balanced.gradient_cosine,
            "final_shared_gradient_norm": balanced.final_gradient_norm,
            "path_shared_gradient_norm": balanced.path_gradient_norm,
            "path_shared_parameter_count": float(balanced.shared_parameter_count),
            "path_025_direction": details_025["direction"],
            "path_075_direction": details_075["direction"],
        }
    if objective not in {
        "blrpng", "blrpng_safe", "blrpng_annealed", "blrpng_safe_annealed",
        "blrpng_v2", "blrpng_plus",
    }:
        raise ValueError(f"Unknown training objective: {objective}")
    auxiliary_target = str(training_config.get('auxiliary_target', 'newton_correction'))
    if auxiliary_target == 'newton_correction':
        correction_target = batch['newton_correction']
    elif auxiliary_target == 'endpoint_increment':
        correction_target = batch.get(
            'endpoint_increment',
            batch['y'] - batch['x'][:, :model.config.out_channels],
        )
    else:
        raise ValueError(f'Unknown auxiliary target: {auxiliary_target}')
    path_loss, details = correction_direction_magnitude_loss(
        probe,
        correction_target,
        batch["valid_correction_mask"],
        log_magnitude_weight=float(training_config["path_log_magnitude_weight"]),
        epsilon=float(training_config["path_epsilon"]),
        field_scales=batch.get("field_scales"),
    )
    target_ratio = float(training_config["path_gradient_ratio"])
    if "path_anneal_start" in training_config or "path_anneal_end" in training_config:
        start = float(training_config.get("path_anneal_start", 0.0))
        end = float(training_config.get("path_anneal_end", 1.0))
        if not 0.0 <= start < end <= 1.0:
            raise ValueError("path annealing requires 0 <= start < end <= 1")
        if epoch_progress < start or epoch_progress >= end:
            target_ratio = 0.0
        else:
            target_ratio *= (end - epoch_progress) / (end - start)
    elif objective in {"blrpng_v2", "blrpng_plus"}:
        gamma = float(training_config.get("path_anneal_gamma", 2.0))
        target_ratio *= max(1.0 - epoch_progress, 0.0) ** gamma
    balanced = balanced_blrpng_loss(
        final_loss=final_loss,
        path_loss=path_loss,
        path_parameters=model.path_active_parameters(layer),
        target_ratio=target_ratio,
        path_details=details,
        conflict_gate=objective in {
            "blrpng_safe", "blrpng_safe_annealed", "blrpng_v2", "blrpng_plus"
        },
    )
    return balanced.total, {
        **metrics,
        "path_loss": float(path_loss.detach()),
        "path_weight": balanced.path_weight,
        "gradient_ratio": balanced.gradient_ratio,
        "gradient_cosine": balanced.gradient_cosine,
        "final_shared_gradient_norm": balanced.final_gradient_norm,
        "path_shared_gradient_norm": balanced.path_gradient_norm,
        "path_shared_parameter_count": float(balanced.shared_parameter_count),
        "path_target_ratio": target_ratio,
        "path_gate_active": float(balanced.path_weight > 0.0),
        **{f"path_{key}": value for key, value in details.items()},
    }


def _averages(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    result = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and row[key] == row[key]]
        result[key] = float(sum(values) / len(values)) if values else float("nan")
    return result


def train(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    objective: str | None = None,
    initialization_checkpoint: str | Path | None = None,
    adapter_initialization_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    output_checkpoint: str | Path | None = None,
    train_data_path: str | Path | None = None,
    validation_data_paths: list[str | Path] | None = None,
    boundary_budget: int | None = None,
    representation_override: str | None = None,
) -> dict[str, Any]:
    experiment = config["experiment"]
    data_config = config["data"]
    training_config = config["training"]
    seed = int(experiment.get("seed", 42) if seed is None else seed)
    objective = str(training_config.get("objective", "blrpng") if objective is None else objective)
    representation = str(
        representation_override
        or ("raw_legacy" if objective == "legacy_path" else data_config.get("representation", "lifted"))
    )
    model_config = copy.deepcopy(config["model"])
    output_channels = int(model_config.get("out_channels", 1))
    condition_channels = int(model_config.get("condition_channels", 0))
    if representation == "raw_legacy":
        model_config["in_channels"] = output_channels + 2 + condition_channels
    elif representation in {"boundary_augmented", "lift_conditioned"}:
        model_config["in_channels"] = 3 * output_channels + 2 + condition_channels
    else:
        model_config["in_channels"] = 2 * output_channels + 2 + condition_channels
    set_seed(seed, bool(experiment.get("deterministic", True)))

    actual_train_path = train_data_path or data_config["train_path"]
    actual_validation_paths = validation_data_paths or [data_config["validation_path"]]
    train_is_index = str(actual_train_path).endswith(".index.json")
    train_is_stream = train_is_index and bool(training_config.get("stream_v4", True))
    if train_is_stream:
        train_dataset: Dataset = RolloutShardDataset(
            actual_train_path,
            cache_rollouts=max(int(training_config["batch_size"]), 2),
        )
        if boundary_budget is not None:
            boundary_ids = sorted({
                int(row["boundary_id"]) for row in train_dataset.index["shards"]
            })
            if boundary_budget > len(boundary_ids):
                raise ValueError(
                    f"Boundary budget {boundary_budget} exceeds available {len(boundary_ids)}"
                )
            selected = set(boundary_ids[:boundary_budget])
            train_dataset.samples = [
                sample for sample in train_dataset.samples
                if int(train_dataset.index["shards"][sample[0]]["boundary_id"]) in selected
            ]
        train_manifest_sha256 = train_dataset.index["sha256"]
    else:
        train_bundle = _boundary_budget(
            materialize_index(actual_train_path)
            if train_is_index
            else load_dataset(actual_train_path, modes2=int(model_config["modes2"])),
            boundary_budget,
        )
        train_dataset = BundleDataset(train_bundle)
        train_manifest_sha256 = train_bundle.meta["manifest_sha256"]
    validation_sources = [
        str(path) if str(path).endswith(".index.json") and bool(training_config.get("stream_v4", True))
        else materialize_index(path) if str(path).endswith(".index.json")
        else load_dataset(path, modes2=int(model_config["modes2"]))
        for path in actual_validation_paths
    ]
    model = build_model(model_config, representation=representation)
    initialization = _load_initialization(
        model,
        initialization_checkpoint,
        representation=representation,
    )
    adapter_config = dict(training_config.get("adapter", {"strategy": "full"}))
    adapter_info = model.configure_adaptation(
        str(adapter_config.get("strategy", "full")),
        rank=int(adapter_config.get("rank", 4)),
        alpha=float(adapter_config.get("alpha", 4.0)),
        train_last_blocks=int(adapter_config.get("train_last_blocks", 0)),
    )
    adapter_initialization = _load_adapter_initialization(
        model,
        adapter_initialization_checkpoint,
        base_checkpoint=initialization_checkpoint,
    )
    initialization_sha256 = state_dict_sha256(model.state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer_name = str(training_config.get("optimizer", "adam")).lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = optimizer_class(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config["lr"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler_name = str(training_config.get("lr_scheduler", "multistep"))
    if scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(training_config["lr_drop_epoch"])],
            gamma=float(training_config["lr_drop_gamma"]),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(training_config["epochs"]), 1),
            eta_min=float(training_config.get("minimum_lr", 1e-6)),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_config.get("lr_plateau_factor", 0.5)),
            patience=int(training_config.get("plateau_patience", 3)),
            threshold=float(training_config.get("relative_improvement_threshold", 0.0025)),
            threshold_mode="rel",
            min_lr=float(training_config.get("minimum_lr", 1e-5)),
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")
    ema = ExponentialMovingAverage(model, decay=float(training_config.get("ema_decay", 0.995)))
    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_raw: dict[str, torch.Tensor] | None = None
    best_ema: dict[str, torch.Tensor] | None = None
    optimizer_steps = 0
    if resume_checkpoint is not None:
        resume_path = resolve_in_project(resume_checkpoint, must_exist=True)
        resumed = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["raw_weights"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        ema.load_state_dict(resumed["ema"])
        start_epoch = int(resumed["epoch"]) + 1
        history = list(resumed["history"])
        best_metric = float(resumed["best_validation_metric"])
        best_raw = resumed.get("best_raw_weights")
        best_ema = resumed.get("best_ema_weights")
        optimizer_steps = int(resumed.get("optimizer_steps", 0))
        if best_raw is None or best_ema is None:
            # Compatibility with v1 recovery files: preserve a valid resume
            # point even though the historical best snapshot was not embedded.
            best_raw = copy.deepcopy(resumed["raw_weights"])
            best_ema = copy.deepcopy(resumed["ema"]["shadow"])

    if train_is_stream:
        loader = DataLoader(
            train_dataset,
            batch_sampler=RolloutBatchSampler(
                train_dataset,
                batch_size=int(training_config["batch_size"]),
                seed=seed,
                start_epoch=start_epoch,
            ),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
    checkpoint_path = ensure_artifact_path(
        output_checkpoint
        or training_config.get(
            "checkpoint_path",
            f"artifacts/checkpoints/{experiment['name']}_{objective}_seed{seed}.pt",
        )
    )
    started = time.time()
    energy_joules = 0.0
    energy_available = device.type == "cuda"
    energy_sample_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epochs = int(training_config["epochs"])
    max_optimizer_steps = training_config.get("max_optimizer_steps")
    max_optimizer_steps = int(max_optimizer_steps) if max_optimizer_steps is not None else None
    early_stopping_patience = training_config.get("early_stopping_patience")
    early_stopping_patience = (
        int(early_stopping_patience) if early_stopping_patience is not None else None
    )
    minimum_epochs = int(training_config.get("minimum_epochs", 1))
    epochs_without_improvement = 0
    stopped_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            stopped_early = True
            break
        model.train()
        rows = []
        accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
        if accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, cpu_batch in enumerate(loader):
            if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                break
            cpu_batch["field_scales"] = torch.tensor(
                model.field_scales, dtype=cpu_batch["x"].dtype
            ).repeat(cpu_batch["x"].shape[0], 1)
            batch = _batch_to_device(cpu_batch, device)
            epoch_training_config = training_config
            if objective in {"blrpng_safe", "blrpng_annealed", "blrpng_safe_annealed"}:
                epoch_training_config = dict(training_config)
                progress = epoch / max(epochs - 1, 1)
                continuation = 1.0 if progress <= 0.5 else 2.0 * (1.0 - progress)
                epoch_training_config["path_gradient_ratio"] = (
                    float(training_config["path_gradient_ratio"])
                    * max(continuation, 0.0)
                )
            epoch_training_config = dict(epoch_training_config)
            epoch_training_config["equation"] = data_config["equation"]
            epoch_training_config["equation_parameters"] = data_config
            loss, metrics = _train_batch(
                model=model,
                batch=batch,
                objective=objective,
                training_config=epoch_training_config,
                epoch_progress=epoch / max(epochs - 1, 1),
            )
            (loss / accumulation_steps).backward()
            update_due = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if update_due:
                clip_norm = training_config.get("gradient_clip_norm")
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad and parameter.grad is not None
                        ],
                        float(clip_norm),
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                ema.update(model)
            rows.append({"total_loss": float(loss.detach()), **metrics})
        if scheduler_name != "plateau":
            scheduler.step()
        if device.type == "cuda":
            sample_time = time.time()
            power = _gpu_power_watts()
            if power is None:
                energy_available = False
            elif energy_available:
                energy_joules += power * (sample_time - energy_sample_time)
            energy_sample_time = sample_time

        raw_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema.shadow)
        validation_strata = [
            projected_rollout_index(
                model, source, device=device,
                rollout_batch_size=int(training_config.get('validation_batch_size', 16)),
                field_balanced=training_config.get('validation_metric') == 'field_balanced_relative_l2',
            )
            if isinstance(source, str)
            else projected_rollout(model, source, device=device)
            for source in validation_sources
        ]
        validation = {
            "rollout_rel_l2": float(
                sum(float(item["rollout_rel_l2"]) for item in validation_strata)
                / len(validation_strata)
            ),
            "strata": validation_strata,
        }
        model.load_state_dict(raw_state)
        metric = float(validation["rollout_rel_l2"])
        if scheduler_name == "plateau":
            scheduler.step(metric)
        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": _averages(rows),
            "validation": validation,
        }
        history.append(epoch_record)
        if metric < best_metric:
            best_metric = metric
            best_raw = copy.deepcopy(raw_state)
            best_ema = copy.deepcopy(ema.shadow)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        last_epoch = epoch

        progress_interval = int(training_config.get("progress_interval", 1))
        if progress_interval <= 0:
            raise ValueError("training.progress_interval must be positive")
        if (epoch + 1) % progress_interval == 0 or epoch + 1 == epochs:
            write_json(
                f"artifacts/reports/progress/{checkpoint_path.stem}.json",
                {
                    "schema": "blrpng_training_progress_v1",
                    "checkpoint": str(checkpoint_path),
                    "epoch": epoch,
                    "epochs_requested": epochs,
                    "optimizer_steps": optimizer_steps,
                    "validation_rollout_rel_l2": metric,
                    "best_validation_metric": best_metric,
                    "epochs_without_improvement": epochs_without_improvement,
                    "elapsed_seconds": time.time() - started,
                },
            )

        recovery_interval = int(
            training_config.get("recovery_interval", max(1, epochs // 4))
        )
        if (epoch + 1) % recovery_interval == 0 or epoch + 1 == epochs:
            recovery_payload = {
                "epoch": epoch,
                "raw_weights": raw_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "history": history,
                "best_validation_metric": best_metric,
                "best_raw_weights": best_raw,
                "best_ema_weights": best_ema,
                "optimizer_steps": optimizer_steps,
            }
            recovery_path = ensure_artifact_path(
                f"artifacts/checkpoints/{checkpoint_path.stem}_recovery.pt"
            )
            atomic_torch_save(recovery_payload, recovery_path)

        if (
            early_stopping_patience is not None
            and epoch + 1 >= minimum_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            break

    if best_raw is None or best_ema is None:
        raise RuntimeError("Training ran no epochs; check resume/epoch configuration.")
    if device.type == "cuda" and energy_available:
        final_energy_time = time.time()
        final_power = _gpu_power_watts()
        if final_power is None:
            energy_available = False
        else:
            energy_joules += final_power * (final_energy_time - energy_sample_time)
    elapsed_seconds = time.time() - started
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    checkpoint = {
        "format": "blrpng_checkpoint_v1",
        "model_config": model.model_config_dict(),
        "representation": representation,
        "objective": objective,
        "loss_config": {
            'final_loss': str(training_config.get('final_loss', 'joint_relative_l2')),
            'validation_metric': str(training_config.get('validation_metric', 'joint_relative_l2')),
            'auxiliary_target': str(training_config.get('auxiliary_target', 'newton_correction')),
            "path_layer": int(training_config["path_layer"]),
            "path_gradient_ratio": float(training_config["path_gradient_ratio"]),
            "path_log_magnitude_weight": float(training_config["path_log_magnitude_weight"]),
            "legacy_lambda": 0.01 if objective == "legacy_path" else None,
            "path_anneal_gamma": float(training_config.get("path_anneal_gamma", 0.0)),
            "commutator_weight": float(training_config.get("commutator_weight", 0.0)),
            "defect_weight": float(training_config.get("defect_weight", 0.0)),
            "conflict_gate": objective in {
                "blrpng_safe", "blrpng_safe_annealed", "blrpng_v2", "blrpng_plus"
            },
        },
        "data_sha256": {
            "train": sha256_file(resolve_in_project(actual_train_path)),
            "validation": [
                sha256_file(resolve_in_project(path)) for path in actual_validation_paths
            ],
        },
        "manifest_sha256": train_manifest_sha256,
        "initialization_sha256": initialization_sha256,
        "initialization": initialization,
        "adapter_initialization": adapter_initialization,
        "training_seed": seed,
        "boundary_budget": boundary_budget,
        "config_sha256": config["_config_sha256"],
        "config_path": config["_config_path"],
        "best_validation_metric": best_metric,
        "best_metric_name": (
            'projected_rollout_field_balanced_rel_l2'
            if training_config.get('validation_metric') == 'field_balanced_relative_l2'
            else 'ten_step_projected_rollout_rel_l2'
        ),
        "raw_weights": best_raw,
        "ema_weights": best_ema,
        "epoch": last_epoch,
        "optimizer_steps": optimizer_steps,
        "stopped_early": stopped_early,
        "lr_scheduler": scheduler_name,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "gpu_hours": elapsed_seconds / 3600.0 if device.type == "cuda" else 0.0,
        "energy_joules": energy_joules if energy_available else None,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "adapter": {**adapter_config, **adapter_info},
        "torch_version": torch.__version__,
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    report_path = ensure_artifact_path(
        f"artifacts/reports/{checkpoint_path.stem}_training.json"
    )
    write_json(
        report_path,
        {
            "checkpoint": str(checkpoint_path),
            "best_validation_metric": best_metric,
            "initialization_sha256": initialization_sha256,
            "history": history,
        },
    )
    return {
        "checkpoint": str(checkpoint_path),
        "best_validation_metric": best_metric,
        "history": history,
    }
