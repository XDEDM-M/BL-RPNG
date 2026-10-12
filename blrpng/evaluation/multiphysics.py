from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from blrpng.data.shards import read_index
from blrpng.paths import resolve_in_project
from blrpng.training.losses import per_sample_relative_l2
from blrpng.training.physics import (
    gray_scott_defect,
    hyperelastic_defect,
    navier_stokes_defect,
    poisson_defect,
)


def fieldwise_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    return [
        float(per_sample_relative_l2(prediction[:, index : index + 1], target[:, index : index + 1]).mean())
        for index in range(prediction.shape[1])
    ]


def pattern_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    def wavelength(value: torch.Tensor) -> float:
        spectrum = torch.fft.rfft2(value, norm="ortho").abs().square().mean(dim=(0, 1))
        height, half = spectrum.shape
        ky = torch.fft.fftfreq(height, device=value.device)[:, None]
        kx = torch.fft.rfftfreq((half - 1) * 2, device=value.device)[None]
        radius = torch.sqrt(kx.square() + ky.square())
        spectrum = spectrum.clone(); spectrum[0, 0] = 0.0
        mean_frequency = (radius * spectrum).sum() / spectrum.sum().clamp_min(1e-12)
        return float(1.0 / mean_frequency.clamp_min(1e-12))

    pred_flat = prediction.flatten(2)
    target_flat = target.flatten(2)
    pred_peak = torch.argmax(pred_flat, dim=2)
    target_peak = torch.argmax(target_flat, dim=2)
    width = prediction.shape[-1]
    pred_xy = torch.stack((pred_peak // width, pred_peak % width), dim=-1).float()
    target_xy = torch.stack((target_peak // width, target_peak % width), dim=-1).float()
    peak_error = (pred_xy - target_xy).norm(dim=-1).mean() / max(width - 1, 1)
    pred_wave, target_wave = wavelength(prediction), wavelength(target)
    return {
        "pattern_wavelength_prediction": pred_wave,
        "pattern_wavelength_target": target_wave,
        "pattern_wavelength_relative_error": abs(pred_wave - target_wave) / max(target_wave, 1e-12),
        "peak_location_error": float(peak_error),
    }


def hyperelastic_metrics(displacement: torch.Tensor, *, mu: float = 1.0, lam: float = 2.0) -> dict[str, float]:
    h = 1.0 / (displacement.shape[-1] - 1)
    ux, uy = displacement[:, 0:1], displacement[:, 1:2]
    exx = (ux[..., 2:, 1:-1] - ux[..., :-2, 1:-1]) / (2 * h)
    eyy = (uy[..., 1:-1, 2:] - uy[..., 1:-1, :-2]) / (2 * h)
    exy = 0.5 * (
        (ux[..., 1:-1, 2:] - ux[..., 1:-1, :-2]) / (2 * h)
        + (uy[..., 2:, 1:-1] - uy[..., :-2, 1:-1]) / (2 * h)
    )
    trace = exx + eyy
    sxx, syy, sxy = 2 * mu * exx + lam * trace, 2 * mu * eyy + lam * trace, 2 * mu * exy
    von_mises = torch.sqrt((sxx - syy).square() + 3.0 * sxy.square())
    energy = 0.5 * (sxx * exx + syy * eyy + 2.0 * sxy * exy)
    # Boundary traction proxies are sufficient for paired surrogate/ground-truth
    # comparisons; the formal FEM evaluator additionally records assembled
    # nodal reactions from the worker.
    reaction_x = sxx[..., -1, :].mean()
    reaction_y = syy[..., :, -1].mean()
    return {
        "von_mises_mean": float(von_mises.mean()),
        "von_mises_max": float(von_mises.max()),
        "strain_energy": float(energy.mean()),
        "right_reaction_x": float(reaction_x),
        "top_reaction_y": float(reaction_y),
    }


def navier_stokes_metrics(fields: torch.Tensor, *, reynolds: float = 100.0) -> dict[str, float]:
    u, v, p = fields[:, 0:1], fields[:, 1:2], fields[:, 2:3]
    h = 1.0 / (fields.shape[-1] - 1)
    du_dx = (u[..., 2:, 1:-1] - u[..., :-2, 1:-1]) / (2 * h)
    du_dy = (u[..., 1:-1, 2:] - u[..., 1:-1, :-2]) / (2 * h)
    dv_dx = (v[..., 2:, 1:-1] - v[..., :-2, 1:-1]) / (2 * h)
    dv_dy = (v[..., 1:-1, 2:] - v[..., 1:-1, :-2]) / (2 * h)
    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    inflow = torch.trapezoid(u[..., 0, :], dx=h, dim=-1).mean()
    outflow = torch.trapezoid(u[..., -1, :], dx=h, dim=-1).mean()
    # Integrated rectangular-wall stress proxies.  They are reported as drag
    # and lift only for the wall whose orientation is fixed by the task.
    top_shear = (u[..., :, -1] - u[..., :, -2]) / h / reynolds
    right_normal = -p[..., -1, :]
    return {
        "divergence_l2": float(divergence.square().mean().sqrt()),
        "divergence_max": float(divergence.abs().max()),
        "vorticity_l2": float(vorticity.square().mean().sqrt()),
        "pressure_gauge_error": float(p.mean().abs()),
        "kinetic_energy": float(0.5 * (u.square() + v.square()).mean()),
        "inflow_mass_flux": float(inflow),
        "outflow_mass_flux": float(outflow),
        "mass_flux_mismatch": float((outflow - inflow).abs()),
        "top_wall_drag": float(torch.trapezoid(top_shear, dx=h, dim=-1).mean()),
        "right_wall_pressure_force": float(torch.trapezoid(right_normal, dx=h, dim=-1).mean()),
    }


@torch.no_grad()
def evaluate_multiphysics_index(
    model: torch.nn.Module,
    index_path: str,
    *,
    device: torch.device,
    equation: str,
    equation_parameters: dict[str, Any],
    horizon: int | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen schema-v4 index without selecting a checkpoint."""
    defect_functions = {
        "poisson": poisson_defect,
        "gray_scott": gray_scott_defect,
        "hyperelastic": hyperelastic_defect,
        "navier_stokes": navier_stokes_defect,
    }
    if equation not in defect_functions:
        raise ValueError(f"Unsupported multiphysics equation: {equation}")
    index = read_index(index_path, verify_hashes=True)
    model.eval()
    records = []
    field_errors: list[list[float]] = []
    residual_errors: list[float] = []
    prediction_defects: list[float] = []
    target_defects: list[float] = []
    boundary_errors: list[float] = []
    near_errors: list[float] = []
    interior_errors: list[float] = []
    final_predictions, final_targets = [], []
    field_names: list[str] | None = None
    for row in index["shards"]:
        shard = torch.load(
            resolve_in_project(row["path"], must_exist=True),
            map_location="cpu", weights_only=False,
        )
        trajectory = shard["trajectory"].to(device)
        steps = trajectory.shape[0] - 1
        if horizon is not None:
            steps = min(steps, horizon)
        resolution = trajectory.shape[-1]
        axis = torch.linspace(0.0, 1.0, resolution, device=device, dtype=trajectory.dtype)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((xx, yy)).unsqueeze(0)
        condition_payload = shard.get(
            "condition_fields", trajectory.new_empty((0, resolution, resolution)).cpu()
        )
        lift = shard["boundary_lift"].unsqueeze(0).to(device)
        mask = shard["constraint_mask"].unsqueeze(0).to(device).bool()
        current = trajectory[0:1]
        step_rel = []
        velocity_step_rel: list[float] = []
        pressure_step_rel: list[float] = []
        vorticity_step_rel: list[float] = []
        for step in range(steps):
            conditions = (
                condition_payload[step]
                if condition_payload.ndim == 4
                else condition_payload
            ).unsqueeze(0).to(device)
            target = trajectory[step + 1 : step + 2]
            prediction, _ = model(torch.cat((current, grid, conditions), dim=1), lift, mask)
            step_rel.append(float(per_sample_relative_l2(prediction, target)[0]))
            if equation == "navier_stokes":
                velocity_step_rel.append(
                    float(per_sample_relative_l2(prediction[:, :2], target[:, :2])[0])
                )
                pressure_step_rel.append(
                    float(per_sample_relative_l2(prediction[:, 2:3], target[:, 2:3])[0])
                )
                spacing = 1.0 / float(resolution - 1)

                def vorticity(value: torch.Tensor) -> torch.Tensor:
                    return (
                        value[:, 1:2, 2:, 1:-1]
                        - value[:, 1:2, :-2, 1:-1]
                        - value[:, 0:1, 1:-1, 2:]
                        + value[:, 0:1, 1:-1, :-2]
                    ) / (2.0 * spacing)

                vorticity_step_rel.append(
                    float(per_sample_relative_l2(vorticity(prediction), vorticity(target))[0])
                )
            per_field = fieldwise_relative_l2(prediction, target)
            field_errors.append(per_field)
            boundary_errors.append(
                float((prediction[mask] - target[mask]).abs().max()) if mask.any() else 0.0
            )
            domain_boundary = mask.any(dim=1, keepdim=True)
            near = F.max_pool2d(domain_boundary.float(), kernel_size=9, stride=1, padding=4).bool()
            near = near.expand_as(prediction)
            interior = ~near
            for region, collection in ((near, near_errors), (interior, interior_errors)):
                numerator = ((prediction - target) * region).flatten(1).norm(dim=1)
                denominator = (target * region).flatten(1).norm(dim=1).clamp_min(1e-12)
                collection.append(float((numerator / denominator).mean()))
            params = dict(equation_parameters)
            if conditions.shape[1]:
                params["condition_fields"] = conditions
            pred_defect = defect_functions[equation](current, prediction, params)
            true_defect = defect_functions[equation](current, target, params)
            prediction_defects.append(float(pred_defect.square().mean().sqrt()))
            target_defects.append(float(true_defect.square().mean().sqrt()))
            residual_errors.append(float(
                (pred_defect - true_defect).flatten(1).norm(dim=1).mean()
                / true_defect.flatten(1).norm(dim=1).mean().clamp_min(1e-12)
            ))
            current = prediction
        final_predictions.append(current)
        final_targets.append(trajectory[steps : steps + 1])
        field_names = list(shard["meta"]["field_names"])
        record = {
            "target_task_id": int(shard["ids"]["target_task_id"]),
            "boundary_id": int(shard["ids"]["boundary_id"]),
            "initial_condition_id": int(shard["ids"]["initial_condition_id"]),
            "rollout_relative_l2": float(sum(step_rel) / max(len(step_rel), 1)),
            "final_relative_l2": step_rel[-1],
            "steps": steps,
        }
        if equation == "navier_stokes":
            record.update(
                {
                    "velocity_rollout_relative_l2": float(
                        sum(velocity_step_rel) / max(len(velocity_step_rel), 1)
                    ),
                    "velocity_final_relative_l2": velocity_step_rel[-1],
                    "pressure_rollout_relative_l2": float(
                        sum(pressure_step_rel) / max(len(pressure_step_rel), 1)
                    ),
                    "vorticity_rollout_relative_l2": float(
                        sum(vorticity_step_rel) / max(len(vorticity_step_rel), 1)
                    ),
                }
            )
        records.append(record)
    prediction = torch.cat(final_predictions)
    target = torch.cat(final_targets)
    field_count = prediction.shape[1]
    result: dict[str, Any] = {
        "equation": equation,
        "field_names": field_names,
        "rollouts": records,
        "primary_rollout_relative_l2": float(sum(row["rollout_relative_l2"] for row in records) / len(records)),
        "final_fieldwise_relative_l2": fieldwise_relative_l2(prediction, target),
        "mean_step_fieldwise_relative_l2": [
            float(sum(row[field] for row in field_errors) / len(field_errors))
            for field in range(field_count)
        ],
        "boundary_error_max": max(boundary_errors),
        "near_boundary_relative_l2": float(sum(near_errors) / len(near_errors)),
        "interior_relative_l2": float(sum(interior_errors) / len(interior_errors)),
        "pde_residual_relative_error": float(sum(residual_errors) / len(residual_errors)),
        "prediction_pde_defect_rms": float(sum(prediction_defects) / len(prediction_defects)),
        "ground_truth_pde_defect_rms": float(sum(target_defects) / len(target_defects)),
    }
    if equation == "gray_scott":
        result["pattern"] = pattern_metrics(prediction, target)
    elif equation == "hyperelastic":
        predicted_metrics = hyperelastic_metrics(prediction)
        target_metrics = hyperelastic_metrics(target)
        result["hyperelastic_prediction"] = predicted_metrics
        result["hyperelastic_target"] = target_metrics
        result["hyperelastic_relative_errors"] = {
            key: abs(predicted_metrics[key] - target_metrics[key]) / max(abs(target_metrics[key]), 1e-12)
            for key in predicted_metrics
        }
    elif equation == "navier_stokes":
        reynolds = float(equation_parameters.get("reynolds", 100.0))
        predicted_metrics = navier_stokes_metrics(prediction, reynolds=reynolds)
        target_metrics = navier_stokes_metrics(target, reynolds=reynolds)
        result["flow_prediction"] = predicted_metrics
        result["flow_target"] = target_metrics
        result["flow_absolute_errors"] = {
            key: abs(predicted_metrics[key] - target_metrics[key]) for key in predicted_metrics
        }
        result["primary_velocity_rollout_relative_l2"] = float(
            sum(row["velocity_rollout_relative_l2"] for row in records) / len(records)
        )
        result["pressure_rollout_relative_l2"] = float(
            sum(row["pressure_rollout_relative_l2"] for row in records) / len(records)
        )
        result["vorticity_rollout_relative_l2"] = float(
            sum(row["vorticity_rollout_relative_l2"] for row in records) / len(records)
        )
    return result
