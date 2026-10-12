from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from blrpng.paths import resolve_in_project


REQUIRED_KEYS = {
    "x",
    "y",
    "trajectory",
    "boundary_values",
    "boundary_mask",
    "boundary_lift",
    "newton_correction",
    "residual_progress",
    "valid_correction_mask",
    "case_id",
    "pair_id",
    "boundary_id",
    "initial_condition_id",
    "rollout_id",
    "local_step_id",
    "fidelity",
    "solver_converged",
    "num_newton_iters",
    "meta",
}


@dataclass
class DatasetBundle:
    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any]
    validation: dict[str, Any]

    @property
    def x(self) -> torch.Tensor:
        return self.tensors["x"]

    @property
    def y(self) -> torch.Tensor:
        return self.tensors["y"]

    def __len__(self) -> int:
        return int(self.x.shape[0])


def validate_dataset(data: dict[str, Any], *, modes2: int | None = None) -> dict[str, Any]:
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise KeyError(f"Missing BL-RPNG schema keys: {sorted(missing)}")
    if data["meta"].get("schema_version") != "blrpng_dataset_v3":
        raise ValueError(f"Unsupported dataset schema: {data['meta'].get('schema_version')}")
    if (
        data["meta"].get("equation") == "allen_cahn"
        and data["meta"].get("correction_semantics")
        != "cumulative_from_initial_newton_state"
    ):
        raise ValueError(
            "Allen--Cahn data must use cumulative Newton-path corrections "
            "conditioned on the initial Newton state."
        )
    sample_count = int(data["x"].shape[0])
    if data["x"].ndim != 4 or data["y"].ndim != 4:
        raise ValueError("x and y must have shape [N,C,H,W].")
    for key in REQUIRED_KEYS - {"meta", "fidelity"}:
        value = data[key]
        if isinstance(value, torch.Tensor) and value.shape and int(value.shape[0]) != sample_count:
            raise ValueError(f"Sample-count mismatch for {key}: {value.shape[0]} != {sample_count}")
    height, width = data["x"].shape[-2:]
    if modes2 is not None and modes2 > width // 2 + 1:
        raise ValueError(
            f"modes2={modes2} exceeds rFFT capacity {width // 2 + 1} for width={width}"
        )
    if modes2 is not None and modes2 > height // 2:
        raise ValueError(
            f"modes1={modes2} exceeds two-sided FFT capacity {height // 2} for height={height}"
        )
    boundary_mask = data["boundary_mask"].bool()
    field_channels = int(data["y"].shape[1])
    if boundary_mask.shape[1] == 1 and field_channels > 1:
        boundary_mask = boundary_mask.expand(-1, field_channels, -1, -1)
    expected_mask = torch.zeros_like(boundary_mask)
    expected_mask[:, :, (0, -1), :] = True
    expected_mask[:, :, :, (0, -1)] = True
    if not torch.equal(boundary_mask, expected_mask):
        raise ValueError("boundary_mask is not the complete rectangular boundary.")
    boundary_error = max(
        float(torch.max(torch.abs(data["x"][:, :field_channels][boundary_mask] - data["boundary_values"][boundary_mask]))),
        float(torch.max(torch.abs(data["y"][boundary_mask] - data["boundary_values"][boundary_mask]))),
    )
    lift_error = float(
        torch.max(torch.abs(data["boundary_lift"][boundary_mask] - data["boundary_values"][boundary_mask]))
    )
    correction_boundary_error = float(
        torch.max(torch.abs(data["newton_correction"][boundary_mask]))
    )
    num_rollouts = int(data["meta"]["num_rollouts"])
    num_steps = int(data["meta"]["num_time_steps"])
    if sample_count != num_rollouts * num_steps:
        raise ValueError("Dataset is not rollout-major with the declared dimensions.")
    expected_rollouts = torch.arange(num_rollouts).repeat_interleave(num_steps)
    expected_steps = torch.arange(num_steps).repeat(num_rollouts)
    ordering_ok = bool(
        torch.equal(data["rollout_id"].cpu(), expected_rollouts)
        and torch.equal(data["local_step_id"].cpu(), expected_steps)
    )
    if not ordering_ok:
        raise ValueError("rollout_id/local_step_id are not in rollout-major order.")
    finite_ok = all(
        bool(torch.isfinite(value).all())
        for value in data.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    if not finite_ok:
        raise ValueError("Dataset contains non-finite floating-point values.")
    transition_norm = (
        data["y"] - data["x"][:, : data["y"].shape[1]]
    ).flatten(1).norm(dim=1)
    nontrivial_transition_fraction = float((transition_norm > 1e-8).float().mean())
    if nontrivial_transition_fraction < 0.5:
        raise ValueError(
            "Degenerate temporal dataset: fewer than half of y-x_t transitions "
            f"are nonzero ({nontrivial_transition_fraction:.3f})."
        )
    return {
        "samples": sample_count,
        "shape": [int(v) for v in data["x"].shape],
        "boundary_error_max": boundary_error,
        "lifting_boundary_error_max": lift_error,
        "correction_boundary_error_max": correction_boundary_error,
        "solver_converged_fraction": float(data["solver_converged"].float().mean()),
        "valid_correction_fraction": float(data["valid_correction_mask"].float().mean()),
        "ordering_ok": ordering_ok,
        "finite_ok": finite_ok,
        "nontrivial_transition_fraction": nontrivial_transition_fraction,
        "transition_norm_mean": float(transition_norm.mean()),
    }


def load_dataset(path: str | Path, *, modes2: int | None = None) -> DatasetBundle:
    dataset_path = resolve_in_project(path, must_exist=True)
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    validation = validate_dataset(data, modes2=modes2)
    tensors = {key: value for key, value in data.items() if isinstance(value, torch.Tensor)}
    return DatasetBundle(tensors=tensors, meta=dict(data["meta"]), validation=validation)


def validate_coarse_fine_pairing(
    coarse: DatasetBundle,
    fine: DatasetBundle,
) -> dict[str, Any]:
    keys = ("pair_id", "boundary_id", "initial_condition_id", "local_step_id")
    if len(coarse) != len(fine):
        raise ValueError(f"Coarse/fine sample counts differ: {len(coarse)} != {len(fine)}")
    for key in keys:
        if not torch.equal(coarse.tensors[key], fine.tensors[key]):
            raise ValueError(f"Coarse/fine pairing mismatch in {key}")
    if coarse.meta["manifest_sha256"] != fine.meta["manifest_sha256"]:
        raise ValueError("Coarse/fine files use different frozen manifests.")
    if not bool(torch.all(coarse.tensors["fidelity"] == 0)):
        raise ValueError("Coarse file has non-coarse fidelity labels.")
    if not bool(torch.all(fine.tensors["fidelity"] == 1)):
        raise ValueError("Fine file has non-fine fidelity labels.")
    coarse_initial = torch.nn.functional.interpolate(
        coarse.tensors["x"][:, : fine.tensors["y"].shape[1]],
        size=fine.tensors["x"].shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    fine_initial = fine.tensors["x"][:, : fine.tensors["y"].shape[1]]
    initial_relative = (
        (coarse_initial - fine_initial).flatten(1).norm(dim=1)
        / fine_initial.flatten(1).norm(dim=1).clamp_min(1e-12)
    )
    if float(initial_relative.max()) > 0.25:
        raise ValueError(
            "Coarse/fine initial fields are inconsistent: "
            f"max interpolated relative error={float(initial_relative.max()):.3g}"
        )
    return {
        "paired_samples": len(coarse),
        "manifest_sha256": coarse.meta["manifest_sha256"],
        "ids_match": True,
        "initial_interpolation_relative_error_max": float(initial_relative.max()),
    }
