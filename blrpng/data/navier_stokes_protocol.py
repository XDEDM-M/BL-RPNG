from __future__ import annotations

import copy
from typing import Any

import numpy as np

from blrpng.paths import sha256_json, write_json


PROTOCOL_SCHEMA = "blrpng_navier_stokes_sfepy_protocol_v1"


def _boundary_spec(
    rng: np.random.Generator,
    *,
    family: str,
    amplitude: float | None = None,
) -> dict[str, Any]:
    if family == "source":
        frequencies = sorted(int(v) for v in rng.choice(np.arange(1, 5), size=2, replace=False))
        coefficient_scale = 0.16
        bump_amplitude = float(rng.uniform(0.0, 0.18))
        bump_sigma = float(rng.uniform(0.12, 0.24))
    elif family == "high_frequency":
        frequencies = [5, 6]
        coefficient_scale = 0.18
        bump_amplitude = 0.0
        bump_sigma = 0.16
    elif family == "localized_bump":
        frequencies = [2, 4]
        coefficient_scale = 0.08
        bump_amplitude = float(rng.uniform(0.35, 0.55))
        bump_sigma = float(rng.uniform(0.045, 0.075))
    elif family == "amplitude_ood":
        frequencies = [3, 5]
        coefficient_scale = 0.14
        bump_amplitude = float(rng.uniform(0.05, 0.20))
        bump_sigma = float(rng.uniform(0.10, 0.20))
    elif family == "combined_reynolds_ood":
        frequencies = [5, 6]
        coefficient_scale = 0.18
        bump_amplitude = float(rng.uniform(0.20, 0.40))
        bump_sigma = float(rng.uniform(0.06, 0.12))
    else:
        raise ValueError(f"Unknown Navier--Stokes boundary family: {family}")
    return {
        "family": family,
        "amplitude": float(amplitude if amplitude is not None else rng.uniform(0.75, 1.25)),
        "fourier_frequencies": frequencies,
        "fourier_coefficients": [float(v) for v in rng.uniform(-coefficient_scale, coefficient_scale, 2)],
        "fourier_phases": [float(v) for v in rng.uniform(0.0, 2.0 * np.pi, 2)],
        "bump_amplitude": bump_amplitude,
        "bump_center": float(rng.uniform(0.25, 0.75)),
        "bump_sigma": bump_sigma,
    }


def _forcing_spec(rng: np.random.Generator, *, active: bool) -> dict[str, Any]:
    return {
        "active": bool(active),
        "amplitude": float(rng.uniform(0.015, 0.05)) if active else 0.0,
        "kx": int(rng.integers(1, 4)),
        "ky": int(rng.integers(1, 4)),
        "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
    }


def _rollout(
    *,
    case_id: int,
    boundary_id: int,
    initial_condition_id: int,
    target_task_id: int,
    role: str,
    boundary: dict[str, Any],
    reynolds: float,
    initial_seed: int,
    forcing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": int(case_id),
        "pair_id": int(case_id),
        "boundary_id": int(boundary_id),
        "initial_condition_id": int(initial_condition_id),
        "target_task_id": int(target_task_id),
        "rollout_id": int(case_id),
        "role": role,
        "boundary": copy.deepcopy(boundary),
        "reynolds": float(reynolds),
        "initial_seed": int(initial_seed),
        "forcing": copy.deepcopy(forcing),
    }


def _source_cases(
    rng: np.random.Generator,
    *,
    count: int,
    role: str,
    offset: int,
    reynolds_values: list[int],
    forcing_fraction: float,
) -> list[dict[str, Any]]:
    cases = []
    for index in range(count):
        boundary = _boundary_spec(rng, family="source")
        cases.append(
            _rollout(
                case_id=offset + index,
                boundary_id=offset + index,
                initial_condition_id=offset + index,
                target_task_id=-1,
                role=role,
                boundary=boundary,
                reynolds=float(rng.choice(reynolds_values)),
                initial_seed=int(rng.integers(0, 2**31 - 1)),
                forcing=_forcing_spec(rng, active=bool(rng.random() < forcing_fraction)),
            )
        )
    return cases


def build_navier_stokes_protocol(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    seed = int(config["experiment"]["seed"])
    rng = np.random.default_rng(seed)
    source_re = [int(value) for value in data["reynolds_source"]]
    forcing_fraction = float(data["source_forcing_fraction"])

    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "seed": seed,
        "config_sha256": config["_config_sha256"],
        "query_locked": True,
        "design": {
            "support_budgets": list(data["support_budgets"]),
            "source_steps": int(data["source_steps"]),
            "support_steps": int(data["support_steps"]),
            "query_steps": int(data["query_steps"]),
            "output_resolutions": copy.deepcopy(data["output_resolutions"]),
            "mesh_resolutions": copy.deepcopy(data["mesh_resolutions"]),
            "condition_names": list(data["condition_names"]),
        },
    }
    protocol["development_train"] = _source_cases(
        rng,
        count=int(data["development_train_rollouts"]),
        role="development_train",
        offset=1_000_000,
        reynolds_values=source_re,
        forcing_fraction=forcing_fraction,
    )
    protocol["development_validation"] = _source_cases(
        rng,
        count=int(data["development_validation_rollouts"]),
        role="development_validation",
        offset=1_100_000,
        reynolds_values=source_re,
        forcing_fraction=forcing_fraction,
    )
    protocol["source_train"] = _source_cases(
        rng,
        count=int(data["source_train_rollouts"]),
        role="source_train",
        offset=2_000_000,
        reynolds_values=source_re,
        forcing_fraction=forcing_fraction,
    )
    protocol["source_validation"] = _source_cases(
        rng,
        count=int(data["source_validation_rollouts"]),
        role="source_validation",
        offset=2_500_000,
        reynolds_values=source_re,
        forcing_fraction=forcing_fraction,
    )

    targets: dict[str, Any] = {}
    in_distribution_re = [int(value) for value in data["reynolds_target_in_distribution"]]
    ood_re = [int(value) for value in data["reynolds_target_ood"]]
    maximum_support = max(int(value) for value in data["support_budgets"])
    for task_id in range(int(data["target_tasks"])):
        if task_id < 6:
            family, amplitude = "high_frequency", None
        elif task_id < 12:
            family, amplitude = "localized_bump", None
        elif task_id < 18:
            family, amplitude = "amplitude_ood", 1.5
        else:
            family, amplitude = "combined_reynolds_ood", None
        boundary = _boundary_spec(rng, family=family, amplitude=amplitude)
        reynolds = float(rng.choice(ood_re if task_id >= 18 else in_distribution_re))
        boundary_id = 10_000 + task_id
        task: dict[str, Any] = {
            "target_task_id": task_id,
            "boundary_id": boundary_id,
            "family": family,
            "boundary": boundary,
            "reynolds": reynolds,
            "primary_boundary_endpoint": task_id < 18,
        }
        for role, count, offset in (
            ("support", maximum_support, 3_000_000),
            ("validation", int(data["validation_initial_conditions"]), 4_000_000),
            ("query", int(data["query_initial_conditions"]), 5_000_000),
        ):
            task[role] = []
            for local_index in range(count):
                case_id = offset + task_id * 1000 + local_index
                task[role].append(
                    _rollout(
                        case_id=case_id,
                        boundary_id=boundary_id,
                        initial_condition_id=case_id,
                        target_task_id=task_id,
                        role=role,
                        boundary=boundary,
                        reynolds=reynolds,
                        initial_seed=int(rng.integers(0, 2**31 - 1)),
                        forcing=_forcing_spec(rng, active=False),
                    )
                )
        task["oracle_extra"] = []
        if task_id in {int(value) for value in data["oracle_tasks"]}:
            maximum_oracle = max(int(value) for value in data["oracle_budgets"])
            for local_index in range(maximum_support, maximum_oracle):
                case_id = 6_000_000 + task_id * 1000 + local_index
                task["oracle_extra"].append(
                    _rollout(
                        case_id=case_id,
                        boundary_id=boundary_id,
                        initial_condition_id=case_id,
                        target_task_id=task_id,
                        role="oracle_support",
                        boundary=boundary,
                        reynolds=reynolds,
                        initial_seed=int(rng.integers(0, 2**31 - 1)),
                        forcing=_forcing_spec(rng, active=False),
                    )
                )
        targets[str(task_id)] = task
    protocol["targets"] = targets
    protocol["frozen_task_ids"] = [int(value) for value in data["frozen_tasks"]]
    protocol["oracle_task_ids"] = [int(value) for value in data["oracle_tasks"]]
    protocol["sha256"] = sha256_json(protocol)
    validate_navier_stokes_protocol(protocol)
    return protocol


def validate_navier_stokes_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    expected = protocol.get("sha256")
    unsigned = {key: value for key, value in protocol.items() if key != "sha256"}
    if expected != sha256_json(unsigned):
        raise ValueError("Navier--Stokes protocol SHA256 mismatch.")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Unsupported Navier--Stokes protocol: {protocol.get('schema')}")
    source_boundary_ids = {
        int(case["boundary_id"])
        for split in ("development_train", "development_validation", "source_train", "source_validation")
        for case in protocol[split]
    }
    target_boundary_ids = {int(task["boundary_id"]) for task in protocol["targets"].values()}
    if source_boundary_ids & target_boundary_ids:
        raise ValueError("Source/target boundary leakage in Navier--Stokes protocol.")
    case_ids: list[int] = []
    initial_ids: list[int] = []
    for split in ("development_train", "development_validation", "source_train", "source_validation"):
        for case in protocol[split]:
            case_ids.append(int(case["case_id"]))
            initial_ids.append(int(case["initial_condition_id"]))
    for task in protocol["targets"].values():
        for role in ("support", "validation", "query", "oracle_extra"):
            for case in task[role]:
                if int(case["boundary_id"]) != int(task["boundary_id"]):
                    raise ValueError("A target rollout does not use its task boundary.")
                case_ids.append(int(case["case_id"]))
                initial_ids.append(int(case["initial_condition_id"]))
    if len(case_ids) != len(set(case_ids)) or len(initial_ids) != len(set(initial_ids)):
        raise ValueError("Navier--Stokes case or initial-condition ID leakage.")
    if sorted(int(key) for key in protocol["targets"]) != list(range(24)):
        raise ValueError("Navier--Stokes protocol must contain target tasks 0..23.")
    return {
        "passed": True,
        "source_boundaries": len(source_boundary_ids),
        "target_boundaries": len(target_boundary_ids),
        "target_tasks": len(protocol["targets"]),
        "unique_case_ids": len(case_ids),
        "query_locked": bool(protocol["query_locked"]),
    }


def write_navier_stokes_protocol(config: dict[str, Any], path: str) -> Any:
    protocol = build_navier_stokes_protocol(config)
    return write_json(path, protocol)
