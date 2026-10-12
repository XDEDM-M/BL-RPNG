from __future__ import annotations

import json
from typing import Any

import numpy as np

from blrpng.data.manifest import evaluate_boundary
from blrpng.paths import ensure_artifact_path, sha256_json
from blrpng.paths import resolve_in_project


FAMILIES = (
    "high_frequency",
    "amplitude_ood",
    "matern_gp",
    "localized_bump",
    "mixed_fourier",
    "piecewise_smooth",
)


def square_perimeter_coordinate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Counter-clockwise perimeter coordinate in [0, 4), with shared corners."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    tolerance = 1.0e-10
    result = np.full(x.shape, np.nan, dtype=np.float64)
    bottom = np.abs(y) <= tolerance
    right = (np.abs(x - 1.0) <= tolerance) & ~bottom
    top = (np.abs(y - 1.0) <= tolerance) & ~right
    left = (np.abs(x) <= tolerance) & ~bottom & ~top
    result[bottom] = x[bottom]
    result[right] = 1.0 + y[right]
    result[top] = 3.0 - x[top]
    result[left] = 4.0 - y[left]
    if np.isnan(result).any():
        raise ValueError("Boundary evaluator received non-boundary coordinates.")
    return np.mod(result, 4.0)


def _raw_periodic_value(s: np.ndarray, descriptor: dict[str, Any]) -> np.ndarray:
    family = descriptor["family"]
    phase = float(descriptor["phase"])
    z = 2.0 * np.pi * (s / 4.0)
    if family in {"high_frequency", "amplitude_ood", "mixed_fourier"}:
        values = np.zeros_like(s)
        for frequency, sine, cosine in zip(
            descriptor["frequencies"], descriptor["sine"], descriptor["cosine"]
        ):
            values += float(sine) * np.sin(float(frequency) * z + phase)
            values += float(cosine) * np.cos(float(frequency) * z + phase)
        return values
    if family == "matern_gp":
        values = np.zeros_like(s)
        lengthscale = float(descriptor["lengthscale"])
        for center, weight in zip(descriptor["centers"], descriptor["weights"]):
            distance = np.minimum(np.abs(s - float(center)), 4.0 - np.abs(s - float(center)))
            scaled = np.sqrt(3.0) * distance / lengthscale
            values += float(weight) * (1.0 + scaled) * np.exp(-scaled)
        return values
    if family == "localized_bump":
        center = float(descriptor["center"])
        distance = np.minimum(np.abs(s - center), 4.0 - np.abs(s - center))
        width = float(descriptor["width"])
        return np.exp(-0.5 * (distance / width) ** 2) - float(descriptor["offset"])
    if family == "piecewise_smooth":
        centers = np.asarray(descriptor["centers"], dtype=np.float64)
        weights = np.asarray(descriptor["weights"], dtype=np.float64)
        sharpness = float(descriptor["sharpness"])
        values = np.zeros_like(s)
        for center, weight in zip(centers, weights):
            angle = np.pi * (s - center) / 4.0
            values += weight * np.tanh(sharpness * np.sin(angle))
        return values
    raise ValueError(f"Unknown boundary family: {family}")


def evaluate_blind_boundary(
    x: np.ndarray, y: np.ndarray, descriptor: dict[str, Any]
) -> np.ndarray:
    s = square_perimeter_coordinate(x, y)
    raw = _raw_periodic_value(s, descriptor)
    return float(descriptor["amplitude"]) * raw / float(descriptor["normalizer"])


def evaluate_case_boundary(
    x: np.ndarray, y: np.ndarray, case: dict[str, Any]
) -> np.ndarray:
    """Evaluate a legacy global boundary function or a frozen blind trace.

    Blind descriptors live on the perimeter. Interior coordinates, used only
    to construct a warm-start field, are projected to their closest edge; the
    PDE solver still imposes only the exact perimeter trace.
    """
    if "boundary_descriptor" not in case:
        return evaluate_boundary(
            x, y, case["boundary_coefficients"], case["boundary_frequencies"]
        )
    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    distances = np.stack((y_values, 1.0 - x_values, 1.0 - y_values, x_values))
    edge = np.argmin(distances, axis=0)
    s = np.select(
        (edge == 0, edge == 1, edge == 2, edge == 3),
        (x_values, 1.0 + y_values, 3.0 - x_values, 4.0 - y_values),
    )
    raw = _raw_periodic_value(np.mod(s, 4.0), case["boundary_descriptor"])
    descriptor = case["boundary_descriptor"]
    return float(descriptor["amplitude"]) * raw / float(descriptor["normalizer"])


def sample_boundary_descriptor(
    family: str, *, boundary_id: int, seed: int
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family: {family}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, boundary_id]))
    descriptor: dict[str, Any] = {
        "family": family,
        "boundary_id": int(boundary_id),
        "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        "amplitude": 0.65 if family == "amplitude_ood" else 0.35,
    }
    if family == "high_frequency":
        descriptor.update(frequencies=[5, 6], sine=rng.normal(size=2).tolist(), cosine=rng.normal(size=2).tolist())
    elif family == "amplitude_ood":
        descriptor.update(frequencies=[5, 6], sine=rng.normal(size=2).tolist(), cosine=rng.normal(size=2).tolist())
    elif family == "mixed_fourier":
        descriptor.update(frequencies=[1, 3, 5, 8], sine=rng.normal(size=4).tolist(), cosine=rng.normal(size=4).tolist())
    elif family == "matern_gp":
        descriptor.update(
            centers=np.linspace(0.0, 4.0, 16, endpoint=False).tolist(),
            weights=rng.normal(size=16).tolist(),
            lengthscale=float(rng.uniform(0.18, 0.42)),
        )
    elif family == "localized_bump":
        center = float(rng.uniform(0.0, 4.0))
        width = float(rng.uniform(0.08, 0.22))
        # The offset makes the trace contain both signs without breaking
        # periodic corner compatibility.
        descriptor.update(center=center, width=width, offset=float(width * np.sqrt(2.0 * np.pi) / 4.0))
    elif family == "piecewise_smooth":
        descriptor.update(
            centers=np.sort(rng.uniform(0.0, 4.0, size=3)).tolist(),
            weights=rng.normal(size=3).tolist(),
            sharpness=float(rng.uniform(5.0, 10.0)),
        )
    sample_s = np.linspace(0.0, 4.0, 8193, endpoint=False)
    normalizer = float(np.max(np.abs(_raw_periodic_value(sample_s, {**descriptor, "normalizer": 1.0}))))
    if not np.isfinite(normalizer) or normalizer <= 1.0e-12:
        raise RuntimeError("Degenerate frozen boundary descriptor.")
    # Guard against a continuous maximum falling between the dense samples.
    descriptor["normalizer"] = normalizer * (1.0 + 1.0e-6)
    descriptor["sha256"] = sha256_json(descriptor)
    return descriptor


def freeze_reconfirmation_manifest(config: dict[str, Any]) -> str:
    protocol = config["blind_protocol"]
    tasks_per_family = int(protocol["tasks_per_family"])
    seed = int(config["experiment"]["seed"])
    tasks = []
    boundary_id = int(protocol.get("boundary_id_offset", 3_000_000))
    for family in protocol["families"]:
        for local_task in range(tasks_per_family):
            descriptor = sample_boundary_descriptor(family, boundary_id=boundary_id, seed=seed)
            tasks.append({
                "target_task_id": len(tasks),
                "family": family,
                "family_task_id": local_task,
                "boundary_id": boundary_id,
                "boundary": descriptor,
                "split_seeds": {
                    "support": seed + 10_000 + boundary_id,
                    "validation": seed + 20_000 + boundary_id,
                    "query": seed + 30_000 + boundary_id,
                    "query_512": seed + 40_000 + boundary_id,
                },
            })
            boundary_id += 1
    payload = {
        "schema": "blrpng_allen_cahn_reconfirmation_manifest_v1",
        "experiment": config["experiment"]["name"],
        "created_for_protocol_date": "2026-08-20",
        "seed": seed,
        "families": list(protocol["families"]),
        "tasks_per_family": tasks_per_family,
        "support_budgets": list(protocol["support_budgets"]),
        "support_resolution": int(protocol["support_resolution"]),
        "query_resolutions": list(protocol["query_resolutions"]),
        "rollout_horizons": list(protocol["rollout_horizons"]),
        "query_opened": False,
        "query_open_condition": "method_config_checkpoint_rule_and_oracle_saturation_frozen",
        "tasks": tasks,
    }
    payload["sha256"] = sha256_json(payload)
    output = ensure_artifact_path("artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output)


def write_reconfirmation_split_manifest(
    frozen_path: str, *, role: str, initial_conditions: int
) -> str:
    """Materialize IDs for one role without opening or generating any solution."""
    allowed = {"support", "validation", "query", "query_512"}
    if role not in allowed:
        raise ValueError(f"Unsupported reconfirmation role: {role}")
    source_path = resolve_in_project(frozen_path, must_exist=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = source["sha256"]
    unsigned = {key: value for key, value in source.items() if key != "sha256"}
    if expected != sha256_json(unsigned):
        raise ValueError("Frozen reconfirmation manifest hash mismatch.")
    role_offsets = {"support": 0, "validation": 100_000, "query": 200_000, "query_512": 300_000}
    cases = []
    for task in source["tasks"]:
        task_id = int(task["target_task_id"])
        rng = np.random.default_rng(int(task["split_seeds"][role]))
        for local_initial in range(initial_conditions):
            identifier = 4_000_000 + role_offsets[role] + task_id * 1_000 + local_initial
            cases.append({
                "case_id": identifier,
                "pair_id": identifier,
                "boundary_id": int(task["boundary_id"]),
                "initial_condition_id": identifier,
                "split": role,
                "boundary_coefficients": [0.0] * 7,
                "boundary_frequencies": [1.0, 2.0],
                "boundary_amplitude": float(task["boundary"]["amplitude"]),
                "boundary_descriptor": task["boundary"],
                "initial_coefficients": rng.normal(size=4).tolist(),
                "initial_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                "target_task_id": task_id,
                "boundary_family": task["family"],
            })
    payload = {
        "schema": "blrpng_manifest_v1",
        "metadata": {
            "experiment": source["experiment"],
            "role": role,
            "initial_conditions_per_task": initial_conditions,
            "source_manifest": str(source_path),
            "source_sha256": expected,
            "contains_solutions": False,
        },
        "cases": cases,
    }
    payload["sha256"] = sha256_json(payload)
    output = ensure_artifact_path(f"artifacts/manifests/allen_cahn_reconfirmation_{role}.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output)
