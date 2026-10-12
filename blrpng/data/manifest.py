from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_json, write_json


@dataclass(frozen=True)
class CaseSpec:
    case_id: int
    pair_id: int
    boundary_id: int
    initial_condition_id: int
    split: str
    boundary_coefficients: list[float]
    boundary_frequencies: list[float]
    boundary_amplitude: float
    initial_coefficients: list[float]
    initial_phase: float


def _boundary_values(
    points: np.ndarray,
    coefficients: np.ndarray,
    frequencies: tuple[float, float],
) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    f1, f2 = frequencies
    return (
        coefficients[0]
        + coefficients[1] * (x - 0.5)
        + coefficients[2] * (y - 0.5)
        + coefficients[3] * np.sin(f1 * np.pi * x)
        + coefficients[4] * np.sin(f1 * np.pi * y)
        + coefficients[5] * np.sin(f2 * np.pi * x)
        + coefficients[6] * np.sin(f2 * np.pi * y)
    )


def evaluate_boundary(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: list[float] | np.ndarray,
    frequencies: list[float] | tuple[float, float],
) -> np.ndarray:
    """Evaluate the smooth global extension whose trace defines Dirichlet data."""
    x_values, y_values = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    points = np.column_stack([x_values.reshape(-1), y_values.reshape(-1)])
    values = _boundary_values(
        points,
        np.asarray(coefficients, dtype=np.float64),
        (float(frequencies[0]), float(frequencies[1])),
    )
    return values.reshape(x_values.shape)


def sample_boundary_coefficients(
    rng: np.random.Generator,
    frequencies: tuple[float, float],
    amplitude: float,
) -> np.ndarray:
    coefficients = rng.uniform(-1.0, 1.0, size=7)
    line = np.linspace(0.0, 1.0, 513)
    points = np.concatenate(
        [
            np.column_stack([line, np.zeros_like(line)]),
            np.column_stack([line, np.ones_like(line)]),
            np.column_stack([np.zeros_like(line), line]),
            np.column_stack([np.ones_like(line), line]),
        ]
    )
    maximum = float(np.max(np.abs(_boundary_values(points, coefficients, frequencies))))
    if maximum <= 0.0:
        raise RuntimeError("Sampled a zero boundary function.")
    return coefficients * (float(amplitude) / maximum)


def make_cases(
    *,
    count: int,
    split: str,
    seed: int,
    frequencies: tuple[float, float],
    amplitude_range: tuple[float, float],
    case_offset: int = 0,
    boundary_offset: int = 0,
    initial_offset: int = 0,
    fixed_boundaries: int | None = None,
    initial_conditions_per_boundary: int = 1,
) -> list[CaseSpec]:
    if count <= 0:
        return []
    if fixed_boundaries is not None:
        expected = fixed_boundaries * initial_conditions_per_boundary
        if expected != count:
            raise ValueError(f"count={count} must equal fixed_boundaries*ICs={expected}")
    rng = np.random.default_rng(seed)
    boundary_count = count if fixed_boundaries is None else fixed_boundaries
    boundaries: list[tuple[np.ndarray, float]] = []
    for _ in range(boundary_count):
        amplitude = float(rng.uniform(*amplitude_range))
        boundaries.append(
            (sample_boundary_coefficients(rng, frequencies, amplitude), amplitude)
        )

    cases: list[CaseSpec] = []
    for local_idx in range(count):
        boundary_local = (
            local_idx
            if fixed_boundaries is None
            else local_idx // initial_conditions_per_boundary
        )
        ic_local = local_idx if fixed_boundaries is None else local_idx % initial_conditions_per_boundary
        coefficients, amplitude = boundaries[boundary_local]
        cases.append(
            CaseSpec(
                case_id=case_offset + local_idx,
                pair_id=case_offset + local_idx,
                boundary_id=boundary_offset + boundary_local,
                initial_condition_id=initial_offset + local_idx,
                split=split,
                boundary_coefficients=[float(v) for v in coefficients],
                boundary_frequencies=[float(v) for v in frequencies],
                boundary_amplitude=amplitude,
                initial_coefficients=[float(v) for v in rng.normal(size=4)],
                initial_phase=float(0.2 * ic_local),
            )
        )
    return cases


def write_manifest(
    path: str | Path,
    cases: Iterable[CaseSpec],
    *,
    metadata: dict[str, Any],
) -> Path:
    case_values = [asdict(case) for case in cases]
    payload = {
        "schema": "blrpng_manifest_v1",
        "metadata": metadata,
        "cases": case_values,
    }
    payload["sha256"] = sha256_json(payload)
    return write_json(path, payload)


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = resolve_in_project(path, must_exist=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = payload.pop("sha256")
    actual = sha256_json(payload)
    payload["sha256"] = expected
    if actual != expected:
        raise ValueError(f"Manifest hash mismatch: expected {expected}, got {actual}")
    if payload.get("schema") != "blrpng_manifest_v1":
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema')}")
    return payload


def validate_split_isolation(payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload["cases"]
    case_ids = [int(case["case_id"]) for case in cases]
    pair_ids = [int(case["pair_id"]) for case in cases]
    initial_ids = [int(case["initial_condition_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id in frozen manifest.")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate pair_id in frozen manifest.")
    if len(initial_ids) != len(set(initial_ids)):
        raise ValueError("Initial-condition leakage: IDs are not globally unique.")
    split_boundaries: dict[str, set[int]] = {}
    for case in cases:
        split_boundaries.setdefault(case["split"], set()).add(int(case["boundary_id"]))
    a_boundaries = set().union(
        *(values for key, values in split_boundaries.items() if key.startswith("a_"))
    ) if any(key.startswith("a_") for key in split_boundaries) else set()
    b_boundaries = set().union(
        *(values for key, values in split_boundaries.items() if key.startswith(("b_", "blind_")))
    ) if any(key.startswith(("b_", "blind_")) for key in split_boundaries) else set()
    if a_boundaries & b_boundaries:
        raise ValueError("A/B boundary leakage detected.")
    allowed_shared = {
        frozenset(("b_train", "b_validation_seen")),
        frozenset(("b_train", "blind_seen")),
        frozenset(("b_validation_seen", "blind_seen")),
    }
    split_names = sorted(split_boundaries)
    for i, first in enumerate(split_names):
        for second in split_names[i + 1 :]:
            overlap = split_boundaries[first] & split_boundaries[second]
            if overlap and frozenset((first, second)) not in allowed_shared:
                raise ValueError(
                    f"Unexpected boundary leakage between {first} and {second}: {overlap}"
                )
    return {
        "cases": len(cases),
        "split_boundary_counts": {
            key: len(value) for key, value in split_boundaries.items()
        },
        "unique_case_ids": True,
        "unique_pair_ids": True,
        "unique_initial_condition_ids": True,
        "a_b_boundary_disjoint": True,
    }
