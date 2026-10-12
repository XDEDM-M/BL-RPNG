from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from blrpng.data.manifest import CaseSpec, read_manifest, write_manifest


def _source_boundaries(path: str | Path, count: int) -> list[dict[str, Any]]:
    source = read_manifest(path)
    by_boundary: dict[int, dict[str, Any]] = {}
    for case in source["cases"]:
        if case["split"] == "b_train":
            by_boundary.setdefault(int(case["boundary_id"]), case)
    selected = [by_boundary[key] for key in sorted(by_boundary)[:count]]
    if len(selected) != count:
        raise ValueError(f"Requested {count} target boundaries, found {len(selected)}.")
    return selected


def _cases(
    boundaries: list[dict[str, Any]],
    *,
    split: str,
    initial_conditions: int,
    seed: int,
    case_offset: int,
    initial_offset: int,
    task_offset: int = 0,
) -> list[CaseSpec]:
    rng = np.random.default_rng(seed)
    rows = []
    for task_index, boundary in enumerate(boundaries):
        for local_ic in range(initial_conditions):
            index = len(rows)
            task_id = task_offset + task_index
            rows.append(
                CaseSpec(
                    case_id=case_offset + index,
                    pair_id=case_offset + index,
                    boundary_id=int(boundary["boundary_id"]),
                    initial_condition_id=initial_offset + index,
                    split=split,
                    boundary_coefficients=[float(v) for v in boundary["boundary_coefficients"]],
                    boundary_frequencies=[float(v) for v in boundary["boundary_frequencies"]],
                    boundary_amplitude=float(boundary["boundary_amplitude"]),
                    initial_coefficients=[float(v) for v in rng.normal(size=4)],
                    initial_phase=float(0.17 * local_ic + 0.031 * task_index),
                )
            )
    return rows


def create_calibration_manifest(config: dict[str, Any]) -> Path:
    data = config["data"]
    boundaries = _source_boundaries(data["source_manifest"], int(data["target_boundaries"]))
    seed = int(config["experiment"]["seed"])
    split_specs = (
        ("support_lr", int(data["support_initial_conditions"]), seed + 1),
        ("validation_lr", int(data["validation_initial_conditions"]), seed + 2),
        ("query_hr", int(data["query_initial_conditions"]), seed + 3),
        ("query_512", int(data["query_512_initial_conditions"]), seed + 4),
        ("oracle_train", int(data["oracle_initial_conditions"]), seed + 5),
        ("oracle_validation", int(data["oracle_validation_initial_conditions"]), seed + 6),
    )
    rows: list[CaseSpec] = []
    next_id = 2_000_000
    split_ranges = {}
    for split, initial_conditions, split_seed in split_specs:
        added = _cases(
            boundaries, split=split, initial_conditions=initial_conditions,
            seed=split_seed, case_offset=next_id, initial_offset=next_id,
        )
        rows.extend(added)
        split_ranges[split] = {
            "case_start": next_id, "cases": len(added),
            "initial_conditions_per_task": initial_conditions,
        }
        next_id += len(added)
    return write_manifest(
        "artifacts/manifests/allen_cahn_v2_calibration.json",
        rows,
        metadata={
            "experiment": config["experiment"]["name"],
            "seed": seed,
            "source_manifest": data["source_manifest"],
            "target_boundary_ids": [int(row["boundary_id"]) for row in boundaries],
            "task_mapping": {
                str(int(row["boundary_id"])): index for index, row in enumerate(boundaries)
            },
            "split_ranges": split_ranges,
            "support_budgets": list(data["support_budgets"]),
            "oracle_budgets": list(data["oracle_budgets"]),
            "frozen": True,
        },
    )


def validate_task_isolation(manifest: dict[str, Any]) -> dict[str, Any]:
    cases = manifest["cases"]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_split.setdefault(case["split"], []).append(case)
    required = {"support_lr", "validation_lr", "query_hr", "query_512", "oracle_train", "oracle_validation"}
    if not required <= set(by_split):
        raise ValueError(f"Calibration manifest misses splits: {sorted(required - set(by_split))}")
    initial_sets = {
        split: {int(row["initial_condition_id"]) for row in values}
        for split, values in by_split.items()
    }
    names = sorted(initial_sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if initial_sets[first] & initial_sets[second]:
                raise ValueError(f"Initial-condition leakage between {first} and {second}.")
    boundary_sets = {
        split: {int(row["boundary_id"]) for row in values}
        for split, values in by_split.items()
    }
    reference = next(iter(boundary_sets.values()))
    if any(values != reference for values in boundary_sets.values()):
        raise ValueError("All calibration splits must contain exactly the same target boundaries.")
    task_mapping = manifest["metadata"]["task_mapping"]
    if set(map(int, task_mapping)) != reference:
        raise ValueError("Task mapping does not match target boundaries.")
    return {
        "tasks": len(reference),
        "splits": {key: len(value) for key, value in by_split.items()},
        "support_query_initials_disjoint": True,
        "boundary_tasks_aligned": True,
    }
