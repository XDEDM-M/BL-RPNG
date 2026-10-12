from __future__ import annotations

from typing import Any

import numpy as np

from blrpng.paths import sha256_json


FAMILIES = ("fourier", "bump", "piecewise_smooth")


def build_poisson_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Build the complete protocol before any formal target query is generated."""
    data = config["data"]
    split_seed = int(config["experiment"]["split_seed"])
    rng = np.random.default_rng(split_seed)

    def source_rows(count: int, *, offset: int, role: str) -> list[dict[str, Any]]:
        rows = []
        for local in range(count):
            case_id = offset + local
            rows.append({
                "case_id": case_id,
                "pair_id": case_id,
                "boundary_id": case_id,
                "forcing_id": case_id,
                "family": FAMILIES[local % len(FAMILIES)],
                "boundary_seed": int(rng.integers(0, 2**31 - 1)),
                "forcing_seed": int(rng.integers(0, 2**31 - 1)),
                "regime": "source",
                "role": role,
            })
        return rows

    train_count = int(data["pretrain_cases"])
    validation_count = int(data["validation_cases"])
    source_train = source_rows(train_count, offset=0, role="source_train")
    source_validation = source_rows(
        validation_count, offset=train_count, role="source_validation"
    )

    task_count = int(data["target_tasks"])
    tasks_per_family = int(data["tasks_per_family"])
    if task_count != len(FAMILIES) * tasks_per_family:
        raise ValueError("Poisson target_tasks must equal 3 * tasks_per_family.")
    support_count = max(int(value) for value in data["support_budgets"])
    validation_per_task = int(data["target_validation_cases"])
    query_per_task = int(data["target_query_cases"])
    targets = []
    for task_id in range(task_count):
        family = FAMILIES[task_id // tasks_per_family]
        boundary_seed = int(rng.integers(0, 2**31 - 1))
        boundary_id = 10_000 + task_id

        def samples(count: int, *, slot_offset: int, role: str) -> list[dict[str, Any]]:
            values = []
            for local in range(count):
                slot = slot_offset + local
                pair_id = 1_000_000 + task_id * 1_000 + slot
                values.append({
                    "case_id": pair_id,
                    "pair_id": pair_id,
                    "forcing_id": 100_000 + task_id * 1_000 + slot,
                    "forcing_seed": int(rng.integers(0, 2**31 - 1)),
                    "role": role,
                })
            return values

        targets.append({
            "target_task_id": task_id,
            "boundary_id": boundary_id,
            "boundary_seed": boundary_seed,
            "family": family,
            "regime": "target",
            "support": samples(support_count, slot_offset=0, role="target_support"),
            "validation": samples(
                validation_per_task, slot_offset=100, role="target_validation"
            ),
            "query": samples(query_per_task, slot_offset=200, role="target_query"),
        })

    protocol: dict[str, Any] = {
        "schema": "blrpng_poisson_negative_control_protocol_v1",
        "frozen": True,
        "experiment_seed": int(config["experiment"]["seed"]),
        "split_seed": split_seed,
        "design": {
            "linear_newton_iterations": 1,
            "source_resolution": int(data["pretrain_resolution"]),
            "support_resolution": int(data["support_resolution"]),
            "query_resolutions": list(data["query_resolutions"]),
            "support_budgets": list(data["support_budgets"]),
            "primary_support_budget": 4,
            "primary_query_resolution": 256,
            "source_objectives": ["final_only", "blrpng"],
            "target_adapter": "nlt_final_only",
            "negative_control_threshold": 0.05,
            "primary_contrast": "path_pretrain_nlt_vs_final_pretrain_nlt",
            "query_for_checkpoint_selection": False,
        },
        "source_train": source_train,
        "source_validation": source_validation,
        "targets": targets,
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    return protocol


def validate_poisson_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    expected = protocol.get("protocol_sha256")
    unhashed = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if expected != sha256_json(unhashed):
        raise ValueError("Frozen Poisson protocol hash mismatch.")
    boundary_ids: set[int] = set()
    forcing_ids: dict[str, set[int]] = {key: set() for key in ("support", "validation", "query")}
    pairs: set[int] = set()
    for task in protocol["targets"]:
        boundary_id = int(task["boundary_id"])
        if boundary_id in boundary_ids:
            raise ValueError("Duplicate target boundary_id.")
        boundary_ids.add(boundary_id)
        for split in forcing_ids:
            for sample in task[split]:
                forcing_id = int(sample["forcing_id"])
                pair_id = int(sample["pair_id"])
                if forcing_id in set().union(*(forcing_ids[name] for name in forcing_ids)):
                    raise ValueError("Poisson forcing leaked across target splits.")
                if pair_id in pairs:
                    raise ValueError("Duplicate Poisson coarse/fine pair id.")
                forcing_ids[split].add(forcing_id)
                pairs.add(pair_id)
    source_boundaries = {
        int(row["boundary_id"])
        for split in ("source_train", "source_validation")
        for row in protocol[split]
    }
    if source_boundaries & boundary_ids:
        raise ValueError("Source/target Poisson boundary leakage.")
    return {
        "passed": True,
        "source_train_cases": len(protocol["source_train"]),
        "source_validation_cases": len(protocol["source_validation"]),
        "target_tasks": len(protocol["targets"]),
        "target_samples": {key: len(values) for key, values in forcing_ids.items()},
        "boundary_leakage": False,
        "forcing_split_leakage": False,
    }
