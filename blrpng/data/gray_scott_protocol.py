from __future__ import annotations

from typing import Any

import numpy as np

from blrpng.paths import sha256_json


SHIFTS = ("high_frequency", "amplitude_ood", "correlation_shift", "parameter_ood")


def build_gray_scott_protocol(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    rng = np.random.default_rng(int(config["experiment"]["split_seed"]))

    def source_row(case_id: int, role: str) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "pair_id": case_id,
            "boundary_id": case_id,
            "initial_condition_id": case_id,
            "boundary_seed": int(rng.integers(0, 2**31 - 1)),
            "initial_seed": int(rng.integers(0, 2**31 - 1)),
            "boundary_regime": "source",
            "correlation": float(rng.uniform(0.2, 0.8)),
            "amplitude": float(rng.uniform(0.06, 0.14)),
            "feed": float(rng.uniform(0.035, 0.045)),
            "kill": float(rng.uniform(0.055, 0.065)),
            "role": role,
        }

    train_count = int(data["source_train_rollouts"])
    validation_count = int(data["source_validation_rollouts"])
    source_train = [source_row(case, "source_train") for case in range(train_count)]
    source_validation = [
        source_row(train_count + local, "source_validation")
        for local in range(validation_count)
    ]

    tasks = []
    tasks_per_shift = int(data["tasks_per_shift"])
    if int(data["target_tasks"]) != tasks_per_shift * len(SHIFTS):
        raise ValueError("Gray--Scott target_tasks must equal 4*tasks_per_shift.")
    support_count = max(int(value) for value in data["support_budgets"])
    validation_per_task = int(data["target_validation_rollouts"])
    query_per_task = int(data["target_query_rollouts"])
    for task_id in range(int(data["target_tasks"])):
        shift = SHIFTS[task_id // tasks_per_shift]
        correlation, amplitude, feed, kill = 0.5, 0.10, 0.04, 0.06
        if shift == "amplitude_ood":
            amplitude = float(data["target_boundary_amplitude_ood"])
        elif shift == "correlation_shift":
            correlation = -0.65
        elif shift == "parameter_ood":
            correlation = float((-0.2, 0.0, 0.2, 0.4)[task_id % tasks_per_shift])
            feed = float((0.047, 0.050, 0.052, 0.048)[task_id % tasks_per_shift])
            kill = float((0.066, 0.067, 0.065, 0.069)[task_id % tasks_per_shift])
            amplitude = 0.12
        boundary_id = 10_000 + task_id
        boundary_seed = int(rng.integers(0, 2**31 - 1))

        def samples(count: int, slot_offset: int, role: str) -> list[dict[str, Any]]:
            rows = []
            for local in range(count):
                slot = slot_offset + local
                pair_id = 2_000_000 + task_id * 1_000 + slot
                rows.append({
                    "case_id": pair_id,
                    "pair_id": pair_id,
                    "initial_condition_id": 200_000 + task_id * 1_000 + slot,
                    "initial_seed": int(rng.integers(0, 2**31 - 1)),
                    "role": role,
                })
            return rows

        tasks.append({
            "target_task_id": task_id,
            "boundary_id": boundary_id,
            "boundary_seed": boundary_seed,
            "boundary_regime": shift,
            "correlation": correlation,
            "amplitude": amplitude,
            "feed": feed,
            "kill": kill,
            "support": samples(support_count, 0, "target_support"),
            "validation": samples(validation_per_task, 100, "target_validation"),
            "query": samples(query_per_task, 200, "target_query"),
        })

    protocol: dict[str, Any] = {
        "schema": "blrpng_gray_scott_seed42_protocol_v1",
        "frozen": True,
        "seed": int(config["experiment"]["seed"]),
        "split_seed": int(config["experiment"]["split_seed"]),
        "design": {
            "source_resolution": int(data.get("pretrain_resolution", 128)),
            "source_steps": int(data["num_train_time_steps"]),
            "support_resolution": 64,
            "support_steps": int(data["num_train_time_steps"]),
            "query_resolutions": list(data["query_grids"]),
            "query_steps": int(data["num_blind_time_steps"]),
            "support_budgets": list(data["support_budgets"]),
            "primary_support_budget": 4,
            "primary_query_resolution": 256,
            "primary_horizon": 20,
            "long_horizon": 50,
            "query_for_checkpoint_selection": False,
            "shift_families": list(SHIFTS),
        },
        "source_train": source_train,
        "source_validation": source_validation,
        "targets": tasks,
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    return protocol


def validate_gray_scott_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    expected = protocol.get("protocol_sha256")
    unhashed = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if expected != sha256_json(unhashed):
        raise ValueError("Frozen Gray--Scott protocol hash mismatch.")
    source_boundaries = {
        int(row["boundary_id"])
        for split in ("source_train", "source_validation")
        for row in protocol[split]
    }
    target_boundaries: set[int] = set()
    split_initials = {split: set() for split in ("support", "validation", "query")}
    pair_ids: set[int] = set()
    shifts: dict[str, int] = {shift: 0 for shift in SHIFTS}
    for task in protocol["targets"]:
        boundary = int(task["boundary_id"])
        if boundary in target_boundaries:
            raise ValueError("Duplicate Gray--Scott target boundary.")
        target_boundaries.add(boundary)
        shifts[task["boundary_regime"]] += 1
        for split in split_initials:
            for sample in task[split]:
                initial = int(sample["initial_condition_id"])
                pair = int(sample["pair_id"])
                if any(initial in values for values in split_initials.values()):
                    raise ValueError("Gray--Scott initial condition leaked across splits.")
                if pair in pair_ids:
                    raise ValueError("Duplicate Gray--Scott pair id.")
                split_initials[split].add(initial)
                pair_ids.add(pair)
    if source_boundaries & target_boundaries:
        raise ValueError("Gray--Scott source/target boundary leakage.")
    return {
        "passed": True,
        "source_train": len(protocol["source_train"]),
        "source_validation": len(protocol["source_validation"]),
        "target_tasks": len(protocol["targets"]),
        "target_shifts": shifts,
        "target_samples": {key: len(value) for key, value in split_initials.items()},
        "boundary_leakage": False,
        "initial_condition_leakage": False,
    }
