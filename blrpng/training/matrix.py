from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FineTuneTask:
    task_id: str
    seed: int
    boundary_budget: int
    initialization: str
    target_objective: str
    initialization_group: str
    data_order_group: str


def causal_matrix(config: dict[str, Any]) -> list[FineTuneTask]:
    training = config["training"]
    budgets = config["data"]["b_budgets"]
    tasks = []
    for budget in budgets:
        for seed in training["seeds"]:
            # The group IDs are explicit audit fields: all six causal cells for
            # one seed/budget must use the same scratch draw and minibatch order.
            init_group = f"seed{seed}"
            order_group = f"seed{seed}_budget{budget}"
            for initialization in training["source_initializations"]:
                for objective in training["target_objectives"]:
                    task_id = (
                        f"b{budget}_s{seed}_{initialization}_to_{objective}"
                    )
                    tasks.append(
                        FineTuneTask(
                            task_id=task_id,
                            seed=int(seed),
                            boundary_budget=int(budget),
                            initialization=str(initialization),
                            target_objective=str(objective),
                            initialization_group=init_group,
                            data_order_group=order_group,
                        )
                    )
    if len(tasks) != 120:
        raise ValueError(f"Formal causal matrix must have 120 tasks, got {len(tasks)}")
    return tasks


def ablation_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in config["training"]["seeds"]:
        rows.extend(
            [
                {
                    "task_id": f"ablation_b4_s{seed}_raw_scratch_final",
                    "seed": seed,
                    "boundary_budget": 4,
                    "representation": "raw_legacy",
                    "initialization": "scratch",
                    "objective": "final_only",
                },
                {
                    "task_id": f"ablation_b4_s{seed}_raw_legacy_path",
                    "seed": seed,
                    "boundary_budget": 4,
                    "representation": "raw_legacy",
                    "initialization": "legacy_pretrained",
                    "objective": "legacy_path",
                    "lambda": 0.01,
                },
                {
                    "task_id": f"ablation_b4_s{seed}_dual_stage",
                    "seed": seed,
                    "boundary_budget": 4,
                    "representation": "lifted",
                    "initialization": "dual_stage_pretrained",
                    "objective": "dual_stage",
                    "probes": [
                        {"layer": 1, "progress": 0.25},
                        {"layer": 3, "progress": 0.75},
                    ],
                },
            ]
        )
    return rows


def matrix_payload(config: dict[str, Any]) -> dict[str, Any]:
    causal = [asdict(task) for task in causal_matrix(config)]
    return {
        "schema": "blrpng_experiment_matrix_v1",
        "config_sha256": config["_config_sha256"],
        "causal_tasks": causal,
        "causal_task_count": len(causal),
        "ablations": ablation_matrix(config),
        "matching_invariants": [
            "same seed => identical scratch initialization",
            "same seed and boundary budget => identical data order",
            "identical epochs, optimizer steps, and sample exposure",
        ],
    }
