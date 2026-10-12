from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from blrpng.analysis.bootstrap import select_oracle_plateau
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file, write_json


PATTERN = re.compile(
    r"v2_task(?P<task>\d+)_m(?P<budget>\d+)_s(?P<seed>\d+)_"
    r"(?P<method>hr_oracle(?:_ordinary|_path)?)_conv_(?P<initialization>scratch|ordinary|path)_"
    r"lr(?P<lr>[^_]+)_e(?P<epochs>\d+)\.pt$"
)


def _curve_audit(history: list[dict[str, Any]], plateau_fraction: float) -> dict[str, Any]:
    metrics = [float(row["validation"]["rollout_rel_l2"]) for row in history]
    learning_rates = [float(row["lr"]) for row in history]
    if len(metrics) < 20:
        return {
            "epochs": len(metrics), "validation_curve": metrics,
            "learning_rate_curve": learning_rates,
            "optimization_plateau": False, "reason": "fewer_than_20_epochs",
        }
    window = max(10, min(30, len(metrics) // 5))
    previous = min(metrics[-2 * window : -window])
    recent = min(metrics[-window:])
    improvement = (previous - recent) / max(previous, 1.0e-15)
    best = min(metrics)
    best_epoch = metrics.index(best)
    return {
        "epochs": len(metrics),
        "validation_curve": metrics,
        "learning_rate_curve": learning_rates,
        "window_epochs": window,
        "initial_metric": metrics[0],
        "best_metric": best,
        "best_epoch": best_epoch,
        "final_metric": metrics[-1],
        "previous_window_best": previous,
        "recent_window_best": recent,
        "recent_relative_improvement": improvement,
        "optimization_plateau_threshold": plateau_fraction,
        "optimization_plateau": bool(improvement <= plateau_fraction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HR specialist optimization and data saturation.")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/v2")
    parser.add_argument("--output", default="artifacts/reports/oracle_convergence_audit.json")
    parser.add_argument("--optimization-threshold", type=float, default=0.005)
    parser.add_argument("--data-threshold", type=float, default=0.01)
    args = parser.parse_args()
    directory = (PROJECT_ROOT / args.checkpoint_dir).resolve()
    rows = []
    for checkpoint_path in sorted(directory.glob("*conv_*_e*.pt")):
        match = PATTERN.match(checkpoint_path.name)
        if not match:
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        audit = _curve_audit(checkpoint["history"], args.optimization_threshold)
        rows.append({
            **{key: int(value) if key in {"task", "budget", "seed", "epochs"} else value
               for key, value in match.groupdict().items()},
            "learning_rate_tag": match.group("lr"),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "stopped_early": bool(checkpoint.get("stopped_early", False)),
            "optimizer_steps": int(checkpoint.get("optimizer_steps", 0)),
            **audit,
        })
    by_task_budget: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_task_budget.setdefault((row["task"], row["budget"]), []).append(row)
    selected = []
    for (task, budget), candidates in sorted(by_task_budget.items()):
        best = min(candidates, key=lambda row: row["best_metric"])
        selected.append({"task": task, "budget": budget, **best})

    data_curves: dict[str, Any] = {}
    tasks = sorted({row["task"] for row in selected})
    for task in tasks:
        values = {row["budget"]: row["best_metric"] for row in selected if row["task"] == task}
        if len(values) >= 2:
            data_curves[str(task)] = select_oracle_plateau(values, improvement_threshold=args.data_threshold)
    sufficient_budgets = bool(data_curves) and all(len(curve["relative_improvements"]) >= 2 for curve in data_curves.values())
    optimization_pass = bool(selected) and all(row["optimization_plateau"] for row in selected)
    def two_consecutive_plateaus(curve: dict[str, Any]) -> bool:
        improvements = [
            float(value) for value in curve["relative_improvements"].values()
        ]
        return any(
            improvements[index] < args.data_threshold
            and improvements[index + 1] < args.data_threshold
            for index in range(len(improvements) - 1)
        )

    data_pass = sufficient_budgets and all(
        two_consecutive_plateaus(curve) for curve in data_curves.values()
    )
    payload = {
        "schema": "blrpng_oracle_convergence_audit_v1",
        "rows": rows,
        "selected_per_task_budget": selected,
        "data_curves": data_curves,
        "gates": {
            "optimization_plateau": optimization_pass,
            "at_least_three_budgets": sufficient_budgets,
            "two_consecutive_data_improvements_below_1pct": data_pass,
            "saturated_hr_oracle": bool(optimization_pass and data_pass),
        },
        "interpretation": (
            "Saturated HR oracle remains prohibited until optimization plateaus and two consecutive "
            "support doublings improve less than one percent."
        ),
    }
    with workspace_audit("summarize_oracle_convergence") as workspace:
        output = write_json(args.output, payload)
        gate_output = None
        if payload["gates"]["saturated_hr_oracle"]:
            gate_output = write_json(
                "artifacts/manifests/allen_cahn_oracle_saturation_gate.json",
                {
                    "schema": "blrpng_oracle_saturation_gate_v1",
                    "passed": True,
                    "oracle_status": "saturated",
                    "audit_report": str(output.relative_to(PROJECT_ROOT)),
                    "audit_sha256": sha256_file(output),
                    "selected_per_task_budget": selected,
                },
            )
        workspace.update({"output": str(output), "gate_output": str(gate_output) if gate_output else None})
    print(json.dumps({
        "output": str(output), "gate_output": str(gate_output) if gate_output else None,
        "runs": len(rows), "gates": payload["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
