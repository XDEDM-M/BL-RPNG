from __future__ import annotations

import json
from statistics import mean

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


METHODS = ("ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated")


def main() -> None:
    values = {method: [] for method in METHODS}
    hashes = {method: [] for method in METHODS}
    histories = {method: [] for method in METHODS}
    for task in range(16):
        for method in METHODS:
            path = resolve_in_project(
                f"artifacts/checkpoints/gray_scott/"
                f"gray_scott_task{task:02d}_m4_s42_{method}_e100.adapter.pt",
                must_exist=True,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload["metadata"]
            history = metadata["history"]
            validation = [float(row["validation"]["rollout_rel_l2"]) for row in history]
            if len(history) < 60:
                raise RuntimeError(f"Convergence run stopped before 60 epochs: {path}")
            values[method].append(float(metadata["best_validation_metric"]))
            hashes[method].append(sha256_file(path))
            earlier = mean(validation[-20:-10])
            later = mean(validation[-10:])
            histories[method].append({
                "task": task,
                "epochs": len(history),
                "best_epoch": min(range(len(validation)), key=validation.__getitem__),
                "relative_late_improvement": (earlier - later) / earlier,
                "best_validation_metric": min(validation),
            })
    means = {method: mean(rows) for method, rows in values.items()}
    family_means = {
        method: [mean(rows[start:start + 4]) for start in range(0, 16, 4)]
        for method, rows in values.items()
    }
    comparisons = {}
    for method in METHODS[1:]:
        family_gaps = [
            candidate / baseline - 1.0 for candidate, baseline in zip(
                family_means[method], family_means["ordinary_nlt"]
            )
        ]
        comparisons[method] = {
            "relative_gap_to_ordinary": means[method] / means["ordinary_nlt"] - 1.0,
            "task_wins": sum(
                candidate < baseline for candidate, baseline in zip(
                    values[method], values["ordinary_nlt"]
                )
            ),
            "family_relative_gaps": family_gaps,
        }
    payload = {
        "schema": "blrpng_gray_scott_converged_lr_summary_v1",
        "frozen": True,
        "passed": True,
        "seed": 42,
        "query_used_for_checkpoint_selection": False,
        "validation_means": means,
        "shift_family_means": family_means,
        "comparisons": comparisons,
        "convergence": histories,
        "maximum_relative_late_improvement": max(
            row["relative_late_improvement"]
            for rows in histories.values() for row in rows
        ),
        "adapter_sha256": hashes,
        "precommitted_confirmation_methods": [
            *METHODS, "lifted_scratch"
        ],
        "authorized_next_experiment": (
            "Train and freeze the matched converged lifted-scratch baselines; "
            "then generate the unopened confirmation query."
        ),
    }
    with workspace_audit("summarize_gray_scott_converged_lr") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_converged_lr_summary.json", payload
        )
        audit.update({"output": str(output), "query_selection_used": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
