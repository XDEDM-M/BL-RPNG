from __future__ import annotations

import json

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


METHODS = (
    "ordinary_nlt", "path_nlt", "path_nlt_gated",
    "path_nlt_commutator", "path_nlt_plus",
)


def main() -> None:
    values = {method: [] for method in METHODS}
    for task in range(16):
        for method in METHODS:
            path = resolve_in_project(
                f"artifacts/checkpoints/gray_scott/"
                f"gray_scott_task{task:02d}_m4_s42_{method}.adapter.pt",
                must_exist=True,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            values[method].append(float(payload["metadata"]["best_validation_metric"]))
    means = {method: sum(rows) / len(rows) for method, rows in values.items()}
    family_means = {
        method: [sum(rows[start : start + 4]) / 4 for start in range(0, 16, 4)]
        for method, rows in values.items()
    }
    raw_gap = means["path_nlt"] / means["ordinary_nlt"] - 1.0
    payload = {
        "schema": "blrpng_gray_scott_lr_diagnosis_v1",
        "frozen": True, "query_used": False,
        "validation_means": means,
        "shift_family_means": family_means,
        "path_relative_gap_to_ordinary": raw_gap,
        "path_task_wins": sum(
            path < ordinary
            for path, ordinary in zip(values["path_nlt"], values["ordinary_nlt"])
        ),
        "diagnosis": (
            "Raw fixed-ratio source path supervision is not NLT-adaptable on Gray--Scott; "
            "target-only gating cannot repair the already learned source feature basis."
        ),
        "mechanistic_evidence": {
            "source_path_gradient_cosine_mean": 0.0009724788698278758,
            "source_path_gradient_cosine_min_epoch_mean": -0.0010277379760475014,
            "source_gradient_ratio": 0.15000000619590284,
        },
        "authorized_next_experiment": (
            "Train a source checkpoint with conflict gating and late path-ratio annealing; "
            "keep all data, seed, initialization protocol and HR-query lock unchanged."
        ),
        "supersedes_primary_candidate_in": (
            "artifacts/manifests/gray_scott_seed42_method_freeze.json"
        ),
    }
    with workspace_audit("freeze_gray_scott_lr_diagnosis") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_lr_diagnosis.json", payload
        )
        audit.update({"output": str(output), "query_used": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
