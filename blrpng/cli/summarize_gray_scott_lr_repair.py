from __future__ import annotations

import json
from statistics import mean

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


METHODS = ("ordinary_nlt", "safe_path_nlt", "safe_path_nlt_gated")


def _adapter(task: int, method: str) -> tuple[float, dict, str]:
    path = resolve_in_project(
        f"artifacts/checkpoints/gray_scott/"
        f"gray_scott_task{task:02d}_m4_s42_{method}.adapter.pt",
        must_exist=True,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        float(payload["metadata"]["best_validation_metric"]),
        payload,
        sha256_file(path),
    )


def main() -> None:
    protocol_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_repair_protocol.json",
        must_exist=True,
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("frozen") or protocol.get("query_used") is not False:
        raise PermissionError("The precommitted LR repair protocol is invalid.")

    source_path = resolve_in_project(
        "artifacts/checkpoints/gray_scott/"
        "gray_scott_source_s42_blrpng_safe_annealed.inference.pt",
        must_exist=True,
    )
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    history = source["history"]
    train_rows = [row["train"] for row in history]
    source_diagnostics = {
        "best_validation_metric": float(source["best_validation_metric"]),
        "checkpoint_sha256": sha256_file(source_path),
        "mean_gradient_cosine": mean(row.get("gradient_cosine", 0.0) for row in train_rows),
        "minimum_epoch_mean_gradient_cosine": min(
            row.get("gradient_cosine", 0.0) for row in train_rows
        ),
        "mean_gate_active_fraction": mean(
            row.get("path_gate_active", 0.0) for row in train_rows
        ),
        "mean_actual_path_weight": mean(row.get("path_weight", 0.0) for row in train_rows),
        "final_epoch_actual_path_weight": train_rows[-1].get("path_weight", 0.0),
        "mean_realized_gradient_ratio": mean(
            row.get("gradient_ratio", 0.0) for row in train_rows
        ),
    }

    values = {method: [] for method in METHODS}
    hashes = {method: [] for method in METHODS}
    for task in range(16):
        for method in METHODS:
            value, payload, digest = _adapter(task, method)
            values[method].append(value)
            hashes[method].append(digest)
            if method.startswith("safe_path_"):
                if payload["base_checkpoint_sha256"] != source_diagnostics["checkpoint_sha256"]:
                    raise RuntimeError("Adapter base hash does not match the frozen safe source.")

    means = {method: mean(rows) for method, rows in values.items()}
    family_means = {
        method: [mean(rows[start : start + 4]) for start in range(0, 16, 4)]
        for method, rows in values.items()
    }
    ordinary = values["ordinary_nlt"]
    rule = protocol["mechanism_support_rule"]
    candidates = {}
    for method in METHODS[1:]:
        family_gaps = [
            candidate / baseline - 1.0
            for candidate, baseline in zip(
                family_means[method], family_means["ordinary_nlt"]
            )
        ]
        wins = sum(candidate < baseline for candidate, baseline in zip(values[method], ordinary))
        improved_families = sum(gap < 0.0 for gap in family_gaps)
        supported = (
            means[method] < means["ordinary_nlt"]
            and improved_families >= rule["minimum_improved_shift_families_out_of_4"]
            and wins >= rule["minimum_task_wins_out_of_16"]
            and max(family_gaps) <= rule["maximum_allowed_family_degradation"]
        )
        candidates[method] = {
            "relative_gap_to_ordinary": means[method] / means["ordinary_nlt"] - 1.0,
            "task_wins": wins,
            "improved_shift_families": improved_families,
            "family_relative_gaps": family_gaps,
            "mechanism_supported": supported,
        }

    supported = [method for method, row in candidates.items() if row["mechanism_supported"]]
    primary = min(supported, key=lambda method: means[method]) if supported else "ordinary_nlt"
    payload = {
        "schema": "blrpng_gray_scott_lr_repair_summary_v1",
        "frozen": True,
        "passed": True,
        "seed": 42,
        "query_used": False,
        "source_diagnostics": source_diagnostics,
        "validation_means": means,
        "shift_family_means": family_means,
        "candidate_tests": candidates,
        "mechanism_supported_on_lr_validation": bool(supported),
        "lr_selected_primary": primary,
        "precommitted_hr_methods": protocol["precommitted_hr_methods"],
        "adapter_sha256": hashes,
        "authorized_next_experiment": "Generate and evaluate the frozen 128-grid query only.",
    }
    with workspace_audit("summarize_gray_scott_lr_repair") as audit:
        output = write_json(
            "artifacts/manifests/gray_scott_seed42_lr_repair_summary.json", payload
        )
        audit.update({"output": str(output), "query_used": False})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
