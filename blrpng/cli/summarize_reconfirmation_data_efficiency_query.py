from __future__ import annotations

import argparse
import json
from collections import defaultdict

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap, paired_ratio_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


BUDGETS = (1, 2, 4, 8)


def _families() -> dict[int, str]:
    payload = json.loads(resolve_in_project(
        "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    return {int(row["target_task_id"]): str(row["family"]) for row in payload["tasks"]}


def _report(task: int, budget: int, resolution: int, horizon: int) -> tuple[str, dict]:
    relative = (
        "artifacts/reports/reconfirmation/"
        f"task{task:02d}_m{budget}_s42_nlt_query{resolution}_h{horizon}.json"
    )
    path = resolve_in_project(relative, must_exist=True)
    return relative, json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze HR-query support-size and long-horizon statistics."
    )
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    args = parser.parse_args()
    families = _families()
    records = []
    reports = []
    family_values: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task, family in sorted(families.items()):
        reference_ids = None
        for budget in BUDGETS:
            relative, payload = _report(task, budget, args.resolution, 20)
            rows = payload["metrics"]["rollouts"]
            ids = [int(row["initial_condition_id"]) for row in rows]
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError(f"Unpaired support-size query units for task {task}.")
            for row in rows:
                value = float(row["relative_l2"])
                family_values[family][budget].append(value)
                records.append({
                    "seed": 42,
                    "boundary_id": task,
                    "initial_condition_id": int(row["initial_condition_id"]),
                    "method": f"m{budget}_h20",
                    "relative_l2": value,
                })
            reports.append({"path": relative, "sha256": sha256_file(resolve_in_project(relative))})
        relative, payload = _report(task, 4, args.resolution, 50)
        rows = payload["metrics"]["rollouts"]
        if [int(row["initial_condition_id"]) for row in rows] != reference_ids:
            raise ValueError(f"Unpaired long-horizon query units for task {task}.")
        for row in rows:
            records.append({
                "seed": 42,
                "boundary_id": task,
                "initial_condition_id": int(row["initial_condition_id"]),
                "method": "m4_h50",
                "relative_l2": float(row["relative_l2"]),
            })
        reports.append({"path": relative, "sha256": sha256_file(resolve_in_project(relative))})
    means = {
        label: sum(row["relative_l2"] for row in records if row["method"] == label)
        / sum(row["method"] == label for row in records)
        for label in [*(f"m{budget}_h20" for budget in BUDGETS), "m4_h50"]
    }
    comparisons = {
        f"m4_vs_m{budget}": paired_hierarchical_bootstrap(
            records,
            method="m4_h20",
            baseline=f"m{budget}_h20",
            replicates=args.replicates,
            seed=20260910 + budget,
        )
        for budget in (1, 2, 8)
    }
    horizon_ratio = paired_ratio_bootstrap(
        records,
        method="m4_h50",
        baseline="m4_h20",
        replicates=args.replicates,
        seed=20260920,
    )
    family_means = {
        family: {
            str(budget): sum(values) / len(values)
            for budget, values in sorted(curves.items())
        }
        for family, curves in sorted(family_values.items())
    }
    payload = {
        "schema": "blrpng_reconfirmation_query_efficiency_v1",
        "frozen": True,
        "seed": 42,
        "resolution": args.resolution,
        "rollouts_per_budget": 768,
        "mean_relative_l2": means,
        "support_size_comparisons": comparisons,
        "horizon50_to_horizon20_ratio": horizon_ratio,
        "family_mean_h20_relative_l2": family_means,
        "selection_note": "m4 was frozen as primary before opening HR query; no post-query reselection.",
        "evaluation_reports": reports,
    }
    with workspace_audit(
        f"summarize_reconfirmation_data_efficiency_query_{args.resolution}"
    ) as audit:
        output = write_json(
            "artifacts/reports/"
            f"allen_cahn_reconfirmation_seed42_query{args.resolution}_efficiency.json",
            payload,
        )
        freeze = write_json(
            "artifacts/manifests/"
            f"allen_cahn_reconfirmation_seed42_query{args.resolution}_efficiency_freeze.json",
            {
                "schema": "blrpng_reconfirmation_query_efficiency_freeze_v1",
                "passed": True,
                "resolution": args.resolution,
                "report": str(output.relative_to(PROJECT_ROOT)),
                "report_sha256": sha256_file(output),
            },
        )
        audit.update({"output": str(output), "freeze": str(freeze)})
    print(json.dumps({
        "output": str(output),
        "mean_relative_l2": means,
        "support_size_comparisons": comparisons,
        "horizon50_to_horizon20_ratio": horizon_ratio,
    }, indent=2))


if __name__ == "__main__":
    main()
