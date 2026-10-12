from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from blrpng.analysis.bootstrap import (
    paired_equivalence_test,
    paired_hierarchical_bootstrap,
    paired_ratio_bootstrap,
)
from blrpng.analysis.cost import cost_summary
from blrpng.analysis.gates_v2 import calibration_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.paths import PROJECT_ROOT, resolve_in_project, write_json


def _load(path: str) -> dict:
    return json.loads(resolve_in_project(path, must_exist=True).read_text())


def _timing(index_path: str, *, limit: int | None = None, horizon: int | None = None) -> dict[str, float]:
    index = read_index(index_path)
    rows = index["shards"][:limit]
    assembly = solve = 0.0
    for row in rows:
        shard = torch.load(resolve_in_project(row["path"], must_exist=True), map_location="cpu", weights_only=False)
        meta = shard["meta"]
        assembly += float(meta.get("assembly_time", 0.0))
        timings = meta.get("solver_timing", [])
        solve += sum(float(value) for value in (timings[:horizon] if horizon else timings))
    return {"assembly_seconds": assembly, "solve_seconds": solve}


def _training_seconds(records: list[dict], method: str, task: int) -> float:
    for row in records:
        if row["method"] == method and int(row["target_task_id"]) == task:
            return float(row.get("training_seconds", 0.0))
    raise KeyError((method, task))


def _inference_seconds(records: list[dict], method: str, task: int) -> float:
    values = [float(row["inference_seconds_per_query"]) for row in records
              if row["method"] == method and int(row["target_task_id"]) == task
              and "inference_seconds_per_query" in row]
    if not values:
        raise KeyError((method, task, "inference_seconds_per_query"))
    return float(np.mean(values))


def _task_bootstrap(values: list[float], *, replicates: int, seed: int) -> tuple[float, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(replicates, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze A1 statistics, target costs, and strict gate.")
    parser.add_argument("--matrix", default="artifacts/reports/v2_seed42_query256_h20.json")
    parser.add_argument("--oracle", default="artifacts/reports/v2_seed42_oracle_query256_h20.json")
    parser.add_argument("--curves", default="artifacts/reports/v2_oracle_curves.json")
    parser.add_argument("--matrix128", default="artifacts/reports/v2_seed42_query128_h20.json")
    parser.add_argument("--method", default="blrpng_v2")
    parser.add_argument("--scratch", default="lr_scratch")
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--output", default="artifacts/reports/v2_seed42_calibration_summary.json")
    args = parser.parse_args()
    matrix, oracle, curves = _load(args.matrix), _load(args.oracle), _load(args.curves)
    matrix128 = _load(args.matrix128) if resolve_in_project(args.matrix128).exists() else {"records": []}
    records = matrix["records"] + oracle["records"]
    ratio = paired_ratio_bootstrap(records, method=args.method, baseline="hr_oracle",
                                   replicates=args.replicates)
    margin = .05
    equivalence = paired_equivalence_test(
        records,
        method=args.method,
        baseline="hr_oracle",
        noninferiority_margin=1.0 + margin,
        equivalence_lower=1.0 - margin,
        replicates=args.replicates,
    )
    scratch = paired_hierarchical_bootstrap(records, method=args.method, baseline=args.scratch,
                                            replicates=args.replicates)
    costs, speedups = {}, []
    for task in range(8):
        selected = int(curves["tasks"][str(task)]["selected_budget"])
        low = _timing(f"artifacts/data/tasks/task{task:02d}_support_lr_64_m4.index.json")
        high = _timing(f"artifacts/data/tasks/task{task:02d}_oracle_train_256_m{selected}.index.json")
        candidate_cost = {**low,
            "adaptation_seconds": _training_seconds(matrix["records"], args.method, task),
            "inference_seconds_per_query": _inference_seconds(matrix["records"], args.method, task)}
        oracle_cost = {**high,
            "adaptation_seconds": _training_seconds(oracle["records"], "hr_oracle", task),
            "inference_seconds_per_query": _inference_seconds(oracle["records"], "hr_oracle", task)}
        costs[str(task)] = cost_summary(candidate_cost, oracle_cost, queries=16)
        costs[str(task)]["selected_oracle_budget"] = selected
        costs[str(task)]["method_components"] = candidate_cost
        costs[str(task)]["oracle_components"] = oracle_cost
        speedups.append(costs[str(task)]["marginal_speedup"])
    speedup_point, speedup_ci = _task_bootstrap(speedups, replicates=args.replicates, seed=20260813)
    gate = calibration_gate(
        oracle_ratio_ci95=tuple(equivalence["one_sided_ci"]), marginal_speedup_ci95=tuple(speedup_ci),
        scratch_improvement=float(scratch["relative_improvement"]), oracle_margin=margin,
    )
    methods = sorted({row["method"] for row in matrix["records"]})
    method_rows = []
    for method in methods:
        selected = [r for r in matrix["records"] if r["method"] == method]
        per_task_costs = []
        for task in range(8):
            low = _timing(f"artifacts/data/tasks/task{task:02d}_support_lr_64_m4.index.json")
            per_task_costs.append(
                low["assembly_seconds"] + low["solve_seconds"]
                + _training_seconds(matrix["records"], method, task)
                + 16 * _inference_seconds(matrix["records"], method, task)
            )
        method_rows.append({
            "method": method,
            "relative_l2": float(np.mean([r["relative_l2"] for r in selected])),
            "cost_seconds": float(np.mean(per_task_costs)),
        })
    method_rows.append({
        "method": "hr_oracle", "relative_l2": float(np.mean([r["relative_l2"] for r in oracle["records"]])),
        "cost_seconds": float(np.mean([value["marginal_oracle_seconds"] for value in costs.values()])),
    })
    heatmap_baseline = "ordinary_final"
    heatmap_methods = [m for m in methods if m not in {args.scratch, heatmap_baseline}]
    heatmap = []
    for method in heatmap_methods:
        row = []
        for task in range(8):
            candidate = [r["relative_l2"] for r in matrix["records"] if r["method"] == method and int(r["target_task_id"]) == task]
            baseline = [r["relative_l2"] for r in matrix["records"] if r["method"] == heatmap_baseline and int(r["target_task_id"]) == task]
            row.append(float((np.mean(baseline) - np.mean(candidate)) / max(np.mean(baseline), 1e-15)))
        heatmap.append(row)
    resolution_curves = {}
    for method in methods:
        at128 = [r["relative_l2"] for r in matrix128["records"] if r["method"] == method]
        at256 = [r["relative_l2"] for r in matrix["records"] if r["method"] == method]
        if at128 and at256:
            resolution_curves[method] = {"128": float(np.mean(at128)), "256": float(np.mean(at256))}
    checkpoint = torch.load(PROJECT_ROOT / "artifacts/checkpoints/v2/v2_task00_m4_s42_blrpng_v2.adapter.pt", map_location="cpu", weights_only=False)
    history = checkpoint["metadata"]["history"]
    gradient_history = [{"epoch": i, **{k: float(item["train"].get(k, 0.0)) for k in
        ("gradient_cosine", "path_target_ratio", "path_gate_active", "path_weight", "commutator_loss")}}
        for i, item in enumerate(history)]
    pretrain = torch.load(resolve_in_project("artifacts/checkpoints/allen_cahn_formal_a_blrpng_seed42.pt", must_exist=True), map_location="cpu", weights_only=False)
    pretrain_seconds = float(pretrain.get("elapsed_seconds", 0.0))
    exploratory = {}
    exploratory_path = resolve_in_project("artifacts/reports/v2_seed42_query256_h20_defect_weight_sweep.json")
    if exploratory_path.exists():
        exploratory_records = _load(str(exploratory_path))["records"]
        combined = matrix["records"] + exploratory_records
        for method_name in sorted({row["method"] for row in exploratory_records}):
            exploratory[method_name] = paired_hierarchical_bootstrap(
                combined, method=method_name, baseline=args.method,
                replicates=args.replicates, seed=20260813,
            )
        exploratory["protocol_note"] = (
            "Post-HR exploratory defect-weight scan; excluded from confirmatory gate."
        )
    horizon50_path = resolve_in_project("artifacts/reports/v2_seed42_query256_h50.json")
    horizon50_means = {}
    if horizon50_path.exists():
        horizon50_records = _load(str(horizon50_path))["records"]
        for method_name in sorted({row["method"] for row in horizon50_records}):
            values = [row["relative_l2"] for row in horizon50_records if row["method"] == method_name]
            horizon50_means[method_name] = float(np.mean(values))
    summary = {
        "schema": "blrpng_v2_calibration_summary_v1", "frozen": True,
        "primary_endpoint": "256x256 20-step per-rollout relative L2",
        "method": args.method, "oracle_ratio": ratio, "scratch_improvement": scratch,
        "equivalence_tests": equivalence,
        "speedup": {"point": speedup_point, "ci95": speedup_ci, "per_task": speedups},
        "costs": costs, "source_pretraining_seconds": pretrain_seconds, "gate": gate,
        "horizon50_means": horizon50_means, "exploratory": exploratory,
        "methods": method_rows,
        "oracle_curves": {f"task{task}": curve["errors"] for task, curve in curves["tasks"].items()},
        "resolution_curves": resolution_curves,
        "paired_heatmap": {"baseline": heatmap_baseline, "methods": heatmap_methods,
                           "tasks": [str(i) for i in range(8)], "values": heatmap},
        "gradient_history": gradient_history,
    }
    with workspace_audit("summarize_calibration"):
        output = write_json(args.output, summary)
        gate_output = write_json("artifacts/manifests/allen_cahn_v2_seed42_gate.json", {
            "schema": "blrpng_v2_seed42_gate_v1", "passed": bool(gate["passed"]),
            "primary_summary": str(output.relative_to(PROJECT_ROOT)),
            "primary_method": args.method, "gate": gate,
        })
    print(json.dumps({"output": str(output), "gate_output": str(gate_output), "gate": gate,
                      "oracle_ratio": ratio, "speedup": summary["speedup"]}, indent=2))


if __name__ == "__main__":
    main()
