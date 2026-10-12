from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


REFERENCE_TASKS = (0, 2, 7)


def _timing(index_path: str) -> dict[str, float]:
    index = read_index(index_path, verify_hashes=True)
    solve = assembly = 0.0
    for row in index["shards"]:
        shard = torch.load(
            resolve_in_project(row["path"], must_exist=True),
            map_location="cpu",
            weights_only=False,
        )
        solve += sum(float(value) for value in shard["meta"].get("solver_timing", []))
        assembly += float(shard["meta"].get("assembly_time", 0.0))
        del shard
    return {
        "rollouts": len(index["shards"]),
        "solver_seconds": solve,
        "assembly_seconds": assembly,
        "index_sha256": sha256_file(resolve_in_project(index_path)),
    }


def _bootstrap_speedup(
    low: np.ndarray,
    high: np.ndarray,
    *,
    replicates: int = 50_000,
) -> dict:
    rng = np.random.default_rng(20260930)
    low_samples = rng.choice(low, size=(replicates, len(low)), replace=True).mean(axis=1)
    high_samples = rng.choice(high, size=(replicates, len(high)), replace=True).mean(axis=1)
    samples = high_samples / np.maximum(low_samples, 1e-12)
    return {
        "point": float(high.mean() / low.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze target-adaptation cost against the HR specialist reference."
    )
    parser.add_argument("--resolution", type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    resolution = int(args.resolution)
    low_rows = []
    for task in range(48):
        path = resolve_in_project(
            "artifacts/reports/reconfirmation/"
            f"task{task:02d}_m4_s42_nlt_query{resolution}_h20.json",
            must_exist=True,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        support = payload["support_cost"]
        # evaluate_bundle measures both true-fine and interpolated-coarse
        # initial-state rollouts.  The deployment endpoint uses one of them,
        # hence half the measured evaluation wall time for Q=16.
        inference_q16 = float(payload["metrics"]["evaluation_seconds"]) / 2.0
        setup = (
            float(support["solver_seconds"])
            + float(support["assembly_seconds"])
            + float(payload["training_seconds"])
        )
        low_rows.append({
            "task": task,
            "setup_seconds": setup,
            "q16_inference_seconds": inference_q16,
            "target_cost_q16_seconds": setup + inference_q16,
            "report": str(path.relative_to(PROJECT_ROOT)),
            "report_sha256": sha256_file(path),
        })
    oracle_report_path = resolve_in_project(
        "artifacts/reports/v2_seed42_oracle_query256_h20.json", must_exist=True
    )
    oracle_report = json.loads(oracle_report_path.read_text(encoding="utf-8"))
    high_rows = []
    for task in REFERENCE_TASKS:
        timing = _timing(
            f"artifacts/data/tasks/task{task:02d}_oracle_train_256_m256.index.json"
        )
        checkpoint_path = resolve_in_project(
            "artifacts/checkpoints/v2/"
            f"v2_task{task:02d}_m256_s42_hr_oracle_path_conv_path_lr3em04_e100.pt",
            must_exist=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        inference_values = [
            float(row["inference_seconds_per_query"])
            for row in oracle_report["records"]
            if int(row["target_task_id"]) == task and row["method"] == "hr_oracle"
        ]
        inference_q16 = 16.0 * float(np.mean(inference_values))
        setup = (
            timing["solver_seconds"]
            + timing["assembly_seconds"]
            + float(checkpoint["elapsed_seconds"])
        )
        high_rows.append({
            "task": task,
            "support_budget": 256,
            "data_timing": timing,
            "training_seconds": float(checkpoint["elapsed_seconds"]),
            "q16_inference_seconds": inference_q16,
            "setup_seconds": setup,
            "target_cost_q16_seconds": setup + inference_q16,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        del checkpoint
    low_setup = np.asarray([row["setup_seconds"] for row in low_rows])
    high_setup = np.asarray([row["setup_seconds"] for row in high_rows])
    low_q16 = np.asarray([row["target_cost_q16_seconds"] for row in low_rows])
    high_q16 = np.asarray([row["target_cost_q16_seconds"] for row in high_rows])
    setup_speedup = _bootstrap_speedup(low_setup, high_setup)
    q16_speedup = _bootstrap_speedup(low_q16, high_q16)
    source_checkpoint = resolve_in_project(
        "artifacts/checkpoints/allen_cahn_formal_a_blrpng_seed42.pt", must_exist=True
    )
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    pretraining_seconds = float(source.get("elapsed_seconds", 0.0))
    training_only_break_even_tasks = pretraining_seconds / max(
        float(high_setup.mean() - low_setup.mean()), 1e-12
    )
    payload = {
        "schema": "blrpng_reconfirmation_target_cost_v1",
        "frozen": True,
        "seed": 42,
        "deployment_queries": 16,
        "deployment_query_resolution": resolution,
        "low_fidelity_method": "BL-RPNG-NLT m4, 64x64 support",
        "high_fidelity_reference": (
            "resource-capped m256 256x256 HR specialist on representative tasks 0/2/7"
        ),
        "low_target_rows": low_rows,
        "high_reference_rows": high_rows,
        "setup_speedup": setup_speedup,
        "q16_target_cost_speedup": q16_speedup,
        "source_pretraining_training_seconds": pretraining_seconds,
        "training_only_amortized_break_even_tasks": training_only_break_even_tasks,
        "pretraining_cost_limit": (
            "Break-even includes source-network training time only; legacy A data-generation "
            "timing remains a separate amortization audit and is not silently treated as zero."
        ),
        "ground_truth_cost_excluded": True,
        "claim_limit": (
            "The HR reference is resource-capped and unsaturated, so its cost is valid as an "
            "observed engineering reference but not as a saturated-oracle accuracy claim."
        ),
        "oracle_report": str(oracle_report_path.relative_to(PROJECT_ROOT)),
        "oracle_report_sha256": sha256_file(oracle_report_path),
    }
    report_name = (
        "allen_cahn_reconfirmation_seed42_target_cost.json"
        if resolution == 128
        else f"allen_cahn_reconfirmation_seed42_target_cost_query{resolution}.json"
    )
    gate_name = (
        "allen_cahn_reconfirmation_seed42_cost_gate.json"
        if resolution == 128
        else f"allen_cahn_reconfirmation_seed42_cost_query{resolution}_gate.json"
    )
    with workspace_audit(f"summarize_reconfirmation_cost_{resolution}") as audit:
        output = write_json(
            f"artifacts/reports/{report_name}",
            payload,
        )
        gate = write_json(
            f"artifacts/manifests/{gate_name}",
            {
                "schema": "blrpng_reconfirmation_cost_gate_v1",
                "passed": bool(setup_speedup["ci95"][0] > 10.0),
                "criterion": "marginal setup speedup 95% CI lower bound > 10x",
                "report": str(output.relative_to(PROJECT_ROOT)),
                "report_sha256": sha256_file(output),
                "speedup": setup_speedup,
            },
        )
        audit.update({"output": str(output), "gate": str(gate)})
    print(json.dumps({
        "output": str(output),
        "gate": str(gate),
        "mean_low_setup_seconds": float(low_setup.mean()),
        "mean_high_setup_seconds": float(high_setup.mean()),
        "setup_speedup": setup_speedup,
        "q16_target_cost_speedup": q16_speedup,
        "training_only_break_even_tasks": training_only_break_even_tasks,
    }, indent=2))


if __name__ == "__main__":
    main()
