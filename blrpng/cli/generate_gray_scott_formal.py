from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

# Sparse direct solves are parallelized by rollout.  Prevent every worker from
# creating a nested BLAS team and exhausting memory/CPU determinism.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity, require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.gray_scott_protocol import validate_gray_scott_protocol
from blrpng.data.shards import validate_shard, write_index
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, resolve_in_project, sha256_file, write_json


DATA_ROOT = "artifacts/data/gray_scott_seed42"
PROTOCOL = "artifacts/manifests/gray_scott_seed42_protocol.json"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    from blrpng.solvers.gray_scott import generate_gray_scott_rollout

    torch.set_num_threads(1)
    output = ensure_artifact_path(job.pop("output"))
    order = int(job.pop("order"))
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        validate_shard(payload)
        expected_ids = {
            "case_id": int(job["case_id"]),
            "pair_id": int(job["pair_id"]),
            "boundary_id": int(job["boundary_id"]),
            "initial_condition_id": int(job["initial_condition_id"]),
            "rollout_id": int(job["case_id"]),
            "target_task_id": int(job["target_task_id"]),
        }
        if payload["ids"] != expected_ids or payload["meta"].get("boundary_seed") != int(job["boundary_seed"]):
            raise ValueError(f"Existing Gray--Scott shard conflicts with protocol: {output}")
    else:
        rollout = generate_gray_scott_rollout(**job)
        payload = rollout.payload()
        partial = ensure_artifact_path(output.with_suffix(".partial"))
        torch.save(payload, partial)
        partial.replace(output)
    return {
        "order": order,
        "path": _relative(output),
        "sha256": sha256_file(output),
        "steps": int(payload["trajectory"].shape[0] - 1),
        "target_task_id": int(payload["ids"]["target_task_id"]),
        "boundary_id": int(payload["ids"]["boundary_id"]),
        "initial_condition_id": int(payload["ids"]["initial_condition_id"]),
        "pair_id": int(payload["ids"]["pair_id"]),
        "role": payload["meta"]["role"],
        "resolution": int(payload["meta"]["resolution"]),
        "solver_converged": bool(payload["meta"]["solver_converged"]),
        "valid_fraction": float(payload["valid_correction_mask"].float().mean()),
        "final_residual_max": max(payload["meta"]["final_residual_norms"]),
    }


def _run_jobs(jobs: list[dict[str, Any]], workers: int, progress_name: str) -> list[dict[str, Any]]:
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(_worker, dict(job)) for job in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 10 == 0 or completed == len(futures):
                write_json(
                    f"artifacts/reports/progress/{progress_name}.json",
                    {"completed": completed, "total": len(futures), "workers": workers},
                )
    return sorted(rows, key=lambda row: row.pop("order"))


def _base(config: dict[str, Any], *, resolution: int, steps: int) -> dict[str, Any]:
    data = config["data"]
    return {
        "resolution": resolution,
        "steps": steps,
        "dt": float(data["dt"]),
        "diffusion_u": float(data["diffusion_u"]),
        "diffusion_v": float(data["diffusion_v"]),
        "progress_threshold": float(data["residual_progress_threshold"]),
    }


def _source_jobs(
    protocol: dict[str, Any], config: dict[str, Any], split: str,
    *, data_root: str = DATA_ROOT, resolution_override: int | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    resolution = int(
        protocol["design"]["source_resolution"]
        if resolution_override is None else resolution_override
    )
    steps = int(protocol["design"]["source_steps"])
    directory = ensure_artifact_path(f"{data_root}/{split}_{resolution}")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    specs = protocol[split] if limit is None else protocol[split][: int(limit)]
    for order, spec in enumerate(specs):
        jobs.append({
            **_base(config, resolution=resolution, steps=steps),
            "seed": int(spec["case_id"]), "case_id": int(spec["case_id"]),
            "boundary_seed": int(spec["boundary_seed"]),
            "initial_seed": int(spec["initial_seed"]),
            "boundary_regime": "source",
            "correlation": float(spec["correlation"]),
            "amplitude": float(spec["amplitude"]),
            "feed": float(spec["feed"]), "kill": float(spec["kill"]),
            "role": split, "fidelity": 1, "target_task_id": -1,
            "boundary_id": int(spec["boundary_id"]),
            "initial_condition_id": int(spec["initial_condition_id"]),
            "pair_id": int(spec["pair_id"]), "order": order,
            "output": str(directory / f"case_{int(spec['case_id']):06d}.pt"),
        })
    return jobs, resolution, steps


def _target_jobs(
    protocol: dict[str, Any], config: dict[str, Any], split: str, resolution: int,
) -> tuple[list[dict[str, Any]], int]:
    steps = (
        int(protocol["design"]["support_steps"])
        if split in {"support", "validation"}
        else int(protocol["design"]["query_steps"])
    )
    jobs = []
    order = 0
    for task in protocol["targets"]:
        task_id = int(task["target_task_id"])
        directory = ensure_artifact_path(f"{DATA_ROOT}/task{task_id:02d}_{split}_{resolution}")
        directory.mkdir(parents=True, exist_ok=True)
        for sample in task[split]:
            jobs.append({
                **_base(config, resolution=resolution, steps=steps),
                "seed": int(sample["case_id"]), "case_id": int(sample["case_id"]),
                "boundary_seed": int(task["boundary_seed"]),
                "initial_seed": int(sample["initial_seed"]),
                "boundary_regime": task["boundary_regime"],
                "correlation": float(task["correlation"]),
                "amplitude": float(task["amplitude"]),
                "feed": float(task["feed"]), "kill": float(task["kill"]),
                "role": f"target_{split}",
                "fidelity": 0 if resolution == 64 else 1,
                "target_task_id": task_id, "boundary_id": int(task["boundary_id"]),
                "initial_condition_id": int(sample["initial_condition_id"]),
                "pair_id": int(sample["pair_id"]), "order": order,
                "output": str(directory / f"pair_{int(sample['pair_id']):07d}.pt"),
            })
            order += 1
    return jobs, steps


def _metadata(protocol: dict[str, Any], split: str, resolution: int, steps: int) -> dict[str, Any]:
    return {
        "equation": "gray_scott", "condition_channels": 3,
        "split": split, "resolution": resolution, "steps": steps,
        "protocol_sha256": protocol["protocol_sha256"],
    }


def _write_target_indices(
    rows: list[dict[str, Any]], protocol: dict[str, Any], split: str,
    resolution: int, steps: int,
) -> list[str]:
    outputs = []
    for task in protocol["targets"]:
        task_id = int(task["target_task_id"])
        selected = [row for row in rows if int(row["target_task_id"]) == task_id]
        output = write_index(
            f"{DATA_ROOT}/task{task_id:02d}_{split}_{resolution}_all.index.json",
            selected,
            metadata={
                **_metadata(protocol, f"target_{split}", resolution, steps),
                "target_task_id": task_id, "boundary_regime": task["boundary_regime"],
            },
        )
        outputs.append(str(output))
        if split == "support":
            for budget in protocol["design"]["support_budgets"]:
                nested = write_index(
                    f"{DATA_ROOT}/task{task_id:02d}_support_{resolution}_m{int(budget)}.index.json",
                    selected[: int(budget)],
                    metadata={
                        **_metadata(protocol, "target_support", resolution, steps),
                        "target_task_id": task_id, "support_budget": int(budget),
                    },
                )
                outputs.append(str(nested))
    aggregate = write_index(
        f"{DATA_ROOT}/target_{split}_{resolution}_all.index.json", rows,
        metadata=_metadata(protocol, f"target_{split}", resolution, steps),
    )
    outputs.append(str(aggregate))
    return outputs


def _solver_gate(config: dict[str, Any]) -> dict[str, Any]:
    from blrpng.analysis.stage_gates import solver_gate
    from blrpng.solvers.gray_scott import generate_gray_scott_rollout

    records = []
    regimes = ("source", "high_frequency", "amplitude_ood", "correlation_shift", "parameter_ood")
    for resolution in (32, 64, 128):
        for index, regime in enumerate(regimes):
            correlation = -0.65 if regime == "correlation_shift" else 0.5
            amplitude = 0.22 if regime == "amplitude_ood" else 0.1
            feed, kill = (0.05, 0.067) if regime == "parameter_ood" else (0.04, 0.06)
            rollout = generate_gray_scott_rollout(
                resolution=resolution, steps=2, seed=42 + index,
                boundary_seed=100 + index, initial_seed=200 + index,
                boundary_regime=regime, correlation=correlation, amplitude=amplitude,
                feed=feed, kill=kill,
            )
            payload = rollout.payload()
            records.append({
                "resolution": resolution, "regime": regime,
                "converged": bool(payload["meta"]["solver_converged"]),
                "valid_fraction": float(payload["valid_correction_mask"].float().mean()),
                "boundary_error": float(payload["trajectory"][:, payload["constraint_mask"]].sub(
                    payload["boundary_values"][payload["constraint_mask"]]
                ).abs().max()),
                "residual_max": max(payload["meta"]["final_residual_norms"]),
                "iterations": payload["meta"]["nonlinear_iterations"],
            })
    gate = solver_gate(
        convergence_fraction=sum(row["converged"] for row in records) / len(records),
        valid_correction_fraction=sum(row["valid_fraction"] for row in records) / len(records),
        boundary_error=max(row["boundary_error"] for row in records),
    )
    return {"schema": "blrpng_gray_scott_solver_gate_v1", **gate, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen Gray--Scott schema-v4 data.")
    parser.add_argument("--config", default="configs/gray_scott_confirmation.yaml")
    parser.add_argument(
        "--section", choices=("solver_gate", "source", "target_lr", "query128", "query256"),
        required=True,
    )
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    require_prerequisite_gate(config["experiment"]["prerequisite_gate"])
    if args.section == "query128":
        require_prerequisite_gate(
            "artifacts/manifests/gray_scott_seed42_lr_repair_summary.json"
        )
    elif args.section == "query256":
        require_prerequisite_gate(
            "artifacts/manifests/gray_scott_seed42_query128_evaluation_gate.json"
        )
    workers = int(args.workers or config["resources"]["generation_workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    if args.section == "solver_gate":
        with workspace_audit("gray_scott_solver_gate") as audit:
            gate = _solver_gate(config)
            output = write_json("artifacts/manifests/gray_scott_seed42_solver_gate.json", gate)
            audit.update({"output": str(output), "passed": gate["passed"]})
        print(json.dumps({"output": str(output), **gate}, indent=2))
        return
    protocol = json.loads(resolve_in_project(PROTOCOL, must_exist=True).read_text())
    validate_gray_scott_protocol(protocol)
    estimates = {"source": 4, "target_lr": 1, "query128": 4, "query256": 12}
    require_capacity(int(estimates[args.section] * 2**30))
    outputs = []
    with workspace_audit(f"generate_gray_scott_formal_{args.section}") as audit:
        if args.section == "source":
            for split in ("source_train", "source_validation"):
                jobs, resolution, steps = _source_jobs(protocol, config, split)
                rows = _run_jobs(jobs, workers, f"gray_scott_{split}_{resolution}")
                output = write_index(
                    f"{DATA_ROOT}/{split}_{resolution}.index.json", rows,
                    metadata=_metadata(protocol, split, resolution, steps),
                )
                outputs.append(str(output))
        else:
            split = "support" if args.section == "target_lr" else "query"
            resolution = 64 if args.section == "target_lr" else int(args.section.removeprefix("query"))
            splits = ("support", "validation") if args.section == "target_lr" else (split,)
            for current_split in splits:
                jobs, steps = _target_jobs(protocol, config, current_split, resolution)
                rows = _run_jobs(
                    jobs, workers, f"gray_scott_target_{current_split}_{resolution}"
                )
                outputs.extend(_write_target_indices(
                    rows, protocol, current_split, resolution, steps
                ))
        audit.update({"section": args.section, "workers": workers, "outputs": outputs})
    print(json.dumps({
        "section": args.section, "workers": workers,
        "indices_written": len(outputs), "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
