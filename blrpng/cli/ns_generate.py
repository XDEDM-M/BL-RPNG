"""Generate gated SfePy Navier--Stokes v4 rollout shards."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity, require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.navier_stokes_protocol import (
    build_navier_stokes_protocol,
    validate_navier_stokes_protocol,
)
from blrpng.data.navier_stokes_sfepy import run_sfepy_navier_stokes_case
from blrpng.data.shards import write_index
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file, write_json


STAGES = (
    "development_train",
    "development_validation",
    "source_train",
    "source_validation",
    "target_support",
    "target_validation",
    "query128",
    "query256",
    "frozen512",
    "oracle256",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navier_stokes_sfepy_seed42.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _cases(protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage in {
        "development_train",
        "development_validation",
        "source_train",
        "source_validation",
    }:
        return list(protocol[stage])
    targets = protocol["targets"]
    if stage == "target_support":
        return [case for task in targets.values() for case in task["support"]]
    if stage == "target_validation":
        return [case for task in targets.values() for case in task["validation"]]
    if stage in {"query128", "query256"}:
        return [case for task in targets.values() for case in task["query"]]
    if stage == "frozen512":
        selected = {int(value) for value in protocol["frozen_task_ids"]}
        return [
            case
            for task_id, task in targets.items()
            if int(task_id) in selected
            for case in task["query"][:2]
        ]
    if stage == "oracle256":
        selected = {int(value) for value in protocol["oracle_task_ids"]}
        result = []
        for task_id, task in targets.items():
            if int(task_id) in selected:
                result.extend(task["support"])
                result.extend(task["oracle_extra"])
        return result
    raise ValueError(stage)


def _stage_geometry(config: dict[str, Any], stage: str) -> tuple[int, int, int, int]:
    data = config["data"]
    if stage.startswith("development"):
        resolution = int(data["output_resolutions"]["development"])
        steps = int(data["source_steps"])
        fidelity = 1
    elif stage.startswith("source"):
        resolution = int(data["output_resolutions"]["source"])
        steps = int(data["source_steps"])
        fidelity = 2
    elif stage in {"target_support", "target_validation"}:
        resolution = int(data["output_resolutions"]["support"])
        steps = int(data["support_steps"])
        fidelity = 0
    elif stage == "query128":
        resolution, steps, fidelity = 128, int(data["query_steps"]), 1
    elif stage in {"query256", "oracle256"}:
        resolution = 256
        steps = int(data["query_steps"] if stage == "query256" else data["support_steps"])
        fidelity = 2
    elif stage == "frozen512":
        resolution, steps, fidelity = 512, int(data["query_steps"]), 3
    else:
        raise ValueError(stage)
    mesh = int(data["mesh_resolutions"][resolution])
    return resolution, mesh, steps, fidelity


def _estimate_bytes(cases: int, resolution: int, steps: int) -> int:
    # trajectory + Newton target + endpoint + four condition fields and metadata,
    # with a conservative 25% serialization/compression safety margin.
    float_planes = (steps + 1) * 3 + steps * 3 * 2 + 3 + 3 + 4
    boolean_planes = 3 + steps
    return int(1.25 * cases * resolution * resolution * (4 * float_planes + boolean_planes))


def _query_stage(stage: str) -> bool:
    return stage in {"query128", "query256", "frozen512"}


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    gate = require_prerequisite_gate(
        "artifacts/manifests/navier_stokes_sfepy_solver_gate.json"
    )
    if _query_stage(args.stage):
        require_prerequisite_gate(
            "artifacts/manifests/navier_stokes_sfepy_method_freeze.json"
        )
    protocol = build_navier_stokes_protocol(config)
    validation = validate_navier_stokes_protocol(protocol)
    protocol_path = write_json(
        "artifacts/manifests/navier_stokes_sfepy_protocol.json", protocol
    )
    cases = _cases(protocol, args.stage)
    resolution, mesh, steps, fidelity = _stage_geometry(config, args.stage)
    estimate = _estimate_bytes(len(cases), resolution, steps)
    capacity = require_capacity(
        estimate, maximum_gb=float(config["resources"]["max_total_artifacts_gib"])
    )
    preflight = {
        "stage": args.stage,
        "cases": len(cases),
        "resolution": resolution,
        "mesh_resolution": mesh,
        "steps": steps,
        "estimated_new_bytes": estimate,
        "capacity": capacity,
        "protocol_validation": validation,
        "solver_gate_sha256": sha256_file(
            "artifacts/manifests/navier_stokes_sfepy_solver_gate.json"
        ),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    requested = int(args.workers or config["resources"]["max_generation_workers"])
    if resolution >= 512:
        worker_count = 1
    elif resolution >= 256:
        worker_count = min(requested, int(config["resources"]["max_256_workers"]))
    else:
        worker_count = min(requested, int(config["resources"]["max_generation_workers"]))
    if worker_count < 1:
        raise ValueError("workers must be positive")
    output_dir = ensure_artifact_path(
        f"artifacts/data/navier_stokes_sfepy/{args.stage}_{resolution}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dt = float(gate["selected_dt"])

    def generate(case: dict[str, Any]) -> dict[str, Any]:
        return run_sfepy_navier_stokes_case(
            case,
            output_shard=output_dir / f"rollout_{int(case['case_id']):08d}.pt",
            output_resolution=resolution,
            mesh_resolution=mesh,
            steps=steps,
            dt=selected_dt,
            config=config,
            fidelity=fidelity,
            linear_solver=("gmres_ilu" if resolution >= 512 else "direct"),
        )

    with workspace_audit(f"ns_generate_{args.stage}") as audit:
        if worker_count == 1:
            rows = [generate(case) for case in cases]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                rows = list(executor.map(generate, cases))
        rows.sort(key=lambda row: (int(row["target_task_id"]), int(row["initial_condition_id"])))
        index_path = write_index(
            f"artifacts/data/navier_stokes_sfepy/{args.stage}_{resolution}.index.json",
            rows,
            metadata={
                "equation": "navier_stokes",
                "discretization": "taylor_hood_p2_p1",
                "lifting": "discrete_stokes",
                "dt": selected_dt,
                "resolution": resolution,
                "mesh_resolution": mesh,
                "steps": steps,
                "condition_names": list(config["data"]["condition_names"]),
                "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
                "protocol_sha256": protocol["sha256"],
                "query_locked_before_generation": _query_stage(args.stage),
            },
        )
        audit.update(
            {
                **preflight,
                "workers": worker_count,
                "index": str(index_path.relative_to(PROJECT_ROOT)),
                "index_sha256": sha256_file(index_path),
            }
        )
    print(
        json.dumps(
            {
                "index": str(index_path),
                "cases": len(rows),
                "resolution": resolution,
                "workers": worker_count,
                "all_converged": all(bool(row["solver_converged"]) for row in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
