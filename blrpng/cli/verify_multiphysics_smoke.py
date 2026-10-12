from __future__ import annotations

import json

import torch

from blrpng.analysis.stage_gates import solver_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import read_index, validate_shard
from blrpng.paths import resolve_in_project, write_json


INDICES = {
    "poisson": "artifacts/data/poisson_smoke.index.json",
    "gray_scott": "artifacts/data/gray_scott_v4_smoke.index.json",
    "hyperelastic": "artifacts/data/hyperelastic_smoke.index.json",
    "navier_stokes": "artifacts/data/navier_stokes_smoke.index.json",
}


def main() -> None:
    reports = {}
    for equation, index_path in INDICES.items():
        index = read_index(index_path, verify_hashes=True)
        validations = []
        metas = []
        for row in index["shards"]:
            payload = torch.load(
                resolve_in_project(row["path"], must_exist=True),
                map_location="cpu", weights_only=False,
            )
            validations.append(validate_shard(payload)); metas.append(payload["meta"])
        convergence = sum(bool(meta["solver_converged"]) for meta in metas) / len(metas)
        valid = sum(
            float(torch.load(resolve_in_project(row["path"]), map_location="cpu", weights_only=False)[
                "valid_correction_mask"
            ].float().mean())
            for row in index["shards"]
        ) / len(index["shards"])
        gate = solver_gate(
            convergence_fraction=convergence,
            valid_correction_fraction=valid,
            boundary_error=max(row["trace_error_max"] for row in validations),
            divergence_error=max(float(meta.get("divergence_max", 0.0)) for meta in metas)
            if equation == "navier_stokes" else None,
        )
        reports[equation] = {
            "index": index_path, "rollouts": len(index["shards"]),
            "validation": validations, "solver_gate": gate,
        }
    payload = {
        "schema": "blrpng_multiphysics_smoke_report_v1",
        "passed": all(report["solver_gate"]["passed"] for report in reports.values()),
        "formal_stage_unlocked": False,
        "reports": reports,
    }
    with workspace_audit("verify_multiphysics_smoke"):
        output = write_json("artifacts/reports/multiphysics_smoke_verification.json", payload)
    print(json.dumps({"output": str(output), "passed": payload["passed"]}, indent=2))


if __name__ == "__main__":
    main()
