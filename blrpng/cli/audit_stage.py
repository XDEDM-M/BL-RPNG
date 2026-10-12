from __future__ import annotations

import json
from pathlib import Path

from blrpng.data.capacity import artifact_size_bytes
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, sha256_json, write_json


REQUIRED = (
    "artifacts/reports/current_state_20260820.json",
    "artifacts/reports/current_results_20260820.csv",
    "artifacts/reports/nlt_causal_seed42_summary_exploratory.json",
    "artifacts/reports/multiphysics_smoke_verification.json",
    "artifacts/manifests/cleanup_20260820_complete.json",
    "artifacts/manifests/allen_cahn_v2_gate_supersession_20260820.json",
    "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json",
    "artifacts/reports/current_state_20260831.json",
    "artifacts/manifests/allen_cahn_reconfirmation_query_256_data_gate.json",
    "artifacts/manifests/allen_cahn_reconfirmation_seed42_cost_query256_gate.json",
    "artifacts/manifests/allen_cahn_reconfirmation_seed42_stage_gate.json",
)


def _verify_embedded_hash(payload: dict) -> bool | None:
    if "sha256" not in payload:
        return None
    expected = payload["sha256"]
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    return expected == sha256_json(unsigned)


def main() -> None:
    with workspace_audit("audit_stage") as audit:
        files = []
        for relative in REQUIRED:
            path = resolve_in_project(relative, must_exist=True)
            row = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                row["json_reload_passed"] = True
                row["embedded_hash_passed"] = _verify_embedded_hash(payload)
            files.append(row)
        smoke = json.loads(resolve_in_project(REQUIRED[3], must_exist=True).read_text())
        blind = json.loads(resolve_in_project(REQUIRED[6], must_exist=True).read_text())
        cleanup = json.loads(resolve_in_project(REQUIRED[4], must_exist=True).read_text())
        stage_gate = json.loads(
            resolve_in_project(REQUIRED[-1], must_exist=True).read_text()
        )
        current_bytes = artifact_size_bytes()
        reservations_gib = {
            "allen_cahn": 45.0,
            "poisson": 3.0,
            "gray_scott": 25.0,
            "hyperelastic": 30.0,
            "navier_stokes": 55.0,
        }
        reserved_gib = sum(reservations_gib.values())
        payload = {
            "schema": "blrpng_stage_audit_v2",
            "as_of": "2026-08-31",
            "project_root": str(PROJECT_ROOT),
            "files": files,
            "storage": {
                "bytes": current_bytes,
                "gib": current_bytes / 2**30,
                "hard_limit_gib": 300.0,
                "passed": current_bytes <= 300 * 2**30,
                "stage_reservations_gib": reservations_gib,
                "reserved_total_gib": reserved_gib,
                "future_reservations_are_sequential_preflight_estimates": True,
                "next_stage": "poisson_negative_control",
                "next_stage_reservation_gib": reservations_gib["poisson"],
                "projected_after_next_stage_gib": (
                    current_bytes / 2**30 + reservations_gib["poisson"]
                ),
                "next_stage_fits": (
                    current_bytes / 2**30 + reservations_gib["poisson"] <= 300.0
                ),
            },
            "cleanup": {
                "released_gib": cleanup.get("released_gib"),
                "validation_passed": cleanup.get("validation_passed"),
            },
            "multiphysics_smoke_passed": bool(smoke.get("passed", False)),
            "blind_protocol": {
                "tasks": len(blind["tasks"]),
                "families": blind["families"],
                "query_opened": bool(blind["query_opened"]),
                "lock_respected": not bool(blind["query_opened"]),
            },
            "allen_cahn_stage_gate_passed": bool(stage_gate.get("passed", False)),
            "formal_stage_unlocked": bool(
                stage_gate.get("passed", False)
                and current_bytes / 2**30 + reservations_gib["poisson"] <= 300.0
            ),
            "next_stage": stage_gate.get("next_stage"),
            "oracle_claim_limit": (
                "The resource-capped m256 HR reference is unsaturated.  The user "
                "waived m512/m1024 continuation, so this unlock applies only to the "
                "Poisson negative control and is not an oracle-parity claim."
            ),
        }
        payload["implementation_audit_passed"] = bool(
            payload["storage"]["passed"]
            and payload["storage"]["next_stage_fits"]
            and payload["multiphysics_smoke_passed"]
            and payload["blind_protocol"]["lock_respected"]
            and payload["allen_cahn_stage_gate_passed"]
            and all(row.get("embedded_hash_passed") is not False for row in files)
        )
        output = write_json("artifacts/reports/stage_audit_20260831.json", payload)
        audit["output"] = str(output)
        audit["implementation_audit_passed"] = payload["implementation_audit_passed"]
    print(json.dumps({
        "output": str(output),
        "implementation_audit_passed": payload["implementation_audit_passed"],
        "formal_stage_unlocked": payload["formal_stage_unlocked"],
        "artifact_gib": payload["storage"]["gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
