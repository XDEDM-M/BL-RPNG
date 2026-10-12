from __future__ import annotations

import argparse
import json

from blrpng.cli.summarize_extended_oracle import build_report
from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the m256 resource-capped HR reference without claiming oracle saturation."
        )
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    args = parser.parse_args()
    report = build_report([0, 2, 7], [64, 128, 256], replicates=args.replicates)
    if not report["gates"]["all_representative_tasks_have_three_budgets"]:
        raise RuntimeError("All task0/2/7 m64/m128/m256 frozen query reports are required.")
    if report["gates"]["saturated_hr_oracle"]:
        raise RuntimeError(
            "This command is only for an explicitly unsaturated resource-capped reference."
        )
    with workspace_audit("freeze_resource_capped_oracle") as audit:
        report_path = write_json(
            "artifacts/reports/oracle_extension_query_audit_seed42.json", report
        )
        amendment = write_json(
            "artifacts/manifests/allen_cahn_oracle_resource_cap_amendment_20260827.json",
            {
                "schema": "blrpng_protocol_amendment_v1",
                "passed": True,
                "authorized_stage_progression": True,
                "authorization": "explicit_user_instruction_2026-08-27",
                "change": "Stop HR support expansion after m256 and continue to fresh blind reconfirmation.",
                "unchanged_scientific_fact": "HR oracle data saturation was not established.",
                "supersedes_query_open_condition_only_of": (
                    "artifacts/manifests/allen_cahn_reconfirmation_tasks_20260820.json"
                ),
                "does_not_modify_frozen_boundary_descriptors": True,
            },
        )
        gate = write_json(
            "artifacts/manifests/allen_cahn_oracle_resource_capped_gate.json",
            {
                "schema": "blrpng_oracle_resource_capped_gate_v1",
                "passed": True,
                "oracle_status": "resource_capped_unsaturated",
                "saturated_hr_oracle": False,
                "authorized_stage_progression": True,
                "maximum_hr_support_rollouts": 256,
                "representative_tasks": [0, 2, 7],
                "seed": 42,
                "audit_report": str(report_path.relative_to(PROJECT_ROOT)),
                "audit_sha256": sha256_file(report_path),
                "protocol_amendment": str(amendment.relative_to(PROJECT_ROOT)),
                "protocol_amendment_sha256": sha256_file(amendment),
                "claim_limitations": [
                    "This gate does not establish a saturated HR oracle.",
                    "Oracle parity and five-percent non-inferiority cannot be claimed from it.",
                    "The HR comparison must be named resource-capped m256 HR reference.",
                ],
            },
        )
        audit.update({
            "report": str(report_path),
            "amendment": str(amendment),
            "gate": str(gate),
        })
    print(json.dumps({
        "gate": str(gate),
        "oracle_status": "resource_capped_unsaturated",
        "stage_progression_authorized": True,
        "saturated_hr_oracle": False,
    }, indent=2))


if __name__ == "__main__":
    main()
