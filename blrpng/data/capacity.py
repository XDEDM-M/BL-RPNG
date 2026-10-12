from __future__ import annotations

import json
from pathlib import Path

from blrpng.analysis.stage_gates import storage_preflight
from blrpng.paths import ARTIFACT_ROOT, PROJECT_ROOT, resolve_in_project


def artifact_size_bytes() -> int:
    return sum(path.stat().st_size for path in ARTIFACT_ROOT.rglob("*") if path.is_file())


def require_capacity(estimated_new_bytes: int, *, maximum_gb: float = 300.0) -> dict:
    result = storage_preflight(
        current_bytes=artifact_size_bytes(), estimated_new_bytes=estimated_new_bytes,
        maximum_gb=maximum_gb,
    )
    if not result["passed"]:
        raise RuntimeError(f"Artifact capacity preflight failed: {result}")
    return result


def require_prerequisite_gate(path: str | Path) -> dict:
    resolved = resolve_in_project(path, must_exist=True)
    relative = str(resolved.relative_to(PROJECT_ROOT))
    for supersession_path in (ARTIFACT_ROOT / "manifests").glob("*supersession*.json"):
        supersession = json.loads(supersession_path.read_text(encoding="utf-8"))
        if supersession.get("supersedes_claim_interpretation_of") == relative:
            raise PermissionError(
                f"Prerequisite gate was superseded and cannot unlock a formal stage: {resolved}"
            )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not payload.get("passed", False):
        raise PermissionError(f"Prerequisite stage gate has not passed: {resolved}")
    if payload.get("oracle_status") == "current_budget_only":
        raise PermissionError("A current-budget oracle gate cannot unlock a formal stage.")
    if payload.get("oracle_status") == "resource_capped_unsaturated" and not (
        payload.get("authorized_stage_progression") is True
        and payload.get("saturated_hr_oracle") is False
    ):
        raise PermissionError(
            "An unsaturated resource-capped oracle requires explicit progression authority."
        )
    return payload
