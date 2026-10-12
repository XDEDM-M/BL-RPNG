from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import ARTIFACT_ROOT, ensure_artifact_path, resolve_in_project, sha256_file, write_json


ARCHIVE_ROOT = "artifacts/reports/archives/cleanup_20260820"


def _artifact_size() -> int:
    return sum(path.stat().st_size for path in ARTIFACT_ROOT.rglob("*") if path.is_file())


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(resolve_in_project(".")))


def _final_for_recovery(path: Path) -> Path:
    name = path.name
    if not name.endswith("_recovery.pt"):
        raise ValueError(f"Not a recovery checkpoint: {path}")
    return path.with_name(name.removesuffix("_recovery.pt") + ".pt")


def _validate_final_checkpoint(path: Path, *, reload: bool) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing non-empty final checkpoint for recovery: {path}")
    details: dict[str, Any] = {
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "reloaded": False,
    }
    if reload:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model_config", "representation", "best_validation_metric"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Final checkpoint {path} lacks {missing}")
        if not ("ema_weights" in payload or "adapter_state" in payload):
            raise ValueError(f"Final checkpoint {path} has no evaluation weights")
        details.update(
            {
                "reloaded": True,
                "best_validation_metric": float(payload["best_validation_metric"]),
                "training_seed": payload.get("training_seed"),
            }
        )
        del payload
    return details


def _archive_json_tree(source: Path, label: str, *, apply: bool) -> list[dict[str, Any]]:
    records = []
    for path in sorted(source.rglob("*.json")) if source.exists() else []:
        destination = ensure_artifact_path(
            Path(ARCHIVE_ROOT) / label / path.relative_to(source)
        )
        records.append(
            {
                "source": _relative(path),
                "archive": _relative(destination),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return records


def _safe_remove_tree(path: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(ARTIFACT_ROOT.resolve())
    if resolved in {ARTIFACT_ROOT.resolve(), resolve_in_project(".")}:
        raise ValueError(f"Refusing broad deletion: {resolved}")
    shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audited conservative BL-RPNG artifact cleanup.")
    parser.add_argument("--apply", action="store_true", help="Execute the recorded deletion plan.")
    parser.add_argument(
        "--reload-finals",
        action="store_true",
        help="Fully torch.load each final checkpoint before deleting its recovery.",
    )
    args = parser.parse_args()

    invalid = resolve_in_project(
        "artifacts/experiments/invalid_zero_transition_20260730"
    )
    superseded = resolve_in_project(
        "artifacts/experiments/superseded_local_newton_target_20260730"
    )
    tmp = resolve_in_project("artifacts/tmp")
    excluded_roots = {invalid.resolve(), superseded.resolve()}
    recoveries = []
    for path in ARTIFACT_ROOT.rglob("*_recovery.pt"):
        if any(root in path.resolve().parents for root in excluded_roots):
            continue
        final = _final_for_recovery(path)
        if not final.exists():
            continue
        recoveries.append((path, final))

    before = _artifact_size()
    entries = []
    final_cache: dict[Path, dict[str, Any]] = {}
    for recovery, final in sorted(recoveries):
        if final not in final_cache:
            final_cache[final] = _validate_final_checkpoint(
                final, reload=args.reload_finals
            )
        entries.append(
            {
                "path": _relative(recovery),
                "size_bytes": recovery.stat().st_size,
                "reason": "redundant_recovery_with_validated_final",
                "corresponding_final": final_cache[final],
            }
        )

    archived = {
        "invalid_zero_transition": _archive_json_tree(
            invalid, "invalid_zero_transition_20260730", apply=args.apply
        ),
        "superseded_local_newton": _archive_json_tree(
            superseded, "superseded_local_newton_target_20260730", apply=args.apply
        ),
    }
    tree_entries = []
    for path, reason in (
        (invalid, "invalid_zero_transition_tensors"),
        (superseded, "superseded_local_newton_tensors"),
        (tmp, "temporary_files"),
    ):
        if path.exists():
            tree_entries.append(
                {
                    "path": _relative(path),
                    "size_bytes": sum(
                        item.stat().st_size for item in path.rglob("*") if item.is_file()
                    ),
                    "reason": reason,
                }
            )

    manifest = {
        "schema": "blrpng_cleanup_manifest_v1",
        "created_unix": time.time(),
        "mode": "apply" if args.apply else "dry_run",
        "artifact_bytes_before": before,
        "recovery_entries": entries,
        "tree_entries": tree_entries,
        "archived_diagnostics": archived,
        "planned_release_bytes": sum(row["size_bytes"] for row in entries + tree_entries),
        "policy": "conservative_keep_all_final_checkpoints_data_reports_figures_manifests",
    }
    manifest_path = write_json(
        f"artifacts/manifests/cleanup_20260820_{manifest['mode']}.json", manifest
    )

    if args.apply:
        with workspace_audit("cleanup_artifacts"):
            for row in entries:
                resolve_in_project(row["path"], must_exist=True).unlink()
            for row in tree_entries:
                path = resolve_in_project(row["path"])
                if path.exists():
                    _safe_remove_tree(path)
            after = _artifact_size()
            completion = {
                **manifest,
                "artifact_bytes_after": after,
                "released_bytes": before - after,
                "completed": True,
                "manifest_sha256_before_completion": sha256_file(manifest_path),
            }
            write_json("artifacts/manifests/cleanup_20260820_complete.json", completion)
        print(json.dumps({"released_gib": (before - after) / 2**30, "after_gib": after / 2**30}, indent=2))
    else:
        print(json.dumps({
            "manifest": str(manifest_path),
            "recovery_count": len(entries),
            "planned_release_gib": manifest["planned_release_bytes"] / 2**30,
        }, indent=2))


if __name__ == "__main__":
    main()
