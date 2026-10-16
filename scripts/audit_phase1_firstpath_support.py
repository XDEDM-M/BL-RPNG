from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blrpng.data.shards import read_index
from blrpng.paths import PROJECT_ROOT, sha256_file, write_json


OLD_INDEX = Path(
    "artifacts/data/phase1_development/hrdev_20261012_t20_ss271829/"
    "support_s271829_64.index.json"
)
NEW_INDEX = Path(
    "artifacts/data/phase1_development/hrdev_20261013_firstpath_ss271829/"
    "support_s271829_64.index.json"
)
AUDIT_PATH = Path(
    "artifacts/data/phase1_development/hrdev_20261013_firstpath_ss271829/"
    "support_s271829_64.index.generation_audit.json"
)
WORKSPACE_AUDIT_LATEST = Path(
    "artifacts/reports/phase1_dev_generate_support_workspace_audit.json"
)
WORKSPACE_AUDIT_DIR = Path("artifacts/reports/workspace_audits")
TASK_MANIFEST = Path("artifacts/manifests/phase1_development/hrdev_20261012/tasks.json")
SFEPY_ROOT = Path("/gemini/code/sfepy")
SFEPY_NLS = SFEPY_ROOT / "sfepy/solvers/nls.py"
SFEPY_PYTHON = Path("/opt/miniconda3/envs/sfepy_env/bin/python")
SFEPY_WORKER = Path("blrpng/solvers/sfepy_worker.py")
EPSILON = 1e-8
COMMON_TENSORS = (
    "trajectory",
    "boundary_values",
    "boundary_lift",
    "constraint_mask",
    "newton_correction",
    "residual_progress",
    "valid_correction_mask",
    "selected_newton_state_index",
    "selected_newton_state_index_p075",
)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "mean": None, "q10": None, "median": None, "q90": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "q10": finite[int(0.10 * (len(finite) - 1))],
        "median": statistics.median(finite),
        "q90": finite[int(0.90 * (len(finite) - 1))],
        "max": finite[-1],
    }


def _key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["target_task_id"]),
        int(row["support_rank"]),
        int(row["initial_condition_id"]),
    )


def _command_options(command: list[str]) -> dict[str, str]:
    if len(command) < 3 or command[1:3] != ["-m", "blrpng.solvers.sfepy_worker"]:
        raise ValueError("Workspace audit does not contain the expected SfePy worker command.")
    options = command[3:]
    if len(options) % 2:
        raise ValueError("SfePy worker command contains a flag without a value.")
    pairs = dict(zip(options[::2], options[1::2]))
    if len(pairs) != len(options) // 2 or any(not key.startswith("--") for key in pairs):
        raise ValueError("Malformed or duplicate SfePy worker command options.")
    return pairs


def _verified_generation_provenance(generation: dict, manifest_sha256: str) -> dict:
    latest_path = PROJECT_ROOT / WORKSPACE_AUDIT_LATEST
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if (
        latest.get("command") is None
        or latest.get("campaign_id") != generation.get("campaign")
        or latest.get("role") != "support"
        or latest.get("support_seed") != 271829
        or latest.get("manifest_sha256") != manifest_sha256
        or latest.get("case_count") != 96
        or latest.get("query_used") is not False
        or latest.get("outside_workspace_unchanged") is not True
    ):
        raise ValueError("Workspace audit does not match the sealed first-path support generation.")

    unique_candidates = []
    for candidate in (PROJECT_ROOT / WORKSPACE_AUDIT_DIR).glob("phase1_dev_generate_support_*_*.json"):
        record = json.loads(candidate.read_text(encoding="utf-8"))
        if (
            record.get("pid") == latest.get("pid")
            and record.get("started_unix") == latest.get("started_unix")
        ):
            unique_candidates.append(candidate)
    if len(unique_candidates) != 1:
        raise ValueError(f"Expected one immutable workspace-audit record; found {len(unique_candidates)}.")
    unique_path = unique_candidates[0]
    unique_record = json.loads(unique_path.read_text(encoding="utf-8"))
    if unique_record != latest:
        raise ValueError("Latest and immutable workspace-audit records differ.")

    options = _command_options(latest["command"])
    expected = {
        "--manifest": str(PROJECT_ROOT / "artifacts/manifests/phase1_development/hrdev_20261012_t20_ss271829/support_s271829.json"),
        "--split": "support",
        "--output-resolution": "64",
        "--mesh-resolution": "65",
        "--steps": "20",
        "--fidelity": "0",
        "--dt": "0.2",
        "--epsilon": "0.03",
        "--gamma": "5.0",
        "--progress-threshold": "0.25",
        "--case-start": "0",
        "--limit": "96",
    }
    if options != expected | {
        "--output-npz": options.get("--output-npz"),
        "--output-meta": options.get("--output-meta"),
    }:
        raise ValueError(f"Unexpected SfePy generation command arguments: {options}")

    worker_path = PROJECT_ROOT / SFEPY_WORKER
    worker_source = worker_path.read_text(encoding="utf-8")
    current_worker_sha256 = sha256_file(worker_path)
    generation_worker_sha256 = generation.get("worker_sha256")
    if not isinstance(generation_worker_sha256, str) or len(generation_worker_sha256) != 64:
        raise ValueError("Sealed generation audit is missing the generation-time worker hash.")
    defaults = {}
    for option, pattern, expected_value in (
        ("max_newton_iterations", r'parser\.add_argument\("--max-newton-iters", type=int, default=(\d+)\)', "20"),
        ("newton_absolute_tolerance", r'parser\.add_argument\("--newton-atol", type=float, default=([\deE.+-]+)\)', "1e-10"),
        ("newton_relative_tolerance", r'parser\.add_argument\("--newton-rtol", type=float, default=([\deE.+-]+)\)', "1e-8"),
    ):
        match = re.search(pattern, worker_source)
        if match is None or match.group(1) != expected_value:
            raise ValueError(f"SfePy worker default changed for {option}.")
        defaults[option] = float(match.group(1)) if "tolerance" in option else int(match.group(1))
    return {
        "latest_path": str(WORKSPACE_AUDIT_LATEST),
        "immutable_path": str(unique_path.relative_to(PROJECT_ROOT)),
        "immutable_sha256": sha256_file(unique_path),
        "started_unix": latest["started_unix"],
        "finished_unix": latest["finished_unix"],
        "pid": latest["pid"],
        "outside_workspace_unchanged": latest["outside_workspace_unchanged"],
        "command": latest["command"],
        "command_options": options,
        "worker_source": str(SFEPY_WORKER),
        "generation_worker_sha256": generation_worker_sha256,
        "current_worker_sha256": current_worker_sha256,
        "worker_hash_matches_generation": generation_worker_sha256 == current_worker_sha256,
        "worker_source_drift_note": (
            "Generation-time source hash is preserved from the sealed data audit. The current source hash is recorded separately; exact generation-time source bytes were not archived."
        ),
        "worker_default_solver_parameters": defaults,
        "worker_default_parameters_checked_against_sha256": current_worker_sha256,
    }


def audit(output: Path) -> dict:
    old = read_index(PROJECT_ROOT / OLD_INDEX, verify_hashes=True)
    new = read_index(PROJECT_ROOT / NEW_INDEX, verify_hashes=True)
    generation = json.loads((PROJECT_ROOT / AUDIT_PATH).read_text(encoding="utf-8"))
    new_index_file_sha256 = sha256_file(PROJECT_ROOT / NEW_INDEX)
    task_manifest = json.loads((PROJECT_ROOT / TASK_MANIFEST).read_text(encoding="utf-8"))
    if generation.get("index_sha256") != new_index_file_sha256 or generation.get("query_read") is not False:
        raise ValueError("The sealed generation audit does not match the new support index.")
    manifest_sha256 = new["metadata"]["manifest_sha256"]
    generation_provenance = _verified_generation_provenance(generation, manifest_sha256)
    if len(old["shards"]) != 96 or len(new["shards"]) != 96:
        raise ValueError("Expected 96 old and regenerated support rollouts.")
    old_rows = {_key(row): row for row in old["shards"]}
    new_rows = {_key(row): row for row in new["shards"]}
    if old_rows.keys() != new_rows.keys():
        raise ValueError("Old and regenerated support rollout identities differ.")
    families = {
        int(task["target_task_id"]): str(task["family"])
        for task in task_manifest["tasks"]
    }
    geometry: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "D": [], "norm_ratio": [], "cosine": [], "angle_degrees": [],
            "projection_residual": [],
            "near_zero_endpoint": 0, "transitions": 0, "rollouts": 0,
        }
    )
    compared_tensors = 0
    compared_rollouts = 0
    unconverged = []
    first_ordinals: dict[str, int] = defaultdict(int)
    first_invalid = 0
    first_zero_norm = 0
    boundary_trace_max = 0.0

    for key, new_row in sorted(new_rows.items()):
        old_row = old_rows[key]
        old_payload = torch.load(PROJECT_ROOT / old_row["path"], map_location="cpu", weights_only=False)
        new_payload = torch.load(PROJECT_ROOT / new_row["path"], map_location="cpu", weights_only=False)
        if not bool(new_payload["meta"].get("solver_converged", False)):
            unconverged.append(key)
        if new_payload["ids"] != old_payload["ids"]:
            raise ValueError(f"Regenerated rollout identity changed: {key}")
        for name in COMMON_TENSORS:
            if name not in old_payload or name not in new_payload:
                raise ValueError(f"Missing common tensor {name}: {key}")
            if not torch.equal(old_payload[name], new_payload[name]):
                raise ValueError(f"Physical/common field differs after regeneration: {key}/{name}")
            compared_tensors += 1
        if new_payload["trajectory"].shape[0] != 21:
            raise ValueError(f"Expected a 20-step physical trajectory: {key}")

        task_id = int(new_payload["ids"]["target_task_id"])
        family = families[task_id]
        bucket = geometry[family]
        bucket["rollouts"] = int(bucket["rollouts"]) + 1
        bucket["transitions"] = int(bucket["transitions"]) + 20
        first = new_payload["newton_correction_first"].to(torch.float64)
        endpoint = (
            new_payload["trajectory"][1:].to(torch.float64)
            - new_payload["trajectory"][:-1].to(torch.float64)
        )
        mask = new_payload["constraint_mask"].bool().expand_as(first)
        if mask.any():
            boundary_trace_max = max(
                boundary_trace_max, float(first[mask].abs().max().item())
            )
        flat_first = first.flatten(1)
        flat_endpoint = endpoint.flatten(1)
        first_norm = torch.linalg.vector_norm(flat_first, dim=1)
        endpoint_norm = torch.linalg.vector_norm(flat_endpoint, dim=1)
        valid = endpoint_norm > EPSILON
        first_valid = first_norm > EPSILON
        first_invalid += int((~new_payload["valid_correction_mask_first"].bool().reshape(-1)).sum())
        first_zero_norm += int((~first_valid).sum())
        for value in new_payload["selected_newton_state_index_first"].tolist():
            first_ordinals[str(int(value))] += 1
        if valid.any():
            a = flat_first[valid]
            b = flat_endpoint[valid]
            an = first_norm[valid]
            bn = endpoint_norm[valid]
            dot = (a * b).sum(dim=1)
            bucket["D"].extend(
                (torch.linalg.vector_norm(a - b, dim=1) / (bn + EPSILON)).tolist()
            )
            bucket["norm_ratio"].extend((an / (bn + EPSILON)).tolist())
            cos_valid = an > EPSILON
            cosine = (dot[cos_valid] / (an[cos_valid] * bn[cos_valid])).clamp(-1.0, 1.0)
            bucket["cosine"].extend(cosine.tolist())
            bucket["angle_degrees"].extend(torch.rad2deg(torch.acos(cosine)).tolist())
            alpha = dot / bn.square()
            projection = a - alpha[:, None] * b
            bucket["projection_residual"].extend(
                (torch.linalg.vector_norm(projection, dim=1) / (an + EPSILON)).tolist()
            )
        bucket["near_zero_endpoint"] = int(bucket["near_zero_endpoint"]) + int((~valid).sum())
        compared_rollouts += 1

    if unconverged:
        raise ValueError(f"Non-converged regenerated rollouts: {unconverged[:5]}")
    if first_invalid or first_zero_norm:
        raise ValueError(
            f"Unexpected invalid/near-zero first correction labels: {first_invalid}/{first_zero_norm}"
        )
    if not math.isclose(boundary_trace_max, 0.0, abs_tol=0.0):
        raise ValueError(f"First Newton correction violates exact boundary trace: {boundary_trace_max}")

    summaries = {}
    for family, values in sorted(geometry.items()):
        summaries[family] = {
            "rollouts": int(values["rollouts"]),
            "transitions": int(values["transitions"]),
            "near_zero_endpoint_transitions": int(values["near_zero_endpoint"]),
            "D_norm_first_minus_endpoint": _stats(values["D"]),
            "first_to_endpoint_norm_ratio": _stats(values["norm_ratio"]),
            "cosine_first_vs_endpoint": _stats(values["cosine"]),
            "angle_first_vs_endpoint_degrees": _stats(values["angle_degrees"]),
            "best_endpoint_projection_residual_over_first_norm": _stats(
                values["projection_residual"]
            ),
        }
    all_values = {
        key: [value for row in geometry.values() for value in row[key]]
        for key in ("D", "norm_ratio", "cosine", "angle_degrees", "projection_residual")
    }
    sfepy_version = subprocess.run(
        [str(SFEPY_PYTHON), "-c", "import sfepy; print(sfepy.__version__)"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    sfepy_commit = subprocess.run(
        ["git", "-C", str(SFEPY_ROOT), "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    result = {
        "schema": "phase1_firstpath_support_audit_v3",
        "query_read": False,
        "old_index": str(OLD_INDEX),
        "old_index_sha256": old["sha256"],
        "regenerated_index": str(NEW_INDEX),
        "regenerated_index_sha256": new["sha256"],
        "regenerated_index_file_sha256": new_index_file_sha256,
        "generation_audit": str(AUDIT_PATH),
        "generation_audit_sha256": sha256_file(PROJECT_ROOT / AUDIT_PATH),
        "generation_workspace_audit": generation_provenance,
        "task_manifest_sha256": task_manifest.get("sha256"),
        "comparison": {
            "rollouts": compared_rollouts,
            "common_tensors_exactly_identical": compared_tensors,
            "common_tensor_names": list(COMMON_TENSORS),
            "all_regenerated_solver_rollouts_converged": True,
            "newton_selection_replay_checks": generation["selection_replay_checks"],
            "first_raw_trace_ordinal_histogram": generation["first_index_histogram"],
            "first_correction_invalid_count": first_invalid,
            "first_correction_near_zero_count": first_zero_norm,
            "first_correction_boundary_trace_max_abs": boundary_trace_max,
        },
        "geometry_definition": {
            "q0": "natural previous physical state u_t",
            "endpoint_increment": "trajectory[t+1] - trajectory[t]",
            "first_increment": "stored first accepted cumulative Newton displacement from q0",
            "epsilon": EPSILON,
            "epsilon_scale": 1.0,
            "near_zero_endpoint_rule": "endpoint L2 norm <= epsilon; excluded from D/ratio/cosine/angle/projection summaries and counted separately",
            "angle": "acos(clamp(dot(first,endpoint)/(norm(first)*norm(endpoint)), -1, 1)) in degrees",
            "projection_residual": "norm(first - alpha*endpoint)/(norm(first)+epsilon), alpha=dot(first,endpoint)/norm(endpoint)^2",
            "quantiles": "floor-index empirical order statistic at floor(p*(n-1)); median uses statistics.median",
            "aggregation": "per transition, across six development families; transitions within a rollout are correlated and are descriptive, not independent replicates",
        },
        "all_families_geometry": {
            "D_norm_first_minus_endpoint": _stats(all_values["D"]),
            "first_to_endpoint_norm_ratio": _stats(all_values["norm_ratio"]),
            "cosine_first_vs_endpoint": _stats(all_values["cosine"]),
            "angle_first_vs_endpoint_degrees": _stats(all_values["angle_degrees"]),
            "best_endpoint_projection_residual_over_first_norm": _stats(
                all_values["projection_residual"]
            ),
        },
        "by_family": summaries,
        "generation_command_parameters_verified": {
            "manifest_sha256": manifest_sha256,
            "role": "support",
            "rollouts": int(generation_provenance["command_options"]["--limit"]),
            "steps": int(generation_provenance["command_options"]["--steps"]),
            "grid_resolution": int(generation_provenance["command_options"]["--output-resolution"]),
            "mesh_resolution": int(generation_provenance["command_options"]["--mesh-resolution"]),
            "fidelity": int(generation_provenance["command_options"]["--fidelity"]),
            "dt": float(generation_provenance["command_options"]["--dt"]),
            "epsilon": float(generation_provenance["command_options"]["--epsilon"]),
            "gamma": float(generation_provenance["command_options"]["--gamma"]),
            "progress_threshold": float(generation_provenance["command_options"]["--progress-threshold"]),
            "first_selection_threshold": float(generation["selection_metadata"]["first_selection_threshold"]),
            "case_start": int(generation_provenance["command_options"]["--case-start"]),
            "generation_workers": 1,
        },
        "worker_defaults_observed_current_not_generation_verified": {
            **generation_provenance["worker_default_solver_parameters"],
            "source_path": str(SFEPY_WORKER),
            "source_sha256": generation_provenance["worker_default_parameters_checked_against_sha256"],
            "limitation": "The generation-time worker hash differs and its source bytes were not archived; these defaults describe the current worker, not verified historical solver arguments.",
        },
        "solver_environment_observed_at_audit_time_not_generation_verified": {
            "newton_source_path": str(SFEPY_NLS),
            "newton_source_sha256": sha256_file(SFEPY_NLS),
            "sfepy_version": sfepy_version,
            "sfepy_git_commit": sfepy_commit,
            "limitation": "These values were measured during this audit; the generation workspace audit did not seal the SfePy environment source hash, version, or commit.",
        },
        "generation_code_hashes_from_sealed_data_audit": {
            "generation_worker_sha256": generation["worker_sha256"],
            "progress_selector_sha256": generation["selector_sha256"],
            "shard_writer_sha256": generation["shards_module_sha256"],
        },
        "source_code_sha256": sha256_file(Path(__file__)),
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the regenerated Phase-1 first-accepted Newton support labels.")
    parser.add_argument(
        "--output",
        required=True,
        help="Fresh artifact path for the audit report; existing paths are refused.",
    )
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Preserve prior evidence; select a fresh output path: {output}")
    result = audit(output)
    print(json.dumps({
        "output": str(output),
        "common_rollouts_exact": result["comparison"]["rollouts"],
        "all_converged": result["comparison"]["all_regenerated_solver_rollouts_converged"],
        "first_ordinal_histogram": result["comparison"]["first_raw_trace_ordinal_histogram"],
        "D_median": result["all_families_geometry"]["D_norm_first_minus_endpoint"]["median"],
        "cosine_median": result["all_families_geometry"]["cosine_first_vs_endpoint"]["median"],
        "projection_residual_median": result["all_families_geometry"]["best_endpoint_projection_residual_over_first_norm"]["median"],
        "angle_degrees_median": result["all_families_geometry"]["angle_first_vs_endpoint_degrees"]["median"],
    }, indent=2))


if __name__ == "__main__":
    main()
