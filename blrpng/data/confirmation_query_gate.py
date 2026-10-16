from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from blrpng.paths import PROJECT_ROOT, resolve_in_project, sha256_file, sha256_json


PREREG_PATH = "docs/prereg/CONFIRMATORY.md"
VERIFIER_PATH = "blrpng/data/confirmation_query_gate.py"
WORKER_PATH = "blrpng/solvers/sfepy_worker.py"
QUERY_CASE_MANIFEST_PATH = "artifacts/manifests/phase1_confirmation/{campaign}/validation_hr.json"
QUERY_IC_SEED_OFFSET = 30_000
QUERY_CASE_ID_BASE = 7_000_000
QUERY_SOLVER_PROTOCOL = {
    "output_resolution": 256,
    "mesh_resolution": 257,
    "steps": 20,
    "fidelity": 1,
    "dt": 0.2,
    "epsilon": 0.03,
    "gamma": 5.0,
    "progress_threshold": 0.25,
    "max_newton_iters": 20,
    "newton_atol": 1e-10,
    "newton_rtol": 1e-8,
    "case_start": 0,
}
QUERY_RUNTIME_CODE_PATHS = (
    "blrpng/paths.py",
    "blrpng/data/manifest.py",
    "blrpng/data/lifting.py",
    "blrpng/data/blind_boundaries.py",
    "blrpng/data/shards.py",
    "blrpng/solvers/progress.py",
    "blrpng/data/generate.py",
)
SFEPY_PYTHON = Path("/opt/miniconda3/envs/sfepy_env/bin/python")


def current_query_runtime_environment() -> dict[str, str]:
    import numpy as np
    import scipy
    import sfepy

    numerical_config = {
        "numpy": np.show_config(mode="dicts"),
        "scipy": scipy.show_config(mode="dicts"),
    }
    encoded = json.dumps(
        numerical_config, sort_keys=True, separators=(",", ":"), default=str
    )
    import hashlib

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sfepy": str(getattr(sfepy, "__version__", "unknown")),
        "blas_lapack_build_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def capture_query_runtime_environment() -> dict[str, str]:
    project_root = PROJECT_ROOT.resolve()
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(project_root)!r}); "
        "from blrpng.data.confirmation_query_gate import current_query_runtime_environment; "
        "print(json.dumps(current_query_runtime_environment(), sort_keys=True))"
    )
    result = subprocess.run(
        [str(SFEPY_PYTHON), "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _load_sealed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sha256") != sha256_json(
        {key: value for key, value in payload.items() if key != "sha256"}
    ):
        raise ValueError(f"Invalid self-seal: {path}")
    return payload


def verify_confirmation_query_gate(
    campaign: str,
    task_manifest: dict,
    *,
    query_rollouts: int,
    query_steps: int,
    runtime_parameters: dict | None = None,
    query_numpy_version: str | None = None,
    query_runtime_environment: dict[str, str] | None = None,
) -> None:
    digest_cache: dict[str, str] = {}

    def verify_file(rel: str, expected: str) -> None:
        path = resolve_in_project(rel, must_exist=True)
        key = str(path)
        actual = digest_cache.get(key)
        if actual is None:
            actual = sha256_file(path)
            digest_cache[key] = actual
        if actual != expected:
            raise ValueError(f"Confirmation artifact changed after query-gate freeze: {path}")

    gate_path = PROJECT_ROOT / "artifacts/manifests/phase1_confirmation" / campaign / "query_gate.json"
    if not gate_path.is_file():
        raise PermissionError("Confirmation HR query is locked until its full matrix gate is frozen.")
    gate = _load_sealed(gate_path)
    if gate.get("schema") != "blrpng_phase1_confirmation_query_gate_v1":
        raise ValueError("Unexpected confirmation query-gate schema.")
    if (
        gate.get("campaign_id") != campaign
        or gate.get("task_manifest_sha256") != task_manifest.get("sha256")
        or gate.get("task_manifest_path")
        != f"artifacts/manifests/phase1_confirmation/{campaign}/tasks.json"
        or gate.get("matrix_manifest_path")
        != f"artifacts/manifests/phase1_confirmation/{campaign}/matrix.json"
        or gate.get("checkpoint_manifest_path")
        != f"artifacts/manifests/phase1_confirmation/{campaign}/checkpoints.json"
        or gate.get("preregistration_path") != PREREG_PATH
    ):
        raise ValueError("Confirmation query gate is bound to a different campaign or task manifest.")
    expected_preunlock_paths = [
        f"artifacts/data/phase1_confirmation/{campaign}/validation_hr_256.index.json",
        f"artifacts/data/phase1_confirmation/{campaign}/validation_hr_256",
        f"artifacts/tmp/phase1_confirmation_{campaign}_validation_hr_256.npz",
        f"artifacts/tmp/phase1_confirmation_{campaign}_validation_hr_256.json",
    ]
    if (
        gate.get("pre_unlock_query_outputs_absent") is not True
        or gate.get("pre_unlock_query_output_paths_checked") != expected_preunlock_paths
    ):
        raise ValueError("Confirmation query gate lacks pre-unlock query-output absence evidence.")

    verify_file(gate["preregistration_path"], gate.get("preregistration_sha256", ""))
    matrix_path = resolve_in_project(gate["matrix_manifest_path"], must_exist=True)
    verify_file(gate["matrix_manifest_path"], gate.get("matrix_manifest_sha256", ""))
    matrix = _load_sealed(matrix_path)
    if (
        matrix.get("campaign_id") != campaign
        or matrix.get("task_manifest_sha256") != task_manifest.get("sha256")
        or int(matrix.get("query_rollouts", -1)) != query_rollouts
        or int(matrix.get("query_steps", -1)) != query_steps
        or int(matrix.get("query_ic_seed", -1))
        != int(task_manifest.get("seed", -1)) + QUERY_IC_SEED_OFFSET
        or int(matrix.get("query_case_id_base", -1)) != QUERY_CASE_ID_BASE
        or matrix.get("query_solver_protocol") != QUERY_SOLVER_PROTOCOL
        or not matrix.get("query_numpy_version")
        or (
            query_numpy_version is not None
            and matrix.get("query_numpy_version") != query_numpy_version
        )
    ):
        raise ValueError("Confirmation query dimensions differ from the frozen task matrix.")
    if runtime_parameters is not None and runtime_parameters != QUERY_SOLVER_PROTOCOL:
        raise ValueError("Confirmation query solver arguments differ from the frozen protocol.")
    if query_runtime_environment is not None and matrix.get(
        "query_runtime_environment"
    ) != query_runtime_environment:
        raise ValueError("Confirmation solver Python/numerical-library environment changed after freeze.")
    if matrix.get("query_runtime_environment") != gate.get("query_runtime_environment"):
        raise ValueError("Query gate runtime environment differs from the frozen matrix.")

    for row in gate.get("method_configs", {}).values():
        verify_file(row["config_path"], row.get("config_sha256", ""))
    verify_file(
        gate["checkpoint_manifest_path"], gate.get("checkpoint_manifest_sha256", "")
    )
    runtime_hashes = gate.get("runtime_code_hashes", {})
    if set(runtime_hashes) != set(QUERY_RUNTIME_CODE_PATHS):
        raise ValueError("Query gate is missing a frozen solver/runtime dependency hash.")
    for rel, digest in runtime_hashes.items():
        verify_file(rel, digest)
    for row in gate.get("checkpoints", []):
        for path_key, hash_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("task_result_path", "task_result_sha256"),
            ("support_index", "support_index_sha256"),
            ("validation_index", "validation_index_sha256"),
        ):
            verify_file(row[path_key], row.get(hash_key, ""))
        source_checkpoint = row.get("source_checkpoint")
        source_sha256 = row.get("source_checkpoint_sha256")
        if source_checkpoint is None:
            if source_sha256 is not None:
                raise ValueError("Query gate has a source-checkpoint hash without a path.")
        else:
            verify_file(source_checkpoint, source_sha256)

    for rel, digest in gate.get("analysis_files", {}).items():
        verify_file(rel, digest)

    frozen_code = (
        (gate.get("query_generator_path"), gate.get("query_generator_sha256")),
        (gate.get("gate_freezer_path"), gate.get("gate_freezer_sha256")),
        (gate.get("query_worker_path"), gate.get("query_worker_sha256")),
        (gate.get("query_gate_verifier_path"), gate.get("query_gate_verifier_sha256")),
    )
    if (
        gate.get("query_generator_path") != "scripts/prepare_phase1_development.py"
        or gate.get("gate_freezer_path") != "scripts/freeze_phase1_confirmation_query.py"
        or gate.get("query_worker_path") != WORKER_PATH
        or gate.get("query_gate_verifier_path") != VERIFIER_PATH
    ):
        raise ValueError("Query gate does not bind the canonical solver worker and gate verifier.")
    for rel, digest in frozen_code:
        if not rel or not digest:
            raise ValueError("Query gate is missing a frozen query-generation code hash.")
        verify_file(rel, digest)

    if not gate.get("query_role_unlocked"):
        raise PermissionError("Confirmation query gate is not unlocked.")
