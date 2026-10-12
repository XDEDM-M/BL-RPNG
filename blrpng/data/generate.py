from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from blrpng.data.manifest import CaseSpec, make_cases, read_manifest, write_manifest
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    git_status_outside_project,
    sha256_file,
    write_json,
)


SFEPY_PYTHON = Path("/opt/miniconda3/envs/sfepy_env/bin/python")


def _case_chunks(total: int, workers: int) -> list[tuple[int, int]]:
    if total <= 0:
        raise ValueError("Cannot split an empty case set.")
    if workers <= 0:
        raise ValueError("Generation worker count must be positive.")
    workers = min(total, workers)
    base, remainder = divmod(total, workers)
    chunks = []
    start = 0
    for index in range(workers):
        count = base + (1 if index < remainder else 0)
        chunks.append((start, count))
        start += count
    return chunks


@contextmanager
def workspace_audit(name: str) -> Iterator[dict[str, Any]]:
    """Fail if any path outside BL-RPNG changes during a command."""
    before = git_status_outside_project()
    started_ns = time.time_ns()
    record: dict[str, Any] = {
        "command": name,
        "started_unix": time.time(),
        "pid": os.getpid(),
        "outside_status_before": before,
    }
    try:
        yield record
    finally:
        after = git_status_outside_project()
        record["finished_unix"] = time.time()
        record["outside_status_after"] = after
        record["outside_workspace_unchanged"] = before == after
        # Preserve every parallel invocation as well as the backwards-
        # compatible latest record.  The unique record is essential when two
        # GPUs finish the same command at nearly the same time.
        write_json(
            f"artifacts/reports/workspace_audits/"
            f"{name}_{started_ns}_{os.getpid()}.json",
            record,
        )
        write_json(f"artifacts/reports/{name}_workspace_audit.json", record)
        if before != after:
            raise RuntimeError(
                "Workspace outside BL-RPNG changed during execution. "
                f"before={before!r}, after={after!r}"
            )


def _frequency_pair(pool: list[int | float], index: int) -> tuple[float, float]:
    if len(pool) < 2:
        raise ValueError("A frequency pool needs at least two values.")
    first = float(pool[index % len(pool)])
    second = float(pool[(index + 1) % len(pool)])
    return first, second


def _make_varied_cases(
    *,
    count: int,
    split: str,
    seed: int,
    frequency_pool: list[int | float],
    amplitude_range: tuple[float, float],
    case_offset: int,
    boundary_offset: int,
    initial_offset: int,
) -> list[CaseSpec]:
    # Independent deterministic streams make every case invariant to future
    # additions to other splits.
    cases: list[CaseSpec] = []
    for idx in range(count):
        cases.extend(
            make_cases(
                count=1,
                split=split,
                seed=seed + 104729 * idx,
                frequencies=_frequency_pair(frequency_pool, idx),
                amplitude_range=amplitude_range,
                case_offset=case_offset + idx,
                boundary_offset=boundary_offset + idx,
                initial_offset=initial_offset + idx,
            )
        )
    return cases


def _repeat_boundaries_with_new_initials(
    source: list[CaseSpec],
    *,
    split: str,
    initial_conditions_per_boundary: int,
    seed: int,
    case_offset: int,
    initial_offset: int,
) -> list[CaseSpec]:
    by_boundary: dict[int, CaseSpec] = {}
    for case in source:
        by_boundary.setdefault(case.boundary_id, case)
    rng = np.random.default_rng(seed)
    result = []
    for boundary_index, source_case in enumerate(by_boundary.values()):
        for local_ic in range(initial_conditions_per_boundary):
            index = len(result)
            result.append(
                CaseSpec(
                    case_id=case_offset + index,
                    pair_id=case_offset + index,
                    boundary_id=source_case.boundary_id,
                    initial_condition_id=initial_offset + index,
                    split=split,
                    boundary_coefficients=list(source_case.boundary_coefficients),
                    boundary_frequencies=list(source_case.boundary_frequencies),
                    boundary_amplitude=source_case.boundary_amplitude,
                    initial_coefficients=[float(value) for value in rng.normal(size=4)],
                    initial_phase=float(0.2 * local_ic + 0.07),
                )
            )
    return result


def create_manifest(config: dict[str, Any]) -> Path:
    data = config["data"]
    seed = int(config["experiment"]["seed"])
    manifest_path = data.get(
        "manifest_path",
        f"artifacts/manifests/{config['experiment']['name']}.json",
    )
    cases: list[CaseSpec] = []
    if "num_rollouts" in data:  # smoke
        train_count = int(data["num_rollouts"])
        validation_count = int(data["num_validation_rollouts"])
        frequencies = tuple(float(v) for v in data["boundary_frequencies"])
        amplitude_range = tuple(float(v) for v in data["boundary_amplitude"])
        cases += make_cases(
            count=train_count,
            split="train",
            seed=seed,
            frequencies=frequencies,
            amplitude_range=amplitude_range,
        )
        cases += make_cases(
            count=validation_count,
            split="validation",
            seed=seed + 1,
            frequencies=frequencies,
            amplitude_range=amplitude_range,
            case_offset=train_count,
            boundary_offset=train_count,
            initial_offset=train_count,
        )
    else:
        train_count = int(data["a_train_rollouts"])
        validation_count = int(data["a_validation_rollouts"])
        pool = list(data["a_frequency_pool"])
        amplitude_range = tuple(float(v) for v in data["a_amplitude_range"])
        cases += _make_varied_cases(
            count=train_count,
            split="a_train",
            seed=seed,
            frequency_pool=pool,
            amplitude_range=amplitude_range,
            case_offset=0,
            boundary_offset=0,
            initial_offset=0,
        )
        cases += _make_varied_cases(
            count=validation_count,
            split="a_validation",
            seed=seed + 1,
            frequency_pool=pool,
            amplitude_range=amplitude_range,
            case_offset=train_count,
            boundary_offset=train_count,
            initial_offset=train_count,
        )
        b_count = int(data.get("b_fixed_boundary_count", data.get("b_boundary_count", 4)))
        ic_per_boundary = int(data["b_initial_conditions_per_boundary"])
        b_cases = b_count * ic_per_boundary
        b_train = make_cases(
            count=b_cases,
            split="b_train",
            seed=seed + 2,
            frequencies=tuple(float(v) for v in data["b_frequencies"]),
            amplitude_range=(float(data["b_amplitude"]), float(data["b_amplitude"])),
            case_offset=train_count + validation_count,
            boundary_offset=train_count + validation_count,
            initial_offset=train_count + validation_count,
            fixed_boundaries=b_count,
            initial_conditions_per_boundary=ic_per_boundary,
        )
        cases += b_train
        next_case = train_count + validation_count + b_cases
        next_initial = next_case
        b_seen = _repeat_boundaries_with_new_initials(
            b_train,
            split="b_validation_seen",
            initial_conditions_per_boundary=2,
            seed=seed + 3,
            case_offset=next_case,
            initial_offset=next_initial,
        )
        cases += b_seen
        next_case += len(b_seen)
        next_initial += len(b_seen)
        unseen_count = int(data.get("b_unseen_validation_boundaries", b_count))
        b_unseen = make_cases(
            count=unseen_count * 2,
            split="b_validation_unseen",
            seed=seed + 4,
            frequencies=tuple(float(v) for v in data["b_frequencies"]),
            amplitude_range=(float(data["b_amplitude"]), float(data["b_amplitude"])),
            case_offset=next_case,
            boundary_offset=train_count + validation_count + b_count,
            initial_offset=next_initial,
            fixed_boundaries=unseen_count,
            initial_conditions_per_boundary=2,
        )
        cases += b_unseen
        next_case += len(b_unseen)
        next_initial += len(b_unseen)
        if "blind_unseen_boundaries" in data:
            blind_seen = _repeat_boundaries_with_new_initials(
                b_train,
                split="blind_seen",
                initial_conditions_per_boundary=2,
                seed=seed + 5,
                case_offset=next_case,
                initial_offset=next_initial,
            )
            cases += blind_seen
            next_case += len(blind_seen)
            next_initial += len(blind_seen)
            blind_specs = (
                (
                    "blind_unseen",
                    int(data["blind_unseen_boundaries"]),
                    tuple(float(v) for v in data["b_frequencies"]),
                    float(data["b_amplitude"]),
                    seed + 6,
                ),
                (
                    "blind_frequency_ood",
                    int(data["blind_ood_frequency_boundaries"]),
                    tuple(float(v) for v in data["ood_frequencies"]),
                    float(data["b_amplitude"]),
                    seed + 7,
                ),
                (
                    "blind_amplitude_ood",
                    int(data["blind_ood_amplitude_boundaries"]),
                    tuple(float(v) for v in data["b_frequencies"]),
                    float(data["ood_amplitude"]),
                    seed + 8,
                ),
            )
            next_boundary = max(case.boundary_id for case in cases) + 1
            for split, boundary_count, frequencies, amplitude, split_seed in blind_specs:
                added = make_cases(
                    count=boundary_count * 2,
                    split=split,
                    seed=split_seed,
                    frequencies=frequencies,
                    amplitude_range=(amplitude, amplitude),
                    case_offset=next_case,
                    boundary_offset=next_boundary,
                    initial_offset=next_initial,
                    fixed_boundaries=boundary_count,
                    initial_conditions_per_boundary=2,
                )
                cases += added
                next_case += len(added)
                next_initial += len(added)
                next_boundary += boundary_count
    return write_manifest(
        manifest_path,
        cases,
        metadata={
            "experiment": config["experiment"]["name"],
            "seed": seed,
            "config_sha256": config["_config_sha256"],
            "frozen": True,
        },
    )


def _run_worker(
    *,
    config: dict[str, Any],
    manifest_path: Path,
    split: str,
    output_path: str | Path,
    output_resolution: int,
    mesh_resolution: int,
    steps: int,
    fidelity: int,
    limit: int | None = None,
) -> Path:
    if not SFEPY_PYTHON.exists():
        raise FileNotFoundError(f"SfePy worker interpreter not found: {SFEPY_PYTHON}")
    output = ensure_artifact_path(output_path)
    stem = f"{output.stem}_{split}_{output_resolution}"
    data = config["data"]
    manifest = read_manifest(manifest_path)
    split_size = sum(case["split"] == split for case in manifest["cases"])
    requested_total = min(split_size, limit) if limit is not None else split_size
    requested_workers = int(os.environ.get("BLRPNG_GENERATION_WORKERS", "1"))
    chunks = _case_chunks(requested_total, requested_workers)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    processes = []
    temporary_files: list[tuple[Path, Path]] = []
    for chunk_index, (case_start, case_count) in enumerate(chunks):
        suffix = "" if len(chunks) == 1 else f".part{chunk_index:03d}"
        temporary_npz = ensure_artifact_path(f"artifacts/tmp/{stem}{suffix}.npz")
        temporary_meta = ensure_artifact_path(f"artifacts/tmp/{stem}{suffix}.json")
        temporary_files.append((temporary_npz, temporary_meta))
        command = [
            str(SFEPY_PYTHON),
            "-m",
            "blrpng.solvers.sfepy_worker",
            "--manifest",
            str(manifest_path),
            "--split",
            split,
            "--output-npz",
            str(temporary_npz),
            "--output-meta",
            str(temporary_meta),
            "--output-resolution",
            str(output_resolution),
            "--mesh-resolution",
            str(mesh_resolution),
            "--steps",
            str(steps),
            "--fidelity",
            str(fidelity),
            "--dt",
            str(data["dt"]),
            "--epsilon",
            str(data["eps"]),
            "--gamma",
            str(data["gamma"]),
            "--progress-threshold",
            str(data["residual_progress_threshold"]),
            "--case-start",
            str(case_start),
            "--limit",
            str(case_count),
        ]
        log_path = ensure_artifact_path(f"artifacts/tmp/{stem}{suffix}.log")
        log_stream = log_path.open("w", encoding="utf-8")
        processes.append(
            (
                command,
                subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    text=True,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                ),
                log_path,
                log_stream,
            )
        )
    failures = []
    for command, process, log_path, log_stream in processes:
        process.wait()
        log_stream.close()
        if process.returncode != 0:
            worker_log = log_path.read_text(encoding="utf-8", errors="replace")
            failures.append(
                "SfePy worker failed.\n"
                f"command={' '.join(command)}\nlog:\n{worker_log}"
            )
    if failures:
        raise RuntimeError("\n\n".join(failures))

    arrays_by_key: dict[str, list[np.ndarray]] = {}
    metas = []
    for temporary_npz, temporary_meta in temporary_files:
        with np.load(temporary_npz, allow_pickle=False) as archive:
            for key in archive.files:
                arrays_by_key.setdefault(key, []).append(archive[key])
        metas.append(json.loads(temporary_meta.read_text(encoding="utf-8")))
    tensors = {
        key: torch.from_numpy(
            values[0] if len(values) == 1 else np.concatenate(values, axis=0)
        )
        for key, values in arrays_by_key.items()
    }
    meta = dict(metas[0])
    meta.update(
        {
            "num_rollouts": sum(int(item["num_rollouts"]) for item in metas),
            "num_samples": sum(int(item["num_samples"]) for item in metas),
            "case_start": 0,
            "generation_workers": len(chunks),
            "convergence_failures": [
                failure
                for item in metas
                for failure in item.get("convergence_failures", [])
            ],
            "all_converged": all(bool(item.get("all_converged")) for item in metas),
        }
    )
    meta.update(
        {
            "schema_version": "blrpng_dataset_v3",
            "num_time_steps": int(steps),
            "config_sha256": config["_config_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "generator_python": str(SFEPY_PYTHON),
            "converter_python": sys.executable,
        }
    )
    tensors["meta"] = meta
    temporary_output = ensure_artifact_path(f"artifacts/tmp/{output.name}.partial")
    torch.save(tensors, temporary_output)
    temporary_output.replace(output)
    for temporary_npz, temporary_meta in temporary_files:
        temporary_npz.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)
    for _command, _process, log_path, _log_stream in processes:
        log_path.unlink(missing_ok=True)
    return output


def _formal_data_path(config: dict[str, Any], split: str, fidelity: str) -> str:
    return f"artifacts/data/{config['experiment']['name']}_{split}_{fidelity}.pt"


def _require_blind_gate(config: dict[str, Any]) -> None:
    gate_path = ensure_artifact_path("artifacts/manifests/formal_gate_passed.json")
    if not gate_path.exists():
        raise PermissionError(
            "Fine-grid blind generation is locked until the development gate writes "
            "artifacts/manifests/formal_gate_passed.json."
        )
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if not payload.get("passed") or payload.get("config_sha256") != config["_config_sha256"]:
        raise PermissionError("Blind-test gate is absent, failed, or belongs to another config.")


def generate_from_config(config: dict[str, Any], *, stage: str = "all") -> list[Path]:
    manifest_path = create_manifest(config)
    if stage == "manifest":
        return [manifest_path]
    data = config["data"]
    outputs: list[Path] = [manifest_path]
    if "num_rollouts" in data:
        outputs += [
            _run_worker(
                config=config,
                manifest_path=manifest_path,
                split="train",
                output_path=data["train_path"],
                output_resolution=int(data["coarse_grid"]),
                mesh_resolution=int(data["coarse_mesh"]),
                steps=int(data["num_time_steps"]),
                fidelity=0,
            ),
            _run_worker(
                config=config,
                manifest_path=manifest_path,
                split="validation",
                output_path=data["validation_path"],
                output_resolution=int(data["coarse_grid"]),
                mesh_resolution=int(data["coarse_mesh"]),
                steps=int(data["num_time_steps"]),
                fidelity=0,
            ),
            _run_worker(
                config=config,
                manifest_path=manifest_path,
                split="validation",
                output_path=data["test_path"],
                output_resolution=int(data["fine_grid"]),
                mesh_resolution=int(data["fine_mesh"]),
                steps=int(data["num_time_steps"]),
                fidelity=1,
            ),
        ]
        return outputs

    # Formal/development generation separates A/B from the locked blind set.
    a_grid = int(data.get("a_grid", data.get("fine_grid", 128)))
    a_mesh = int(data.get("a_mesh", data.get("fine_mesh", 129)))
    # Do not use ``dict.get(key, data[legacy_key])`` here: Python evaluates the
    # default expression eagerly, so a formal config that intentionally omits
    # the legacy smoke-only key would fail even when the new key is present.
    steps_key = (
        "num_train_time_steps"
        if "num_train_time_steps" in data
        else "num_time_steps"
    )
    steps = int(data[steps_key])
    if stage in {"all", "a"}:
        for split in ("a_train", "a_validation"):
            path = _formal_data_path(config, split, "fine")
            outputs.append(
                _run_worker(
                    config=config,
                    manifest_path=manifest_path,
                    split=split,
                    output_path=path,
                    output_resolution=a_grid,
                    mesh_resolution=a_mesh,
                    steps=steps,
                    fidelity=1,
                )
            )
    if stage in {"all", "b-coarse"}:
        coarse_grid = int(data.get("b_coarse_grid", data.get("coarse_grid", 64)))
        coarse_mesh = int(data.get("b_coarse_mesh", data.get("coarse_mesh", 65)))
        for split in ("b_train", "b_validation_seen", "b_validation_unseen"):
            outputs.append(
                _run_worker(
                    config=config,
                    manifest_path=manifest_path,
                    split=split,
                    output_path=_formal_data_path(config, split, "coarse"),
                    output_resolution=coarse_grid,
                    mesh_resolution=coarse_mesh,
                    steps=steps,
                    fidelity=0,
                )
            )
    if stage == "blind":
        _require_blind_gate(config)
        fine_grid = int(data["b_fine_grid"])
        fine_mesh = int(data["b_fine_mesh"])
        blind_steps = int(data["num_blind_time_steps"])
        for split in (
            "b_train",
            "blind_seen",
            "blind_unseen",
            "blind_frequency_ood",
            "blind_amplitude_ood",
        ):
            outputs.append(
                _run_worker(
                    config=config,
                    manifest_path=manifest_path,
                    split=split,
                    output_path=_formal_data_path(config, split, "fine"),
                    output_resolution=fine_grid,
                    mesh_resolution=fine_mesh,
                    steps=blind_steps,
                    fidelity=1,
                )
            )
    return outputs
