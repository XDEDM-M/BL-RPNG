from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

from blrpng.data.schema import DatasetBundle, load_dataset
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
    sha256_json,
    write_json,
)


INDEX_SCHEMA = "blrpng_shard_index_v4"
SHARD_SCHEMA = "blrpng_rollout_v4"


def _grid(resolution: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, resolution, dtype=dtype)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def validate_shard(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "trajectory", "boundary_values", "boundary_lift",
        "constraint_mask", "newton_correction", "residual_progress",
        "valid_correction_mask", "ids", "meta",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Missing v4 rollout keys: {sorted(missing)}")
    if payload["schema"] != SHARD_SCHEMA:
        raise ValueError(f"Unsupported rollout shard schema: {payload['schema']}")
    trajectory = payload["trajectory"]
    correction = payload["newton_correction"]
    if trajectory.ndim != 4 or correction.ndim != 4:
        raise ValueError("trajectory and correction must be [T+1,C,H,W] and [T,C,H,W].")
    steps, fields, height, width = correction.shape
    if tuple(trajectory.shape) != (steps + 1, fields, height, width):
        raise ValueError("v4 trajectory/correction shape mismatch.")
    if height != width or height < 3:
        raise ValueError("v4 currently requires a square grid of size at least 3.")
    for key in ("boundary_values", "boundary_lift", "constraint_mask"):
        if tuple(payload[key].shape) != (fields, height, width):
            raise ValueError(f"{key} has an incompatible shape.")
    if "condition_fields" in payload:
        conditions = payload["condition_fields"]
        if conditions.ndim == 3:
            if tuple(conditions.shape[-2:]) != (height, width):
                raise ValueError("condition_fields must have shape [K,H,W].")
        elif conditions.ndim == 4:
            if conditions.shape[0] != steps or tuple(conditions.shape[-2:]) != (height, width):
                raise ValueError("time-dependent condition_fields must have shape [T,K,H,W].")
        else:
            raise ValueError("condition_fields must have shape [K,H,W] or [T,K,H,W].")
    if "endpoint_increment" in payload:
        endpoint = payload["endpoint_increment"]
        if tuple(endpoint.shape) != (steps, fields, height, width):
            raise ValueError("endpoint_increment must match newton_correction shape.")
    if "predictor_endpoint_increment" in payload:
        predictor_endpoint = payload["predictor_endpoint_increment"]
        if tuple(predictor_endpoint.shape) != (steps, fields, height, width):
            raise ValueError("predictor_endpoint_increment must match newton_correction shape.")
    if "accepted_newton_residuals" in payload:
        histories = payload["accepted_newton_residuals"]
        counts = payload.get("accepted_newton_counts")
        if histories.ndim != 2 or histories.shape[0] != steps:
            raise ValueError("accepted_newton_residuals must have shape [T,K].")
        if counts is None or tuple(counts.shape) != (steps,):
            raise ValueError("accepted_newton_counts must have shape [T].")
    if "path_information_mask" in payload:
        information_mask = payload["path_information_mask"]
        if tuple(information_mask.shape) != tuple(payload["valid_correction_mask"].shape):
            raise ValueError("path_information_mask must match valid_correction_mask shape.")
    mask = payload["constraint_mask"].bool()
    trace_error = max(
        float((trajectory[0][mask] - payload["boundary_values"][mask]).abs().max()) if mask.any() else 0.0,
        float((payload["boundary_lift"][mask] - payload["boundary_values"][mask]).abs().max()) if mask.any() else 0.0,
        float(correction[:, mask].abs().max()) if mask.any() else 0.0,
        float(payload["endpoint_increment"][:, mask].abs().max())
        if mask.any() and "endpoint_increment" in payload else 0.0,
        float(payload["predictor_endpoint_increment"][:, mask].abs().max())
        if mask.any() and "predictor_endpoint_increment" in payload else 0.0,
    )
    if not all(
        torch.isfinite(value).all()
        for value in payload.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ):
        raise ValueError("v4 shard contains non-finite values.")
    return {
        "steps": int(steps), "fields": int(fields), "resolution": int(height),
        "trace_error_max": trace_error,
    }


def read_index(path: str | Path, *, verify_hashes: bool = False) -> dict[str, Any]:
    index_path = resolve_in_project(path, must_exist=True)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    expected = payload.pop("sha256")
    actual = sha256_json(payload)
    payload["sha256"] = expected
    if actual != expected:
        raise ValueError(f"v4 index hash mismatch: {actual} != {expected}")
    if payload.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"Unsupported v4 index schema: {payload.get('schema')}")
    seen = set()
    for row in payload["shards"]:
        shard_path = resolve_in_project(row["path"], must_exist=True)
        if str(shard_path) in seen:
            raise ValueError(f"Duplicate shard in index: {shard_path}")
        seen.add(str(shard_path))
        if verify_hashes and sha256_file(shard_path) != row["sha256"]:
            raise ValueError(f"Shard hash mismatch: {shard_path}")
    return payload


class RolloutShardDataset(Dataset):
    """Transition-level lazy dataset backed by factorized rollout shards."""

    def __init__(self, index_path: str | Path, *, cache_rollouts: int = 2):
        self.index_path = resolve_in_project(index_path, must_exist=True)
        self.index = read_index(self.index_path)
        self.cache_rollouts = max(int(cache_rollouts), 1)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.samples: list[tuple[int, int]] = []
        for shard_index, row in enumerate(self.index["shards"]):
            self.samples.extend((shard_index, step) for step in range(int(row["steps"])))

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, shard_index: int) -> dict[str, Any]:
        row = self.index["shards"][shard_index]
        key = row["path"]
        if key not in self._cache:
            payload = torch.load(resolve_in_project(key, must_exist=True), map_location="cpu", weights_only=False)
            validate_shard(payload)
            self._cache[key] = payload
            while len(self._cache) > self.cache_rollouts:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, step = self.samples[index]
        payload = self._load(shard_index)
        trajectory = payload["trajectory"]
        fields, resolution = trajectory.shape[1], trajectory.shape[-1]
        ids = payload["ids"]
        condition_fields = payload.get(
            "condition_fields",
            trajectory.new_empty((0, resolution, resolution)),
        )
        if condition_fields.ndim == 4:
            condition_fields = condition_fields[step]
        sample = {
            "x": torch.cat(
                (
                    trajectory[step],
                    _grid(resolution, trajectory.dtype),
                    condition_fields,
                ),
                dim=0,
            ),
            "y": trajectory[step + 1],
            "trajectory": trajectory[step + 1],
            "boundary_values": payload["boundary_values"],
            "boundary_lift": payload["boundary_lift"],
            "boundary_mask": payload["constraint_mask"],
            "constraint_mask": payload["constraint_mask"],
            "newton_correction": payload["newton_correction"][step],
            "endpoint_increment": payload.get(
                "endpoint_increment", trajectory[1:] - trajectory[:-1]
            )[step],
            "predictor_endpoint_increment": payload.get(
                "predictor_endpoint_increment",
                payload.get("endpoint_increment", trajectory[1:] - trajectory[:-1]),
            )[step],
            "residual_progress": payload["residual_progress"][step],
            "valid_correction_mask": payload["valid_correction_mask"][step],
            "path_information_mask": payload.get(
                "path_information_mask", payload["valid_correction_mask"]
            )[step],
            "local_step_id": torch.tensor(step, dtype=torch.long),
        }
        for key in ("case_id", "pair_id", "boundary_id", "initial_condition_id", "rollout_id", "target_task_id"):
            sample[key] = torch.tensor(int(ids[key]), dtype=torch.long)
        sample["fidelity"] = torch.tensor(int(payload["meta"]["fidelity"]), dtype=torch.int8)
        sample["solver_converged"] = torch.tensor(bool(payload["meta"].get("solver_converged", True)))
        sample["num_newton_iters"] = torch.tensor(
            int(payload["meta"].get("nonlinear_iterations", [0] * (step + 1))[step]), dtype=torch.int16
        )
        return sample


def _rollout_groups(bundle: DatasetBundle) -> Iterator[tuple[int, list[int]]]:
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(bundle.tensors["rollout_id"].tolist()):
        groups.setdefault(int(value), []).append(index)
    for rollout_id, indices in sorted(groups.items()):
        yield rollout_id, sorted(indices, key=lambda i: int(bundle.tensors["local_step_id"][i]))


def convert_v3_to_v4(
    source_path: str | Path,
    index_path: str | Path,
    *,
    target_task_by_boundary: dict[int, int] | None = None,
    role: str = "support",
) -> Path:
    """Factorize a v3 file into atomic rollout shards without changing values."""
    bundle = load_dataset(source_path)
    index_output = ensure_artifact_path(index_path)
    shard_dir = ensure_artifact_path(index_output.with_suffix(""))
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = int(bundle.tensors["y"].shape[1])
    for rollout_id, indices in _rollout_groups(bundle):
        first = indices[0]
        states = [bundle.tensors["x"][first, :fields]]
        states.extend(bundle.tensors["y"][index] for index in indices)
        boundary_id = int(bundle.tensors["boundary_id"][first])
        task_id = int((target_task_by_boundary or {}).get(boundary_id, boundary_id))
        mask = bundle.tensors["boundary_mask"][first]
        if mask.shape[0] == 1 and fields > 1:
            mask = mask.expand(fields, -1, -1).clone()
        payload = {
            "schema": SHARD_SCHEMA,
            "trajectory": torch.stack(states).contiguous(),
            # clone() is required here: a contiguous one-sample view can still
            # retain the storage of the entire materialized dataset in torch.save.
            "boundary_values": bundle.tensors["boundary_values"][first].clone(),
            "boundary_lift": bundle.tensors["boundary_lift"][first].clone(),
            "constraint_mask": mask.clone(),
            "newton_correction": bundle.tensors["newton_correction"][indices].contiguous(),
            "residual_progress": bundle.tensors["residual_progress"][indices].contiguous(),
            "valid_correction_mask": bundle.tensors["valid_correction_mask"][indices].contiguous(),
            "ids": {
                key: int(bundle.tensors[key][first])
                for key in ("case_id", "pair_id", "boundary_id", "initial_condition_id", "rollout_id")
            } | {"target_task_id": task_id},
            "meta": {
                "equation": bundle.meta["equation"], "role": role,
                "resolution": int(bundle.tensors["y"].shape[-1]),
                "fidelity": int(bundle.tensors["fidelity"][first]),
                "field_names": bundle.meta.get("field_names", [f"field_{i}" for i in range(fields)]),
                "field_scales": bundle.meta.get("field_scales", [1.0] * fields),
                "solver_converged": bool(bundle.tensors["solver_converged"][indices].all()),
                "nonlinear_iterations": [int(v) for v in bundle.tensors["num_newton_iters"][indices]],
                "source_v3_sha256": sha256_file(resolve_in_project(source_path)),
            },
        }
        validate_shard(payload)
        shard_path = ensure_artifact_path(shard_dir / f"rollout_{rollout_id:06d}.pt")
        temporary = ensure_artifact_path(shard_path.with_suffix(".partial"))
        torch.save(payload, temporary)
        temporary.replace(shard_path)
        rows.append({
            "path": _relative(shard_path), "sha256": sha256_file(shard_path),
            "steps": len(indices), "target_task_id": task_id,
            "boundary_id": boundary_id, "initial_condition_id": int(bundle.tensors["initial_condition_id"][first]),
            "role": role, "resolution": int(bundle.tensors["y"].shape[-1]),
        })
    index = {
        "schema": INDEX_SCHEMA, "source": _relative(resolve_in_project(source_path)),
        "source_sha256": sha256_file(resolve_in_project(source_path)), "shards": rows,
    }
    index["sha256"] = sha256_json(index)
    return write_json(index_output, index)


def write_v4_rollouts_from_arrays(
    arrays: dict[str, torch.Tensor],
    meta: dict[str, Any],
    *,
    shard_dir: str | Path,
    target_task_by_boundary: dict[int, int],
    role: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Write factorized shards directly from one solver-worker chunk."""
    directory = ensure_artifact_path(shard_dir)
    directory.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(arrays["rollout_id"].tolist()):
        groups.setdefault(int(value), []).append(index)
    rows = []
    fields = int(arrays["y"].shape[1])
    for rollout_id, indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda index: int(arrays["local_step_id"][index]))
        first = indices[0]
        states = [arrays["x"][first, :fields]] + [arrays["y"][index] for index in indices]
        boundary_id = int(arrays["boundary_id"][first])
        mask = arrays["boundary_mask"][first]
        if mask.shape[0] == 1 and fields > 1:
            mask = mask.expand(fields, -1, -1).clone()
        payload = {
            "schema": SHARD_SCHEMA,
            "trajectory": torch.stack(states).contiguous(),
            "boundary_values": arrays["boundary_values"][first].clone(),
            "boundary_lift": arrays["boundary_lift"][first].clone(),
            "constraint_mask": mask.clone(),
            "newton_correction": arrays["newton_correction"][indices].contiguous(),
            "residual_progress": arrays["residual_progress"][indices].contiguous(),
            "valid_correction_mask": arrays["valid_correction_mask"][indices].contiguous(),
            "ids": {
                key: int(arrays[key][first])
                for key in ("case_id", "pair_id", "boundary_id", "initial_condition_id", "rollout_id")
            } | {"target_task_id": int(target_task_by_boundary[boundary_id])},
            "meta": {
                "equation": meta["equation"], "role": role,
                "resolution": int(arrays["y"].shape[-1]),
                "fidelity": int(arrays["fidelity"][first]),
                "field_names": meta.get("field_names", [f"field_{i}" for i in range(fields)]),
                "field_scales": meta.get("field_scales", [1.0] * fields),
                "solver_converged": bool(arrays["solver_converged"][indices].all()),
                "nonlinear_iterations": [int(v) for v in arrays["num_newton_iters"][indices]],
                "solver_timing": [float(v) for v in arrays.get("solver_seconds", torch.zeros(len(arrays["y"])))[indices]],
                "assembly_time": float(arrays.get("assembly_seconds", torch.zeros(len(arrays["y"])))[indices].sum()),
                "linear_iterations": [0] * len(indices),
                "energy_joules": None,
                "source_sha256": source_sha256,
            },
        }
        if "condition_fields" in arrays:
            payload["condition_fields"] = arrays["condition_fields"][first].clone()
        validate_shard(payload)
        shard_path = ensure_artifact_path(directory / f"rollout_{rollout_id:06d}.pt")
        temporary = ensure_artifact_path(shard_path.with_suffix(".partial"))
        torch.save(payload, temporary)
        temporary.replace(shard_path)
        rows.append({
            "path": _relative(shard_path), "sha256": sha256_file(shard_path),
            "steps": len(indices), "target_task_id": int(target_task_by_boundary[boundary_id]),
            "boundary_id": boundary_id, "initial_condition_id": int(arrays["initial_condition_id"][first]),
            "role": role, "resolution": int(arrays["y"].shape[-1]),
        })
    return rows


def write_index(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> Path:
    payload = {"schema": INDEX_SCHEMA, "metadata": metadata, "shards": rows}
    payload["sha256"] = sha256_json(payload)
    return write_json(path, payload)


def subset_index(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_task_id: int,
    max_rollouts: int | None = None,
) -> Path:
    source = read_index(source_path)
    rows = [
        row for row in source["shards"]
        if int(row["target_task_id"]) == int(target_task_id)
    ]
    rows.sort(key=lambda row: (int(row["initial_condition_id"]), row["path"]))
    if max_rollouts is not None:
        rows = rows[: int(max_rollouts)]
    if not rows:
        raise ValueError(f"No shards for target_task_id={target_task_id}")
    return write_index(
        output_path,
        rows,
        metadata={
            **source.get("metadata", {}),
            "parent_index": str(resolve_in_project(source_path).relative_to(PROJECT_ROOT)),
            "parent_sha256": source["sha256"],
            "target_task_id": int(target_task_id),
            "max_rollouts": max_rollouts,
        },
    )


def materialize_index(index_path: str | Path) -> DatasetBundle:
    dataset = RolloutShardDataset(index_path)
    rows = [dataset[index] for index in range(len(dataset))]
    keys = rows[0].keys()
    tensors = {key: torch.stack([row[key] for row in rows]) for key in keys}
    index = dataset.index
    meta = {
        "schema_version": "blrpng_dataset_v4", "num_samples": len(rows),
        "num_rollouts": len(index["shards"]), "num_time_steps": index["shards"][0]["steps"],
        "manifest_sha256": index["sha256"], "equation": dataset._load(0)["meta"]["equation"],
    }
    return DatasetBundle(tensors=tensors, meta=meta, validation={"samples": len(rows), "schema": 4})
