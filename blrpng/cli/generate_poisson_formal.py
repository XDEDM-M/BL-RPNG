from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from blrpng.config import load_config
from blrpng.data.capacity import require_capacity, require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.poisson_protocol import validate_poisson_protocol
from blrpng.data.shards import validate_shard, write_index
from blrpng.paths import (
    PROJECT_ROOT,
    ensure_artifact_path,
    resolve_in_project,
    sha256_file,
)
from blrpng.solvers.poisson import generate_poisson_rollout


PROTOCOL_PATH = "artifacts/manifests/poisson_negative_control_seed42_protocol.json"
DATA_ROOT = "artifacts/data/poisson_negative_control"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _save_rollout(path: Path, rollout: Any) -> dict[str, Any]:
    output = ensure_artifact_path(path)
    payload = rollout.payload()
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        validate_shard(existing)
        if existing["ids"] != payload["ids"] or existing["meta"].get("boundary_seed") != payload["meta"].get("boundary_seed"):
            raise ValueError(f"Existing Poisson shard conflicts with frozen protocol: {output}")
    else:
        temporary = ensure_artifact_path(output.with_suffix(".partial"))
        torch.save(payload, temporary)
        temporary.replace(output)
    return {
        "path": _relative(output),
        "sha256": sha256_file(output),
        "steps": 1,
        "target_task_id": int(payload["ids"]["target_task_id"]),
        "boundary_id": int(payload["ids"]["boundary_id"]),
        "initial_condition_id": int(payload["ids"]["initial_condition_id"]),
        "pair_id": int(payload["ids"]["pair_id"]),
        "role": payload["meta"]["role"],
        "resolution": int(payload["meta"]["resolution"]),
    }


def _metadata(protocol: dict[str, Any], *, split: str, resolution: int) -> dict[str, Any]:
    return {
        "equation": "poisson",
        "negative_control": True,
        "linear_newton_iterations": 1,
        "condition_channels": 1,
        "split": split,
        "resolution": resolution,
        "protocol_sha256": protocol["protocol_sha256"],
    }


def _generate_source(protocol: dict[str, Any], resolution: int) -> list[str]:
    outputs = []
    for split in ("source_train", "source_validation"):
        rows = []
        directory = ensure_artifact_path(f"{DATA_ROOT}/{split}_{resolution}")
        directory.mkdir(parents=True, exist_ok=True)
        for spec in protocol[split]:
            rollout = generate_poisson_rollout(
                resolution=resolution,
                seed=int(spec["case_id"]),
                case_id=int(spec["case_id"]),
                family=spec["family"],
                role=split,
                regime="source",
                boundary_seed=int(spec["boundary_seed"]),
                forcing_seed=int(spec["forcing_seed"]),
                target_task_id=-1,
                boundary_id=int(spec["boundary_id"]),
                forcing_id=int(spec["forcing_id"]),
                pair_id=int(spec["pair_id"]),
                fidelity=1,
            )
            rows.append(_save_rollout(directory / f"case_{int(spec['case_id']):06d}.pt", rollout))
        index = write_index(
            f"{DATA_ROOT}/{split}_{resolution}.index.json",
            rows,
            metadata=_metadata(protocol, split=split, resolution=resolution),
        )
        outputs.append(str(index))
    return outputs


def _target_specs(task: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [
        {
            **sample,
            "target_task_id": int(task["target_task_id"]),
            "boundary_id": int(task["boundary_id"]),
            "boundary_seed": int(task["boundary_seed"]),
            "family": task["family"],
            "regime": task["regime"],
        }
        for sample in task[split]
    ]


def _generate_target_split(
    protocol: dict[str, Any], *, split: str, resolution: int,
) -> list[str]:
    all_rows = []
    outputs = []
    for task in protocol["targets"]:
        task_id = int(task["target_task_id"])
        directory = ensure_artifact_path(
            f"{DATA_ROOT}/task{task_id:02d}_{split}_{resolution}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        task_rows = []
        for spec in _target_specs(task, split):
            rollout = generate_poisson_rollout(
                resolution=resolution,
                seed=int(spec["case_id"]),
                case_id=int(spec["case_id"]),
                family=spec["family"],
                role=f"target_{split}",
                regime="target",
                boundary_seed=int(spec["boundary_seed"]),
                forcing_seed=int(spec["forcing_seed"]),
                target_task_id=task_id,
                boundary_id=int(spec["boundary_id"]),
                forcing_id=int(spec["forcing_id"]),
                pair_id=int(spec["pair_id"]),
                fidelity=0 if resolution == 64 else 1,
            )
            row = _save_rollout(directory / f"pair_{int(spec['pair_id']):07d}.pt", rollout)
            task_rows.append(row)
            all_rows.append(row)
        task_index = write_index(
            f"{DATA_ROOT}/task{task_id:02d}_{split}_{resolution}_all.index.json",
            task_rows,
            metadata={
                **_metadata(protocol, split=f"target_{split}", resolution=resolution),
                "target_task_id": task_id,
                "boundary_family": task["family"],
                "boundary_seed": int(task["boundary_seed"]),
            },
        )
        outputs.append(str(task_index))
        if split == "support":
            for budget in protocol["design"]["support_budgets"]:
                nested = write_index(
                    f"{DATA_ROOT}/task{task_id:02d}_support_{resolution}_m{int(budget)}.index.json",
                    task_rows[: int(budget)],
                    metadata={
                        **_metadata(protocol, split="target_support", resolution=resolution),
                        "target_task_id": task_id,
                        "support_budget": int(budget),
                        "nested_order": [int(row["initial_condition_id"]) for row in task_rows],
                    },
                )
                outputs.append(str(nested))
    aggregate = write_index(
        f"{DATA_ROOT}/target_{split}_{resolution}_all.index.json",
        all_rows,
        metadata=_metadata(protocol, split=f"target_{split}", resolution=resolution),
    )
    outputs.append(str(aggregate))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen formal Poisson protocol.")
    parser.add_argument("--config", default="configs/poisson_negative_control.yaml")
    parser.add_argument(
        "--section", choices=("source", "target_lr", "query128", "query256", "all"),
        default="all",
    )
    parser.add_argument("--protocol", default=PROTOCOL_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    require_prerequisite_gate(config["experiment"]["prerequisite_gate"])
    protocol_path = resolve_in_project(args.protocol, must_exist=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation = validate_poisson_protocol(protocol)
    require_capacity(int(2.0 * 2**30))
    outputs: list[str] = []
    with workspace_audit(f"generate_poisson_formal_{args.section}") as audit:
        if args.section in {"source", "all"}:
            outputs.extend(_generate_source(protocol, int(config["data"]["pretrain_resolution"])))
        if args.section in {"target_lr", "all"}:
            resolution = int(config["data"]["support_resolution"])
            outputs.extend(_generate_target_split(protocol, split="support", resolution=resolution))
            outputs.extend(_generate_target_split(protocol, split="validation", resolution=resolution))
        if args.section in {"query128", "all"}:
            outputs.extend(_generate_target_split(protocol, split="query", resolution=128))
        if args.section in {"query256", "all"}:
            outputs.extend(_generate_target_split(protocol, split="query", resolution=256))
        audit.update({
            "section": args.section, "protocol_sha256": protocol["protocol_sha256"],
            "outputs": outputs, "protocol_validation": validation,
        })
    print(json.dumps({
        "section": args.section, "indices_written": len(outputs),
        "protocol_sha256": protocol["protocol_sha256"], "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
