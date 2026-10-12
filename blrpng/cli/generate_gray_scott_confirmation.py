from __future__ import annotations

import argparse
import json

from blrpng.cli.generate_gray_scott_formal import _base, _run_jobs
from blrpng.config import load_config
from blrpng.data.capacity import require_capacity, require_prerequisite_gate
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import write_index
from blrpng.paths import ensure_artifact_path, resolve_in_project


ROOT = "artifacts/data/gray_scott_seed42_confirmation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the unopened Gray--Scott confirmation query.")
    parser.add_argument("--config", default="configs/gray_scott_confirmation.yaml")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    require_prerequisite_gate(
        "artifacts/manifests/gray_scott_seed42_converged_lr_summary.json"
    )
    require_prerequisite_gate(
        "artifacts/manifests/gray_scott_seed42_saturated_scratch_freeze.json"
    )
    if args.resolution == 256:
        require_prerequisite_gate(
            "artifacts/manifests/gray_scott_seed42_256_continuation_gate.json"
        )
    require_capacity((4 if args.resolution == 128 else 12) * 2**30)
    config = load_config(args.config)
    protocol_path = resolve_in_project(
        "artifacts/manifests/gray_scott_seed42_confirmation_query_protocol.json",
        must_exist=True,
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("frozen") or protocol.get("query_generated") is not False:
        raise PermissionError("Confirmation protocol is not unopened and frozen.")
    jobs = []
    order = 0
    for target in protocol["tasks"]:
        task = int(target["target_task_id"])
        directory = ensure_artifact_path(f"{ROOT}/task{task:02d}_query_{args.resolution}")
        directory.mkdir(parents=True, exist_ok=True)
        for sample in target["query"]:
            jobs.append({
                **_base(config, resolution=args.resolution, steps=int(protocol["steps"])),
                "seed": int(sample["case_id"]),
                "case_id": int(sample["case_id"]),
                "boundary_seed": int(target["boundary_seed"]),
                "initial_seed": int(sample["initial_seed"]),
                "boundary_regime": str(target["boundary_regime"]),
                "correlation": float(target["correlation"]),
                "amplitude": float(target["amplitude"]),
                "feed": float(target["feed"]),
                "kill": float(target["kill"]),
                "role": "target_confirmation_query",
                "fidelity": 1,
                "target_task_id": task,
                "boundary_id": int(target["boundary_id"]),
                "initial_condition_id": int(sample["initial_condition_id"]),
                "pair_id": int(sample["pair_id"]),
                "order": order,
                "output": str(directory / f"pair_{int(sample['pair_id']):07d}.pt"),
            })
            order += 1
    with workspace_audit(f"generate_gray_scott_confirmation_{args.resolution}") as audit:
        rows = _run_jobs(
            jobs, args.workers, f"gray_scott_confirmation_{args.resolution}"
        )
        outputs = []
        metadata = {
            "equation": "gray_scott",
            "condition_channels": 3,
            "split": "target_confirmation_query",
            "resolution": args.resolution,
            "steps": int(protocol["steps"]),
            "protocol_sha256": protocol["protocol_sha256"],
        }
        for target in protocol["tasks"]:
            task = int(target["target_task_id"])
            selected = [row for row in rows if int(row["target_task_id"]) == task]
            output = write_index(
                f"{ROOT}/task{task:02d}_query_{args.resolution}_all.index.json",
                selected,
                metadata={
                    **metadata,
                    "target_task_id": task,
                    "boundary_regime": target["boundary_regime"],
                },
            )
            outputs.append(str(output))
        outputs.append(str(write_index(
            f"{ROOT}/target_query_{args.resolution}_all.index.json",
            rows,
            metadata=metadata,
        )))
        audit.update({"outputs": outputs, "resolution": args.resolution})
    print(json.dumps({
        "resolution": args.resolution,
        "rollouts": len(rows),
        "indices": len(outputs),
        "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
