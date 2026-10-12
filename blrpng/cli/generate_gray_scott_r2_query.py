from __future__ import annotations

import argparse
import json

from blrpng.cli.generate_gray_scott_formal import _base, _run_jobs
from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import write_index
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_json


DATA_ROOT = "artifacts/data/gray_scott_r2_seed42/blind_query"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen Gray--Scott R2 blind query.")
    parser.add_argument("--config", default="configs/gray_scott_r2_development.yaml")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Post-crash safety limit: query workers must be in [1,4].")
    require_capacity(4 * 2**30)
    protocol = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_blind_query_protocol.json", must_exist=True
    ).read_text(encoding="utf-8"))
    expected = protocol.pop("protocol_sha256")
    actual = sha256_json(protocol)
    protocol["protocol_sha256"] = expected
    if expected != actual or not protocol.get("frozen"):
        raise ValueError("R2 blind-query protocol hash mismatch or protocol is not frozen.")
    config = load_config(args.config)
    jobs = []
    order = 0
    steps = int(protocol["query_horizon"])
    for task in protocol["targets"]:
        task_id = int(task["target_task_id"])
        directory = ensure_artifact_path(
            f"{DATA_ROOT}/task{task_id:02d}_query_{args.resolution}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        for sample in task["query"]:
            jobs.append({
                **_base(config, resolution=args.resolution, steps=steps),
                "seed": int(sample["case_id"]),
                "case_id": int(sample["case_id"]),
                "boundary_seed": int(task["boundary_seed"]),
                "initial_seed": int(sample["initial_seed"]),
                "boundary_regime": task["boundary_regime"],
                "correlation": float(task["correlation"]),
                "amplitude": float(task["amplitude"]),
                "feed": float(task["feed"]),
                "kill": float(task["kill"]),
                "role": "r2_blind_query",
                "fidelity": 1,
                "target_task_id": task_id,
                "boundary_id": int(task["boundary_id"]),
                "initial_condition_id": int(sample["initial_condition_id"]),
                "pair_id": int(sample["pair_id"]),
                "order": order,
                "output": str(directory / f"pair_{int(sample['pair_id']):07d}.pt"),
            })
            order += 1
    with workspace_audit(f"generate_gray_scott_r2_query{args.resolution}") as audit:
        rows = _run_jobs(
            jobs, args.workers, f"gray_scott_r2_blind_query_{args.resolution}"
        )
        outputs = []
        for task in protocol["targets"]:
            task_id = int(task["target_task_id"])
            selected = [row for row in rows if int(row["target_task_id"]) == task_id]
            output = write_index(
                f"{DATA_ROOT}/task{task_id:02d}_query_{args.resolution}.index.json",
                selected,
                metadata={
                    "equation": "gray_scott",
                    "condition_channels": 3,
                    "split": "r2_blind_query",
                    "resolution": args.resolution,
                    "steps": steps,
                    "target_task_id": task_id,
                    "boundary_regime": task["boundary_regime"],
                    "protocol_sha256": expected,
                    "selection_used_query": False,
                },
            )
            outputs.append(str(output))
        audit.update({
            "resolution": args.resolution,
            "outputs": outputs,
            "workers": args.workers,
            "query_opened_for_generation": True,
            "query_used_for_selection": False,
        })
    print(json.dumps({
        "resolution": args.resolution,
        "rollouts": len(rows),
        "outputs": outputs,
        "query_used_for_selection": False,
    }, indent=2))


if __name__ == "__main__":
    main()
