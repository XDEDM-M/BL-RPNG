from __future__ import annotations

import argparse
import json

from blrpng.cli.generate_gray_scott_formal import _base, _run_jobs
from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.shards import write_index
from blrpng.paths import ensure_artifact_path, resolve_in_project, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate second fresh high-frequency query.")
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Post-crash worker safety limit is four.")
    require_capacity(2 * 2**30)
    protocol = json.loads(resolve_in_project(
        "artifacts/manifests/gray_scott_r2_highfreq_confirmation_protocol.json",
        must_exist=True,
    ).read_text(encoding="utf-8"))
    expected = protocol.pop("protocol_sha256")
    actual = sha256_json(protocol)
    protocol["protocol_sha256"] = expected
    if expected != actual:
        raise ValueError("High-frequency confirmation protocol hash mismatch.")
    config = load_config("configs/gray_scott_r2_development.yaml")
    task = protocol["task"]
    root = "artifacts/data/gray_scott_r2_seed42/highfreq_confirmation"
    directory = ensure_artifact_path(f"{root}/query_{args.resolution}")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for order, sample in enumerate(protocol["query"]):
        jobs.append({
            **_base(config, resolution=args.resolution, steps=50),
            "seed": int(sample["case_id"]), "case_id": int(sample["case_id"]),
            "boundary_seed": int(task["boundary_seed"]),
            "initial_seed": int(sample["initial_seed"]),
            "boundary_regime": task["boundary_regime"],
            "correlation": float(task["correlation"]),
            "amplitude": float(task["amplitude"]),
            "feed": float(task["feed"]), "kill": float(task["kill"]),
            "role": "r2_highfreq_confirmation", "fidelity": 1,
            "target_task_id": 0, "boundary_id": int(task["boundary_id"]),
            "initial_condition_id": int(sample["initial_condition_id"]),
            "pair_id": int(sample["pair_id"]), "order": order,
            "output": str(directory / f"pair_{int(sample['pair_id']):07d}.pt"),
        })
    with workspace_audit(f"generate_gray_scott_r2_highfreq_confirmation{args.resolution}") as audit:
        rows = _run_jobs(
            jobs, args.workers, f"gray_scott_r2_highfreq_confirmation_{args.resolution}"
        )
        output = write_index(
            f"{root}/query_{args.resolution}.index.json", rows,
            metadata={
                "equation": "gray_scott", "condition_channels": 3,
                "split": "r2_highfreq_confirmation", "resolution": args.resolution,
                "steps": 50, "target_task_id": 0,
                "boundary_regime": "high_frequency", "protocol_sha256": expected,
                "selection_used_query": False,
            },
        )
        audit.update({"output": str(output), "rollouts": len(rows), "workers": args.workers})
    print(json.dumps({"output": str(output), "rollouts": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
