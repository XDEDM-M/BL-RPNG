from __future__ import annotations

import argparse
import json

from blrpng.cli.generate_gray_scott_formal import _metadata, _run_jobs, _source_jobs
from blrpng.config import load_config
from blrpng.data.capacity import require_capacity
from blrpng.data.generate import workspace_audit
from blrpng.data.gray_scott_protocol import validate_gray_scott_protocol
from blrpng.data.shards import write_index
from blrpng.paths import resolve_in_project


ROOT = "artifacts/data/gray_scott_r2_seed42"
PROTOCOL = "artifacts/manifests/gray_scott_r2_seed42_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gray--Scott R2 paired source data.")
    parser.add_argument("--config", default="configs/gray_scott_r2_development.yaml")
    parser.add_argument("--section", choices=("source256", "paired128"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = json.loads(resolve_in_project(PROTOCOL, must_exist=True).read_text(encoding="utf-8"))
    validate_gray_scott_protocol(protocol)
    require_capacity((4 if args.section == "source256" else 1) * 2**30)
    outputs = []
    with workspace_audit(f"generate_gray_scott_r2_{args.section}") as audit:
        for split in ("source_train", "source_validation"):
            if args.section == "source256":
                resolution, limit = 256, None
            else:
                resolution = 128
                limit = int(protocol["design"][
                    "paired_train_rollouts" if split == "source_train" else
                    "paired_validation_rollouts"
                ])
            jobs, actual_resolution, steps = _source_jobs(
                protocol, config, split, data_root=ROOT,
                resolution_override=resolution, limit=limit,
            )
            rows = _run_jobs(jobs, args.workers, f"gray_scott_r2_{split}_{resolution}")
            suffix = "paired_128" if args.section == "paired128" else "256"
            output = write_index(
                f"{ROOT}/{split}_{suffix}.index.json", rows,
                metadata={
                    **_metadata(protocol, split, actual_resolution, steps),
                    "r2_protocol_sha256": protocol["protocol_sha256"],
                    "paired": args.section == "paired128",
                },
            )
            outputs.append(str(output))
        audit.update({"section": args.section, "outputs": outputs, "workers": args.workers})
    print(json.dumps({"section": args.section, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
