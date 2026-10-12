from __future__ import annotations

import argparse
import json

from blrpng.analysis.bootstrap import paired_hierarchical_bootstrap
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--metric", default="relative_l2")
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", default="artifacts/reports/bootstrap.json")
    args = parser.parse_args()
    records_path = resolve_in_project(args.records, must_exist=True)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    with workspace_audit("analyze"):
        report = paired_hierarchical_bootstrap(
            records,
            method=args.method,
            baseline=args.baseline,
            metric=args.metric,
            replicates=args.replicates,
            seed=args.seed,
        )
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
