from __future__ import annotations

import argparse
import json

from blrpng.analysis.gates import development_gate
from blrpng.config import load_config
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply development gates and unlock frozen fine-grid generation."
    )
    parser.add_argument("--development-config", required=True)
    parser.add_argument("--formal-config", required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help=(
            "JSON with data_reports, gradient_ratios, seen_blrpng, seen_pretrained, "
            "unseen_blrpng, unseen_pretrained."
        ),
    )
    args = parser.parse_args()
    development = load_config(args.development_config)
    formal = load_config(args.formal_config)
    evidence_path = resolve_in_project(args.evidence, must_exist=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    with workspace_audit("gate_formal"):
        result = development_gate(config=development, **evidence)
        result.update(
            {
                "config_sha256": formal["_config_sha256"],
                "development_config_sha256": development["_config_sha256"],
                "evidence": str(evidence_path),
            }
        )
        write_json("artifacts/manifests/formal_gate_passed.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
