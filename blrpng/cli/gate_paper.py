from __future__ import annotations

import argparse
import json

from blrpng.analysis.gates import paper_claims
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the five frozen Allen--Cahn paper claims."
    )
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    evidence_path = resolve_in_project(args.evidence, must_exist=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    with workspace_audit("gate_paper"):
        report = paper_claims(**evidence)
        report["evidence"] = str(evidence_path)
        write_json("artifacts/manifests/allen_cahn_paper_gate.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
