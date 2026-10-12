from __future__ import annotations

import argparse
import json

from blrpng.analysis.visualize import generate_core_figures
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create frozen BL-RPNG v2 figures without checkpoint selection.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default="artifacts/reports/v2_figure_manifest.json")
    args = parser.parse_args()
    summary_path = resolve_in_project(args.summary, must_exist=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("frozen", False):
        raise PermissionError("Visualization only accepts a frozen summary.")
    with workspace_audit("visualize"):
        figures = generate_core_figures(summary)
        output = write_json(args.output, {"schema": "blrpng_figure_manifest_v1", "summary": str(summary_path), "figures": figures})
    print(json.dumps({"output": str(output), "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
