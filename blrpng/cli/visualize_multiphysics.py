from __future__ import annotations

import argparse
import json

import numpy as np

from blrpng.analysis.visualize import multifield_comparison
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a frozen multiphysics prediction snapshot.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest_path = resolve_in_project(args.manifest, must_exist=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blrpng_visualization_input_v1":
        raise ValueError("Unsupported visualization manifest schema.")
    snapshot_path = resolve_in_project(manifest["snapshot_path"], must_exist=True)
    if sha256_file(snapshot_path) != manifest["snapshot_sha256"]:
        raise ValueError("Frozen visualization snapshot hash mismatch.")
    snapshot = np.load(snapshot_path, allow_pickle=False)
    with workspace_audit("visualize_multiphysics"):
        result = multifield_comparison(
            prediction=snapshot["prediction"],
            target=snapshot["target"],
            boundary_mask=snapshot["boundary_mask"],
            field_names=list(manifest["field_names"]),
            equation=str(manifest["equation"]),
            stem=str(manifest["stem"]),
        )
        report = write_json(
            f"artifacts/reports/{manifest['stem']}_multiphysics_visualization.json",
            {"input_manifest": str(manifest_path), "input_sha256": sha256_file(manifest_path), "figure": result},
        )
    print(json.dumps({"report": str(report), "figure": result}, indent=2))


if __name__ == "__main__":
    main()

