from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blrpng.analysis.visualize import STYLE, _save
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frozen HR specialist convergence curves.")
    parser.add_argument("--audit", default="artifacts/reports/oracle_convergence_audit.json")
    parser.add_argument("--stem", default="oracle_specialist_convergence")
    args = parser.parse_args()
    audit_path = resolve_in_project(args.audit, must_exist=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    tasks = sorted({int(row["task"]) for row in audit["rows"]})
    with workspace_audit("visualize_oracle_convergence"):
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(len(tasks), 1, figsize=(6.2, max(3.0, 2.8 * len(tasks))),
                                     squeeze=False, constrained_layout=True)
            for axis, task in zip(axes[:, 0], tasks):
                for row in audit["rows"]:
                    if int(row["task"]) != task:
                        continue
                    label = f"{row['initialization']}, lr={row['learning_rate_tag']}, m={row['budget']}"
                    axis.plot(range(1, len(row["validation_curve"]) + 1), row["validation_curve"], label=label)
                    axis.scatter([int(row["best_epoch"]) + 1], [float(row["best_metric"])], s=18)
                axis.set_yscale("log")
                axis.set_xlabel("Epoch")
                axis.set_ylabel("LR validation rollout relL2")
                axis.set_title(f"Target task {task}: HR specialist convergence")
                axis.legend(fontsize=7)
        figure = _save(args.stem, fig, {
            "source_audit": str(audit_path),
            "gates": audit["gates"],
            "runs": audit["rows"],
        })
        report = write_json(
            f"artifacts/reports/{args.stem}_figure_manifest.json",
            {"source": str(audit_path), "figure": figure},
        )
    print(json.dumps({"report": str(report), "figure": figure}, indent=2))


if __name__ == "__main__":
    main()

