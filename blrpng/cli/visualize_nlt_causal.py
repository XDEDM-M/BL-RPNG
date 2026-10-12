from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blrpng.analysis.visualize import STYLE, _save
from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, write_json


LABELS = {
    "ordinary_nlt": "ordinary pretrain→NLT",
    "nlt": "path pretrain→NLT",
    "nlt_path": "path pretrain→NLT+path",
    "nlt_commutator": "NLT+commutator",
    "nlt_commutator_defect": "NLT+commutator+defect",
}


def _load(path: str) -> dict:
    return json.loads(resolve_in_project(path, must_exist=True).read_text(encoding="utf-8"))


def main() -> None:
    summary = _load("artifacts/reports/nlt_causal_seed42_summary_exploratory.json")
    evaluation = _load("artifacts/reports/nlt_causal_seed42_query256_h20_exploratory.json")
    methods = list(LABELS)
    records = evaluation["records"]
    with workspace_audit("visualize_nlt_causal"):
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            values = [
                [row["relative_l2"] for row in records if row["method"] == method]
                for method in methods
            ]
            ax.boxplot(values, showfliers=False)
            ax.scatter(
                np.arange(1, len(methods) + 1), [np.mean(value) for value in values],
                color="#d62828", zorder=4, label="mean",
            )
            ax.set_xticks(np.arange(1, len(methods) + 1), [LABELS[m] for m in methods], rotation=18, ha="right")
            ax.set_ylabel("256²/20-step relative L2")
            ax.legend()
        first = _save("nlt_causal_error_distribution", fig, {
            "status": summary["status"], "methods": methods,
            "means": summary["means"], "values": values,
        })

        tasks = sorted({int(row["target_task_id"]) for row in records})
        baseline = {
            (int(row["target_task_id"]), int(row["initial_condition_id"])): row["relative_l2"]
            for row in records if row["method"] == "ordinary_nlt"
        }
        heat = np.zeros((len(methods) - 1, len(tasks)))
        for row_index, method in enumerate(methods[1:]):
            for column, task in enumerate(tasks):
                selected = [row for row in records if row["method"] == method and int(row["target_task_id"]) == task]
                candidate = np.mean([row["relative_l2"] for row in selected])
                reference = np.mean([baseline[(task, int(row["initial_condition_id"]))] for row in selected])
                heat[row_index, column] = (reference - candidate) / reference
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(6.6, 3.2))
            bound = max(float(np.max(np.abs(heat))), 1e-3)
            image = ax.imshow(heat, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
            ax.set_xticks(range(len(tasks)), tasks); ax.set_xlabel("Target boundary task")
            ax.set_yticks(range(len(methods) - 1), [LABELS[m] for m in methods[1:]])
            fig.colorbar(image, ax=ax, label="Improvement vs ordinary-pretrain NLT")
        second = _save("nlt_causal_task_heatmap", fig, {
            "tasks": tasks, "methods": methods[1:], "values": heat.tolist(),
        })
        output = write_json("artifacts/reports/nlt_causal_figure_manifest.json", {
            "schema": "blrpng_figure_manifest_v1", "source_status": summary["status"],
            "figures": [first, second],
        })
    print(json.dumps({"output": str(output), "figures": 2}, indent=2))


if __name__ == "__main__":
    main()
