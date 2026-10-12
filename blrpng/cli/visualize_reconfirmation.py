from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, resolve_in_project, sha256_file, write_json


METHOD_LABELS = {
    "nlt": "BL-RPNG-NLT",
    "ordinary_nlt": "Final-pretrain + NLT",
    "nlt_path": "NLT + target path",
    "nlt_commutator": "NLT + commutator",
    "nlt_commutator_defect": "NLT + comm. + defect",
    "lr_scratch": "Lifted LR scratch",
}


def _load(path: str) -> dict:
    return json.loads(resolve_in_project(path, must_exist=True).read_text(encoding="utf-8"))


def _save_figure(figure: plt.Figure, stem: str) -> list[dict[str, str]]:
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = ensure_artifact_path(f"artifacts/figures/{stem}.{suffix}")
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render frozen Allen-Cahn blind-query summary figures."
    )
    parser.add_argument("--resolution", type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    resolution = int(args.resolution)
    primary_path = (
        "artifacts/reports/"
        f"allen_cahn_reconfirmation_seed42_query{resolution}_h20_m4.json"
    )
    efficiency_path = (
        "artifacts/reports/"
        f"allen_cahn_reconfirmation_seed42_query{resolution}_efficiency.json"
    )
    primary = _load(primary_path)
    efficiency = _load(efficiency_path)
    methods = [
        "nlt", "ordinary_nlt", "nlt_path", "nlt_commutator",
        "nlt_commutator_defect", "lr_scratch",
    ]
    colors = ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), constrained_layout=True)

    values = [100 * primary["method_mean_relative_l2"][method] for method in methods]
    axes[0, 0].bar(range(len(methods)), values, color=colors)
    axes[0, 0].set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=28, ha="right")
    axes[0, 0].set_ylabel("20-step relative L2 (%)")
    axes[0, 0].set_title(f"(a) Frozen method matrix, {resolution}²")
    axes[0, 0].grid(axis="y", alpha=0.25)

    families = list(primary["family_mean_relative_l2"])
    x = np.arange(len(families))
    width = 0.25
    for offset, method, color in zip((-1, 0, 1), ("nlt", "ordinary_nlt", "lr_scratch"), (colors[0], colors[1], colors[-1])):
        axes[0, 1].bar(
            x + offset * width,
            [100 * primary["family_mean_relative_l2"][family][method] for family in families],
            width,
            label=METHOD_LABELS[method],
            color=color,
        )
    axes[0, 1].set_xticks(x, [value.replace("_", "\n") for value in families], rotation=18, ha="right")
    axes[0, 1].set_ylabel("20-step relative L2 (%)")
    axes[0, 1].set_title("(b) Six unseen boundary families")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.25)

    budgets = [1, 2, 4, 8]
    curve = [100 * efficiency["mean_relative_l2"][f"m{budget}_h20"] for budget in budgets]
    axes[1, 0].plot(budgets, curve, "o-", color=colors[0], linewidth=2.2)
    axes[1, 0].scatter([4], [curve[2]], s=95, facecolors="none", edgecolors="#D55E00", linewidths=2, label="Pre-frozen primary")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(budgets, [str(value) for value in budgets])
    axes[1, 0].set_xlabel("LR support rollouts per target boundary")
    axes[1, 0].set_ylabel("20-step relative L2 (%)")
    axes[1, 0].set_title("(c) Target-boundary data efficiency")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.25)

    horizon_values = [
        100 * efficiency["mean_relative_l2"]["m4_h20"],
        100 * efficiency["mean_relative_l2"]["m4_h50"],
    ]
    axes[1, 1].bar([0, 1], horizon_values, color=[colors[0], colors[2]], width=0.6)
    axes[1, 1].set_xticks([0, 1], ["20 steps", "50 steps"])
    axes[1, 1].set_ylabel("Per-rollout relative L2 (%)")
    axes[1, 1].set_title("(d) Long-horizon stability (m=4)")
    axes[1, 1].grid(axis="y", alpha=0.25)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)

    csv_path = ensure_artifact_path(
        "artifacts/figures/"
        f"allen_cahn_reconfirmation_seed42_query{resolution}_summary.csv"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("panel", "group", "method_or_budget", "relative_l2"))
        for method in methods:
            writer.writerow(("method", "all", method, primary["method_mean_relative_l2"][method]))
        for family in families:
            for method in ("nlt", "ordinary_nlt", "lr_scratch"):
                writer.writerow(("family", family, method, primary["family_mean_relative_l2"][family][method]))
        for budget in budgets:
            writer.writerow(("support", "all", budget, efficiency["mean_relative_l2"][f"m{budget}_h20"]))
        writer.writerow(("horizon", "all", 20, efficiency["mean_relative_l2"]["m4_h20"]))
        writer.writerow(("horizon", "all", 50, efficiency["mean_relative_l2"]["m4_h50"]))

    with workspace_audit(f"visualize_reconfirmation_{resolution}") as audit:
        figures = _save_figure(
            figure,
            f"allen_cahn_reconfirmation_seed42_query{resolution}_summary",
        )
        plt.close(figure)
        metadata = write_json(
            "artifacts/figures/"
            f"allen_cahn_reconfirmation_seed42_query{resolution}_summary.json",
            {
                "schema": "blrpng_frozen_figure_v1",
                "selection_performed": False,
                "source_reports": [
                    {"path": primary_path, "sha256": sha256_file(resolve_in_project(primary_path))},
                    {"path": efficiency_path, "sha256": sha256_file(resolve_in_project(efficiency_path))},
                ],
                "csv": str(csv_path.relative_to(PROJECT_ROOT)),
                "csv_sha256": sha256_file(csv_path),
                "figures": figures,
            },
        )
        audit.update({"metadata": str(metadata), "figures": len(figures)})
    print(json.dumps({
        "metadata": str(metadata),
        "csv": str(csv_path),
        "figures": figures,
    }, indent=2))


if __name__ == "__main__":
    main()
