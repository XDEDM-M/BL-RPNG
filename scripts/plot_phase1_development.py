"""Export a static development-only scientific comparison from sealed summaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blrpng.paths import PROJECT_ROOT, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    source = PROJECT_ROOT / args.input
    output = PROJECT_ROOT / args.output_root
    if output.exists():
        raise FileExistsError(f"Use a fresh figure directory: {output}")
    data = json.loads(source.read_text())
    if data.get("query_read") is not False or data.get("confirmation_evidence") is not False:
        raise ValueError("Only audited exploratory development summaries are accepted")
    labels = {
        "ordinary_lr1e-3_cosine100": "Ordinary source + FT",
        "final_lr1e-3_cosine100": "Path source + FT",
        "comm_lr1e-3_cw005_r1": "Path source + consistency",
    }
    rows = {r["trial"]: r for r in data["hr_development"] if r["complete_for_screen"] and r["m"] == 4
            and r["model_seed"] == 42 and r["support_seed"] == 271829}
    training = {r["trial"]: r for r in data["training"] if r["m"] == 4
                and r["model_seed"] == 42 and r["support_seed"] == 271829}
    selected = [key for key in labels if key in rows]
    if len(selected) < 2:
        raise ValueError("At least two matched development methods are required")
    families = [c["family"] for c in rows[selected[0]]["cells"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colors = ["#64748b", "#2563eb", "#0f766e"]
    x = np.arange(len(families)); width = 0.8 / len(selected)
    for i, method in enumerate(selected):
        cells = rows[method]["cells"]
        if [c["family"] for c in cells] != families:
            raise ValueError("Method family pairing differs")
        axes[0].bar(x + (i - (len(selected)-1)/2) * width,
                    [100*c["mean"] for c in cells], width, label=labels[method], color=colors[i])
        axes[1].plot(["LR validation", "HR development"],
                     [100*training[method]["lr_mean"], 100*rows[method]["hr_mean"]],
                     marker="o", label=labels[method], color=colors[i])
    axes[0].set_xticks(x, [f.replace("_", "\n") for f in families], fontsize=9)
    axes[0].set_ylabel("Mean rollout relative L2 (%)")
    axes[0].set_title("Independent 256-grid HR development")
    axes[1].set_ylabel("Four-family mean relative L2 (%)")
    axes[1].set_title("64-grid adaptation to 256-grid inference")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    fig.suptitle("Exploratory only: seed 42, one support draw, m=4, 20-step rollout", fontsize=11)
    fig.tight_layout()
    output.mkdir(parents=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"development_comparison.{extension}", dpi=180, bbox_inches="tight")
    write_json(output / "evidence.json", {"source": args.input,
               "source_sha256": sha256_file(source), "methods": selected,
               "query_read": False, "confirmation_evidence": False,
               "uncertainty": "No CI: a single model/support seed and four development tasks."})
    print(output)


if __name__ == "__main__":
    main()
