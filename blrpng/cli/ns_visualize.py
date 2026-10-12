"""Create frozen 300-dpi PNG diagnostics for the SfePy-NS stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from blrpng.data.generate import workspace_audit
from blrpng.paths import PROJECT_ROOT, ensure_artifact_path, sha256_file, write_json


def _save(figure: plt.Figure, name: str, sources: list[str]) -> Path:
    path = ensure_artifact_path(
        f"artifacts/figures/navier_stokes_sfepy_seed42/{name}.png"
    )
    figure.savefig(path, dpi=300)
    plt.close(figure)
    write_json(
        path.with_suffix(".json"),
        {
            "png_sha256": sha256_file(path),
            "sources": [
                {"path": source, "sha256": sha256_file(PROJECT_ROOT / source)}
                for source in sources
            ],
            "checkpoint_selection_performed": False,
        },
    )
    return path


def _vorticity(field: torch.Tensor) -> torch.Tensor:
    spacing = 1.0 / float(field.shape[-1] - 1)
    return torch.gradient(field[1], spacing=(spacing,), dim=(0,))[0] - torch.gradient(
        field[0], spacing=(spacing,), dim=(1,)
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="artifacts/reports/navier_stokes_sfepy_seed42/prescreen.json",
    )
    args = parser.parse_args()
    report_path = PROJECT_ROOT / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["records"]
    reynolds = [int(value) for value in report["reynolds"]]
    dts = [float(value) for value in report["dt_candidates"]]
    sources = [args.report]
    outputs: list[Path] = []
    with workspace_audit("ns_visualize"):
        discrepancy = np.zeros((len(reynolds), len(dts)))
        iterations = np.zeros_like(discrepancy)
        for row_index, re_value in enumerate(reynolds):
            for column_index, dt in enumerate(dts):
                selected = [
                    row
                    for row in records
                    if int(row["reynolds"]) == re_value and float(row["dt"]) == dt
                ]
                discrepancy[row_index, column_index] = 100.0 * np.median(
                    [row["correction_endpoint_discrepancy"] for row in selected]
                )
                iterations[row_index, column_index] = np.median(
                    [row["newton_iterations"] for row in selected]
                )
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        shown = axes[0].imshow(
            np.maximum(discrepancy, 1e-6),
            norm=colors.LogNorm(vmin=1e-6, vmax=max(5.0, float(discrepancy.max()))),
            cmap="viridis",
            aspect="auto",
        )
        for i in range(len(reynolds)):
            for j in range(len(dts)):
                axes[0].text(j, i, f"{discrepancy[i, j]:.3g}%", ha="center", va="center", color="white", fontsize=8)
        figure.colorbar(shown, ax=axes[0], label="discrepancy (%)")
        shown_iterations = axes[1].imshow(iterations, vmin=1, vmax=max(3, float(iterations.max())), cmap="magma", aspect="auto")
        for i in range(len(reynolds)):
            for j in range(len(dts)):
                axes[1].text(j, i, f"{iterations[i, j]:.1f}", ha="center", va="center", color="white")
        figure.colorbar(shown_iterations, ax=axes[1], label="iterations")
        for axis, title in zip(axes, ("Newton target vs endpoint", "Accepted Newton iterations")):
            axis.set_xticks(range(len(dts)), [str(value) for value in dts])
            axis.set_yticks(range(len(reynolds)), [str(value) for value in reynolds])
            axis.set_xlabel("dt")
            axis.set_ylabel("Re")
            axis.set_title(title)
        figure.suptitle("Path-information gate: all tested settings fail the 5% discrepancy criterion")
        outputs.append(_save(figure, "02_path_richness_heatmap", sources))

        figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for re_value in reynolds:
            selected = [row for row in records if int(row["reynolds"]) == re_value]
            axes[0].scatter(
                [row["grid_divergence_rms"] for row in selected],
                [row["weak_continuity"] for row in selected],
                s=20,
                alpha=0.75,
                label=f"Re={re_value}",
            )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].axvline(1e-3, color="#D55E00", linestyle=":")
        axes[0].axhline(1e-6, color="#D55E00", linestyle=":")
        axes[0].set_xlabel("collocated-grid divergence RMS")
        axes[0].set_ylabel("Taylor–Hood weak continuity residual")
        axes[0].legend(frameon=False)
        summaries = report["summary_by_dt"]
        axes[1].plot(dts, [100 * summaries[str(dt)]["time_refinement_velocity_error_max"] for dt in dts], "o-", label="velocity")
        axes[1].plot(dts, [100 * summaries[str(dt)]["time_refinement_pressure_error_max"] for dt in dts], "s--", label="pressure")
        axes[1].plot(dts, [100 * summaries[str(dt)]["time_refinement_vorticity_error_max"] for dt in dts], "^--", label="vorticity")
        axes[1].axhline(1.0, color="#D55E00", linestyle=":", label="velocity gate")
        axes[1].set_xlabel("dt")
        axes[1].set_ylabel("dt vs dt/2 error (%)")
        axes[1].legend(frameon=False)
        axes[2].semilogy(dts, [summaries[str(dt)]["maximum_relative_residual"] for dt in dts], "o-", label="relative nonlinear residual")
        axes[2].semilogy(dts, [summaries[str(dt)]["maximum_weak_continuity"] for dt in dts], "s--", label="weak continuity")
        axes[2].axhline(1e-8, color="#D55E00", linestyle=":")
        axes[2].set_xlabel("dt")
        axes[2].set_ylabel("maximum diagnostic")
        axes[2].legend(frameon=False)
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.suptitle("Numerical fidelity: weak FEM conservation is distinct from grid interpolation divergence")
        outputs.append(_save(figure, "03_numerical_fidelity", sources))

        dt = max(dts)
        shard_sources = []
        figure, axes = plt.subplots(len(reynolds), 4, figsize=(14, 3.5 * len(reynolds)), constrained_layout=True)
        for row_index, re_value in enumerate(reynolds):
            relative = f"artifacts/data/navier_stokes_sfepy_prescreen/re{re_value}_dt{dt:g}.pt"
            shard_sources.append(relative)
            payload = torch.load(PROJECT_ROOT / relative, map_location="cpu", weights_only=False)
            field = payload["trajectory"][-1]
            speed = torch.sqrt(field[0].square() + field[1].square())
            values = (speed, field[2], _vorticity(field))
            titles = ("speed", "pressure", "vorticity")
            for column, (value, title) in enumerate(zip(values, titles)):
                image = axes[row_index, column].imshow(value.T, origin="lower", cmap="coolwarm" if title != "speed" else "viridis")
                figure.colorbar(image, ax=axes[row_index, column], fraction=0.046)
                axes[row_index, column].set_title(f"Re={re_value}: {title}")
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
            counts = payload["accepted_newton_counts"]
            residuals = payload["accepted_newton_residuals"]
            for step in range(residuals.shape[0]):
                count = int(counts[step])
                normalized = residuals[step, :count] / residuals[step, 0].clamp_min(1e-30)
                axes[row_index, 3].semilogy(range(count), normalized, "o-", alpha=0.75, label=f"step {step + 1}")
            axes[row_index, 3].set_xlabel("accepted Newton state")
            axes[row_index, 3].set_ylabel("relative residual")
            axes[row_index, 3].set_title(f"Re={re_value}: Newton convergence")
            axes[row_index, 3].grid(alpha=0.2)
        axes[0, 3].legend(frameon=False, fontsize=7)
        figure.suptitle(f"Representative transient fields and accepted Newton paths (dt={dt:g})")
        outputs.append(_save(figure, "04_fields_and_newton_paths", sources + shard_sources))

        index_path = write_json(
            "artifacts/figures/navier_stokes_sfepy_seed42/index.json",
            {
                "schema": "blrpng_ns_sfepy_prescreen_figure_index_v1",
                "source_report": args.report,
                "source_report_sha256": sha256_file(report_path),
                "figures": [
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "sha256": sha256_file(path),
                    }
                    for path in outputs
                ],
                "model_comparison_figures_available": False,
                "reason": report["decision"],
            },
        )
    print(json.dumps({"index": str(index_path), "figures": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
