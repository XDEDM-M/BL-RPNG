# Allen–Cahn seed42 paper visualization set

> The complete generated set remains in the local artifact store. A compact
> repository-safe selection is available under `docs/figures/`.

All figures are deterministic PNG outputs from frozen checkpoints and frozen
256² query data. Representative fields use the lowest pre-frozen task ID in
each boundary family and the lowest frozen query rollout ID. They were not
selected by their HR error.

| Figure | Purpose |
|---|---|
| `paper_seed42_performance_overview.png` | Main error, boundary-family generalization, support efficiency, resolution transfer, and long-horizon stability. |
| `paper_seed42_paired_task_effects.png` | 48/48 paired task wins and per-task improvement heatmaps. |
| `paper_seed42_rollout_error_distributions.png` | ECDFs over all 768 query rollouts, including final-step error. |
| `paper_seed42_physics_diagnostics.png` | One-step/rollout/near-boundary/interior, PDE residual, Fourier shells, and 128²→256² transfer. |
| `paper_seed42_accuracy_cost.png` | Marginal target accuracy–cost trade-off and the 10× engineering threshold. |
| `paper_seed42_boundary_conditions.png` | Representative Dirichlet perimeter profiles and their harmonic lifts. |
| `paper_seed42_representative_method_comparison.png` | Ground truth, three methods, and absolute errors on a shared scale for all six families. |
| `paper_seed42_representative_error_growth.png` | 50-step instantaneous error growth for three methods on the same rollouts. |
| `paper_seed42_evolution_<family>.png` | Ground truth, BL-RPNG-NLT, and shared-scale absolute error at steps 0/1/5/10/20/35/50. |

The machine-readable provenance manifest is
`artifacts/figures/paper_seed42_visualizations.json`; accompanying plotted data
are in `artifacts/figures/paper_seed42_visualization_data.csv`.

Claim limits remain unchanged: these are seed42 results; the HR specialist
marker is resource-capped, unsaturated, and contextual rather than a paired
oracle-parity comparison. The 50-step ordinary/scratch trajectories are
post-freeze visualization-only rollouts and did not affect model selection.
