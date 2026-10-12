# Representative frozen figures

This directory contains a compact PNG subset copied from the local artifact
store. Plotting scripts remain under `blrpng/cli/`; full-resolution field and
trajectory panels remain local-only so that the public source snapshot stays
small. They can be regenerated from the frozen local records and plotting
commands documented in `docs/EXPERIMENT_STATUS.md`.

- `allen_cahn_performance_overview.png`: primary accuracy and transfer summary.
- `allen_cahn_paired_task_effects.png`: task-wise paired improvements.
- `allen_cahn_accuracy_cost.png`: target accuracy--cost comparison.
- `poisson_negative_control.png`: linear negative-control comparison.
- `gray_scott_performance.png`: final mixed Gray--Scott result.
- `gray_scott_error_growth.png`: long-rollout behavior.
- `ns_transient_diagnostics.png`: energy, enstrophy and path-information traces.

The following high-resolution panels are intentionally retained only in the
ignored local artifact store:

- Allen--Cahn ground-truth/method field comparison.
- Transient NS physical-time evolution.
- Transient NS Newton-path state/correction comparison.
