# Phase-1 exploratory progress, 2026-10-13

Status: IN_PROGRESS. This is not the final confirmation report. Check `docs/STATE.md` for the active job; GPU budgets are charged from terminal ledger rows, with active time and early untimed HR-development overhead disclosed separately.

## Current measured result

On four frozen development boundaries, seed42, support draw271829, four independent 64-grid support rollouts, and independent eight-rollout 256-grid HR development per task:

| Method/config | LR-validation mean | HR-development mean |
|---|---:|---:|
| Ordinary source, NLT, lr1e-3 cosine100 | 0.0342082131 | 0.0621957691 |
| Path source, NLT final-only, lr1e-3 cosine100 | 0.0279160336 | 0.0560245670 |
| Path source, 64-to128 one-step consistency weight0.05 | 0.0279855519 | 0.0543033645 |
| Path source, 64-to256 two-step consistency weight0.05 | 0.0281098347 | 0.0528792817 |
| Path source, NLT final-only, lr3e-4 cosine100 | 0.0281588136 | 0.0564806820 |
| Lifted scratch, best of three cosine100 LR screens, lr1e-4 | 0.5347863933 | 0.6103115785 |

Evidence recomputed with `scripts/summarize_phase1_progress.py`: `artifacts/reports/phase1_progress_20261013_1839.json`. Each row references sealed summary/task artifacts and independent HR-development reports. The initial figure exports predate the direct256 candidate: `artifacts/figures/phase1/development_20261013_1828/development_comparison.{png,svg,pdf}`.

The path-source final-only system is 18.3938% better than ordinary transfer on LR validation and 9.9222% better on HR development; the piecewise-smooth family instead favors ordinary transfer by 13.8857% relative to that baseline. One-step consistency lowers the path-source HR-development mean by 3.0722%, improving all four families, while its LR-validation mean is 0.2490% higher. This supports prioritizing measured cross-resolution stability rather than assuming FNO grid transfer is accurate.

These are one-seed development screens. Ordinary full FT and shared rollout/consistency improvements still require fair opportunities; a five-model-seed by three-support-seed independent confirmation table is not available. No confidence interval or confirmation superiority is claimed.

Direct256 two-step consistency further lowers the HR-development mean to0.0528792817, 5.6141% below R1 and14.9793% below ordinary transfer. Piecewise-smooth is still2.3370% worse than ordinary. Its four-task adaptation charge0.49481273GPU-hours is substantially higher, so its accuracy/cost tradeoff must be retained. These new HR-development reports include evaluator hashes and synchronized timing; selection still uses LR validation.

## Target path FT: completed negative evidence

Same path-source initialization, support draw, four tasks, LR1e-3/cos100:

| Path gradient-ratio setting | Mean LR-validation rollout error |
|---|---:|
| No target auxiliary path | 0.0279160336 |
| 0.05 | 0.0279272979 |
| 0.15 | 0.0280449178 |
| 0.30 | 0.0281933126 |

All three real path-FT settings are slightly worse, despite valid labels, nonzero shared gradients and gate activity near 50%. Lowering LR to3e-4 also gave a slightly worse paired result at ratio0.05. Raw trial artifacts are in `results/phase1/R1/path_*`; do not attribute this to a silent disabled path objective.

The retained LR path is nearly endpoint-collinear: median D3.02e-5, orthogonal projection residual2.80e-5, cosine0.9999999996. The planned layer3 mapping and matched endpoint/direction controls are frozen in `configs/phase1_r3_path_controls.yaml`, but have not yet been scored. The evidence currently supports a real accepted-path training implementation; it does not demonstrate an extra Newton-geometry benefit.

## Implemented and verified engineering improvements

- True LR H3 recursive supervision: same-rollout windows, differentiable predicted-state recursion, optional growing horizon, and explicit rejection of unsupported time-dependent auxiliary geometry. Sixteen selected CPU tests pass, including a real lifted-FNO engine backward pass; scored GPU trial pending.
- Compact Newton-selection evidence: saved float64 residual and step-norm traces, explicit counts, corrected raw-hook ordinal semantics, and exact threshold replay. Real SfePy smoke passed for one support rollout/two steps with normal dt/nonlinearity/tolerances; packaged shard boundary error0 and four replay checks passed. `artifacts/experiments/phase1/provenance_smoke_20261013/verified_durable/audit.json`.
- Future shard writes use atomic save/fsync and read-back validation. Old frozen shards are unchanged. The compact evidence replays selection but does not reconstruct full unsaved node states.
- Subsequent HR-development reports include execution code hashes, synchronized timing/ledger entries, checkpoint-hash checks and protection against overwriting prior reports. Earlier missing cost measurements are disclosed.

## Remaining research obligations

Continue R2 direct256 and H3 candidates, execute R3, strengthen ordinary/scratch baselines, then evaluate all six development families and fixed pilot seeds0/1/42. Freeze the confirmation protocol before independent query generation. Complete HR-expert convergence/data curves, few-shot curves, measured deployment costs, controls and final independent review under the mission's total120 GPU-hour cap and48-hour reserve. `docs/PHASE1_COMPLETION.md` records unfinished evidence without treating planned work as completed results.
