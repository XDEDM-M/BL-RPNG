# Ordinary full-FT + direct256 baseline: LR=3e-5 seed1 replication

Date: 2026-10-16. This append-only amendment authorizes exactly one development
replication of the completed seed0 low-side result. It supersedes the original
LR3E5 card's local 1.8 GPU-h dispatch gate and 0.9 GPU-h uncertainty buffer for
this bounded replication only; the mission's 72 GPU-h exploration ceiling and
48 GPU-h confirmation/control reserve remain binding. It does not authorize any
additional exploration or query/test access.

## Question and frozen protocol

Does the seed0 signal at target LR `3e-5` reproduce for another predeclared
model seed under the same ordinary full-FT + direct256 protocol?

- Model seed `1`, support seed `271829`, tasks `[0,1,2,3,4,5]`, `m=4`.
- Ordinary final-only source checkpoint, full-parameter fine-tuning, final-only
  loss plus direct64-to256 two-step consistency (`weight=0.05`), 100 epochs,
  cosine schedule, minimum LR `1e-6`, and the same batch size as the seed0 card.
- Only model seed changes from the seed0 LR=`3e-5` screen. Reuse its six
  canonical support and LR-validation manifests. Select checkpoints on LR-dev
  only. Run all six HR-dev evaluations only after all six fits are complete.
- No query/test data may be generated, loaded, evaluated, or used for selection.
- This is a two-seed development replication. It is not the frozen
  confirmation, does not establish a general cross-seed ranking, and cannot
  promote a champion by itself.

## Hard budget gate and stop rule

The corrected terminal ledger at dispatch is `70.935533173839 GPU-h` against
the `72 GPU-h` exploration ceiling, leaving `1.064466826161 GPU-h` nominally.
Reserve an estimated `0.020674800773 GPU-h` for 24 historical untimed HR-dev
reports, leaving a deployable ceiling of `1.043792025388 GPU-h`. The
conservative empirical replication estimate is `0.931556 GPU-h`: six times the prior maximum observed
LR-task charge (`0.15439786`) plus six times the maximum timed HR-dev charge
(`0.00086145`). These are observed bounds, not guarantees.

- Run on GPU0 only, with no concurrent GPU work. Dispatch each task as a
  separate command with a task-specific attempt suffix, so the ledger and
  artifacts remain immutable and a budget check can occur between fits.
- Before each LR fit and HR-dev evaluation, refresh the terminal ledger and
  include the active/remaining jobs in a completion-cost projection. Do not
  start a cell if its conservative projected total would cross the 72 GPU-h
  exploration ceiling after preserving the 48 GPU-h reserve.
- Stop immediately if measured task cost exceeds its projected envelope enough
  that all remaining frozen cells no longer fit under the hard ceiling. Record
  every completed and interrupted attempt. Do not substitute tasks or seeds.
- No further exploration follows this one replication, regardless of outcome.

## Artifacts

- Config: `configs/phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml`.
- LR command template (run once for each task ID `N=0..5`; set `NN` to its
  two-digit form `00..05`):
  `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_trials.py --config configs/phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml --trial ordinary_full_comm256_lr3e-5 --task N --seed 1 --attempt-suffix lr3e5seed1rep-taskNN`.
- After all six LR fits finish, run the matching HR-dev command for each task:
  `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python scripts/evaluate_phase1_hrdev.py --task-result results/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5seed1rep-taskNN_m4_s1_ss271829/taskNN.json --task N`.

## Outcome

All six 100-epoch fits and six post-selection HR-dev evaluations completed.
Seed1 LR-dev mean is `0.025756364181`; HR-dev mean is `0.044410349642`.
Together with the seed0 screen, the matched two-seed LR=`3e-5` HR-dev mean is
`0.044567777087`, descriptively 20.86% below the matched seed0/seed1 ordinary
LR=`1e-4` mean (`0.056316934007`). The two-seed matched path-source C256 mean
is `0.054183115603`. This is development evidence on one support draw and six
fixed families, not confirmation or a three-seed ranking. Full metrics, costs,
commands and raw artifact hashes are in
`artifacts/reports/PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md` and its
machine-readable summary.
