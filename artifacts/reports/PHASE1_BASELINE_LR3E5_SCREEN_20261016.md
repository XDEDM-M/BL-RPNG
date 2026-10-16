# Ordinary full-FT + direct256 baseline: LR=3e-5 screen

Date: 2026-10-16. This is a one-seed, six-family development screen. It is a
local hyperparameter comparison, not a confirmation result.

## Frozen protocol

The ordinary final-only source checkpoint, full-parameter target fine-tuning,
final-only loss plus direct64-to256 two-step consistency (`weight=0.05`),
`m=4`, 100 epochs, cosine schedule to `1e-6`, model seed 0, and support seed
271829 were fixed. Relative to ordinary full-FT + comm256 at LR=`1e-4`, only
the target LR changed to `3e-5`. The six tasks use the same support indices
and canonical 64x64 LR-development-validation indices. Checkpoints were
selected by LR-dev only, and HR-dev inference was run only after all six LR
fits completed. All six HR-dev reports say `query_read=false`; no query/test
data was opened.

The runner config fingerprint is
`bbb44e453a507c1046e9b092f114bec4b8b381237e029917a63120336d1e226d`; the raw
YAML byte SHA-256 is
`0a47b09387437ea2f90f72352b2603dacf51b5344bb5e77fce5251d493eb14dc`. The
resolved trial-spec SHA-256 is also recorded in the frozen run summary. The
training engine SHA-256 is
`6a2ab12001562413f62c19918ea66824308077ef2b4c141317a98c1c0ddea5b0`.

## Development results

| Task / family | LR-dev `1e-4` | LR-dev `3e-5` | LR-dev `3e-4` | HR-dev `1e-4` | HR-dev `3e-5` | HR-dev `3e-4` |
|---|---:|---:|---:|---:|---:|---:|
| 0 / high-frequency | 0.04337541 | 0.02526402 | 0.16078084 | 0.07271281 | 0.06192119 | 0.17564816 |
| 1 / amplitude-OOD | 0.04804767 | 0.02828506 | 0.11531357 | 0.04805644 | 0.02917969 | 0.09549712 |
| 2 / matern-GP | 0.03322085 | 0.02325362 | 0.09026610 | 0.04177176 | 0.03364962 | 0.11483409 |
| 3 / localized-bump | 0.03694564 | 0.02536993 | 0.10606825 | 0.06366513 | 0.05134735 | 0.16877616 |
| 4 / mixed-Fourier | 0.04212978 | 0.02609319 | 0.12062085 | 0.08776810 | 0.05136385 | 0.20089716 |
| 5 / piecewise-smooth | 0.03767342 | 0.02086638 | 0.15312910 | 0.04268695 | 0.04088952 | 0.12378177 |
| **Equal-task mean** | **0.04023213** | **0.02485537** | **0.12436312** | **0.05944353** | **0.04472520** | **0.14657241** |

The matched seed0/support271829 HR-dev mean improves by `24.76%` at LR=`3e-5`
relative to `1e-4`, and every family improves. LR-dev mean improves by
`38.22%`. The LR-selected best epochs (zero-based) are `[3,11,18,6,6,13]`;
each task still ran the full 100 epochs. The `3e-4` high-side point is worse
than `3e-5` on all six tasks; its HR-dev mean is `0.14657241`, so the low-side
point is `69.49%` lower than that paired local bracket result.

This is strong evidence that LR=`1e-4` was not the best setting for this
ordinary baseline under this direct256 objective. It is still only one model
seed and one support draw. The three-seed Stage-2 ordinary baseline mean at
LR=`1e-4` is `0.05488117`, while path-source + comm256 is `0.05347209`; the
single-seed LR=`3e-5` mean is lower than both, but it cannot replace either
three-seed aggregate or establish a cross-seed ranking. The new ordinary
baseline result materially strengthens the required baseline comparison and
means the apparent Stage-2 path-source edge must not be presented as robust
until the lower LR is replicated. It does not by itself confirm or refute the
main hypothesis.

## Cost and remaining budget

- Six LR training tasks cost `0.924687 GPU-h`; the six HR-dev evaluations cost
  `0.004221 GPU-h`; total measured GPU use is `0.928908 GPU-h`.
- The first post-screen snapshot at
  `artifacts/reports/phase1_progress_after_r5_complete_20261016_1349.json`
  undercounted separate retry attempts because the prior aggregator collapsed
  by logical trial ID. The corrected snapshot
  `artifacts/reports/phase1_progress_after_r5_complete_20261016_1355_corrected.json`
  charges `70.935533 GPU-h`; the correction adds `0.300653 GPU-h` in distinct
  attempts and removes duplicate exclusion annotations. The independent audit
  and reconciliation are recorded in `docs/agents/REVIEW.md` and
  `docs/BUDGET.md`.
- After the seed0 screen, nominal exploration headroom was `1.064467 GPU-h`;
  after estimating `0.020675 GPU-h` for historical untimed HR-dev reports, the
  conservative headroom was `1.043792 GPU-h`. A dated amendment superseded
  this card's local 0.9-GPU-h uncertainty buffer for exactly one predeclared
  seed1 replication. That replication cost `0.930489 GPU-h` and is documented
  in `artifacts/reports/PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md`.
  No further six-task exploration fits the remaining allocation; the 48-GPU-h
  confirmation/control reserve remains protected.

## Reproduction and raw evidence

- Frozen card/config:
  `docs/experiment_cards/BASELINE_FULLCOMM256_LR3E5_20261016.md`,
  `configs/phase1_baseline_fullcomm256_lr3e-5_screen.yaml`.
- LR command:
  `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_trials.py --config configs/phase1_baseline_fullcomm256_lr3e-5_screen.yaml --trial ordinary_full_comm256_lr3e-5 --seed 0 --attempt-suffix lr3e5screen`.
- Task reports/checkpoints and summary:
  `results/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5screen_m4_s0_ss271829/`.
- HR-dev reports:
  `artifacts/reports/phase1_hrdev_evaluations/ordinary_full_comm256_lr3e-5-lr3e5screen_task{00..05}_s0_ss271829.json`.
- Each HR-dev reproduction command follows:
  `CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python scripts/evaluate_phase1_hrdev.py --task-result results/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5screen_m4_s0_ss271829/taskXX.json --task X`.
- Paired `1e-4` task and HR-dev results are in the `S2PILOT` results folder
  and `pilot_ordinary_full_comm256_taskXX_s0_ss271829.json` reports. The
  `3e-4` screen is documented in
  `artifacts/reports/PHASE1_BASELINE_LR3E4_SCREEN_20261016.md`.
- Terminal ledger and corrected snapshots:
  `docs/EXPERIMENTS.csv`,
  `artifacts/reports/phase1_progress_after_r5_complete_20261016_1355_corrected.json`,
  `artifacts/reports/phase1_progress_after_seed1_hrdev_complete_20261016.json`.
- Independent result audit is recorded in `docs/agents/REVIEW.md` after
  completion.
