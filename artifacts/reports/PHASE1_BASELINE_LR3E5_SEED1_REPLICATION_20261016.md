# R5 ordinary full-FT + direct256: LR=3e-5 seed1 replication

**Status: completed development replication; not confirmation.** This is the
predeclared seed1 follow-up to the seed0 local LR screen. All six tasks used
`m=4`, model seed `1`, support seed `271829`, and the same ordinary final-only
source initialization, full-parameter target fine-tuning, final loss plus
direct64-to256 two-step commutator consistency (`weight=0.05`), 100 epochs,
cosine schedule, and minimum LR `1e-6`. Each checkpoint was selected on
64×64, 20-step LR development validation. The six 256×256, 20-step HR-dev
evaluations ran after all six LR fits. Each HR report says `query_read=false`.
No query/test artifact was opened.

## Per-family results

| Task / family | LR-dev, seed1 | HR-dev, seed1, LR=3e-5 | Paired ordinary full + comm256, LR=1e-4 | Paired path-source C256 |
|---|---:|---:|---:|---:|
| 0 / high-frequency | 0.02517443 | 0.05992937 | 0.04924395 | 0.05184649 |
| 1 / amplitude-OOD | 0.02966063 | 0.03002297 | 0.04480050 | 0.02836940 |
| 2 / matern-GP | 0.02405344 | 0.03127426 | 0.04595283 | 0.04754582 |
| 3 / localized-bump | 0.02605149 | 0.05221141 | 0.05564401 | 0.06014429 |
| 4 / mixed-Fourier | 0.02521360 | 0.04919595 | 0.07098463 | 0.06478832 |
| 5 / piecewise-smooth | 0.02438459 | 0.04382814 | 0.05251610 | 0.05333420 |
| **Equal-family mean** | **0.02575636** | **0.04441035** | **0.05319034** | **0.05100475** |

The seed1 LR=`3e-5` HR-dev mean is 16.51% below its paired ordinary LR=`1e-4`
mean, improving in five of six families; high-frequency is worse. Together
with seed0, the two-seed LR=`3e-5` mean is `0.04456778`. On the matched
seed0/seed1 subset, ordinary LR=`1e-4` averages `0.05631693`, so the lower-LR
ordinary arm is 20.86% lower descriptively. The matched two-seed path-source
C256 average is `0.05418312`; the lower-LR ordinary arm is 17.75% lower on
that same two-seed subset. The family-by-family differences are in the
machine-readable summary.

These are two model seeds on one support draw and six fixed development
families. They are not a confirmatory interval or a generalization estimate.
The full three-seed Stage-2 aggregates remain ordinary LR=`1e-4` at
`0.05488117` and path-source C256 at `0.05347209`; do not replace either with
this two-seed subset or claim a stable cross-seed ranking. In particular, the
new result means the original 2.57% Stage-2 path-source advantage over the
ordinary LR=`1e-4` arm should not be treated as robust to baseline tuning.

## Cost and campaign budget

| Work | GPU-h |
|---|---:|
| Six LR target fits | 0.926292 |
| Six HR-dev evaluations | 0.004197 |
| **Seed1 incremental target-stage cost** | **0.930489** |

The fixed LR support, validation and HR-dev data were reused; this replication
generated no new PDE labels. The source checkpoint is shared campaign asset
`artifacts/checkpoints/phase1/R1/source_final_s1_scratch.pt`, so the table is
incremental target adaptation and evaluation cost, not source pretraining or
hyperparameter-search cost.

The final corrected terminal ledger is `71.866022 GPU-h` against the
`72 GPU-h` exploration allocation. It preserves the `48 GPU-h`
confirmation/control reserve. The ledger also excludes some historical
untimed HR-dev executions; the conservative estimate for those is
`0.020675 GPU-h`, leaving about `0.113303 GPU-h` for further exploration. That
remaining amount is below the approximately `0.93 GPU-h` needed for another
complete six-family LR-plus-HR screen. Snapshot:
[`phase1_progress_after_seed1_hrdev_complete_20261016.json`](phase1_progress_after_seed1_hrdev_complete_20261016.json).

The amended three-method, 48-task, three-seed primary confirmation matrix is
projected at about `158.2 GPU-h`, before the few-shot curve, HR expert
saturation, mechanism controls and confirmation-scale CPU data generation.
The full required evidence cannot fit the current remaining allocation while
preserving the reserve. Query generation remains gated.

## Reproduction and provenance

Frozen amendment and config:
[`BASELINE_FULLCOMM256_LR3E5_SEED1_REPLICATION_20261016.md`](../../docs/experiment_cards/BASELINE_FULLCOMM256_LR3E5_SEED1_REPLICATION_20261016.md),
[`phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml`](../../configs/phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml).
Run one task at a time (`N=0..5`) with its unique immutable attempt namespace:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python -u scripts/run_phase1_trials.py \
  --config configs/phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml \
  --trial ordinary_full_comm256_lr3e-5 --task N --seed 1 \
  --attempt-suffix lr3e5seed1rep-taskNN
```

After the six LR tasks terminate, evaluate each selected task result:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/ai4pde/bin/python \
  scripts/evaluate_phase1_hrdev.py \
  --task-result results/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5seed1rep-taskNN_m4_s1_ss271829/taskNN.json \
  --task N
```

Recompute the paired development report with
`/opt/miniconda3/envs/ai4pde/bin/python scripts/summarize_phase1_r5_lr3e5_replication.py`.
It writes
[`phase1_r5_lr3e5_seed1_replication_summary_20261016.json`](phase1_r5_lr3e5_seed1_replication_summary_20261016.json)
from the task results, HR-dev reports and ledger.

- Repository commit recorded by the runner: `6ccea77c9ef35a92ec95bce75b240d73c88e047d`.
- Raw trial YAML SHA-256: `e3bb759291ed462f81dfa2adb4dfcbd19a53c5b7b20ac225413cd31df2ac6ddd`.
- Seed1 final-only source checkpoint SHA-256: `8b9509f482b2029c0c12302449c93509762991e72a3343978a2bd2b55273507b`.
- Task manifest SHA-256: `d1013c914076b85cd8b19c1e89c3e69f4948a667a2693e613253e43e79a301b2`.
- Runner SHA-256: `00ac1f09c65f1e167159963eda165d9582581b20a1f234bf8183d2942c640f88`; training engine SHA-256: `6a2ab12001562413f62c19918ea66824308077ef2b4c141317a98c1c0ddea5b0`.
- Environment: Python `3.9.23`, PyTorch `2.5.1+cu121`, PyTorch CUDA runtime `12.1`, NVIDIA GeForce RTX 4090 (driver `580.178.04`).
- Raw task outputs/checkpoints: `results/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5seed1rep-task{00..05}_m4_s1_ss271829/` and `artifacts/checkpoints/phase1/R5/ordinary_full_comm256_lr3e-5-lr3e5seed1rep-task{00..05}_m4_s1_ss271829/`.
- HR-dev reports: `artifacts/reports/phase1_hrdev_evaluations/ordinary_full_comm256_lr3e-5-lr3e5seed1rep-task{00..05}_task{00..05}_s1_ss271829.json`.

Independent provenance, metric and ledger review is recorded in
[`docs/agents/REVIEW.md`](../../docs/agents/REVIEW.md). The result is a
development replication only; final confirmation, support redraws, the
few-shot curve, HR expert saturation and cost break-even remain outstanding.
