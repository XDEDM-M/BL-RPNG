# BL-RPNG Phase-1 final research report

**Disposition: `PARTIAL_BUDGET_EXHAUSTED` at the frozen full-experiment
granularity.** The required three optimization rounds, path-fine-tuning
comparisons, strong-baseline search and a predeclared second-seed replication
were actually run. They did not complete the frozen confirmation matrix,
few-shot curve, saturated HR expert, full cost comparison or mechanism
controls. The remaining conservative exploration allocation is too small for
another complete six-family experiment. No confirmation query/test data were
opened.

## Main result

The two-seed tuned ordinary baseline now has the lowest observed six-family
development mean, but it does not yet establish a stable advantage over the
path-source system. Stage-2's full three-seed comparison used ordinary
LR=`1e-4`; a subsequent three-point local LR bracket showed that this baseline
was not well tuned at that setting. At LR=`3e-5`, seed0 improved on all six
families and the predeclared seed1 replication improved the aggregate again.
Because the new candidate has only two seeds and one support draw, it cannot
replace the three-seed results or support a confirmatory claim.

| Candidate / setting | Model seeds | HR-dev mean relative L2 | Evidence scope |
|---|---:|---:|---|
| Path-source + target final-only NLT + comm256 (`pilot_pathsource_comm256`) | 3 | 0.05347209 | Stage-2, tasks 0–5, support seed 271829, `m=4` |
| Ordinary final-only source + full FT + comm256, LR=`1e-4` | 3 | 0.05488117 | Stage-2, tasks 0–5, support seed 271829, `m=4` |
| Ordinary final-only source + full FT + comm256, LR=`3e-5` | 2 | **0.04456778** | Matched seeds 0/1, tasks 0–5, support seed 271829, `m=4` |
| Lifted LR scratch, 500 epochs, LR=`3e-4`, WD=`1e-4` | 3 | 0.50904359 | Stage-2, tasks 0–5, `m=4` |

On the matched seed0/seed1 subset, LR=`3e-5` is 20.86% below ordinary
LR=`1e-4` (`0.05631693`) and 17.75% below path-source C256 (`0.05418312`). The
first comparison shows that baseline tuning is material. The second is not a
three-seed comparison, and the path-source method also uses a different
adaptation strategy. The frozen T2 requirement—at least 15% lower error than a
fully tuned ordinary-transfer baseline with a paired 95% interval on
confirmation tasks—remains **NOT_MET**.

### Seed1 replication by development family

Each cell below is the 256×256, 20-step HR-dev mean over eight rollouts. LR
checkpoints were selected using 64×64 LR-dev validation before any HR-dev
evaluation.

| Family | LR-dev, seed1 `3e-5` | HR-dev, seed1 `3e-5` | Paired ordinary seed1 `1e-4` | Paired path-source C256 seed1 |
|---|---:|---:|---:|---:|
| High-frequency | 0.02517443 | 0.05992937 | 0.04924395 | 0.05184649 |
| Amplitude-OOD | 0.02966063 | 0.03002297 | 0.04480050 | 0.02836940 |
| Matérn GP | 0.02405344 | 0.03127426 | 0.04595283 | 0.04754582 |
| Localized bump | 0.02605149 | 0.05221141 | 0.05564401 | 0.06014429 |
| Mixed Fourier | 0.02521360 | 0.04919595 | 0.07098463 | 0.06478832 |
| Piecewise smooth | 0.02438459 | 0.04382814 | 0.05251610 | 0.05333420 |
| **Equal-family mean** | **0.02575636** | **0.04441035** | **0.05319034** | **0.05100475** |

The LR=`3e-5` arm is lower than the paired seed1 ordinary LR=`1e-4` arm on
five of six families; high-frequency is worse. On the two-seed matched subset,
mean error is lower for LR=`3e-5` on all six family averages. This remains a
small development panel: two model seeds, one support draw and six fixed
families. No confidence interval is presented as task-generalization evidence.

Full baseline replication report and machine-readable aggregate:
[`PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md`](../artifacts/reports/PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md),
[`phase1_r5_lr3e5_seed1_replication_summary_20261016.json`](../artifacts/reports/phase1_r5_lr3e5_seed1_replication_summary_20261016.json).
The seed0 local bracket, including the high-side failure at `3e-4`, is in
[`PHASE1_BASELINE_LR3E5_SCREEN_20261016.md`](../artifacts/reports/PHASE1_BASELINE_LR3E5_SCREEN_20261016.md)
and [`PHASE1_BASELINE_LR3E4_SCREEN_20261016.md`](../artifacts/reports/PHASE1_BASELINE_LR3E4_SCREEN_20261016.md).

## Path information and mechanism

Real target accepted-Newton-path fine-tuning was implemented and run from the
same path-source initialization. The six-task, three-seed Stage-2 means were
first-path A1 `0.05409390`, endpoint A2 `0.05402562`, and endpoint-direction
A3 `0.05423021`; the corresponding C256 candidate was `0.05347209`. A1 was
0.126% worse than A2 and 0.251% better than A3; these small differences varied
by seed. The focused R4 A1/A2 + comm256 interaction did not meet its
predeclared improvement/repeatability rule. There is no complete A0/A1/A2/A3/A4
matrix and no confirmation-level causal evidence. The evidence supports
transfer as a useful system component; it does not show that target Newton
path supervision adds a repeatable benefit. See
[`PHASE1_STAGE2_SUMMARY_20261015.md`](../artifacts/reports/PHASE1_STAGE2_SUMMARY_20261015.md),
[`PHASE1_R4_PATHCOMM_20261016.md`](../artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md),
and [`docs/agents/MECHANISM.md`](../docs/agents/MECHANISM.md).

## HR expert gap

The four-task HR-from-zero scratch `m=4` study used 500 epochs and reached
mean HR-dev error `0.4421034`, versus `0.0561567` for its matched path-source
reference. The nested `m=16` scratch supplement used 500 epochs, had mean
`0.1642690`, and was still `2.925×` the matched path-source mean. A
source-initialized HR full-FT `m=16`, 250-epoch study averaged `0.0706924`,
25.88% above the matched path-source mean, and missed each of its four task
references. These results do not establish expert saturation and are not an
oracle comparison. Details and costs are in
[`PHASE1_HR_SCRATCH_END_TO_END_20261015.md`](../artifacts/reports/PHASE1_HR_SCRATCH_END_TO_END_20261015.md),
[`PHASE1_HR_SCRATCH_M16_END_TO_END_20261016.md`](../artifacts/reports/PHASE1_HR_SCRATCH_M16_END_TO_END_20261016.md),
and [`PHASE1_HR_FULLFT_M16_EXPERT_END_TO_END_20261016.md`](../artifacts/reports/PHASE1_HR_FULLFT_M16_EXPERT_END_TO_END_20261016.md).

## Cost and remaining budget

The seed0 LR=`3e-5` six-task fit plus HR-dev cost `0.928908 GPU-h`. Its seed1
replication cost `0.930489 GPU-h` (`0.926292` training + `0.004197` HR-dev).
The fixed support, validation and HR-dev data were reused, so this replication
generated no new PDE labels. These totals exclude shared source pretraining
and earlier hyperparameter search. Other target-generation, HR scratch and
source costs are separately detailed in the linked reports. Direct HR solver
and neural HR-inference timing were not measured on a matched repeated
protocol, so a solver break-even point and end-to-end cost advantage cannot
be stated.

The corrected terminal budget snapshot is
[`phase1_progress_after_seed1_hrdev_complete_20261016.json`](../artifacts/reports/phase1_progress_after_seed1_hrdev_complete_20261016.json):
`71.866022 GPU-h` charged, `0.133978 GPU-h` nominal exploration remainder,
and the protected 48 GPU-h reserve intact. The ledger excludes historical
untimed HR-dev executions; a conservative estimate for these leaves
`0.113303 GPU-h` for exploration. A complete six-family baseline screen costs
about `0.93 GPU-h`, so further full-scope baseline search does not fit.

The frozen three-method, 48-task, three-seed primary confirmation matrix
projects to about `158.2 GPU-h`; current total nominal remainder is only
`48.134 GPU-h`, including the protected reserve, for a shortfall of at least
`110.066 GPU-h`. Adding the representative few-shot curve raises the known
subtotal to `187.7 GPU-h`, short by `139.566 GPU-h`, before expert saturation,
mechanism controls, confirmation-scale CPU label generation and review. The
query gate remains closed; no query/test data were generated or read.

## Reproducibility and audit

The seed1 protocol/config, six per-task train/evaluate commands, environment,
source/config/data/code hashes and raw artifact paths are recorded in the
replication report and frozen experiment card. The independent reviewer
recomputed the six task histories, LR minima, HR metrics, checkpoint/index
seals, query flags, ledger charges and final snapshot; see
[`docs/agents/REVIEW.md`](../docs/agents/REVIEW.md). The campaign's current
best-config record is [`docs/BEST_CONFIG.md`](../docs/BEST_CONFIG.md); it
labels the low-LR setting as a development candidate, not a confirmed
champion. The full itemized completion checklist is
[`docs/PHASE1_COMPLETION.md`](../docs/PHASE1_COMPLETION.md).
