# Phase-1 completion evidence (2026-10-16)

**Disposition: `PARTIAL_BUDGET_EXHAUSTED` at the frozen full-experiment granularity.**
The planned optimization rounds and the budget-feasible baseline search have
been run. The core confirmation, expert, few-shot, cost and full mechanism
criteria remain incomplete; this report does not label Phase-1 scientifically
complete. `NOT_MET` means the required evidence is absent, not that the
hypothesis was disproved.

| Item | Status | Evidence and current limit |
|---|---|---|
| Trustworthy starting point | DONE | Repository/data/provenance audits: `docs/audit/REPO_MAP.md`, `artifacts/reports/phase1_r1_input_audit_resume2_20261013.json`, and the independently audited Stage-2 results. Target LR support is genuine 64×64 solve data; HR-dev validation is independent; confirmation query remains gated. |
| Effect optimization | DONE | R1, R2 and R3 each produced executed improvement/search evidence; Stage-2, R4 interaction and R5 ordinary-baseline LR bracket/replication are complete. Decision cards and raw records are in `docs/experiment_cards/`, `docs/EXPERIMENTS.csv`, and `docs/LOG.md`. This satisfies the required optimization rounds, not a positive main claim. |
| True path fine-tuning | DONE | Same path-source initialization was compared under final-only and real target accepted-path supervision in R1 and the Stage-2/R4 controls. The target path-specific gain was not stable: R4 A1 did not beat endpoint A2 consistently. Development evidence only; see `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md` and `docs/agents/MECHANISM.md`. |
| Strong baselines | DONE | Ordinary full-FT + comm256 received a three-seed Stage-2 run and a local `{3e-5,1e-4,3e-4}` LR bracket. The promising `3e-5` setting was replicated on a second predeclared seed. The two-seed mean `0.04456778` is below the matched two-seed LR=`1e-4` mean `0.05631693`; it cannot replace the three-seed aggregates. LR scratch also received 500-epoch cosine/weight-decay screens. Evidence: `artifacts/reports/PHASE1_BASELINE_LR3E5_SCREEN_20261016.md` and `artifacts/reports/PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md`. |
| Confirmation main table | NOT_MET | Frozen protocol requires the full 48-task × 3-model-seed `[0,1,42]` × 3-crossed-support-seed matrix for the selected methods. No complete confirmation checkpoint matrix or query outputs exist. The projected primary costs about `158.2 GPU-h`, beyond available total capacity. Protocol: `docs/prereg/CONFIRMATORY.md`. |
| HR experts | NOT_MET | Four-task HR scratch `m=4`/500-epoch and `m=16`/500-epoch runs plus a source-initialized HR full-FT `m=16`/250-epoch study are complete. They are one seed and do not establish saturation; neither HR-from-scratch budget reached its matched path-source reference. Reports: `artifacts/reports/PHASE1_HR_SCRATCH_END_TO_END_20261015.md`, `artifacts/reports/PHASE1_HR_SCRATCH_M16_END_TO_END_20261016.md`, and `artifacts/reports/PHASE1_HR_FULLFT_M16_EXPERT_END_TO_END_20261016.md`. |
| Few-shot curve | NOT_MET | Most main comparisons use `m=4`. The m=16 scratch supplement is not the nested `m∈{1,2,4,8,16}` curve across path method, strongest ordinary baseline and LR scratch. |
| Cost | NOT_MET | GPU training, target adaptation, HR-dev and selected CPU data-generation costs are measured. Direct HR solver versus HR inference timing, complete end-to-end workflow costs, lifecycle break-even and a full expert workflow remain unmeasured; historical source HR-data generation time is unavailable. See `docs/BUDGET.md` and the HR scratch reports. |
| Mechanism | NOT_MET | Stage-2/R4 compare first accepted path, endpoint and endpoint-direction targets; differences are small/inconsistent and no A0/A4 full matrix exists. The path-source C256 winner does not isolate target Newton-path fine-tuning. See `docs/agents/MECHANISM.md` and `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md`. |
| Review | DONE | Independent reviewers recomputed Stage-2 aggregates, the R5 LR screens and seed1 replication, HR expert reports, selection/provenance, query flags, and corrected ledger totals. Findings: `docs/agents/REVIEW.md`. Scope remains development artifacts; no confirmation result was available to review. |
| Reproduction | DONE | The current low-LR ordinary development candidate has frozen YAML/card, per-task selected checkpoints, reproducible train/evaluate commands, hashes, and a summary script: `configs/phase1_baseline_fullcomm256_lr3e-5_seed1_replication.yaml`, `scripts/summarize_phase1_r5_lr3e5_replication.py`, and `artifacts/reports/PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md`. It is not a confirmed champion. |

## Remaining budget and stopping basis

- Corrected terminal ledger: `71.866022 GPU-h` of the 120 GPU-h campaign cap.
  Nominal total remainder is `48.133978 GPU-h`: `0.133978 GPU-h` exploration
  plus the protected `48 GPU-h` confirmation/control reserve.
- The ledger omits some historical untimed HR-dev executions. Estimating those
  at the maximum observed timed report cost gives `0.020675 GPU-h` and
  `0.113303 GPU-h` conservative exploration headroom. This is an estimate,
  not a ledger charge. A complete six-family LR+HR screen costs about
  `0.93 GPU-h`, so no further full baseline search fits without taking from
  the protected reserve.
- The amended three-method primary confirmation matrix projects to about
  `158.2 GPU-h`, a minimum `110.066 GPU-h` shortfall against all nominal total
  remainder. The primary plus the representative few-shot curve is about
  `187.7 GPU-h`, a `139.566 GPU-h` shortfall, before expert saturation,
  mechanism controls, confirmation-scale CPU data generation and final review.
  Detailed assumptions and the corrected attempt-level ledger are in
  `docs/BUDGET.md`.
- No GPU task is active. The 48-hour reserve stays intact and the confirmation
  query gate stays closed. The remaining `0.113 GPU-h` cannot complete any
  frozen full-family search; no single-task or shortened-schedule result is
  substituted for the predeclared evidence.

## Required final deliverables

- Scientific summary and evidence limits: `reports/PHASE1_FINAL.md`.
- Direction and claim decision: `docs/DECISION_MEMO.md`.
- Reproducible development candidate: `docs/BEST_CONFIG.md`.
- Frozen completion and state: this file and `docs/STATE.md`.
- Run history and complete append-only trial ledger: `docs/LOG.md` and
  `docs/EXPERIMENTS.csv`.
