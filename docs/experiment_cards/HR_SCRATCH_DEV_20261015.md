# Direct HR scratch baseline on frozen Stage-2 support

## Question

Does a randomly initialized FNO trained on the target's 256×256 rollouts reach the same independent HR-development error as the selected LR-adaptation path, and what is the measured end-to-end cost after charging the HR support-label generation?

## Frozen inputs and scope

- Development tasks: `[0, 1, 3, 5]` (`high_frequency`, `amplitude_ood`, `localized_bump`, `piecewise_smooth`), the mission's four preselected representative tasks. GPU1 starts task 5. After the frozen Stage-2 queue drains, GPU0 runs only task 3 via a physical-GPU/task scheduling override; it uses the same config, seed, support and validation indices. GPU1's serial runner verifies and skips task 3 when it reaches it. The two active HR scratch jobs therefore cover different boundary families.
- Model seed: `0`; support draw: `271829`; support count: `m=4` per task. No additional seed or support redraw.
- Each HR support case copies the exact IC coefficients, phase, case ID, and boundary from the Stage-2 LR support manifest. It is solved at 256×256 with the frozen 257-mesh, 20-step, fidelity-1 Allen–Cahn protocol.
- Training starts from random initialization, uses the base FNO/representation, full parameter updates, the frozen scratch settings (500 cosine epochs, LR `3e-4`, weight decay `1e-4`), batch size 2 at HR, and no source checkpoint.
- The independent eight-rollout 256×256 HR development index selects the best epoch. The same HR-development index is the reported development metric; this is a development comparison, not confirmation evidence.
- The query gate remains closed. No confirmation query/test data is generated or read.

## Cost accounting

Record separately:

1. CPU end-to-end HR support generation wall time, including manifest/index validation and packaging.
2. Solver-only time summed from the 16×20 HR support transitions and the separately recorded assembly time.
3. Per-task GPU time for the four HR scratch fits, including HR validation and checkpoint compaction.
4. For the LR-adaptation path, target LR adaptation GPU time and the already measured LR support generation wall time. Report source pretraining separately, and show both unamortized and task-amortized cost where traceable.
5. Compare task-paired HR-development error first. Any speedup is reported only for comparable error; do not combine CPU seconds and GPU-hours into one ratio without an explicit wall-clock schedule.

## Parallel dispatch audit

The config and data splits remain unchanged. The initial runner source snapshot is `artifacts/reports/phase1_hr_scratch_runner_at_initial_dispatch.py` (SHA-256 `1221ae5e...1c7822`). The task-splitting runner at task03 dispatch was `scripts/run_phase1_hr_scratch_dev.py` (SHA-256 `0c212440...74144d`); task03 records that runner hash. Its scheduling additions were explicit physical-GPU and frozen-task selection flags. Per-task reports share the canonical run namespace. After all fits completed, summary aggregation first stopped on a field-name mismatch: the saved task report uses `config_sha256`, while the runner checked `descriptor_sha256`. The skip-path check was corrected (current runner SHA-256 `80a09f40...0cb5982`) and the runner reused all four reports without retraining to write the canonical summary.

## Limits

This is one `m=4` point on four development tasks, with one model seed and one support draw. It does not establish expert saturation, an HR support-count curve, or confirmation performance. HR-development checkpoint selection can favor the scratch model on this same development metric; query data remains required for a paper-level claim.

## Completed comparison

All four frozen 500-epoch fits are complete. Best HR-dev errors are task00 `0.640716` (epoch150), task01 `0.300194` (epoch159), task03 `0.485944` (epoch229), and task05 `0.341559` (epoch324); none reached the paired path-source reference. Their total training cost is `2.642104` GPU-h. Generating and packaging the 16 HR support rollouts cost `1187.548 s` CPU wall; all `320/320` solver transitions converged.

The paired, by-family comparison, path-source training/adaptation cost, shared LR-label cost, source-data-generation caveat, time-to-match status, and evidence-limited conclusion are in `artifacts/reports/PHASE1_HR_SCRATCH_END_TO_END_20261015.md` and its JSON companion. The canonical run summary is `results/phase1/HRSCRATCHDEV/hr_scratch_from_zero_hr256_m4_s0_ss271829/summary.json`. No extra seed or query/test access was used.
