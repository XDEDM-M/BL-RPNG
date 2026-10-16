# Phase-1 confirmatory protocol

This file is append-only. Amendments must be appended before the affected data
role or result is observed; do not rewrite earlier entries.

## Initial freeze (2026-10-14)

- Campaign: `ac_phase1_confirmation_20261014`.
- Frozen task manifest: `artifacts/manifests/phase1_confirmation/ac_phase1_confirmation_20261014/tasks.json`, SHA-256 `bb9dcdf9d5e3c14ae658e74d08cd2b3bbc0fb45dd789e4e86957e722ba4aac5b`.
- The manifest has 48 unseen boundary descriptors, eight in each of the six established boundary families. It supports generalization to new boundaries within those families; it does not establish generalization to a previously unseen function family.
- Confirmation model seeds are `[0, 1, 2, 3, 42]`. Use at least three crossed support draws: `[271829, 271830, 271831]`. The primary support size is `m=4`; support subsets are nested from a frozen 16-rollout order for `m∈{1,2,4,8,16}`.
- LR support and LR validation use independent rollouts. All target checkpoint selection uses LR validation only. HR query initial conditions are independent of both LR roles and shared across candidate methods for paired comparisons.
- Primary query metric is the same 20-step rollout relative-L2 implementation used for HR development. Report all six family means, all 48 task values, paired candidate differences, worst task/family, failures, and 95% uncertainty intervals. Resample boundary tasks within family; model seeds and support draws are crossed repeated measures, not independent tasks.
- No query data may be generated, loaded, evaluated, or summarized until the complete selected method × model seed × support seed × task checkpoint matrix is frozen, the query evaluator and analysis code are hash-frozen, and the final matrix amendment below is appended. A locked generator enforces these conditions through `scripts/freeze_phase1_confirmation_query.py`.
- The development pilot will choose at most three methods by its predeclared rule: retain the best path-source candidate by six-family HR-development mean, the strongest ordinary-transfer baseline, and lifted LR scratch. Do not select model seeds or support draws from HR results. Exact candidate IDs/config hashes are to be appended after the pilot and before confirmation training.
- No threshold, candidate, exclusion, or metric may be changed after any confirmation HR query is opened. If a genuine protocol correction is required, append a dated amendment before query access and report it explicitly.

## Amendment log

Amendments below record the role counts and the revised model-seed count/method selection. The exact configuration hashes, frozen checkpoint matrix path, and matrix hash must still be appended here before the confirmation query gate can be unlocked.

## Amendment 1 — role sample counts and horizons (2026-10-14)

- Generate 16 true 64×64 LR support rollouts per task for each support seed `[271829, 271830, 271831]`, then form nested `m∈{1,2,4,8,16}` subsets from the frozen support-rank ordering.
- Generate 8 independent 64×64 LR validation rollouts per task and 8 independent 256×256 HR query rollouts per task. All three roles use 20 time steps. HR query generation must be a separate command after the final query gate is frozen.
- The final query generator requires the command's HR rollout count and horizon to match the frozen matrix; a command-line override cannot silently change the evaluation sample or horizon.

## Amendment 2 — model-seed count and selected methods (2026-10-15)

- Per the user's updated instruction, use the three already-run model seeds `[0, 1, 42]` for the confirmation matrix. Do not train or search additional model seeds. This supersedes only the initial five-model-seed list; it does not change the 48-task manifest, the three crossed support draws `[271829, 271830, 271831]`, `m=4`, the nested support ordering, independent LR validation/HR query roles, 20-step horizon, or query gate.
- The three selected comparison methods are `pilot_pathsource_comm256`, `pilot_ordinary_full_comm256`, and `pilot_lifted_scratch_lr3e-4_cos500_wd1e-4`, selected by the frozen Stage-2 development rule. Their confirmation-task configurations and code hashes still must be frozen in the checkpoint-matrix amendment before any HR query is generated.
- The reduced model-seed count narrows seed-level uncertainty. Treat model seeds and support draws as crossed repeated measures and boundary tasks as the generalization units; do not interpret the three model seeds as independent tasks or claim the precision of the original five-seed design.
- No confirmation support/query data or confirmation result has been observed when this amendment is recorded. HR query access remains locked until the complete selected method × model seed × support seed × task checkpoint matrix, evaluator, and analysis hashes are frozen.
