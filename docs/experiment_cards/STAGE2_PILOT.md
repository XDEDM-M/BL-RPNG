# Stage 2 six-family, three-seed shortlist pilot

- **Question:** Does the path-source, few-shot LR adaptation pipeline retain its HR advantage across every development boundary family and three model seeds, and does genuine target path supervision add value over matched endpoint controls?
- **Development-only candidates:** path-source NLT with direct 64-to-256 consistency; path-source NLT with first-accepted target path supervision; matched endpoint-increment and endpoint-direction controls; the strongest measured ordinary baseline (ordinary HR source + full target FT + direct256 consistency); and lifted LR scratch with 500-epoch cosine and weight decay.
- **Frozen families/tasks:** the six sealed development tasks 0–5, one task per family, m=4 LR support, support seed 271829. These tasks are development-only and cannot serve as final confirmation tasks.
- **Model seeds:** 0, 1, and 42, with source pretraining performed from random initialization for each objective/seed pair using the frozen HR source train/validation data. Target adaptations use matching source and adaptation seeds.
- **Checkpoint selection:** LR validation only for every target task. HR development is evaluated once per selected task checkpoint, after the complete LR screen. No query/test data is used.
- **Primary summary:** unweighted mean across the six family-specific 20-step HR-dev rollout relative-L2 values; report paired per-family values, model-seed dispersion and worst-family result. Mechanism inference compares first-path A1 against endpoint A2 and direction A3 under the same source, layer, schedule, seed, support and task.
- **Configuration:** `configs/phase1_stage2_pilot.yaml`. Run serially on GPU0. Source-model jobs and target adaptation jobs share the same GPU lock/queue and run one at a time.
- **Budget:** six source pretrains × 60 epochs, plus six arms × three model seeds × six tasks. No exploratory hyperparameter changes inside this pilot.
- **Acceptance:** all 18 task-family cells per arm have durable reports/checkpoints, finite histories, unique LR-validation minima, valid source/support/validation/checkpoint hashes and query-free records. Any failed or interrupted task remains excluded and is rerun in a fresh immutable namespace.
- **Promotion:** choose at most three candidates using the frozen six-family HR-dev summary, preserving the strongest path-source method and ordinary/scratch baselines for the independent confirmation. Do not select model or support seeds based on HR values.

## Execution status

- Frozen before dispatch. R2C independent provenance review passed and is recorded in `docs/agents/REVIEW.md`.
- Seed0/path and seed0/final from scratch completed 60 epochs; best source-HR validation `0.007275705315` / `0.007762998303`, charges `2.068093906690` / `1.402507711082` GPU-h, checkpoint SHAs `b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf` / `46e5486749f01e12fe1eac86398970168337c880b8bfe9370066f1682273d554`, and `query_used=false`. Seed1/path attempt1 and retry1 were interrupted and excluded (conservative charges `0.183888888889` and `1.500833333333` GPU-h); the epoch39 full-state recovery completed 60 epochs. Seed1/final has since completed 60 epochs. See terminal records below. Stage-2 LR adaptation is active. Seed0/pathsource_comm256 and seed0/pathsource_firstpath completed all six task cells; their query-free summaries are recorded below. The canonical seed0 endpoint-increment arm failed at task01 inside Adam multi-tensor update and is excluded; a fresh immutable retry1 namespace is being used for all six cells on each of the same model seeds [0,1,42]. The frozen LR matrix audit and HR-dev evaluations start only after all accepted LR cells finish.
- Before target training, CPU-only provenance preflight passed all six pilot tasks: m=4 support, eight 64²/20-step LR validation rollouts, correct task/boundary/role/manifest bindings, unique ICs within each split, and no support/validation overlap. The runner now enforces the frozen validation count and split-internal IC uniqueness for S2PILOT/CONFIRM.
- Confirmation query authorization is shared by the wrapper and SfePy worker; direct worker invocation checks the full gate, canonical case-manifest path/seal, deterministic IC construction, NumPy version, full solver parameters, source/model/data hashes, and frozen generator/freezer/worker/runtime code before solving. Static syntax and whitespace checks passed; no confirmation query was generated.
- The worker now restricts HR generation to the exact pinned development HR manifest/task hashes or the gated confirmation query manifest, blocking rewritten self-sealed task/case manifests and copied/re-encoded unregistered cases. The query freezer will fail closed on pre-existing canonical query index/shards or wrapper temp outputs and record the absence check. The frozen matrix now pins solver Python/NumPy/SciPy/SfePy and BLAS/LAPACK build fingerprint. The worker routes confirmation cases regardless of split/fidelity labels, compares copied boundaries on exact LR/HR FE edge nodes, rejects non-LR HR-like requests, and requires the pinned HR protocol for development HR. Independent static re-review found no simple copy/relabel/resolution/Fourier alias bypass with the worker code fixed. Syntax/diff checks pass; query data remains ungenerated.
- Source runs now take the same nonblocking `phase1_training_gpu0` advisory lock as target training and HR-dev evaluation.
- Independent review found and fixed the runner's `S2PILOT` namespace rejection. Confirmation query generation now fails closed on manifest-root mismatch and requires a hash-frozen full confirmation checkpoint matrix, preregistration, evaluator, and analysis code; see `docs/prereg/CONFIRMATORY.md` and the two freeze scripts.

- A separate 48-boundary confirmation manifest is sealed at `artifacts/manifests/phase1_confirmation/ac_phase1_confirmation_20261014/tasks.json`. The generation helper now requires explicit confirmation roles and routes the campaign to `artifacts/data/phase1_confirmation`; confirmation `validation_hr` data is the final query role and will remain ungenerated until checkpoints are frozen.


Interruption record (2026-10-14 13:47 Asia/Shanghai): controller and worker were absent at process check; no exception/audit or system journal entry established the cause. The epoch-3 progress snapshot is hash-bound in the incident JSON. The retry uses the same frozen source data, config, seed, and 60-epoch/patience schedule with a distinct immutable output namespace; the interrupted attempt remains excluded.


Second interruption / recovery (2026-10-14 15:18 Asia/Shanghai): retry1 process disappeared after epoch41 progress without terminal report. Recovery checkpoint captured epoch39, best source validation `0.013992019284`, 100,000 steps, 40 epochs; SHA-256 `629194c90e10e981825f532244248ebe550894ac1166fe150c7b642baa75433e`. Retried in `resume_from_scratch` mode so no unrelated warm-start source checkpoint is falsely attributed; full-state resume goes to a new artifact namespace.


Detached continuation (2026-10-14 15:21 Asia/Shanghai): queue supervisor PID `418560` runs `artifacts/workflows/phase1_stage2_complete.sh`; current recovery worker PID `418565` resumed epoch39 in the immutable `resume_from_scratch` namespace and reached epoch42/60, best source-HR validation `0.012975909679`, latest `0.015169511166`, 107,500 steps, 377.8 s in this attempt, stale1. The script serializes all remaining source models and the frozen three-seed LR pilot, audits all 108 unique LR-validation minima and checkpoint/index hashes, then evaluates HR-dev only after that full matrix passes. Per-step logs and queue status are under `artifacts/logs/phase1_stage2_queue/`.


### Queue status snapshot (2026-10-14 15:29 Asia/Shanghai)

The previous `retry1` process disappeared after epoch41 and is excluded. Its epoch39 full-state snapshot was resumed in a new immutable `resume_from_scratch` attempt. Detached supervisor PID `418560` / worker PID `418565` are live; the recovery attempt reached epoch42/60, best source-HR validation `0.012975909679`, latest `0.015169511166`, 107,500 cumulative optimizer steps, 377.8 s measured training time, stale1. Durable progress is `artifacts/reports/progress/source_path_s1_resume_scratch_recovery1.json`. Queue order is seed1/final, seed42 path/final, three frozen LR pilot screens, full LR matrix audit, then HR-dev.


### Latest queue status (2026-10-14 15:33 Asia/Shanghai)

The recovery attempt reached epoch44/60 and improved the source-HR validation best to `0.010447250983` (latest equals best), with 112,500 cumulative optimizer steps, 629.7 s measured training time, and stale0. Queue supervisor PID `418560` and worker PID `418565` remained live; the three remaining source pretrains and all target adaptation/HR-dev work had not started. The new confirmation query gate remains unopened.


Queue progress (2026-10-14 15:35 Asia/Shanghai): recovery reached epoch45/60, latest source-HR validation `0.011151085810`, best `0.010447250983`, 115,000 cumulative steps, 756.6 s measured training time, stale1. The mechanism review in `docs/agents/MECHANISM.md` clarifies that the frozen A1/A2/A3 target-path comparison has no zero-auxiliary A0 and cannot alone identify path FT versus final-only; preserve this limitation in pilot interpretation.


### Latest queue status (2026-10-14 15:56 Asia/Shanghai)

Seed1/path recovery reached epoch55/60; best source-HR validation `0.008093109012` at epoch54, latest `0.008355821932`, 140,000 cumulative optimizer steps, 2020.1 s measured resume time, stale1. Worker PID `418565` remains under queue supervisor PID `418560`. Queue still has seed1/final and seed42 path/final pretraining, then the frozen 108-cell LR pilot and its 108 HR-dev evaluations; target adaptation has not started.


### Seed1/path recovery terminal (2026-10-14 16:04 Asia/Shanghai)

The epoch39 full-state recovery completed the frozen 60-epoch source run. Best source-HR validation is `0.007934059578` at epoch58; measured charge `0.704184163002` GPU-h; checkpoint SHA-256 `eb4863d41026adc92bcda4ae421feff6609baa005f6818e135e533ce29bd3770`; no query used. Ledger RUNNING/DONE rows are 426/427. The queue verified and wrote the canonical alias in `artifacts/reports/seed1_path_source_alias.json`, then started seed1/final from scratch (PID `420652`).


### Seed1/final source terminal (2026-10-14 17:28 Asia/Shanghai)

The seed1/final source run completed 60 epochs. Best source-HR validation `0.008050473349` at epoch59; charge `1.401436295762` GPU-h; checkpoint SHA-256 `8b9509f482b2029c0c12302449c93509762991e72a3343978a2bd2b55273507b`; query unused. Ledger rows 428/429 record RUNNING/DONE. Queue PID `418560` then began seed42/path source training (PID `431820`).

### Execution boundary confirmed (2026-10-14)

The user confirmed continuation of this frozen Stage-2 plan with no added model seeds or temporary seed search. Preserve its configurations and data splits, finish source pretraining → planned target LR adaptation → full matrix audit → HR-dev, then report by seed and boundary family and assess the paper claims. No additional seed experiment starts before the user reviews that report.


Queue status (2026-10-14 17:37 Asia/Shanghai): seed42/path from-scratch is active under supervisor PID `418560` / worker PID `431820`, epoch index 3/60, best source-HR validation `0.076835690560`, latest `0.093235848343`, 10,000 steps. Run-specific progress: `artifacts/reports/progress/source_path_s42_scratch.json`; measured active time is not terminally charged. Stage-2 LR adaptation has not started.

Queue progress (2026-10-14 17:39 Asia/Shanghai): seed42/path from-scratch reached epoch index 4/60 and improved best/latest source-HR validation to `0.072543017093`; 12,500 optimizer steps and 617.7 s measured training time. Supervisor PID `418560` / worker PID `431820` remain active on GPU0.

Queue progress (2026-10-14 17:41 Asia/Shanghai): seed42/path from-scratch reached epoch index 5/60 and improved best/latest source-HR validation to `0.067776328371`; 15,000 optimizer steps and 742.0 s measured training time. Supervisor PID `418560` / worker PID `431820` remain active.

Queue progress (2026-10-14 17:43 Asia/Shanghai): seed42/path reached epoch index 6/60, best source-HR validation `0.067776328371`, latest `0.080627630841`, 17,500 steps, 865.1 s, stale1. Queue remains active under PID `418560` / worker `431820`.

Queue progress (2026-10-14 17:45 Asia/Shanghai): seed42/path from scratch reached epoch index 7/60 and improved best/latest source-HR validation to `0.046958894125`; 20,000 optimizer steps, 989.7 s training time, stale0.

Queue progress (2026-10-14 17:47 Asia/Shanghai): seed42/path reached epoch index 8/60; best source-HR validation `0.046958894125`, latest `0.056221514360`, 22,500 steps, 1,115.8 s, stale1. Worker PID `431820` remains active under queue supervisor `418560`.

Queue progress (2026-10-14 17:49 Asia/Shanghai): seed42/path reached epoch index 9/60; best source-HR validation `0.046958894125`, latest `0.074478857358`, 25,000 steps, 1,240.4 s, stale2. Worker PID `431820` remains active.

Queue progress (2026-10-14 17:51 Asia/Shanghai): seed42/path reached epoch index 10/60 and improved best/latest source-HR validation to `0.041768468455`; 27,500 steps, 1,366.4 s, stale0.

Queue progress (2026-10-14 17:53 Asia/Shanghai): seed42/path reached epoch index 11/60; best source-HR validation `0.041768468455`, latest `0.043707700368`, 30,000 steps, 1,490.6 s, stale1. Worker PID `431820` remains active.

Queue progress (2026-10-14 17:55 Asia/Shanghai): seed42/path reached epoch index 12/60; best source-HR validation `0.041768468455`, latest `0.051025932006`, 32,500 steps, 1,615.4 s, stale2.

Queue progress (2026-10-14 17:57 Asia/Shanghai): seed42/path reached epoch index 13/60; best source-HR validation `0.041768468455`, latest `0.042073625668`, 35,000 steps, 1,739.7 s, stale3.

Queue progress (2026-10-14 17:59 Asia/Shanghai): seed42/path reached epoch index 14/60; best source-HR validation `0.041768468455`, latest `0.042915275911`, 37,500 steps, 1,863.9 s, stale4.

Queue progress (2026-10-14 18:01 Asia/Shanghai): seed42/path reached epoch index 15/60 and improved best/latest source-HR validation to `0.038299596927`; 40,000 steps, 1,988.7 s, stale0.

Queue progress (2026-10-14 18:04 Asia/Shanghai): seed42/path reached epoch index 16/60; best source-HR validation `0.038299596927`, latest `0.045637392704`, 42,500 steps, 2,113.1 s, stale1.

Queue progress (2026-10-14 18:06 Asia/Shanghai): seed42/path reached epoch index 17/60; best source-HR validation `0.038299596927`, latest `0.038761919239`, 45,000 steps, 2,237.4 s, stale2.

Queue progress (2026-10-14 18:08 Asia/Shanghai): seed42/path reached epoch index 18/60 and improved best/latest source-HR validation to `0.034888776215`; 47,500 steps, 2,361.1 s, stale0.

Queue progress (2026-10-14 18:10 Asia/Shanghai): seed42/path reached epoch index 19/60 and improved best/latest source-HR validation to `0.033169452250`; 50,000 steps, 2,485.7 s, stale0.

Queue progress (2026-10-14 18:11 Asia/Shanghai): seed42/path reached epoch index 20/60 and improved best/latest source-HR validation to `0.031992271335`; 52,500 steps, 2,611.8 s, stale0.

Queue progress (2026-10-14 18:13 Asia/Shanghai): seed42/path reached epoch index 21/60 and improved best/latest source-HR validation to `0.030164192409`; 55,000 steps, 2,736.5 s, stale0.

Queue progress (2026-10-14 18:16 Asia/Shanghai): seed42/path reached epoch index 22/60; best source-HR validation `0.030164192409`, latest `0.034589929610`, 57,500 steps, 2,860.5 s, stale1.

Queue progress (2026-10-14 18:18 Asia/Shanghai): seed42/path reached epoch index 23/60 and improved best/latest source-HR validation to `0.027674669798`; 60,000 steps, 2,985.2 s, stale0.

Queue progress (2026-10-14 18:22 Asia/Shanghai): seed42/path reached epoch index 25/60; best source-HR validation `0.027674669798`, latest `0.032190485535`, 65,000 steps, 3,234.6 s, stale2.

Queue progress (2026-10-14 18:24 Asia/Shanghai): seed42/path reached epoch index 26/60; best source-HR validation `0.027674669798`, latest `0.029475475527`, 67,500 steps, 3,359.0 s, stale3.

Queue progress (2026-10-14 18:26 Asia/Shanghai): seed42/path reached epoch index 27/60 and improved best/latest source-HR validation to `0.026572181578`; 70,000 steps, 3,483.3 s, stale0.

Queue progress (2026-10-14 18:28 Asia/Shanghai): seed42/path reached epoch index 28/60 and improved best/latest source-HR validation to `0.020557355141`; 72,500 steps, 3,607.5 s, stale0.

Queue progress (2026-10-14 18:30 Asia/Shanghai): seed42/path reached epoch index 29/60 and improved best/latest source-HR validation to `0.019237580123`; 75,000 steps, 3,731.9 s, stale0.

Queue progress (2026-10-14 18:32 Asia/Shanghai): seed42/path reached epoch index 30/60 and improved best/latest source-HR validation to `0.019006389616`; 77,500 steps, 3,858.6 s, stale0.

Queue progress (2026-10-14 18:34 Asia/Shanghai): seed42/path reached epoch index 31/60; best source-HR validation `0.019006389616`, latest `0.028105755996`, 80,000 steps, 3,983.2 s, stale1.

Queue progress (2026-10-14 18:37 Asia/Shanghai): seed42/path reached epoch index 32/60; best source-HR validation `0.019006389616`, latest `0.020950556179`, 82,500 steps, 4,107.6 s, stale2.

Queue progress (2026-10-14 18:39 Asia/Shanghai): seed42/path reached epoch index 33/60 and improved best/latest source-HR validation to `0.016983818134`; 85,000 steps, 4,232.2 s, stale0.

Queue progress (2026-10-14 18:41 Asia/Shanghai): seed42/path reached epoch index 34/60; best source-HR validation `0.016983818134`, latest `0.018801517975`, 87,500 steps, 4,357.4 s, stale1.

Queue progress (2026-10-14 18:43 Asia/Shanghai): seed42/path reached epoch index 35/60 and improved best/latest source-HR validation to `0.016860493390`; 90,000 steps, 4,481.4 s, stale0.

Queue progress (2026-10-14 18:45 Asia/Shanghai): seed42/path reached epoch index 36/60; best source-HR validation `0.016860493390`, latest `0.017589007859`, 92,500 steps, 4,605.7 s, stale1.

Queue progress (2026-10-14 18:47 Asia/Shanghai): seed42/path reached epoch index 37/60; best source-HR validation `0.016860493390`, latest `0.022223603209`, 95,000 steps, 4,729.6 s, stale2.

Queue progress (2026-10-14 18:50 Asia/Shanghai): seed42/path reached epoch index 38/60 and improved best/latest source-HR validation to `0.015057366399`; 97,500 steps, 4,854.0 s, stale0.

Queue progress (2026-10-14 18:51 Asia/Shanghai): seed42/path reached epoch index 39/60; best source-HR validation `0.015057366399`, latest `0.015066730943`, 100,000 steps, 4,978.3 s, stale1.

Queue progress (2026-10-14 18:53 Asia/Shanghai): seed42/path reached epoch index 40/60 and improved best/latest source-HR validation to `0.011955843147`; 102,500 steps, 5,104.3 s, stale0.

Queue progress (2026-10-14 18:55 Asia/Shanghai): seed42/path reached epoch index 41/60 and improved best/latest source-HR validation to `0.011749192793`; 105,000 steps, 5,228.6 s, stale0.

Queue progress (2026-10-14 18:57 Asia/Shanghai): seed42/path reached epoch index 42/60; best source-HR validation `0.011749192793`, latest `0.012587298823`, 107,500 steps, 5,353.0 s, stale1.

Queue progress (2026-10-14 19:01 Asia/Shanghai): seed42/path reached epoch index 44/60 and improved best/latest source-HR validation to `0.011328501280`; 112,500 steps, 5,602.7 s, stale0.

Queue progress (2026-10-14 19:04 Asia/Shanghai): seed42/path reached epoch index 45/60 and improved best/latest source-HR validation to `0.010356816802`; 115,000 steps, 5,727.4 s, stale0.

Queue progress (2026-10-14 19:06 Asia/Shanghai): seed42/path reached epoch index 46/60; best source-HR validation `0.010356816802`, latest `0.010978513507`, 117,500 steps, 5,852.0 s, stale1.

Queue progress (2026-10-14 19:08 Asia/Shanghai): seed42/path reached epoch index 47/60 and improved best/latest source-HR validation to `0.009532859863`; 120,000 steps, 5,976.6 s, stale0.

Queue progress (2026-10-14 19:10 Asia/Shanghai): seed42/path reached epoch index 48/60 and improved best/latest source-HR validation to `0.008910416278`; 122,500 steps, 6,101.8 s, stale0.

Queue progress (2026-10-14 19:12 Asia/Shanghai): seed42/path reached epoch index 49/60 and improved best/latest source-HR validation to `0.008090428975`; 125,000 steps, 6,225.9 s, stale0.

Queue progress (2026-10-14 19:14 Asia/Shanghai): seed42/path reached epoch index 50/60; best source-HR validation `0.008090428975`, latest `0.008873769475`, 127,500 steps, 6,351.5 s, stale1.

Queue progress (2026-10-14 19:16 Asia/Shanghai): seed42/path reached epoch index 51/60; best source-HR validation `0.008090428975`, latest `0.008311238940`, 130,000 steps, 6,475.2 s, stale2.

Queue progress (2026-10-14 19:18 Asia/Shanghai): seed42/path reached epoch index 52/60 and improved best/latest source-HR validation to `0.007924223903`; 132,500 steps, 6,598.8 s, stale0.

Queue progress (2026-10-14 19:21 Asia/Shanghai): seed42/path reached epoch index 53/60 and improved best/latest source-HR validation to `0.007721010593`; 135,000 steps, 6,722.7 s, stale0.

Queue progress (2026-10-14 19:23 Asia/Shanghai): seed42/path reached epoch index 54/60; best source-HR validation `0.007721010593`, latest `0.007998109450`, 137,500 steps, 6,847.7 s, stale1.

Queue progress (2026-10-14 19:25 Asia/Shanghai): seed42/path reached epoch index 55/60 and improved best/latest source-HR validation to `0.007664484706`; 140,000 steps, 6,972.1 s, stale0.

Queue progress (2026-10-14 19:26 Asia/Shanghai): seed42/path reached epoch index 56/60 and improved best/latest source-HR validation to `0.007291000329`; 142,500 steps, 7,097.2 s, stale0.

Queue progress (2026-10-14 19:28 Asia/Shanghai): seed42/path reached epoch index 57/60; best source-HR validation `0.007291000329`, latest `0.007337792483`, 145,000 steps, 7,222.0 s, stale1.

Queue progress (2026-10-14 19:30 Asia/Shanghai): seed42/path reached epoch index 58/60 and improved best/latest source-HR validation to `0.007089049045`; 147,500 steps, 7,346.8 s, stale0.

Queue progress (2026-10-14 19:32 Asia/Shanghai): seed42/path reached epoch index 59/60 (60th epoch); best source-HR validation `0.007089049045`, latest `0.007238718543`, 150,000 steps, 7,471.1 s, stale1. Worker is finishing its terminal report/hash audit.


Seed42/path source terminal (2026-10-14 19:33 Asia/Shanghai): 60 epochs completed; best source-HR validation `0.007089049045` at epoch58; charge `2.078342349826` GPU-h; checkpoint SHA-256 `31da8e8e9dde42992aba21b4ecf4efb2d8535fb4f7c15e7b152fbb16ecba20fd`; report SHA-256 `d998f4cd6cc70216c7ea3df3d44ae33c712d5ae7cf04c74832bf274ecce64449`; `query_used=false`. Reported and actual checkpoint hashes match. Queue immediately started seed42/final from scratch (PID `529349`).

Queue progress (2026-10-14 19:34 Asia/Shanghai): seed42/final from-scratch source training is active under supervisor PID `418560` / worker `529349`; epoch index 0/60, best/latest source-HR validation `0.119320945248`, 2,500 steps, 83.8 s measured training time.

Queue progress (2026-10-14 19:36 Asia/Shanghai): seed42/final reached epoch index 1/60; best/latest source-HR validation `0.102829012595`, 5,000 steps, 167.2 s measured time, stale0.

Queue progress (2026-10-14 19:37 Asia/Shanghai): seed42/final reached epoch index 2/60; best source-HR validation `0.102829012595`, latest `0.107752513873`, 7,500 steps, 250.7 s, stale1.

Queue progress (2026-10-14 19:38 Asia/Shanghai): seed42/final reached epoch index 3/60 and improved best/latest source-HR validation to `0.081193803598`; 10,000 steps, 334.8 s, stale0.

Queue progress (2026-10-14 19:40 Asia/Shanghai): seed42/final reached epoch index 4/60; best source-HR validation `0.081193803598`, latest `0.086052400206`, 12,500 steps, 418.9 s, stale1.

Queue progress (2026-10-14 19:41 Asia/Shanghai): seed42/final reached epoch index 5/60 and improved best/latest source-HR validation to `0.079654236708`; 15,000 steps, 502.9 s, stale0.

Queue progress (2026-10-14 19:43 Asia/Shanghai): seed42/final reached epoch index 6/60 and improved best/latest source-HR validation to `0.059932398685`; 17,500 steps, 586.4 s, stale0.

Queue progress (2026-10-14 19:44 Asia/Shanghai): seed42/final reached epoch index 7/60; best source-HR validation `0.059932398685`, latest `0.061666350638`, 20,000 steps, 670.1 s, stale1.

Queue progress (2026-10-14 19:45 Asia/Shanghai): seed42/final reached epoch index 8/60 and improved best/latest source-HR validation to `0.059392035427`; 22,500 steps, 753.6 s, stale0.

Queue progress (2026-10-14 19:47 Asia/Shanghai): seed42/final reached epoch index 9/60 and improved best/latest source-HR validation to `0.049013975236`; 25,000 steps, 837.8 s, stale0.

Queue progress (2026-10-14 19:48 Asia/Shanghai): seed42/final reached epoch index 10/60; best source-HR validation `0.049013975236`, latest `0.057853407457`, 27,500 steps, 923.7 s, stale1.

Queue progress (2026-10-14 19:50 Asia/Shanghai): seed42/final reached epoch index 11/60 and improved best/latest source-HR validation to `0.044508291421`; 30,000 steps, 1,007.5 s, stale0.

Queue progress (2026-10-14 19:51 Asia/Shanghai): seed42/final reached epoch index 12/60 and improved best/latest source-HR validation to `0.035851253191`; 32,500 steps, 1,091.4 s, stale0.

Queue progress (2026-10-14 19:52 Asia/Shanghai): seed42/final reached epoch index 13/60; best source-HR validation `0.035851253191`, latest `0.036295593482`, 35,000 steps, 1,175.2 s, stale1.

Queue progress (2026-10-14 19:54 Asia/Shanghai): seed42/final reached epoch index 14/60; best source-HR validation `0.035851253191`, latest `0.044298461481`, 37,500 steps, 1,258.9 s, stale2.

Queue progress (2026-10-14 19:55 Asia/Shanghai): seed42/final reached epoch index 15/60; best source-HR validation `0.035851253191`, latest `0.039425721863`, 40,000 steps, 1,342.8 s, stale3.

Queue progress (2026-10-14 19:57 Asia/Shanghai): seed42/final reached epoch index 16/60; best source-HR validation `0.035851253191`, latest `0.041430506801`, 42,500 steps, 1,426.8 s, stale4.

Queue progress (2026-10-14 19:58 Asia/Shanghai): seed42/final reached epoch index 17/60; best source-HR validation `0.035851253191`, latest `0.041760589396`, 45,000 steps, 1,510.5 s, stale5.

Queue progress (2026-10-14 19:59 Asia/Shanghai): seed42/final reached epoch index 18/60 and improved best/latest source-HR validation to `0.031467584707`; 47,500 steps, 1,594.2 s, stale0.

Queue progress (2026-10-14 20:01 Asia/Shanghai): seed42/final reached epoch index 19/60; best source-HR validation `0.031467584707`, latest `0.033049584635`, 50,000 steps, 1,677.6 s, stale1.

Queue progress (2026-10-14 20:02 Asia/Shanghai): seed42/final reached epoch index 20/60; best source-HR validation `0.031467584707`, latest `0.031885118736`, 52,500 steps, 1,763.4 s, stale2.

Queue progress (2026-10-14 20:03 Asia/Shanghai): seed42/final reached epoch index 21/60; best source-HR validation `0.031467584707`, latest `0.033701061440`, 55,000 steps, 1,847.5 s, stale3.

Queue progress (2026-10-14 20:05 Asia/Shanghai): seed42/final reached epoch index 22/60; best source-HR validation `0.031467584707`, latest `0.033057612885`, 57,500 steps, 1,931.3 s, stale4.

Queue progress (2026-10-14 20:07 Asia/Shanghai): seed42/final reached epoch index 23/60 and improved best/latest source-HR validation to `0.026768417986`; 60,000 steps, 2,015.4 s, stale0.

Queue progress (2026-10-14 20:08 Asia/Shanghai): seed42/final reached epoch index 24/60 and improved best/latest source-HR validation to `0.024275010564`; 62,500 steps, 2,099.1 s, stale0.

Queue progress (2026-10-14 20:10 Asia/Shanghai): seed42/final reached epoch index 25/60 and improved best/latest source-HR validation to `0.024082175260`; 65,000 steps, 2,183.3 s, stale0.

Queue progress (2026-10-14 20:11 Asia/Shanghai): seed42/final reached epoch index 26/60; best source-HR validation `0.024082175260`, latest `0.034989043945`, 67,500 steps, 2,266.9 s, stale1.
Stage-2 source progress (2026-10-14 20:16 Asia/Shanghai): seed42/final reached epoch index 30/60; best source-HR validation `0.020662988904`, latest `0.022867361065`, 77,500 steps, 2,605.0 s measured time, stale2. Frozen schedule and source-HR validation selection remain unchanged.
Stage-2 source progress (2026-10-14 20:19 Asia/Shanghai): seed42/final reached epoch index 32/60 and improved best/latest source-HR validation to `0.016555860228`; 82,500 steps, 2,773.0 s measured time, stale0. Continue the frozen full schedule.
Stage-2 source progress (2026-10-14 20:24 Asia/Shanghai): seed42/final reached epoch index 35/60; best source-HR validation `0.016555860228`, latest `0.017414869515`, 90,000 steps, 3,024.1 s measured time, stale3. The frozen source run remains active.
Stage-2 source progress (2026-10-14 20:25 Asia/Shanghai): seed42/final reached epoch index 36/60 and improved best/latest source-HR validation to `0.016008725294`; 92,500 steps, 3,108.0 s measured time, stale0. Frozen source schedule remains active.
Stage-2 source progress (2026-10-14 20:28 Asia/Shanghai): seed42/final reached epoch index 38/60 and improved best/latest source-HR validation to `0.013026897602`; 97,500 steps, 3,275.6 s measured time, stale0. Continue the full frozen schedule.
Stage-2 source progress (2026-10-14 20:30 Asia/Shanghai): seed42/final reached epoch index 40/60; best source-HR validation `0.013026897602`, latest `0.013822125502`, 102,500 steps, 3,445.9 s measured time, stale2. Full frozen source schedule remains active.
Stage-2 source progress (2026-10-14 20:32 Asia/Shanghai): seed42/final reached epoch index 41/60 and improved best/latest source-HR validation to `0.011723511593`; 105,000 steps, 3,530.0 s measured time, stale0. Frozen source queue continues.
Stage-2 source progress (2026-10-14 20:34 Asia/Shanghai): seed42/final reached epoch index 43/60 and improved best/latest source-HR validation to `0.010703610336`; 110,000 steps, 3,697.9 s measured time, stale0. The queue remains within the frozen source phase.
Stage-2 source progress (2026-10-14 20:36 Asia/Shanghai): seed42/final reached epoch index 44/60 and improved best/latest source-HR validation to `0.010058916678`; 112,500 steps, 3,781.9 s measured time, stale0. Continue the frozen schedule.
Stage-2 source progress (2026-10-14 20:37 Asia/Shanghai): seed42/final reached epoch index 45/60 and improved best/latest source-HR validation to `0.010020153583`; 115,000 steps, 3,865.5 s measured time, stale0. The frozen source phase remains active.
Stage-2 source progress (2026-10-14 20:40 Asia/Shanghai): seed42/final reached epoch index 47/60 and improved best/latest source-HR validation to `0.009299825739`; 120,000 steps, 4,033.0 s measured time, stale0. This remains source validation only; target HR-dev is deferred until LR selection and audit.
Stage-2 source progress (2026-10-14 20:43 Asia/Shanghai): seed42/final reached epoch index 49/60 and improved best/latest source-HR validation to `0.007941124384`; 125,000 steps, 4,201.0 s measured time, stale0. The source run continues through the frozen final epochs.
Stage-2 source progress (2026-10-14 20:47 Asia/Shanghai): seed42/final reached epoch index 52/60 and improved best/latest source-HR validation to `0.007895633997`; 132,500 steps, 4,455.9 s measured time, stale0. Seven frozen epochs remain.
Stage-2 source progress (2026-10-14 20:49 Asia/Shanghai): seed42/final reached epoch index 53/60 and improved best/latest source-HR validation to `0.007569379194`; 135,000 steps, 4,539.6 s measured time, stale0. Six frozen epochs remain, followed by source report/checkpoint validation and the queued LR screen.
Stage-2 source progress (2026-10-14 20:51 Asia/Shanghai): seed42/final reached epoch index 55/60; best source-HR validation `0.007569379194`, latest `0.007616869289`, 140,000 steps, 4,706.9 s measured time, stale2. Five frozen epochs remain.
Stage-2 source progress (2026-10-14 20:53 Asia/Shanghai): seed42/final reached epoch index 56/60 and improved best/latest source-HR validation to `0.007309890054`; 142,500 steps, 4,790.7 s measured time, stale0. Three frozen epochs remain.
Stage-2 source progress (2026-10-14 20:56 Asia/Shanghai): seed42/final reached epoch index 58/60 and improved best/latest source-HR validation to `0.007083074395`; 147,500 steps, 4,959.2 s measured time, stale0. One requested epoch remains before terminal report/checkpoint audit.
Stage-2 source terminal (2026-10-14 20:57 Asia/Shanghai): seed42/final completed 60 epochs; best source-HR validation `0.007083074395` at epoch58; measured charge `1.403775779834` GPU-h; `query_used=false`. Checkpoint SHA-256 `9932f22a443070dec2ea95de10fe00dd0562e315a2961ce2482ce7e56a02a21d` matches; report SHA-256 `42fbc5431ba69f1ed746e26f0170b4f7fb9471cf06f3b96068008eb433779ddc`. The queue automatically started the frozen seed0 LR screen.
Stage-2 LR progress (2026-10-14 20:58 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task00 (high_frequency) is active at epoch index19/100; best LR-validation relative L2 `0.022617392603`, 400 updates, 87.3 s. HR-dev remains deferred.
Stage-2 LR progress (2026-10-14 20:59 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task00 (high_frequency) reached epoch index33/100 and improved best LR-validation relative L2 to `0.021835869987`; 680 updates, 149.6 s. No HR-dev evaluation has started.
Stage-2 LR progress (2026-10-14 21:00 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task00 (high_frequency) reached epoch index47/100; best LR-validation relative L2 `0.021210711403`, 960 updates, 210.4 s. Continue unchanged.
Stage-2 LR progress (2026-10-14 21:02 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task00 (high_frequency) reached epoch index62/100 and improved best LR-validation relative L2 to `0.021049735037`; 1,260 updates, 277.2 s.
Stage-2 LR progress (2026-10-14 21:03 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task00 (high_frequency) reached epoch index88/100; best LR-validation relative L2 `0.020957346121`, 1,780 updates, 391.7 s. The selected checkpoint is still based on LR validation only.
Stage-2 LR progress (2026-10-14 21:04 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task00 (high_frequency) completed 100 epochs; unique LR minimum `0.020945114305` at epoch99, `query_metric=null`, checkpoint SHA-256 `813a8490af42ffb546efd636b6b535865a33f6caadd63a04bcad5a697e665cd8`, source path checkpoint hash matches frozen seed0 source (`b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf`). task01 (amplitude_ood) has started; no HR-dev evaluation yet.
Stage-2 LR progress (2026-10-14 21:05 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task01 (amplitude_ood) reached epoch index16/100; current best LR-validation relative L2 `0.028950416518`, 340 updates, 73.9 s. task00 report is durable with `query_metric=null`.
Stage-2 LR progress (2026-10-14 21:07 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task01 (amplitude_ood) reached epoch index30/100; best LR-validation relative L2 `0.028574091109`, 620 updates, 136.0 s.
Stage-2 LR progress (2026-10-14 21:08 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task01 (amplitude_ood) reached epoch index56/100 and improved best LR-validation relative L2 to `0.026982509415`; 1,140 updates, 250.5 s.
Stage-2 LR progress (2026-10-14 21:09 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task01 (amplitude_ood) reached epoch index70/100; best LR-validation relative L2 `0.026647796633`, 1,420 updates, 311.3 s.
Stage-2 LR progress (2026-10-14 21:10 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task01 (amplitude_ood) reached epoch index83/100; best LR-validation relative L2 `0.026476548426`, 1,680 updates, 369.2 s, stale3. Frozen epoch limit remains 100.
Stage-2 LR progress (2026-10-14 21:11 Asia/Shanghai): seed0 `pilot_pathsource_comm256` task01 (amplitude_ood) reached epoch index97/100 and improved best LR-validation relative L2 to `0.026392835588`; 1,960 updates, 429.9 s, stale0.
Stage-2 LR progress (2026-10-14 21:12 Asia/Shanghai): seed0 path-source comm256 task01 (amplitude_ood) completed 100 epochs; LR minimum `0.026371586975` at epoch98, `query_metric=null`, checkpoint SHA `18671f01ddd6d269bca94b38e4a84ea96c1ea8fe3fb409998e036758ce4740d1`; seed0 path source SHA binds to `b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf`. task02 (matern_gp) is active at epoch3/100.
Stage-2 LR progress (2026-10-14 21:13 Asia/Shanghai): seed0 path-source comm256 task02 (matern_gp) reached epoch index17/100; current best LR-validation relative L2 `0.021731198090`, 360 updates, 78.2 s.
Stage-2 LR progress (2026-10-14 21:14 Asia/Shanghai): seed0 path-source comm256 task02 (matern_gp) reached epoch index30/100 and improved best LR-validation relative L2 to `0.019303115306`; 620 updates, 136.1 s.
Stage-2 LR progress (2026-10-14 21:15 Asia/Shanghai): seed0 path-source comm256 task02 (matern_gp) reached epoch index44/100 and improved best LR-validation relative L2 to `0.018609846686`; 900 updates, 196.9 s.
Stage-2 LR progress (2026-10-14 21:16 Asia/Shanghai): seed0 path-source comm256 task02 (matern_gp) reached epoch index57/100; best LR-validation relative L2 `0.018391846732`, 1,160 updates, 254.9 s.
Stage-2 LR progress (2026-10-14 21:17 Asia/Shanghai): seed0 path-source comm256 task02 (matern_gp) reached epoch index71/100; best LR-validation relative L2 `0.018367303273`, 1,440 updates, 315.7 s, stale3.
Stage-2 LR progress (2026-10-14 21:18 Asia/Shanghai): seed0 path-source comm256 task02 (matern_gp) reached epoch index85/100; best LR-validation relative L2 `0.018362553441`, 1,720 updates, 378.0 s, stale12. Continue under the existing runner stop rule.
Stage-2 LR progress (2026-10-14 21:19 Asia/Shanghai): seed0 path-source comm256 task02 (matern_gp) completed 100 epochs; unique LR minimum `0.018362553441` at epoch73, `query_metric=null`, checkpoint SHA-256 `4a814d7f336848588fbbc4b4bb9b2cfa547ddc2572bff8b1fe617477297b545a`. task03 (localized_bump) active at epoch4/100.
Stage-2 LR progress (2026-10-14 21:20 Asia/Shanghai): seed0 path-source comm256 task03 (localized_bump) reached epoch index18/100; current best LR-validation relative L2 `0.029898693075`, 380 updates, 82.5 s.
Stage-2 LR progress (2026-10-14 21:22 Asia/Shanghai): seed0 path-source comm256 task03 (localized_bump) reached epoch index32/100 and improved best LR-validation relative L2 to `0.028910981043`; 660 updates, 144.7 s.
Stage-2 LR progress (2026-10-14 21:23 Asia/Shanghai): seed0 path-source comm256 task03 (localized_bump) reached epoch index59/100 and improved best LR-validation relative L2 to `0.028072058200`; 1,200 updates, 263.4 s.
Stage-2 LR progress (2026-10-14 21:24 Asia/Shanghai): seed0 path-source comm256 task03 (localized_bump) reached epoch index72/100 and improved best LR-validation relative L2 to `0.027713671955`; 1,460 updates, 319.8 s.
Stage-2 LR progress (2026-10-14 21:25 Asia/Shanghai): seed0 path-source comm256 task03 (localized_bump) reached epoch index86/100 and improved best LR-validation relative L2 to `0.027488541219`; 1,740 updates, 382.0 s.
Stage-2 LR progress (2026-10-14 21:27 Asia/Shanghai): seed0 path-source comm256 task03 (localized_bump) completed 100 epochs; LR minimum `0.027411289205` at epoch99, `query_metric=null`, checkpoint SHA-256 `d8481ef45fd1a4dee266c352c31d7f2eec27f5f1c5c9de68f605cee2060ab20a`, source checkpoint SHA matches seed0 path source. task04 (mixed_fourier) is active at epoch3/100.
Stage-2 LR progress (2026-10-14 21:29 Asia/Shanghai): seed0 path-source comm256 task04 (mixed_fourier) reached epoch index31/100; best LR-validation relative L2 `0.020883995132`, 640 updates, 140.3 s.
Stage-2 LR progress (2026-10-14 21:30 Asia/Shanghai): seed0 path-source comm256 task04 (mixed_fourier) reached epoch index44/100 and improved best LR-validation relative L2 to `0.020474364824`; 900 updates, 196.7 s, stale3.
Stage-2 LR progress (2026-10-14 21:32 Asia/Shanghai): seed0 path-source comm256 task04 (mixed_fourier) reached epoch index70/100; best LR-validation relative L2 `0.020474364824`, 1,420 updates, 311.1 s, stale29. It remains active under the current fixed schedule.
Stage-2 LR progress (2026-10-14 21:34 Asia/Shanghai): seed0 path-source comm256 task04 (mixed_fourier) reached epoch index96/100; best LR-validation relative L2 `0.020474364824`, 1,940 updates, 425.3 s, stale55.
Stage-2 LR progress (2026-10-14 21:34 Asia/Shanghai): seed0 path-source comm256 task04 (mixed_fourier) completed 100 epochs; LR minimum `0.020474364824` at epoch41, `query_metric=null`, checkpoint SHA-256 `d799a2ba542ffa111680d590c1b64fff992eb829fe3bbfdcf686bbb502f79e56`, source SHA binds to seed0 path checkpoint. task05 (piecewise_smooth) started at epoch1/100.
Stage-2 LR progress (2026-10-14 21:36 Asia/Shanghai): seed0 path-source comm256 task05 (piecewise_smooth) reached epoch index28/100; current best LR-validation relative L2 `0.024704037054`, 580 updates, 127.3 s.
Stage-2 LR progress (2026-10-14 21:38 Asia/Shanghai): seed0 path-source comm256 task05 (piecewise_smooth) reached epoch index54/100 and improved best LR-validation relative L2 to `0.023158727883`; 1,100 updates, 241.7 s.
Stage-2 LR progress (2026-10-14 21:40 Asia/Shanghai): seed0 path-source comm256 task05 (piecewise_smooth) reached epoch index80/100 and improved best LR-validation relative L2 to `0.022696169565`; 1,620 updates, 356.0 s.
Stage-2 LR progress (2026-10-14 21:41 Asia/Shanghai): seed0 path-source comm256 task05 (piecewise_smooth) reached epoch index94/100 and improved best LR-validation relative L2 to `0.022578512289`; 1,900 updates, 416.7 s.
Stage-2 LR arm terminal (2026-10-14 21:41 Asia/Shanghai): seed0 `pilot_pathsource_comm256` completed 6/6 tasks; mean LR validation `0.022688864408`, worst task `0.027411289205`, task0–5 `0.020945114305 / 0.026371586975 / 0.018362553441 / 0.027411289205 / 0.020474364824 / 0.022568277700`; `query_opened_or_loaded=false`. These are LR-screen values only, not the final HR-dev comparison. The queue started path-source firstpath-ratio015 task00 (high_frequency), epoch21/100, best LR validation `0.022698199895`.
Stage-2 LR progress (2026-10-14 21:43 Asia/Shanghai): seed0 path-source firstpath-ratio015 task00 (high_frequency) completed 100 epochs; LR minimum `0.021023234463` at epoch98, `query_metric=null`, checkpoint SHA-256 `45a32ff2563cf4eb2f67e0b14100f634b7ec9402967815b771c503b269ac2ed7`, source SHA binds to seed0 path checkpoint. task01 (amplitude_ood) is at epoch49/100, current best `0.027959529881`.
Stage-2 LR progress (2026-10-14 21:44 Asia/Shanghai): seed0 path-source firstpath-ratio015 task01 (amplitude_ood) completed 100 epochs; LR minimum `0.027059846418` at epoch93, `query_metric=null`, checkpoint SHA-256 `2d14b5dc346aaaf5723def633c065f2909912df73e1ac55829ecff79f27aac8a`, source SHA matches frozen seed0 path source. task02 (matern_gp) is active at epoch55/100; best LR validation `0.018644537206`.
Stage-2 LR progress (2026-10-14 21:45 Asia/Shanghai): seed0 path-source firstpath-ratio015 task02 (matern_gp) completed 100 epochs; LR minimum `0.018328954562` at epoch99, `query_metric=null`, checkpoint SHA-256 `60d94e200812b55e6562250a80812b475b54f7cd4d6440f0db00738af90407e8`; source SHA binds to seed0 path source. task03 (localized_bump) is active at epoch24/100, best LR validation `0.028885171923`.
Stage-2 LR progress (2026-10-14 21:46 Asia/Shanghai): seed0 path-source firstpath-ratio015 task03 (localized_bump) completed 100 epochs; LR minimum `0.027136980376` at epoch97, `query_metric=null`, checkpoint SHA-256 `8456ed1adf86d26eab8527881ce7c467506eb0f3dd81bd4d544685eb890c9ac3`, source SHA matches seed0 path source. task04 (mixed_fourier) is active at epoch14/100, current best `0.022038587049`.
Stage-2 LR progress (2026-10-14 21:47 Asia/Shanghai): seed0 path-source firstpath-ratio015 task04 (mixed_fourier) completed 100 epochs; LR minimum `0.019998141384` at epoch99, `query_metric=null`, checkpoint SHA-256 `21cbb9b16767859a178754acca0e6ef53990d7eccc3eecb2a30b811912f9196f`, source SHA matches seed0 path source. task05 (piecewise_smooth) is active at epoch16/100, current validation `0.025999197818`.
Stage-2 LR arm terminal (2026-10-14 21:48 Asia/Shanghai): seed0 path-source firstpath-ratio015 completed 6/6 tasks; mean LR validation `0.022583043681`, worst `0.027136980376`, task0–5 `0.021023234463 / 0.027059846418 / 0.018328954562 / 0.027136980376 / 0.019998141384 / 0.021951104881`, `query_opened_or_loaded=false`. This is LR-screen evidence only. The queue started path-source endpoint-increment-ratio015 task00 (high_frequency), epoch20/100, current best LR validation `0.022746797063`.
Stage-2 LR progress (2026-10-14 21:49 Asia/Shanghai): seed0 path-source endpoint-increment-ratio015 task00 (high_frequency) completed 100 epochs; LR minimum `0.021029376594` at epoch96, `query_metric=null`, checkpoint SHA-256 `f7794d56146a499170e81cd2a567fff87235902f5a48d82fa5d823700fb9cf9d`, source SHA matches seed0 path source. task01 (amplitude_ood) is active at epoch17/100, current best `0.029293631896`.
Stage-2 LR incident/recovery (2026-10-14 21:53 Asia/Shanghai): canonical endpoint-increment task01 (seed0/amplitude_ood) failed at Adam multi-tensor optimizer update with `RuntimeError: tensor does not have a device`; ledger charge `0.003062378770` GPU-h. The initial task00 and failed task01 artifact remain preserved but excluded from the complete arm. Queue stopped fail-closed; GPU is idle, no HR-dev/query access occurred. Recovery will rerun the full endpoint-increment arm in the immutable `retry1` namespace for each frozen seed [0,1,42], then continue the unchanged remaining arm order, LR audit, HR-dev matrix, and summary.
Stage-2 recovery preflight (2026-10-14 21:57 Asia/Shanghai): first detached script startup vanished before child log, ledger, or GPU work. The session-managed retry reached `run_phase1_trials.py` reuse validation and stopped before training because it compares raw LR-index file SHA to the report’s internal self-seal. The recovery workflow now skips the two complete seed0 canonical arms (which remain preserved) and includes them in the unified read-only LR audit. This did not create another experiment ledger row or consume GPU time.
Stage-2 recovery dispatch (2026-10-14 21:57 Asia/Shanghai): the corrected session-managed workflow skipped preserved seed0 comm256 and firstpath arms without GPU dispatch, then started the uniform endpoint-increment `retry1` arm at seed0/task00 on GPU0 (supervisor PID `630857`, worker PID `630863`).
Stage-2 retry progress (2026-10-14 21:58 Asia/Shanghai): endpoint-increment retry1 seed0 task00 (high_frequency) completed 100 epochs; LR minimum `0.021029376594` at epoch96, `query_metric=null`, checkpoint SHA-256 `37422c96bcff9040c418edb6711d0ea6be29dd4eec2ae19971e7f942f620e7af`, frozen seed0 path source SHA. Its LR result exactly matches canonical task00; retry1 task01 is at epoch17/100, best LR validation `0.029293631896`.
Stage-2 retry progress (2026-10-14 21:59 Asia/Shanghai): endpoint-increment retry1 seed0 task01 (amplitude_ood) completed 100 epochs; LR minimum `0.026970080764` at epoch99, `query_metric=null`, checkpoint SHA-256 `aeec4c17e054ec04f886bb04f1df94a1cb40c09a1afa0f52eab17f2c4fdcad0a`, source SHA binds to seed0 path source. The fresh attempt passed Adam updates; task02 (matern_gp) is active at epoch20/100, current best `0.021187970496`.
Stage-2 retry progress (2026-10-14 22:00 Asia/Shanghai): endpoint-increment retry1 seed0 task02 (matern_gp) completed 100 epochs; LR minimum `0.018283999743` at epoch98, `query_metric=null`, checkpoint SHA-256 `6c3b65229ee77f41297ce9ffd7c1e691c0339da1c8c0a113c3a2185a482191eb`, source SHA binds to seed0 path source. task03 (localized_bump) is active at epoch23/100, current best `0.029248312488`.
Stage-2 retry progress (2026-10-14 22:01 Asia/Shanghai): endpoint-increment retry1 seed0 task03 (localized_bump) completed 100 epochs; LR minimum `0.027264112333` at epoch99, `query_metric=null`, checkpoint SHA-256 `b3aeb99384db858a5407e4b8d358560c746cf4956db2d2df3da7fa68c5d404f7`, source SHA binds to seed0 path source. task04 (mixed_fourier) is active at epoch27/100, best `0.020648432412`.
Stage-2 retry progress (2026-10-14 22:03 Asia/Shanghai): endpoint-increment retry1 seed0 task04 (mixed_fourier) completed 100 epochs; LR minimum `0.019895660743` at epoch96, `query_metric=null`, checkpoint SHA-256 `93568d168b2e51620700d7908ccaf437e682e0aabf1fe922f550fa4ff5c4b895`, source SHA binds to seed0 path source. task05 (piecewise_smooth) is active at epoch30/100, current best `0.024282879679`.
Stage-2 LR arm terminal (2026-10-14 22:04 Asia/Shanghai): endpoint-increment `retry1` seed0 completed 6/6 tasks; mean LR validation `0.022564816523`, worst `0.027264112333`, task0–5 `0.021029376594 / 0.026970080764 / 0.018283999743 / 0.027264112333 / 0.019895660743 / 0.021945668961`, `query_opened_or_loaded=false`. Accepted retry1 arm remains distinct from the failed canonical attempt. Queue is running seed0 endpoint-direction-ratio015 task00 (high_frequency) at epoch33/100, best `0.021638662246`.
Stage-2 LR progress (2026-10-14 22:05 Asia/Shanghai): seed0 endpoint-direction-ratio015 task00 (high_frequency) completed 100 epochs; LR minimum `0.021027999127` at epoch98, `query_metric=null`, checkpoint SHA-256 `ea52bd23b5bf329316c9a0c3b631537d5900e8416ee9b8a96b1171c81a00c97b`, source SHA binds to seed0 path source. task01 (amplitude_ood) is at epoch44/100, best LR validation `0.028142695245`.
Stage-2 LR progress (2026-10-14 22:06 Asia/Shanghai): seed0 endpoint-direction-ratio015 task01 (amplitude_ood) completed 100 epochs; LR minimum `0.026972036302` at epoch99, `query_metric=null`, checkpoint SHA-256 `2a7a2e093e9fcf70dfbf2276cb2f9230487f483d1a48264b8be981a3ee22c634`, source SHA binds to seed0 path source. task02 (matern_gp) is active at epoch49/100, best LR validation `0.019130298035`.
Stage-2 LR progress (2026-10-14 22:07 Asia/Shanghai): seed0 endpoint-direction-ratio015 task02 (matern_gp) completed 100 epochs; LR minimum `0.018267683027` at epoch99, `query_metric=null`, checkpoint SHA-256 `1c8d47dc9e8195bdba4043de774f41fcaaa034464aebb0db5bca4aa9feafb58f`, source SHA binds to seed0 path source. task03 (localized_bump) is active at epoch57/100, current best `0.027704891289`.
Stage-2 LR progress (2026-10-14 22:08 Asia/Shanghai): seed0 endpoint-direction-ratio015 task03 (localized_bump) completed 100 epochs; LR minimum `0.027177954529` at epoch99, `query_metric=null`, checkpoint SHA-256 `9e2c299bf4f20890fbf7f12a91ac707907bcb1ffd3348468f5834d584c357ecf`, source SHA binds to seed0 path source. task04 (mixed_fourier) is active at epoch59/100, current best `0.019849618216`.


### Recovery queue update (2026-10-14 22:11 Asia/Shanghai)

Seed0 endpoint-direction completed all six LR tasks (mean `0.022538409140`, task order high_frequency, amplitude_ood, matern_gp, localized_bump, mixed_fourier, piecewise_smooth: `0.021027999127 / 0.026972036302 / 0.018267683027 / 0.027177954529 / 0.019849618216 / 0.021935163642`). Endpoint-increment uses the consistent `retry1` alias. The recovery workflow is now training ordinary full-FT + comm256 seed0/task00. The current ordinary task is at epoch8/100, best LR validation `0.043375408137`; no HR-dev evaluation has started. Queue status and logs: `artifacts/logs/phase1_stage2_recovery/status.tsv` and `artifacts/logs/phase1_stage2_recovery/`.


Queue progress (2026-10-14 22:20 Asia/Shanghai): ordinary full-FT + comm256 seed0 task00 ended at LR minimum `0.043375408137` (epoch5). Task01 is training at epoch6/100 with best `0.054916321579` and latest `0.063978090649`. These are LR-validation results only; all planned arms, LR audit, and HR-dev evaluations remain in the serial queue.


Queue progress (2026-10-14 22:35 Asia/Shanghai): ordinary full-FT + comm256 seed0 tasks00–02 have LR minima `0.043375408137 / 0.048047667043 / 0.033220852772`, selected at epochs 5/9/1. Task03 is active at epoch7/100, current best `0.036945638969`. HR-dev remains deferred until the full 108-cell audit passes.


Queue progress (2026-10-14 22:43 Asia/Shanghai): ordinary full-FT + comm256 seed0 task03 completed at LR minimum `0.036945638969` (epoch6); task04 is active at epoch1/100, current best `0.047093198646`. Earlier task00–02 minima: `0.043375408137 / 0.048047667043 / 0.033220852772`.


Queue progress (2026-10-14 22:52 Asia/Shanghai): ordinary full-FT + comm256 seed0 task04 completed at LR minimum `0.042129778909` (epoch6). Final task05 is active, epoch1/100, initial best `0.070600183913`. Tasks00–03 are complete; HR-dev remains deferred.


### Host restart recovery (2026-10-15 10:36 Asia/Shanghai)

The previous queue advanced through all six seed0 arms and started seed1/pathsource comm256/task00. After host restart, no process remained; the task directory has only `config.json`, with no progress snapshot, checkpoint, or task result. The attempt is excluded and conservatively charged for 10h27m06s through host reboot (`10.451666666667` GPU-h). The unchanged seed1 arm is rerun under immutable `pilot_pathsource_comm256-recovery1`; its alias is mapped back to the frozen candidate in the audit and summary. Existing complete arms are skipped.


Recovery dispatch (2026-10-15 10:40 Asia/Shanghai): the revised workflow verified and skipped the six completed seed0 arms, then started seed1/pathsource comm256 as `pilot_pathsource_comm256-recovery1` (supervisor PID `29364`, worker PID `29391`). The per-seed alias is recorded in `artifacts/reports/phase1_stage2_attempt_aliases_20261015.json` and will be reflected in the final LR audit and HR-dev summary.


Recovery progress (2026-10-15 10:56 Asia/Shanghai): seed1/pathsource comm256 recovery1 task00 and task01 completed at LR minima `0.022703295085` (epoch42) and `0.028957645321` (epoch12), query-free. Task02 is active at epoch8/100, best `0.023649407661`.


Recovery progress (2026-10-15 11:03 Asia/Shanghai): seed1/pathsource comm256 recovery1 task02 completed at LR minimum `0.018329907398` (epoch99). Task03 is active at epoch12/100, current best `0.026171641622`; tasks00–02 are complete and query-free.


Recovery progress (2026-10-15 11:11 Asia/Shanghai): seed1/pathsource comm256 recovery1 task03 completed at LR minimum `0.024261596930` (epoch98). Task04 is active at epoch6/100, best `0.022715760727`; tasks00–03 are complete and query-free.


Recovery progress (2026-10-15 11:18 Asia/Shanghai): seed1/pathsource comm256 recovery1 task04 completed at LR minimum `0.019176766777` (epoch25). Task05 is active at epoch2/100, current best `0.031314128515`; tasks00–04 are complete and query-free.


Recovery arm terminal (2026-10-15 11:25 Asia/Shanghai): seed1/pathsource comm256 recovery1 completed all six tasks; LR minima in family order are `0.022703295085 / 0.028957645321 / 0.018329907398 / 0.024261596930 / 0.019176766777 / 0.020289749623`, mean `0.022286494515`, query-free. The queue started seed1/pathsource firstpath task00.


Queue progress (2026-10-15 11:26 Asia/Shanghai): seed1/pathsource firstpath-ratio015 task00 completed at LR minimum `0.021914835734` (epoch58); task01 is active. The seed1 comm256 recovery1 arm is complete, mean LR `0.022286494515`, query-free.


Recovery arm terminal (2026-10-15 11:32 Asia/Shanghai): seed1/pathsource firstpath-ratio015 completed all six tasks, mean LR validation `0.02194768`, query-free. Task00–05 best LR values are recorded in the result JSONs. The queue started seed1 endpoint-increment retry1 task00.


Queue progress (2026-10-15 11:35 Asia/Shanghai): seed1 endpoint-increment retry1 tasks00–02 completed at LR minima `0.021927399928 / 0.029363005329 / 0.018380045210`; task03 is active.


### 2026-10-15 11:38 progress

Seed1 endpoint-increment-ratio015 retry1 finished all six LR cells (mean LR validation `0.022016226081`; per-family results in the task reports and current state). Its measured charge was `0.105512133350` GPU-h. The serial queue has started seed1 endpoint-direction-ratio015. The 108-cell LR audit and HR-dev phase remain deferred; no query or HR-dev has been read.


### 2026-10-15 11:44 progress

Seed1 endpoint-direction-ratio015 finished all six LR cells (mean `0.022106576939`, query-free; task details in `docs/STATE.md`) at `0.106539569751` GPU-h. The serial queue has started ordinary full-FT + comm256 seed1.


### 2026-10-15 12:21 interruption and recovery

The managed queue and worker were absent when checked; GPU was idle. Seed1 ordinary-full+comm256 task00 and task01 terminal reports remain preserved; task02 had only a progress snapshot through epoch36 and no terminal result. The incomplete canonical arm is excluded. The upper-bound process-age charge is 1229 s (`0.341388888889` GPU-h), plus terminal task00–01 use (`0.277525144359` GPU-h). Recovery will rerun all six cells under a fresh immutable namespace using the same seed/config/splits.


Recovery dispatch (2026-10-15 12:22 Asia/Shanghai): verified no old process was active; resumed the alias-aware queue. Complete seed0 and seed1 arms were skipped; ordinary full-FT + comm256 seed1 is restarting under recovery1 from task00 with all frozen parameters and splits.


Recovery arm terminal (2026-10-15 13:12 Asia/Shanghai): seed1 ordinary full-FT + comm256 recovery1 completed all six LR tasks. Mean `0.039740790356`; family order high_frequency, amplitude_ood, matern_gp, localized_bump, mixed_fourier, piecewise_smooth: `0.037029018078 / 0.045844315796 / 0.037664393603 / 0.039230308012 / 0.032245909900 / 0.046430796746`. Charge `0.832853350532` GPU-h. Its task00–01 minima exactly reproduce the preserved partial canonical attempt. The queue started seed1 lifted scratch.


Queue progress (2026-10-15 13:49 Asia/Shanghai): seed1 lifted scratch lr3e-4/cosine-500/wd1e-4 completed six LR tasks, mean `0.460710063057`, per-family values `0.538656096254 / 0.393020512909 / 0.474808148760 / 0.410299671907 / 0.545446345443 / 0.402029603068`, charge `0.614725686001` GPU-h. Queue started the last model seed, seed42/pathsource comm256. No HR-dev/query access.

Stage-2 LR terminal and HR-dev queue (2026-10-15 16:21 Asia/Shanghai): all 108 frozen LR cells across six candidates, three model seeds, and six boundary families completed. The LR provenance audit passed all 108 cells with `query_access=false`, including source, support, validation, checkpoint and manifest seals; interrupted canonical namespaces remain excluded behind recorded recovery aliases. Seed42 arm means were pathsource comm256 `0.019426900716`, firstpath `0.019214181132`, endpoint-increment retry1 `0.019215993804`, endpoint-direction `0.019245741632`, ordinary full-FT+comm256 `0.036084177286`, and lifted scratch `0.455993099650`. HR-dev evaluation is now running serially after LR checkpoint selection; 12/108 reports are complete. Query remains untouched.


### HR-dev terminal and paper-claim review (2026-10-15 16:30 Asia/Shanghai)

All 108 frozen HR-dev cells completed from the six fixed candidates × seeds `[0,1,42]` × six boundary families; no training/evaluation rerun was used after the summary verifier correction. The protocol auditor independently recomputed every metric from raw per-step reports and verified unique LR-selected checkpoints, source/checkpoint/evaluator hashes, alias mapping, index seals and shards, and `query_read=false`.

Candidate means (worst family mean): pathsource comm256 `0.05347209` (`0.06726502`), endpoint A2 `0.05402562` (`0.07218725`), firstpath A1 `0.05409390` (`0.07254368`), endpoint direction A3 `0.05423021` (`0.07212134`), ordinary full+comm256 `0.05488117` (`0.07618120`), lifted scratch `0.50904359` (`0.68809062`). C256 is 2.57% below the ordinary arm but nominal paired 95% CI across three seed means includes zero. A1 is 0.126% worse than A2 and 0.251% better than A3; differences are small and seed-dependent. No stable additional path-information gain is established; no A0 is present. This is development evidence on six fixed tasks and one support draw, not confirmation or cross-family inference.

The summarizer fix changes only verification semantics: task/evaluation index SHA fields are canonical content seals, not raw-file digests. Seal recomputation and per-shard SHA checks remain enforced. Full seed×family table and claim assessment: `artifacts/reports/PHASE1_STAGE2_SUMMARY_20261015.md`; machine-readable source: `artifacts/reports/phase1_stage2_pilot_hrdev_summary_20261015.json`.
