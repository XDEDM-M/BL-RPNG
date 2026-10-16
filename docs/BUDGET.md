# Phase-1 execution budget

```yaml
hardware: 2 x NVIDIA GeForce RTX 4090 (24 GiB each), CUDA 13.0
fresh_campaign_gpu_hours_cap: 120
confirmation_and_controls_reserve_fraction: 0.40
main_method_trial_cap: 24
baseline_trial_cap_per_method: 12
required_main_improvement_rounds: 3
screen_model_seeds: [42]
pilot_model_seeds: [0, 1, 42]
confirm_model_seeds: [0, 1, 42]  # amended 2026-10-15; see CONFIRMATORY.md
confirm_support_seeds_min: 3
solver_generation_workers: 1
allow_external_downloads: false
```

## Current authoritative reconciliation (2026-10-16, after R5 seed1 replication)

The corrected `docs/EXPERIMENTS.csv` ledger charges **71.866022173522 GPU-h**
at snapshot `artifacts/reports/phase1_progress_after_seed1_hrdev_complete_20261016.json`.
Against the 120 GPU-h campaign cap, nominal total remainder is
**48.133977826478 GPU-h**: **0.133977826478 GPU-h** of exploration capacity
plus the protected **48 GPU-h** confirmation/control reserve. The current
snapshot excludes historical untimed HR-dev executions; estimating those 24
reports at the largest observed timed HR-dev cost (`0.000861450032 GPU-h` each)
uses another **0.020674800768 GPU-h**, leaving about **0.113303025710 GPU-h**
as a conservative exploration remainder. This is an uncertainty estimate,
not an additional ledger charge. No Phase-1 GPU job is active.

The completed seed0 LR=`3e-5` screen cost **0.928907764596 GPU-h**. Its one
predeclared seed1 replication cost **0.930488999683 GPU-h** (`0.926292034617`
LR training + `0.004196965067` HR-dev). Together they are **1.859396764279
GPU-h**. Reproduction and metrics are in
`artifacts/reports/PHASE1_BASELINE_LR3E5_SCREEN_20261016.md` and
`artifacts/reports/PHASE1_BASELINE_LR3E5_SEED1_REPLICATION_20261016.md`.
The seed1 task01 charge exceeded the prior per-task maximum by only
`0.00003272 GPU-h`; the complete replication remained `0.001067 GPU-h` below
its predeclared aggregate estimate. The current remaining exploration balance
cannot fund another complete six-family LR+HR screen (about `0.93 GPU-h`).

Budget aggregation was corrected to count distinct training attempts by
`(trial_id, config_sha256, raw_output_path)` while collapsing exact duplicate
contention annotations, and to count each completed HRDEV execution. An
independent audit reconciled the former R5 snapshot's `70.634879848141 GPU-h`
to `70.935533173839 GPU-h` before seed1 replication: three distinct retry
attempts add `0.300653325698 GPU-h`; four duplicate exclusion annotations
would add `0.122081095404 GPU-h` if naïvely summed. See the budget-ledger
follow-up in `docs/agents/REVIEW.md` and the corrected snapshot
`artifacts/reports/phase1_progress_after_r5_complete_20261016_1355_corrected.json`.

The amended three-method primary confirmation matrix is projected at about
**158.2 GPU-h**, before the nested `m` curve, HR expert saturation, additional
mechanism controls, confirmation-scale CPU data generation or final review.
Current nominal total remainder is only `48.133978 GPU-h`, so the primary
shortfall is at least **110.066 GPU-h**. The representative four-task few-shot
curve raises the known subtotal to about `187.7 GPU-h`, a shortfall of at least
**139.566 GPU-h**. These remain projections; preserve all 48 tasks, three
crossed support draws and the query gate. No confirmation query has been
generated or read.

## Historical reconciliation (2026-10-15, after Amendment 2)

The append-only `docs/EXPERIMENTS.csv` ledger charges **45.989272304442 GPU-h** by the official `terminal_gpu_hours()` function in `scripts/summarize_phase1_progress.py`. Against the 120 GPU-h cap, nominal total remainder is **74.010727695558 GPU-h**: **26.010727695558 GPU-h** of exploration capacity plus the protected **48 GPU-h** confirmation/control reserve. No Phase-1 training or evaluation job is active. Historical early HR-dev evaluations were not all timed, so this charge is a lower bound and remaining capacity is optimistic.

The original five-seed estimate for the three-method primary confirmation matrix was **271.09 GPU-h**. Amendment 2 in `docs/prereg/CONFIRMATORY.md` reduces the model seeds to the already-used `[0,1,42]` and preserves the 48 tasks, three crossed support draws, methods, and query roles. Proportional scaling of the measured target-cell and query costs projects the amended primary at about **158.2 GPU-h**, before HR-expert training and the nested few-shot curve. That still exceeds nominal remaining capacity by about **84.2 GPU-h**. The representative four-task `m` curve adds an estimated **29.5 GPU-h** after reusing `m=4`, for a known subtotal of about **187.7 GPU-h** and a shortfall of **113.7 GPU-h**. HR-expert training, confirmation-scale CPU data generation, additional path-FT confirmation arms, and final review remain uncosted. These are projections, not measured confirmation results. Do not open the query gate or reduce the frozen task/support matrix. Full assumptions are in `docs/agents/PERFORMANCE.md` and `docs/PHASE1_COMPLETION.md`.

The four-task HR-from-zero comparison is complete. Its fits used **2.642104 GPU-h**; target HR support labels and packaging took **1187.548 s CPU wall**. The path-source comparator used **2.068094 GPU-h** for source pretraining and **0.486617 GPU-h** for the four target LR adaptations. The end-to-end report adds the shared six-family LR support generation cost and shows CPU/GPU costs separately, including a clearly labeled serial-equivalent accounting. No HR-from-zero task reached its matched path-source HR-dev error, so a finite equal-accuracy speedup is unavailable. Source HR-training dataset generation time is not traceable and is excluded rather than treated as zero. See `artifacts/reports/PHASE1_HR_SCRATCH_END_TO_END_20261015.md`.

The 120 GPU-hour cap follows the mission default. It counts each occupied GPU
hour separately; running one job on each card concurrently consumes two GPU
hours per wall-clock hour. The campaign starts at 0 newly consumed GPU hours;
the frozen historical assets predate this budget and their untraceable cost is
not counted as new work. Preserve 48 GPU hours for confirmation and controls,
leaving at most 72 GPU hours for exploration. CPU data generation and storage
are tracked separately. At discovery, 1.1 TiB is free and no GPU jobs are
running. Do not assign multiple GPUs to a single training job.

Terminal GPU ledger charges total `11.143806207176` GPU-hours as of 2026-10-14 02:48 Asia/Shanghai, recomputed by `scripts/summarize_phase1_progress.py` into `artifacts/reports/phase1_progress_20261014_0248_hr.json`. Exploration balance is `60.856193792824` GPU-hours out of 72; 48 GPU-hours remain reserved for confirmation and controls. The total includes failed, interrupted, and contention-excluded reservations once per trial/task ID. It includes timed R3, R2B, ordinary direct256, and full-FT HR-development evaluations; early HR-development evaluations were not instrumented, so the ledger omits their overhead and is not total campaign occupancy. CPU-only data generation is tracked separately. The 24-trial E1 method-screen cap is closed (R2B added one documented method trial beyond it); no further exploratory main-method configurations should be added. A campaign-wide advisory lock prevents duplicate Phase-1 training runners.

Operational update (2026-10-14 01:20 Asia/Shanghai): ordinary-initialization direct256 two-step baseline `ordinary_comm256_lr1e-3_cw005_r2` is active under PID 383074 on GPU0. Its task-0 active remainder is not yet charged. A duplicate runner attempt for H3 baseline controls was rejected by the lock before GPU work; it incurs no experiment charge. The terminal total above is still the 01:14 snapshot and excludes this active trial.

Progress update (2026-10-14 01:22 Asia/Shanghai): task 0 completed at `0.12382200` GPU-h; task 1 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0122.json` reports terminal charges `9.763974874180` GPU-h and exploration balance `62.236025125820` GPU-h, excluding the active remainder.

Progress update (2026-10-14 01:30 Asia/Shanghai): task 1 completed at `0.12355290` GPU-h; task 3 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0130.json` reports terminal charges `9.887527776417` GPU-h and exploration balance `62.112472223583` GPU-h, excluding the active remainder.

Progress update (2026-10-14 01:37 Asia/Shanghai): task 3 completed at LR-dev `0.034490032378` (best epoch 99) and cost `0.12355546` GPU-h; task 5 is active. Snapshot `artifacts/reports/phase1_progress_20261014_posttask03.json` reports terminal charges `10.011083239101` GPU-h and exploration balance `61.988916760899` GPU-h, excluding the active remainder.

Operational update (2026-10-14 01:48 Asia/Shanghai): full-FT implementation smoke `ordinary_full_lr3e-5_cos100-capacity-smoke1-smoke` passed on task 0 (two epochs, finite LR-dev `0.03721306`, charge `0.00267525` GPU-h). Capacity screen PID 385765 is active on GPU0 with ordinary full-FT LR `3e-5/1e-4` and scratch 500-epoch LR `1e-4/3e-4` arms; the first full-FT task is active. A redundant smoke dispatch was rejected by the campaign lock before setup and incurred no GPU charge. Snapshot `artifacts/reports/phase1_progress_20261014_0148_capacity1.json` reports terminal charges `10.140444360577` GPU-h and exploration balance `61.859555639423` GPU-h; active remainder excluded.

Progress update (2026-10-14 01:51 Asia/Shanghai): first ordinary full-FT task 3 completed at LR-dev `0.030504894169`, best epoch 41, cost `0.02302757` GPU-h; task 5 is active. HRDEV duplicate-evaluation execution overhead `0.002365039979` GPU-h is now included by the corrected progress accounting. Snapshot `artifacts/reports/phase1_progress_20261014_0150_capacity1_corrected.json` reports terminal charges `10.211916119974` GPU-h and exploration balance `61.788083880026` GPU-h, excluding active remainder.

Progress update (2026-10-14 01:53 Asia/Shanghai): ordinary full-FT LR `3e-5` completed all four tasks with LR mean `0.030261243111` (best epochs `95/31/41/68`; cost `0.09207531` GPU-h); full-FT LR `1e-4` task 0 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0153_capacity1.json` reports terminal charges `10.234884709260` GPU-h and exploration balance `61.765115290740` GPU-h, excluding active remainder.

Progress update (2026-10-14 01:58 Asia/Shanghai): ordinary full-FT LR `1e-4` completed all four tasks at LR mean `0.029712766977` (best epochs `25/12/22/48`; cost `0.09152865` GPU-h), 1.81% below the LR `3e-5` arm. The 500-epoch scratch LR `1e-4`, weight-decay `1e-4` screen is active on task 0. Snapshot `artifacts/reports/phase1_progress_20261014_0158_capacity1.json` reports terminal charges `10.326413355115` GPU-h and exploration balance `61.673586644885` GPU-h, excluding active remainder.

Progress update (2026-10-14 02:03 Asia/Shanghai): first scratch 500-epoch LR `1e-4` task 0 completed at LR validation `0.542029540986` (best epoch 365), charge `0.101571864810` GPU-h; task 1 is active. This single task is not a family-balanced conclusion. Snapshot `artifacts/reports/phase1_progress_20261014_0203_capacity1.json` reports terminal charges `10.427985219925` GPU-h and exploration balance `61.572014780075` GPU-h; active remainder excluded.

Progress update (2026-10-14 02:10 Asia/Shanghai): scratch LR `1e-4` task 1 completed at LR validation `0.471997121815` (best epoch 290), charge `0.101496578717` GPU-h; task 3 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0210_capacity1.json` reports terminal charges `10.529481798641` GPU-h and exploration balance `61.470518201359` GPU-h; active remainder excluded.

Progress update (2026-10-14 02:16 Asia/Shanghai): scratch LR `1e-4` task 3 completed at LR validation `0.429840025399` (best epoch 354), charge `0.101777361027` GPU-h; task 5 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0216_capacity1.json` reports terminal charges `10.631259159668` GPU-h and exploration balance `61.368740840332` GPU-h; active remainder excluded.

Progress update (2026-10-14 02:22 Asia/Shanghai): scratch LR `1e-4`, cosine-500, WD `1e-4` finished the four tasks at LR mean `0.462373684091`, with training charge `0.406252357090` GPU-h. Best epochs are `365/290/354/498`; inspect task-5 trajectory because its minimum is at the schedule end. Scratch LR `3e-4` task 0 is now active. Snapshot `artifacts/reports/phase1_progress_20261014_0222_capacity1.json` reports terminal charges `10.732665712205` GPU-h and exploration balance `61.267334287795` GPU-h; active remainder excluded.

Progress update (2026-10-14 02:28 Asia/Shanghai): scratch LR `3e-4`, cosine-500 task 0 completed at LR validation `0.547625704296` (best epoch 207), charge `0.101373062031` GPU-h; task 1 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0228_capacity1.json` reports terminal charges `10.834038774236` GPU-h and exploration balance `61.165961225764` GPU-h; active remainder excluded.

Progress update (2026-10-14 02:34 Asia/Shanghai): scratch LR `3e-4` task 1 completed at LR validation `0.380540154641` (best epoch 314), charge `0.101465167409` GPU-h; task 3 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0234_capacity1.json` reports terminal charges `10.935503941645` GPU-h and exploration balance `61.064496058355` GPU-h; active remainder excluded.

Progress update (2026-10-14 02:40 Asia/Shanghai): scratch LR `3e-4` task 3 completed at LR validation `0.399693717621` (best epoch 371), charge `0.101443309265` GPU-h; task 5 is active. Snapshot `artifacts/reports/phase1_progress_20261014_0240_capacity1.json` reports terminal charges `11.036947250910` GPU-h and exploration balance `60.963052749090` GPU-h; active remainder excluded.
Progress update (2026-10-14 01:46 Asia/Shanghai): ordinary-source direct256 baseline completed (LR `0.034392807039`, HR-dev `0.057153851914`, charge `0.49436767` GPU-h); all four HR reports are timed and query-free. Fresh report `artifacts/reports/phase1_progress_20261014_0146.json` records total `10.137769113949` GPU-h and exploration balance `61.862230886051` GPU-h. No training/evaluation process is active.
Progress update (2026-10-14 02:48 Asia/Shanghai): frozen R1B capacity screen completed. Full-FT arms cost `0.09208/0.09153` GPU-h with LR/HR means `0.03026124/0.06297840` and `0.02971277/0.06203766`; scratch 500-epoch arms cost `0.40625/0.40568` GPU-h with LR means `0.46237368/0.43320730`. Fresh report `artifacts/reports/phase1_progress_20261014_0248_hr.json` records `11.143806207176` terminal GPU-h and `60.856193792824` exploration GPU-h remaining. No training process was active at 02:48, before the R2C follow-up.

Operational update (2026-10-14 02:49 Asia/Shanghai): the ordinary full-FT+direct256 consistency smoke passed (2 epochs, `0.00504648` GPU-h). Formal R2C task 0 is active under PID `398283` on GPU0; active remainder is not charged. Snapshot `artifacts/reports/phase1_progress_20261014_0249_fullcomm_active.json` reports terminal `11.148852687807` GPU-h and exploration balance `60.851147312193` GPU-h.

Progress update (2026-10-14 02:58 Asia/Shanghai): R2C task 0 completed at LR-dev `0.033986920293` (runner best epoch 33, independent uniqueness audit pending), charge `0.138872744551` GPU-h. Task 1 is active under the same serialized runner. Fresh snapshot `artifacts/reports/phase1_progress_20261014_0258_fullcomm_task00.json` reports terminal charges `11.287725432358` GPU-h and exploration balance `60.712274567642` GPU-h, excluding active remainder.

Progress update (2026-10-14 03:08 Asia/Shanghai): R2C task 1 completed at LR-dev `0.031091272814` (runner best epoch 32), charge `0.138551898925` GPU-h; task 3 is active. Fresh snapshot `artifacts/reports/phase1_progress_20261014_0308_fullcomm_task01.json` reports terminal charges `11.426277331283` GPU-h and exploration balance `60.573722668717` GPU-h, excluding active remainder.

Progress update (2026-10-14 03:35 Asia/Shanghai): R2C ordinary full-FT+direct256 completed all four LR tasks and sequential query-free HR-development checks. LR/HR means are `0.030274311443/0.055875992816`; training charge is `0.554417839611` GPU-h. Fresh report `artifacts/reports/phase1_progress_20261014_0335_r2c_complete.json` records terminal campaign usage `11.706041885620` GPU-h, leaving `60.293958114380` GPU-h of the 72-hour exploration allocation; the 48-hour confirmation reserve remains untouched. Independent provenance audit is pending.

Operational update (2026-10-14 10:15 Asia/Shanghai): Stage-2 pilot source pretraining started serially on GPU0 with seed 0/path from random initialization, orchestrator PID 404356 and GPU worker PID 404360. At the latest read it had 7,500 optimizer updates and 370.3 elapsed seconds; terminal GPU-hours are not yet charged. The frozen pilot schedules six source pretrains and 108 target task adaptations; confirmation reserve remains unspent.

Data-freeze update (2026-10-14 10:20 Asia/Shanghai): the independent 48-task/six-family confirmation boundary manifest was frozen and seal-verified; no rollout data was generated and no GPU budget was charged. Current source pretrain remains active at epoch 4/60 with 617.2 s elapsed; its terminal charge is not yet recorded.

Progress update (2026-10-14 10:35 Asia/Shanghai): seed0/path source pretraining reached epoch 6/60, source-HR validation `0.049681636885`, 17,500 optimizer steps, elapsed `864.7` s, improving from the previous best `0.069865718108`. Active GPU hours remain uncharged until the source task finishes. The confirmation generator was hardened and no solver data/query was opened.

Progress update (2026-10-14 10:38 Asia/Shanghai): source seed0/path is at epoch 7/60, best source-HR validation `0.049681636885`, 20,000 updates, 987.7 s elapsed; active work remains uncharged.

Progress update (2026-10-14 10:44 Asia/Shanghai): seed0/path source reached epoch 9/60 at best validation `0.049681636885`, 25,000 optimizer steps, 1,235.1 s elapsed, three stale epochs. Active cost is not terminally charged; all 60 exploration GPU-h remain except completed prior trials.

Progress update (2026-10-14 10:46 Asia/Shanghai): source seed0/path reached epoch 10/60, refreshed best validation to `0.047623343658`, 27,500 steps, 1,361.4 s. GPU0 had about 7.4 GiB allocated. Terminal charge remains pending.

Progress update (2026-10-14 10:35 Asia/Shanghai): recovery inspection found seed0/path at epoch 11/60, source-HR validation `0.044590274550`, 30,000 optimizer steps, elapsed `1,485.6` s. This corrects the prose snapshot, which had lagged at epoch 10; the progress JSON is the source of truth. GPU0 is the sole active training device; GPU1 is idle. No terminal charge is booked until this source job exits.


Operational/progress update (2026-10-14 10:40 Asia/Shanghai): independent review found that the frozen `S2PILOT` round name was rejected by the target runner; the constrained namespace validator now accepts it without changing the pilot config. The confirmation generator now rejects confirmation/development manifest-root mismatches, and final HR query generation requires a sealed full method × five model seed × at least three support seed × every frozen task checkpoint matrix plus frozen preregistration and evaluator/analysis hashes. The append-only preregistration is `docs/prereg/CONFIRMATORY.md`; query data remains absent.

Progress update (2026-10-14 10:40 Asia/Shanghai): seed0/path source pretraining reached epoch 14/60. Best source-HR validation is `0.040410235524`; latest epoch validation is `0.042579648640`; 37,500 optimizer updates, 1,857.4 s elapsed, one stale epoch. The next five frozen source runs remain queued. No terminal GPU hours are booked while the source run is active; no target or HR query role has been opened.

Live budget note (2026-10-14 10:40 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h, leaving `60.293958114380` exploration GPU-h before the active job. Seed0/path has accumulated `1,857.4` s (`0.5160` GPU-h) on GPU0; estimated live exploration balance is about `59.778` GPU-h. It is not yet a terminal ledger row.

Progress update (2026-10-14 10:47 Asia/Shanghai): seed0/path source pretraining reached epoch 18/60, best source-HR validation `0.029791256829`, latest `0.029801228945`, 47,500 optimizer steps and 2,352.9 s elapsed (`0.65357` GPU-h; one stale epoch). Terminal campaign ledger remains `11.706041885620` GPU-h, so estimated exploration balance after charging active elapsed time is `59.64039` GPU-h; this is an estimate until the source process exits. Confirmation reserve remains untouched. No target HR-dev or query role has been opened.

Progress update (2026-10-14 10:49 Asia/Shanghai): seed0/path source pretraining reached epoch 19/60, best source-HR validation remains `0.029791256829`, latest `0.109989479869`, 50,000 optimizer steps and 2,476.5 s elapsed (`0.68791` GPU-h; two stale epochs). Terminal ledger remains `11.706041885620` GPU-h; live estimated exploration balance is `59.60605` GPU-h pending terminal accounting.

Progress update (2026-10-14 10:51 Asia/Shanghai): seed0/path source pretraining reached epoch 20/60 and improved best source-HR validation to `0.022170161670`, 52,500 optimizer steps, 2,602.4 s elapsed (`0.72288` GPU-h; zero stale epochs). Terminal ledger remains `11.706041885620` GPU-h, live estimated exploration balance `59.57108` GPU-h until task completion. The six LR-only pilot index pairs passed CPU provenance preflight; no additional GPU-hours were incurred.

Operational update (2026-10-14 10:54 Asia/Shanghai): confirmation query access is now gated in both the wrapper and the SfePy worker itself, with hashes for source checkpoints, all frozen data/checkpoints/analysis, and the generation/verifier code. No GPU-hours or query data were consumed by this hardening. Active source seed0/path remains at epoch 21/60, elapsed 2,726.3 s (`0.75731` GPU-h); the terminal ledger remains `11.706041885620` GPU-h, live estimated exploration balance `59.53665` GPU-h.

Progress update (2026-10-14 11:01 Asia/Shanghai): source seed0/path reached epoch 24/60, best source-HR validation `0.022170161670`, latest `0.036692537165`, 62,500 updates and 3,098.1 s (`0.86058` GPU-h; four stale epochs). Terminal ledger remains `11.706041885620` GPU-h; live estimated exploration balance is `59.43338` GPU-h pending task completion. Query manifest/IC/solver/runtime protocol hardening used no GPU and generated no query data.

Progress update (2026-10-14 11:02 Asia/Shanghai): source seed0/path reached epoch 25/60, best `0.022170161670`, latest `0.025897626639`, 65,000 updates and 3,221.9 s (`0.89497` GPU-h; five stale epochs). Terminal ledger remains `11.706041885620` GPU-h; estimated live exploration balance is `59.39898` GPU-h. Confirmation matrix code now records the deterministic query IC generator, NumPy version and exact HR solver protocol; no query rollouts were generated.

Progress update (2026-10-14 11:04 Asia/Shanghai): source seed0/path reached epoch 26/60, best validation `0.020828590123`, latest equal to best, 67,500 updates and 3,345.6 s (`0.92934` GPU-h; zero stale epochs). Terminal ledger remains `11.706041885620` GPU-h; estimated live exploration balance is `59.36462` GPU-h pending task completion. Query-worker verifier import was moved after independent stdlib hash validation; targeted static checks passed, no query data was generated.

Progress update (2026-10-14 11:07 Asia/Shanghai): source seed0/path reached epoch 27/60, best `0.020828590123`, latest `0.027335827455`, 70,000 updates and 3,469.3 s (`0.96370` GPU-h; one stale epoch). Terminal ledger remains `11.706041885620` GPU-h; live estimated exploration balance is `59.33026` GPU-h pending completion. Worker identity checks now include path/schema downgrade cases; the query freezer will reject known pre-unlock query outputs. No solver/query data was generated.

Progress update (2026-10-14 11:10 Asia/Shanghai): source seed0/path reached epoch 29/60, best `0.020828590123`, latest `0.023294494523`, 75,000 updates and 3,717.2 s (`1.03255` GPU-h; three stale epochs). Terminal ledger remains `11.706041885620` GPU-h; live estimated exploration balance is `59.26141` GPU-h pending task completion. Environment fingerprint capture used CPU-only package/config inspection; no solver rollouts or extra GPU hours were incurred.

Progress update (2026-10-14 11:13 Asia/Shanghai): source seed0/path reached epoch 30/60, best `0.020828590123`, latest `0.033007201679`, 77,500 updates and 3,842.6 s (`1.06740` GPU-h; four stale epochs). Terminal ledger remains `11.706041885620` GPU-h; live estimated exploration balance is `59.22656` GPU-h until process completion. The minimum-epoch gate is met; patience is not. No additional GPU task or query rollout was started.

Progress update (2026-10-14 11:17 Asia/Shanghai): source seed0/path reached epoch 32/60, best `0.019571552367`, latest `0.020264994977`, 82,500 updates and 4,089.6 s (`1.13600` GPU-h; one stale epoch). Terminal ledger remains `11.706041885620` GPU-h; live estimated exploration balance is `59.15796` GPU-h pending completion. Query-worker HR generation now permits only canonical development cases or a gated confirmation query; no GPU work or query data was added by this guard.

Progress update (2026-10-14 11:18 Asia/Shanghai): source seed0/path reached epoch 33/60, best `0.017241928730`, 85,000 updates and 4,213.3 s (`1.17035` GPU-h; zero stale epochs). Terminal ledger remains `11.706041885620` GPU-h; live estimated exploration balance is `59.12361` GPU-h. Static query-worker changes did not consume GPU hours.

Live budget note (2026-10-14 11:21 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h. The active seed0/path source snapshot has `4,336.7` elapsed seconds (`1.20465` GPU-h), giving estimated exploration balance `59.08931` GPU-h of 72; this charge remains provisional until the source task exits. The 48 GPU-h confirmation reserve is untouched.

Live budget note (2026-10-14 11:25 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h. The active seed0/path source snapshot has `4,584.4` elapsed seconds (`1.27344` GPU-h), giving estimated exploration balance `59.02052` GPU-h of 72; this remains provisional until the source task exits. Confirmation reserve remains 48 GPU-h. Static gating work added no GPU charge.

Live budget note (2026-10-14 11:27 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h. Active source snapshot: `4,708.1` s (`1.30782` GPU-h); estimated exploration balance `58.98614` GPU-h of 72. The 48 GPU-h reserve is untouched. Independent static review consumed no GPU time.

Live budget note (2026-10-14 11:29 Asia/Shanghai): terminal ledger `11.706041885620` GPU-h; active source snapshot `4,831.8` s (`1.34217` GPU-h), estimated exploration balance `58.95178` GPU-h of 72. The 48 GPU-h confirmation reserve remains untouched.

Live budget note (2026-10-14 11:31 Asia/Shanghai): terminal ledger is `11.706041885620` GPU-h. Active source elapsed is `4,955.3` s (`1.37648` GPU-h), leaving an estimated `58.91748` GPU-h in the exploration allocation; the 48 GPU-h confirmation reserve is untouched.

Live budget note (2026-10-14 11:33 Asia/Shanghai): terminal campaign usage remains `11.706041885620` GPU-h. The active source run has `5,081.2` s (`1.41145` GPU-h) elapsed, leaving estimated exploration balance `58.88251` GPU-h; the 48 GPU-h confirmation reserve is untouched.

Live budget note (2026-10-14 11:35 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h. Active source elapsed `5,205.1` s (`1.44586` GPU-h) leaves estimated exploration balance `58.84810` GPU-h of 72. The 48 GPU-h confirmation reserve remains untouched.

Live budget note (2026-10-14 11:37 Asia/Shanghai): terminal campaign usage `11.706041885620` GPU-h; active source elapsed `5,328.9` s (`1.48026` GPU-h); estimated exploration balance `58.81370` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 11:39 Asia/Shanghai): terminal campaign usage `11.706041885620` GPU-h; active elapsed `5,452.9` s (`1.51469` GPU-h); estimated exploration balance `58.77927` GPU-h. The 48 GPU-h confirmation reserve remains untouched.

Live budget note (2026-10-14 11:43 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h. Active source elapsed is `5,700.5` s (`1.58348` GPU-h), estimated exploration balance `58.71048` GPU-h of 72. The 48 GPU-h confirmation reserve is untouched.

Live budget note (2026-10-14 11:45 Asia/Shanghai): terminal ledger `11.706041885620` GPU-h; active elapsed `5,824.5` s (`1.61790` GPU-h), estimated exploration balance `58.67605` GPU-h. Confirmation reserve remains untouched at 48 GPU-h.

Live budget note (2026-10-14 11:48 Asia/Shanghai): terminal ledger is `11.706041885620` GPU-h. Active source elapsed `5,948.4` s (`1.65233` GPU-h) gives estimated exploration balance `58.64163` GPU-h; 48 GPU-h remain reserved for confirmation.

Live budget note (2026-10-14 11:50 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h. Active source elapsed is `6,072.2` s (`1.68673` GPU-h), leaving estimated exploration balance `58.60722` GPU-h. The 48 GPU-h reserve remains untouched.

Live budget note (2026-10-14 11:52 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h; active source elapsed `6,196.0` s (`1.72111` GPU-h), estimated exploration balance `58.57285` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 11:54 Asia/Shanghai): terminal campaign ledger `11.706041885620` GPU-h; active elapsed `6,321.6` s (`1.75599` GPU-h), estimated exploration balance `58.53797` GPU-h; confirmation reserve 48 GPU-h is untouched.

Live budget note (2026-10-14 11:58 Asia/Shanghai): terminal campaign usage `11.706041885620` GPU-h. Active source elapsed `6,568.2` s (`1.82450` GPU-h) gives estimated exploration balance `58.46946` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:00 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h; active source elapsed `6,691.4` s (`1.85873` GPU-h), estimated exploration balance `58.43525` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:02 Asia/Shanghai): terminal campaign usage `11.706041885620` GPU-h; active elapsed `6,815.1` s (`1.89310` GPU-h), estimated exploration balance `58.40088` GPU-h. The 48 GPU-h reserve remains untouched.

Live budget note (2026-10-14 12:04 Asia/Shanghai): terminal campaign usage is `11.706041885620` GPU-h. Active source elapsed `6,938.7` s (`1.92743` GPU-h) leaves estimated exploration balance `58.36655` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:06 Asia/Shanghai): terminal campaign usage remains `11.706041885620` GPU-h. Active source elapsed `7,062.7` s (`1.96185` GPU-h) gives estimated exploration balance `58.33210` GPU-h. The 48 GPU-h confirmation reserve is untouched.

Live budget note (2026-10-14 12:08 Asia/Shanghai): terminal ledger remains `11.706041885620` GPU-h; active source elapsed `7,186.6` s (`1.99628` GPU-h), estimated exploration balance `58.29868` GPU-h. Confirmation reserve is untouched at 48 GPU-h.

Terminal source-task charge (2026-10-14 12:12 Asia/Shanghai): seed0/path completed at `2.068093906690` GPU-h. Terminal campaign ledger is now `13.774135792310` GPU-h, leaving `58.225864207690` GPU-h in the 72-hour exploration allocation before the active seed0/final run. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:13 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active final-only source elapsed `83.8` s (`0.02328` GPU-h), estimated exploration balance `58.20258` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:16 Asia/Shanghai): terminal ledger is `13.774135792310` GPU-h; active seed0/final elapsed `251.4` s (`0.06983` GPU-h), estimated exploration balance `58.15604` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:18 Asia/Shanghai): terminal campaign use `13.774135792310` GPU-h; active source elapsed `418.7` s (`0.11629` GPU-h), estimated exploration balance `58.10957` GPU-h. The confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:19 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `502.6` s (`0.13961` GPU-h), estimated exploration balance `58.08626` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:21 Asia/Shanghai): terminal use `13.774135792310` GPU-h; active seed0/final elapsed `586.3` s (`0.16285` GPU-h), estimated exploration balance `58.06301` GPU-h. The 48 GPU-h confirmation reserve remains untouched.

Live budget note (2026-10-14 12:23 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `753.7` s (`0.20935` GPU-h), estimated exploration balance `58.01651` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:26 Asia/Shanghai): terminal use `13.774135792310` GPU-h; active seed0/final elapsed `922.9` s (`0.25637` GPU-h), estimated exploration balance `57.96949` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:28 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `1,090.2` s (`0.30284` GPU-h), estimated exploration balance `57.92302` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:30 Asia/Shanghai): terminal use remains `13.774135792310` GPU-h; active source elapsed `1,257.8` s (`0.34939` GPU-h), estimated exploration balance `57.87647` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:32 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `1,425.2` s (`0.39589` GPU-h), estimated exploration balance `57.82998` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:34 Asia/Shanghai): terminal campaign use `13.774135792310` GPU-h; active source elapsed `1,592.7` s (`0.44242` GPU-h), estimated exploration balance `57.78345` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:36 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active seed0/final elapsed `1,762.3` s (`0.48952` GPU-h), estimated exploration balance `57.73634` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:38 Asia/Shanghai): terminal campaign usage `13.774135792310` GPU-h; active seed0/final elapsed `1,930.1` s (`0.53613` GPU-h), estimated exploration balance `57.68973` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:40 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active elapsed `2,097.8` s (`0.58273` GPU-h), estimated exploration balance `57.64314` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:42 Asia/Shanghai): terminal campaign usage `13.774135792310` GPU-h; active source elapsed `2,265.8` s (`0.62938` GPU-h), estimated exploration balance `57.59648` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:44 Asia/Shanghai): terminal campaign ledger `13.774135792310` GPU-h; active source elapsed `2,433.5` s (`0.67597` GPU-h), estimated exploration balance `57.54989` GPU-h. The confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:46 Asia/Shanghai): terminal campaign use `13.774135792310` GPU-h; active source elapsed `2,603.3` s (`0.72313` GPU-h), estimated exploration balance `57.50270` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:48 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `2,770.9` s (`0.76968` GPU-h), estimated exploration balance `57.45618` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:51 Asia/Shanghai): terminal campaign use `13.774135792310` GPU-h; active source elapsed `3,022.2` s (`0.83951` GPU-h), estimated exploration balance `57.38636` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:54 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `3,273.7` s (`0.90937` GPU-h), estimated exploration balance `57.31649` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 12:57 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `3,527.4` s (`0.97982` GPU-h), estimated exploration balance `57.24604` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:00 Asia/Shanghai): terminal campaign usage `13.774135792310` GPU-h; active source elapsed `3,778.8` s (`1.04967` GPU-h), estimated exploration balance `57.17619` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:02 Asia/Shanghai): terminal campaign ledger `13.774135792310` GPU-h; active source elapsed `4,114.0` s (`1.14278` GPU-h), estimated exploration balance `57.08308` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:04 Asia/Shanghai): terminal use `13.774135792310` GPU-h; active source elapsed `4,283.9` s (`1.18998` GPU-h), estimated exploration balance `57.03588` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:07 Asia/Shanghai): terminal ledger is `13.774135792310` GPU-h; active source elapsed `4,535.4` s (`1.25983` GPU-h), estimated exploration balance `56.96604` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:10 Asia/Shanghai): terminal ledger `13.774135792310` GPU-h; active source elapsed `4,786.7` s (`1.32965` GPU-h), estimated exploration balance `56.89622` GPU-h. Confirmation reserve remains 48 GPU-h.

Terminal source-task charge (2026-10-14 13:36 Asia/Shanghai): seed0/final used `1.402507711082` GPU-h. Combined with seed0/path, terminal campaign ledger is `15.176643503393` GPU-h; exploration balance is `56.823356496608` GPU-h before active seed1/path elapsed time. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:40 Asia/Shanghai): terminal campaign ledger `15.176643503393` GPU-h; active seed1/path elapsed `123.5` s (`0.03431` GPU-h), estimated exploration balance `56.78905` GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:43 Asia/Shanghai): terminal campaign use `15.176643503393` GPU-h; active seed1/path elapsed `370.6` s (`0.10293` GPU-h), estimated exploration balance `56.72043` GPU-h. Confirmation reserve remains 48 GPU-h.

Interrupted source attempt (2026-10-14 13:47 Asia/Shanghai): seed1/path attempt1 charged conservatively at 662 s = 0.183888888889 GPU-h (runtime through terminal process check), excluded from ranking. Added to prior terminal use 15.176643503393 GPU-h: campaign use 15.360532392282 GPU-h; exploration balance 56.639467607718 of 72 GPU-h before retry; 48 GPU-h confirmation reserve remains. Process exit cause is unknown.

Retry live (2026-10-14 13:49 Asia/Shanghai): seed1/path `retry1` owns GPU0; reserve 48 GPU-h unchanged. Terminal campaign use is 15.360532392282 GPU-h, before this retry runtime; exploration remaining before it 56.639467607718 GPU-h.

Live budget note (2026-10-14 13:51 Asia/Shanghai): terminal usage before retry is 15.360532392282 GPU-h; retry1 elapsed 0.0342587209 GPU-h; estimated exploration remaining 56.6052088868 of 72 GPU-h. Confirmation reserve is 48 GPU-h.

Live budget note (2026-10-14 13:52 Asia/Shanghai): terminal campaign use before retry 15.360532392282 GPU-h; retry1 elapsed 0.0685400826; estimated exploration remaining 56.5709275251 GPU-h. Prior pilot task timings are 63–83 s/path-target task and ~499 s/ordinary full-FT task; full pilot actuals will set the confirmation capacity estimate.

Live budget note (2026-10-14 13:55 Asia/Shanghai): terminal campaign use before retry 15.360532392282 GPU-h; retry1 elapsed 0.1027848421; estimated exploration remaining 56.5366827656 GPU-h. Confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 13:57 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.1370594046; estimated exploration remaining 56.5024082031 GPU-h.

Live budget note (2026-10-14 14:00 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.1713575566; estimated exploration remaining 56.4681100511 GPU-h.

Live budget note (2026-10-14 14:01 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.2058604743; estimated exploration remaining 56.4336071334 GPU-h.

Live budget note (2026-10-14 14:04 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.2401370011; estimated exploration remaining 56.3993306066 GPU-h.

Live budget note (2026-10-14 14:05 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.2743777045; estimated exploration remaining 56.3650899032 GPU-h.

Live budget note (2026-10-14 14:07 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.3086891135; estimated exploration remaining 56.3307784942 GPU-h.

Live budget note (2026-10-14 14:10 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.3428979050; estimated exploration remaining 56.2965697027 GPU-h.

Live budget note (2026-10-14 14:11 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.3777597681; estimated exploration remaining 56.2617078396 GPU-h.

Live budget note (2026-10-14 14:14 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.4121472919; estimated exploration remaining 56.2273203158 GPU-h.

Live budget note (2026-10-14 14:16 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.4465609198; estimated exploration remaining 56.1929066879 GPU-h.

Live budget note (2026-10-14 14:18 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.4810740050; estimated exploration remaining 56.1583936027 GPU-h.

Live budget note (2026-10-14 14:20 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.5155705247; estimated exploration remaining 56.1238970831 GPU-h.

Live budget note (2026-10-14 14:22 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.5499082661; estimated exploration remaining 56.0895593416 GPU-h.

Live budget note (2026-10-14 14:24 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.5842724725; estimated exploration remaining 56.0551951353 GPU-h.

Live budget note (2026-10-14 14:27 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.6186305977; estimated exploration remaining 56.0208370101 GPU-h.

Live budget note (2026-10-14 14:29 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.6530170980; estimated exploration remaining 55.9864505097 GPU-h.

Live budget note (2026-10-14 14:31 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.6873623459; estimated exploration remaining 55.9521052618 GPU-h.

Live budget note (2026-10-14 14:33 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.7222836115; estimated exploration remaining 55.9171839962 GPU-h.

Live budget note (2026-10-14 14:34 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.7565037154; estimated exploration remaining 55.8829638923 GPU-h.

Live budget note (2026-10-14 14:36 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.7907433644; estimated exploration remaining 55.8487242434 GPU-h.

Live budget note (2026-10-14 14:39 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.8249317369; estimated exploration remaining 55.8145358708 GPU-h.

Live budget note (2026-10-14 14:41 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.8592123955; estimated exploration remaining 55.7802552122 GPU-h.

Live budget note (2026-10-14 14:43 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.8934257338; estimated exploration remaining 55.7460418739 GPU-h.

Live budget note (2026-10-14 14:45 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.9278053547; estimated exploration remaining 55.7116622531 GPU-h.

Live budget note (2026-10-14 14:47 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.9620981669; estimated exploration remaining 55.6773694408 GPU-h.

Live budget note (2026-10-14 14:49 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 0.9964720113; estimated exploration remaining 55.6429955964 GPU-h.

Live budget note (2026-10-14 14:51 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.0307914734; estimated exploration remaining 55.6086761343 GPU-h.

Live budget note (2026-10-14 14:53 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.0656194054; estimated exploration remaining 55.5738482023 GPU-h.

Live budget note (2026-10-14 14:55 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.0999184883; estimated exploration remaining 55.5395491194 GPU-h.

Live budget note (2026-10-14 14:57 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.1341683351; estimated exploration remaining 55.5052992727 GPU-h.

Live budget note (2026-10-14 14:59 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.1684602797; estimated exploration remaining 55.4710073280 GPU-h.

Live budget note (2026-10-14 15:01 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.2027032288; estimated exploration remaining 55.4367643789 GPU-h.

Live budget note (2026-10-14 15:03 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.2369934491; estimated exploration remaining 55.4024741586 GPU-h.

Live budget note (2026-10-14 15:05 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.2713095678; estimated exploration remaining 55.3681580399 GPU-h.

Live budget note (2026-10-14 15:08 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.3056383744; estimated exploration remaining 55.3338292333 GPU-h.

Live budget note (2026-10-14 15:10 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.3400233015; estimated exploration remaining 55.2994443062 GPU-h.

Live budget note (2026-10-14 15:11 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.3743502216; estimated exploration remaining 55.2651173861 GPU-h.

Live budget note (2026-10-14 15:13 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.4091723031; estimated exploration remaining 55.2302953046 GPU-h.

Live budget note (2026-10-14 15:16 Asia/Shanghai): terminal use before retry 15.360532392282 GPU-h; retry1 elapsed 1.4434826219; estimated exploration remaining 55.1959849858 GPU-h.

Terminal interrupted retry1 charge (2026-10-14 15:18 Asia/Shanghai): 5,403 s = 1.500833333333 GPU-h upper bound; total campaign use 16.861365725615 GPU-h, exploration balance 55.138634274385 of 72 GPU-h before recovery resume, confirmation reserve 48 GPU-h. Recovery is retained; the attempt is excluded from ranking.

Live budget note (2026-10-14 15:21 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exploration balance 55.138634274385 GPU-h before recovery resume; confirmation reserve 48 GPU-h. Detached queue supervisor PID 418560, current recovery worker PID 418565.

Live budget note (2026-10-14 15:24 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; recovery resume elapsed 0.0348291934; estimated exploration remaining 55.1038050810 GPU-h. The resume cost is incremental after the interrupted attempt charge; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:26 Asia/Shanghai): terminal campaign use before recovery resume 16.861365725615 GPU-h; incremental recovery elapsed 0.0700754530; estimated exploration remaining 55.0685588214 GPU-h.

Live budget note (2026-10-14 15:28 Asia/Shanghai): terminal use before resume 16.861365725615 GPU-h; incremental recovery elapsed 0.1049407713; exploration remaining estimate 55.0336935031 GPU-h.

Live budget note (2026-10-14 15:29 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; recovery worker process age 456 s (~0.1266666667 GPU-h on its exclusive GPU0 slot), estimated exploration remaining 55.0119676077 GPU-h; confirmation reserve remains 48 GPU-h. The 377.8 s training progress counter is also preserved in the source report.

Live budget note (2026-10-14 15:31 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 610 s (~0.1694444444 GPU-h), estimated exploration remaining 54.9691898299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:33 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; recovery slot has run 708 s (~0.1966666667 GPU-h), estimated exploration remaining 54.9419676077 GPU-h; 48 GPU-h confirmation reserve remains.

Live budget note (2026-10-14 15:35 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 824 s (~0.2288888889 GPU-h), estimated exploration remaining 54.9097453855 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:37 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 941 s (~0.2613888889 GPU-h), estimated exploration remaining 54.8772453855 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:40 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1105 s (~0.3069444444 GPU-h), estimated exploration remaining 54.8316898299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:41 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1175 s (~0.3263888889 GPU-h), estimated exploration remaining 54.8112453855 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:43 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1308 s (~0.3633333333 GPU-h), estimated exploration remaining 54.7753009418 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:44 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1385 s (~0.3847222222 GPU-h), estimated exploration remaining 54.7539120528 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:45 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1457 s (~0.4047222222 GPU-h), estimated exploration remaining 54.7339120528 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:47 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1528 s (~0.4244444444 GPU-h), estimated exploration remaining 54.7141898299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:49 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1666 s (~0.4627777778 GPU-h), estimated exploration remaining 54.6758564966 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:51 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1804 s (~0.5011111111 GPU-h), estimated exploration remaining 54.6375231633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:53 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 1940 s (~0.5388888889 GPU-h), estimated exploration remaining 54.5997453855 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:56 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 2074 s (~0.5761111111 GPU-h), estimated exploration remaining 54.5625231633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:58 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 2212 s (~0.6144444444 GPU-h), estimated exploration remaining 54.5241898299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 15:59 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 2282 s (~0.6338888889 GPU-h), estimated exploration remaining 54.5047453855 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:01 Asia/Shanghai): terminal campaign use 16.861365725615 GPU-h; exclusive recovery process age 2415 s (~0.6708333333 GPU-h), estimated exploration remaining 54.4678019418 GPU-h; confirmation reserve remains 48 GPU-h.

Terminal source charge (2026-10-14 16:04 Asia/Shanghai): seed1/path recovery1 report records `0.704184163002` GPU-h. Terminal campaign use is now `17.565549888617` GPU-h; estimated exploration remaining `54.434450111383` GPU-h before active seed1/final; 48 GPU-h confirmation reserve remains untouched.

Live budget note (2026-10-14 16:07 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final active process age 235 s (~0.0652777778 GPU-h), estimated exploration remaining 54.3691723336 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:10 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final active process age 375 s (~0.1041666667 GPU-h), estimated exploration remaining 54.3302834449 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:11 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 445 s (~0.1236111111 GPU-h), estimated exploration remaining 54.3108398299 GPU-h; 48 GPU-h confirmation reserve remains.

Live budget note (2026-10-14 16:14 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 655 s (~0.1819444444 GPU-h), estimated exploration remaining 54.2525056670 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:17 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 791 s (~0.2197222222 GPU-h), estimated exploration remaining 54.2147281633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:18 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 862 s (~0.2394444444 GPU-h), estimated exploration remaining 54.1950064966 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:20 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 995 s (~0.2763888889 GPU-h), estimated exploration remaining 54.1580610522 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:22 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 1129 s (~0.3136111111 GPU-h), estimated exploration remaining 53.8447281633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:26 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 1330 s (~0.3694444444 GPU-h), estimated exploration remaining 54.0650064966 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:28 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 1463 s (~0.4063888889 GPU-h), estimated exploration remaining 54.0280610522 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:31 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 1660 s (~0.4611111111 GPU-h), estimated exploration remaining 53.9733398299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:32 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 1736 s (~0.4822222222 GPU-h), estimated exploration remaining 53.9522287188 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:34 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 1807 s (~0.5019444444 GPU-h), estimated exploration remaining 53.9316176077 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:36 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 1944 s (~0.54 GPU-h), estimated exploration remaining 53.8944501114 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:39 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2144 s (~0.5955555556 GPU-h), estimated exploration remaining 53.8388943330 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:40 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2215 s (~0.6152777778 GPU-h), estimated exploration remaining 53.8191723336 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:42 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2285 s (~0.6347222222 GPU-h), estimated exploration remaining 53.7997281633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:44 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2422 s (~0.6727777778 GPU-h), estimated exploration remaining 53.7616723336 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:45 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2496 s (~0.6933333333 GPU-h), estimated exploration remaining 53.7411176077 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:47 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2631 s (~0.7308333333 GPU-h), estimated exploration remaining 53.7036176077 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:51 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2828 s (~0.7855555556 GPU-h), estimated exploration remaining 53.6488943330 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:52 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2898 s (~0.805 GPU-h), estimated exploration remaining 53.6294501114 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:53 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 2969 s (~0.8247222222 GPU-h), estimated exploration remaining 53.6097281633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:56 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3172 s (~0.8811111111 GPU-h), estimated exploration remaining 53.5533398299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:58 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3242 s (~0.9005555556 GPU-h), estimated exploration remaining 53.5338943330 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 16:59 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3312 s (~0.92 GPU-h), estimated exploration remaining 53.5144501114 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:00 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3380 s (~0.9388888889 GPU-h), estimated exploration remaining 53.4955610005 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:03 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3590 s (~0.9972222222 GPU-h), estimated exploration remaining 53.4372281633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:04 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3660 s (~1.0166666667 GPU-h), estimated exploration remaining 53.4177834449 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:06 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3730 s (~1.0361111111 GPU-h), estimated exploration remaining 53.3983398299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:09 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3929 s (~1.0913888889 GPU-h), estimated exploration remaining 53.3430610005 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:10 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 3999 s (~1.1108333333 GPU-h), estimated exploration remaining 53.3236176077 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:12 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4141 s (~1.1502777778 GPU-h), estimated exploration remaining 53.2841723336 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:16 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4350 s (~1.2083333333 GPU-h), estimated exploration remaining 53.2261165550 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:18 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4486 s (~1.2461111111 GPU-h), estimated exploration remaining 53.1883398299 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:19 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4559 s (~1.2663888889 GPU-h), estimated exploration remaining 53.1680610005 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:22 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4694 s (~1.3038888889 GPU-h), estimated exploration remaining 53.1305610005 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:24 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4833 s (~1.3425 GPU-h), estimated exploration remaining 53.0919501114 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:25 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4904 s (~1.3622222222 GPU-h), estimated exploration remaining 53.0722281633 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:26 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 4974 s (~1.3816666667 GPU-h), estimated exploration remaining 53.0527834449 GPU-h; confirmation reserve remains 48 GPU-h.

Live budget note (2026-10-14 17:28 Asia/Shanghai): terminal campaign use 17.565549888617 GPU-h; seed1/final process age 5047 s (~1.4020 GPU-h), estimated exploration remaining 53.0322281633 GPU-h; confirmation reserve remains 48 GPU-h.

Terminal source charge (2026-10-14 17:28 Asia/Shanghai): seed1/final measured `1.401436295762` GPU-h. Terminal campaign use is now `18.966986184379` GPU-h; exploration balance is `53.033013815621` GPU-h before active seed42/path; the 48 GPU-h confirmation reserve remains untouched.


Terminal source charge (2026-10-14 19:33 Asia/Shanghai): seed42/path completed 60 epochs with measured charge `2.078342349826` GPU-h. Terminal campaign use is `21.045328534204` GPU-h; exploration balance before the active seed42/final source is `50.954671465796` GPU-h; the 48 GPU-h confirmation reserve remains untouched.

Terminal source charge (2026-10-14 20:57 Asia/Shanghai): seed42/final measured `1.403775779834` GPU-h. Terminal campaign use is now `22.449104314038` GPU-h; exploration balance before the active Stage-2 LR screen is `49.550895685962` GPU-h; the 48 GPU-h confirmation reserve remains untouched.

Stage-2 live terminal ledger (2026-10-14 21:54 Asia/Shanghai): seed42/final source is charged `1.403775779834` GPU-h. The 13 completed seed0 adaptation cells plus the failed endpoint task01 total `0.868228228718` GPU-h; the failed cell alone is `0.003062378770` GPU-h. Newly charged campaign use is `23.317332542757` GPU-h, leaving `48.682667457243` GPU-h in the 72-GPU-h exploration allotment; the 48-GPU-h confirmation reserve remains untouched. Active target remainder is excluded until terminal task rows are written.


Stage-2 recovery accounting (2026-10-15 10:36 Asia/Shanghai): terminal target LR cells completed after the prior Stage-2 budget note add `1.653482089915` GPU-h (endpoint-increment retry1 seed0, endpoint-direction seed0, ordinary-full comm256 seed0, and lifted-scratch seed0; 24 cells). The orphaned seed1/pathsource comm256/task00 RUNNING attempt is charged conservatively through host reboot: 10h27m06s = `10.451666666667` GPU-h. With previous total `23.317332542757`, current campaign use is `35.422481299339` GPU-h; exploration balance is `36.577518700661` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched. The partial canonical attempt is excluded from results and will be replaced by the same seed/arm in a fresh namespace.


Stage-2 continuation charge (2026-10-15 11:25 Asia/Shanghai): seed1/pathsource comm256 recovery1 six terminal LR cells charged `0.742520025154` GPU-h. Campaign use is `36.165001324493` GPU-h; exploration balance is `35.834998675507` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched.

Stage-2 continuation charge (2026-10-15 11:32 Asia/Shanghai): seed1/pathsource firstpath-ratio015 six terminal LR cells charged `0.105832168281` GPU-h. Campaign use is `36.270833492774` GPU-h; exploration balance is `35.729166507226` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched.

Stage-2 continuation charge (2026-10-15 11:38 Asia/Shanghai): seed1 endpoint-increment retry1 six terminal LR cells charged `0.105512133350` GPU-h. Campaign use is `36.376345626124` GPU-h; exploration balance is `35.623654373876` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched.

Stage-2 continuation charge (2026-10-15 11:44 Asia/Shanghai): seed1 endpoint-direction-ratio015 six terminal LR cells charged `0.106539569751` GPU-h. Campaign use is `36.482885195875` GPU-h; exploration balance is `35.517114804125` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched.

Stage-2 interruption/recovery accounting (2026-10-15 12:21 Asia/Shanghai): seed1 ordinary full-FT + comm256 task00–01 terminal cells charged `0.277525144359` GPU-h. Task02 ended without a terminal report; process-age upper bound 1229 s charged as `0.341388888889` GPU-h. The incomplete canonical arm is excluded and will be rerun under recovery1; successful task00–01 entries remain separately charged. Campaign use is `37.101799229123` GPU-h; exploration balance is `34.898200770877` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched.

Stage-2 recovery charge (2026-10-15 13:12 Asia/Shanghai): seed1 ordinary full-FT + comm256 recovery1 six terminal LR cells charged `0.832853350532` GPU-h; the earlier canonical attempt charges remain recorded separately. Campaign use is `37.934652579655` GPU-h; exploration balance is `34.065347420345` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched.

Stage-2 continuation charge (2026-10-15 13:49 Asia/Shanghai): seed1 lifted scratch six terminal LR cells charged `0.614725686001` GPU-h. Campaign use is `38.549378265656` GPU-h; exploration balance is `33.450621734344` of 72 GPU-h. The 48 GPU-h confirmation/control reserve remains untouched.

Live dual-GPU update (2026-10-15 18:38 Asia/Shanghai): official `terminal_gpu_hours()` reconciliation is `42.883145342583` GPU-h; nominal remainder `77.116854657417` GPU-h comprises `29.116854657417` GPU-h exploration capacity plus the protected 48 GPU-h confirmation/control reserve. Active rows are not yet included. Physical GPU0 runs the frozen Stage-2 optimization queue (currently LR-scratch task02, epoch 383/500); physical GPU1 runs the distinct four-task HR-from-random baseline (currently task05, epoch 49/500). No seed/config/split changes. The HR support-label generation for four selected m=4 tasks used `1187.548 s` CPU wall including packaging (`405.171 s` summed solver time, `0.305 s` assembly); this cost is tracked separately and will be included in the end-to-end comparison. Query remains gated.

Live continuation (2026-10-15 18:47 Asia/Shanghai): terminal campaign usage recomputed by `terminal_gpu_hours()` is `43.113805872780` GPU-h. Nominal remainder is `76.886194127220` GPU-h (`28.886194127220` exploration and 48 reserved). GPU0's optimized LR-scratch tasks02/03 are terminal at `0.115666`/`0.114995` GPU-h; task04 is active. GPU1's HR-from-zero task05 is at 167/500 epochs, best HR-dev `0.3638010` so far versus matched path-source `0.0513598`. It remains a same-seed, same-support, development-only four-task baseline; no test/query data touched.

Live continuation (2026-10-15 18:53 Asia/Shanghai): terminal GPU usage is `43.229863323318` GPU-h; `28.770136676682` exploration GPU-h and the protected 48 GPU-h reserve remain. GPU0 completed optimized LR-scratch task04 at `0.116057` GPU-h and is running task05. GPU1's HR-from-zero task05 has reached epoch 241/500, with best HR-dev `0.3502571`. All four HR labels remain one-time CPU cost `1187.548 s` end-to-end; no query data was accessed.

Live update (2026-10-15 19:04 Asia/Shanghai): official terminal campaign charge is `43.347168698073` GPU-h; nominal remainder is `76.652831301927` GPU-h (`28.652831301927` exploration plus the protected 48 GPU-h confirmation/control reserve). Stage-2 optimization tasks00–05 are complete and audited at exact per-epoch/model-state parity; full method runtimes and contexts are in `artifacts/reports/phase1_syncopt2_parity_speed_summary_20261015.json`. Direct HR scratch jobs now run different tasks in parallel: task03 on GPU0, task05 on GPU1. Active rows are not yet charged. The target HR labels cost `1187.548 s` CPU wall, separate from GPU-hours.

HR scratch task05 terminal charge (2026-10-15 19:15 Asia/Shanghai): 500-epoch direct-HR fit used `0.687396242098` GPU-h, best HR-dev `0.341559250840` at epoch324, with no time-to-match against path-source `0.051359781786`. Updated official terminal use is `44.034564940171` GPU-h; exploration balance `27.965435059829` GPU-h, confirmation/control reserve remains 48 GPU-h. GPU0 task03 and GPU1 task00 are active as separate tasks; live elapsed is not in this terminal total. Add HR support-generation CPU wall `1187.548 s` separately in the end-to-end comparison.


HR scratch queue terminal charge (2026-10-15 20:34 Asia/Shanghai): all four frozen direct-HR fits completed at 500 epochs. Task03/00/01 charges after the earlier task05 update total `1.954707364271` GPU-h. Official `terminal_gpu_hours()` is now `45.989272304442` GPU-h; total nominal remaining is `74.010727695558` GPU-h, exploration headroom `26.010727695558` GPU-h, and the protected 48 GPU-h confirmation/control reserve remains intact. The 16 HR target support labels cost `1187.548 s` CPU wall including packaging; this CPU cost is tracked separately. Final by-family cost and quality comparison: `artifacts/reports/PHASE1_HR_SCRATCH_END_TO_END_20261015.md`. No confirmation query was opened.

R4 live budget update (2026-10-15 23:45 Asia/Shanghai): the official append-only ledger is `48.435457132849` GPU-h after three R4 smokes and A1/A2 seed0. Total nominal remainder is `71.564542867151` GPU-h, with `23.564542867151` exploration headroom and the 48 GPU-h confirmation/control reserve untouched. R4 A1/A2 seed0 target training cost is `1.208605`/`1.217631` GPU-h; the paired development HR evaluations have not started. At observed cell costs, remaining R4 cells are projected to fit its 8 GPU-h amended cap. This lower bound still omits untimed historic HR-dev overhead.

R4 live update (2026-10-16 01:01 Asia/Shanghai): official ledger charge is `49.702356921008` GPU-h after A1 seeds0/1 and A2 seed0. Exploration remainder is `22.297643078992` GPU-h; the 48 GPU-h confirmation/control reserve remains intact. The remaining 12 target cells are projected at about `2.45` GPU-h from measured cell cost, within the amended 8 GPU-h R4 cap. HR-dev evaluation is still pending and will follow LR-only checkpoint selection.

R4 live update (2026-10-16 01:13 Asia/Shanghai): official terminal ledger is `49.916186344226` GPU-h after A2 seed1/task00; that cell used `0.213829423218` GPU-h (`769.786 s`), selected LR checkpoint epoch39 at `0.02285532`. Nominal remaining is `70.083813655774` GPU-h, of which `22.083813655774` is exploration and 48 remains reserved. A2 seed1/task01 is active on GPU0 and not yet charged; the R4 HR-dev queue remains gated on all LR cells. The separate development-only HR-scratch m=16 task05 is also active on GPU1 at epoch425/500 after `7955.1 s` engine elapsed (`2.21` GPU-h-equivalent so far, not terminally charged). R4 has 17 target cells remaining after task00; at the measured `0.214` GPU-h/cell, about `3.64` GPU-h remains, within its 8-GPU-h card cap. Do not treat the nominal exploration balance as free capacity until the active m=16 job and its remaining frozen tasks are charged.

Live process update (2026-10-16 01:20 Asia/Shanghai): R4 A2 seed1/task01 is at epoch56/100, best LR validation `0.02901856`. HR scratch m=16 task05 reached epoch457/500 with provisional best HR-dev `0.1026967`, `8538.3 s` engine elapsed. These GPU-hour charges remain pending terminal task records; the official total is still the 01:13 snapshot `49.916186344226` GPU-h.

R4 live charge (2026-10-16 01:26 Asia/Shanghai): A2 seed1/task01 is terminal at `0.204659344510` GPU-h, LR validation `0.02901856` (best epoch31). Official terminal campaign use is `50.120845688736` GPU-h; exploration balance `21.879154311264` GPU-h, 48 reserved. Task02/seed1 is active on GPU0. HR scratch m=16 task05 is at epoch480/500 after `8878.4 s` engine elapsed; it remains a pending GPU-hour charge. R4 has 16 target cells left; the observed mean terminal cost across its 20 completed cells is `0.205581` GPU-h/cell, or about `3.29` GPU-h at that rate.

Process update (2026-10-16 01:28 Asia/Shanghai): R4 A2 seed1/task02 reached epoch19/100, LR validation best `0.02042858`. HR scratch m=16 task05 reached epoch488/500, provisional HR-dev best `0.1026967`. The full 64-rollout HR m=16 support-label generation cost is now verified at `4786.624 s` CPU wall including packaging; training GPU charges remain terminally unposted until task completion.

HR-scratch m=16 terminal update (2026-10-16 01:32 Asia/Shanghai): task05 used `2.545328593614` GPU-h at 500 epochs, selecting best epoch442 with HR-dev `0.102696693154`; the matched path-source reference is `0.051359781786`, and no epoch reached it. Official terminal ledger is `52.666174282349` GPU-h, exploration balance `19.333825717651` GPU-h, confirmation reserve 48. Task00 on GPU1 has started. R4 A2 seed1/task02 is active on GPU0 at epoch51/100; terminal ledger does not include either active cell.

R4/m=16 live update (2026-10-16 01:38 Asia/Shanghai): A2 seed1/task02 completed at `0.202567522252` GPU-h, LR validation `0.01835452` (best epoch99). Official terminal ledger is `52.868741804601` GPU-h, with `19.131258195399` GPU-h exploration and 48 reserved. R4 has 15 target cells remaining; mean charge over its 21 completed cells is `0.205438` GPU-h/cell, projecting `3.08` GPU-h at the observed rate. A2 task03/seed1 is active on GPU0; HR scratch m=16 task00 is active on GPU1 at epoch29/500, provisional HR-dev `0.2786980`, `443.6 s` engine elapsed. Neither active process is charged.

R4/m=16 live update (2026-10-16 01:50 Asia/Shanghai): A2 seed1/task03 completed at `0.202695874220` GPU-h, LR validation `0.02429196` (best epoch96). Official terminal ledger is `53.071437678821` GPU-h, with `18.928562321179` GPU-h exploration and 48 reserved. R4 has 14 target cells remaining; mean charge over its 22 completed cells is approximately `0.205313` GPU-h/cell, projecting `2.87` GPU-h. A2 task04/seed1 is active on GPU0. HR scratch m=16 task00 is active on GPU1 at epoch80/500, provisional HR-dev `0.1878847`, engine elapsed `1197.7 s`; terminal ledger excludes both active cells.

R4/m=16 live update (2026-10-16 02:03 Asia/Shanghai): A2 seed1/task04 completed at `0.202824966354` GPU-h, LR validation `0.01949338` (best epoch40). Official terminal ledger is `53.274262645176` GPU-h, with `18.725737354824` GPU-h exploration and 48 reserved. R4 has 13 cells remaining; mean cost over 23 completed cells is `0.205205` GPU-h/cell, projecting `2.67` GPU-h. Task05/seed1 is active on GPU0. HR scratch m=16 task00 is active on GPU1 at epoch128/500, provisional HR-dev `0.1841019`, engine elapsed `1911.3 s`; pending jobs are not included in terminal charge.

R4/m=16 live update (2026-10-16 02:15 Asia/Shanghai): A2 seed1/task05 completed at `0.202755803416` GPU-h, LR validation `0.02029603` (best epoch99). Official terminal ledger is `53.477018448592` GPU-h, with `18.522981551408` GPU-h exploration and 48 reserved. A2 seed1's six-task LR mean is `0.022384962736`; R4 has 12 cells remaining and the observed mean over 24 completed R4 cells is `0.205103` GPU-h/cell, projecting `2.46` GPU-h. A1 seed42/task00 is active on GPU0 at epoch11/100; HR scratch m=16 task00 is active on GPU1 at epoch181/500. Neither active cell is terminally charged.

R4/m=16 live update (2026-10-16 02:27 Asia/Shanghai): R4 A1 seed42/task00 completed at `0.203002648607` GPU-h, LR validation `0.02005800` (best epoch34). Official terminal ledger is `53.680021097199` GPU-h; exploration balance `18.319978902801` GPU-h, reserve48. R4 has 11 cells remaining; mean cost over 25 completed cells is about `0.205019` GPU-h/cell, projecting `2.26` GPU-h. Task01/seed42 is active on GPU0. HR scratch m=16 task00 is active on GPU1 at epoch226/500 with provisional HR-dev `0.1630438`; neither active run is terminally charged.

R4/m=16 live update (2026-10-16 02:39 Asia/Shanghai): R4 A1 seed42/task01 completed at `0.202641691385` GPU-h, LR validation `0.02229431` (best epoch82). Official terminal ledger is `53.882662788584` GPU-h; exploration balance `18.117337211416` GPU-h, reserve48. R4 has 10 cells remaining; mean cost over 26 completed cells is `0.204927` GPU-h/cell, projecting `2.05` GPU-h. Task02/seed42 is active on GPU0. HR scratch m=16 task00 is active on GPU1 at epoch275/500 with provisional HR-dev `0.1630438`, engine elapsed `4085.8 s`; neither active process is terminally charged.

R4/m=16 live update (2026-10-16 02:51 Asia/Shanghai): R4 A1 seed42/task02 completed at `0.202667380700` GPU-h, LR validation `0.01824379` (best epoch98). Official terminal ledger is `54.085330169284` GPU-h; exploration balance `17.914669830716` GPU-h, reserve48. R4 has 9 cells remaining; mean cost over 27 completed cells is about `0.204844` GPU-h/cell, projecting `1.84` GPU-h. Task03/seed42 is active on GPU0. HR scratch m=16 task00 is active on GPU1 at epoch324/500, provisional HR-dev `0.1602344`, engine elapsed `4808.6 s`; pending task costs are uncharged.

R4/m=16 live update (2026-10-16 03:03 Asia/Shanghai): R4 A1 seed42/task03 completed at `0.202778679157` GPU-h, LR validation `0.01817778` (best epoch97). Official terminal ledger is `54.288108848442` GPU-h; exploration balance `17.711891151558` GPU-h, reserve48. R4 has 8 cells remaining; mean cost over 28 completed cells is approximately `0.204770` GPU-h/cell, projecting `1.64` GPU-h. Task04/seed42 is active on GPU0. HR scratch m=16 task00 is active on GPU1 at epoch373/500, provisional HR-dev `0.1587605`, engine elapsed `5530.1 s`; active charges remain unposted.

R4/m=16 live update (2026-10-16 03:15 Asia/Shanghai): R4 A1 seed42/task04 completed at `0.202626602352` GPU-h, LR validation `0.01783190` (best epoch56). Official terminal ledger is `54.490735450794` GPU-h; exploration balance `17.509264549206` GPU-h, reserve48. R4 has 6 target cells remaining; mean charge over 29 completed cells is about `0.204696` GPU-h/cell, projecting `1.23` GPU-h. A1 task05/seed42 is active on GPU0. HR scratch m=16 task00 is active on GPU1 at epoch423/500, provisional HR-dev `0.1587605`, engine elapsed `6273.3 s`; active work is not yet terminally charged.

## Live queue reconciliation (2026-10-16 03:29 Asia/Shanghai)

The official terminal ledger is `54.693615076374 GPU-h`, leaving `17.306384923626 GPU-h` for unreserved exploration; the 48 GPU-h confirmation/control reserve remains intact. The fresh accounting output is `artifacts/reports/phase1_progress_r4_full_a1_seed42_20261016_0329.json`.

R4 has 30/36 new LR cells complete. Six A2/seed42 cells remain, with task00 active; their projected cost is about `1.23 GPU-h` at the observed per-cell rate. The completion-gated HR-dev queue has 36 new evaluations; existing evaluations take about 2.7 seconds each, so their projected GPU cost is about `0.03 GPU-h`. HR-scratch m=16 has task05 terminal at `2.545329 GPU-h`; task00 is still active (last verified at epoch473/500, provisional HR-dev `0.1587605`) and tasks01/03 remain queued. Conservatively projecting three remaining fits at the observed task05 cost gives about `7.64 GPU-h` pending. Thus the current queued work is projected to leave roughly `8.4 GPU-h` of exploration capacity, with runtime variation possible; no new search has been scheduled against that balance. No confirmation query was generated or read.

## Live queue update (2026-10-16 03:35 Asia/Shanghai)

After HR-scratch m=16/task00 terminated, official ledger use is `56.750629143217 GPU-h`; nominal total remainder is `63.249370856783 GPU-h`, with `15.249370856783 GPU-h` unreserved for exploration and the 48 GPU-h confirmation/control reserve intact. Accounting output: `artifacts/reports/phase1_progress_after_m16_task00_20261016_0335.json`.

Task00 cost `2.057014 GPU-h`, best epoch364, HR-dev `0.15876050`; its matched path-source result is `0.06831793`, so direct scratch error is 2.324× higher. Task01 is active and task03 queued. Using the larger completed m=16 fit cost (`2.545329 GPU-h`) for each of the two remaining fits gives a conservative `5.09 GPU-h` projection. R4 remains at 30/36 new LR cells, with six A2/seed42 cells left (~`1.23 GPU-h` at observed rate), followed by ~`0.03 GPU-h` for its 36 HR-dev evaluations. Pending frozen work is therefore about `6.35 GPU-h`, leaving roughly `8.9 GPU-h` of exploration capacity if runtime stays within this projection. No confirmation query was generated or read.

## Ledger reconciliation and active queue (2026-10-16 03:39 Asia/Shanghai)

A read-only budget audit found two abandoned attempts whose last ledger rows still said `RUNNING` despite no live process or final result. Append-only terminal rows now charge the measurable progress time: R1 source continuation epoch50/60, `8961.294399 s` (`2.489248 GPU-h`), and R3 first-path task03 epoch72/100, `41.025256 s` (`0.011396 GPU-h`). These are progress-time lower bounds, not accepted metrics; neither attempt has a final result/checkpoint. This adds `2.500644 GPU-h` to the prior ledger and preserves both original rows.

Official ledger use is now `59.251273492076 GPU-h`; nominal remainder `60.748726507924 GPU-h`, unreserved exploration `12.748726507924 GPU-h`, and 48 GPU-h confirmation/control reserve unchanged. Fresh accounting output: `artifacts/reports/phase1_progress_after_stale_rows_20261016_0339.json`.

After the two stale charges, conservatively expected pending R4 training + its HR-dev pass and m=16 task01/task03 is `6.37–6.56 GPU-h`, leaving `6.19–6.38 GPU-h` unreserved if estimates hold. There is also unquantified early HR-dev evaluator overhead, so this remainder is not fully spendable with confidence. The R4 gate remains closed pending all LR cells; m=16 task01 is active and task03 is queued. No confirmation query was generated or read.


## Live ledger refresh (2026-10-16 03:42 Asia/Shanghai)

R4 A2 seed42/task00 has completed at `0.202887210506 GPU-h`, LR validation `0.02015377` (best epoch33). This was appended after the 03:39 stale-row snapshot. The fresh official terminal total is `59.454160702583 GPU-h`, nominal remainder `60.545839297417 GPU-h`, exploration capacity `12.545839297417 GPU-h`, with 48 GPU-h protected. Snapshot: `artifacts/reports/phase1_progress_active_20261016_0342.json`.

R4 now has 31/36 terminal LR cells; task01/seed42 is active at epoch18/100 and four additional tasks follow. The five remaining LR cells are projected at about `1.02 GPU-h`; the frozen 36-checkpoint HR-dev pass is projected at `0.027 GPU-h`, subject to early evaluator overhead. HR-scratch m=16 task01 is active at epoch30/500, with task03 still queued; completed m=16 task00/task05 costs are `2.057014/2.545329 GPU-h`. A conservative `5.1 GPU-h` allowance for the two remaining scratch fits yields about `6.15 GPU-h` total pending and `6.4 GPU-h` unreserved exploration afterward if these estimates hold. Do not schedule against the full projection until the terminal charges and R4 paired results arrive. Query/test remains unopened.


## Queue progress (2026-10-16 03:51 Asia/Shanghai)

R4 A2 seed42/task01 terminated at `0.202828267892 GPU-h`, LR validation `0.02224321` at epoch42. The fresh official total is `59.656988970474 GPU-h`, exploration headroom `12.343011029526 GPU-h`, reserve48 intact; snapshot `artifacts/reports/phase1_progress_r4_task01_seed42_20261016_0351.json`. R4 has 32/36 cells terminal; four remain, with task02 now active. HR-scratch m=16 task01 is at epoch71/500 and task03 remains queued.


## Queue progress (2026-10-16 04:05 Asia/Shanghai)

R4 A2 seed42/task02 is terminal at `0.202990087599 GPU-h`, LR validation `0.01827783` (best epoch98). Fresh ledger total `59.859979058073 GPU-h`; exploration balance `12.140020941927`, reserve48 intact. Snapshot: `artifacts/reports/phase1_progress_r4_task02_seed42_20261016_0405.json`. Three R4 LR cells remain, task03 active. m=16 scratch task01 is at epoch120/500, with task03 queued.


## Queue progress (2026-10-16 04:17 Asia/Shanghai)

R4 A2 seed42/task03 is terminal at `0.202631119870 GPU-h`, LR validation `0.01816410` (best epoch95). Fresh ledger total `60.062610177943 GPU-h`; exploration balance `11.937389822057`, reserve48 intact. Snapshot: `artifacts/reports/phase1_progress_r4_task03_seed42_20261016_0417.json`. Two R4 LR cells remain; task04 is active. HR-scratch m=16 task01 is at epoch169/500, with task03 queued.


## R4 LR completion and HR-dev queue (2026-10-16 04:39 Asia/Shanghai)

The final R4 A2 seed42/task05 cell terminated at `0.202562926708 GPU-h`, LR validation `0.02016054` (best epoch91). All 36 new LR cells plus six arm×seed summaries are now present. Seed42 A2 LR mean is `0.01943460`, nearly matching A1's `0.01943917`; this remains only an LR selection metric. The completion-gated HR-dev evaluator is active on GPU0 and has not yet produced paired results. Fresh terminal total `60.468819256837 GPU-h`; unreserved exploration `11.531180743163 GPU-h`, reserve48 intact. Snapshot `artifacts/reports/phase1_progress_r4_lr_complete_20261016_0439.json`. HR scratch m=16 task01 remains active at epoch268/500; task03 queued.


## R4 HR-dev completion (2026-10-16 04:45 Asia/Shanghai)

All 36 R4 HR-dev reports passed the queue verifier (`query_read=false`). Paired macro means: A0 pathsource final-only+comm256 `0.05347209`; A1 genuine first-path+comm256 `0.05387774`; A2 endpoint+comm256 `0.05363236`. A1 is 0.7586% worse than A0 and 0.4575% worse than A2, fails the frozen 1% advantage rule, and has only 1/3 paired seed wins vs A0 and 1/3 vs A2. Family wins are 1/6 vs A0, 2/6 vs A2. Both seed-level paired intervals span zero. No further path-specific search is justified from this R4 result. Full report: `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md`; machine summary `artifacts/reports/phase1_r4_pathcomm_paired_summary_20261016.json`.

Timed evaluation added `0.028249887529 GPU-h`; terminal total is `60.497069144366 GPU-h`, exploration balance `11.502930855634`, reserve48 intact. Fresh snapshot: `artifacts/reports/phase1_progress_r4_hrdev_complete_20261016_0445.json`. HR-scratch m=16 task01 remains active at epoch280/500; task03 queued. Query/test remains unopened.

## R4 closure and HR expert follow-up (2026-10-16 04:48 Asia/Shanghai)

The independent R4 audit passed; no query/test data was accessed. R4's failure closes the path-specific search branch. The active m=16 random-init HR scratch task01 is at epoch295/500 with best HR-dev `0.103308130940`; task03 remains queued. These are live progress values and are not terminal ledger charges. After the four-task scratch queue terminates, refresh the append-only ledger before scheduling the HR full-FT expert. The frozen candidate is source-initialized, 250 epochs, four tasks, using the exact same m=16 HR support and HR-dev sets. Its forecast cap is `4.7 GPU-h`; start only with at least `5.6 GPU-h` exploration remaining after terminal scratch charges, preserving a `0.9 GPU-h` operating buffer and the protected 48 GPU-h reserve. Frozen artifacts: `configs/phase1_hr_fullft_m16_dev.yaml`, `docs/experiment_cards/HR_FULLFT_M16_20261016.md`, `scripts/run_phase1_hr_fullft_m16_dev.py`. The expert is budget-limited and cannot establish saturation.

Live progress update (2026-10-16 04:51 Asia/Shanghai): m=16 HR-scratch task01 reached epoch316/500; current HR-dev `0.105484522204`, best `0.101591570093` (new low). Task03 remains queued. The active task's provisional elapsed time is excluded from terminal ledger balance until the runner records a terminal row. Preserve the 250-epoch expert dispatch gate pending both queued scratch fits and fresh reconciliation.

Live progress update (2026-10-16 04:52 Asia/Shanghai): m=16 HR-scratch task01 reached epoch323/500 with best HR-dev `0.100062042871` and current HR-dev `0.107091395231`. Task03 remains queued. Active elapsed work remains outside terminal ledger totals pending task completion.

Live progress update (2026-10-16 04:57 Asia/Shanghai): m=16 HR-scratch task01 reached epoch335/500; current HR-dev `0.104999110987`, best `0.099455473758`. Task03 remains queued. The provisional elapsed work is still excluded from terminal ledger totals.

Live progress update (2026-10-16 04:59 Asia/Shanghai): task01 of the HR-scratch m=16 queue reached epoch350/500 with current/best HR-dev `0.097096322733`. Task03 remains queued; active cost is not included in terminal-ledger balance.

Frozen optional baseline screen (2026-10-16 05:03 Asia/Shanghai): one ordinary full-FT + direct256 candidate at LR `3e-4`, seed0 across all six tasks, is budget-gated after the HR expert. Matching LR=`1e-4` training costs `0.832 GPU-h` for six tasks; timed HR-dev checks historically add <`0.005 GPU-h`. Run only if ≥`1.8 GPU-h` exploration remains after expert terminal accounting, retaining 0.9 GPU-h for cost uncertainty. This one-seed LR bracket cannot replace the three-seed baseline. No charge yet.

Live progress update (2026-10-16 05:05 Asia/Shanghai): m=16 HR-scratch task01 reached epoch369/500 with a new current/best HR-dev `0.096853650245`. Task03 remains queued; terminal ledger excludes the in-flight remainder.

Live progress update (2026-10-16 05:09 Asia/Shanghai): task01 reached epoch383/500 with latest HR-dev `0.099535053049` and best `0.096853650245` at epoch369. Task03 remains queued; no terminal budget entry yet.

Live progress update (2026-10-16 05:13 Asia/Shanghai): task01 reached epoch401/500, latest HR-dev `0.099472547590`, best `0.096464462450` at epoch388. Task03 remains queued; the remaining runtime forecast is provisional and is not a ledger charge.

Live progress update (2026-10-16 05:15 Asia/Shanghai): task01 reached epoch411/500; best HR-dev `0.096462116065` at epoch407, latest `0.098734667245`. Task03 remains queued.

Live progress update (2026-10-16 05:18 Asia/Shanghai): m=16 scratch task01 reached epoch421/500, best HR-dev `0.095592029695` at epoch416 and latest `0.097557427024`. Task03 remains queued.

Live progress update (2026-10-16 05:23 Asia/Shanghai): m=16 scratch task01 reached epoch439/500, best `0.095592029695` at epoch416 and latest `0.097359162115`; task03 remains queued. The full 500-epoch schedule is preserved.

Live progress update (2026-10-16 05:27 Asia/Shanghai): task01 reached epoch457/500; best HR-dev remains `0.095592029695` at epoch416, latest `0.097664185310`. Task03 remains queued.

Live progress update (2026-10-16 05:31 Asia/Shanghai): scratch task01 reached epoch475/500 with current HR-dev `0.096958800277` and best `0.095592029695` at epoch416. Task03 is queued after this task terminates.

Queue progress (2026-10-16 05:38 Asia/Shanghai): HR-scratch m=16 task01 is terminal at `2.051367177739 GPU-h` after 500 epochs, best HR-dev `0.095592029695` at epoch416, endpoint `0.096935368144`; its matched path-source value is `0.044732853770` and time-to-match is right-censored. Task03 has started on GPU1 at epoch0/500. Fresh official terminal charge is `62.548436322105 GPU-h`; exploration remaining `9.451563677895`, protected reserve48 intact. Snapshot: `artifacts/reports/phase1_progress_after_m16_task01_20261016_0537.json`.

## 2026-10-16 07:41 Asia/Shanghai — m=16 scratch closeout and HR expert dispatch

Fresh official ledger snapshot: `artifacts/reports/phase1_progress_after_m16_complete_20261016_0741.json`. Terminal charge is `64.588707116179 GPU-h` of 120; nominal balance `55.411292883821 GPU-h`; exploration remaining `7.411292883821 GPU-h`; the protected 48 GPU-h confirmation/control reserve remains intact. The four-task HR scratch m=16 run charged `8.693980632271 GPU-h`; target HR support generation used `4786.624 s` CPU wall. Mean best HR-dev was `0.164269032492` over tasks `[0,1,3,5]`. Task03 terminal charge is `2.040270794075 GPU-h`, HR-dev `0.300026903744`, best epoch491. The source-initialized HR full-FT m=16 expert started on GPU0 at attempt `fit1` after meeting the frozen `5.6 GPU-h` gate; cap `4.7 GPU-h`, buffer `0.9 GPU-h`. The optional one-seed LR=`3e-4` baseline screen requires at least `1.8 GPU-h` exploration remaining after the expert terminates.

## 2026-10-16 11:53 Asia/Shanghai — HR full-FT expert closeout and baseline LR screen

Fresh official ledger snapshot: `artifacts/reports/phase1_progress_after_hr_fullft_m16_complete_20261016_1153.json`. Terminal use is `68.777250620168 GPU-h` of 120; nominal balance `51.222749379832 GPU-h`; exploration headroom `3.222749379832 GPU-h`; the protected 48 GPU-h confirmation/control reserve remains intact. The source-initialized HR full-FT m=16 four-task expert charged `4.188543503989 GPU-h`, with mean best HR-dev `0.070692431461`; all four task reports passed independent audit. The frozen ordinary full-FT + direct256 LR=`3e-4` one-seed/six-family screen was dispatched after exceeding its `1.8 GPU-h` gate, retaining the 0.9 GPU-h uncertainty buffer. Attempt suffix `lr3e4screen`; task00 is running on GPU0. Training cost is estimated near `0.832 GPU-h`; HR-dev evaluation follows the six LR-only selections. Query/test remains closed.
