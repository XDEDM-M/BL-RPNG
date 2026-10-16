# R2B exploratory card — combine LR rollout and direct HR-grid consistency

- **Question:** Can the LR rollout signal that lowers 64² development error combine with direct 64→256 consistency that lowers HR-development error?
- **Evidence:** Fixed H3 supervision lowered the four-task LR mean from 0.02791603 to 0.02642924, but HR development was 0.05596649, essentially tied with R1 and 5.84% worse than the direct-256 consistency candidate at 0.05287928.
- **Design:** Add the true three-step LR support rollout loss to the two-step 64→256 consistency loss, with the same m=4, model seed 42, support seed 271829, tasks 0/1/3/5, 100 epochs, LR 1e-3 and cosine schedule. `commutator_only` suppresses target Newton-path supervision while retaining final rollout and consistency losses.
- **Fair comparison:** Run path-source and ordinary final-only source initializations with identical settings; the ordinary arm receives the same generic objective. Checkpoints are selected only by frozen 64² LR validation. Evaluate shortlisted checkpoints afterwards on independent HR development validation.
- **Budget:** Four tasks per source initialization, one GPU at a time; stop after both paired screens and independent HR-development reports. No query access.
- **Decision:** Promote only if HR development improves over the current 0.05287928 reference without a material LR-validation regression; otherwise record a negative combination result and keep the simpler direct-256 candidate.

## Implementation check (read-only)

- Mechanism review verified that `rollout_horizon=3` adds recursive supervised H3 loss from genuine LR support trajectories, while `commutator_only` disables path loss but retains direct 64→256 consistency at two autoregressive steps. The smoke decomposition closes: `0.024751783245 + 0.05 × 0.029285529732 = 0.026216059716`; step losses are `0.02509721` and `0.03347385`, horizon is 3, and path loss is zero.
- The runner constructs support and separate LR-validation roles only (4 and 8 rollouts for task 0; IC overlap 0); selection is LR validation only, and the smoke reports `query_opened_or_loaded=false`. No query/test data were read.
- Interpretation limits: the 256 branch is label-free self-consistency, not an HR physical label, so it can be self-consistently wrong. Three-step rollout windows overlap and are correlated. The smoke establishes finite combined losses but does not log separate component gradient norms/conflict. The `commutator_only` objective name does not mean H3-free; reports must disclose the H3 term. H3 supervises three steps while the commutator term covers two.

## Frozen paired LR screens (seed 42, support 271829; HR-dev pending)

| Initialization | Task 0 | Task 1 | Task 3 | Task 5 | Four-task mean |
|---|---:|---:|---:|---:|---:|
| Path source | 0.02479727 | 0.03344349 | 0.02398095 | 0.02421145 | 0.02660829 |
| Ordinary final-only source | 0.03246689 | 0.03472367 | 0.03311954 | 0.02999098 | 0.03257527 |

- Both use identical frozen H3+comm256 settings, m=4 and tasks 0/1/3/5; each arm consumed ~0.488 GPU-h. Ordinary initialization has 22.43% higher LR error than path initialization across this small development cohort. This remains a single-seed development comparison.

## Independent HR-development results

| Initialization | Task 0 | Task 1 | Task 3 | Task 5 | Four-task mean |
|---|---:|---:|---:|---:|---:|
| Path source | 0.06836318 | 0.03426582 | 0.06209227 | 0.05067259 | 0.05384847 |
| Ordinary final-only source | 0.07468464 | 0.04103025 | 0.06834128 | 0.05027085 | 0.05858175 |

- All eight reports use LR-selected checkpoints and record `query_read=false`. Path source improves the paired ordinary HR mean by 8.08%, but its mean is 1.83% higher than the current path-source direct256-only result `0.052879281671`.
- **Decision:** Do not promote H3+comm256 above direct256-only. The paired source comparison favors path pretraining under the combined objective, while the combination itself does not improve the HR-development leader. These are exploratory screens, not confirmation evidence.
- Reports: `artifacts/reports/phase1_hrdev_evaluations/{pathsource,ordinary}_h3_comm256_cw005_lr1e-3-serial1_task*_s42_ss271829.json`.
