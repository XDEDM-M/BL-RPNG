# BL-NPF Development Results

> Historical development snapshot. It is retained for provenance but is
> superseded by `docs/EXPERIMENT_STATUS.md` for current conclusions and claim
> limits.

Date: 2026-07-30  
Status: development gate passed; formal fine-grid blind data remains unopened.

## Valid data audit

The final development datasets use
`correction_semantics=cumulative_from_initial_newton_state`.

| Split | Samples | Grid | Solve convergence | Valid path | Nontrivial transition | Mean path/final cosine |
|---|---:|---:|---:|---:|---:|---:|
| A train | 1000 | 128² | 100% | 100% | 100% | 0.9999999 |
| A validation | 200 | 128² | 100% | 100% | 100% | 1.0000002 |
| B train | 160 | 64² | 100% | 100% | 100% | 0.9999996 |
| B validation seen | 80 | 64² | 100% | 100% | 100% | 0.9999992 |
| B validation unseen | 80 | 64² | 100% | 100% | 100% | 0.9999999 |

Input, target and lifting boundary errors, and correction boundary traces are
all exactly zero in the stored float32 tensors.

## A pretraining

The metric is mean 10-step projected-rollout relative L2 on the 20
high-resolution A validation rollouts.

| Objective | Best epoch | Best relL2 | Final/best |
|---|---:|---:|---:|
| Final-only | 18 | 0.074449 | 1.0390 |
| BL-NPF | 18 | **0.070141** | **1.0030** |

BL-NPF reduces error by **5.79%** and is substantially more stable at the final
epoch.

## B four-boundary causal matrix

Checkpoint selection equally weights seen-boundary/new-IC and unseen-boundary
validation strata.

| A initialization | B objective | Overall | Seen | Unseen |
|---|---|---:|---:|---:|
| Scratch | Final-only | 0.339789 | 0.338458 | 0.341120 |
| Scratch | BL-NPF | 0.357744 | 0.358434 | 0.357054 |
| Final-only pretrained | Final-only | 0.101725 | 0.100314 | 0.103135 |
| Final-only pretrained | BL-NPF | 0.106215 | 0.107823 | 0.104608 |
| BL-NPF pretrained | Final-only | 0.102809 | 0.107345 | **0.098273** |
| BL-NPF pretrained | BL-NPF | **0.098974** | **0.093062** | 0.104886 |

The main model provides:

- **70.87%** lower error than scratch final-only;
- **3.73%** lower error than BL-NPF-pretrained → final-only, isolating the
  fine-tuning path effect;
- **2.70%** lower error than final-only-pretrained → final-only;
- exact per-batch path/final gradient ratio of 0.15 within numerical precision.

An annealed-path fine-tuning ablation reaches 0.100171 and does not improve the
fixed-ratio main model. Fixed guidance is therefore retained for the formal
matrix.

## Interpretation and limitations

The development evidence supports both high-resolution Newton-path
pretraining and path-guided low-resolution fine-tuning. The effect is
initialization-dependent: path guidance hurts scratch and final-only-pretrained
B training, but helps after a path-compatible pretraining representation has
formed. This interaction is itself mechanistically useful and is why the full
factorial matrix is required.

These are seed-42 development results, not paper claims. The five-seed formal
matrix, frozen 128² blind suite, hierarchical bootstrap, data-budget curve and
OOD tests are still required.

Two earlier runs are explicitly excluded:

- `artifacts/experiments/invalid_zero_transition_20260730/`: SfePy mutated the
  input array before previous-state sampling;
- `artifacts/experiments/superseded_local_newton_target_20260730/`: a local
  correction was conditioned on an unobserved intermediate iterate.

Both directories retain their diagnostics and invalidation metadata.
