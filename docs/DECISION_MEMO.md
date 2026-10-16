# Phase-1 decision memo (2026-10-16)

## Decision

Retain multi-boundary path pretraining followed by few-shot LR adaptation as a
promising transfer system, but do not claim that Newton-path information gives
a robust advantage over an independently tuned ordinary baseline. Keep the
ordinary full-FT + direct256 LR=`3e-5` configuration as the strongest current
development candidate, not as a confirmed champion.

## Evidence behind the decision

- In the frozen Stage-2 three-seed pilot, path-source C256 averaged
  `0.05347209` HR-dev relative L2 and ordinary full-FT + comm256 at LR=`1e-4`
  averaged `0.05488117`, a 2.57% descriptive difference whose paired interval
  crossed zero. Both used six fixed development families, one support draw,
  and `m=4`.
- The baseline LR bracket `{3e-5,1e-4,3e-4}` found a strong seed0 result at
  `3e-5`; the predeclared seed1 replication yielded `0.04441035`. On the
  matched seed0/seed1 subset, LR=`3e-5` averages `0.04456778`, versus
  `0.05631693` for ordinary LR=`1e-4` and `0.05418312` for path-source C256.
  Those are 20.86% and 17.75% lower errors, respectively, but the comparison
  has only two model seeds, a single support draw, and six development
  families. It is descriptive evidence that baseline tuning may erase the
  Stage-2 path-source edge, not a confirmation result.
- The target path-FT evidence does not establish extra Newton-path benefit.
  Stage-2 first-path, endpoint and direction controls were close and
  seed-dependent; in the focused R4 interaction, A1 first-path did not meet
  its practical/repeatability rule against A2 endpoint. The C256 path-source
  candidate also changes the source objective and adaptation strategy, so it
  cannot isolate target path-FT causally.
- Transfer remains far ahead of the tested lifted LR scratch baseline
  (`0.50904` Stage-2 mean). However, HR-from-scratch and source-initialized
  HR experts did not reach the matched path reference on their four development
  tasks. The expert studies were budget-limited and do not show saturation.
- The latest two-seed candidate's measured incremental target-stage cost is
  `0.930489 GPU-h` for six LR fits and six HR-dev evaluations. This excludes
  source pretraining and reuses existing target data. Direct HR solver versus
  HR inference timing and complete lifecycle break-even remain unavailable.

## Claim boundaries

The evidence supports a practical multi-boundary transfer result on these six
fixed development boundary families. It does not meet the frozen T2 target of
at least 15% improvement with a paired 95% interval on the confirmation
matrix; it does not establish non-inferiority to a saturated HR expert, an
end-to-end cost advantage, or a path-specific causal mechanism. No query/test
data were opened for selection.

## Next decision if additional compute is allocated

Before opening any query, amend and freeze a full confirmation design that
includes the tuned ordinary LR=`3e-5` candidate alongside the selected
path-source and scratch methods. Preserve the 48 tasks and three crossed
support draws, select checkpoints on LR validation only, and set the analysis
and evaluator hashes before generating query data. The present 120-GPU-h
campaign cannot fit the existing primary matrix: it projects to about
`158.2 GPU-h`, while only `48.134 GPU-h` total nominal capacity remains and
48 GPU-h is protected. The budget-limited handoff and all remaining checklist
gaps are in `docs/PHASE1_COMPLETION.md` and `reports/PHASE1_FINAL.md`.
