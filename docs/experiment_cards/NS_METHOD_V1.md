# Navier--Stokes method validation, V1

## Frozen design

The study tests whether solver-intermediate information is learnable. The
supervision is the earliest complete non-endpoint PIMPLE outer state selected
by the existing first-hook D/S gate. It makes no Newton-specific claim.

HR source pretraining uses 28 trajectories at 128²: the seven eligible V0
source-support trajectories plus three independent initial conditions for each
of those seven source tasks. Source validation uses the 14 independent V0
source-query trajectories. The
previously exposed F4/a060 task is excluded in full. The target extension uses
F5 for development and F6 for confirmation. Each task has `m=4` LR support
trajectories, four independent LR validation trajectories, and four HR query
trajectories. The two new query ICs per task are paired with the two existing
V0 query ICs. The V0 manifest and its original access labels are unchanged.

## Matched comparisons

All transfer arms use the same lifted FNO, HR source data, source-validation
selection, and target NLT adaptation schedule:

1. Selected PIMPLE path supervision.
2. Ordinary final-only source training.
3. Matched endpoint-direction control with the selected path norm.
4. Lifted LR scratch from random initialization.

Only one outer state per physical step contributes path supervision. The target
stage uses final-only support labels. Source checkpoints are chosen on source
HR validation; target checkpoints are chosen on independent LR rollout
validation. Every source and target checkpoint plus the evaluator is frozen
before any target HR query field is loaded.

## Evidence and limits

The complete training config is `configs/ns_pimple_method_v1.yaml`; the
resolved, amended machine-readable pre-registration is
`docs/prereg/ns_pimple_method_v1_amended.json` (the initial source coverage
screen and its unused checkpoint are preserved separately). Results will report source and target
training curves, validation-selected epochs, every query rollout, and task-level
means. The design covers two target boundary families and four tasks, so it can
support a focused NS validity check but not broad population-level claims.

The earlier V0 pilot is retained as an exploratory result. Its small source set,
single support trajectory, four-transition adaptation budget, and query result
are not treated as evidence for or against the method.
