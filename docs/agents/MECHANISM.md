# Phase-1 mechanism and numerical audit (2026-10-13)

Scope: read-only audit of Allen--Cahn source/target path semantics, the current
Phase-1 development support data, LR/HR alignment, lifting/normalization, and
the R1 path-FT route. I did not launch training or generation and did not read
query/test data. Only this note was edited.

## Executive finding

The R1 code does implement real target-side LR path fine-tuning, and it is safe
to rerun the interrupted smoke on one GPU. The interruption is attributable to
the recorded host kernel/fsync incident, not to a path-loss exception. However,
the current Allen--Cahn selected path target is numerically almost the endpoint
increment. Consequently, an A1-vs-A0 win would show an auxiliary-supervision
effect, not a nontrivial Newton-geometry effect. An endpoint auxiliary arm is a
mandatory control before making the latter claim.

## Accepted-path definition and provenance

- `sfepy_worker.py` passes independent copies of the previous physical nodal
  field as `state0` and `u_prev` (`problem.solve(...)`, lines 255--260). SfePy's
  Newton hook is called before every iteration, starting from `vec_x0`; after a
  line search, the accepted `vec_x` becomes the next hook state
  (`/gemini/code/sfepy/sfepy/solvers/nls.py`, lines 617--651). Thus recorded
  state 0 is the natural time-step initial guess `q0 = u_t`, and subsequent
  recorded states are accepted iterates.
- `select_progress_update()` chooses the first nonzero accepted state crossing
  normalized log-residual progress 0.25 and returns the cumulative displacement
  `q_k-q_0`, not the local correction `q_k-q_{k-1}`
  (`blrpng/solvers/progress.py`, lines 7--57). The endpoint is the converged
  implicit state `q*=u_{t+1}`. Therefore `q*-q0` is also the physical time
  increment in this Allen--Cahn implementation, while the stored path label is
  an intermediate cumulative Newton displacement.
- The worker interpolates pre/post accepted states separately, subtracts them,
  and zeros the boundary trace (`blrpng/solvers/sfepy_worker.py`, lines
  288--317). New shards persist the selected accepted-state ordinal and declare
  its zero-based semantics. All 1,920 transitions in the current development
  support have a valid correction.
- Limitation: v4 shards retain only the selected correction, progress, ordinal,
  and total nonlinear-iteration count. `write_v4_rollouts_from_arrays()` drops
  the worker's p=0.75 correction/progress and does not store the accepted
  residual/state history (`blrpng/data/shards.py`, lines 320--376). The worker's
  temporary NPZ is then deleted. Selection is well supported by the generator
  source and immutable shard hashes, but cannot be independently replayed from
  the retained campaign artifact. This falls short of the mission's strongest
  “save or verify accepted states/corrections/residuals” evidence requirement.

## Current LR support path geometry

I recomputed diagnostics from the 96 genuine 64-grid SfePy support rollouts in
`hrdev_20261012_t20_ss271829` (1,920 transitions), using
`delta_k=newton_correction` and `delta*=trajectory[t+1]-trajectory[t]`.
No query/test shard was opened.

| statistic over valid transitions | min | Q1 | median | Q3 | max |
|---|---:|---:|---:|---:|---:|
| cosine(`delta_k`,`delta*`) | 0.99995195 | 0.9999999961 | 0.9999999996 | 0.99999999997 | 1.00000000 |
| `||delta_k||/||delta*||` | 0.99999990 | 1.00000172 | 1.00000975 | 1.00004669 | 1.00475253 |
| `D_k=||delta_k-delta*||/||delta*||` | 5.59e-7 | 7.15e-6 | 3.02e-5 | 1.03e-4 | 1.09e-2 |
| best endpoint-projection residual / `||delta_k||` | 5.39e-7 | 6.78e-6 | 2.80e-5 | 8.79e-5 | 9.80e-3 |

Selected ordinal counts are `{3:1171, 4:552, 5:76, 6:109, 7:12}`; none is
the recorded final ordinal. Residual progress ranges 0.25005--0.57490 (median
0.43189). Thus the selector genuinely chooses a pre-final accepted iterate,
but Newton has already reached an endpoint-like displacement: the median
non-collinear residue is only 2.8e-5 of path magnitude. All six families show
the same result (per-family median `D_k` 1.76e-5--3.97e-5).

This is the principal mechanism warning. A fixed endpoint fraction would be an
even weaker/equivalent control; the useful control is the actual endpoint with
the same hook/loss, plus the endpoint direction scaled to the observed path
norm if a geometry-specific contrast is desired.

## LR/HR, lifting, and normalization

- LR support/validation are genuine 64-grid, mesh-65 SfePy solves (`fidelity=0`),
  not HR downsampling. HR development validation is an independent 256-grid,
  mesh-257 solve (`fidelity=1`). The frozen task boundary mapping, `dt=0.2`, and
  20-step physical horizon agree; support, LR validation, and HR validation IC
  sets have zero overlap. Evidence:
  `artifacts/reports/phase1_hrdev_20261012_t20_integrity_audit.json`.
- LR and HR validation are not rollout-paired by initial condition. That is
  appropriate for independent adaptation/evaluation, but it does not support a
  paired LR-discretization-error calculation. A separate paired development
  asset is needed for that Stage-3 diagnostic.
- Allen--Cahn boundary descriptors are time-independent. Each resolution gets
  its own discrete harmonic lift; the model consumes `(u-Eg)/scale`, `Eg/scale`,
  and grid coordinates, predicts a residual, and exactly overwrites boundary
  values (`blrpng/models/fno.py`, lines 324--381). Path probes and labels are
  both zero-trace. This is internally consistent for the present static
  boundary tasks; it must not be generalized silently to time-dependent data.
- There are no train/query mean or variance statistics. The sole Allen--Cahn
  field scale is the fixed value 1.0 from configuration/checkpoint. The path
  loss includes grid-cell weighting and uses the same field scale, so there is
  no target HR statistic leakage or LR/HR normalization mismatch.

## Path-FT implementation and smoke evidence

`scripts/run_phase1_trials.py` maps current path trials to `objective=blrpng_v2`
with NLT adaptation and the fixed path-source checkpoint. `engine.py` reads
`batch['newton_correction']`, applies the layer-1 probe direction/log-magnitude
loss, balances it on the 384 trainable shared layer-0/1 NLT scale/bias values,
and uses the conflict gate. Probe parameters remain trainable but are correctly
excluded from “shared gradient” balancing because the final head does not use
them. Current R1 sets `commutator_weight=0`, so the added target signal is only
the accepted-path loss.

The resumed current-config smoke completed at
`results/phase1/R1/path_lr1e-4_ratio015-smoke_m4_s42_ss271829/task00.json`.
It has valid path labels, 384 shared parameters, finite loss/gradients, and
epoch-0 requested ratio 0.15 with mean achieved ratio 0.075; epoch 1 correctly
anneals to zero. The incident-era directory with `smoke_path_ft_...` in its
name remains an interrupted, metric-free record and is not ranked.

The conflict gate is material: batches with negative final/path gradient cosine
receive zero path weight. Formal trials must report the active-gate fraction
and achieved gradient ratio; otherwise an experiment named path-FT could in
practice be mostly final-only.

## Must-fix items

1. **Before any Newton-geometry claim:** add a matched endpoint auxiliary arm
   (`auxiliary_target=endpoint_increment`) using the identical probe, loss,
   schedule, NLT capacity, support draw, and checkpoint rule. The current R1
   A1-vs-A0 grid alone cannot distinguish path geometry from deep supervision.
2. **Before confirmation-quality path provenance:** retain accepted residual
   histories and enough iterate/correction information to replay the selected
   ordinal, or emit a sealed generation-time verification report before the
   NPZ is deleted. Record `D_k`, norm ratio, angle, projection residual, selected
   ordinal, and near-zero endpoint handling per sample.
3. **Before ranking a path trial:** enforce/report finite path/final gradient
   norms, `path_valid`, path gate-active fraction, achieved gradient ratio, and
   path weight. Treat a mostly closed conflict gate as a different effective
   intervention.
4. **For Stage-3 cross-resolution attribution:** create/use a development-only
   LR/HR pair with identical boundary and IC IDs. Do not infer discretization
   error from the present independent validation cohorts.

## Next minimum credible path-FT control

The current task-0 two-epoch smoke has passed. Run one 100-epoch diagnostic
triplet on task 0 with the current path-source checkpoint, model seed 42,
support seed 271829, m=4, cosine schedule, and LR `1e-3` (the current matched
final-only screen leader):

1. A0: final-only NLT;
2. A1: accepted `newton_correction`, layer 1, `blrpng_v2`, ratio 0.15;
3. A2: `endpoint_increment`, otherwise bit-for-bit identical to A1.

Select checkpoints only on the frozen LR validation rollout metric, then apply
the three frozen checkpoints once to task-0 HR development validation. This is
the smallest interpretable check, not a performance claim. Given the measured
geometry, A1 should be expected to track A2 closely. Only if A1 separates from
A2 should the key comparison be repeated with symmetric local ratio/LR tuning
and additional tasks/seeds. Regardless of outcome, the broader R1 path-ratio
screen is still needed for the mission's practical final-only-vs-path-FT result.

## Addendum: A2 routing after the R1 negative result

The engine already fully supports A2; no numerical data regeneration or new
label field is required:

- `RolloutShardDataset.__getitem__()` exposes `endpoint_increment`. If it is not
  explicitly stored in a v4 shard, it uses the exact retained transition
  `trajectory[t+1]-trajectory[t]` (`blrpng/data/shards.py`, lines 187--191).
- `_train_batch()` accepts `auxiliary_target=endpoint_increment` and routes that
  tensor through the **same** layer probe, direction/log-magnitude loss,
  gradient balancing, annealing, and conflict gate used by accepted-path A1
  (`blrpng/training/engine.py`, lines 430--497).
- `scripts/run_phase1_trials.py::_training_config()` already copies a trial's
  `auxiliary_target` into the effective training config (lines 94--105). The
  checkpoint `loss_config` records it, while the runner's descriptor hashes the
  full trial and engine.

The R1 negative is not explained by a dead path route. Across the four tasks
and 100 epochs, mean path-gate-active fractions for ratios 0.05/0.15/0.30 were
0.5043/0.4951/0.5010. The corresponding mean achieved gradient ratios were
0.00907/0.02673/0.05418 because gamma-2 annealing reduces the requested ratio
over time and roughly half the batches have negative gradient cosine. A2 must
retain the same gate and schedule: removing the gate only for A2 would change
the effective intervention.

Create A2 in a **new control spec**, not by editing the already-scored
`configs/phase1_r1.yaml`: its whole-file SHA is embedded in every descriptor,
so modifying it would invalidate clean reuse/provenance for the completed R1
arms. The minimal trial entry is:

```yaml
- id: endpoint_lr1e-3_ratio015
  objective: blrpng_v2
  auxiliary_target: endpoint_increment
  commutator_weight: 0.0
  lr: 0.001
  path_gradient_ratio: 0.15
  path_layer: 1
  path_anneal_gamma: 2.0
  epochs: 100
  scheduler: cosine
  minimum_lr: 0.000001
```

The separate spec should copy the frozen R1 experiment block exactly (same
source checkpoint, task IDs, m=4, support seed 271829, support/validation
templates, and LR-only selection metric) and use a unique filename/trial ID.
Run A2 against the already-completed A0/A1 artifacts; do not rerun them merely
to obtain a syntactically three-arm directory. The current endpoint arm uses
`valid_correction_mask`, deliberately matching A1's sample inclusion. All
current support labels are valid, so this is immaterial here; for future data
with invalid paths it means A2 is an A1-matched control, not an endpoint arm
over every transition.

## R2 cross-resolution numerical and normalization risks

The existing commutator is useful as a consistency regularizer, but it is not
HR-supervised mixed-resolution training. It evaluates the same operator on a
coarse state and on a bilinearly prolonged coarse state, then restricts the
fine prediction. The following issues must be controlled before interpreting
an R2 gain as improved LR-to-HR physics.

1. **The refined input is not a genuine fine-grid state.** `fine_previous` is
   bilinear interpolation of the LR numerical solution. It carries LR
   dispersion/diffusion error to the fine branch and may reward invariance to
   an interpolant rather than agreement with a true HR solve. Keep a paired,
   development-only LR/HR diagnostic separate; call the training term
   resolution consistency, not HR supervision.
2. **“Exact refined lift” is exact only for an interpolated coarse trace.**
   `_exact_refined_lift()` interpolates the coarse lift to obtain boundary
   values and recomputes a discrete harmonic interior. It does not reevaluate
   the analytic frozen boundary descriptor at the new grid. This can attenuate
   or alias high-frequency and piecewise-smooth boundaries. Before R2 ranking,
   compare this trace to direct descriptor evaluation at 128/256 on development
   tasks and record boundary/near-boundary errors.
3. **The commutator is measured on the absolute physical state.** Exact
   Dirichlet overwrite and the shared lift can dominate agreement, especially
   for large-amplitude boundaries. Its denominator is the restricted fine
   state's norm, so task amplitude and lift energy affect weighting. Also log
   an interior lifted-residual commutator `(u-Eg)` with the boundary excluded;
   use fixed field scales and a declared norm floor.
4. **Restriction/prolongation are bilinear, not discretization-aware.** They
   are neither conservative restriction nor the FEM transfer used by SfePy.
   `align_corners=True` aligns endpoints but can still smooth frequencies and
   create an easy low-pass objective. Compare bilinear restriction against a
   predeclared area/FEM-consistent alternative on paired development fields;
   do not choose it on query performance.
5. **Multi-step consistency can amplify self-generated error.** With
   `commutator_rollout_steps>1`, both branches autoregress on predictions and
   may become consistently wrong. Increase the horizon only after one-step
   consistency is finite and HR-development rollout improves; log each step's
   commutator separately (the engine already exposes these metrics).
6. **Resolution-dependent loss geometry remains.** Final relative L2 uses raw
   pixel Euclidean norms; ratios mostly cancel cell count, but the fraction of
   exactly overwritten boundary pixels changes with resolution. The path loss
   uses a cell-area factor, whereas commutator and final losses use different
   normalizations. Compare interior quadrature-weighted metrics and check
   gradient norms before assigning a weight such as 0.05.
7. **Fixed field scaling is safe only if frozen and adequate.** Allen--Cahn's
   fixed scale 1.0 leaks no target statistics, and the commutator uses model
   field scales. Do not estimate new scales from HR development/query. For
   vector PDEs, source-only or LR-support scales must be frozen and applied
   identically before restriction and per-field aggregation.
8. **Fine-grid defect losses are highly resolution-sensitive.** Finite
   differences contain `h^-2` Laplacians, magnifying interpolation noise; the
   current relative defect divides by state norm rather than a resolution-
   calibrated residual scale. Do not combine defect/PINO with R2 until defect
   magnitude and gradient norms are checked independently at each resolution.
9. **Existing `train()` is single-index training.** A naive index mixing 64,
   128, and 256 shards will fail ordinary tensor collation when a batch contains
   different shapes. Genuine HR/LR mixed pretraining therefore needs
   resolution-homogeneous batches (or alternating loaders), explicit sampling
   weights per resolution/family, and separately logged losses. The current
   synthetic commutator path does not implement this part of Mission R2.
10. **Static-boundary assumptions are embedded.** Refined geometry reuses one
    lift/mask through commutator rollout. This is valid for the present static
    Allen--Cahn boundaries, but not for time-varying boundary conditions without
    recomputing the lift at every step.

The lowest-risk R2 screen is therefore: retain A0/final-only target adaptation,
add one-step 64-to-128 residual-space consistency at a small weight, keep field
scales fixed at source value 1.0, and compare both LR validation and independent
HR development. Only then test 64-to-128-to-256 or multi-step consistency.

## Confirmation provenance patch specification (read-only design)

This section prepares a minimal patch for future confirmation generation. It
does not apply to the currently running direct-256 job, which must remain
untouched. The present chain has a clear gap: the worker holds every accepted
iterate/residual in memory, `select_progress_update()` consumes them, but only
the selected correction/progress/index survives conversion. Interestingly,
`validate_shard()` already contains partial optional checks for
`accepted_newton_residuals` and `accepted_newton_counts`; neither field is
currently produced or copied.

### Minimal retained evidence, not full state history

Do **not** store every accepted FEM state. For each physical step retain these
small fixed-width arrays, padded conservatively to
`Kmax=max_newton_iters+2`:

| field | dtype / shape in worker arrays | active meaning | padding |
|---|---|---|---|
| `accepted_newton_residuals` | float64 `[N,Kmax]` | exact residual 2-norm for recorded `q_0...q_{K-1}` | `0.0` |
| `accepted_newton_step_norms` | float64 `[N,Kmax]` | `0` at k=0; `||q_k-q_{k-1}||_2` for k>0 | `0.0` |
| `accepted_newton_counts` | int16 `[N]` | legacy-compatible key: number K of raw recorded trace states, including q0/appended final and possible duplicate q0 | none |

The extra slot is necessary because SfePy hooks at the start of iterations and
the worker may append the returned converged state after the last hook.
Residuals plus step norms are sufficient to replay the selector: progress uses
only residuals, while candidate eligibility uses only whether the step norm is
above `1e-13`. This adds at most `2 * 22 * 8 = 352` bytes per transition
under the current 20-iteration cap, versus storing full FEM states. Keep
float64: an evidence trail should not change a threshold crossing through
float32 rounding.

Also retain the already-produced p=0.75 selected correction/progress/valid mask
in v4 (`newton_correction_p075`, `residual_progress_p075`,
`valid_correction_mask_p075`), not only its ordinal. This costs one extra grid
field per transition; if storage is a concern, the p=0.75 **grid correction**
may be omitted because it is not needed to replay selection, but its progress,
valid bit, and ordinal should remain. The p=0.25 correction used for training
must remain unchanged.

### `progress.py`: one canonical replay function

Factor selection into a pure helper, for example:

```text
replay_progress_selection(residuals, step_norms, threshold,
                          residual_floor=1e-30,
                          log_denominator_floor=1e-12,
                          nonzero_step_threshold=1e-13)
    -> (selected_index, selected_progress, valid)
```

Its exact rules must match the current selector:

1. Use the first `count` values; require equal residual/step-norm lengths.
2. Clamp residuals below `1e-30` before logarithms.
3. Set denominator to
   `max(log(r0)-log(r_final), 1e-12)` and clip every progress to `[0,1]`.
4. Candidate indices are k=1..K-1 with `step_norm[k] > 1e-13`.
5. Select the first candidate with progress >= threshold; if none crosses,
   select the last candidate. With no candidate return `(-1,0,false)`.

`select_progress_update()` should compute step norms from states, call this
helper, and only then form `states[k]-states[0]`. Both worker generation and v4
validation must call the same helper rather than maintain duplicate selection
logic. Preserve the old return signature for existing callers.

### `sfepy_worker.py`: evidence capture

Immediately after the optional final state append and before either 0.25/0.75
selection:

1. Assert `len(iter_states)==len(iter_residuals)>=1` and all values are finite.
2. Compute the accepted step norms once in reduced FEM space; k=0 is exactly 0.
3. Pad residuals/norms to Kmax and append them plus the count to `records`.
4. Replay both thresholds with the canonical helper and assert exact ordinal and
   valid-bit equality with the values returned by `select_progress_update()`;
   compare progress with tight float64 tolerance before casting output progress
   to float32.
5. Record the constants and semantics in metadata, including that q0 is the
   natural previous physical state and ordinals refer to the **raw recorded
   trace**, not a compacted accepted-update list.

There is a subtle existing metadata error to fix for new data. SfePy invokes
the hook before `apply_line_search()`. At iteration 0 it evaluates q0 and forms
the first Newton direction; the iteration-1 hook can therefore record q0 again
before the first nonzero accepted update appears. The selector correctly skips
this duplicate through its `>1e-13` step-norm test, so current labels are not
invalid, but the present string “first accepted state is 1” is not generally
true. Use semantics such as
`zero_based_raw_hook_trace_q0_index_0_nonzero_steps_are_accepted_updates_final_may_be_appended`.

Recommended metadata object:

```json
"newton_path_selection": {
  "schema": "blrpng_newton_selection_evidence_v1",
  "thresholds": [0.25, 0.75],
  "residual_floor": 1e-30,
  "log_denominator_floor": 1e-12,
  "nonzero_step_norm_threshold": 1e-13,
  "fallback": "last_nonzero_candidate",
  "q0_semantics": "natural_previous_physical_state",
  "trace_count_includes_q0_and_appended_final": true,
  "trace_may_contain_repeated_q0": true
}
```

Do not infer trace count from `num_newton_iters`: persist the explicit count.
The current value `len(iter_states)-1` is a recorded-transition count, not
necessarily SfePy's Newton iteration count because the trace can contain a
repeated q0 and an appended final state. Also persist `status['n_iter']` under a
separately named solver-iteration field; do not silently change the legacy
field's meaning.

### `write_v4_rollouts_from_arrays()` and shard validation

The writer should slice the three evidence arrays by rollout step and copy the
selection metadata verbatim into each shard. Make the evidence group all-or-
none: if any of residuals, step norms, or counts is present, require all three.
Historical v4 shards remain valid because the group is optional, but a new
confirmation gate should require evidence schema v1.

`validate_shard()` should enforce:

- shapes `[T,Kmax]`, `[T,Kmax]`, `[T]`; counts in `[1,Kmax]`;
- finite, non-negative active residuals and step norms; k=0 norm exactly zero;
- zero padding at positions `k>=count` (so top-level finite checks still pass);
- selected ordinals are `-1` iff invalid, otherwise `1 <= k < count`;
- replayed p=0.25 and p=0.75 ordinal/valid values equal stored values exactly;
- replayed progress matches stored float32 progress with an explicit absolute
  tolerance such as `5e-7` (not a loose scientific-result tolerance);
- legacy `nonlinear_iterations == count-1` only for shards produced before a
  schema clarification; separately validate the new solver-iteration field
  against `status['n_iter']` without conflating it with trace count.

Return replay-pass counts in the validation summary and include them in a
sealed generation report before deleting the temporary NPZ. The report should
contain worker/progress/shards code hashes, manifest hash, selection constants,
transition count, invalid/fallback counts, selected-index histograms, and the
hashes of the final v4 index/shards. This provides evidence even if a later
consumer does not load every shard.

### Fixed geometry and near-zero rules

Compute geometry on the retained **output-grid representation**, after boundary
zeroing, because this is exactly the representation supervised by the model.
For vector PDEs divide each field by its frozen source/LR-deployment field scale
and report each field separately in addition to the joint scaled vector. Do not
derive scales from HR validation/query.

Let `delta*=q*-q0`, `delta_k=q_k-q0`, and define a resolution-consistent norm
using the same rectangle quadrature for all quantities:

```text
||v||_h = sqrt(hx*hy * sum_field sum_grid trapezoid_weight * (v_field/scale_field)^2)
```

Boundary entries are zero for static Dirichlet corrections, but retaining
trapezoid weights makes the definition explicit. Freeze
`epsilon_delta = 1e-8` in these normalized physical units for Allen--Cahn; do
not tune it after looking at confirmation outcomes. Store it in the protocol
and report a sensitivity table on source/development only if another PDE needs
a different physical scale.

For `a=||delta_k||_h`, `b=||delta*||_h`, and inner product `<.,.>_h`:

- `nearzero_endpoint = (b <= epsilon_delta)`; report/count these separately and
  exclude them from aggregate D, norm-ratio, cosine, angle, and projection
  quantiles. Do not silently turn them into ordinary zero-error samples.
- `nearzero_path = (a <= epsilon_delta)`. If endpoint is non-nearzero, D and
  norm ratio remain defined, but cosine/angle/projection-direction metrics are
  invalid and counted separately.
- `D = ||delta_k-delta*||_h / max(b, epsilon_delta)`.
- `norm_ratio = a / max(b, epsilon_delta)`.
- `cosine = clamp(<delta_k,delta*>_h/(a*b),-1,1)` only when both are non-nearzero;
  `angle_degrees = acos(cosine)*180/pi`.
- `alpha = <delta_k,delta*>_h / b^2` when endpoint is non-nearzero.
- `projection_residual = ||delta_k-alpha*delta*||_h / max(a,epsilon_delta)`;
  this is the fraction of the path target orthogonal to the endpoint direction.
  Optionally also record the endpoint-normalized numerator, but never mix the
  two denominators in one table.

Use validity masks plus zero-filled tensor slots rather than NaN/Inf inside v4,
because shard validation globally rejects non-finite floating tensors. JSON
reports may use `null` for undefined per-sample metrics. Aggregate by rollout,
task, and boundary family; time steps are repeated observations, not independent
generalization units.

### Minimal tests before confirmation generation

No solver run is needed to implement unit tests. Add synthetic selector tests
covering threshold crossing, fallback, zero step, residual floor, denominator
floor, and no-candidate behavior; then build a synthetic v4 shard and verify
that changing one residual, count, step norm, selected ordinal, progress, or
padding value is rejected. A small future SfePy smoke may follow only after the
currently running job completes and the main thread schedules it.

### Implementation status (2026-10-13)

The compact selection-replay portion is now implemented in the assigned files:

- `progress.py` owns the constants and canonical
  `replay_progress_selection()` helper; `select_progress_update()` delegates to
  it while preserving its four-value default return and optional five-value
  indexed return. Historical residual edge-padding behavior is retained.
- The Allen--Cahn worker checks the `max_newton_iters+2` raw-trace bound, stores
  float64 padded residual/step-norm matrices and explicit trace counts, records
  SfePy `status['n_iter']` separately, verifies both selections in memory, and
  emits corrected raw-hook semantics. It explicitly states that full node
  states are not retained and evidence supports selection replay rather than
  field-trajectory reconstruction.
- The v4 writer persists the new evidence group only when complete, copies the
  p=0.75 correction/progress/valid/index fields and metadata, and leaves old
  shards untouched. Validator replay is strict for evidence-v1 shards. Legacy
  residual+count-only shards remain accepted without claiming replay evidence.
- `tests/test_phase1_path_provenance.py` covers signature compatibility,
  residual edge padding, threshold fallback, no-candidate behavior, residual
  and denominator floors, two-threshold shard replay, corruption rejection,
  all-or-none enforcement, and legacy compatibility.

CPU-only fnotest results after the first-nonzero extension: the new suite
passed 8/8; existing selector tests passed 4/4; relevant
v2/window/Navier--Stokes shard suites passed 25/25. Python
compilation also passed. No solver, GPU, data generation, or query/test data was
used. Geometry D/norm/cos/projection summaries remain analysis/report work;
they are deliberately not inserted into training batches by this patch.

## Decision

The current-data smoke and three `lr=1e-3` ratio arms establish that path-FT is
implemented but does not improve the four-task LR-validation mean. Run the
new-spec A2 endpoint control to classify that negative result, while continuing
lower-LR R1 search under the frozen rules. Label A1 “accepted-path auxiliary
supervision,” not “nonlinear Newton geometry.” For R2, begin with one-step
64-to-128 residual-space consistency and explicit gradient/norm diagnostics;
do not describe the existing interpolated commutator as mixed HR/LR training.
Before confirmation generation, implement the compact replay evidence above
and correct the raw-trace ordinal semantics; no change is requested for the
already-running direct-256 development job.

## R3 first-nonzero cumulative-path candidate (2026-10-13)

The targeted candidate is now implemented without changing the solve, q0,
physics, tolerances, or the existing p=0.25/p=0.75 labels. For every time step
the worker calls the same audited `select_progress_update()` with threshold
`0.0`. Because the canonical selector excludes step norms at or below
`1e-13`, this skips repeated q0 hook records and selects the earliest nonzero
accepted raw-trace state. The stored correction is still cumulative from the
natural previous physical state q0; it is not a local correction from an
unobserved iterate and is not produced by perturbing the initial guess.

The optional shard group is atomic:

- `newton_correction_first`
- `residual_progress_first`
- `valid_correction_mask_first`
- `selected_newton_state_index_first`

New evidence metadata records `first_selection_threshold: 0.0`. The writer and
validator require all four fields together, require the complete float64 replay
evidence, replay the ordinal/progress/valid bit at exactly threshold zero, and
require an exactly zero Dirichlet boundary trace on the first correction. The
dataset exposes these fields only when actually stored. Historical shards keep
working with no synthetic field and, in particular, never receive an endpoint
increment as a fallback first-path target.

The retained hook trace contains accepted solver states, with one important
indexing wrinkle: q0 is recorded twice before the first nonzero accepted state.
The threshold-zero selector skips that duplicate by its step-norm rule. What is
strictly proven by the persisted compact evidence is the first-nonzero accepted
state selection and its cumulative q0 displacement; full node-state
trajectories remain unretained and cannot be reconstructed from the compact
audit.

CPU-only synthetic coverage now includes threshold-zero replay, repeated-q0
skipping, corruption rejection, optional-group atomicity, exact boundary-zero
checking, and Dataset presence/absence behavior. No solver, generation, GPU,
engine/runner, or query/test access was used for this addition.

### Accepted-state callback data-flow recheck

The exact SfePy source used by `sfepy_env` resolves to
`/gemini/code/sfepy/sfepy/solvers/nls.py` (SfePy
`2025.4+git.68eb845dc02fd197eb8395fd9b593b5c3509a8e1`). Its file SHA-256 is:

```text
a0535b5f2e970d34500a308259024158720c60a0cc9a7220bb23e746ede80c21
```

The `vec_x` flow in `Newton.__call__()` establishes accepted-state semantics,
not merely untested line-search candidates:

1. Lines 611--614 initialize `vec_x=vec_x0`, `vec_x_last=vec_x0`, and
   `vec_dx=0`. The iteration-0 hook at lines 626--628 therefore records q0.
2. Lines 635--638 call `apply_line_search(vec_x_last, ..., vec_dx, ...)` and
   assign its returned state directly to `vec_x`. On iteration 0, `vec_dx=0`,
   so this returns q0. The first Newton direction is computed at lines 672--674.
3. The loop tail scales that direction and increments the iteration, then only
   executes `vec_x_last=vec_x.copy()` at lines 680--685. There is no separate
   `vec_x -= vec_dx` at the loop tail. Consequently the iteration-1 hook again
   sees q0, explaining the duplicate zero step.
4. During iteration 1, `apply_line_search_bt()` forms trial states as
   `vec_x0-vec_dx` (line 146), backtracks as needed (lines 175--212), and
   returns the line-search-selected state (line 214). The caller assigns that
   returned state to `vec_x`. At the loop tail it becomes `vec_x_last`; the
   next hook therefore sees the state already selected by the preceding line
   search, i.e. the accepted solver iterate, not the current iteration's
   pre-line-search trial.
5. If convergence is detected at lines 645--647 before another hook, the
   solver returns that accepted `vec_x` at line 708. The worker obtains this as
   `final_reduced` and appends it only when it differs from the last hook state
   (`sfepy_worker.py` lines 283--290). Thus this final optional entry is also
   the solver-returned accepted state.

Therefore the raw sequence is operationally
`[q0, q0, q1_accepted, q2_accepted, ...]`, optionally followed by the returned
final accepted state when convergence occurs before a subsequent hook. The
first threshold-zero nonzero entry is genuinely the first accepted Newton
path displacement from the natural q0. The word “raw” describes duplicate-q0
and optional-final indexing; it does not demote nonzero hook entries to
unaccepted line-search candidates. No solve behavior or initial guess was
changed for this conclusion.

## R3 mapping / endpoint audit handoff (2026-10-13, Asia/Shanghai)

Read-only audit of the current full R3 outputs and raw task JSON; no training,
data generation, query access, or result edits were performed.

- `results/phase1/R3/path_l3_lr1e-3_ratio005_m4_s42_ss271829/`: each of the four
  100-epoch task histories selects exactly the saved `best_epoch`; task checkpoint
  SHA-256s verify. Recomputing the unweighted mean of task-level LR-validation
  rollout errors gives **0.02799854681798**, exactly the saved summary. All task
  JSONs have `query_metric=null`. Per-task values (0/1/3/5) are
  0.02744002539 / 0.03378129333 / 0.02527297619 / 0.02549989237.
- Paired raw-task comparison against the existing R1 layer-1 path arm uses the
  same task IDs, support-index hashes, LR-validation-index hashes, source
  checkpoint SHA, model seed, LR, epochs, scheduler, and path ratio. Layer 1's
  mean is 0.02792729793437; layer 3 is **+0.000071248884 (+0.2551%)** worse,
  with positive error deltas on all four tasks (+0.062% to +0.444%). Against
  same-screen final-only (0.02791603357764), layer 3 is +0.2956% worse. This is
  a single-seed screen: no evidence of a layer-3 mapping benefit.
- The endpoint-L3 arm is still incomplete in the append-only ledger: as of this
  audit tasks 0/1/3 are DONE and task 5 is RUNNING. The paired 3-task means are
  0.02882005419 endpoint vs 0.02883143163 path (endpoint lower by 0.0394%); do
  not promote this partial mean to a full-trial conclusion. Endpoint-direction
  and endpoint-L1 scored arms have no full outputs yet. The one-task first-path
  smoke is implementation-only.
- Mechanism remains unresolved. Existing support geometry shows p=.25 selected
  accepted displacements almost equal the endpoint (median D=3.02e-5, median
  endpoint-projection residual/path norm=2.80e-5). Thus path-vs-final-only is not
  evidence for nontrivial Newton geometry; the matched same-hook endpoint and
  direction arms must finish before attribution.
- Provenance caveat: the R3 layer-3 path run records engine SHA
  `55b1e103...`, matching committed source at git `6ccea77`; the R1 layer-1
  comparator records engine SHA `d0b5109...`, which does **not** match
  `blrpng/training/engine.py` at its recorded git `b026ef0` (actual committed
  SHA `7313406f...`). Therefore the mapping comparison is paired on data and
  hyperparameters, but not strictly code-matched. The running endpoint config
  is a distinct descriptor (`b7276d30...`, engine SHA `4c38203e...`); preserve
  its exact source/runtime identity before treating it as a clean A2 contrast.

Next useful control: finish the four-task same-layer endpoint and endpoint-
direction arms. For a geometry-specific follow-up, compare genuine first
accepted cumulative-update supervision against endpoint direction scaled to
that same first-update norm, with identical layer, initialization, support,
schedule, and code hash; the current p=.25 target is too endpoint-like to test
nonredundant Newton information.

### R3 endpoint-L3 completion update (2026-10-13, 23:22 Asia/Shanghai)

A read-only follow-up found task 5 complete. Recomputed from all four raw task
histories (best epoch and checkpoint SHA verified), the endpoint-L3 mean is
**0.02795536657650** versus path-L3 **0.02799854681798**: endpoint is lower by
0.00004318024 (**0.1542%**). Paired endpoint-minus-path changes for tasks 0/1/3/5
are -0.0000453383 (-0.1652%), +0.0000092911 (+0.0275%), +0.0000019149
(+0.0076%), and -0.0001385886 (-0.5435%). Support and validation-index hashes
match path-L3 task by task; source checkpoint, seed, layer, optimizer schedule,
loss ratio, and 100-epoch budget also match. This is a small, mixed-direction,
single-seed difference: it supplies no evidence that the selected accepted-path
target beats the endpoint target.

The endpoint descriptor has a different engine/runner SHA from path-L3, though
both say git `6ccea77`. Diff against that commit shows the engine changes are
first-accepted-label routing plus initialization/plumbing of the valid mask;
the existing `endpoint_increment` route remains equivalent (`valid_correction_mask`
was already passed directly). Runner changes add the base-config hash and allow
support budget 16. No active training-path change was found for this A1/A2 pair,
but retain both code hashes because the run-time source was not represented by
the git commit alone. The earlier R1-versus-R3 layer mapping caveat still stands.

Next control: screen first accepted cumulative updates against
`endpoint_direction_first_norm` on the same replay-audited LR support, layer,
initialization, seed, and LR-validation rule. This isolates update geometry from
scale after A1/A2 showed the p=.25 cumulative displacement is effectively an
endpoint target. First-path currently has only a two-epoch smoke, not a scored
result.

Latest R3 ledger snapshot (23:22 +08): the scored `direction_l3` arm has task 00
DONE and task 01 RUNNING; scored `firstpath_l1` task 00 is RUNNING. These are
partial screens and are not included in the completed A1/A2 comparison above.

## Confirmation manifest and Stage-2 A1/A2/A3 audit (2026-10-14)

Read-only audit. I read boundary/task manifests, split/audit metadata, development
pilot outputs, and runner/config code only. I did not open confirmation rollout
or HR-query data, launch compute, or edit anything except this handoff.

### Frozen 48-task boundary manifest

`artifacts/manifests/phase1_confirmation/ac_phase1_confirmation_20261014/tasks.json`
has a valid top-level SHA (`bb9dcdf9…aac5b5`). It contains target IDs 0–47 and
48 unique boundary IDs (6000000–6000047); each of the six declared families has
exactly eight family-local IDs 0–7. All 48 nested descriptor hashes are unique
and valid, and each descriptor's family/ID agrees with its task row. No ID or
descriptor-hash collision was found against the frozen Phase-1 development task
manifests. It is a boundary-only freeze: no confirmation support, LR-validation,
or HR-query role manifest/index/data exists under the confirmation campaign
root yet.

The six family names are exactly the six already used for development selection
(`high_frequency`, `amplitude_ood`, `matern_gp`, `localized_bump`,
`mixed_fourier`, `piecewise_smooth`). Thus this is a sound test of new boundary
descriptors within six known function-family distributions, not a held-out
function-family test. Report boundary-task results and family-specific results;
48 tasks (eight/family) are the prospective generalization units. Model/support
seeds are repeated measurements crossed within those tasks, and rollout steps
or Newton transitions are not independent samples. The six families are a
purposeful set, not a random sample of all possible boundary families.

### Role separation and query gate

`prepare_phase1_development.py` routes explicit confirmation campaigns to the
separate `artifacts/{data,manifests}/phase1_confirmation` roots, reuses the same
frozen boundary descriptor for every role, and constructs separate case/IC
streams. Current defaults give support IC seed `support_seed+101` (with the
support seed also in its `SeedSequence`), LR-validation seed `manifest_seed+20000`,
and HR-query seed `manifest_seed+30000`; case-ID ranges are separately assigned.
The HR role is labeled `evaluation_role=query`, uses 256-grid/fidelity-1 solves,
and output metadata says `final_confirmation_query`. The role generator requires
an explicit `--roles` argument for confirmation and reports `query_generated`.

This is not yet a hard freeze-before-query gate: passing
`--roles validation_hr` generates the final query without checking for a sealed
checkpoint/analysis freeze artifact. The manifest's query policy is descriptive
text. The confirmatory protocol file and role-specific confirmation manifests
are also not present. Therefore separation is correctly designed in code but
not yet empirically verifiable from generated role IDs, and query timing still
relies on operator discipline. Before the final role is generated, freeze the
support/model seed matrix, methods/checkpoints, LR-only selection, horizon,
aggregation/statistics, and analysis code; then add/check a hard prerequisite
gate. Do not interpret the currently empty campaign root as evidence that future
query access is technically prevented.

### Stage-2 pilot pairing and mechanism scope

`configs/phase1_stage2_pilot.yaml` pairs A1 (`newton_correction_first`), A2
(`endpoint_increment`), and A3 (`endpoint_direction_first_norm`) at layer 1,
ratio .15, cosine-100, gamma-2 annealing, zero commutator weight, and the same
per-seed path-source checkpoint template, m=4 support index, LR-validation
index, tasks 0–5, and validation-only checkpoint rule. A3 keeps endpoint
direction and rescales it per support sample to the first-path norm, so A1 vs
A3 is the useful direction-at-matched-scale contrast; A1 vs A2 changes both
orientation and magnitude. `STAGE2_PILOT.md` specifies seeds 0/1/42, but the YAML
only defaults to 42; the runner supports one `--seed` override per invocation,
so the full crossed seed matrix must be explicitly dispatched and ledgered.
At this audit there are no S2PILOT target result rows/directories; seed-0 source
pretraining is active. The pilot design is therefore not yet pilot evidence.
It uses one fixed support seed, which is appropriate for arm pairing but does not
measure support-draw variability.

The completed serial2 development A1/A2/A3 screen is a clean within-task,
within-code pair (four tasks 0/1/3/5, seed 42, fixed m=4 support): raw-task
histories select the saved LR-validation best epochs; checkpoint hashes, source
checkpoint, support and LR-validation index hashes match across arms. Their
independent HR-development means are A1 first-path .15 **0.05515037**, A2
endpoint .15 **0.05518705**, and A3 endpoint-direction/first-norm .15
**0.05568520**. A1 is only 0.0665% lower than A2, with its gain confined to the
piecewise-smooth task while the other three are slightly worse; A3 is 0.97%
worse than A1, driven by that same task. All reports state `query_read=false`
and that HR validation was not used for checkpoint selection. This is neutral
exploratory evidence, not a robust mechanism effect; there is one model seed and
only four boundary instances, one in each represented family.

The first-path labels themselves are meaningfully less redundant than the old
p=.25 labels: all 1,920 regenerated support transitions select raw trace ordinal
2 (`[q0,q0,q1_accepted]`), are valid, and pass 5,760 selection-replay checks;
864 common old/new physical tensors are exactly identical and all regenerated
solves converged. The recorded geometry has median D(first−endpoint)=0.03178,
median cosine 0.999622, median angle 1.5744°, and median endpoint-projection
residual/first norm 0.02748. Thus A3 is a relevant control, but the first update
is still usually close to the endpoint direction. These transition-level
summaries are descriptive because transitions within the same rollout are
correlated. The generation audit retains worker/selector/writer hashes and replay
traces, but the generation-time source bytes and SfePy environment hash are not
archived; avoid claiming a fully hermetic reconstruction of the solver path.

For final inference, treat the result as evidence about this selected first
cumulative update versus endpoint targets under the tested Allen–Cahn setup;
it does not establish learning of the full Newton trajectory or generalization
to unseen boundary families. The 3-seed Stage-2 pilot can test development
stability across the six fixed families, while the independent 48-boundary
campaign is needed for boundary-instance inference.

## Mechanism priority audit: R1–R3 and frozen Stage 2 (2026-10-14)

Read-only synthesis against `docs/MISSION_PHASE1.md` §§4.3/Stage 3,
`docs/STATE.md`, the R1/R2/R2B/R3 cards, the frozen Stage-2 YAML/card, the
confirmation preregistration, and the raw-output audits cited below. No GPU,
training, generation, query access, code edits, or protocol edits were used.

### Mechanism claim worth carrying forward

The strongest current mechanistic lead is **source path-guided pretraining may
improve transfer under a fixed target adaptation recipe**. In R2's paired
direct-256 screen, path-source NLT and ordinary-source NLT both use the same
target consistency objective; the HR-development means are `0.052879281671`
and `0.057153851914`, respectively. R2B repeats the direction under a common
H3+direct-256 objective (`0.053848467514` path-source vs `0.058581753005`
ordinary-source). Evidence: `docs/experiment_cards/R2.md` (ordinary-source
direct256 section), `docs/experiment_cards/R2B.md`, and the independent
recomputation in `docs/agents/REVIEW.md` (R2/R2B sections); raw task outputs
are under `results/phase1/R2/` and `results/phase1/R2B/`, with HR reports in
`artifacts/reports/phase1_hrdev_evaluations/`.

This is a promising one-seed, four-task development signal, not yet a robust
causal conclusion. It supports the **source-objective contrast under those
matched target recipes**, not the stronger statement that intermediate Newton
states themselves are the cause. The strongest ordinary system is R2C ordinary
full-FT+direct256 (`0.055875992816`), but it changes target adaptation capacity
and learning rate relative to path-source NLT direct256. The path-source
candidate is numerically better (`0.052879281671`), yet this cross-system
comparison does not isolate source pretraining. In the direct256 paired
comparison, the piecewise-smooth task remains worse for path-source than for
ordinary NLT (`0.0473763` vs `0.0462944`). Keep the claim scoped to the four
tested families and expose this tail result. The R2B pair has the same limited
seed/task coverage and does not promote H3+direct256 above direct256-only.

Target-side Newton-path FT is not presently the best-supported mechanism. R1's
old progress-0.25 labels were almost endpoint-collinear (median `D=3.02e-5`);
its matched final-only screen did not show an LR gain at either paired learning
rate. The regenerated first-accepted labels are less redundant (median
`D=0.03178`, median angle `1.5744°`), but the clean R3 serial2 A1/A2/A3 HR
contrast remains nearly null at ratio .15: first-path `0.055150370795`,
endpoint `0.055187052739` (A1 lower by 0.0665%); A1 is worse on three of four
tasks, with the aggregate gain coming from task 5. Direction-matched A3 is
`0.055685197838` (0.97% worse than A1). All five first-path/endpoint/direction
HR screens are also 4.29–6.05% worse than the direct256 leader. See
`docs/experiment_cards/R3.md`, `results/phase1/R3/*serial2_m4_s42_ss271829/`,
the corresponding `artifacts/reports/phase1_hrdev_evaluations/*serial2*` files,
and the independent raw recomputation in `docs/agents/REVIEW.md` (R3
first-accepted serial2 review). This is weak evidence against a useful
first-update-specific benefit at the tested setting; it is not an equivalence
claim or a rejection of all target-path methods.

### A0/A1/A2 and direction control coverage

For the mission's minimal mechanism interpretation, keep the arms distinct:

- **A0, no auxiliary target:** asks whether adding target auxiliary supervision
  helps at all.
- **A1, accepted intermediate displacement:** the selected path target.
- **A2, endpoint displacement through the same hook/head/loss:** controls for
  generic deep supervision. A1 vs A2 is the key path-specific contrast.
- **A3, endpoint direction with support-sample path norm:** tests direction at
  matched scale. A1 vs A3 is informative about orientation, subject to the
  fact that A3's scale uses the support path label.

R3 serial2 supplies a clean, same-source/same-code/same-task/same-support
A1/A2/A3 comparison at ratios .05 and .15 where available. It does **not**
include A0 in the same run matrix. R1 does contain a matched final-only A0 and
old progress-label A1 screen, but it has no endpoint A2 and its path labels are
the endpoint-like p=.25 targets. R3 regenerates physically identical rollout
fields for first-path labels, but uses a distinct sealed data root and later
runner/engine provenance. Together these rounds triangulate the question, but
they are not one exact A0/A1/A2 factorial on the regenerated first-path data.
The frozen `configs/phase1_stage2_pilot.yaml` repeats A1/A2/A3 at ratio .15 across
six tasks and three model seeds, but contains no zero-auxiliary A0. Its
`pilot_pathsource_comm256` arm is not A0: it has a nonzero resolution-consistency
loss. The ordinary-source arm is full FT with that same consistency loss, not
ordinary-source path FT. Thus the pilot can estimate first-path vs endpoint
auxiliary effects under path-source initialization, but cannot by itself
estimate path-FT vs final-only or the source-path × target-path interaction.
It also fixes one support draw, so it does not estimate support-draw
variability. See `configs/phase1_stage2_pilot.yaml`,
`docs/experiment_cards/STAGE2_PILOT.md`, and `docs/prereg/CONFIRMATORY.md`.

The first-accepted A1/A2/A3 pilot controls are otherwise well paired for their
intended contrast: same per-seed path-source checkpoint, target model seed,
task, m=4 support and LR-validation indexes, layer 1, ratio .15, optimizer
schedule, loss balancing/gating, and LR-only checkpoint selection. A1 vs A2
changes the target only; A1 vs A3 keeps direction/scale information explicitly
matched by each support rollout. This is adequate for a development estimate of
those auxiliary-target contrasts, provided completed reports verify these
bindings. It is not a complete mechanism factorial. Mission A4 (sample-matched
permutation) is also absent; because A2/A3 are the more discriminating controls,
A4 remains a lower-priority negative control.

### Confirmation scope and no-query next steps

The frozen confirmation manifest contains 48 new boundary descriptors, eight
per each of six already-known families. It supports new-boundary inference
within those families, not a held-out function-family claim. The protocol
crosses five model seeds with three support seeds and nested support sizes;
checkpoint selection is LR-validation-only, and the HR query is held until the
selected checkpoint matrix, evaluator, and analysis are frozen. The current
query-freezer implementation checks the canonical checkpoint matrix,
preregistration/matrix hashes, complete cells and hashes, LR-only selection,
analysis/evaluator hashes, and absence of canonical query outputs before
unlocking the query. This supersedes the earlier paragraph in “Role separation
and query gate,” which described the pre-hardening state. Current references:
`docs/prereg/CONFIRMATORY.md`, `docs/experiment_cards/STAGE2_PILOT.md`,
`scripts/freeze_phase1_confirmation_query.py`, and the latest state in
`docs/STATE.md`. No confirmation support, LR validation, or query rollouts have
been generated as of this audit.

Next steps that need no query access:

1. Leave the frozen pilot arms untouched. Complete and audit the full 3-seed ×
   6-task LR matrix, then evaluate each LR-selected checkpoint once on the
   pinned HR-development split as the card specifies. Report paired taskwise
   A1−A2/A3 effects and seed dispersion; do not count time steps as replicates.
2. Report the existing R1 A0/A1 evidence separately from the first-path pilot.
   If a direct path-FT-vs-final-only estimate is still needed after the pilot,
   predeclare a separate development-only A0 diagnostic with the same source
   checkpoint per seed, support/validation indexes, architecture, schedule,
   and checkpoint rule as A1; keep it outside the frozen pilot and do not use
   HR-development results for within-task selection.
3. Prioritize replication of the source-pretraining contrast under the same
   target recipe on the six development families/seeds, and compare it with
   the independently tuned ordinary system only as a system-level result.
   Keep the path-FT claim conditional on A1 separating from A2 and effective
   A3; do not infer mechanism from A1 beating only A0.

No query result is needed for any of these development-stage decisions. The
final confirmation remains necessary for boundary-instance inference and may
not be used to choose between these mechanism hypotheses.


## R4 follow-up recommendation (2026-10-16)

R1–R3 path-search rounds are closed and the 3×3×48 confirmation matrix cannot fit inside the remaining campaign budget; a reduced matrix would not count as confirmation. Finish the frozen R4 A1/A2/A0 evaluation and the active m=16 HR-scratch controls. If R4 clears its predeclared A1-over-A0-and-A2 threshold, the preferred next experiment is a fixed-recipe support-draw replication; if it fails, stop path-specific tuning and report the mechanism claim as unsupported. An m=16 source-initialized HR full-FT expert would be informative but is not to be started unless conservative measured cost leaves at least 0.9 GPU-h buffer; present remaining headroom is marginal after queued jobs.


## R4 outcome (2026-10-16)

The frozen R4 HR-dev matrix does not support a repeatable target-path fine-tuning advantage: A1 first-path+comm256 macro error is `0.05387774`, versus `0.05347209` for A0 final-only+comm256 and `0.05363236` for the endpoint control. A1 misses the preregistered ≥1% rule against both, with seed wins 1/3 and family wins 1/6 vs A0, 1/3 and 2/6 vs A2; both paired intervals over three seed means include zero. Do not continue the path-specific search on this draw. Report the mechanism as unsupported under this development recipe; this does not refute all path supervision. Detailed evidence: `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md`.
