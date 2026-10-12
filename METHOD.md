# BL-RPNG-NLT method contract

The current primary candidate is **boundary-lifted Newton-path pretraining with
target-specific neuron-affine adaptation**.

For Dirichlet data `g`, the state is decomposed as `u = Eg + v`, where `Eg` is
the discrete harmonic extension and `v` has zero trace. The operator receives
`[v, Eg, grid, optional condition fields]`; its residual output is projected to
zero trace before reconstructing `u`. This makes the discrete boundary error
zero by construction.

During source pretraining, the first accepted Newton update reaching normalized
log-residual progress 0.25 supervises a hidden probe. The correction also has
zero trace and is therefore privileged numerical information in the same
tangent space as the learned residual state. Gradient balancing is computed
only on representation parameters shared by final and path losses. Probe-only
parameters are excluded from that norm.

For a target boundary, the default adapter freezes the backbone and learns
per-layer channel-wise affine parameters

`h_l' = gamma_l * h_l + beta_l`

together with the output projection. Target-stage Newton path supervision is
not part of the primary method unless it wins on a pre-frozen blind set. The
current seed42 post-query audit finds a strong source-path effect and only a
small target-path effect.

The intended error analysis separates: numerical discretization, source
representation, low-dimensional adapter estimation, and resolution
commutator errors. The cost analysis separates shared source pretraining from
target LR solves, adaptation, and repeated inference.

The frozen baseline contract separates the effects that are otherwise easily
confounded:

- `raw_legacy`: `[u, grid]`, with no explicit boundary projection;
- `boundary_augmented`: `[u, g on boundary, mask, grid]`, with exact output
  projection but no boundary-to-interior extension;
- `lift_conditioned`: `[u, Eg, mask, grid]`, with exact projection but without
  the residual-space coordinate change;
- `lifted`: `[u-Eg, Eg, grid]`, the BL-RPNG representation.

The `pino` target objective uses unlabeled fine-grid residual calibration with
no target HR solution. It is kept distinct from Newton-path supervision. The
secondary UNO backbone implements the same lifting and NLT contract and remains
locked until the FNO blind gate is passed.
