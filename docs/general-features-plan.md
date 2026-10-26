# Plan: general noise model, apply_inverse, marginalisation, prediction, joint members

Agreed 2026-10-25. Prompted by the WASP103b_CHEOPS brief
(`WASP103b_CHEOPS/docs/upstream/anvil-gp_prompt.md`; reply sent as
`anvil-gp_reply.md` beside it, our copy `docs/wasp103-reply.md`),
but scoped as general library features: anvil-gp is a general-purpose,
celerite2-like O(N) GP library for MLX, and nothing here is specific to
one science case.

## Out of scope, as a principle

Kernels of more than one input. The celerite class is sums and products
of exponential-family terms of **one sorted input** (which can be any
sorted quantity, not only time); a distance over several inputs has no
total order, so no semiseparable form and no O(N) filter. Sums of 1-D
kernels on different inputs, and (cos, sin) embeddings for circular
inputs, are excluded for the same reason. No dense O(N^3) path.
`docs/kernels.md` states this as the library's principle (item 5).

## Items, in order

### 1. General noise model: `noise_fn` (2-3 days)

celerite2 takes `diag`, a per-point noise variance the user builds; our
`jitter` (added variance, per instrument) is one expression for it. The
general API is `noise_fn(v, yerr) -> (n_chains, N)` variance, alongside
`mean_fn`, with gradients by autodiff through it. The shipped default
stays `yerr^2 + jitter^2`, so nothing changes for existing users.

- MLX and `.hi` paths: the variance array replaces `yerr^2 + jit^2`.
- Fused kernel: read a per-chain, per-datum variance array in place of
  `r_ref[i] + jit2[instrument]`; the backward writes a per-datum
  variance adjoint (same shape as the residual adjoint it already
  writes). That retires `MAX_JITTER` and the kernel's instrument index.
- **Measure first:** the cost of the variance array and its adjoint at
  N = 1e5 against today's compact jitter. If it is small (guess
  +10-20%), the compact path is retired; if not, both stay and the
  compact one serves the default.
- Unchanged: `r_ref` stays the reference variance for the log-det
  reference subtraction (float32.md rule 2) and the leaf floor; the
  kappa guard reads the actual variance.
- Priors: `scale_hyper` and helpers for the common forms (error scale
  s^2 yerr^2, scale plus jitter, per-instrument); a bounded default.
- Tests: the default reproduces today bit for bit or to float32
  rounding; a scale-only `noise_fn` equals the analytic reparametrised
  likelihood (data / s, kernel / s, jitter / s, minus N ln s); kernel vs
  MLX vs `.hi` value and gradient for heteroscedastic noise; the old
  per-instrument jitter tests expressed through `noise_fn`. Folded in:
  irregular non-time input with exact ties, both paths against the
  dense oracle (the WASP reply's Q1 check, as a general test).

### 2. Joint targets: member protocol in anvil-gp, composition in anvil (1-2 days)

"Several independent log-densities under one target, some parameters
shared" is generic sampler-side composition, so it is proposed to anvil
as a primitive (note: `docs/anvil-joint-target-note.md`, needs sign-off).
anvil-gp's side: `GPTarget` as a well-behaved member -- names, `u0`,
`natural()`, refusals, `.hi`, `model_log_norm` -- in whatever protocol
anvil defines. Fallback if anvil declines: anvil-gp builds the
composition itself, on the same member protocol.

### 3. `apply_inverse` / `gram`, then linear marginalisation (3-5 days)

**The primitive.** `gp.apply_inverse(v, Y)` = K^-1 Y and `gp.gram(v, Y)`
= Y^T K^-1 Y for any columns Y, batched over chains, differentiable. The
filter's covariance recursion does not depend on the data, so p + 1
columns cost p + 1 copies of the m-vector mean recursion and share
everything else; the Gram matrix comes from whitened innovations with
no cancellation (polarisation by pseudo-chains is the fallback only: it
subtracts ~N-sized numbers in float32).

Staged:
1. `.hi` and the MLX path, multi-column (the MLX path is autodiff).
2. Fused-kernel forward with p + 1 mean columns; the square-root
   boundary's entry means per column, covariances shared.
3. Fused-kernel backward: with cotangent Lambda on the Gram matrix, the
   hyperparameter gradient is a sum of quadratic-form adjoints along
   Lambda's eigenvectors, which the existing single-residual backward
   computes -- p + 1 sweeps, no kernel change. A shared-covariance
   multi-column backward only if the cost matters.

**Marginalisation layer.** `linear_basis(v, x_mean, m0) -> Phi
(chains, N, p)`, rebuilt each evaluation; flat or Gaussian prior on
beta. Restricted-likelihood form from the Gram matrix of [r Phi],
keeping -1/2 ln|Phi^T K^-1 Phi|; the p x p solve and log-det on the host
side through `linalg.inv_logdet` (MLX ops, differentiable). Basis
columns rescaled internally (Phi^T K^-1 Phi ~ N / sigma^2). A singular
normal matrix gets the finite failure value. `linear_posterior(v)`
returns beta-hat and its covariance. The existing per-instrument
offsets are linear parameters, so marginalising them becomes an option.

Acceptance: brute-force quadrature over beta to 1e-8 relative (p = 2,
N ~ 500); gradients against float64 finite differences on every path; a
calibration test (marginalised and beta-sampled targets agree within
MCMC error on a known-truth problem); a benchmark line for what
marginalisation costs.

### 4. `predict`, a layer on the primitive (1-1.5 days)

Conditional mean and variance at new inputs (sorted into the data's
coordinate), batched over chains: the mean from K_* K^-1 r via
`apply_inverse`, the variance from a Kalman smoother over the same
recursion. A separate method; it does not touch the likelihood path or
its cost. Checked against `oracle.dense_predict` in float64.

### 5. Docs and benchmarks (~1 day)

`kernels.md`: the one-input principle and why. README and tutorial:
`noise_fn`, `apply_inverse` / `gram`, `linear_basis`, `predict`, joint
targets. Benchmarks: noise_fn cost, marginalisation cost, predict cost.

## Size

8-12 working days. The least certain parts: item 1's kernel cost
(measured first), item 3's stage 3 cost, and item 2's dependence on
anvil's answer.

## Item 5 done: docs and benchmarks (2026-10-26)

- `kernels.md`: "One input, and why" -- the O(N)-only rule, and what a
  second input maps to instead (mean / `linear_basis`, `noise_fn`, a dense
  joint member).
- README: "Beyond the plain likelihood" (each feature, its cost, the
  refusal); `noise.py` and `predict.py` in the layout; two open items
  (parallel-in-time `predict`, a gradient for `apply_inverse`); this plan
  and the WASP reply in the documentation table.
- `tutorial.md`: "Beyond the plain likelihood", five runnable sections on
  the tutorial's data (noise model, marginalised level + slope with
  `linear_posterior`, `predict`, `gram` / `apply_inverse`, a two-member
  joint target), figures from one run; `tests/test_docs.py` pins the
  names, shapes and the `gram` agreement.
- `benchmarks.md`: "What the general features cost", hand-written from
  the measurements above; the stale prediction paragraph replaced.

The plan is complete.

## Parked behind this

The boundary plan's open items: the square-root backward (rotation at
TESS 20-s, the 2-s floor), the kernel thread layout (m = 5-6 speed,
m > 6), IV (deferred), the seed-3 margin, the >= 100x gate.

From item 4: **parallel-in-time `predict`.** Today's smoother is one
host-side step per merged point, bound by NumPy's per-call overhead (~20
us a point: 2.0 s at N = 1e5 for one chain, 7.3 s for 64). Proposed:
split the merged grid into ~sqrt(N + M) blocks; form each block's
filtering element in float64 (vectorised over blocks), a prefix over the
blocks for their entry states, then the within-block filter vectorised
over chains x blocks; backward, each block's exit adjoints from the
suffix in closed form (`r = G^T (eta_s - J_s mu)`, `Jt`, harmless in
float64), then the within-block smoother vectorised the same way.
Sequential steps fall from 2(N + M) to ~3L + 2B (L, B the block length
and count): ~1700 instead of 2e5 at N = 1e5, estimated ~10x. Same
answers to float64 rounding; the dense tests in `tests/test_predict.py`
are the gate.

## Item 1 done: `noise_fn` (2026-10-25)

- **API.** `GPLogLike(..., noise_fn=)` with `noise_fn(v_noise, d)` -> the
  total per-datum variance, `d` an `anvilgp.noise.NoiseData` (`yerr`,
  `x` = `x_mean` or `t`, `inst`) sliced alike, since the reference path
  evaluates per block; `ParamMap.noise` for its columns (raw model-space
  values, last in `v`). `anvilgp.noise`: `NoiseModel` (fn + specs +
  `natural`), `error_scale`, `scale_and_jitter`, each per instrument;
  `make_target(noise=)`, with `n_jitter` defaulting to 0 then. Combining
  `noise_fn` with `param_map.jitter` raises. The plan's `scale_hyper`
  became the noise models' own bounded log-uniform specs.
- **Kernel.** A third `_jitter_src` variant: a per-datum extra variance
  `(npts, n_chains, 1)` in the residual's layout, read per observation,
  with the backward writing each observation's variance adjoint. Checked
  bit-identical in value and to 1e-7 in gradient against the compact
  variant at m = 2, 4, 6. Cost at 1024 chains, with the array built and
  laid out each call: m = 2 forward 1.04-1.10x, value + gradient
  1.05-1.06x; m = 4 1.00x. So the default single jitter keeps its hoisted
  read (bit-identical, no cost), and **several per-instrument jitters now
  go per datum too: the compact variant and `MAX_JITTER` (eight) are
  retired**; twelve jitters run on the kernel and match the MLX path.
- **Floor.** `sqrt_blocks` floored the leaf variance at `r_ref / 2`, which
  an error scale below 0.71 would hit; now `1e-6 r_ref`, the innovation
  floor's default. Unchanged for every variance >= that.
- **Found on the way.** MLX's float64 `exp` and `sin` are float32-accurate
  (known; `special.exp_acc` / `sincos` exist for it), so a hand-written
  `noise_fn` or `mean_fn` using `mx.exp` puts a float32 floor under `.hi`
  (1e-5 in log L, and float64 finite differences off by 2%). The noise
  models use `exp_acc`; the docs say so.
- **Tests** (`tests/test_noise.py`): every path against the dense float64
  oracle for m = 2, 4, 6 (float64 to 1e-8, float32 gradient to 1e-4 of
  float64 differences); the error scale as an exact reparametrisation;
  s = 0.3 over several kernel blocks; a convolved term's correction;
  per-instrument noise; the refusals; kappa; `make_target`; the per-block
  slicing; and the WASP reply's check as a general test -- an irregular
  non-time input with a quarter of the spacings tied, both paths.

## Item 2, member side done (2026-10-25)

The note went to anvil (`docs/anvil-joint-target-note.md`); the
composition itself waits on its answer. anvil-gp's side, done now:

- `GPTarget` exposes the member protocol: `model_log_prob` /
  `model_log_prob_hi` (model space, no Jacobian), `model_log_norm`,
  `specs` (one `SpecHint` per column, in `names` order), alongside the
  existing `u_ref()` and `natural()`. The class docstring says how to
  compose: sum the model-space functions and apply one joint transform.
- **Bug fixed:** `make_target` never passed the likelihood's constant
  (`-1/2 sum log(2 pi yerr^2)`, kept out of the float32 graph) to anvil as
  `model_log_norm`, so `anvil.evidence` and `target.log_norm_const` on a
  GP target were off by it. Relative evidences on the same data were
  unaffected (the constant depends only on the error bars).
- `tests/test_members.py`: two members with a shared mean parameter,
  composed by hand as anvil or a fallback would -- one transform from the
  members' specs, the model-space sum; the naive sum of members'
  `log_prob(u)` differs by exactly the shared column's Jacobian (the trap
  the note describes); the shared column's gradient is the members' sum;
  `log_prob_hi` and `validate_precision` work on the composition; a
  refused chain in one member fails the joint chain; the constant
  reproduces the dense oracle's absolute value to 1e-8.

Next: item 3 (`apply_inverse` / `gram`, then marginalisation), which does
not depend on anvil. The composition (anvil's or our fallback) is picked
up when anvil answers.

## Item 3, first half done: marginalisation by polarisation (2026-10-25)

Decision C (agreed): polarisation now, on every path, as the general
implementation and the test oracle; the multi-column kernel next.

- **Measured first.** Prototype (no library change): float64 exact;
  float32 kernel marginal error 1e-3 (s/e 10) to 2e-2 (s/e 100) at first.
  That was cancellation in the marginal *formula*, not in polarisation: a
  200-sigma baseline left in r made r^T K^-1 r and its projection agree to
  ~1e-6. Centring r on a white-noise least-squares beta0 first (exact:
  the marginal is invariant to r -> r - Phi beta0) took it to 7e-5, about
  twice the plain likelihood's error on the same residual.
- **Built.** `GPLogLike(linear_basis=, linear_prior=)`,
  `linear_basis(v_mean, x, m0) -> Phi (n, N, p)`; flat prior (restricted
  likelihood, -1/2 ln|C| kept) or Gaussian `(mu, cov)`;
  `linear_posterior(v)` (float64); `make_target(linear_basis=,
  linear_prior=)` and `GPTarget.linear_posterior`. Internally
  `_marginal`: centre, scale the columns, evaluate the existing
  likelihood once on all `1 + k(k+1)/2` combinations as extra chains
  (`_raw_resid`, a residual supplied in place of y - mean on every path),
  polarise into the Gram matrix, small solves by `linalg.inv_logdet`. A
  refused chain or a singular basis gets the failure value. (Revised
  2026-10-26 after review: a Gaussian prior makes a singular basis
  proper, and it was getting a wrong value; see docs/review-2026-10-26.md,
  "A2 as fixed".)
- **Accuracy.** float64: the brute-force quadrature over beta of the plain
  likelihood (the acceptance test asked for) to 1e-8 relative; the
  Gaussian prior against the dense marginal; `linear_posterior` against
  dense GLS to 1e-7. float32: value within 5e-3 of float64 at s/e 10,
  gradients (mean parameter included, which moves the basis through m0)
  to 1e-4 relative on the kernel, 1e-3 on the MLX path. The float32 floor
  is set by forming the residual on the device: the plain likelihood with
  the same device-side mean is 3e-4 off here, the marginal ~4x that.
- **Calibration** (slow test): ensemble sampler on the marginalised and
  the beta-sampled targets, posterior means of depth, GP amplitude, rho
  and jitter agree to 0.02-0.05 posterior sd.
- **Cost** (kernel, value / value + gradient, against the plain
  likelihood): p = 1 1.6-2.3x at 256 chains, 3.3-4.8x at 1024; p = 2
  3.1-5.5x at 256, 7.0-11.8x at 1024 (N = 1e4 and 1e5). Under-filled GPUs
  hide part of it; at 1024 chains it is the full 7 evaluations and more.
  This is what the multi-column kernel is for.

## Item 2, composition: anvil built it (2026-10-25)

anvil answered the note by building it: 0.12.0 (in anvil's working
tree, not yet committed when read) adds `anvil.joint(members, shared)` ->
`JointLogDensity` -- shared columns first, then `"<member>.<name>"`, one
Jacobian and one set of declared priors, `model_log_norm` summed, a
float64 path when every member has one, failures kept -- makes
`TransformedLogDensity.model_log_prob` / `model_log_prob_hi` public (our
one request), and makes `Transform` reject duplicate names, which broke
the hand-composed tests of the member side (they concatenated two
members' specs under the same names). No fallback is needed.

anvil-gp's remaining piece: `anvilgp.anvil.joint_u0(jt, members,
n_chains)`, a joint start from each member's `u_ref()` placed by
`jt.member_columns`. `tests/test_members.py` now runs against
`anvil.joint` (skipped before anvil 0.12.0). Caveat, documented on
`joint_u0`: sharing a GP *hyperparameter* across members needs equal
specs, which `make_target`'s data-derived bounds do not give.

## Item 3, second half, Stage A: the multi-column forward (2026-10-26)

Agreed staging (option C, revised after review): A, the forward on the
kernel; measure; C, a multi-column backward. Stage B (the gradient as
k + 1 single-column backward runs) was dropped as a shipping stage: each
of its runs replays the covariance recursion that A removes from the
forward, and polarisation is already the gradient oracle.

- **The filter.** `metal.scan_gram(...)` returns `-1/2 log|K|` (relative
  to `r_ref`) and `Z^T K^-1 Z` for k columns from one filter: the
  covariance recursion, `S` and the gain once, k means
  (`_FWD_MC`). `_FWD_STEP`'s data-dependent lines are now tokens
  (`MEAN_PREDICT`, `INNOVATION`, `MEAN_UPDATE`), so the single-column
  source is unchanged (a declaration split, diffed) and the sqrt
  backward, when it comes, ports the multi-column variant mechanically.
- **Entry states.** Covariances from the existing boundary run once, on
  column 0 (they are not a function of the data). Each column's entry
  means from the blocks' mean maps `m_exit = T_b m_entry + d_b`: one
  affine pass from the shared entry covariances (`_SRC_AFFINE`) and a
  prefix over blocks (`_SRC_MEAN_PREFIX`). No inverse anywhere, unlike
  the information-form boundary. The first version ran the boundary with
  the columns as k virtual chains, and the boundary was then most of the
  cost: 8.7 / 16.1 ms at k = 3 / 6 against 3.0 for one (N = 1e4 x 1024
  chains, SHO).
- **Wiring.** `GPLogLike._gram_metal`, a `custom_function` whose VJP
  differentiates `_gram_polarised` (the polarisation, factored out of
  `_marginal`). The gradient therefore costs polarisation's plus this
  forward until Stage C: +6-15% on value + gradient, p = 2-10. The MLX
  path and `.hi` polarise as before.
- **Two costs that were not the filter**, found by profiling, fixed for
  every route (polarisation included):
  - the white-noise centring fit's batched `Phi^T W Phi` -- MLX's batched
    matmul for a p x p result over a long N ran at ~5 GB/s, 3.9 ms at
    N = 1e4 and 38 ms at 1e5 x 256 chains, more than the likelihood. Now
    `metal.normal_equations`, a reduction kernel (a shared basis is read
    once): 0.3 / 1.1 ms at p = 2.
  - the p x p inverses: `linalg.inv_logdet` is p^3 unrolled launches,
    12 ms forward and 74 ms gradient for 1024 10 x 10 matrices.
    `linalg.inv_logdet_fast`: one Metal thread per matrix forward
    (`metal.small_inv_logdet`), the analytic VJP; 0.30 / 0.57 ms.
- **A compiler cliff.** The Gram accumulation fully unrolled ran 10-15x
  slower at exactly k = 7-8 than on either side; the loop shape kept
  (outer rolled, inner unrolled, `1/S` hoisted) was the fastest of four
  at every k from 3 to 16.
- **Accuracy.** Gram against dense float64 to 2-5e-6 relative, one block
  and eight, both boundaries. The marginal is as close to float64 as
  polarisation or closer: over 24 evaluations at m = 6 (N = 4096, k = 8)
  median 1.6e-3, max 4.0e-3, against 2.7e-3 and 7.0e-3; per column it
  equals the plain single-column likelihood's own floor.
- **Cost, forward** (N = 1e4, 1024 chains, SHO, quiet machine), against
  the plain likelihood:

  | p (k) | polarised | multi-column |
  |---|---|---|
  | 2 (3) | 5.6x | 2.2x |
  | 5 (6) | 15.9x | 3.1x |
  | 10 (11) | 47x | 7.2x |

  At k = 3 the remainder is about one-third filter passes (boundary 2.8
  ms, affine 1.8, replay 2.3) and one-third the marginal's own algebra
  (columns built and transposed, ~3 ms). `MAX_GRAM_COLS = 32` only
  bounds the kernel's size: the filter beat polarisation at every k
  measured, 1.5-3.3x (k = 4-26, m = 2) and 4-62x (k = 4-21, m = 6), and
  polarising 26 columns at m = 6 tripped the GPU watchdog.
- `benchmarks/marginal_cost.py` reproduces the table (`--grad` for value
  + gradient).

Next: Stage C, the multi-column backward, which removes the polarised
gradient (and its +6-15%).

## Item 3, Stage C: the multi-column backward (2026-10-26)

`scan_gram` is differentiable on its own now, and `GPLogLike._gram_metal`
no longer polarises for the gradient.

- **Seeds.** The single column's closed form (docs/m4-plan.md), `m_bar =
  cot r`, `P_bar = (cot/2)(r r^T - Jt)`, generalises to the cotangent
  `c_ld` on the log-determinant half and `gs` (symmetrised) on the Gram
  matrix as `seed_m[a] = -2 (gs R)_a`, `seed_C = -1/2 c_ld Jt - R gs
  R^T`; one column, `c_ld = cot`, `gs = -cot/2`, is the original exactly.
  `-1/2 c_ld Jt` is the existing seed of the zero residual at cotangent
  `c_ld` (so the forward's one boundary now runs on zeros, not column 0).
  `R` (one `r_c` per column at every block exit) does not come from the
  suffix element's `eta_s - J_s mu` -- the cancellation the M4 plan
  measured amplifying float32 error ~1e3 -- but from a linear suffix of
  the actual innovations: the mean adjoint's recursion does not involve
  the covariance adjoint, so each block's contribution `g_b = sum_t
  (h^T Phi_t T_{t-1})^T v_t / S_t` comes from one pass from the true entry
  means (`_SRC_RSENS`) and `r(b) = g(b+1) + T(b+1)^T r(b+1)`
  (`_SRC_R_SUFFIX`).
- **The kernel.** `backward_ckpt_mc` is `_SRC_BWD_CKPT` with nine
  asserted replacements (`_BWD_MC_EDITS`): per-column means, checkpoints,
  adjoints and `dy`; the covariance adjoint -- most of the kernel -- is
  the single-column source. Step terms `v_bar = 2 (gs v)/S`, `s_bar =
  -c_ld/(2S) - v^T gs v / S^2`. All 78 existing kernel sources hash
  identically before and after (the single-column VJP's fold, budget and
  seed code moved into shared helpers, comments with them).
- **Accuracy.** Against the same functional built by polarisation through
  `scan_loglike`'s backward, random cotangents: 1e-8-8e-7 relative in the
  parameter and the column gradients, m = 2, 4, 6 and a convolved SHO,
  no / one / per-datum noise, one block and many, both boundaries. In the
  hard regime (sigma/yerr 100, rho/dt 1000, 16 blocks) the marginal's
  gradient is within 2.1e-6 of float64 finite differences (polarisation
  3.7e-6); 1.0e-6 / 5.8e-7 at 32 / 64 blocks.
- **Cost, value + gradient** (N = 1e4, SHO, quiet machine), against the
  plain likelihood:

  | p (k) | 1024 chains: polarised | multi-column | 256: polarised | multi-column |
  |---|---|---|---|---|
  | 1 (2) | 3.2x | 1.7x | | |
  | 2 (3) | 5.3x | 2.0x | 2.4x | 1.5x |
  | 5 (6) | 14.9x | 3.1x | 5.8x | 2.1x |
  | 10 (11) | 44.6x | 7.5x | | |

  Forward: 1.8x, 2.0x, 2.6x, 6.3x at 1024 chains. At m = 4 and 6 the
  covariance adjoint dominates and three columns cost the backward kernel
  +8-9% over one. The backward's sub-block is 16, measured (sizing it to
  the single-column register budget gave 4 at k = 11, 12% slower).

## Item 4 done: `predict` (2026-10-26)

`GPLogLike.predict(v, t_pred, return_var=True)` (and `GPTarget.predict`)
-> `(mean, var)`, float64 `(n_chains, M)`: the GP component's posterior
at `t_pred` given the data. The mean model and the noise are not included
(celerite2's `include_mean=False`): `x_mean` can be any covariate array,
so only the caller can evaluate `mean_fn` at new points.

- **Method** (`anvilgp/predict.py`): a Kalman filter then the modified
  Bryson-Frazier smoother over the merged, sorted grid of data and
  prediction points, prediction points being steps without an
  observation. The MBF form is the adjoint `(lambda, Lambda)` -- `E[x|y]
  = m_pred + P_pred lambda`, `Cov = P_pred - P_pred Lambda P_pred` -- and
  needs no inverse of a predicted covariance, which the RTS form does and
  which is singular at duplicate timestamps. O(N + M), float64 on the
  host, vectorised over chains (chunked to ~512 MB), and entirely off the
  likelihood path.
- **General by construction**: the residual and per-datum noise variance
  come from `leaf_inputs`, the likelihood's own, so `mean_fn`, offsets,
  jitter and `noise_fn` are what the likelihood uses. A `linear_basis`
  is smoothed as extra means beside the residual (one covariance) and the
  linear parameters integrated over `linear_posterior`: mean `s_r - s_Phi
  b`, variance `+ s_Phi C_beta s_Phi^T`. A TermConvolution predicts the
  exposure-averaged process (variance includes its zero-lag excess) and
  refuses prediction points within the exposure width of anything, where
  the state-space form does not hold.
- **Accuracy**: against the dense float64 conditional to 1e-14-1e-15 on
  m = 1, 2, 3, 6 terms, inside, outside and on the data, duplicate
  timestamps, mean model + offsets + `noise_fn`, flat and Gaussian
  marginalised bases, and a convolved SHO (tests at 1e-10 / 1e-9).
- **Cost** (SHO, M = 1000): 0.21 s at N = 1e4 and 2.0 s at N = 1e5 for
  one chain; 0.55 / 7.3 s for 64 chains; 2.9 s for 512 at N = 1e4. The
  loop is bound by NumPy's per-call overhead (~20 us per merged point).
  Not optimised further: the next step would be parallel-in-time blocks
  (block elements, a prefix over blocks, then the within-block filter and
  smoother vectorised over blocks), estimated ~10x at N = 1e5, if
  prediction cost turns out to matter.
- Found on the way: `GPTarget.linear_posterior` passed `mx.array` of a
  float64 array, which MLX makes float32, so the "float64" posterior was
  computed from float32-rounded draws. Fixed (`dtype=mx.float64`).

## Item 3's primitive, public: `gram` and `apply_inverse` (2026-10-26)

The plan put marginalisation and `predict` on a public `apply_inverse` /
`gram` primitive (and the WASP-103 update said so). What was built first
was narrower: the Gram matrix internally (kernel multi-column filter,
polarisation elsewhere) and `predict` on the smoother, with no public
primitive and no `apply_inverse` at all. Now:

- `GPLogLike.gram(v, Y)` -> `(n_chains, k, k)` = `Y^T K^-1 Y`,
  differentiable in `v` and `Y`, `Y` `(N,)` / `(N, k)` / `(n_chains, N,
  k)`. float32 on the kernel by `scan_gram` (forward and backward), on the
  MLX path by polarisation; float64 `v` by polarisation of the float64
  path on the CPU. Refused chains get NaN. Measured against dense float64
  solves: 5-7e-8 (kernel), 2-3e-7 (MLX; polarisation degrades for large
  or nearly parallel columns), 5-7e-15 (float64); gradients in `v` against
  float64 finite differences and in `Y` against `K^-1 Y (W + W^T)`.
- `GPLogLike.apply_inverse(v, Y)` -> `K^-1 Y`, float64, value only, from
  `predict`'s smoother (now shared, `predict._smooth`): `-d l/d y_t =
  v_t / S_t - K_t^T lambda_t` -- no `y - E[f|y]` subtraction. 7e-14
  against `np.linalg.solve`, sums of terms, exact ties and a convolved
  term included; `Y^T apply_inverse(Y)` equals the float64 `gram` to
  1e-10. `GPTarget.apply_inverse` passes through.
- **Not built: a differentiable `apply_inverse`** (gradients of `K^-1 Y`
  in the hyperparameters). It needs second derivatives through the filter,
  which neither the kernel's VJP nor the MLX path provides, and no feature
  here needs it. Built on request.
- Found on the way (writing the tutorial): `linear_posterior` evaluated
  every draw in one float64 batch, ~6 MB per draw at N = 2000, so a whole
  posterior (7.7e4 draws) asked for ~440 GiB and pushed the machine's free
  memory to 24% before it was stopped. Now tiled by `hi_chain_block`, as
  `hi` is: flat 0.40 GiB from 64 to 4096 draws, ~3.3 ms a draw. The
  float64 `gram` is tiled the same way.
