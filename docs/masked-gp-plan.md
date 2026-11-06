# Plan: held-out ("masked") GP conditioning

Agreed 2026-11-05. Prompted by a request from the TOI-199 b JWST analysis
(`~/Downloads/anvilgp_masked_gp_request.md`), scoped as a general library
feature: any fit whose measurement is the *shape* of a short feature
(ingress/egress, spot crossings, flares, moon dips), where a GP fitted to
all the data is free to absorb the feature as noise.

## The construction

Split the data into S (conditions the GP) and M (held out), a fixed
partition given at construction.  With ``r = y - mean(v) - offsets``:

    log L = log N(r_S; 0, C) + log N(r_M - P r_S; 0, W_M)
    C = K_SS + W_S,   P = K_MS C^-1   (the GP's two-sided conditional mean)

M is scored with its **white noise only**; that is the masking.  Adding
the predictive covariance ``K_MM - K_MS C^-1 K_SM`` to the M term gives
back the full GP exactly (``p(r) = p(r_S) p(r_M | r_S)``), so that mode
is not an option -- it is a test.  The construction is the joint Gaussian

    Sigma = [ C       K_SM                ]
            [ K_MS    K_MS C^-1 K_SM + W_M ]

whose Schur complement is ``W_M``, so it is a proper likelihood and its
evidence is defined.  Every primitive that consumes "the covariance"
(`gram`, the linear marginalisation, `linear_posterior`) consumes
``Sigma`` in place of ``K``; ``log|Sigma| = log|C| + log|W_M|``, and the
host constant `log_norm_const` is unchanged (it is per datum, over all
N).  Parameter map, priors, `natural()` and `default_hypers` are
untouched: a masked fit has the same parameter vector as an unmasked one.

Not modelled: correlated noise *inside* M.  Far inside a window longer
than the kernel's timescale ``P r_S -> 0`` and M is scored white about the
bare mean, which can overstate a feature's significance unless the M
timescales are independently shown to be white.  The docstring says so.
A sampled inflation of M's white noise needs no new API: label the M
points as their own instrument and give it a jitter (or a `noise_fn`
term).

## How it is computed (O(N), same element algebra)

The S points split into runs ("segments") at the windows of M.  Each
segment is reduced to one scan element as the likelihood reduces its
blocks, but with its first step taken at lag 0 (``Phi = I, Q = 0``), so
the element's input is the state *at* its first point.  The true gaps
become explicit transition elements.  Then:

* the S term is the root of ``[start, E_0, G_1, E_1, ..., E_K]`` (start =
  the stationary prior), reduced by the pairwise tree;
* a prefix over the segments gives the filtered state at each window's
  left edge, a suffix the information ``(eta, J)`` at its right edge;
* each held-out point fuses the two through its own lags to each edge
  (three batched `combine`s, vectorised over the window's points).

Sequential depth grows with the number of windows, not of held-out
points.  Everything is MLX ops, so float64 on the CPU is the `hi` path
and float32 with autodiff is the gradient: one implementation, both
precisions.  The hook is a single one, `_evaluate`, which `hi`, `gram`'s
float64 route and the MLX path all go through; the marginalisation,
`gram` and the refusals follow from it unchanged.  `predict` and
`linear_posterior` run the float64 smoother on S, with the M points as
prediction points for the masked Gram.

## Phases

### 0. API, oracle, float64 value

- `GPLogLike(..., held_out=...)`: a boolean ``(N,)`` mask or a list of
  ``(t_lo, t_hi)`` windows, resolved to a fixed mask at construction (so
  it cannot depend on the parameters).  At least one S point; an empty
  mask is the unmasked likelihood.  `make_target` passes it through.
- `oracle.dense_masked_loglike`: ``log N(r; 0, Sigma)`` from ``Sigma``
  above, built directly.
- `hi` via the masked evaluation; `predict` (S-conditioned, M points get
  ``P r_S``), `linear_posterior` (masked Gram), `gram` (``Y^T Sigma^-1
  Y``).  `apply_inverse` refuses a masked likelihood (nothing needs
  ``Sigma^-1 Y`` yet).
- Tests: masked = dense ``Sigma`` to ~1e-10 over interior, leading,
  trailing and several windows, single points, ties across a window
  edge, m = 1 to 6, `noise_fn`, offsets and instruments, `mean_fn`;
  the predictive-scoring identity on the oracle; marginalised basis
  (flat and Gaussian) and its posterior against dense GLS under
  ``Sigma``; `predict` at M against dense ``P r_S``; refusals.

### 1. float32 gradient on the MLX path

The same evaluation in float32, differentiated by MLX.  A masked
likelihood takes the MLX path (`GPPolicy(metal=True)` warns and falls
back, as for any problem the kernel cannot serve).  Gate: value and
gradient against float64 within the float32 budgets the unmasked MLX
path meets on the same data; ChEES runs; cost measured on a quiet GPU.

**Status 2026-11-05.** Phase 0 done (ff1e712): float64 equals the dense
``Sigma`` to ~1e-11 for m = 1 to 6, convolved terms, ties at a window
edge, `mean_fn`/offsets/`noise_fn`, and bases (shared and per chain, flat
and Gaussian prior).  Phase 1 measured, all but the cost:

- *float32, small problems* (N = 300): value 4e-7 relative at m = 2 (a
  few ulps), 3.6e-6 at m = 6 with a leading window (2e-7 unmasked);
  gradients 1e-6 to 6e-6 relative, as unmasked.
- *float32 at the requester's scale* (N = 2800, one-minute cadence, SHO,
  ingress/egress or the whole transit held out): where masking is
  appropriate -- the GP's interpolation into the window comparable to
  the white noise -- the error is the unmasked path's: 1e-4 to 1e-3
  log-units at sigma <= 3 yerr, 0.05 against 0.065 for slow systematics
  at sigma = 30 yerr, rho = 30 hr.  Gradients 1e-6 to 3e-5.
- *Where it is not*: with a large GP left unmodelled inside the window
  (sigma = 300 yerr, rho = 10 hr: residuals of 96 sigma there, an M term
  of -2e5) the error is 88 log-units against 0.9.  Located: the forward
  state and the backward information are each float32-accurate (1e-5,
  4e-7); the fusion deep inside the window loses the information gain
  over the prior there (~1e5, the same factor kappa measures).  The
  caveat already excludes that regime and a sampler does not stay in it
  (the likelihood there is 1e5 log-units down); a covariance-form
  backward pass would remove it if it ever mattered.
- *ChEES* on a masked target (N = 1000, 256 chains, 300 + 300): acceptance
  0.96, no chains lost, widths within 4-15% of the float64 Laplace widths
  (log jitter 0.094 against 0.089).  A reduced version is a slow test.
- *Memory* at 1024 chains x N = 2800, value + gradient: 2.25 GiB masked
  against 2.02 GiB unmasked on the MLX path (the kernel: 0.04 GiB).
- *Cost*: measured by the requester on their own idle GPU, 2026-11-06
  (`~/Downloads/anvilgp_masked_speed_gap.md`): a joint two-detector fit,
  N = 2818 each, Matern-3/2, `error_scale`, a per-chain polynomial basis
  marginalised, 14% held out in two windows.  32 optimiser starts 561 s
  against 17.4 s on the kernel unmasked (32x); an ensemble step 0.21 s at
  96 chains against 0.028 s at 128 (8x as measured); a fit to R-hat < 1.1
  1.5-2.5 hr projected against 7-10 min.  Their correctness check on real
  posterior draws: `hi` within 1e-6 to 3.4e-6 of an independent celerite2
  reference, float32 within 2.5e-4 to 1e-3 log-units.  Not yet measured
  here (needs a quiet GPU): masked against unmasked *on the MLX path*, to
  split masking's own cost from the MLX path's.
- *Optimiser iterations* (CPU device, no timing; N = 2818, Matern-3/2,
  quadratic basis, two windows of 199 points, 32 starts): median 9-10
  iterations masked against 10 unmasked, all converged.  Masking does not
  make the objective harder; the 32x is cost per evaluation.
- *Where the cost comes from, from the code*: the masked evaluation is
  the unmasked MLX reduction plus O(windows + held-out points) of glue,
  so masking itself should cost little over the MLX path.  But the MLX
  path polarises a marginalised basis -- ``1 + (p+1)(p+2)/2`` likelihood
  evaluations, batched along the chains (7 at p = 2) -- where the kernel
  carries every column in one filter (1.7-1.9x).  That and the MLX path's
  own ~40x (value + gradient) account for the ratios above.

Verdict: too slow at the requester's scale; Phase 2 is needed, and it has
to include the marginalised basis to help them.

### 2. Kernel path (only if Phase 1 is too slow at the requester's scale)

Window edges aligned to the kernel's block boundaries, the window
fusion fed by the seeds kernel's prefix and suffix, and the M term's
cotangent injected at those boundaries in the VJP.

Scoped 2026-11-06, from reading `metal.py`; not started.  The S points
are just an irregular series, which the kernel already serves; what is
new is the M term and its gradient.  Four pieces, each usable alone:

- *2a. Blocks that start where segments start.*  The kernels place
  block ``i`` at ``i * blk_len``.  A block-start array instead, through
  the forward, replay, seeds, backward and square-root kernels, so every
  window edge is a block boundary.  Mechanical, but in every kernel.
- *2b. Value.*  The S term is the existing replay.  The filtered state
  at each window's left edge is the entry state `entry_states` already
  returns for the block after it.  The information at the right edge is
  the suffix the seeds kernel already forms (``suf_eta``, ``suf_J``), or,
  more accurately in float32, the entry state of a second, time-reversed
  run of the same kernels.  In the ``P_inf = I`` gauge every term but a
  convolution uses, reversal is ``Phi -> Phi^T``, ``Q -> I - Phi^T Phi``.
  Fusing the two at the held-out points is the glue `_evaluate_masked`
  already has, in MLX.  This alone serves an ensemble sampler (values
  only).
- *2c. Gradient.*  MLX differentiates the glue and gives cotangents on
  those entry states.  The backward's seeds are the adjoints of each
  block's entry state, formed by a backward recursion over the blocks.
  An extra cotangent on one block's entry state enters that recursion
  additively, so the seeds kernel (dense and square-root) takes a
  per-block injection, and `backward_ckpt` runs unchanged.  The
  derivation has to be checked against docs/m4-plan.md first.
- *2d. A marginalised basis.*  ``Gram_S + D^T W_M^-1 D``, ``D = Z_M - P
  Z_S``, needs each column's smoothed mean at M: the column versions of
  2b and 2c (`column_entry_means`, `scan_gram`'s VJP).  The requester's
  fit needs this; without it a masked fit with a basis stays on the MLX
  path.

Acceptance: the requester's numbers in Phase 1 (``hi`` within ~1e-6 of the
dense ``Sigma``, float32 within the unmasked kernel's budget on their
posterior draws), the dense tests in tests/test_masked.py on the kernel,
and a cost within ~1.5-2x the unmasked kernel, measured on a quiet GPU.

**Status 2026-11-06: values done (2a, 2b, 2d's value), then the gradient
(2c, 2d's gradient).**  Agreed order: values first.  As built, two
things differ from the scope above:

- *The right edge comes from a time-reversed run, not the suffix.*  In
  every term's ``P_inf = I`` state, time reversal is a sign flip ``S``
  with ``Phi^T = S Phi S``, ``I - Phi^T Phi = S Q S`` and ``h S = h``
  (`Term.reversal_signs`; pinned for every term by
  `test_time_reversal_is_a_sign_flip`).  So the kernel run unchanged
  over S reversed, with the same lag table, is the backward filter: no
  new numerics, and the square-root boundary on both sides.  A convolved
  term has no ``S`` and stays on the MLX path.
- *Block exits, not entries.*  The replay can write each block's exit
  state (`metal._EXIT_1`), so the trailing and leading windows need no
  special case: a window's left edge is the exit of the forward block
  that ends there, its right edge the exit of the backward block that
  ends there.

The window step is the two-filter form in covariance terms,
``B G^-1 a + A G^-T b`` with ``G = A + B - A B`` (`_held_out_mean`): no
inverse of either side's covariance, so it does not lose the information
gain the MLX path's information-form fusion loses.

*The gradient (2c).*  `metal.scan_masked` (and `scan_masked_mc` for a
basis) returns S's value and every block's exit state in both series;
the window step downstream is MLX, so MLX hands the scan's VJP
cotangents on the exit states.  Those enter the backward sweep through
the seeds.  The seed of a block is the adjoint of its exit state; the
likelihood's own part keeps its closed form, and the exit states' part
is a linear recursion backward over the blocks (`metal.inject_seeds`):

    a <- T^T a,   Pi <- T^T Pi T + sum_c sym(a_c g_c^T),

with ``T`` a block's mean map and ``g`` its data's sensitivity to its
entry mean -- both from the existing r-sensitivity pass, both in
covariance form -- from the filter's first-order ``dm_exit = T dm_entry +
T dP_entry g`` and ``dP_exit = T dP_entry T^T``.  The backward kernels
are unchanged.  The placed-block variant passes ``blk_start`` in
``blk_len``'s buffer: the multi-column backward with a split basis
already uses 31, the Metal limit.  (Until 2c a custom VJP differentiated
the MLX evaluation instead.)

Accuracy, float32 against float64 (dense-``Sigma`` verified):

- N = 300, every term (m = 1 to 6) and window layout, auto blocks and
  blocks of 16: 7e-6 to 1.4e-4 log-units, against 2.6e-5 to 6.4e-4 on
  the MLX path.
- Marginalised basis, shared and per chain, flat and Gaussian prior:
  1.5e-5 to 5e-5 (2e-4 to 4.5e-4 on the MLX path).
- The requester's scale (N = 2800, one-minute cadence, transit plus
  ingress and egress held out): sigma = yerr, 8.2e-5 against 1.0e-4
  unmasked; sigma = 30 yerr, rho = 30 hr, 7.6e-4 against 1.6e-4 (the
  held-out residuals all come from the same two edge states, so their
  few-ulp float32 error adds coherently); 60x better than the MLX path's
  5e-2.  The excluded regime (sigma = 300 yerr, rho = 10 hr): 0.49
  against the MLX path's 88.  (These are the window walk's, with anchors
  every 16 points; the per-point version before it measured 2.4e-3 and
  0.54, the unanchored walk 6.4e-3 and 1.75.)

Gradient accuracy, float32 against float64 central differences:

- N = 300, every term and layout, auto blocks and blocks of 16: 3e-7 to
  7e-6 relative (MLX path 1.4e-7 to 2.3e-5).  One jitter and two per
  instrument, sqrt and dense boundary; mean model, offsets and a
  per-datum `noise_fn` (5e-7).
- Marginalised basis: 4.7e-6 to 9.9e-6 (MLX path 1.2e-5 to 6.1e-5).
- The requester's scale: sigma = yerr 9.6e-7, sigma = 30 yerr 7.4e-6,
  sigma = 300 yerr 5.8e-6 -- the unmasked kernel's own (8.5e-7, 5.4e-6,
  1.4e-5), where the MLX path's masked gradient is 1.5e-6, 1.5e-4 and
  1.7e-3.

Cost, measured on a quiet machine (benchmarks.md has the table): at
N = 2818 with 14% held out, 1.7-2.2x the unmasked kernel's value and
2.0-2.5x value + gradient from 128 to 1024 chains; with the requester's
noise model and baseline, 1.7x and 2.0x at 128 chains, 2.0x and 2.0x at
1024.  How it got there (2026-11-06/07):

- First measurement, 10-90x.  The window step was MLX ops, and MLX's
  batched matmul spent 10.7 ms per (chains x points) 2 x 2 product.
- Unrolled (`linalg.matmul`, also now in `inv_logdet_fast`'s VJP), then a
  per-point kernel each way: 2.9x / 4.1x at 1024 chains.  Ablating the
  step showed the rest was MLX plumbing around that kernel -- gathering
  each point's edge states and propagators, the scatter-add backward of
  those gathers -- and the term algebra at the 199 distinct lags from a
  point to its edges.
- The walk (`metal.window_walk`): one thread per (chain, window) reads
  its edge states from the scans' exits and steps through its points
  with the cadence's propagator (the filter's own prediction step, so
  only the few lags already in the table), fusing at each point.  Its
  VJP is the fusion's backward per point (`_SRC_FUSE_VJP`, run wide) and
  the walks reversed per window (`_SRC_WALK_REV`), each step's
  propagator adjoint summed in registers and flushed per lag slot as the
  scans' backward does.  The forward stays one kernel: at 128 chains,
  where launches dominate, splitting it cost 0.3 ms.
- Anchors.  Walking compounds the cadence propagator's float32 rounding:
  for a slow process ``Phi(dt)`` differs from ``I`` only past its fourth
  digit, and 170 steps took the error at sigma = 30 yerr, rho = 30 hr
  from 1.8e-3 (each point propagated directly) to 6.4e-3.  Every 16th
  point from each edge now steps from the edge itself with its own lag
  (15 table rows here, not 199): 7.6e-4, for 0.2 ms of value + gradient
  at 1024 chains.
- The backward series skips its replay unless a window leads the data:
  its block exits are its next blocks' entry states, which the boundary
  already gives.  The two directions share one `q_factor`.

With the window step stubbed out the masked likelihood is 1.8x the
unmasked value and 1.9x value + gradient at 1024 chains, so what is left
is the two scans.

### Status

Phase 2 done; Phase 3's docs done (the tutorial's "Holding a feature out
of the GP", with the warm start from an unmasked fit; benchmarks.md; the
README).  The reply to the requester is drafted
(docs/masked-gp-reply.md) and needs sign-off before it is sent.

### 3. Docs and reply

Docstrings (including the caveat above), a tutorial section, the cost
in benchmarks.md, and a reply to the requester -- the reply needs
sign-off before it is sent.

## Not done, by design

- ``held_out_score="predictive"``: identical to no mask; a test only.
- A guard against a parameter-dependent mask: the mask is a constructor
  argument, so it cannot be one.
- A separate ``held_out_inflate``: an instrument jitter on the M points
  is the same thing, sampled.
