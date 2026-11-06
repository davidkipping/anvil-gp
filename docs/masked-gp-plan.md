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
- *Cost*: owed, on a quiet GPU.  The ChEES run above took 17 min on a
  possibly shared GPU (not a timing); the MLX path is ~40x the kernel
  unmasked, so this decides Phase 2.

### 2. Kernel path (only if Phase 1 is too slow at the requester's scale)

Window edges aligned to the kernel's block boundaries, the window
fusion fed by the seeds kernel's prefix and suffix, and the M term's
cotangent injected at those boundaries in the VJP.

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
