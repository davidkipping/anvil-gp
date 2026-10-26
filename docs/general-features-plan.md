# Plan: general noise model, apply_inverse, marginalisation, prediction, joint members

Agreed 2026-10-25. Prompted by the WASP103b_CHEOPS brief
(`WASP103b_CHEOPS/docs/upstream/anvil-gp_prompt.md`; reply sent as
`anvil-gp_reply.md` beside it, our copy `docs/wasp103-reply-draft.md`),
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

## Parked behind this

The boundary plan's open items: the square-root backward (rotation at
TESS 20-s, the 2-s floor), the kernel thread layout (m = 5-6 speed,
m > 6), IV (deferred), the seed-3 margin, the >= 100x gate.
