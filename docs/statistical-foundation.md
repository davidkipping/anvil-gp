# The statistical foundation

## Every celerite kernel is a linear Gaussian state-space model

A celerite-class covariance function — a sum of damped exponential and
damped-oscillator terms — is exactly the autocovariance of a linear Gaussian
state-space model (SSM). This is the CARMA correspondence (Kelly et al.
2014); it is an identity, not an approximation, and it comes with a
closed-form stationary covariance `P_inf` for initialising the recursion.

Two consequences that this package is built on:

**The Kalman prediction-error decomposition and the celerite semiseparable
Cholesky compute the same number.** Identical log-determinant, identical
quadratic form, in exact arithmetic. The choice between them is purely
numerics plus GPU scheduling. That in turn means the reference
implementation and the fast path *need not share a recursion* — they need
only agree numerically. `anvilgp` exploits this: the sequential filter
(`kalman.py`) and the blocked tree reduction (`elements.py`) are different
recursions over the same leaf inputs, and testing them against each other is
the cheapest strong check available on either.

**The SSM route buys two things celerite cannot give.**

1. *A propagator that is smooth through Q = 1/2.* The celerite coefficient
   parametrisation carries a `1/sqrt(4Q^2 - 1)`, singular at critical
   damping. A 4096-chain warmup crosses Q = 1/2 routinely, and a float32
   sampler cannot survive a pole in its own likelihood. Here the propagator
   is built from `cos(d dt)` and `sin(d dt)/d`, which are **entire functions
   of `u = d^2 dt^2` alone**: they continue analytically from underdamped
   (`u > 0`) through critical (`u = 0`) into overdamped (`u < 0`), where
   they become `cosh` and `sinh(.)/.`. `d` itself never appears.

2. *Exact Matérn-3/2 and Matérn-5/2*, neither of which is
   celerite-representable. Matérn-3/2 needs no new code at all: it **is**
   the SHO at Q = 1/2, with `omega_0 = sqrt(3)/rho`. (`tests/test_terms.py`
   asserts this as a design claim, not as a curiosity.) celerite2's
   `Matern32Term` takes an explicit `eps` approximation parameter for the
   same reason; at its default `eps = 0.01` it differs from the exact
   Matérn-3/2 likelihood by 6.5e-3 log-units on the test problem, and
   converges to `anvilgp`'s answer as `eps -> 0`.

## The likelihood is a reduction, not a scan

Write the contribution of one contiguous stretch of the series as an
*element*: what it does to the state that enters it.

    L(x_out, x_in) = N(x_out; A x_in + b, C)
                     * exp(-1/2 x_in^T J x_in + eta^T x_in + ell)

An affine-Gaussian transition, times the unnormalised likelihood of the
stretch's observations expressed as a function of the incoming state.
Composing adjacent stretches is the Gaussian integral

    L(x_out, x_in) = int L2(x_out, z) L1(z, x_in) dz

which is associative — this is the Särkkä & García-Fernández (1905.13002)
parallel-filter element, with a sixth component, the log-normaliser `ell`,
carried along.

The key structural point: **the marginal likelihood needs only the root of a
binary reduction tree, never the prefix at every leaf.** The first element
has `A = 0`, `J = 0`, `eta = 0` (its "incoming state" is the stationary
prior, already absorbed), so those stay zero all the way up; the composite of
the whole series is `N(x_N; b, C) exp(ell)`, and integrating out `x_N` leaves
`exp(ell)`. **log L is the root element's `ell`.**

That halves the working set relative to a prefix scan, and it makes the
float32 accumulation of both the log-determinant and the quadratic form a
pairwise tree — the friendliest summation order there is (error `~ eps
sqrt(log K)` rather than `eps K`).

### The composition, in closed form

Every inverse is routed through two Cholesky factorisations rather than an
explicit inverse — MLX has no GPU `inv` (issue #1238) and no `cholesky` VJP
(#1781), and the factored route keeps the symmetric objects exactly
symmetric:

    L1 = chol(C1)
    Sig = I + L1^T J2 L1         (SPD, all eigenvalues >= 1: no pivoting)
    Y  = L1 chol(Sig)^{-T}
    M  = Y Y^T   = (I + C1 J2)^{-1} C1
    G  = I - M J2 = (I + C1 J2)^{-1}
    Jt = J2 - (J2 Y)(J2 Y)^T = (I + J2 C1)^{-1} J2

(`Sig` is an `m x m` matrix.  The scalar `S` further down is a different
object -- the one-step innovation variance `S = h Q h^T + r_var`, written
`s_q` in `elements.py`.  They were both called `S` until 2026-09-19.)

    A   = A2 G A1
    b   = A2 (G b1 + M eta2) + b2
    C   = A2 M A2^T + C2
    eta = A1^T G^T (eta2 - J2 b1) + eta1
    J   = A1^T Jt A1 + J1
    ell = ell1 + ell2 - sum log diag(chol Sig)
                      - 1/2 b1^T Jt b1 + eta2^T G b1 + 1/2 eta2^T M eta2

`log|I + C1 J2| = log|Sig|` by Sylvester's identity, so the log-determinant is
read straight off a factor that has already been computed.

That is the algebra; the code evaluates it in the innovation
`d = eta2 - J2 b1` and through the factor, with `V = J2 Y` and `w = Y^T d`:

    G b1 + M eta2           = b1 + Y w
    G^T (eta2 - J2 b1)      = d - V w
    the three cross terms   = b1^T (eta2 - 1/2 J2 b1) + 1/2 |w|^2

The two sides are equal in exact arithmetic and not in float32: when a block
starts just after an observation, `J2 ~ sigma^2/r` and the left-hand sides
cancel ~1e4-sized terms through a `G` that is only good to ~1e-3. See
docs/float32.md section 7.

### The rank-one special case, and why it matters

When `e2` is a raw **leaf**, its information matrix is rank one —
`J2 = z z^T` with `z = (h Phi)^T / sqrt(S)`, and `eta2 = g z` with
`g = y / sqrt(S)`, where `S = h Q h^T + r_var` is the scalar one-step
innovation variance (`s_q` in the code). Every inverse above then collapses under
Sherman–Morrison. With `p = C1 z` and `alpha = 1 + z^T p`:

    (I + C1 J2)^-1     = I - p z^T / alpha
    (I + C1 J2)^-1 C1  = C1 - p p^T / alpha
    (I + J2 C1)^-1 J2  = z z^T / alpha
    log|I + C1 J2|     = log(alpha)              [determinant lemma]

so both Choleskys and the triangular solve disappear and thirteen tiny matrix
products drop to three. The log-normaliser collapses further: with
`beta = g - z^T b1` the three quadratic cross terms sum to
`g^2/2 - beta^2/(2 alpha)`, and once the leaf's own `-g^2/2` cancels what
remains is

    ell = ell1 - 1/2 [ log(alpha S / r_ref) + beta^2 / alpha ]

— exactly the Kalman prediction-error decomposition, with `beta/sqrt(alpha)`
the normalised innovation and `alpha S` the innovation variance. Watching the
filter fall out of the element algebra is a good sign the algebra is right.

This is what `chunked_reduce` exploits, and what M3's fused kernel runs
sequentially in registers: O(m^2) per observation with nothing to factor.

It also means a leaf never has to be *assembled*. Storing `Phi` and `Q` as
blocks plus the three vectors `kappa`, `z`, `g` determines every component of
the element, and `combine_leaf_blocked` substitutes the identities directly —
so `Phi`'s zeros are never multiplied and `A` and `C` are never formed. That
costs `2 sum m_i^2 + 2m + 2` numbers per observation against the dense
element's `3m^2 + 2m + 1`, which at m = 10 is a 5x cut in the largest array
the evaluation holds. See the README's multi-term section for where it pays.

Note the asymmetry between `b` and the `ell` cross term: `b` carries the
extra `M eta2`, `ell` does not. Folding them together (they differ by one
term that looks shareable) double-counts `eta2^T M eta2` — a *constant*
offset per composition, so every relative check still passes and only the
absolute likelihood is wrong. `tests/test_elements.py` guards this by
evaluating the composed element against a brute-force numerical Gaussian
integral as a function of `(x_out, x_in)`, which pins every component at once.

## Where the mean model's gradient comes from

    d logL / d(mean model) = alpha = (K + Sigma)^{-1} r

The mean model's gradient is the GP-whitened residual, chain-ruled into
whatever VJP the mean model already has. This decouples the hard
(hyperparameter) gradient engineering from the solved (transit) path:
Elome's analytic VJP is unchanged and simply receives a different
cotangent.

## What is deliberately excluded

Rejected because they fail the exactness rule — the likelihood a Metropolis
ratio consumes must be the exact marginal likelihood, evaluated
deterministically:

* **FFT/Toeplitz and Whittle.** Gaps break Toeplitz structure, and the
  Whittle likelihood is a quasi-likelihood: its posterior is not the
  posterior. (A periodogram *peak* as a chain initialiser is fine and is
  what `priors.periodogram_period` does.)
* **SKI / BBMM / stochastic-Lanczos log-determinants.** A noisy likelihood
  breaks detailed balance and is not a valid pseudo-marginal scheme (the
  noise is in the log, not in an unbiased estimate of the likelihood).
* **RFF, SVGP.**

Admissible only as a secondary cross-check, never as the likelihood:

* **HSGP** (Solin & Särkkä basis GPs), for low-Q smooth kernels with
  m <~ 100. Blockers, recorded so they are not re-litigated: `m` scales
  roughly linearly with Q (sharp spectral peaks) and with the
  baseline/timescale ratio; published `(m, c)` guidance is not calibrated
  for quasi-periodic kernels; the approximation deficit launders into biased
  Q and inflated jitter in a way that posterior predictive checks do not
  catch; and it needs per-chain m-by-m factorisations that MLX cannot
  GPU-execute natively. Any use requires a posterior-level A/B against the
  exact likelihood.

## Known v1 limitations

* Exposure-time integration (`TermConvolution`) forms its process noise
  by subtraction, so its float32 path is 50–100x less accurate than an
  unconvolved term's on data far off-model (no measurable cost on data
  drawn from the model). See
  [kernels.md](kernels.md#indefinite-and-still-exact).
* No 2D / multiband GPs.
* `predict` / conditional mean is an offline float64 utility
  (`oracle.dense_predict`), deliberately outside the hot path.
