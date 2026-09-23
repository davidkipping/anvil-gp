# The float32 discipline, for a GP likelihood

anvil's rules ("preprocess in float64 on the CPU, sample offsets not
absolutes, reduce with structure, verify don't hope, re-anchor") carry over
unchanged. What a GP adds is that the *recursion itself* has hazards, in
places a chi-squared likelihood does not have any. Each of the five below is
a hard requirement, not a contingency, and each is pinned by a test.

## 1. Process noise via `expm1`, never by difference

`Q(dt) = P_inf - Phi P_inf Phi^T` is **forbidden**. Expand it for the SHO at
`omega_0 dt << 1` and three `O(omega_0 dt)` terms cancel to
`O((omega_0 dt)^3)`. Measured, at `omega_0 dt = 1e-3` in float32 the
difference form returns 1.79e-7 where the answer is 5.50e-8 — **225% wrong**.
By `dt = 1e-4` it has no significant digits at all, returning either exactly
zero or the float32 granule `2^-24`.

The rearrangement that removes it. With `a = c dt`, `u = d^2 dt^2`,
`E = e^{-2a}`, `p = a sinc(u)`:

    Q11/sigma^2 = E [expm1(2a) - 2 p (C + p)]
                = gamma3(2a) + E [8 a u sincm1(4u)
                                  + 2 a^2 u sincm1(u) (1 + sinc(u))]

where `gamma3(z) = 1 - e^{-z}(1 + z + z^2/2)` is a regularised lower
incomplete gamma (`= int_0^z e^{-t} t^2/2 dt`), evaluated by series below
`z = 2`. Every piece is regular at the origin, and the second line is also
bounded — the `expm1(2a)` of the first line overflows at long lag while the
product it sits in does not.

The closed form holds ~1e-7 relative in float32 all the way to
`omega_0 dt = 1e-6`. `Q22` needs no rearrangement (its two terms *add*), and
`Q12` is a product.

Matérn-5/2 gets the same treatment for `m = 3`: its process-noise entries are
combinations of `P(n+1, 2 lambda dt)` for `n <= 5`, by the same series switch.

## 2. Reference-subtracted log-determinant

At N = 1e5 with 500 ppm errors, `log|K| ~ -1.5e6`. Accumulating that in
float32 loses ~20 log-likelihood units — twenty times the scale a Metropolis
decision runs on.

So the recursion accumulates `log(S_n / yerr_n^2)`, an O(0.01–5) quantity,
and the parameter-independent remainder `-1/2 sum log(2 pi yerr_n^2)` is a
float64 host constant held **outside the graph** (`GPLogLike.log_norm_const`).
This is anvil's offset discipline applied to the log-determinant.

The reference is the *fixed* `yerr^2`, never the jitter-dependent
`yerr^2 + jitter^2`: the jitter is a sampled parameter, and its contribution
belongs inside the well-conditioned ratio where it can be differentiated.

Parity tests must compare **absolute** log-likelihood
(`GPLogLike.log_prob_absolute` / `.hi_absolute`) — a wrong additive constant
is invisible to MCMC and to every delta-based check.

## 3. Non-dimensionalised state

Terms hand back a state rescaled so that `P_inf = I` and the amplitude lives
in the observation vector instead (`h = [sigma, 0]` for one SHO). Without
this the SHO's natural state `(x, xdot)` has
`P_inf = sigma^2 diag(1, omega_0^2)`, and a scale-aware Cholesky pivot floor
on one component would sit at the *rounding scale of the other*. With it,
every covariance in the recursion is O(1) and dimensionless, and the entire
conditioning story reduces to the one axis that matters: `dt` versus the
damping and oscillation times.

Matérn-5/2's stationary covariance is not diagonal, so it takes a second
constant similarity (by the Cholesky factor of the lambda-scaled `P_inf`) to
reach `P_inf = I` as well — which lets the element algebra assume `P_inf = I`
uniformly across the whole kernel set.

## 4. No `inf * 0` in the overdamped corner

At long lag with `Q < 1/2`, `cosh(phi)` overflows to `inf` while `e^{-a}`
underflows to `0` — and their product is bounded by 1. Forming the factors
separately gives NaN and loses the chain.

`special.damped_cos_sinc` returns `e^{-a} cos_gen(u)` and `e^{-a} sinc_gen(u)`
as products, from the two real decay rates `a -/+ phi`. The caller supplies
those rates because only it can form `a - phi` without cancellation: for the
SHO it is `omega_0^2 dt / (c + |d|)`, and the literal subtraction loses
`log10(1/(2Q^2))` digits as `Q -> 0`.

## 5. Positive-definiteness failure is a defined outcome

With thousands of chains, warmup **will** visit float32-infeasible corners:
`sigma_GP/yerr` of 10–300 with a timescale far above the cadence makes the
innovation-variance update a catastrophic cancellation, and `phi` rounds to
exactly 1.0 for `c dt <~ 6e-8`.

Requirements, all implemented:

* the leaf innovation variance is floored at `s_floor_rel * yerr^2`
  (`GPPolicy.s_floor_rel`);
* `chol` floors each pivot at `eps * trace(A)/m`, which is what makes
  `dt = 0` (duplicate timestamps — celerite2 accepts them, so we must) a
  defined outcome rather than a NaN;
* a failed evaluation returns a large-negative **finite** log-likelihood
  with both branches of the `mx.where` sanitised, so the VJP contributes
  zero rather than `0 * inf`.

Still owed at M3: the *map* of the float32 feasibility boundary over
`(sigma_GP/yerr, timescale/cadence, Q)`, and the documented ParamSpec bound
guidance derived from it.

## Error growth: measured

`benchmarks/error_scaling.py` scans N, `tau/dt` and `sigma_GP/yerr`, and
decomposes each result into log-determinant and quadratic-form
contributions — they fail differently, so a single number hides which one is
in trouble. They separate cleanly: evaluating at `y = 0` zeroes every
quadratic contribution and leaves exactly `-1/2 sum log(S_n / yerr_n^2)`.

Result (M2 Max, SHO Q = 3, block 1024):

| scan | range | worst absolute | worst **relative** |
|---|---|---:|---:|
| N | 1e3 → 3.2e4 | 1.5e-2 | 8.0e-07 |
| `rho/dt` | 2 → 4e4 (i.e. `rho/T` 3e-4 → 5) | 2.3e-3 | 4.8e-07 |
| `sigma_GP/yerr` | 0.1 → 1000 | 3.4e-2 | 2.2e-06 |

**The relative error is flat at a few times float32 eps across every axis.**
The recursion is not compounding; what grows with N is `|logL|` itself, and
the absolute error follows it. Consequences:

* At N = 1e5 (`|logL| ~ 6e4`) the worst case extrapolates to ~7e-2 log-units
  — an order of magnitude under the ~1-unit Metropolis decision scale.
* At N = 1e6 it reaches ~0.7, i.e. the scale of the decision itself. That is
  where anvil's `reanchor_every` earns its keep, or where the per-block roots
  want composing in float64 on the CPU stream — the blocked structure makes
  that cheap, since there are ~1e3 roots rather than 1e6 leaves.
* The brief predicted a float32-infeasible corner at `sigma_GP/yerr` of
  10–300 with a timescale far above the cadence. Across that whole range the
  worst relative error measured here is 4.3e-7, with no sign of the
  innovation-variance cancellation. **The cancellation-free process noise
  appears to have removed the hazard.** This is one slice, not the map; M3
  still owes the full `(sigma/yerr, timescale/cadence, Q)` survey and the
  ParamSpec bound guidance derived from it.

## Where the tiling goes

`ChunkedGaussianLogLike.hi` tiles over the **data** axis, because a
white-noise likelihood is a sum over independent points. A GP factorisation
is global-sequential over data, so `GPLogLike.hi` tiles over **chains**
(<= 16 per float64 tile) and streams the data axis in recursion order. Peak
working set is one block of elements: `n_tile x block x (3m^2 + 2m + 1)`
numbers. The default 16-chain x 1024-point float64 tile at m = 4 holds
7.5 MB of elements and peaks around 60 MB measured, the rest being the
intermediates of the composition.

## Measured status

On the M2 Max, 256 chains x N = 1000, SHO plus jitter, sampled at the
defaults, `anvil.validate_precision` reports:

    |logL| typical magnitude : 1145
    |fp32 - fp64| median     : 1.4e-04
    |fp32 - fp64| max        : 2.2e-04
    OK: fp32 error is far below the ~1-unit scale of Metropolis accept
    decisions

At N = 1024 against a dense float64 Cholesky the float32 GPU path is within
5.9e-4 log-units, and the float32 gradient agrees with float64 central
differences to 0.14% relative — uniform across components, i.e. float32
rounding rather than a defect.

## Compiled gradients: a hard MLX limit

`mx.compile` over the **backward** pass of the whole likelihood fails for
m ≥ 2 with *"Too many inputs/outputs fused in the Metal Compiled
primitive which exhausted the available argument buffers for the kernel"*.
MLX fuses a long dependent elementwise chain into one Metal kernel with
more buffer arguments than its limit of 24 arrays per fused group allows.

Characterised:

* structural, not size-dependent — identical failure at N = 128 and N = 1000;
* absent at m = 1 (`RealTerm`);
* forward-only compilation is unaffected at every size and every m;
* **it is not the Cholesky.** An earlier version of this note blamed the
  unrolled factorisation, which made the problem look unfixable. Bisected
  2026-09-19: `chol`, `combine`, `combine_leaf`, `leaf_elements` and the
  whole leaf → fold → tree chain each compile under `mx.vjp` on their own,
  and so does `(phi_table, q_table, h) → ell` end to end. What fails is
  the front end — `kernel_params` plus `terms.matrices`, i.e. the fused
  chain inside `_sho_matrices` (`damped_cos_sinc`, `gamma3`, `sincm1` and
  their ~20 intermediates). Splitting the VJP at the term-matrix table
  therefore recovers most of it; that is tracked in the README's "What is
  left" rather than done. M4's backward kernel did not remove it: the
  over-fused group is in the front end both paths share.

anvil's ChEES kernel compiles its leapfrog step, so ChEES-HMC cannot run on a
GP target without `mx.disable_compile()`. With it, 30 ChEES steps over 64
chains complete in 7.2 s with zero divergences and a finite chain. The
gradient-free stretch move never hits it.

M4's backward kernel was expected to be the proper fix, replacing this
graph with a hand VJP. It was not: the kernel path aborts identically,
because the group MLX over-fuses is `terms._sho_matrices`, which the kernel
consumes rather than replaces (a `RealTerm`, whose front end is trivial,
compiles its gradient on both paths). Splitting the VJP at the term-matrix
table is the remaining route. The failure is a process abort from inside
Metal, not a Python exception, so it cannot be caught.

## MLX behaviours worked around

All pinned as regression tests in `tests/test_mlx_gotchas.py`, so an MLX
upgrade that fixes one shows up as a failing test rather than as dead
workaround code.

* **`mx.array(np_float64_array)` silently returns float32.** The dtype must
  be named. Downstream arithmetic with a float64 operand promotes the result
  back, so a "float64 path" can carry float32 data with every dtype it
  reports looking correct. `likelihood.a64` exists solely for this.
  *(Worth checking in anvil itself:
  `ChunkedGaussianLogLike.hi` builds its per-tile `x`, `y` and `w` with bare
  `mx.array(...)` on float64 numpy slices.)*
* **A single-element strided slice has a wrong VJP.** The backward pass
  fills `start:stop` rather than `start:stop:step`, so `x[0:2:2]` — which
  selects only index 0 — returns a gradient of `[1, 1]` instead of `[1, 0]`.
  Values are unaffected; only gradients are wrong. A binary reduction tree's
  last level is always `k = 2`, so this silently corrupted every gradient the
  element algebra produced. `elements._pairs` reshapes to
  `(..., k//2, 2, ...)` and indexes instead of striding. Confirmed on CPU and
  GPU, float32 and float64, MLX 0.32.2 (the current release). A ready-to-file
  upstream report is in
  [mlx-issue-strided-slice-vjp.md](mlx-issue-strided-slice-vjp.md).
* **float64 `exp`/`sin`/`cos` are only float32-accurate**; `log`, `expm1`,
  `sqrt` and arithmetic are true float64. `special.exp_acc` and
  `special.sincos` supply real float64 versions for the verification path.
  Note `exp(-x)` cannot be recovered as `1 + expm1(-x)`: at `x = 20` that
  leaves ~8 significant digits.
* **`mx.linalg` is CPU-only with no VJPs.** Hence the hand-rolled unrolled
  Cholesky and triangular solve in `linalg.py` for the differentiable path,
  with the LAPACK route reserved for the value-only float64 path.
* Numpy-scalar poison, `.item()` forcing evaluation inside `mx.compile`,
  both-branch `mx.where` sanitisation including masked denominators, and
  `mx.compile` retracing on shape change — inherited from MetalPlanet's
  hard-won list and respected throughout.
