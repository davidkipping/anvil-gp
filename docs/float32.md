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
  zero rather than `0 * inf`.  The value is **not a constant**:
  `-1e30 * (1 + pull)`, with `pull` a chain's distance from the reference
  point over the GP's own columns, so it slopes back towards feasibility.
  A constant made the failure region perfectly flat -- in float32
  `-1e30 + (-120) == -1e30`, so even anvil's (finite, correctly inward)
  Jacobian was absorbed -- and a flat region is not a rejection: stretch
  walkers started there diffused to `|u| ~ 1e11`. The slope is relative so
  float32 can resolve it at that magnitude, and `stop_gradient`-ed because
  its true derivative is of order 1e30; failed lanes keep a zero gradient.
  Consequence, pinned in `test_contract.py`: wherever the likelihood
  succeeds, values and gradients are bit-identical to the constant
  sentinel, so no posterior moves -- only recovery from the infeasible
  region changes.

* a chain past the float32 feasibility boundary is **refused** before it is
  evaluated, and gets the same failure value (`GPPolicy.kappa_max`, next
  section).

## 6. The feasibility map, and the kappa guard

Measured 2026-10-12 with `benchmarks/kappa_map.py` (M2 Max, MLX 0.32.2, SHO
kernel, error = |float32 − float64 `.hi`| in log-units). Owed since M3.

**One number predicts the error.** Not amplitude, not timescale, but

    kappa = k(0) / (h Q h^T + r)

the process variance over the variance one step adds: the GP's own process
noise over one median cadence, plus the white noise `r` (median `yerr^2`
plus jitter). It measures how nearly deterministic the state is between
samples, which float32 (`eps ~ 1.2e-7`, `1/eps ~ 8e6`) cannot resolve once
`kappa * eps` nears 1. Amplitude alone does not do it: at `sigma/yerr = 1e14`
with a timescale of two cadences the error is 0.002, because each step adds
a lot of fresh variance; at `sigma/yerr = 100` with `rho/dt = 1000` and
`Q = 30` it is 0.2, because each step adds almost none.
`GPLogLike.kappa(v)` computes it per chain.

**Fixed data, parameters anywhere** (where a wandering chain goes; the
tutorial data, max error over 2125 points spanning `sigma/yerr` 1e-2 to
1e14, `rho/dt` 2 to 1000, `Q` 0.05 to 500, jitter 1e-3 to 1e3 times `yerr`):

| kappa | fused kernel, N = 2e3 / 2e4 / 1e5 | MLX path, N = 2e3 / 2e4 / 1e5 |
|---|---|---|
| 1e4 – 1e6 | ≤ 0.009 / 0.023 / 0.096 | ≤ 0.014 / 0.11 / 0.54 |
| 1e6 – 1e7 | ≤ 0.17 / 0.087 / 0.24 | ≤ 0.11 / 0.65 / 3.3 |
| 1e7 – 3e7 | ≤ 0.75 / 0.71 / 0.84 | fails |
| > 3e7 | fails | fails |

The boundary does not move with N; only the low-kappa floor does, and that
is the summation error of the section above.

**Data drawn at the parameters** (where a posterior lives; N = 4000, one
draw each, 45 settings): harder, and the scatter between draws at the same
kappa is a factor of ten. Fused kernel: ≤ 0.08 below kappa 1e4; 0.008 to
0.2 from 1e4 to 1e5; 0.008 to 0.8 from 1e5 to 1e6; 0.6 to 14 near 3e6; 1.9
to 19 near 2e7; 322 or a failure above 6e7. The MLX path is worse
throughout: up to 1.9 near kappa 1e4, 0.03 to 3 from 2e4 to 8e5, and 7 to
499 from about 1e6.

**The guard.** `GPPolicy.kappa_max` (default `1e6`, `None` to disable)
refuses any chain past it: the likelihood returns the failure value of
section 5 with a zero gradient, without trusting a float32 evaluation that
is wrong there by a log-unit or more. Pinned in `tests/test_contract.py`:
everywhere at or below `kappa_max` the value is bit-identical to the
unguarded one. The gradient agrees to rounding: the MLX path's gradient is
not bit-reproducible run to run, guarded or not (~1e-5), and on the fused
kernel, where kappa reads the forward's own lag table (since 2026-10-24;
rebuilding it cost +29-66% of an eager call at 2000 x 256), the backward
moves by up to 4e-6 relative, with both versions equally close to a float64
finite difference. The guard costs nothing measurable on either path. `make_target`'s defaults put the tutorial posterior at
kappa 64 to 124.

**Why it matters for sampling.** A chain that reaches kappa ~1e9 sees a
density that is rough by a log-unit between neighbouring points, so it
accepts only steps smaller than float32's resolution of `u`, which do not
move it at all; acceptance then reads 1.00. ChEES adapts one step size from
the harmonic mean of all chains' acceptance, so it follows that chain down.
Measured at 1024 chains, `u0(spread=1.0)` (docs/anvil-chees-stuck-chains-measurement.md):
one chain at kappa 3e9 held the step size at 2.9e-7 through 3000 warmup
steps, with 78 ordinary chains frozen behind it. With the guard the same
seed ends warmup at 4.7e-4 (300 steps) and 1.6e-3 (3000 steps), and none of
the chains still behind is a refused one: all have kappa below 6e3. Those
are a wide start that needs a longer warmup, not a precision problem.

**Limits.** Measured on the SHO kernel at a regular cadence; the other terms
use the same formula and are tested for it, not mapped. `kappa` is taken at
the *median* cadence, so a dataset with much shorter gaps can be worse
locally (section 7 measures how much). A posterior whose bulk sits past `kappa_max` cannot be sampled in
float32 at all: every chain is refused. `GPLogLike` warns at construction
when the data's rms over the median error bar exceeds `sqrt(kappa_max)`,
since only then can a posterior get there (`kappa <= sigma^2 / r` for any
timescale).

## 7. Compose in the innovation, through the factor

Found 2026-10-23, fixed 2026-10-24. Sums of terms (m >= 4) were wrong by
1 to 100 log-units, in value and gradient, whenever two observations fell
much closer together than the cadence at high SNR. Kappa saw none of it:
it reads the median lag, and was 3.7.

**Mechanism.** A block whose first observation follows the previous one
almost at once predicts that observation almost exactly: `hQh << r`, so the
block's information matrix `J2` reaches `sigma^2 / r`, 1e4 at
`sigma/yerr = 100`, and `eta2 ~ 2e4`. The general composition formed its
log-normaliser as three cross terms of ~1e4 each, cancelling to O(1):

    -1/2 b1^T Jt b1 + eta2^T G b1 + 1/2 eta2^T M eta2

with `G = I - M J2` built as a matrix. Each entry of `G` carried an
absolute error of `eps |M| |J2| ~ 1e-3`, and `eta2^T G b1` multiplied it by
3e4. The stored elements were not the problem: composing the float32 leaves
in float64 was within 0.008, and the per-composition errors summed to the whole.
The rank-one leaf fold never had the problem, because it is already written
in the innovation `beta`. That is why the default MLX path (16-point folds)
was hurt less than the pure tree, and m = 2 hardly at all.

**Fix.** The same algebra, written in the innovation `d = eta2 - J2 b1` and
through the Cholesky factor `Y` (with `M = Y Y^T`, `V = J2 Y`, `w = Y^T d`):

    b1^T (eta2 - 1/2 J2 b1) + 1/2 |w|^2         (log-normaliser)
    b1 + Y w                                     (G b1 + M eta2)
    d - V w                                      (G^T d, into the next eta)

`elements.combine`, both Metal block folds and the backward seeds use it.
Both halves matter: the innovation form alone left `G^T d` with the same
1e-3 error, and the composite `eta` carried it into the next level, which
cost another log-unit on a uniform-random cadence.

**Measured** (SHO + SHO, N = 4096, data on-model; value error in log-units,
then worst relative gradient error against a float64 finite difference):

| cadence, `sigma/yerr` | before: default / pure tree / kernel | after |
|---|---|---|
| 20 near-duplicates (1e-4 cadences), 100 | -3.8, 53% / +18.3, 260% / -0.002, 0.08% | ≤ 0.007, 0.4% |
| uniform random, 100 | -2.9, 110% / +58.9, 270% / -1.3, 2.9% | ≤ 0.005, 0.13% |
| uniform random (another draw), 100; every `sub_block` 1-1024, dense and blocked leaves, m = 2, 4, 6 | up to 101 | ≤ 0.012 |
| near-duplicates or uniform, 1000 (value only) | up to 2.1e4 / 2.9e5 / 297 | 0.02-0.88 / 0.09-1.2 / 0.03-0.14 |

**What is left** is the last row. At `sigma/yerr = 1000` a near-duplicate
leaf stores `ell = ld - g^2/2` with `g^2 ~ 1e6`, so float32 storage alone
costs a few hundredths of a log-unit per such leaf. Composing the float32
leaves *in float64* is off by 0.25-0.52, and the sequential filter by up to
0.14. That is a limit of the dense information representation, not of the
arithmetic. A square-root information form would remove it, at the price of
a QR per composition. Regular cadences are unaffected at every SNR tried.

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
  appears to have removed the hazard.** This is one slice, not the map. The
  map (section 6) found that the hazard is set by `kappa`, not by
  `sigma/yerr` on its own.

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
  left" rather than done. M4's backward kernel did not remove it, and
  neither did a `custom_function` boundary at the term-matrix table; the
  actual cause was `special.sincm1` and it is fixed — see below.

**Resolved 2026-09-27.** anvil's ChEES kernel compiles its leapfrog step,
so ChEES-HMC could not run on a GP target without `mx.disable_compile()`.
The cause was a single function: `special.sincm1` selects between a series
and a `sinc_gen` call, and `sinc_gen` is itself a three-way `mx.where` with
each arm separately sanitised. That nested both-branch evaluation is what
exhausts the 24-array limit — **collectively, not alone**: `sinc_gen` by
itself compiles, `sincm1` by itself compiles, and only `sincm1` inside
`_sho_matrices`' surrounding expression fails. No single function is the
culprit; `sincm1` is simply the contributor that can be removed, because
`_sho_matrices` calls it only under `|u| <= 0.5`, which bounds the argument
by the series threshold and makes the other branch unreachable there.
`sincm1_series` drops it. A self-contained reproducer is in
[upstream/mlx-issue-fused-argument-limit.md](upstream/mlx-issue-fused-argument-limit.md). The
result is bit-identical over the whole threshold interval in float32 and
float64, and `mx.compile(mx.value_and_grad(...))` now succeeds for every
term on both paths. No `mx.disable_compile()` anywhere.

Two earlier diagnoses were wrong and are recorded because each looked
convincing. The unrolled Cholesky was blamed first; every piece of the
element algebra compiles under `mx.vjp` on its own. `terms._sho_matrices`
was blamed next and the prescribed fix was a `custom_function` boundary at
the term-matrix table — measured, and it changes nothing, because the
fused group is inside the term algebra and a boundary around the whole of
it does not break the chain.

**And the failure was never an uncatchable process abort**, as this
document and the M4 notes claimed. It is an ordinary Python
`RuntimeError`; the process survives. (`mx.eval` *inside* a compiled
function is rejected outright with a different error, so eval placement was
never the difference.)

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
  `mx.compile` retracing on shape change — inherited from Elome's
  hard-won list and respected throughout.
