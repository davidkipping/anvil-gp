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

Found 2026-10-23, fixed 2026-10-24 (`46d392d`). Sums of terms (m >= 4: two
SHOs, SHO + Matern, `RotationTerm`) were wrong by up to 160 log-units in
value, and up to 270% in gradient, whenever the GP was nearly predictable
from one sample to the next. That covers ordinary photometry of an active
star: a regular TESS cadence with a 3-day rotation at s/e = 30 (amplitude
over error bar) was 3 log-units wrong, and 70 at s/e = 100. Near-duplicate
timestamps are the extreme case, not the only one. Kappa was 1e3-1e4 in
every case measured, far below `kappa_max`, so the guard refused none of
them. Single terms (m = 2) were not affected.

**Mechanism.** A block whose first observation is predicted almost exactly
from the state entering it -- `hQh << r`, which is what a timescale much
longer than the cadence does at high s/e, and what a near-duplicate
timestamp does at any timescale -- has an information matrix `J2` of order
`sigma^2 / r` (1e4 at s/e = 100) and `eta2 ~ 2e4`. The general composition
formed its log-normaliser as three cross terms of ~1e4 each, cancelling to
O(1):

    -1/2 b1^T Jt b1 + eta2^T G b1 + 1/2 eta2^T M eta2

with `G = I - M J2` built as a matrix. Each entry of `G` carried an
absolute error of `eps |M| |J2| ~ 1e-3`, and `eta2^T G b1` multiplied it by
3e4. The stored elements were not the problem: composing the float32 leaves
in float64 was within 0.008, and the per-composition errors summed to the
whole. The rank-one leaf fold never had the problem, because it is already
written in the innovation `beta`. That is why the default MLX path
(32-point folds) was hurt less than the pure tree, and why the Metal kernel,
which runs only a dozen or so general compositions per chain, was hurt less
again.

**Fix.** The same algebra, written in the innovation `d = eta2 - J2 b1` and
through the Cholesky factor `Y` (with `M = Y Y^T`, `V = J2 Y`, `w = Y^T d`):

    b1^T (eta2 - 1/2 J2 b1) + 1/2 |w|^2         (log-normaliser)
    b1 + Y w                                     (G b1 + M eta2)
    d - V w                                      (G^T d, into the next eta)

`elements.combine`, both Metal block folds and the backward seeds use it.
Both halves matter: the innovation form alone left `G^T d` with the same
1e-3 error, and the composite `eta` carried it into the next level, which
cost another log-unit on a uniform-random cadence.

**Measured on realistic cadences** (SHO + SHO, N = 4096, data drawn at the
parameters, worst of two draws; value error in log-units, before -> after;
`benchmarks/cadence_exposure.py`, 2026-10-24):

| setup | kappa | default MLX | fused kernel |
|---|---|---|---|
| TESS 2-min, rotation 3 d + granulation 0.2 d, s/e = 10 | 1e2 | 0.07 -> 0.006 | 0.003 -> 0.004 |
| same, s/e = 30 | 1e3 | 3.1 -> 0.015 | 0.20 -> 0.009 |
| same, s/e = 100 | 1e4 | 70 -> 0.04 | 18 -> 0.19 |
| Kepler 30-min, rotation 10 d + granulation 1 d, s/e = 100 | 3e3 | 3.7 -> 0.03 | 0.51 -> 0.006 |
| two instruments, simultaneous to 1 s, s/e = 30 | 1e3 | 2.0 -> 0.16 | 1.8 -> 0.04 |
| same, s/e = 100 | 1e4 | 160 -> 0.59 | 20 -> 0.10 |
| RV, 600 points, 1-3 per night, rotation 25 d, s/e = 3-30 | <= 1e3 | <= 0.08 -> <= 0.001 | <= 0.02 -> <= 0.001 |
| single SHO (m = 2), TESS 2-min or two instruments, s/e = 30-300 | <= 8e4 | unchanged, 0.01-1.7 | unchanged, 0.01-0.17 |

**Measured on stress cadences** (SHO + SHO, N = 4096; value error, then
worst relative gradient error against a float64 finite difference):

| cadence, s/e | before: default / pure tree / kernel | after |
|---|---|---|
| 20 near-duplicates (1e-4 cadences), 100 | -3.8, 53% / +18.3, 260% / -0.002, 0.08% | <= 0.007, 0.4% |
| uniform random, 100 | -2.9, 110% / +58.9, 270% / -1.3, 2.9% | <= 0.005, 0.13% |
| uniform random (another draw), 100; every `sub_block` 1-1024, dense and blocked leaves, m = 2, 4, 6 | up to 101 | <= 0.012 |
| near-duplicates or uniform, 1000 (value only) | up to 2.1e4 / 2.9e5 / 297 | 0.02-0.88 / 0.09-1.2 / 0.03-0.14 |

### What is left: the dense information form

What is left is older than this bug and independent of it. An element
stores its log-normaliser for a *zero* incoming state, so a leaf holds
`ell = ld - g^2/2` and a composite of k leaves holds roughly
`-1/2 sum g^2` over them -- ~1e4 at s/e = 30 when the timescale outlasts
the block -- and every general composition cancels that against a cross
term of the same size. The arithmetic is now right; the float32 *storage*
of those numbers is the floor. It grows with **both** how predictable one
sample is from the last (timescale over cadence, times s/e) and N, and
**kappa does not track it**: kappa saturates at `k(0)/r` once `hQh << r`,
and was only 1.5e3 in the worst case below. `kappa_max` cannot guard it.

**Exposure, measured where a sampler feels it** (`benchmarks/floor_exposure.py`,
2026-10-24). A constant offset is invisible to MCMC; what moves a posterior
is how the error varies over it. The float32 target is the true one times
`exp(e)`, so to first order no posterior mean moves by more than `sd(e)`
posterior standard deviations. Each row: data drawn on-model, 32 draws from
the float64 Laplace posterior, `sd(e)` across them (worst in brackets), and
the float32 gradient error per posterior sigma of motion. Rotation (3 d,
or 10 d for Kepler) plus granulation, SHO + SHO, unless noted:

| setup | N | kappa | default MLX: sd(e) (max), grad | fused kernel, fold (before 2026-10-24) | fused kernel, replay: sd(e) (max), grad |
|---|---|---|---|---|---|
| RV, 600 points, s/e 10-30 | 601 | <= 1e3 | 0.002 (0.005), 0.04 | 0.000 | 0.000, 0.04 |
| Kepler long cadence, one quarter, s/e 30 / 300 | 4.3k | 1e3-6e3 | 0.009 / 0.055 (0.16), <= 0.09 | 0.003 / 0.023 (0.08) | 0.000 / 0.005 (0.013), <= 0.11 |
| TESS 2-min sector, s/e 3 / 10 | 19k | 13 / 150 | 0.006 / 0.036 (0.13), 0.06 | 0.001 / 0.004 | 0.001 / 0.001, 0.06 |
| TESS 2-min sector, s/e 30 | 19k | 1.3e3 | 0.12 (0.24), 0.08 | 0.017 (0.06) | 0.003 (0.007), 0.08 |
| TESS 2-min sector, s/e 100 | 19k | 1.4e4 | **1.05 (2.7)**, 0.18 | 0.09 (0.39) | 0.008 (0.020), **0.48** |
| TESS 2-min, single SHO, s/e 30 / 100 / 300 | 19k | 1e3-1e5 | 0.13 / **0.78 / 3.5 (10.4)**, <= 0.98 | 0.007 / 0.024 / 0.09 | 0.001 / 0.001 / 0.002 (0.005), <= 0.02 |
| Kepler long cadence, 16 quarters, s/e 30 / 100 | 68k | 900-3.5e3 | 0.12 / **0.42 (1.1)**, -- | 0.012 / 0.031 | 0.004 / 0.005 (0.011), <= 0.08 |
| TESS 20-s sector, s/e 10 | 114k | 160 | **0.46 (1.0)**, -- | 0.14 (0.51) | 0.004 (0.010), 0.07 |
| TESS 20-s sector, s/e 30 | 114k | 1.5e3 | **2.8 (8.7)**, -- | **1.7 (5.9)** | 0.023 (0.061), **0.53** |

(`--`: the default MLX gradient exceeds Metal's buffer-count limit at this
N; see the README, "What is left".) Separating the two causes at s/e = 30
(default / old kernel / replay): 20-s cadence at a 2-min sector's N (19k)
gives 0.49 / 0.27 / 0.022, and 2-min cadence at a 20-s sector's N (114k)
gives 0.63 / 0.06 / 0.004.

So: transit fits detrended with a GP on quiet or mildly active stars
(s/e <~ 10), RV, and single Kepler quarters are clear on either path.
Bright active stars (s/e >~ 100), long baselines and TESS 20-s data are
not, on the default path. **`GPPolicy(metal=True)` clears every row above
in value, at m <= 4, since the replay (below) landed on 2026-10-24**; before
it the kernel too was 1.7 (5.9) on a 20-s sector at s/e = 30. An earlier
version of this section said the kernel stayed within 0.2 everywhere
measured; that was true only at N <= 19k and 2-min cadence. The kernel's
*gradient* still carries 0.48-0.53 posterior sigma at the two worst rows
(step 2 below).

**Except on short blocks.** Every row above runs at >= 1024-point blocks.
At N <~ 6k with m = 4, `_auto_metal_block` picks 256, and on densely
sampled, high-s/e data the prefix's entry states are then wrong enough to
reach the replayed value too (step 2). 20-s cadence, N = 4096, SHO + SHO,
s/e = 100, 128 draws from a converged ChEES posterior, against float64:
**0.081 (0.77)** at the default 16 x 256, 0.023 (0.13) at one block. The
pooled posterior still comes out right (measured, step 2), but sampling
it is ~10x slower per gradient. `tests/test_metal.py::
test_the_value_does_not_carry_the_dense_information_floor` probes the same
case within 0.01 of the mode, which is why it did not see this.

### The fix: replay the filter, do not compose ell

The prefix over the block elements already gives every block its entry
state `(b, C)` -- the filtered mean and covariance, O(1) numbers -- and the
kernel's backward already replays each block from it. Taking log L from
that replay instead of from the composed `ell` makes every term an O(1)
innovation, `-1/2 (log S + v^2/S)`, and the ~1e4 numbers never exist.
`benchmarks/block_replay_prototype.py` emulates it in NumPy float32 from the
kernel's own float32 block elements (value only; 16 posterior draws):

| case | default MLX | fused kernel | **replay** | one float32 filter, no blocks |
|---|---|---|---|---|
| TESS 20-s, SHO + SHO, s/e = 30 (N = 114k) | 1.6 (2.9) | 1.7 (5.9) | **0.017 (0.061)** | 0.40 (1.7) |
| TESS 2-min, SHO + SHO, s/e = 100 | 0.73 (1.3) | 0.067 (0.31) | **0.006 (0.019)** | 0.038 (0.092) |
| TESS 2-min, single SHO, s/e = 300 | 4.0 (12.8) | 0.13 (0.28) | **0.005 (0.010)** | 0.060 (0.14) |

sd(e) (max). The replay is 25-100x better than the kernel and leaves the
worst case at 0.06. The last column says the blocks matter for a second
reason: one float32 accumulator over 1e5 O(1) terms loses ~0.4 on its own,
and per-block partial sums are what avoid it.

The replay was chosen ahead of the square-root information form (`J = Z
Z^T`, `eta = Z gamma`, `ell` centred on `gamma`). The square-root form
would change every composition, in the MLX algebra, both Metal folds and
the hand VJPs, and add a QR to each. The replay changes how the value is
read out, reuses the prefix and the replay the backward already has, and
removes the large numbers instead of storing them more carefully.

1. **Kernel: done 2026-10-24.** `metal.scan_loglike` now runs pass 1, then
   `entry_states` (the seeds kernel's own prefix source, so the forward
   replays from exactly the states the backward replays from), then
   `replay_blocks`, which runs the backward's `_FWD_STEP` through each
   block from its entry state and Kahan-sums the innovations; the value is
   the sum of the per-block partials. Measured in the table above:
   **0.023 (0.061) on a 20-s sector at s/e = 30, against 1.7 (5.9)**, and
   <= 0.008 sd on every other row. Pinned by
   `tests/test_metal.py::test_the_value_does_not_carry_the_dense_information_floor`
   (0.45 -> 0.055). Pass 1 no longer forms its blocks' own `ell` on this
   path or the backward's (one precise `log` per observation saved);
   `reduce_blocks` keeps it for the tests that pin the element fold.

   **Cost**, interleaved A/B on a quiet machine against the fold, compiled,
   regular cadence: forward 1.50 -> 2.43 ms at N = 1e4 x 1024 chains, m = 2
   (1.6x); 10.9 -> 19.0 ms at N = 1e5 (1.74x); 11.7 -> 18.1 ms at N = 1e4,
   m = 4 (1.55x). Value + gradient: 6.57 -> 7.10 ms (1.08x), 51.6 -> 57.8
   ms (1.12x), and 53.9 -> 58.0 ms at m = 4, 256 chains (1.08x) -- the
   backward dominates, and it lost its `ell` work too. Forming `Phi C
   Phi^T` as two m^3 products instead of one m^4 sum was tried for the
   shared step and bought nothing (5% slower at m = 4).

   **What it gives up.** On well-conditioned long series the replay's own
   floor is the larger one: a covariance propagated recursively in float32
   puts a small systematic bias into each innovation variance, and it adds
   up with N, where pass 1 read `hQh` from a fixed table. A single SHO,
   `rho/dt = 100`, on-model: 0.0004 -> 0.0016 at N = 25k, s/e = 100, and
   0.004 -> 0.012 at N = 1e5, s/e = 30. Three orders under the decision
   scale, and on every row of the table above the replay is the better of
   the two by 10-100x.
2. **Gradient: diagnosed 2026-10-24, not fixed.** It does not bias a
   sampler -- HMC accepts or rejects on the value, which is now right, and
   leapfrog with any position-dependent force is still reversible and
   volume-preserving -- so a gradient error costs acceptance, not
   correctness. Gradient error per posterior sigma, against float64 central
   differences, SHO + SHO at s/e = 100 (the two amplitudes are the
   affected parameters; the timescales stay at <= 0.03-0.1):

   | case | default kernel | one block (`metal_block = N`) |
   |---|---|---|
   | 2-min cadence, N = 8192, 4 blocks of 2048 | 0.07 / 0.25 | 0.001 / 0.017 |
   | 20-s cadence, N = 4096, 16 blocks of 256 | **3.8 / 2.6** | 0.025 / 0.039 |

   Two boundary quantities fail, both because they compose
   dense-information block elements across blocks:

   * **the seeds** (`block_seeds_kernel`, from the suffix `J, eta`): the
     kernel's are 2-3e-4 relative wrong at 2048-point blocks, and replacing
     them by exact float64 seeds (central differences of a float64 filter
     from the kernel's own entry states) brings the first row to the
     one-block level. So do seeds from block elements *centred on the
     filtered trajectory* -- the same algebra with the innovations `v_t` as
     residuals and `-K_t v_t` added to each leaf's `b`, which makes
     `eta_s - J_s mu` the centred `eta_s` itself, with no cancellation
     (prototyped in MLX float32).
   * **the entry states** (the prefix's `b, C`): at 256-point blocks of
     20-s data the kernel's entry means are up to 7% wrong (0.12 against a
     state of 1.6), in directions the observations barely see -- the
     rotation / granulation split. Near the mode the value replayed from
     them is still right, but not across the posterior: 0.081 (0.77)
     against 0.023 (0.13) at one block (see "Except on short blocks"
     above).

   **Both quantities must be fixed, at any block length** (measured
   2026-10-24 on the second row's problem; "exact seeds" are the adjoints
   of a float64 filter's state at each block boundary, from one reverse-
   mode sweep, and "exact entries" that filter's states; worst over three
   posterior draws, the two amplitudes):

   | blocks | kernel as is | exact seeds only | exact entries + exact seeds |
   |---|---|---|---|
   | 16 x 256 | 3.8 / 2.6 | 5.4 / 4.4 | **0.05 / 0.11** |
   | 4 x 1024 | 10.7 / 9.8 | 0.83 / 0.64 | **0.013 / 0.037** |

   Short blocks are dominated by the entry states, long ones by the
   seeds, and with both right either block length comes within ~2x of one
   block (0.025 / 0.039): nothing inside a float32 block is the problem.
   An earlier version of this section gave 0.41 / 0.30 for "both exact"
   and called the remainder unexplained; those seeds were central
   differences, 3-7e-4 relative off -- as wrong as the kernel's own.

   **The centred elements, prototyped (2026-10-24).** Leaves centred on
   the float32 replay from the kernel's entry states (residual `y_t - h
   Phi m_{t-1}`, and `b' = A m_{t-1} + K y_t - m_t`), folded per block,
   then prefix (entry means = trajectory + correction, entry covariances)
   and suffix (seeds). Same problem, worst over three draws, the two
   amplitudes, with each stage of the composition optionally in float64:

   | elements | 16 x 256 | 4 x 1024 |
   |---|---|---|
   | plain, all float32 (the kernel's algebra) | 4.1 / 3.2 | 1.3 / 0.95 |
   | plain, float64 block prefix only | 4.1 / 3.1 | 1.4 / 0.98 |
   | plain, float32 leaves, float64 fold and prefix | 0.80 / 0.76 | 0.46 / 0.43 |
   | **centred, all float32** | **0.87 / 0.69** | **0.53 / 0.48** |
   | centred, float32 leaves, float64 fold and prefix | 0.20 / 0.32 | 0.12 / 0.20 |
   | all float64 (= exact entries and seeds) | 0.13 / 0.10 | 0.09 / 0.04 |

   What this says:

   * **Centring fixes the means, not the covariances.** It cuts the entry
     means' error 7-9% -> 0.4-1%, but `C` does not depend on the data, so
     its composition is untouched (entry covariances 4% off), and a wrong
     `C` limits the means through every `(I + C J)^-1`. Re-centring a
     second or third time changes nothing: the correction stalls at a
     fixed size instead of shrinking.
   * **The precision is lost inside the block fold, not in the prefix.**
     A float64 prefix over the block elements changes nothing; a float64
     fold of float32 leaves recovers most of it.
   * **Even float32 *leaves* are not enough at s/e = 100.** With them
     folded in float64, entry covariances are still 1% off: the dense
     information form is ill-conditioned enough to amplify the leaves'
     own rounding (the exact entry covariance has eigenvalues 1e-5 to
     0.9).
   * So centring alone buys 4x (short blocks) to 20x (long blocks) and
     leaves 0.5-0.9 sigma at the worst draws; reaching ~0.1-0.3 needs more
     precision in the fold as well (float-float arithmetic in the kernel's
     pass 1), or a better-conditioned form (square-root information
     form), not prototyped.

   Prototype pitfall, for whoever repeats this: `mx.array(x)` of a
   float64 ndarray is *float32* unless `dtype=` is passed. A first run of
   the table above silently did its "float64" stages in float32 and
   showed no effect from any of them.

   **The square-root information form, with centring: prototyped, and it
   works (2026-10-24).** Carry `C = U U^T` and `J = Z Z^T` as triangular
   factors (the square-root parallel filter of Yaghoobi, Corenflos, Hassan
   and Sarkka, 2022): every `(I + C_i J_j)^-1` comes from one QR ("tria")
   of `[[U_i^T Z_j, I], [Z_j, 0]]`, and the composed `U`, `Z` are rebuilt
   by QR, so both stay PSD and carry the square root of the dense form's
   condition number. Leaves use the standard square-root measurement
   update, `tria([[sqrt(r), h L_Q], [0, L_Q]])` with `L_Q L_Q^T = Q`.
   Dense and square-root algebra side by side in the same NumPy float32
   code (LAPACK `sgeqrf` / `sgesv`), the kernel's sequential in-block
   fold, same problem and draws as above; float64 runs of both reproduce
   the exact entry states (1e-10; 1e-8 with `L_Q` stored in float32):

   | elements, float32 | 16 x 256 | 4 x 1024 | entry means | entry covariances |
   |---|---|---|---|---|
   | dense, plain | 3.9 / 2.7 | 13.3 / 11.9 | 17-24% | 3-4% |
   | dense, centred | 1.4 / 1.4 | 4.4 / 3.8 | 2-5% | 3-4% |
   | square-root, plain | 6.0 / 4.9 | 4.6 / 4.4 | 13-15% | **8e-6** |
   | **square-root, centred** | **0.12 / 0.08** | **0.17 / 0.19** | **4-7e-4** | **8e-6** |
   | exact entry states and seeds | 0.37 / 0.05 | 0.33 / 0.08 | -- | -- |

   (These draws differ from the tables above: fixing `floor_exposure.hi`
   moved the Hessian behind `posterior_draws`. At these draws the float32
   backward itself, from exact boundary quantities, carries up to 0.37.)
   The square-root form fixes the covariances, ~5000x; centring fixes the
   means, where `eta_j - J_j b_i` still cancels; each alone fails.

   Against one block, at 256 draws from the converged ChEES posterior (the
   test that exposed the 11-18 sigma tail), median / 99th percentile /
   worst, the two amplitudes:

   | | 16 x 256 | 4 x 1024 |
   |---|---|---|
   | kernel as is | 0.14 / 5.4 / 6.1 | 1.05 / 30 / 54 |
   | **square-root, centred** | **0.024 / 0.62 / 1.05** | **0.022 / 0.32 / 0.55** |

   And the value replayed from the new entry states, at 128 of those draws
   against float64: sd 0.025 (worst 0.19) at 16 x 256 and 0.021 (0.14) at
   4 x 1024, against 0.081 (0.77) for the kernel and 0.023 (0.13) for one
   block -- the short-block value problem goes with it.

   **What a kernel can do.** The table centres the first leaf of each
   block on the previous block's trajectory end, which belongs to another
   thread. The *block-local* form centres each block on its own start
   state and moves the boundary mismatch into the previous block's
   *output* offset (`b` only; `eta`, `J` untouched), and is as good --
   0.12 / 0.10 and 0.16 / 0.18 -- **provided the start states are already
   approximately right**: from the existing kernel's entry states, or the
   uncentred square-root ones. From zero it fails (5.9 sigma) until a
   second round. A second round from a good start buys nothing. Factoring
   `Q` by float32 Cholesky instead of a float64 eigendecomposition changes
   nothing here (0.05 / 0.08, 0.19 / 0.21; `Q`'s eigenvalues at 20 s run
   down to 1e-9), though denser cadences will need the relative floor.

   So the kernel pipeline becomes: today's pass 1 and prefix (approximate
   entry states) -> one centred square-root pass, the replay fused with
   the fold, block-local -> square-root prefix and suffix over the block
   elements, with the output shifts -> entry states and seeds -> the value
   replay and the backward, unchanged. One extra sweep of the data plus a
   heavier fold; cost not measured. Not yet confirmed by a ChEES run,
   which needs the kernel (the prototype is far too slow to sample with).
   Reference implementation: `benchmarks/boundary_prototype.py`.
   **Built 2026-10-25 to `docs/boundary-plan.md` and the kernel's default
   since** (`GPPolicy.metal_boundary`; convolved terms keep the dense
   form): at 16 posterior draws of the problem above the worst gradient
   error goes 4.0 -> 0.31 sigma (16 x 256) and 5.0 -> 0.34 (4 x 1024),
   and the value's spread across them sd 0.066 -> 0.020 and 0.038 ->
   0.025. ChEES confirmation and cost: stage 5 of the plan. Oracle and
   tests: `tests/boundary_oracle.py`, `tests/test_boundary.py`.

   Ruled out, measured: the term front end (chaining the kernel's per-lag
   adjoints to theta in float64 changes nothing), and the per-lag and `dh`
   accumulators (Kahan compensation changes nothing). Refining the seeds
   by a second backward pass -- each block's entry adjoint is exactly the
   seed the previous block should have had -- fixes the first row
   (0.25 -> 0.017) and takes a TESS 20-s sector at s/e = 30 from 0.49 to
   0.17 (its converged floor is 0.155), but makes the second row worse
   (3.8 -> 5.9): it moves information one block per pass, and a 256-point
   block of 20-s data is far shorter than the ~860 (granulation) to ~13,000
   (rotation) steps the filter needs to forget an error. A real fix needs
   both boundary quantities in a well-conditioned form -- the centred
   elements, or the square-root information form, for the prefix and the
   suffix -- and is open. Scripts: the scratch prototypes of 2026-10-24
   (`exact_seeds.py`, `refine_exp.py`, `centred_proto.py`) are not in the
   repository.

   **What it costs a sampler, measured 2026-10-24.** The error is heavy-
   tailed rather than uniformly large: at 256 converged posterior draws of
   the second row's problem (`make_target` defaults, jitter included) the
   median gradient error is 0.13-0.18 sigma in the two amplitudes, but the
   worst points are 11-18 sigma off. ChEES adapts its step size to the
   harmonic mean of the acceptance, so those few points drag it down, the
   trajectory-length criterion follows, and the slow directions (jitter)
   stop mixing. ChEES, 1024 chains, `max_leapfrog = 128`, two seeds each:

   | blocks | warmup | step size | leapfrog / iter | max R-hat | ESS per 1000 gradients | wall |
   |---|---|---|---|---|---|---|
   | one (`metal_block = 4096`) | 300 | 0.0225-0.0228 | 35-38 | **1.11** | **0.81-0.86** | 32-34 min |
   | default, 16 x 256 | 300 | 0.0078-0.0093 | 12 | 2.8-3.3 | -- | 3 min |
   | default, 16 x 256 | 2000 | **0.0042-0.0043** | 33-41 | **1.57-1.74** | **0.084-0.093** | 52-55 min |
   | 4 x 1024 | 300 | 0.0019-0.0063 | 6-11 | **6.8-11** | -- | 4-6 min |

   Longer blocks are *not* a mitigation: at 4 x 1024 the value improves
   (0.035 (0.20) across the posterior, against 0.081 (0.77) at 16 x 256)
   but the gradient is worse (median 0.47-0.98 sigma, worst 36-43 sigma in
   the amplitudes), because the seeds' cancellation grows with the data in
   each block.

   The step size kept shrinking through warmup as chains reached more of
   the posterior. The pooled default-block posterior agrees with the
   one-block one to <= 0.06 sigma in every mean, as it must, but ~10x
   fewer effective samples per gradient outweighs its 4.5x cheaper
   gradient (44 against 196 ms at 1024 chains): one block converged in
   ~60% of the wall time the default blocks spent *not* converging.
   **Fixed 2026-10-25** by the square-root boundary, now the default:
   ChEES at the default 16 x 256 blocks, same settings, step size
   0.0208-0.0225, 27-36 leapfrog / iter, max R-hat 1.12-1.19 (one block's
   own: 1.11), **0.68-0.81 ESS per 1000 gradients**, 9-12 min, pooled
   posterior within 0.022 sigma of one block's (docs/boundary-plan.md,
   stage 5). Before the fix the advice was `metal_block` = N for short,
   densely sampled, high-s/e series; at 2-s cadence and s/e 100 it still
   is (see the plan). Machine
   not quiet during these runs; the per-gradient columns do not depend on
   it, the wall column does. Script: `benchmarks/chees_block_cost.py`.
3. **Default MLX path** (all m, and the only path for m > 4). The same
   structure at sub-block scale: fold leaves into sub-block elements, a
   prefix over them instead of the reduction, then a 32-step replay batched
   over sub-blocks. Not yet attempted.

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
