# Plan: per-instrument jitter, exposure-time integration, and the compile split

Status: **locked 2026-09-27** after an adversarial review that revised it
— see "Stage R" at the end. **Stage 1 is done** (commit below); stages 2
and 3 are as written. The review overturned the
plan's central design claim and inverted its ordering; the stages below
are the revised ones, and the original reasoning is kept where it was
wrong so the correction is legible.

## What was measured first, and what it did and did not show

**Exposure-time integration keeps the rates and rescales the amplitudes.**
celerite2's `TermConvolution` does exactly two things: it rescales the
celerite amplitudes and adds a constant to the diagonal. The rates are
untouched — verified identical to machine precision over eight terms and
six exposure widths, including a real term, two sums with different
rates, and overdamped SHOs — and for lags at or beyond the exposure width
the convolved kernel is exactly a sum of exponentials with the modified
amplitudes (agreement 3e-16, and by construction: celerite2's own
`get_value` *is* that sum in that branch).

**The rescaling breaks the SHO's amplitude ratio.** anvil-gp's SHO has
`b/a = c/d` exactly; at `delta = 1.5` against `rho = 4` the convolution
moves `b/a` from 0.169 to 0.348. Fitting a free two-vector `h` over the
existing SHO basis leaves a 6% residual.

**The original inference from that was wrong.** The plan concluded that a
general stationary covariance `P_inf` was needed. Two facts, both proved
and measured in the review, say otherwise:

1. *No stationary model with positive process noise can realise the
   convolved kernel, whatever `P_inf` is.* For any such model the
   kernel's initial slope is `k'(0+) = -½ h Q_c hᵀ ≤ 0` — the other term,
   `½ h (FP - PFᵀ) hᵀ`, vanishes identically because `FP - PFᵀ` is
   antisymmetric. That is `b/a ≤ c/d`, celerite's own positive-definiteness
   condition. The un-convolved SHO sits exactly on that boundary and the
   convolution pushes it outside for every exposure width (by +1.8e-2 at
   `delta = 0.5`). A brute search over general `(F, P_inf, h)` with
   positive noise enforced bottoms out at 2.1% residual; drop the
   constraint and it reaches 6e-15. So generalising `P_inf` buys nothing.
2. *Every positive-definite complex term is already reachable with
   `P_inf = I`*, by a three-parameter generalisation of the SHO basis:
   `beta = b d / a`, `g = c - beta`, `s = √(beta² + d²)`, `h = (√a, 0)`,
   `F = [[-g, s], [-s, -(2c - g)]]`, `Q_c = diag(2g, 2(2c - g))`. Verified
   against `celerite2.terms.ComplexTerm` on 200 random terms to 1.3e-11
   (via `expm`; 3e-14 with the analytic propagator), including `d → 0`,
   `b = 0`, and `b` on the boundary. The existing `_sho_matrices` is the
   special case `beta = c, s = omega_0`, reproduced to 0.0 and 2e-16.
3. *And the filter does not need positive noise.* It is linear algebra on
   the observed covariance, which is positive-definite whenever every
   spacing is at least the exposure width. The same construction on the
   convolved amplitudes — with `Q_c` formally indefinite — reproduces
   celerite2's `TermConvolution` log-likelihood to 1e-14 … 4e-13 across
   SHO, real, and real-plus-SHO terms at three exposure widths.

So the general complex term, `OriginalCeleriteTerm` and exposure time are
one piece of front-end work, and it is *smaller* than the plan first
said: a third free rate in `_sho_matrices`, not a new stationary
covariance threaded through the package. The kernel is untouched. The
risk moves to one new special function, named under stage 3a.

**The compile failure has a one-function cause.** The stated fix had been
"split the VJP at the term-matrix table". Forcing the existing
`custom_function` wrapper on unconditionally fails identically to the
baseline, same fused kernel. Bisecting instead: the offender is
`special.sincm1`, whose `direct` branch (a three-way nested `where` inside
`sinc_gen`, each arm sanitised) exhausts the 24-buffer limit on its own.
At both `_sho_matrices` call sites the argument is bounded by 2, the
series threshold, so **that branch is provably dead there**: the
series-only form is bit-identical over 20,005 points in float32 and
float64, and with it in place `mx.compile(mx.value_and_grad(...))`
succeeds for every term and both likelihood paths, value bit-identical,
gradient within 8e-7 relative (float32 reassociation). The full suite
passes under the patch except the one test that asserts compilation
fails. This is not an investigation; it is a commit.

**The failure is a catchable `RuntimeError`**, not the "uncatchable
process abort" that `docs/float32.md` and the M4 stage R notes describe.
Twenty probes, process survived every time. `mx.eval` inside a compiled
function is rejected outright, so placement cannot be the difference.

## Stage 1 — the compile fix — **DONE 2026-09-27**

Use the series-only `sincm1` at the two `_sho_matrices` call sites, where
the other branch cannot be taken. `special.sincm1` itself and its mpmath
pinning stay as they are.

| gate | |
|---|---|
| bit-identity | the term matrices are unchanged to the bit in float32 and float64 (already measured; pin it) |
| the point | `mx.compile(mx.value_and_grad(gp))` succeeds on both paths, for every term and a `TermSum` |
| the suite | green, with `test_compiled_value_and_grad_exhausts_metal_argument_buffers` inverted to assert success |
| the docs | `float32.md` and the README's "Known limitation" rewritten; the "process abort" claim corrected; `mx.disable_compile()` removed from the ChEES examples and `tests/test_calibration.py` |
| upstream | file the MLX reproducer: `sincm1` alone at 1 chain x 2 lags is a four-line case |

**Landed**, as `special.sincm1_series` used at the two `_sho_matrices`
call sites. Every gate met: zero differing bits against `sincm1` across the
whole threshold interval in both precisions; `mx.compile(value_and_grad)`
succeeds for all six terms plus a `TermSum` on both paths (14 of 14), value
bit-identical, gradient within 1.3e-5; the gotchas test inverted; every
`mx.disable_compile()` removed from the tree; a **verified** self-contained
MLX reproducer written to `docs/upstream/`.

**One correction the reproducer forced.** The review said "`sincm1` alone at
1 chain x 2 lags is a four-line repro". It is not: `sincm1` alone compiles,
and so does `sinc_gen` alone. Only `sincm1` *inside* `_sho_matrices`'
surrounding expression fails, so the fused budget is spent collectively and
no single function is the offender — `sincm1` is merely the contributor that
can be removed. My first attempt at an upstream report asserted the
reviewer's version and did not reproduce; the shipped one does, and the
claim is corrected in `README.md`, `docs/float32.md` and the
`sincm1_series` docstring.

**The prize is smaller than the review claimed, and size-dependent.** The
review measured 1.95x at 64 chains x N = 4096 and generalised from it. At
three sizes, interleaved:

| | gradient eager | compiled | compile gain | dead-branch gain alone |
|---|---:|---:|---:|---:|
| N = 4,096, 64 chains | 3.24 ms | 1.93 ms | 1.68x | 1.23x |
| N = 1e4, 1024 chains | 8.26 ms | 7.06 ms | 1.17x | 1.09x |
| N = 1e5, 1024 chains | 57.6 ms | 56.3 ms | 1.02x | 1.01x |

So at production sizes it is 1.02-1.17x — which is what the *original*
plan estimated (1.15-1.25x) and the review talked me out of. The reason to
do it was always the workaround, not the clock; the review was right about
the effort and wrong about the payoff, in the opposite direction from the
plan's first error.

**Still not verified:** whether anvil's ChEES leapfrog compiles end to end.
The likelihood's `value_and_grad` does, and `tests/test_calibration.py` now
runs ChEES with compilation on, which is the closest available evidence.

## Stage 2 — per-instrument jitter in the kernel — **DONE 2026-09-27**

The shape is right; three premises in the original were wrong and change
the work.

* **There is no per-instrument accessor.** `ParamMap.jitter_var(v, inst)`
  returns per-*datum* variances, `(n_chains, N)`, via `mx.take`, and with
  `inst=None` it raises. The kernel needs `(n_chains, n_jit)`. Stage 2
  writes that accessor.
* **A wrong instrument index does not necessarily change the value.**
  `priors.jitter_hyper` gives every instrument the same reference, so at
  `v = 0` — anvil's starting point — all jitters are numerically equal and
  permuting the labels leaves the log-likelihood **bit-identical** while
  the jitter gradients swap (measured: 0.0 on the value, 7.65 on the
  gradient). The failure is silent exactly where a sampler starts. The
  permutation gate must use *distinct* jitter values and assert on the
  gradient.
* **`param_map.instrument` is validated nowhere.** Label 7 with 3 jitters
  goes through `mx.take` silently today; in the kernel it would read the
  next chain's row. Stage 2 adds a host range check.

What changes: the forward kernel's once-per-thread jitter read
(`metal.py:114`) becomes a per-observation read indexed by instrument,
at that site and at the two spliced copies of `_FWD_STEP` and the
backward sweep. The backward accumulates `float drv_acc[MAX_JIT]`;
`out_drv` becomes `(n_chains, n_blocks, MAX_JIT)`; the VJP slices
`[:, :n_jit]`. **Zero all `MAX_JIT` slots**, not only the ones a block
uses: `mx.fast.metal_kernel` outputs are not zero-initialised. Instruments
absent from a block are fine (slot stays 0); absent from the whole
dataset give an exactly-zero gradient and should warn.

**Design it as an additive per-(chain, instrument) diagonal term, not as
"the jitter".** Stage 3c needs to add a parameter-dependent constant to
the diagonal, per chain and (with per-instrument exposure times) per
instrument. That is this slot. Build it once.

| gate | |
|---|---|
| correctness | kernel vs `GPPolicy(metal=False)` to 1e-5 and vs float64 central differences to 1e-6, with 1, 2 and 4 instruments, on every cadence in `_fwd_setup` — the MLX reference is trustworthy (its per-instrument jitter adjoints already match float64 to 1.5e-7) |
| the trap | distinct jitter values, labels permuted: the *gradient* test must fail |
| no regression | single-jitter timings unchanged interleaved; the one-jitter path keeps the once-per-thread read behind a compile-time switch (doubles the kernel cache keys) |

Half a day for the code is credible. The gates are a day: 5 cadences x 3
instrument counts x value, gradient, and central differences.

### Landed

`metal.MAX_JITTER = 8`, a register budget like `MAX_STATE`. The kernel takes
an `inst` input and is specialised per jitter count: at most one jitter keeps
the hoisted once-per-thread read it always had, several move the read inside
the loop indexed by the observation's instrument. The backward holds
`float drv_acc[NJIT]`, zeroes **all** slots, and `out_drv` gains a jitter
axis that the VJP slices.

Host side, all three of the review's corrections applied:
`ParamMap.jitter_var_per_instrument` is the new compact accessor (the
existing `jitter_var` is per-*datum* and cannot serve the kernel);
instrument labels are range- and shape-checked at construction; and
`metal_eligible` refuses above `MAX_JITTER` and when several jitters come
without labels.

**Gates, measured.** Against the reference path: worst 6.2e-7 relative over
{1, 2, 3, 4, 8} jitters x 5 cadences, 31 cases. Against float64 central
differences: 5.9e-6 at one jitter, 1.8e-5 at four. The single-jitter path is
unchanged (1.04x forward and 1.07x gradient for *four* jitters, so the
feature costs ~5% only when used).

**One gate as written was unachievable, and the reason is worth keeping.**
It asked for 1e-6 against central differences. The 1.8e-5 at four jitters
sits entirely on the smallest gradient component (−0.0246 against a largest
of −41.7, a factor of 1700), where the absolute error is 4e-7 — float32
epsilon on a quantity of order one. Varying the difference step from 1e-3
to 1e-6 moves it hardly at all, so it is not truncation. The MLX reference
path has the same 1.1e-5 there and is *worse* than the kernel on `sigma`
(1.6e-4 against 1.2e-5). So a relative gate on a near-zero component cannot
be met in float32 by any implementation; the binding evidence is the
6.2e-7 agreement with the path the kernel is verified against.

**Two bugs this stage's own gates caught**, both invisible to a value test:

* The kernel cache key collapsed every jitter count above one into a single
  variant, but `NJIT` sizes the backward's accumulator and strides its
  output — so a three-jitter problem reused the two-jitter kernel, wrote
  `drv_acc[2]` out of bounds and strided `out_drv` wrongly. Value exact,
  jitter gradients wrong by 0.44 relative. Each count above one now gets
  its own kernel.
* The availability probe did not pass the new `inst` input, so
  `metal_available()` returned False and *every* problem silently fell back
  to the MLX path — with a warning, but the tests would have measured the
  wrong path.

**One assertion I wrote was physically wrong.** The permutation test first
asserted that an instrument left alone by the label swap keeps its
gradient. It does not, and should not: the jitters enter one shared filter,
so changing which observations carry which variance changes every later
observation's contribution. The untouched instrument moved 0.2%. The test
now asserts what actually discriminates — agreement with the reference path
under *both* labelings, plus the equal-jitter case where the value is
bit-identical and only the gradient can reveal a wrong index.

**A fourth mistake, and a process one.** I reported the gates met while 102
tests were failing, because I had only run the jitter subset. Changing
`scan_blocks`/`backward_ckpt`/`scan_loglike`'s signatures broke every direct
caller in the suite, and the `drv` output's new jitter axis silently
broadcast against the reference in `_e2e_errors` (a `(n_chains, 1)` against
a `(n_chains,)`, giving a 1.5 relative "error"). **Run the whole suite
before claiming a gate, not the tests for the thing just built.**

**Left for stage 3c**, as planned: the additive per-(chain, instrument)
diagonal slot this builds is what the exposure-time diagonal correction
will use. It is in place; 3c has to route a parameter-dependent constant
through it rather than build one.

626 tests pass; the README verifier passes all ten rows at a 0.9% probe.

## Stage 3 — the general complex term, and exposure time on top of it (was stage 2)

### 3a. Generalise `_sho_matrices` from two rates to three — **DONE**

`(omega_0, Q)` becomes `(c, beta, s)` with `P_inf = I` kept. `Phi` and `Q`
keep their algebraic shape exactly (verified against `expm` and
`I - Phi Phiᵀ` to 2e-15):

```
Q11 = (1 - E) - 2 b_ E C S - 2 b_² E S²
Q22 = (1 - E) + 2 b_ E C S - 2 b_² E S²
Q12 = 2 b_ s_ E S²          with b_ = beta dt, s_ = s dt, E = e^{-2c dt}
```

**The one new numeric object was expected to be the risk, and is not.**
It looked like it needed its own series. It does not: there is an exact
identity,

    1 - e^{-z}(1 + y + y²/2)
      = gamma3(z) + e^{-z} (z - y) (1 + (z + y)/2)
      = gamma3(z) + e^{-z} delta (1 + z - delta/2),   delta = z - y

so it is `gamma3` — already cancellation-free — plus a correction that is a
*product*, not a difference. And `delta = 2(c - beta) dt = 2 g dt` is
**supplied by the parametrisation**, so the one subtraction that could
cancel never happens. `special.gamma3_shifted` is four lines with no
branch. Measured in float32: worst 1.5e-7 relative over the `(z, delta)`
corners including `z = 1e-6`, where the naive form is wrong by a factor of
3.6e11; and at `delta = 0` it is **bit-identical** to `gamma3`, so no
existing term moves.

**Gate met.** `SHOTerm`, `Matern32Term` and `RotationTerm` reproduced
**bit-identically** through the general path — 0 differing bits over
19,584 values, both precisions, `Q` from 0.05 to 300 and `dt` over eight
decades. `gamma3_shifted` pinned against mpmath and against the naive
form. Suite green with no tolerance change. As predicted, `first_point()`,
the pivot floor, the `.hi()` path and `sample_ssm` needed no change,
because `P_inf = I` is kept.

`_complex_matrices` takes `dsq`, `wsq` and `g` rather than deriving them,
so each caller keeps the cancellation-free route its own parametrisation
affords: the SHO still gets `d²` through the factored
`(2Q-1)(2Q+1)/(4Q²)` that vanishes *exactly* at `Q = 1/2`.

### 3b. `ComplexTerm` — **DONE**; `OriginalCeleriteTerm` folded in

**Gate met, tighter than asked.** `ag.ComplexTerm(a, b, c, d)` matches
`celerite2.terms.ComplexTerm` to **0.0** on kernel values over 150 random
terms and to 2.5e-16 through the state-space form, over the corners
`b = 0`, `d → 0`, `c >> d`, `c << d`, `b < 0` and `b` just inside the
positive-definiteness boundary. End to end against celerite2 *and* the
dense float64 oracle: 1.9e-15. On the fused kernel: value within 1e-3 of
the reference path, gradient 3.3e-7 against it and 1.2e-7 against float64
central differences.

`SHOTerm` is pinned as the `b/a = c/d` case of it, which is what makes
3a's refactor safe.

**`OriginalCeleriteTerm` needs no separate class.** In celerite2 it is a
*wrapper* that just forwards another term's coefficients
(`get_coefficients` returns `self.term.get_all_coefficients()`), not a
parametrisation of its own — so `ComplexTerm` already is it. The plan
listed it as a third gap; it was one gap wearing two names.

### 3c. `TermConvolution` — **DONE 2026-09-27**

The same construction on the rescaled amplitudes, with `Q_c` formally
indefinite. Four things the original plan got wrong or left out:

* **The diagonal correction is a function of the sampled parameters.**
  Measured: 0.3% of `k(0)` at `delta/rho = 0.125`, 2.3% at 0.25, 8.5% at
  0.375, and it scales with `sigma²` and varies with `rho` and `Q`. It
  cannot go into `r_ref`, which is per-observation data with no gradient.
  It goes through stage 2's additive per-(chain, instrument) diagonal
  slot, and MLX's chain rule on the host carries its gradient. A user
  with no jitter parameter still needs the slot.
* **Refuse any spacing below the exposure width, including zero.** The
  original rule said "smallest *non-zero* spacing", and zero is the case
  that fails: at `tau = 0` the off-diagonal is the exponential sum's
  value, not the convolved kernel's. For a term with a real component the
  observed matrix goes indefinite (min eigenvalue -3.0e-2 at
  `delta = 0.1`, needing `yerr` of 14% of `√k(0)` to rescue) and the
  filter's innovation variance goes negative — a NaN. For SHO-only terms
  it stays positive but wrong by up to 6e-2 log-units. anvil-gp
  *supports* duplicate timestamps (`linalg.py:20`, the `dupes` and
  `alternating` fixtures), so this is reachable. Two instruments observing
  simultaneously with different exposure times is this case, not a
  separate one. Near-duplicates decay smoothly (2e-5 of `k(0)` at
  `0.01 delta`), so the rule needs no tolerance.
* **Matern-5/2 has no celerite coefficients**, so "take the term's
  coefficients" cannot cover it — and it is the one kernel anvil-gp has
  that celerite2 lacks. Its exposure-convolved kernel is a quadratic times
  the *same* exponential (fitted to 1.6e-15 with the rate fixed; a free
  rate returns the base rate to six figures), so it is a three-state model
  with the base propagator. Implement the convolution at the state-space
  level — same `Phi`, modified amplitude structure — for every term, not
  via coefficients.
* **The clamp comments become false.** `metal.py:150` and
  `elements.leaf_elements` say `h Q hᵀ` is "non-negative in exact
  arithmetic". For a convolved term it goes negative below `dt ≈ delta/2`
  (measured sign flip at `0.50-0.72 delta` across `Q` and `delta`). The
  spacing rule gives ~2x margin, so no failure could be constructed; but
  the comment must say so, and any spacing under `delta/2` would turn a
  clamp documented as unable to bind into a silent value change.

Gates: float64 parity with `celerite2.terms.TermConvolution` to ~1e-11 on
cadences with every spacing ≥ `delta`, over SHO (both damping regimes),
real, Matern-3/2 and sums; `delta → 0` recovers the base term to 1e-12;
`dupes` and `alternating` must **raise**, naming the offending spacing;
the float32 kernel path against the float64 oracle at the usual
tolerances; a gradient check through the diagonal correction.

Two reference caveats. At `delta/rho = 1e-3` celerite2's own two float64
routes disagree with an mpmath quadrature of `k_conv(0)` by up to 5.7e-12,
so a 1e-11 gate there has under 2x margin and is mostly testing celerite2;
use the quadrature `(2/δ²) ∫₀^δ (δ - u) k(u) du` as the reference at the
small-`delta` end. And for a convolved Matern-3/2, celerite2 is
approximate in the base term, so the reference is anvil-gp's own exact
convolved kernel, not celerite2.

**The oracle.** `sample_ssm` draws `x₀ ~ N(0, I)` and clips negative
eigenvalues of `Q` when it factors it, so it cannot sample a convolved
process. For 3c the on-model sampler is the dense one: build the exact
piecewise convolved kernel in `value_np`, Cholesky it. `dense_covariance`
already does this given a correct `value_np`.

Two to three days, as before, but with the risk relocated: from `P_inf`
plumbing (gone) to one new two-argument special function (3a) and the
zero-spacing refusal (3c).

#### Landed

Exposure-time integration on the coefficient route, `delta` fixed data and
the wrapped term's parameters passed through untouched. The state dimension
is unchanged, so a convolved `RotationTerm` is still `m = 4` and still runs
on the fused kernel. The zero-lag correction goes through stage 2's additive
diagonal slot on both paths, exactly as that stage was designed for.

**Gates, measured.** Against `celerite2.terms.TermConvolution` in float64:
worse than 1e-11 nowhere over five term shapes (SHO, real, complex,
SHO+real, rotation) x {1e-2, 0.04} x {at, off} the reference point. Fused
kernel against the reference path: value 1e-5, gradient 1e-3. Gradient
against float64 central differences: 1e-4 over `delta` in {1e-4, 1e-3,
1e-2}. `delta -> 0` recovers the base term to 1e-12 at every lag `>= delta`.
The `dupes` and `alternating` cadences raise, naming the offending pair, and
the same cadences still construct without the convolution. Full suite 724
passed, up from 644.

**The stage shipped with a wrong gradient and a right value, and the value
check was what hid it.** The convolved amplitudes are
`2a(cosh z - 1)/z^2` and the correction `-2a(sinh z - z)/z^2` at
`z = c delta`; written that way both cancel to nothing as `delta -> 0`, and
dividing by `z^2` afterwards cannot restore digits the subtraction already
destroyed. In float32 at `delta = 1e-2` that left ~2 digits: the *value*
matched celerite2 to 1e-13 in float64 and the gradient was **9% wrong**, at
`delta = 1e-3` it was wrong by 247x, and at `1e-4` the analytic gradient was
exactly **zero** while finite differences were not. Both the autodiff path
and the hand-written Metal backward were wrong *identically*, which is what
localised it: a forward whose parameter dependence is destroyed, not a
backward bug.

Fixed analytically rather than by tolerance. `coshm1_sinhm_over_z2` and
`coshm1_sinhm_complex` (new, in `special.py`) return the combinations
already divided, from series in `w^2` with all-positive coefficients. The
complex correction's `O(delta)` part cancels against its own
`(ac + bd) delta (c^2 + d^2)` term by the exact identity
`C2 d - C1 c = -(c^2+d^2)(ac+bd)` (verified in rational arithmetic), so
subtracting `w` inside `sinh w - w` does that cancellation in algebra
instead of in float32. Both helpers hold machine precision in both dtypes
against mpmath across the series/closed-form switch. Gradient error after:
~1e-5, flat in `delta` from 1e-8 to 1e-2.

**This makes anvil-gp the more accurate of the two at small `delta`**, by
six orders of magnitude: on a real term at `delta = 1e-3`, against a
50-digit reference, anvil-gp is 2.2e-16 and celerite2 2.2e-10 — celerite2
evaluates the same numerator as written. That is why the parity gate stops
at `delta = 1e-2` and an mpmath quadrature of
`(2/d^2) int_0^d (d - u) k(u) du` is the reference below it, exactly as the
plan's reference caveat anticipated — though the plan expected the margin to
be thin, not to invert.

**Three things the plan did not have.**

* **`RotationTerm` had no `celerite_coefficients`, so the motivating case
  was refused.** Long-cadence photometry of a rotating star is *the* reason
  to want this feature, and it raised `TypeError`. Added: two complex pairs,
  matching celerite2 to 1e-14. Both modes have `Q >= 1/2 + Q0 > 1/2`, so
  unlike a bare `SHOTerm` it cannot reach the critical-damping division at
  all.
* **The overdamped refusal did not exist.** `SHOTerm.celerite_coefficients`
  documented a call-time check for `Q <= 1/2`; there was none, and `d` is
  floored at zero, so `b = a c / d` is non-finite. What a sampler got was
  not a NaN but the likelihood's non-finite guard: `-1e30` with an exactly
  **zero gradient** — a flat plateau, the failure mode that guard's own
  comment warns about. Now refused in `GPLogLike.__init__` at the reference
  point, where the message can name the cause; the docs state that the prior
  must keep `Q > 1/2`, because a sampler can still walk there.
* **The correction's sign is two orders, not two cases.** The class
  documented it as positive for oscillatory and negative for real
  components. Measured: `corr = (delta/3) k'(0+)` to six digits wherever the
  even extension has a corner, and `k'(0+) = -1/2 h Q_c h^T <= 0` always, so
  there it is negative. A term with `b/a = c/d` (SHO, rotation) has no
  corner, the first-order term vanishes, and the remainder is positive and
  `O(delta^3)`. Both signs occur, at different orders.

**One gate as written was unachievable, for a reason worth keeping.** It
asked that `delta -> 0` recover the base term to 1e-12 — at *every* lag.
At zero lag it cannot: the convolved kernel differs from the base at
**first order** in `delta` there, by `(delta/3) k'(0+)`, which is physics
and not error. The gate now runs over lags `>= delta`, which is not a dodge
but the supported domain, since `GPLogLike` refuses any shorter spacing; the
zero-lag behaviour has its own test against the corner law.

**Also done:** the clamp comments at `metal.py` and `elements.leaf_elements`
now say that `h Q h^T >= 0` holds for a convolved term only because
spacings below `delta` are refused — it genuinely goes negative below
`delta/2`. `sample_ssm` documents that it cannot draw a convolved process.
`docs/kernels.md` gains `ComplexTerm` and `TermConvolution` sections and
loses three stale "planned" rows; `docs/tutorial.md` gains a long-cadence
worked example with every figure pinned in `test_docs.py`.

**Still not done:** `TermConvolution` around `Matern32Term` or
`Matern52Term`, which raise. A repeated root has no celerite coefficients,
so the closed form does not apply; their convolutions are state-space models
of the same order but need the `m = 3` propagator generalised the way 3a
generalised `m = 2`. `OriginalCeleriteTerm` is still absent, now only as a
coefficient conversion away from the implemented `ComplexTerm`.

#### Stage 3d --- the convolution moved to the state-space level, 2026-09-27

The plan's own 3c bullet asked for this and the implementation overruled it:
"Implement the convolution at the state-space level -- same `Phi`, modified
amplitude structure -- for every term, not via coefficients." The
coefficient route shipped first and cost exactly what the bullet predicted:
`Matern32Term`, `Matern52Term` and an overdamped `SHOTerm` were all refused,
and those are *one* limitation, not three -- the celerite coefficients
`(a, b, c, d)` do not exist for a repeated root, a triple root, or a
decomposition with a negative amplitude.

**The derivation.** For lags `tau >= delta`, with `Phi(tau) = expm(F tau)`,

    k_conv(tau) = (1/d^2) int int h expm(F(tau + s - s')) h^T ds ds'
                = h A Phi(tau) B h^T = h Phi(tau) M h^T,

since `A`, `B` and `M = A B` are all functions of `F` and commute with
`Phi`.  `M = f(F)` for the *even* scalar `f(z) = 2(cosh z delta - 1)/(z
delta)^2` -- exactly the scalar factor the coefficient route applied per
exponential, now as a matrix function.  The true zero-lag variance is
`W = 2(e^X - I - X) X^-2`, `X = F delta`, and the diagonal correction is
`h (W - M) h^T`.  Both reduce to the coefficient route's scalar formulas
exactly, which is how the sign error in `W - M` was caught: it must be
`2(sinh X - X)/X^2`, and at `X = -z` that is `2(z - sinh z)/z^2 < 0`.

Being *entire* functions of `F`, both are indifferent to root multiplicity
and never form an amplitude -- so all three exclusions lift at once.  `Phi`
is untouched, so `m` is unchanged: a convolved Matern-3/2 stays at `m = 2`,
correcting this plan's guess that it needed `m = 3` (five lags match a
two-coefficient form to 1.6e-15; the earlier "quadratic times exponential"
came from an over-parameterised fit whose quadratic coefficient was ~0).

**`P_inf` is no longer the identity, and the fix is a rank-one one.** `M` is
*not* symmetric (3e-4 to 1e-2 relative) and no change of basis makes it so,
since it transforms as `T M T^T`. But only `P h^T` is observable, so any
symmetric `P` with `P h^T = M h^T` gives the same kernel:

    v = M h^T,  s = h M h^T,  P = v v^T / s + (s / h h^T)(I - h^T h / h h^T).

No linear solve. The projector term is annihilated by `h^T`, so it is free --
and it is *not* optional: `v v^T / s` alone is **rank one**, the element
algebra Choleskys exactly this matrix, and a floored singular pivot made the
likelihood 1e-3 wrong while the sequential filter, which never factors it,
stayed exact. That asymmetry between the two recursions is what identified
it. The weight `s / h h^T` makes the construction equivariant -- `M = c I`
gives `P = c I` -- so `delta -> 0` *recovers* `P_inf = I` rather than
approaching it.

`Term.generator` is new, one `F` per diagonal block, every implementation
verified against `logm(Phi(dt))/dt` from the term's own propagator to 9e-15.
`elements.first_point` and the Metal table's row 0 take an explicit `P_inf`;
`None` still means the identity, so every other term keeps the constant row
and no per-chain matrix.

**Gates.** Against a dense float64 Cholesky over ten term shapes x `delta`
from 1e-3 to 0.3: 1e-14 to 1e-16 nearly everywhere. `value_np` against an
mpmath quadrature of the definition: 1e-15, including at zero lag and inside
the exposure width, where it now uses a panelled Gauss-Legendre rule because
the exponential sum is the wrong function there. Suite 741.

**One structural caveat, stated rather than hidden.** A convolved kernel
whose exponential-sum continuation is not positive definite *cannot* have a
PSD process noise -- if it could, that continuation would be a valid
covariance -- and the reduction factors `Q`. Measured minimum eigenvalue of
`Q`, relative: exactly 0 for `RealTerm`, `ComplexTerm`, `RotationTerm` and
`SHOTerm` at `Q > 1/2`; -1e-12 to -3e-2 for the three newly enabled shapes,
growing with `delta`. End to end it costs nothing almost everywhere, with one
measured exception: a convolved Matern-5/2 near `delta/rho ~ 0.01` holds
7e-7 rather than 1e-11. Inside the float32 budget, and documented in
`kernels.md`; it was *not* fixed by tuning the projector weight, which was
measured across four values and made things worse in both directions.

**`||F delta|| <= 8` is a new refusal.** The matrix series' coefficients are
positive but `F delta` is not, so far enough out the alternating powers cost
float32 digits, and no algebraic rewrite removes it -- unlike the scalar
cancellation stage 3c fixed. Beyond the bound the exposure is longer than the
fastest correlation time and the process is nearly white across a frame.

## Order, and why it changed

1. **Compile fix.** Half a day, two-line change, bit-identical, 2x on the
   gradient, removes a process-global workaround. Nothing about it should
   wait.
2. **Per-instrument jitter**, designed as the additive diagonal slot
   stage 3c will reuse.
3. **Complex term, then exposure time**, on the three-rate SHO basis with
   the identity stationary covariance kept.

Each stage lands as its own commit with its gates met and the docs
updated in the same commit.

## What this plan does not do

* It does not close the >=100x forward gate.
* It does not add `TermProduct`.
* It does not raise the `m <= 4` kernel ceiling.

## Stage R — the review that revised this plan, 2026-09-27

An independent attacker was given the four claims marked "measured" and
told to assume each was wrong; in parallel, the reasoned parts of the
plan were checked by reading. Every finding below was reproduced
independently before being acted on, except where marked.

**Wrong, and load-bearing:**

* Stage 2a's premise ("a general complex term needs a general `P_inf`").
  Proved unnecessary for the complex term and insufficient for exposure
  time — see "What was measured first". The original experiment fitted a
  two-vector `h` from one starting point with a local optimiser and
  concluded "unreachable"; the right conclusion was "unreachable *in this
  basis*", and the basis needed one more free rate, not a new covariance.
* Stage 3's framing ("a bounded investigation that may end in a bug
  report"). It is a one-function fix with a measured 2x, and the plan had
  pointed it at the wrong file (`terms.py`; the offender is `special.py`).
* Stage 1's premise that a per-instrument accessor exists (it is
  per-datum) and that a wrong instrument index changes the value (it does
  not at `v = 0`).

**Incomplete:**

* The diagonal correction's parameter dependence and where it must flow.
* Zero spacings: the refusal rule said "non-zero" and zero is the case
  that NaNs.
* Matern-5/2, excluded by the coefficient-based design.
* The oracle, which cannot sample a convolved process.
* The clamp comments, which become false for convolved terms.
* The "uncatchable process abort" claim, in two documents.

**Confirmed:** the exposure-time algebra (rates untouched, exponential
sum beyond the width) over a wider grid than the plan used, including
overdamped and summed terms; the `MAX_JIT` accumulator design; that
compile failure is structural and size-independent.

**Reported by the attacker and not independently reproduced here:** the
1.95x gradient timing; the overdamped analytic continuation to 5e-10; the
duplicate-timestamp indefiniteness table; the 5.7e-12 small-`delta`
reference caveat. Each is marked where used.

**Two lessons for the next plan.** A negative result from a local search
is not a proof; the obstruction here had a two-line algebraic proof that
would have settled it before any code ran. And "capped, allowed to fail"
was the wrong shape for stage 3: it encoded the assumption that the fix
was hard, and one afternoon of bisection showed it was not.
