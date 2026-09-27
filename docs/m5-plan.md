# Plan: per-instrument jitter, exposure-time integration, and the compile split

Status: proposed 2026-09-27; **adversarially reviewed the same day and
revised** — see "Stage R" at the end. Not locked. The review overturned the
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

## Stage 1 — the compile fix (was stage 3, was "capped and allowed to fail")

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

The measured prize, one size (64 chains x N = 4096, Metal path,
interleaved): gradient 3.72 ms → 3.04 ms from removing the dead branch,
→ 1.91 ms compiled — **1.95x overall**, against the "1.15-1.25x" the
plan expected. One machine, one size, indicative only. Half a day.

**Not verified:** whether anvil's ChEES leapfrog then compiles end to end.
The likelihood's `value_and_grad` does; the sampler was not run.

## Stage 2 — per-instrument jitter in the kernel (was stage 1)

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

## Stage 3 — the general complex term, and exposure time on top of it (was stage 2)

### 3a. Generalise `_sho_matrices` from two rates to three

`(omega_0, Q)` becomes `(c, beta, s)` with `P_inf = I` kept. `Phi` and `Q`
keep their algebraic shape exactly (verified against `expm` and
`I - Phi Phiᵀ` to 2e-15):

```
Q11 = (1 - E) - 2 b_ E C S - 2 b_² E S²
Q22 = (1 - E) + 2 b_ E C S - 2 b_² E S²
Q12 = 2 b_ s_ E S²          with b_ = beta dt, s_ = s dt, E = e^{-2c dt}
```

**The one new numeric object** is the two-argument
`1 - e^{-2a}(1 + 2b + 2b²)`, which replaces `gamma3(2a) = 1 - e^{-2a}(1 +
2a + 2a²)` and reduces to it at `a = b`. Its small-`dt` behaviour is
`2(a - b) = 2 g dt`, first order rather than third, so the cancellation
is *milder* everywhere except as `g → 0`, the SHO boundary, where
`gamma3` is the limit. **That is where stage 3's risk lives**, and it is
not the risk the original plan named. The `direct`/`series` split in
`special.py` is the template.

Gate: the existing SHO reproduced bit-identically in float64 and float32
through the general path (`beta = c, s = omega_0`); the new function
pinned against mpmath in float32 across the `(a, b)` plane including
`g → 0`; the whole suite green with no tolerance change. Because
`P_inf = I` is kept, `first_point()`, the pivot floor in `linalg.py`, the
`.hi()` path and `sample_ssm` need **no change** for this stage.

### 3b. `ComplexTerm` and `OriginalCeleriteTerm`

The construction above, exposed. Gate: float64 parity with
`celerite2.terms.ComplexTerm` to ~1e-11 over a grid including `d → 0`,
`b = 0`, `c → 0` and `b` on the positive-definiteness boundary (where
`Q_c` has an exact zero eigenvalue); `OriginalCeleriteTerm` by
reparametrisation. Overdamped `Q < 1/2` goes through the analytic
continuation to `d² < 0`; `F` and `h` come out exactly real (measured,
max imaginary part 0.0). **Do not route it through celerite2's two-real-
term decomposition**, which carries a negative amplitude and is the only
form that would need a signed `P_inf`.

### 3c. `TermConvolution`

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
