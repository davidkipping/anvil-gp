# Kernels: what is supported, how to set it up, what is not

Every kernel here is a **linear Gaussian state-space model**. The state
dimension `m` is the number that matters for performance: the fused Metal
kernel holds `3m^2 + 2m + 1` floats per thread in registers, so cost grows
steeply with `m`, and the kernel refuses `m > 6` (m = 5 and 6 run, at
~8x the cost of m = 4).

## Supported kernels

| kernel | `m` | parameters (natural units) | in celerite2? |
|---|---:|---|---|
| `RealTerm` | 1 | `sigma`, `c` | yes |
| `SHOTerm` | 2 | `sigma`, `rho`, `Q` | yes |
| `Matern32Term` | 2 | `sigma`, `rho` | yes, but **approximate** |
| `Matern52Term` | 3 | `sigma`, `rho` | **no** |
| `ComplexTerm` | 2 | `a`, `b`, `c`, `d` | yes |
| `RotationTerm` | 4 | `sigma`, `period`, `Q0`, `dQ`, `f` | yes |
| `TermSum` (via `+`) | sum of parts | concatenated | yes |

### `RealTerm` — `k(tau) = sigma^2 exp(-c tau)`

One state. Note the second parameter is the **rate**, not a timescale:
celerite2's `RealTerm(a, c)` maps as `a = sigma^2`, `c = c`.

```python
ag.RealTerm()                       # params: sigma, c
ParamMap.log([1.2, 3.0])            # sigma = 1.2, c = 3.0 per unit time
```

### `SHOTerm` — stochastically driven damped harmonic oscillator

```python
ag.SHOTerm()                        # params: sigma, rho, Q
ParamMap.log([1.3, 4.0, 3.0])       # sigma, undamped period rho, quality Q
```

`sigma^2` is the stationary variance and `rho = 2 pi / omega_0`. The same
convention as `celerite2.terms.SHOTerm(sigma=, rho=, Q=)`, so parameters
transfer directly.

The `Q = 1/2` point is critical damping, where the general solution has a
removable singularity. anvil-gp's propagator is built to be regular
there — verified against a dense oracle at `Q` in
`{0.1, 1/sqrt2, 1/2 ± eps, 10, 100, 300}` — so you can sample across it
without special-casing.

### `Matern32Term` — `k(tau) = sigma^2 (1 + sqrt3 tau/rho) exp(-sqrt3 tau/rho)`

```python
ag.Matern32Term()                   # params: sigma, rho
```

Exactly the SHO at `Q = 1/2` with `omega_0 = sqrt(3)/rho`, which is why it
costs no new code and is **exact in form** — celerite2 approximates this
kernel rationally.

That buys a real but uneven accuracy advantage. Over 36 on-model
configurations at N = 2000, anvil-gp was more accurate in 27, with a
median error of 1.7e-04 against celerite2's 3.9e-04 and a better worst
case (4.1e-03 against 1.7e-02). It is not a rule you can rely on
pointwise: celerite2 wins about a quarter of the time, because its
approximation error and anvil-gp's float32 error are of similar size here.
See [benchmarks.md](benchmarks.md#accuracy-against-a-float64-dense-cholesky).

### `Matern52Term` — `k(tau) = sigma^2 (1 + x + x^2/3) e^{-x}`, `x = sqrt5 tau/rho`

```python
ag.Matern52Term()                   # params: sigma, rho
```

Three states, a triple characteristic root, process noise built from lower
incomplete gammas. **Not available in celerite2 at all** and not
celerite-representable — a genuine capability difference rather than a
speed one.

### `RotationTerm` — two SHOs at `P` and `P/2`

```python
ag.RotationTerm()                   # params: sigma, period, Q0, dQ, f
```

The primary mode has `Q1 = 1/2 + Q0 + dQ` at damped period `period`; the
secondary has `Q2 = 1/2 + Q0` at `period/2`. Matches the
celerite2/exoplanet rotation kernel.

`f` is the secondary-to-primary **variance ratio**, not the secondary's
share: the code sets `amp = sigma^2/(1+f)` for the primary and `f*amp` for
the secondary, so the secondary carries `f/(1+f)` of the total. At
`f = 0.4` that is 29%, not 40%, and `f = 1` is an even split.

**`f` takes a logit link, not a log one** — it is a ratio in `(0, 1)` in
normal use, and `ParamMap.log` would let it exceed 1 during warmup and
make the "secondary" mode dominate. `make_target` gets this right; if you
build a `ParamMap` by hand, use `KernelParam(link="logit", index=4)` for
`f` and `log` for the rest.

At `m = 4` this sits at the kernel's register ceiling and is the slowest
supported kernel — about 5x a 12-core celerite2 pool, against 17–24x for
`m <= 2`. Still faster, but the margin narrows sharply.

### `ComplexTerm` — `k(tau) = e^{-c tau}(a cos d tau + b sin d tau)`

```python
ag.ComplexTerm()                    # params: a, b, c, d
```

The raw celerite complex-exponential term, matching
`celerite2.terms.ComplexTerm(a, b, c, d)`. `SHOTerm` is the special case
`b/a = c/d` with a physical parametrisation, and is what you want for an
oscillator; `ComplexTerm` is for a decaying oscillation whose amplitude
ratio is a free parameter.

**`b` takes a linear link, not a log one** — it is the only parameter here
that is legitimately negative or zero.

Not every `(a, b, c, d)` is a valid covariance. The constraint is
`k'(0+) = -½ h Q_c hᵀ ≤ 0`, i.e. **`b/a ≤ c/d`**: at `b/a = c/d` the kernel
is smooth at the origin (the SHO), and above it the implied process noise
is not PSD. The state-space form is built on a three-rate basis that
carries the whole valid range with `P_inf = I`.

### `TermConvolution` — exposure-time integration

```python
ag.TermConvolution(ag.RotationTerm(), delta=0.0204)   # Kepler long cadence
```

A detector integrating for `delta` does not sample the process, it samples
its running mean. For a 30-minute cadence against an hours-long signal
that materially smooths the kernel, and ignoring it biases amplitudes and
timescales. Equivalent to `celerite2.terms.TermConvolution`, verified
against it to 1e-11 or better on the log-likelihood over five term shapes,
three widths, on *and* off the parameter map's reference point.

`delta` is fixed data (the exposure time), not a sampled parameter. The
wrapped term's parameters pass through unchanged, so a `ParamMap` built for
the base term needs no edit:

```python
term = ag.TermConvolution(ag.SHOTerm(), delta=0.0204)
pm   = ParamMap.log([1.3, 4.0, 3.0])       # still sigma, rho, Q
```

The state dimension is unchanged, so a convolved `RotationTerm` is still
`m = 4` and still runs on the fused kernel.

**Three rules, all enforced rather than documented-and-hoped.**

*Every spacing must be at least `delta`.* Inside the exposure width the
convolved kernel is simply not the sum of exponentials the state-space form
evaluates, and for a term with a real component the observed matrix goes
indefinite (measured min eigenvalue -3.0e-2 at `delta = 0.1`), which makes
the innovation variance negative and the likelihood NaN. `GPLogLike`
refuses at construction, naming the offending pair. **Duplicate timestamps
are the common case** — anvil-gp otherwise supports them, and two
instruments observing simultaneously is this same case. Bin them, or drop
the convolution. Physically the rule is automatic for one instrument: you
cannot sample faster than you expose.

*The exposure must not be wider than the kernel's fastest correlation
time.* Concretely `||F delta|| <= 8`, where `F` is the generator; beyond it
the matrix series below stop converging. `GPLogLike` refuses at
construction if the reference point is past it, and gives any *chain* that
walks past it the failure value. The regime is degenerate anyway — the
process is nearly white across one frame, so model it as white noise, or
bin.

*Convolve the sum, not a member of it.* `TermConvolution(a + b, delta)` is
supported; `TermConvolution(a, delta) + b` raises. The exposure integrates
the whole signal, and a sum with one convolved member used to be scored as
the wrong kernel, silently (0.04 to 5.3 log-units off).

**Every term with a generator can be wrapped**, which is every term in this
library: `RealTerm`, `SHOTerm` at *any* `Q`, `ComplexTerm`, `Matern32Term`,
`Matern52Term`, `RotationTerm`, and sums. In particular **a convolved
Matern-5/2 is something celerite2 cannot do at all**, and a convolved
overdamped SHO (`Q <= 1/2`) works too — see
[Indefinite, and still exact](#indefinite-and-still-exact) for why that
needed care.

### How it works, and why that matters for what is supported

The convolution leaves the *rates* alone and changes only the stationary
covariance. With `X = F delta`:

```
M = 2 (cosh X - I) X^-2      rescales the amplitudes (lags >= delta)
W = 2 (e^X - I - X) X^-2     the true variance at zero lag
```

`Phi` is untouched, so the state dimension is unchanged and the propagator
is the one already verified for the base term. `M` and `W` are **entire**
functions of `F`, evaluated as series in `X^2` with all-positive
coefficients — and being matrix functions they are indifferent to root
multiplicity. That is the whole reason the Matern terms and the overdamped
SHO work: an earlier version of this feature went through the celerite
coefficients `(a, b, c, d)`, which a repeated root (Matern-3/2), a triple
root (Matern-5/2) and a negative-amplitude decomposition (`Q <= 1/2`) simply
do not have. Nothing is decomposed here, so there is nothing to decompose
badly.

The diagonal correction is `h (W - M) h^T`, and `P_inf` is no longer the
identity: it is the symmetric PSD matrix with `P h^T = M h^T`.

### Indefinite, and still exact

The state-space form above is only *formally* Gaussian. The convolved
exponential sum is not a positive-definite function for every parameter —
a kernel with a corner at zero lag (`RealTerm`, Matern-3/2, a low-`Q` SHO)
has a negative zero-lag correction — so `P_inf`, `Q`, the filtered
covariances and some intermediate determinants are indefinite or negative
for ordinary parameters. The likelihood is still **exact**: the formal model
reproduces the data's covariance, so any algebra that does not assume
positivity gives the right answer. The convolved path therefore drops the
float32 positivity clamps that every other term uses, takes magnitudes of
the intermediate determinants (their product is the data's, which is
positive), and composes blocks through a pivoted direct inverse instead of
a Cholesky factor. On the MLX path, the fused kernel and `.hi` alike.

**History, because the old answers were wrong.** Until 2026-10-23 those
positivity assumptions were applied anyway, and each one silently
substituted a different number: a convolved `RealTerm` at signal-to-noise
10 to 100 came back **1 to 500 log-units wrong**, a convolved Matern-5/2 at
`rho` = 10 to 50 exposures 0.004 to 0.46 wrong, some errors too *high*. The
parity tests against celerite2 ran at signal-to-noise ~4, where none of it
binds. One intermediate version (225b85c) refused those chains instead.

Measured now, over six term shapes, timescales of 1 to 100 exposures and
signal-to-noise 3 to 100 (152 points): float64 matches the dense oracle
everywhere, worst 4e-12 relative; no point is refused; kernel and MLX
gradients match float64 finite differences to 2.3e-4 or better. Wall time is
unchanged (convolved SHO and rotation value+gradient within 1% of before).

**One float32 caveat remains, older than all of this.** `Q(dt) = P - Phi P
Phiᵀ` is formed by subtraction — the form [float32.md](float32.md) forbids
for every other term — because the convolved `P` is not the identity. On
data drawn from the model at moderate signal-to-noise that costs nothing
measurable (float32 error ~1e-4, the same as the unconvolved term, N = 300).
**On bright data it did** -- a TESS 2-min convolved rotation at s/e = 100,
N = 4096, varied by sd 0.19-0.32 across the posterior against 0.018
unconvolved, and its gradient was 10 posterior sigma off at the default
blocks -- **until 2026-10-25, when the lag table moved to float64** (on the
host, rounded to float32; `likelihood._Float64FrontEnd`) and SHO, rotation and
Matern members moved to the square-root block boundary: 0.63 and 0.016
now. [boundary-plan.md](boundary-plan.md), VI. **Far off-model** -- white
noise against a long timescale, as during an early burn-in -- the float32
table was 20-2800x the unconvolved term's error at the same `kappa` for an
SHO (2e-4 at `kappa` ~ 90, 1.3e-3 at ~860, 3.6e-2 at ~9700, N = 300) and
4-40x for a `RealTerm`; with the float64 table it is the unconvolved term's,
1e-5 to 1e-4 on both paths at every `kappa` tried (re-measured 2026-10-25;
the 2026-10-23 figures quoted here before, 0.2 log-units at `kappa` = 50,
came from a script that was not kept and are not reproduced by this one).
A cancellation-free `Q` in float32 would remove the host round-trip; it is
not written.

**One thing is not a kernel.** Below the exposure width the convolution is
not an exponential sum, and the difference at zero lag is a constant added
to every observation's variance. It is a function of the sampled
parameters, so it carries a gradient, and it is **sign-indefinite**: it
equals `(delta/3) k'(0+)` to six digits wherever the kernel's even
extension has a corner, and since `k'(0+) ≤ 0` for any stationary model
with PSD noise, there it *removes* variance. A term with `b/a = c/d` (the
SHO, and `RotationTerm`) has no corner, so that term vanishes and what
remains is positive and `O(delta³)`. You do not need to do anything about
this — it is applied on both evaluation paths — but it is why a convolved
term's variance is not just its kernel at zero.

**Accuracy against celerite2.** anvil-gp is the more accurate of the two at
small `delta`, by a wide margin, and the reason is arithmetic rather than
algorithm: the convolved amplitudes involve `(cosh z - 1)/z²` and
`(sinh z - z)/z²` at `z = c·delta`, both of which cancel to nothing as
`delta → 0`. celerite2 evaluates them as written; anvil-gp uses a series.
At `delta = 1e-3` on a real term, against a 50-digit reference:

| | relative error on the convolved amplitude |
|---|---|
| anvil-gp | 2.2e-16 |
| celerite2 | 2.2e-10 |

That is six orders of magnitude, and it is why the docs' own parity gate
against celerite2 stops at `delta = 1e-2` and uses an mpmath quadrature
below it — a tighter gate there would be measuring celerite2. Written the
naive way, this bug was *invisible in the value* at the reference point and
made the gradient 9% wrong at `delta = 1e-2`, 247x wrong at `1e-3`, and
exactly zero at `1e-4`.

## Sums — `TermSum`

```python
term = ag.SHOTerm() + ag.SHOTerm()          # m = 4, granulation + oscillation
term = ag.SHOTerm() + ag.Matern32Term()     # m = 4
term = ag.RotationTerm() + ag.SHOTerm()     # m = 6, rotation + granulation
```

`Phi` and `Q` become block-diagonal, `h` concatenates, and the states add
up. **The state dimensions add**, which is what pushes a sum past the
kernel's limit:

| combination | `m` | path |
|---|---:|---|
| `SHOTerm() + SHOTerm()` | 4 | fused kernel |
| `SHOTerm() + Matern32Term()` | 4 | fused kernel |
| `SHOTerm() + Matern52Term()` | 5 | fused kernel (~6x the cost of m = 4) |
| `RotationTerm() + SHOTerm()` | 6 | fused kernel (~8x the cost of m = 4) |
| `RotationTerm() + RotationTerm()` | 8 | MLX path (warns) |

Check before you commit to a design:

```python
gp.metal_eligible()        # (True, '') or (False, 'm = 8 exceeds ...')
```

With `GPPolicy(metal=False)` the answer is `(False, 'GPPolicy.metal is
off')`: that reports you turned the kernel off, not that your model is
unsupported. The default (`metal=None`) takes the kernel wherever this
says `True`.

A request the kernel cannot honour **warns and falls back** to the MLX
reference path. That path is correct in float64 and is what the kernel is
verified against, but in float32 it is a fallback rather than a plan:
15-26x slower on the forward at m <= 4, ~4x at m = 5-6 (see
[benchmarks.md](benchmarks.md)), less accurate than the kernel on bright,
densely sampled data (it carries the dense float32 floor of
[float32.md](float32.md) section 7, whose fix on this path is deferred),
and its gradient runs out of Metal buffers from N ~ 16384 at m = 6, sooner
above.

## Not supported

### Celerite-class, but not implemented

| kernel | in celerite2 | status here |
|---|---|---|
| `TermProduct` | (varies) | **not implemented.** Products of kernels are not generally state-space representable at fixed `m`. |

### Coming from celerite or celerite2

`from_celerite_coefficients` takes a celerite coefficient set and returns an
anvil-gp term plus its parameter vector:

```python
import celerite2.terms as c2t
co = c2t.SHOTerm(sigma=1.3, rho=4.0, Q=3.0).get_coefficients()
term, p = ag.from_celerite_coefficients(*co)     # exact, same closed form
```

**This is what `celerite2.terms.OriginalCeleriteTerm` actually is.** An
earlier version of this page called it "a reparametrisation of `ComplexTerm`";
that was wrong. Reading its source, it is a thin wrapper that takes a term
object from the **legacy `celerite` package** and passes
`get_all_coefficients()` straight through, unchanged. So there is no separate
parametrisation to implement and no factor to convert — the coefficients are
already in this convention, and the function above accepts anything either
celerite version can express.

Three coefficient sets are refused, each for a condition documented
elsewhere on this page rather than a new one:

| refused | why |
|---|---|
| `a_r <= 0` or `a <= 0` | `P_inf = I` carries the amplitude in `h` as its square root. The usual source is an SHO below critical damping, whose two real amplitudes have opposite signs — use `SHOTerm` directly, which carries it as one 2-state block at any `Q`. |
| `b/a > c/d` | that is `k'(0+) > 0`, so the implied process noise is not PSD and the set is not a covariance. Equality is the SHO. |
| `d = 0` | a repeated root, not a complex pair — that is `Matern32Term`, which the coefficient form cannot express. |

### Not celerite-class at all

These cannot be written as a sum of exponentials, so neither celerite2 nor
anvil-gp supports them, and no `O(N)` method of this family will:

* **Squared exponential / RBF**, `exp(-tau^2 / 2l^2)`. The classic smooth
  kernel. Approximable by a sum of SHO terms, but not exactly.
* **Periodic / ExpSineSquared**, `exp(-2 sin^2(pi tau / P) / l^2)`. For
  quasi-periodic stellar variability use `RotationTerm`, which is the
  celerite-class stand-in and what the field actually uses.
* **Rational quadratic.**
* **Any non-stationary kernel**, or any kernel on inputs that are not a
  single sorted time coordinate. anvil-gp is one-dimensional and
  stationary by construction.

For those, a dense GP library (george, tinygp, GPyTorch) is the right
tool, at `O(N^3)`.

## Other limits worth knowing

**Noise models.** The observation variance defaults to `yerr^2 +
jitter^2`, with one jitter or one per instrument (any number since
2026-10-25; until then the kernel kept their gradients in registers and
capped them at eight). For anything else, pass `noise_fn` -- celerite2's
`diag` -- giving the **total** per-datum variance from your own
parameters, the error bars, the abscissa and the instrument labels;
gradients come from autodiff, on the fused kernel too (it reads the
variance per observation and writes its adjoint per observation, within
0-10% of the single-jitter cost). `anvilgp.noise` has the common forms,
for `make_target(noise=...)`:

```python
noise.error_scale()                 # (s yerr)^2: error bars right up to a factor
noise.scale_and_jitter(yerr)        # (s yerr)^2 + j^2
noise.error_scale(n_instruments=3)  # one s per instrument (pass instrument=)
```

A hand-written `noise_fn` must act pointwise along the data, and should
use `anvilgp.special.exp_acc` rather than `mx.exp`: MLX's float64 `exp`
is only float32-accurate, which would put a float32 floor under the
float64 `.hi` path. Keep the variance above `1e-6 yerr^2`, where the
innovation floor (`GPPolicy.s_floor_rel`) binds. Several jitters or
per-instrument noise require `param_map.instrument`, whose labels are
range-checked at construction — an out-of-range label used to pass
silently through `mx.take`. Per-instrument *offsets* are supported on both
paths.

**Mean models** are supported on both paths: pass `mean_fn` and
`mean_specs` to `make_target`, or `mean_fn` to `GPLogLike`.

**Cadence.** Any cadence works, including duplicate timestamps and large
gaps. A cadence where every spacing is distinct makes the internal lag
table one row per observation, which costs memory in the *value* (13.2 GiB
at 1024 chains x N = 1e5, against 0.4 GiB for a regular cadence) but is
handled correctly, gradient included.

**`m > 6`** is the hard kernel ceiling, and m = 5 and 6 already spill
registers: value + gradient at m = 6 is ~8x m = 4 where the arithmetic
says 3.4x (still ~10x faster than the MLX path, whose gradient at m = 6
runs out of Metal buffers from N ~ 16384), and their blocks are capped at
128 (m = 5) and 64 (m = 6) points, half the shortest block the GPU
watchdog killed. Raising the ceiling,
or recovering the spill, means restructuring the thread layout; it is
listed under "What is left" in the README.
