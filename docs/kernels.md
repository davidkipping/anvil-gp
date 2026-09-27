# Kernels: what is supported, how to set it up, what is not

Every kernel here is a **linear Gaussian state-space model**. The state
dimension `m` is the number that matters for performance: the fused Metal
kernel holds `3m^2 + 2m + 1` floats per thread in registers, so cost grows
steeply with `m`, and the kernel refuses `m > 4`.

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

*A wrapped `SHOTerm` needs `Q > 1/2`, and your prior must enforce it.*
Below critical damping the SHO is two real exponentials, one with a
**negative** amplitude, which the `P_inf = I` construction cannot carry.
`GPLogLike` refuses a reference point at `Q <= 1/2`, but a sampler can
still walk there, and what it finds is not a rejection: the non-finite
guard returns `-1e30` with an exactly **zero gradient**, a flat plateau a
gradient sampler cannot escape. Bound `Q` away from 1/2 in the prior.
`RotationTerm` has `Q >= 1/2 + Q0 > 1/2` by construction and needs no such
care — which, with `m = 4` and long-cadence photometry of a rotating star
being the motivating case, makes it the term to reach for.

*Only terms with exact celerite coefficients can be wrapped.*
`Matern32Term` (a repeated root) and `Matern52Term` (a triple one) are not
sums of distinct exponentials, so the closed form does not apply and
wrapping them raises `TypeError`. Their convolutions *are* state-space
models of the same order — measured to 1.6e-15 — but reaching them needs
the `m = 3` propagator generalised the way `ComplexTerm` generalised
`m = 2`. Not done; use celerite2's approximation, or an `SHOTerm` at
`Q = 1/2 + eps` for the Matern-3/2 shape.

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
term = ag.RotationTerm() + ag.SHOTerm()     # m = 6 -> MLX path only
```

`Phi` and `Q` become block-diagonal, `h` concatenates, and the states add
up. **The state dimensions add**, which is what pushes a sum past the
kernel's limit:

| combination | `m` | path |
|---|---:|---|
| `SHOTerm() + SHOTerm()` | 4 | fused kernel |
| `SHOTerm() + Matern32Term()` | 4 | fused kernel |
| `SHOTerm() + Matern52Term()` | 5 | MLX path (warns) |
| `RotationTerm() + SHOTerm()` | 6 | MLX path (warns) |

Check before you commit to a design:

```python
gp.metal_eligible()        # (True, '') or (False, 'm = 5 exceeds ...')
```

With the default `GPPolicy` the answer is `(False, 'GPPolicy.metal is
off')`: that reports you did not ask for the kernel, not that your model
is unsupported.

A request the kernel cannot honour **warns and falls back** to the MLX
reference path. That path is correct and is what the kernel is verified
against, but it is 30–70x slower depending on the kernel (see
[benchmarks.md](benchmarks.md)), so it is a fallback rather than a plan.

## Not supported

### Celerite-class, but not implemented

| kernel | in celerite2 | status here |
|---|---|---|
| `OriginalCeleriteTerm` | yes | **not implemented.** A reparametrisation of `ComplexTerm`, which *is* implemented -- convert the coefficients yourself and use `ComplexTerm`. |
| `TermProduct` | (varies) | **not implemented.** Products of kernels are not generally state-space representable at fixed `m`. |
| `TermConvolution` of a Matern | yes | **not implemented.** See `TermConvolution` above: a repeated root has no celerite coefficients, and it raises rather than approximating. |

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

**Per-instrument jitter, up to eight instruments.** The kernel used to
accept one jitter only; since 2026-09-27 it indexes the jitter per
observation from `param_map.instrument`, and the backward gives each
instrument its own gradient. The cap is `metal.MAX_JITTER = 8`, a register
budget like `m <= 4` rather than an arbitrary limit; above it
`metal_eligible()` refuses and the MLX path is correct. Several jitters
require `param_map.instrument`, whose labels are range-checked at
construction — an out-of-range label used to pass silently through
`mx.take`. Per-instrument *offsets* were already supported on both paths.

**Mean models** are supported on both paths: pass `mean_fn` and
`mean_specs` to `make_target`, or `mean_fn` to `GPLogLike`.

**Cadence.** Any cadence works, including duplicate timestamps and large
gaps. A cadence where every spacing is distinct makes the internal lag
table one row per observation, which costs memory in the *value* (13.2 GiB
at 1024 chains x N = 1e5, against 0.4 GiB for a regular cadence) but is
handled correctly, gradient included.

**`m > 4`** is the hard kernel ceiling. Raising it means spilling
registers or restructuring the thread layout; it is listed under "What is
left" in the README.
