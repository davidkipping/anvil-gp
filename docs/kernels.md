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

### Sums — `TermSum`

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
| `ComplexTerm(a, b, c, d)` | yes | **not implemented.** The general complex exponential term. `SHOTerm` covers the oscillatory case with a physical parametrisation; `ComplexTerm` is the raw form. Nothing about the state-space approach prevents it. |
| `OriginalCeleriteTerm` | yes | **not implemented.** The original Foreman-Mackey et al. parametrisation. |
| `TermConvolution` | yes | **not implemented.** Exposure-time integration — convolving the kernel with a finite integration window. Matters for long-cadence photometry where the 30-minute bin smooths short-period signals. If you need this, use celerite2. |
| `TermProduct` | (varies) | **not implemented.** Products of kernels are not generally state-space representable at fixed `m`. |

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

**One jitter parameter for the fused kernel.** A per-instrument jitter is
supported on the MLX path but not in the kernel yet; `metal_eligible()`
reports it. Per-instrument *offsets* are supported on both.

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
