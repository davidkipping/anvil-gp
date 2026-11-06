# Tutorial: anvil-gp for a celerite2 user

This walks through the same fit twice, once in celerite2 and once in
anvil-gp, so the differences are concrete. The numbers quoted are the ones
these blocks actually produce. `tests/test_docs.py` re-runs the load-bearing
ones — the celerite2 agreement, the batched shapes, the kernel/`m` table,
the Matern-3/2 comparison, and the feature sections' parameter lists,
shapes and `gram` agreement — with tolerances loose enough to survive
float32 noise. **The timings and the posterior table below are not
pinned**: they move with machine load and sampler seed, so treat them as
illustrative rather than as a contract.

If you have never used celerite2, the short version is: both libraries
compute the exact marginal likelihood of a Gaussian process whose kernel
is a sum of (possibly complex) exponentials, in `O(N)` rather than
`O(N^3)`. anvil-gp does it as a state-space model on the GPU for
**thousands of parameter vectors at once**; celerite2 does it as a
semi-separable Cholesky factorisation on the CPU for **one at a time**.
That difference is the whole story, and it decides whether anvil-gp is
faster or slower for you — see [benchmarks.md](benchmarks.md).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"     # mlx, numpy (+ celerite2, scipy for tests)
.venv/bin/pip install -e ../anvil      # only for the sampler integration
```

Apple Silicon only: the fast path is a Metal kernel. The library imports
and runs on any machine, but without a GPU you are on the MLX reference
path, which is not competitive with celerite2.

## Some data

```python
import numpy as np
import anvilgp as ag
from anvilgp.oracle import sample_ssm

rng = np.random.default_rng(42)
n = 2000
t = np.arange(n) * 0.02                       # 40 days at ~30 min cadence
yerr = np.full(n, 0.15)
y = sample_ssm(ag.SHOTerm(), [1.2, 5.0, 4.0], 0.02, n, rng)
y = y + rng.standard_normal(n) * yerr         # sigma=1.2, rho=5 d, Q=4
```

`sample_ssm` draws an exact realisation of the process through the
state-space form, which is handy for tests and demos.

## One log-likelihood, both ways

celerite2:

```python
import celerite2
from celerite2 import terms as c2t

gp2 = celerite2.GaussianProcess(c2t.SHOTerm(sigma=1.2, rho=5.0, Q=4.0), mean=0.0)
gp2.compute(t, yerr=yerr)
ll_c2 = gp2.log_likelihood(y)                 # 733.620430
```

anvil-gp:

```python
import mlx.core as mx
from anvilgp import GPLogLike, GPPolicy, ParamMap

gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
               ParamMap.log([1.2, 5.0, 4.0]),   # reference values
               GPPolicy(metal=True))            # the kernel (the default
                                                #   where eligible; True
                                                #   warns if it cannot)
v = mx.zeros((1, 3), dtype=mx.float32)          # v = 0 means "at the references"
with mx.stream(mx.gpu):
    ll = float(np.array(gp.log_prob_absolute(v))[0])   # 733.620472
```

The two agree to **4e-5** on a log-likelihood of 734. That gap is
float32 against float64 and is discussed under
[accuracy](benchmarks.md#accuracy-against-a-float64-dense-cholesky); it is
four orders below the ~1-unit scale at which a Metropolis accept/reject
decision changes. (It was 1.1e-3 until 2026-10-24, when the kernel's
value moved to a replayed filter; the MLX path gives 733.623401, 3e-3.)

### Three things that look different

**You sample offsets, not values.** `ParamMap.log([1.2, 5.0, 4.0])` says
the three kernel parameters are positive scales, sampled as
`ref * exp(v)`. The sampler works in `v`, where the posterior sits near
zero and float32 has the most resolution. `gt.natural(...)` converts back.

**`log_prob_absolute` vs `__call__`.** `__call__` drops a constant
(`-0.5 sum log(2 pi yerr^2)`), which is ~7e5 at N = 1e5 and would
quantise a float32 answer if carried inside. MCMC never sees the
difference. Use `log_prob_absolute` when comparing against another
library, as above, and `__call__` everywhere else.

**Everything is batched.** `v` has shape `(n_chains, n_params)`: one row
per hyperparameter *setting* you want scored, each row holding that
setting's values. So `mx.zeros((1024, 3))` is 1024 different `(sigma, rho,
Q)` triples, all scored against the same light curve in one call. In MCMC
terms the first axis is your walker count; the second is fixed by the
kernel (3 for SHO). That is the point of the library, and the axis every
benchmark in [benchmarks.md](benchmarks.md) varies.

## The thing celerite2 cannot do

```python
u = mx.array(rng.standard_normal((1024, 3)).astype(np.float32) * 0.1)
with mx.stream(mx.gpu):
    lls = np.array(gp(u))                     # shape (1024,)
```

1024 log-likelihoods — one per hyperparameter setting, same data — in
**1.7 ms** at this tutorial's N = 2,000, barely more than the 1.5 ms one
costs. The same work through celerite2 is a Python loop at 0.094 ms per
vector, so about 96 ms. Break-even is ~16 vectors here. Full numbers, and
the cases where celerite2 wins, in [benchmarks.md](benchmarks.md).

And the gradient with respect to every one of them:

```python
with mx.stream(mx.gpu):
    _, grad = mx.vjp(gp, [u], [mx.ones((1024,), dtype=mx.float32)])
    g = np.array(grad[0])                     # shape (1024, 3)
```

This is a hand-derived adjoint running as a second Metal kernel, not
autodiff through the scan. It matches float64 central differences to
~1e-7. celerite2 has no gradient; you would pair it with JAX or finite
differences.

## Long-cadence data: exposure time

A detector integrating for 29.4 minutes does not sample the process, it
samples its running mean. Kepler's long cadence against anything with an
hours-long timescale smooths the kernel materially, and ignoring it biases
amplitudes and timescales. Wrap the term:

```python
cad  = 0.0204                                  # Kepler long cadence, days
t    = np.arange(2000) * cad
term = ag.TermConvolution(ag.SHOTerm(), delta=cad)
pm   = ParamMap.log([1.2, 0.1, 4.0])           # still sigma, rho, Q
gp   = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy(metal=True))
```

`delta` is the exposure time — fixed data, not a sampled parameter — and
the wrapped term's parameters pass through unchanged, so an existing
`ParamMap` needs no edit. The state dimension is unchanged too, so this
still runs on the fused kernel.

On on-model data at `rho = 0.1` d (a 2.4-hour oscillation at a 29.4-minute
cadence), at the true parameters:

| | log-likelihood |
|---|---:|
| anvil-gp, float64 reference | −1982.2663 |
| `celerite2.terms.TermConvolution` | −1982.2663 (rel 5e-15) |
| anvil-gp, fused kernel (float32) | −1982.2662 (rel 5e-08) |
| **the same `SHOTerm` with no convolution** | **−2026.1469** |

Ignoring the exposure costs 44 log-units at the *true* parameters, and
suppresses the zero-lag variance by 12% — a fit that ignores it pays for
the mismatch by distorting `sigma` and `rho`.

**One rule you will meet immediately:** every spacing must be at least
`delta`, duplicate timestamps included. It is refused at construction with a
message naming the offending pair. See
[kernels.md](kernels.md#termconvolution--exposure-time-integration) for why
it is a refusal rather than a warning.

Any term in this library can be wrapped — including `Matern52Term`, which
**celerite2 cannot convolve at all**, and an `SHOTerm` at any `Q`. Wrap a
sum as a whole (`TermConvolution(a + b, delta)`); a convolved term cannot be
one member of a sum. See
[kernels.md](kernels.md#indefinite-and-still-exact) for what makes a
convolved term numerically different from the others, and the one float32
caveat that remains.

## Why the batch is the whole point

An ensemble sampler proposes a new position for every walker at every
step, and needs a log-likelihood for each one — same data, different
hyperparameters. So a run of 600 steps with 1024 walkers is **614,400
log-likelihoods** however they are batched.

Through celerite2 that is a Python loop of 614,400 calls. Through anvil-gp
it is 1,200 batched calls: anvil's `EnsembleKernel` is a red-black scheme,
so each step splits the ensemble and issues **two calls of 512**, which is
what keeps the vectorised move detailed-balanced.

Measured end to end at N = 1,000, that run takes **0.76 s**. The same
614,400 evaluations cost celerite2 36.5 s on one core and 6.0 s on a
12-core pool — 48x and 7.9x. At N = 100,000 it is 10.65 s against 40
minutes and 294 s, so 226x and 28x. Full table and method in
[benchmarks.md](benchmarks.md#what-this-means-for-an-mcmc-run).

You cannot dodge this by using fewer walkers. Ensemble samplers need more
walkers than twice the parameter count to move at all, and your effective
sample size scales with them, so the regime where anvil-gp wins is the
regime you wanted to be in anyway. Full numbers and the end-to-end caveat
in [benchmarks.md](benchmarks.md#what-this-means-for-an-mcmc-run).

## A real fit

```python
import anvil
from anvil.kernels.ensemble import EnsembleKernel
from anvilgp.anvil import make_target

gt = make_target(ag.SHOTerm(), t, y, yerr, policy=GPPolicy(metal=True))
res = anvil.run(EnsembleKernel(gt.target, seed=2), gt.target,
                gt.u0(256, seed=1), n_warmup=1000, n_samples=300,
                seed=3, progress=False)
post = gt.natural(gt.transform.model_np(res.get_chain(flat=True).astype(np.float64)))
```

`make_target` picks the parametrisation, proper default priors, float64
host references and a jitter term. What comes back:

| parameter | median | 16–84% | truth |
|---|---:|---|---:|
| `gp_sigma` | 1.224 | +0.307 / −0.194 | 1.2 |
| `gp_rho` | 5.329 | +0.324 / −0.291 | 5.0 |
| `gp_Q` | 3.867 | +2.310 / −1.184 | 4.0 |
| `jitter` | 0.023 | +0.010 / −0.009 | 0 |

A caution about ensemble widths: on a harder problem the stretch move can
leave a couple of walkers in a secondary mode and inflate a reported
width several-fold while R-hat still looks fine. See
[anvil-stretch-width.md](anvil-stretch-width.md); break widths down by
chain before quoting them.

### With gradients

Because the kernel is differentiable, a gradient sampler works:

```python
res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=32), gt.target,
                gt.u0(256, seed=1), n_warmup=300, n_samples=300, seed=3)
```

On the injection-recovery example this ran in 4.0 s against 126 s on the
MLX graph (2026-09; 5.5 s on the kernel on 2026-10-24, after its value
moved to a replayed filter), and mixes better. Nothing extra is needed: until 2026-09-27
this required a process-global `mx.disable_compile()`, because
`mx.compile` over the backward pass exceeded MLX's fused-argument limit.
One function was responsible and it is fixed; see
[float32.md](float32.md).

## Which path will run?

The fused kernel has limits. Ask before relying on it:

```python
gp.metal_eligible()        # (True, '') or (False, 'reason')
```

For example, `RotationTerm() + RotationTerm()` has `m = 8` and reports
`'m = 8 exceeds the kernel's register budget (max 6)'`.

It returns `False` first of all when `GPPolicy(metal=False)` turned the
kernel off — `(False, 'GPPolicy.metal is off')` means "you turned it off",
not "unsupported"; the default, `metal=None`, takes the kernel wherever
this says `True`. It also returns `False` for `m > 6`, and when
`mx.fast.metal_kernel` is unavailable. A request that cannot be honoured
**warns and falls back** to the MLX path, which is correct but far
slower — it does not fail silently. See [kernels.md](kernels.md) for the
state dimension of each kernel and what combinations stay under the
limit.

## Checking against celerite2 yourself

The parity tests do this over every kernel, cadence and corner
(`tests/test_likelihood.py`), including the cases where celerite2 is
itself approximate. To spot-check your own problem:

```python
with mx.stream(mx.gpu):
    ll = float(np.array(gp.log_prob_absolute(v))[0])
assert abs(ll - ll_c2) < 1e-2 * max(1, abs(ll_c2))
```

(`log_prob_absolute` returns float64, which lives on the CPU stream:
convert it with `np.array` as here rather than indexing it on the GPU,
which raises "float64 is not supported on the GPU".)

There is also a float64 path, `gp.hi(...)`, which runs the same algebra
in double precision on the CPU and agrees with a dense Cholesky to ~1e-11.
Use it when you want to know whether a discrepancy is float32 or a bug.
It refuses the same chains the float32 path does (it is the float64 value
of the same target, and anvil's `reanchor_every` writes it into the chain
state); `gp.hi(v, refuse=False)` gives the raw float64 number everywhere.

## Beyond the plain likelihood

The sections below use the same `t`, `y`, `yerr`, `rng` and the imports
above. Each feature is general (it knows nothing about transits or
instruments) and each keeps the likelihood `O(N)`.

### Noise models

The default observation variance is `yerr^2 + jitter^2`. If you do not
trust the error bars, say so in the model:

```python
from anvilgp import noise

gt = make_target(ag.SHOTerm(), t, y, yerr, noise=noise.scale_and_jitter(yerr))
gt.names   # ['log_gp_sigma', 'log_gp_rho', 'log_gp_Q',
           #  'log_noise_scale', 'log_noise_jitter']
```

That is `(s yerr)^2 + j^2`; `noise.error_scale()` is `(s yerr)^2` alone,
with one `s` per instrument if you ask. Anything else -- flux- or
time-dependent noise -- is a `noise_fn` of your own, celerite2's `diag`;
the two rules it must follow are in
[kernels.md](kernels.md#other-limits-worth-knowing). It runs on the fused
kernel, gradient included, within 0-10% of the plain jitter's cost.

### Linear parameters, marginalised

A baseline that is linear in its coefficients -- a level and a slope here,
`(1 + m0)(a + b t)` around a transit -- need not be sampled. Give
`make_target` the basis and the coefficients are integrated out
analytically:

```python
tc = float(t.mean())

def baseline(v_mean, x, m0):             # -> (N, p): one basis for every chain
    return mx.stack([mx.ones_like(x), x - tc], axis=-1)

y_lin = y + 3.0 + 0.05 * (t - tc)        # an offset and a trend to remove
gt = make_target(ag.SHOTerm(), t, y_lin, yerr, linear_basis=baseline)
gt.names   # ['log_gp_sigma', 'log_gp_rho', 'log_gp_Q', 'log_jitter']: no beta

res = anvil.run(EnsembleKernel(gt.target, seed=2), gt.target,
                gt.u0(256, seed=1), n_warmup=1000, n_samples=300,
                seed=3, progress=False)
v = gt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
vs = v[::200]                            # 384 of the 76,800 draws
beta, cov = gt.linear_posterior(vs)      # beta | v is Gaussian, in float64
draws = beta + np.einsum("cij,cj->ci", np.linalg.cholesky(cov),
                         rng.standard_normal(beta.shape))
```

`draws` are posterior draws of β: one per draw of the rest, from its
exact conditional. Here they give a level of 2.97 (16-84%: 2.88-3.10)
and a slope of 0.061 (0.051-0.071) against 3.0 and 0.05.

The basis comes in two forms, and the first is the one to use when it
fits:

* **Shared, `(N, p)`**, as above: the same columns for every chain. It is
  a constant, so it is evaluated once, when the target is built, in
  float64, and orthonormalised there; float32 never sees the columns as
  you wrote them. So their conditioning does not matter: raw times work
  (`mx.stack([mx.ones_like(x), x], -1)` with `x` a BJD gives exactly the
  same float32 value as the centred form above), nearly parallel columns
  cost nothing, and exactly collinear ones (a global level beside one
  offset per instrument) are fine under a Gaussian `linear_prior` and
  refused as improper under a flat one. It is also the cheapest: the
  kernel reads it once for every chain.
* **Per chain, `(n_chains, N, p)`**, when the columns move with the mean
  model (`(1 + m0)(a + b t)` around a transit, through `m0`) or with
  sampled parameters. It is rebuilt at every evaluation, in float32 on the
  GPU, from a float32 `x` -- which rounds a BJD to 0.25 d before your
  function sees it, as it does for `mean_fn`. So pass `x_mean` relative to
  a reference time, and centre and scale the columns yourself: nearly
  parallel ones cost digits (`[1, 1 + 1e-3 (t - tc)]`, correlation
  0.99998, is off by 1e-3 log-units on the kernel and by 41 on the MLX
  path, whose Gram comes from differences of likelihoods), and exactly
  collinear ones are refused in float32 under either prior (the float64
  paths, `gp.hi`, `linear_posterior` and `predict`, evaluate them under a
  Gaussian one).

The likelihood is the restricted one: it keeps `-1/2 ln|Phi^T K^-1 Phi|`,
which is what makes the marginal posterior of the other parameters match a
fit that samples β under a flat prior. `linear_prior=(mu, S)` gives a
Gaussian prior on β instead. It costs, on the kernel at N = 1e4 x 1024
chains, 1.7x the plain likelihood for p = 2 on value + gradient with a
shared basis, rising to 5.6x at p = 10 (per chain 1.9x and 6.3x, and
about twice the memory at large N;
[benchmarks.md](benchmarks.md#what-the-general-features-cost)).

### Predicting the GP

The GP component's posterior at any times, for each draw, in float64:

```python
t_new = np.linspace(-2.0, 42.0, 500)
mean, var = gt.predict(vs, t_new)        # each (384, 500)
```

That is `E[f(t_new) | y]` and its variance under each draw's
hyperparameters -- not including the mean model or the observation noise,
and with β integrated over its posterior when it is marginalised. It
matches the dense conditional to ~1e-14. It runs on the CPU, as a Kalman
filter and smoother over data and prediction points together, vectorised
over draws: 0.7 s for these 384 draws, and for a single draw about 0.2 s
at N = 1e4 and 2 s at N = 1e5. Thin the chain first. A parallel version is on the open list in the README.

### Holding a feature out of the GP

When the measurement is the *shape* of a short feature -- a transit's
ingress and egress, a spot crossing, a flare -- a flexible GP fitted to
all the data is free to absorb part of it as noise. `held_out` keeps
those points out of the GP's conditioning:

```python
gt = make_target(ag.SHOTerm(), t, y, yerr, held_out=[(18.0, 20.0)])
gt.names   # ['log_gp_sigma', 'log_gp_rho', 'log_gp_Q', 'log_jitter']: as before
```

The GP is conditioned on the rest, S, and applied to the held-out points,
M, through its conditional mean given S; M is scored with its white noise
alone:

    log L = log N(r_S; 0, K_SS + W_S) + log N(r_M - E[f_M | r_S]; 0, W_M)

That is a proper Gaussian likelihood -- the joint covariance whose Schur
complement on M is `W_M` -- so the marginalised baseline, `gram`,
`linear_posterior` and `predict` work with it unchanged; `predict`
conditions on S only, so at a held-out time it gives `E[f_M | r_S]`.
`held_out` takes a boolean mask of length N or a list of `(t_lo, t_hi)`
windows. It is data, fixed when the target is built, and the parameter
vector is the same as without it. (Adding the conditional variance to the
M term instead would give back the unmasked likelihood exactly, which is
why M is scored white.)

What it does not model is correlated noise *inside* a window. Deep inside
one much longer than the kernel's timescale `E[f | r_S]` falls back to
zero, and the points there are scored as white noise about the mean
model, which can overstate a feature's significance unless the noise on
those timescales is independently shown to be white. To sample extra
white noise on the held-out points, label them as their own instrument
and give it a jitter.

It runs on the fused kernel, value and gradient, a marginalised baseline
included: S is scanned forward and time-reversed, and the two are joined
at each held-out point. In float32 that is more accurate than the MLX
path. At N = 2818 with 14% held out in two windows, an error-scale noise
model and a quadratic baseline, a value costs 1.8x the unmasked
likelihood at 128 chains (2.3x at 1024) and value + gradient 2.3x (3.1x)
([benchmarks.md](benchmarks.md#what-the-general-features-cost)). A
convolved term (`TermConvolution`) runs on the MLX path.

**Starting from an unmasked fit.** The parameters are the same, so an
unmasked run's last positions are a start for the masked one; there is
no need to run `anvil.optimize` again:

```python
plain = make_target(ag.SHOTerm(), t, y, yerr)
res0 = anvil.run(EnsembleKernel(plain.target, seed=2), plain.target,
                 plain.u0(256, seed=1), n_warmup=1000, n_samples=300,
                 seed=3, progress=False)
v_last = plain.transform.model_np(
    np.asarray(res0.get_chain()[-1], dtype=np.float64))
u_start = mx.array(gt.transform.from_model_np(v_last),   # via model space;
                   dtype=mx.float32)                     # anvil wants mx
res = anvil.run(EnsembleKernel(gt.target, seed=2), gt.target, u_start,
                n_warmup=500, n_samples=300, seed=4, progress=False)
```

Going through model space keeps this right even if the two targets'
transforms differ (`seed_period`, for one, reads only the kept points).
The posterior moves once the feature stops being absorbed, so keep a
warm-up and check convergence as usual: the start saves the search, not
the burn-in.

### The primitives underneath

The quadratic form `Y^T K^-1 Y` and the solve `K^-1 Y`, for any columns
`Y`, are public:

```python
gp = gt.loglike
Y = np.stack([np.ones(n), t - tc], 1)                       # (N, k)
with mx.stream(mx.gpu):
    G = gp.gram(mx.array(vs[:4], dtype=mx.float32), Y)      # (4, 2, 2)
X = gt.apply_inverse(vs[:4], Y)                             # (4, N, 2)
```

`gram` is batched and differentiable, in `v` and in `Y`, on every path:
the marginalisation above is built on it, and the fused kernel carries all
k columns through one filter (up to 32). `apply_inverse` is float64 and
value-only -- no gradient in `v`. `Y` may be shared, `(N,)` or `(N, k)`,
or per chain, `(n_chains, N, k)`. Here `Y^T X` agrees with `G` to ~1e-6
of its largest entry (5e-7 to 3e-6 across runs), float32 against float64.

### Several datasets at once

Datasets with some parameters in common are one target, composed by
anvil. Two halves of the light curve here, sharing a mean level `mu`:

```python
from anvilgp.anvil import joint_u0
from anvilgp.priors import SpecHint

def level(v_mean, x):
    return mx.broadcast_to(v_mean[:, :1], (v_mean.shape[0], x.shape[-1]))

h, mu = n // 2, [SpecHint("mu", lo=-2.0, hi=2.0)]
a = make_target(ag.SHOTerm(), t[:h], y[:h] + 0.5, yerr[:h],
                mean_fn=level, x_mean=t[:h], mean_specs=mu)
b = make_target(ag.SHOTerm(), t[h:], y[h:] + 0.5, yerr[h:],
                mean_fn=level, x_mean=t[h:], mean_specs=mu)
jt = anvil.joint({"a": a.target, "b": b.target}, shared={"mu": ["a", "b"]})
jt.names   # ['mu', 'a.log_gp_sigma', ..., 'b.log_jitter']
res = anvil.run(EnsembleKernel(jt, seed=2), jt,
                joint_u0(jt, {"a": a, "b": b}, 256),
                n_warmup=1000, n_samples=300, seed=3, progress=False)
```

Each half keeps its own GP and jitter; `mu` comes out at 0.40
(0.29-0.51). `anvil.joint` needs anvil 0.12.0 or later; it composes the
members in model space, so the shared column's prior and Jacobian count
once, and the float64 path and `anvil.validate_precision` work on the
result. `joint_u0` starts each member's columns at its own reference. A
member can be any anvil log-density, so a likelihood that is not a GP, or a
GP anvil-gp does not support, sits beside these. A shared GP
hyperparameter needs the same spec on every member.

## Where to go next

* [kernels.md](kernels.md) — every supported kernel, how to set it up,
  and the common ones that are not supported.
* [benchmarks.md](benchmarks.md) — speed and accuracy against celerite2
  across scales and kernels, including where celerite2 wins, and what
  the features above cost.
* [float32.md](float32.md) — the precision discipline and its measured
  error budget.
* [statistical-foundation.md](statistical-foundation.md) — why the
  state-space form is exact.
