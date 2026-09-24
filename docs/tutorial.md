# Tutorial: anvil-gp for a celerite2 user

This walks through the same fit twice, once in celerite2 and once in
anvil-gp, so the differences are concrete. The numbers quoted are the ones
these blocks actually produce; `tests/test_docs.py` re-runs the claims so
they cannot drift.

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
               GPPolicy(metal=True))            # use the fused kernel
v = mx.zeros((1, 3), dtype=mx.float32)          # v = 0 means "at the references"
with mx.stream(mx.gpu):
    ll = float(np.array(gp.log_prob_absolute(v))[0])   # 733.621570
```

The two agree to **1.1e-3** on a log-likelihood of 734. That gap is
float32 against float64 and is discussed under
[accuracy](benchmarks.md#accuracy-against-a-float64-dense-cholesky); it is
three orders below the ~1-unit scale at which a Metropolis accept/reject
decision changes.

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
**1.7 ms** at this N, barely more than the 1.5 ms one costs. The same work through celerite2 is a Python loop:
0.094 ms per vector here, so about 96 ms, and it is 578 ms at N = 1e5 even
on 12 cores. The break-even is ~16 vectors at this N. Full numbers, and
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
import mlx.core as mx
mx.disable_compile()                   # see "Known limitation" in the README
res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=32), gt.target,
                gt.u0(256, seed=1), n_warmup=300, n_samples=300, seed=3)
mx.enable_compile()
```

On the injection-recovery example this runs in 4.0 s against 126 s on the
MLX graph, and mixes better. `mx.disable_compile()` is still required:
the group MLX over-fuses is in the term front end that both paths share,
which M4's kernel did not replace.

## Which path will run?

The fused kernel has limits. Ask before relying on it:

```python
gp.metal_eligible()        # (True, '') or (False, 'reason')
```

For example, `SHOTerm() + Matern52Term()` has `m = 5` and reports
`'m = 5 exceeds the kernel's register budget (max 4)'`.

It returns `False` for `m > 4`, for per-instrument jitter, and when
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
assert abs(gp.log_prob_absolute(v)[0] - ll_c2) < 1e-2 * max(1, abs(ll_c2))
```

There is also a float64 path, `gp.hi(...)`, which runs the same algebra
in double precision on the CPU and agrees with a dense Cholesky to ~1e-11.
Use it when you want to know whether a discrepancy is float32 or a bug.

## Where to go next

* [kernels.md](kernels.md) — every supported kernel, how to set it up,
  and the common ones that are not supported.
* [benchmarks.md](benchmarks.md) — speed and accuracy against celerite2
  across scales and kernels, including where celerite2 wins.
* [float32.md](float32.md) — the precision discipline and its measured
  error budget.
* [statistical-foundation.md](statistical-foundation.md) — why the
  state-space form is exact.
