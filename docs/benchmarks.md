# Benchmarks against celerite2

Regenerate the three measured tables — ensemble size, by kernel, and
accuracy — with one command:

```bash
.venv/bin/python benchmarks/bench_docs_tables.py
```

The break-even table and the MCMC worked example below are hand-written
from separate runs and are **not** covered by that script.

Measured on an M2 Max (12 CPU cores, 64 GB unified memory) on 2026-09-27,
in a window where a stability probe read 0.7% spread and the GPU was
otherwise idle. Wall times on this
machine move by 2x under background load, so check
`benchmarks/verify_readme.py`'s probe before believing any timing you take
yourself.

**Cadence matters and this page uses a gapped one** — a regular grid with
one 20-day gap, so the lag table has three rows. That is what photometry
looks like and it is *not* the cadence the README's milestone-3 figures
use, which is a bare regular grid. Measured side by side in one window,
the gap costs the fused kernel 1.7x at N = 1e4 and 1.5x at N = 1e5, which
is why this page's ratios (20x the pool at N = 1e4) are lower than the
README's 35-42x for what looks like the same comparison. Neither is
wrong; they are different cadences.

## What is being counted

Three numbers vary independently here, and only one of them is the axis
this library is about:

| | meaning | example |
|---|---|---|
| **N** | data points in the light curve | 10,000 |
| **parameters** | hyperparameters in the kernel — the *length* of one parameter vector | 3 for SHO: `(sigma, rho, Q)` |
| **parameter vectors** | distinct hyperparameter **settings** scored at once, against the same data | 1024 |

"Parameter vectors" is the last of these: how many different `(sigma, rho,
Q)` combinations you need a log-likelihood for in one go. In MCMC terms it
is **your walker count** — an ensemble sampler with 1024 walkers needs
1024 log-likelihoods per step, all on the same light curve, differing only
in hyperparameters. It is `n_chains` in the code, the first axis of `v`,
which has shape `(n_chains, n_params)`.

celerite2 evaluates one parameter vector per call, in a Python loop.
anvil-gp evaluates all of them in one GPU launch. Everything below follows
from that.

## The short version

**anvil-gp is not faster than celerite2 at evaluating one likelihood.** It
is faster at evaluating hundreds or thousands of them at once, which is
what an ensemble sampler needs. If your workflow calls the likelihood one
parameter vector at a time, celerite2 is the better tool and this library
has nothing to offer you.

The crossover — where the fused kernel's fixed ~1.5 ms launch cost is
repaid — depends on N, because celerite2's per-vector cost grows with N
while the kernel's barely does:

| N | vs 1 core | vs a 12-core pool |
|---|---:|---:|
| 1,000 | ~26 | ~158 |
| 10,000 | ~5 | ~40 |
| 100,000 | ~1.4 | ~12 |

Against one core the crossover is small; against a *fully parallelised*
pool — the honest denominator if you are willing to run one — it is 158
parameter vectors at N = 1000, falling to 12 at N = 1e5.

Above that anvil-gp pulls away quickly; at 1024 parameter vectors and
N = 1e5 it is
46x a fully parallelised 12-core celerite2 pool and 336x a single core.

## Speed against celerite2, by ensemble size

SHO kernel (3 hyperparameters), one log-likelihood per parameter vector,
milliseconds for the whole batch. The "parameter vectors" column is the
walker count, not the data size and not the hyperparameter count. celerite2 columns are its per-vector cost times the batch,
single core and a 12-core multiprocess pool.

| N | parameter vectors | anvil-gp kernel | anvil-gp MLX | celerite2 1 core | celerite2 12 cores | kernel vs 1 core | kernel vs pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1 | **1.54** | 25 | 0.06 | 0.25 | 0.04x | 0.16x |
| 1,000 | 64 | **1.82** | 25 | 3.97 | 1.07 | 2.2x | 0.58x |
| 1,000 | 1024 | **1.59** | 27 | 67 | 10 | **42x** | 6.5x |
| 10,000 | 1 | **1.96** | 89 | 0.41 | 0.57 | 0.21x | 0.29x |
| 10,000 | 64 | **2.39** | 90 | 27 | 4.27 | **11x** | 1.8x |
| 10,000 | 1024 | **2.85** | 137 | 421 | 50 | **148x** | **18x** |
| 100,000 | 1 | **5.92** | 831 | 3.90 | 4.27 | 0.66x | 0.72x |
| 100,000 | 64 | **8.55** | 854 | 248 | 35 | **29x** | 4.1x |
| 100,000 | 1024 | **13** | 1300 | 4001 | 490 | **315x** | **39x** |

**Read the first row of each block first.** At one parameter vector,
celerite2 on a single core beats the fused kernel by 26x at N = 1000, by
4.8x at N = 1e4, and by 1.5x at N = 1e5 — and its 12-core pool beats it up
to 64 vectors at N = 1000. That is a fixed GPU launch cost of roughly
1.5 ms that a small problem cannot amortise. It is not a defect to be
tuned away; it is what a GPU is.

**The batch is nearly free.** Going from 1 to 1024 parameter vectors
costs the
kernel 5% at N = 1000 and 46% at N = 1e4. The same step costs celerite2 a
factor of 1024. That is the entire argument for this library.

**The MLX column is the reference path**, kept for verification, float64
parity and kernels with `m > 4`. It is not competitive with celerite2 and
is not meant to be.

## What this means for an MCMC run

**One MCMC step costs one log-likelihood per walker.** That is the whole
connection. An ensemble sampler proposes a new position for every walker
each step and must score every one of them, all against the same light
curve, differing only in hyperparameters. So a run of `S` steps with `W`
walkers is `S x W` log-likelihood evaluations.

**They do not arrive as one call of `W`.** anvil's `EnsembleKernel` is a
red-black scheme, as emcee 3 is: the ensemble splits in half, and each
half is proposed against the *frozen* other half, which is what keeps the
vectorised move detailed-balanced. So a step is **two batched calls of
`W/2`**, and an `S`-step run is `2S` calls — the same `S x W` evaluations,
delivered in twice as many batches of half the width. Read the table row
at `W/2`, not `W`.

That split is not free. At N = 1,000 one call of 1024 costs 1.61 ms and
one of 512 costs about 1.2 ms, so two halves cost ~1.5x what one full
batch would: the kernel's cost is dominated by fixed overhead at this
size, and halving the batch does not halve it. The effect shrinks as N
grows and the work per call starts to dominate.

The walker count is not a free choice you can shrink to suit celerite2.
Ensemble samplers need more walkers than twice the parameter count to move
at all, and in practice you want far more — walkers are where the
independent samples come from, so halving them roughly halves your
effective sample size per step. A 4-parameter GP fit with 256 or 1024
walkers is ordinary.

### A worked example

600 steps, 1024 walkers, the SHO kernel, measured end to end with anvil's
ensemble sampler on the fused kernel:

| | N = 1,000 | N = 100,000 |
|---|---:|---:|
| batched calls | 1,200 (2 per step, 512 wide) | 1,200 |
| **anvil-gp, end to end** | **0.76 s** | **10.65 s** |
| celerite2, 614,400 evaluations, 1 core | 36.5 s | 40 min |
| celerite2, same, 12-core pool | 6.0 s | 294 s |
| **speedup vs 1 core** | **48x** | **226x** |
| **speedup vs the pool** | **7.9x** | **28x** |

anvil-gp's row is the minimum of three end-to-end runs with anvil's
sampler. The celerite2 rows scale its measured per-vector cost (0.059 ms
and 3.92 ms) to 614,400 evaluations; the pool row scales its measured
1024-vector throughput. Nothing here is a 12-core end-to-end run.

**What is not in this table, and why.** Earlier versions split anvil-gp's
wall time into "likelihood" and "sampler overhead" (12%, then 17%). Both
were wrong: timing the likelihood in isolation forces an `mx.eval` per
call that the sampler pipelines away, and the isolated sum comes out
*larger than the whole measured run* — 2.06 s of "likelihood" inside a
0.76 s wall. The split is not measurable that way, so it is gone.

Two things to take from this.

**The end-to-end numbers track the kernel-only ones closely here**, which
was not true of the earlier, contended measurements: 7.9x end to end
against 6.5x kernel-only at N = 1000, and 28x against 39x at N = 1e5. The
sampler's own cost and the red-black split (two half-width calls per step
rather than one full-width one) are real but modest at these sizes. Treat
the kernel-only tables as indicative rather than as a guarantee, and
measure your own problem.

**The advantage grows with N**, because celerite2's per-evaluation cost
grows linearly with the data while the fused kernel's grows far more
slowly — over the range in the table above, 1,000 to 100,000 points,
celerite2 goes 67 ms to 4001 ms per 1024-vector batch (60x) while the
kernel goes 1.59 ms to 13 ms (8x). End to end that is 7.9x a fully
parallelised pool at a thousand points and 28x at a hundred thousand.

### If you use a gradient sampler

ChEES-HMC and NUTS also run many chains at once, and each leapfrog step
needs a value *and* a gradient for every chain. anvil-gp's gradient costs
about 3x its value and comes from a second Metal kernel; celerite2 has no
gradient at all, so the comparison there is against finite differences
(`2 x n_params + 1` likelihood calls per gradient, so 7x for SHO) or
against a different library. On the injection-recovery example, ChEES on
the fused kernel runs in 4.0 s against 126 s on anvil-gp's own MLX
reference path.

## Speed by kernel

N = 10,000 data points, 1024 parameter vectors (i.e. 1024 walkers). The
hyperparameter count varies by kernel and is not the axis here.

| kernel | `m` | anvil-gp kernel | anvil-gp MLX | celerite2 1 core | celerite2 12 cores | kernel vs pool |
|---|---:|---:|---:|---:|---:|---:|
| `real` | 1 | **1.47** | 44 | 278 | 36 | **24x** |
| `sho` | 2 | **2.99** | 140 | 471 | 52 | **17x** |
| `matern32` | 2 | **2.81** | 141 | 485 | 49 | **18x** |
| `matern52` | 3 | **6.01** | 381 | — | — | not in celerite2 |
| `rotation` | 4 | **14** | 983 | 562 | 77 | 5.4x |
| `sho2` (`SHO+SHO`) | 4 | **14** | 997 | 569 | 81 | 5.7x |

The kernel's cost scales roughly as `m^2` — it carries `3m^2+2m+1` floats
per thread — while celerite2's scales closer to `m`. So **the advantage
narrows sharply with state dimension**: 24x the pool at `m = 1`, 5.4x at
`m = 4`. If your model is a rotation kernel plus granulation, expect the
lower end of the range, and note that `m > 4` leaves the kernel entirely.

## Accuracy against a float64 dense Cholesky

Absolute error in log L at N = 2,000, where an exact dense factorisation
is affordable. anvil-gp's production path is float32; celerite2 is float64
throughout. **The data is drawn on model** from each kernel: off-model
residuals are large and the quadratic form dominates, which flatters
everyone, while on model the log-determinant and the quadratic form very
nearly cancel and what survives is what a sampler sees.

| kernel | `m` | \|log L\| | anvil-gp kernel | anvil-gp MLX | celerite2 |
|---|---:|---:|---:|---:|---:|
| `real` | 1 | 1650 | 2.74e-04 | 5.18e-04 | 2.27e-13 |
| `sho` | 2 | 620 | 4.22e-05 | 5.30e-04 | 2.61e-12 |
| `matern32` | 2 | 567 | 1.70e-04 | 9.29e-04 | 1.30e-03 |
| `matern52` | 3 | 457 | 5.64e-04 | 5.35e-04 | not available |
| `rotation` | 4 | 660 | 9.67e-04 | 1.56e-02 | 2.61e-11 |
| `sho2` | 4 | 930 | 5.13e-04 | 3.91e-04 | 1.59e-11 |

**celerite2 is seven to nine orders of magnitude more accurate** on every
kernel it implements exactly. That is float64 against float32 and it is
exactly what you should expect. The question is whether it matters: a
Metropolis accept/reject decision turns on differences of order 1, and
anvil-gp's error is 1e-4 to 1e-3 on a log-likelihood of several hundred.
It does not matter for MCMC. It would matter if you needed the absolute
evidence to better than 1e-3, and for that anvil-gp ships a float64 path
(`gp.hi(...)`) that agrees with the dense oracle to ~1e-11.

**Matern-3/2 is the one kernel where that can reverse**, because
celerite2 approximates it rationally where anvil-gp uses the exact
two-state model (the SHO at critical damping). The advantage is real but
not uniform. Over 36 on-model configurations at N = 2,000 — three `sigma`
by four `rho` by three noise seeds:

| | celerite2 | anvil-gp |
|---|---:|---:|
| more accurate in | 9 of 36 | **27 of 36** |
| median error | 3.9e-04 | **1.7e-04** |
| worst case | 1.7e-02 | **4.1e-03** |
| best case | 3.2e-06 | 3.8e-06 |

Median advantage to anvil-gp is 3.6x, but the per-configuration ratio
ranges from 0.02x to 197x: celerite2's approximation error and anvil-gp's
float32 error are of comparable size here, and which dominates depends on
the parameters in a way neither a single number nor a simple rule
captures. The defensible claim is that anvil-gp wins about three times in
four and has the better worst case.

Two earlier versions of this page got this wrong, and both errors are
instructive. The first claimed a flat 65x, measured on *off-model* white
noise where the quadratic form dominates and flatters the comparison. The
second claimed the winner was monotone in `rho/dt`, fitted to four points
that happened to line up; a wider grid shows `sigma` matters too and the
ordering is not monotone in either.

### How float32 error grows

The errors above are at N = 2000 and a modest amplitude. They grow with
both, and not separably — see [float32.md](float32.md). The short version:
at `sigma_GP/yerr = 100` **and** `rho/dt = 1000` simultaneously, the MLX
path is off by 1.5 log-units where either axis alone stays under 0.2,
while the fused kernel holds 0.02. `benchmarks/error_scaling.py` maps it.

## What is not benchmarked here

**tinygp** is not installed in this environment (it needs JAX, which is
not present), so there are no numbers for it. It would be the interesting
third comparison: it is also a JAX-based `O(N)` celerite implementation
with gradients, and on Apple Silicon it would run on CPU or through an
experimental Metal backend. Treat its absence as "not measured", not as a
claim.

**Gradients against celerite2** are not compared because celerite2 does
not provide them. anvil-gp's gradient costs about 3x its value (8.9 ms at
1024 chains and N = 1e4) and matches float64 central differences to ~1e-7.
The honest comparison would be against tinygp or a JAX-wrapped celerite2.

**Prediction / conditioning.** celerite2 has a mature `predict` API.
anvil-gp has `oracle.dense_predict` for float64 reference use, but
prediction is not the batched, GPU-resident path the library is built
around.

## Choosing between them

Use **celerite2** if you evaluate one parameter vector at a time, need
float64 accuracy in the likelihood itself, need a kernel anvil-gp does not
have (`TermProduct`, or `TermConvolution` wrapped around a Matern), or are
not on Apple Silicon.

Exposure-time integration and `ComplexTerm` are no longer reasons to prefer
celerite2 — both landed in M5, and on the convolution's small-`delta`
arithmetic anvil-gp is the more accurate of the two by six orders of
magnitude (see [kernels.md](kernels.md#termconvolution--exposure-time-integration)).

Use **anvil-gp** if you run an ensemble sampler with hundreds or thousands
of walkers, want gradients for HMC without writing them yourself, or need
Matern-5/2, which celerite2 does not have. Matern-3/2 is a weaker reason:
anvil-gp represents it exactly rather than approximately and is more
accurate in about three quarters of configurations, but not reliably
enough to choose a library over.

The two agree on the likelihood to float32 precision, so moving between
them is a change of engine, not of model — the parity tests in
`tests/test_likelihood.py` pin that across every shared kernel, duplicate
timestamps, gaps, jitter and mean models.
