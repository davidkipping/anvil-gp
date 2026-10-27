# Benchmarks against celerite2

Regenerate the three measured tables — ensemble size, by kernel, and
accuracy — with one command:

```bash
.venv/bin/python benchmarks/bench_docs_tables.py
```

The m = 5 and 6 rows come from `--kernels sho_m52,rot_sho`. The break-even
table is derived from the ensemble-size table by linear interpolation; the
MCMC worked example is `benchmarks/mcmc_end_to_end.py`.

Measured on an M2 Max (12 CPU cores, 64 GB unified memory) on 2026-10-25,
in a window where `verify_readme.py`'s stability probe read 73.1 ms before
and 73.6 ms after (its quiet level is 73.3) at 0.2% spread. That is after
two accuracy fixes that each cost the kernel speed on purpose: the value
moved to a replayed filter on 2026-10-24 (1.6-1.7x on the forward, 10-100x
more accurate on bright, densely sampled data; docs/float32.md, section 7),
and the block boundaries moved to the square-root information form on
2026-10-25 (1.6-1.8x on the forward at m <= 2 and 2.5x at m = 4, where the
watchdog cap of 256-point blocks adds to it; gradient samplers now converge
on bright data; [boundary-plan.md](boundary-plan.md)). Every kernel figure
here is 1.2-2.5x the 2026-10-24 version of this page; the celerite2 figures
moved within noise. Wall times on this
machine move by 2x under background load, so check
`benchmarks/verify_readme.py`'s probe before believing any timing you take
yourself.

**Cadence matters and this page uses a gapped one** — a regular grid with
one 20-day gap, so the lag table has three rows. That is what photometry
looks like and it is *not* the cadence the README's milestone-3 figures
use, which is a bare regular grid. Measured side by side in one window,
the gap costs the fused kernel 1.7x at N = 1e4 and 1.5x at N = 1e5, which
is why this page's ratios (9.4x the pool at N = 1e4) are lower than the
README's 13x for what looks like the same comparison. Neither is
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

The crossover — where the fused kernel's fixed ~1.7 ms launch cost is
repaid — depends on N, because celerite2's per-vector cost grows with N
while the kernel's barely does:

| N | vs 1 core | vs a 12-core pool |
|---|---:|---:|
| 1,000 | ~30 | ~150 |
| 10,000 | ~7 | ~46 |
| 100,000 | ~4 | ~26 |

Against one core the crossover is small; against a *fully parallelised*
pool — the honest denominator if you are willing to run one — it is ~150
parameter vectors at N = 1000, falling to ~26 at N = 1e5.

Above that anvil-gp pulls away; at 1024 parameter vectors and N = 1e5 it
is 11x a fully parallelised 12-core celerite2 pool and 86x a single core.
**This is for m <= 4.** At m = 6 (rotation + granulation) the 12-core pool
is 2.3x faster than the kernel's forward at N = 1e4 and 1024 vectors; see
"Speed by kernel".

## Speed against celerite2, by ensemble size

SHO kernel (3 hyperparameters), one log-likelihood per parameter vector,
milliseconds for the whole batch. The "parameter vectors" column is the
walker count, not the data size and not the hyperparameter count. celerite2 columns are its per-vector cost times the batch,
single core and a 12-core multiprocess pool.

| N | parameter vectors | anvil-gp kernel | anvil-gp MLX | celerite2 1 core | celerite2 12 cores | kernel vs 1 core | kernel vs pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1 | **1.73** | 25 | 0.06 | 0.19 | 0.04x | 0.11x |
| 1,000 | 64 | **1.97** | 25 | 3.92 | 1.19 | 2.0x | 0.61x |
| 1,000 | 1024 | **2.05** | 26 | 64 | 10 | **31x** | 4.9x |
| 10,000 | 1 | **2.89** | 88 | 0.40 | 0.59 | 0.14x | 0.20x |
| 10,000 | 64 | **3.54** | 89 | 27 | 4.42 | 7.5x | 1.2x |
| 10,000 | 1024 | **5.63** | 135 | 408 | 53 | **72x** | 9.4x |
| 100,000 | 1 | **13** | 838 | 3.77 | 3.99 | 0.28x | 0.30x |
| 100,000 | 64 | **21** | 796 | 241 | 35 | **11x** | 1.6x |
| 100,000 | 1024 | **45** | 1278 | 3899 | 488 | **86x** | **11x** |

(Uncompiled calls, as the script has always timed them: the launch cost
here is larger than a compiled sampler step pays.)

**Read the first row of each block first.** At one parameter vector,
celerite2 on a single core beats the fused kernel by ~29x at N = 1000, by
7.2x at N = 1e4, and by 3.4x at N = 1e5 — and its 12-core pool beats it up
to 64 vectors at N = 1000. That is a fixed GPU launch cost of roughly
1.7 ms that a small problem cannot amortise. It is not a defect to be
tuned away; it is what a GPU is.

**The batch is nearly free.** Going from 1 to 1024 parameter vectors
costs the
kernel 18% at N = 1000 and 95% at N = 1e4. The same step costs celerite2 a
factor of 1024. That is the entire argument for this library.

**The MLX column is the reference path**, kept for verification, float64
parity and kernels with `m > 6`. It is not competitive with celerite2 and
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

That split is not free. At N = 1,000 one compiled call of 1024 costs
1.41 ms and one of 512 costs 1.22 ms, so two halves cost ~1.7x what one
full batch would: the kernel's cost is dominated by fixed overhead at this
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
| **anvil-gp, end to end** | **1.44 s** | **33.2 s** |
| celerite2, 614,400 evaluations, 1 core | 38.4 s | 39 min |
| celerite2, same, 12-core pool | 6.0 s | 293 s |
| **speedup vs 1 core** | **27x** | **70x** |
| **speedup vs the pool** | **4.2x** | **8.8x** |

anvil-gp's row is the minimum of three end-to-end runs with anvil 0.11.0's
sampler in one process, 2026-10-25 (`benchmarks/mcmc_end_to_end.py`; the
first run, compilation included, was within 10% of the others). The
2026-10-24 version of this table was 0.89 s and 17.6 s, before the
square-root boundary, from a run whose script was not kept. The celerite2
rows scale its measured per-vector cost (0.0625 ms and 3.81 ms, from the
1024-vector cells) to 614,400 evaluations; the pool row scales its
measured 1024-vector throughput. Nothing here is a 12-core end-to-end
run.

**What is not in this table, and why.** Earlier versions split anvil-gp's
wall time into "likelihood" and "sampler overhead" (12%, then 17%). Both
were wrong: timing the likelihood in isolation forces an `mx.eval` per
call that the sampler pipelines away, and the isolated sum comes out
*larger than the whole measured run* — 2.06 s of "likelihood" inside a
0.76 s wall. The split is not measurable that way, so it is gone.

Two things to take from this.

**The end-to-end numbers track the kernel-only ones closely here**, which
was not true of the earlier, contended measurements: 4.2x end to end
against 4.9x kernel-only at N = 1000, and 8.8x against 11x at N = 1e5. The
sampler's own cost and the red-black split (two half-width calls per step
rather than one full-width one) are real but modest at these sizes. Treat
the kernel-only tables as indicative rather than as a guarantee, and
measure your own problem.

**The advantage grows with N**, because celerite2's per-evaluation cost
grows linearly with the data while the fused kernel's grows far more
slowly — over the range in the table above, 1,000 to 100,000 points,
celerite2 goes 64 ms to 3899 ms per 1024-vector batch (61x) while the
kernel goes 2.05 ms to 45 ms (22x). End to end that is 4.2x a fully
parallelised pool at a thousand points and 8.8x at a hundred thousand.

### If you use a gradient sampler

ChEES-HMC and NUTS also run many chains at once, and each leapfrog step
needs a value *and* a gradient for every chain. anvil-gp's value and
gradient together cost about 2.3x its value alone (10.1 against 4.4 ms at
1024 chains and N = 1e4, the README's cadence) and come from a second
Metal kernel; celerite2 has no
gradient at all, so the comparison there is against finite differences
(`2 x n_params + 1` likelihood calls per gradient, so 7x for SHO) or
against a different library. On the injection-recovery example, ChEES
(64 chains, 300 warmup and 300 samples) on the fused kernel ran in 5.5 s
(2026-10-24, before the square-root boundary; not re-measured); 126 s on anvil-gp's own MLX reference path when that was
last measured, 2026-09.

## Speed by kernel

N = 10,000 data points, 1024 parameter vectors (i.e. 1024 walkers). The
hyperparameter count varies by kernel and is not the axis here.

| kernel | `m` | anvil-gp kernel | anvil-gp MLX | celerite2 1 core | celerite2 12 cores | kernel vs pool |
|---|---:|---:|---:|---:|---:|---:|
| `real` | 1 | **2.70** | 41 | 261 | 31 | **11x** |
| `sho` | 2 | **5.43** | 137 | 409 | 52 | 9.7x |
| `matern32` | 2 | **5.35** | 136 | 409 | 50 | 9.3x |
| `matern52` | 3 | **14** | 366 | — | — | not in celerite2 |
| `rotation` | 4 | **51** | 952 | 536 | 72 | 1.4x |
| `sho2` (`SHO+SHO`) | 4 | **50** | 954 | 558 | 71 | 1.4x |
| `sho_m52` (`SHO+Matern-5/2`) | 5 | **378** | 1932 | — | — | not in celerite2 |
| `rot_sho` (`Rotation+SHO`) | 6 | **461** | 1729 | 1402 | 200 | **0.43x** |

The kernel's cost grows steeply with `m` — it carries `3m^2+2m+1` floats
per thread — while celerite2's grows closer to `m`. So **the advantage
narrows sharply with state dimension and reverses at m = 6**: 11x the pool
at `m = 1`, 9-10x at `m = 2`, 1.4x at `m = 4`, and at `m = 6` the 12-core
pool is 2.3x *faster* than the kernel's forward. Two things compound
above `m = 4`: the element no longer fits in registers (m = 5-6 spill:
value + gradient ~8x m = 4 where the arithmetic says 3.4x), and the GPU
watchdog caps blocks at 128 and 64 points, so the boundary composes more
of them. m = 4 also moved, 20 -> 50 ms, since 2026-10-24: the square-root
boundary and the 256-point watchdog cap. What the kernel still offers at
m = 5-6: a gradient (celerite2 has none; anvil-gp's MLX path runs out of
Metal buffers from N ~ 16384), ~4x the MLX path's forward and ~10x its
value + gradient, and one 1024-chain launch. `m > 6` leaves the kernel
entirely. If your model is rotation + granulation and you sample with a
gradient-free ensemble, a celerite2 pool is the faster engine today.

## Accuracy against a float64 dense Cholesky

Absolute error in log L at N = 2,000, where an exact dense factorisation
is affordable. anvil-gp's production path is float32; celerite2 is float64
throughout. **The data is drawn on model** from each kernel: off-model
residuals are large and the quadratic form dominates, which flatters
everyone, while on model the log-determinant and the quadratic form very
nearly cancel and what survives is what a sampler sees.

| kernel | `m` | \|log L\| | anvil-gp kernel | anvil-gp MLX | celerite2 |
|---|---:|---:|---:|---:|---:|
| `real` | 1 | 1650 | 2.99e-05 | 2.74e-04 | 2.27e-13 |
| `sho` | 2 | 620 | 7.99e-05 | 2.86e-04 | 2.61e-12 |
| `matern32` | 2 | 567 | 4.79e-05 | 7.41e-05 | 1.30e-03 |
| `matern52` | 3 | 457 | 4.66e-05 | 9.01e-04 | not available |
| `rotation` | 4 | 660 | 9.82e-06 | 9.82e-06 | 2.61e-11 |
| `sho2` | 4 | 930 | 2.48e-05 | 1.47e-04 | 1.59e-11 |
| `sho_m52` | 5 | 641 | 4.12e-05 | 1.63e-04 | not available |
| `rot_sho` | 6 | 854 | 1.95e-04 | 6.84e-04 | 2.24e-11 |

The kernel's column is unchanged from 2026-10-24 to three figures although
its boundary changed since: these errors sit at float32's resolution of
log L itself (one ulp at |log L| = 620 is 6.1e-5), below anything the
boundary can move. Its accuracy gains show on bright, densely sampled data,
not here: docs/float32.md, section 7.

**celerite2 is seven to nine orders of magnitude more accurate** on every
kernel it implements exactly. That is float64 against float32 and it is
exactly what you should expect. The question is whether it matters: a
Metropolis accept/reject decision turns on differences of order 1, and
anvil-gp's error is 1e-5 to 1e-3 on a log-likelihood of several hundred.
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
while the fused kernel holds 0.002 (0.02 before its value moved to a
replayed filter, 2026-10-24). `benchmarks/error_scaling.py` maps it.

## What the general features cost

Measured by hand on a quiet machine (2026-10-25/26) rather than by
`bench_docs_tables.py`, against the plain likelihood on the fused kernel.
Each is recorded in more detail in
[general-features-plan.md](general-features-plan.md).

**A noise model** (`noise_fn`, or several per-instrument jitters) costs
almost nothing: the kernel reads one extra variance per observation and
writes one adjoint. At 1024 chains, m = 2: forward 1.04-1.10x, value +
gradient 1.05-1.06x; at m = 4, 1.00x. A single jitter keeps its cheaper
path and is bit-identical.

**Marginalised linear parameters** (`linear_basis`, p columns) need the
Gram matrix of the residual and the p basis columns, k = p + 1 in all.
The kernel carries all k through one filter (one covariance recursion, k
means); the MLX path, and the kernel above 32 columns, polarise instead,
one likelihood per pair. A basis shared by every chain, returned as
`(N, p)`, is read once by the kernel rather than per chain. Value +
gradient, N = 1e4, SHO, against the plain likelihood (quiet machine,
2026-10-27):

| p (k) | 1024 chains: shared basis | per-chain basis | polarised | 256 chains: shared | per-chain | polarised |
|---|---:|---:|---:|---:|---:|---:|
| 1 (2) | 1.5x | 1.7x | 3.1x | | | |
| 2 (3) | 1.7x | 1.9x | 5.1x | 1.4x | 1.4x | 2.3x |
| 5 (6) | 2.1x | 2.5x | 14.4x | 1.8x | 2.0x | 5.5x |
| 10 (11) | 5.5x | 6.3x | 42.5x | | | |

The forward alone at 1024 chains: 1.7x, 1.8x, 2.0x and 6.4x shared,
1.9x, 2.0x, 2.3x and 7.1x per chain. The residual and the basis go to the
filter as separate inputs, never concatenated, and a shared basis is
never expanded per chain; at N = 1e5 x 1024 chains (p = 2) that put the
value + gradient peak at 2.2 GB shared and 4.6 GB per chain, from 7.7 GB
for both before 2026-10-26 (docs/review-2026-10-26.md, E1 and E5/E6).
The white-noise centring fit and the p x p solves run on small Metal
kernels of their own, because MLX's batched matmul and inverse were
slower than the likelihood at these shapes. The one-filter route beat
polarisation at every k measured (up to 26), and polarising 26 columns at
m = 6 tripped the GPU watchdog. Reproduce with
`benchmarks/marginal_cost.py --chains 1024 --p 1 2 5 10 --grad`, adding
`--shared` for the shared basis (the defaults, 256 chains and p = 2, 5,
10, give the 256-chain columns).

**Prediction** (`predict`) is float64 on the CPU, off the likelihood
path: a Kalman filter and smoother over data and prediction points
together, vectorised over chains. SHO, 1000 prediction points: 0.21 s at
N = 1e4 and 2.0 s at N = 1e5 for one chain; 0.56 s and 5.0 s for 64
chains; 3.0 s for 512 at N = 1e4. The mean alone (`return_var=False`)
skips the covariance adjoint: 0.17 and 1.6 s for one chain, 3.7 s for 64
at N = 1e5. With a marginalised basis the linear parameters' posterior
comes from the same pass: 0.66 s for 64 chains at N = 1e4, 8.1 s at
1e5. It is bound by NumPy's per-call
overhead (~20 us per merged point), not arithmetic; a parallel-in-time
version, estimated ~10x at N = 1e5, is on the README's open list.
`apply_inverse` runs the same smoother without the variance: 0.46 s for
two columns, 64 chains, N = 1e4. `linear_posterior` is one forward pass
of the same float64 filter, no backward: 0.35 s for 64 draws and 1.7 s
for 512 at N = 1e4; 0.63 ms a draw at N = 2000, with memory flat at
~1.0 GiB from ~2500 draws up (16384 measured). (Before 2026-10-27 it
polarised the float64 likelihood: 3-5x slower, 0.40 GiB.) Measured on a
quiet machine, 2026-10-27.

## What is not benchmarked here

**tinygp** is not installed in this environment (it needs JAX, which is
not present), so there are no numbers for it. It would be the interesting
third comparison: it is also a JAX-based `O(N)` celerite implementation
with gradients, and on Apple Silicon it would run on CPU or through an
experimental Metal backend. Treat its absence as "not measured", not as a
claim.

**Gradients against celerite2** are not compared because celerite2 does
not provide them. anvil-gp's value and gradient cost about 2.3x the value
alone (10.1 ms at 1024 chains and N = 1e4) and matches float64 central differences to ~1e-7.
The honest comparison would be against tinygp or a JAX-wrapped celerite2.

**Prediction against celerite2.** Both have a `predict`; the two have
not been timed against each other. anvil-gp's (above) is float64 on the
CPU, not the batched, GPU-resident path the library is built around, and
matches the dense conditional to ~1e-14.

## Choosing between them

Use **celerite2** if you evaluate one parameter vector at a time, need
float64 accuracy in the likelihood itself, need a kernel anvil-gp does not
have (`TermProduct`), are not on Apple Silicon, or fit an m = 5-6 model
(rotation + granulation) with a gradient-free sampler and can run a
multi-core pool.

Exposure-time integration and `ComplexTerm` are no longer reasons to prefer
celerite2 — both landed in M5. On the convolution anvil-gp is now ahead on
two counts: it is six orders of magnitude more accurate at small `delta`
(2.2e-16 against 2.2e-10), and it convolves terms celerite2 cannot, namely
Matern-5/2, Matern-3/2 exactly, and an overdamped SHO. See
[kernels.md](kernels.md#termconvolution--exposure-time-integration).

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
