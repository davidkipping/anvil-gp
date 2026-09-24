# Benchmarks against celerite2

Regenerate every table here with one command:

```bash
.venv/bin/python benchmarks/bench_docs_tables.py
```

Measured on an M2 Max (12 CPU cores, 64 GB unified memory) on 2026-09-24,
in a window where a stability probe read 0.4% spread. Wall times on this
machine move by 2x under background load, so check
`benchmarks/verify_readme.py`'s probe before believing any timing you take
yourself.

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

| N | parameter vectors to break even vs 1 core |
|---|---:|
| 1,000 | ~26 |
| 2,000 | ~16 |
| 10,000 | ~5 |
| 100,000 | ~1.4 |

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
| 1,000 | 1 | **1.53** | 26 | 0.06 | 1.47 | 0.04x | 0.96x |
| 1,000 | 64 | **1.61** | 25 | 3.93 | 2.86 | 2.4x | 1.8x |
| 1,000 | 1024 | **1.61** | 26 | 63 | 11 | **39x** | 6.6x |
| 10,000 | 1 | **1.96** | 87 | 0.43 | 0.91 | 0.22x | 0.46x |
| 10,000 | 64 | **2.44** | 94 | 28 | 4.61 | **12x** | 1.9x |
| 10,000 | 1024 | **2.86** | 136 | 430 | 58 | **151x** | **20x** |
| 100,000 | 1 | **5.95** | 830 | 4.14 | 5.18 | 0.70x | 0.87x |
| 100,000 | 64 | **8.62** | 875 | 273 | 41 | **32x** | 4.7x |
| 100,000 | 1024 | **13** | 1307 | 4249 | 578 | **336x** | **46x** |

**Read the first row of each block first.** At one parameter vector,
celerite2 on a single core beats the fused kernel by 25x at N = 1000, by
4.5x at N = 1e4, and by 1.4x at N = 1e5. That is a fixed GPU launch cost
of roughly 1.5 ms that a single small problem cannot amortise. It is not a
defect to be tuned away; it is what a GPU is.

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
walkers is `S x W` log-likelihood evaluations, and the "parameter vectors"
column above is `W`.

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
| log-likelihood, per step | 1.86 ms | 12.4 ms |
| **anvil-gp, 614,400 evaluations** | **1.26 s** | **7.5 s** |
| of which sampler overhead | 0.15 s (12%) | 0.15 s (2%) |
| celerite2, 1 core | 38.6 s | 42 min |
| celerite2, 12-core pool (measured) | 6.6 s | 347 s |
| **speedup vs the pool** | **~5x** | **~46x** |

The N = 1,000 column is an end-to-end measurement with anvil's sampler;
the N = 100,000 column is the same 600 x 1024 evaluations timed on the
likelihood alone, plus the measured sampler overhead, since the sampler's
cost does not grow with N.

Two things to take from this.

**The end-to-end speedup is smaller than the kernel speedup**, because the
sampler itself costs something that no likelihood can speed up. At
N = 1,000 the likelihood is 88% of the wall time and the sampler is the
rest; drive the likelihood to zero and you still pay that 12%. The
kernel-only ratios in the tables above are an upper bound on what a real
fit will see.

**The advantage grows with N**, because celerite2's per-evaluation cost
grows linearly with the data while the fused kernel's barely moves between
1,000 and 10,000 points. At a thousand points anvil-gp is about five times
faster than a fully parallelised pool and you might reasonably not bother.
At a hundred thousand points the same run is 7.5 seconds against six
minutes on twelve cores, or 42 minutes on one.

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
throughout.

| kernel | `m` | \|log L\| | anvil-gp kernel | anvil-gp MLX | celerite2 |
|---|---:|---:|---:|---:|---:|
| `real` | 1 | 5208 | 3.81e-04 | 1.08e-04 | 3.64e-12 |
| `sho` | 2 | 9711 | 2.98e-04 | 2.98e-04 | 5.64e-11 |
| `matern32` | 2 | 9838 | 3.40e-04 | 1.32e-03 | **2.23e-02** |
| `matern52` | 3 | 10007 | 4.66e-04 | 4.66e-04 | not available |
| `rotation` | 4 | 9777 | 3.49e-03 | 4.14e-04 | 5.82e-11 |
| `sho2` | 4 | 8896 | 8.45e-04 | 8.45e-04 | 2.73e-11 |

**celerite2 is seven orders of magnitude more accurate**, everywhere
except Matern-3/2. That is float64 against float32 and it is exactly what
you should expect. The question is whether it matters: a Metropolis
accept/reject decision turns on differences of order 1, and anvil-gp's
error is 3e-4 on a log-likelihood of ~1e4. It does not matter for MCMC. It
would matter if you needed the absolute value of the evidence to better
than 1e-3, and for that anvil-gp ships a float64 path (`gp.hi(...)`) that
agrees with the dense oracle to ~1e-11.

**The exception is Matern-3/2, where anvil-gp is 65x more accurate.**
celerite2 represents Matern-3/2 by a rational approximation; anvil-gp uses
the exact two-state model (it is the SHO at critical damping). Across four
`(sigma, rho)` settings celerite2's error ran 2.6e-3 to 8.2e-2 while
anvil-gp held 4.7e-5 to 7.2e-4.

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
float64 accuracy in the likelihood itself, need exposure-time integration
(`TermConvolution`), need `ComplexTerm`, or are not on Apple Silicon.

Use **anvil-gp** if you run an ensemble sampler with hundreds or thousands
of walkers, want gradients for HMC without writing them yourself, need
Matern-5/2, or need Matern-3/2 done exactly.

The two agree on the likelihood to float32 precision, so moving between
them is a change of engine, not of model — the parity tests in
`tests/test_likelihood.py` pin that across every shared kernel, duplicate
timestamps, gaps, jitter and mean models.
