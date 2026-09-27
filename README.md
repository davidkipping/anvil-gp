# anvil-gp

Batched Gaussian-process likelihoods for the [anvil](../anvil) MCMC engine on
Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx).

**The premise.** Time-series GP likelihoods of the celerite class evaluated
for *thousands of parameter vectors per call* on the Apple GPU, with
gradients, so that anvil samples **GP hyperparameters jointly with mean-model
parameters**. Joint sampling is the requirement, not a nicety: designs that
need hyperparameters frozen or pre-fit are out of scope and do not shape the
architecture.

```python
import anvil, anvilgp as ag
from anvil.kernels.ensemble import EnsembleKernel
from anvilgp.anvil import make_target

from anvilgp import GPPolicy

# metal=True turns on the fused kernel: ~45-75x on the forward and ~100x on
# the gradient, so it suits the stretch move below and ChEES-HMC alike.  It
# is off by default; `gt.loglike.metal_eligible()` says whether it can be honoured
# (m <= 4, one jitter), and a request it cannot honour warns rather than
# falling back in silence.
gt = make_target(ag.SHOTerm(), t, y, yerr,      # defaults, not homework
                 policy=GPPolicy(metal=True))
print(anvil.validate_precision(gt.target, gt.u0(32)))   # trust check first

u0 = gt.u0(256)
res = anvil.run(EnsembleKernel(gt.target), gt.target, u0,
                n_warmup=3000, n_samples=600, thin=3)
posterior = gt.natural(gt.transform.model_np(res.get_chain(flat=True)))
```

`make_target` picks the parametrisation, the float64 host references every
parameter is an offset from, proper default priors, and a jitter — see
[Parametrisation](#parametrisation-and-priors).

## Documentation

| | |
|---|---|
| [docs/tutorial.md](docs/tutorial.md) | Getting started, side by side with celerite2 |
| [docs/kernels.md](docs/kernels.md) | Every supported kernel, how to set it up, and what is not supported |
| [docs/benchmarks.md](docs/benchmarks.md) | Speed and accuracy against celerite2 across scales and kernels — including where celerite2 wins |
| [docs/m5-plan.md](docs/m5-plan.md) | Planned next: per-instrument jitter in the kernel, exposure-time integration, and the compile split |
| [docs/float32.md](docs/float32.md) | The precision discipline and its measured error budget |
| [docs/statistical-foundation.md](docs/statistical-foundation.md) | Why the state-space form is exact |

**The one-line summary of the comparison:** celerite2 is faster for a
single likelihood and seven orders more accurate in absolute terms;
anvil-gp is tens of times a 12-core celerite2 pool once you need hundreds
of *hyperparameter settings* scored at once — an ensemble sampler's walker
count, not the data size — carries its own gradient, and has Matern-5/2,
which celerite2 lacks. Break-even is about 26 walkers at
N = 1000 and about 1.4 at N = 1e5. The exact multiple depends on the
cadence: 35-42x on a regular grid (the milestone-3 figures below) and
20-46x on a gapped one (docs/benchmarks.md), because a gap adds a lag-table
row and costs the kernel ~1.5-1.7x. End to end inside a sampler it is
lower again; benchmarks.md says by how much and why.

## Why a state-space model rather than celerite's Cholesky

Every celerite kernel *is* a linear Gaussian state-space model (the CARMA
correspondence). The Kalman prediction-error decomposition and the celerite
semiseparable Cholesky compute the same marginal likelihood in exact
arithmetic, so the choice is purely numerics and GPU scheduling. The SSM route
buys two things:

* **A propagator that is smooth through Q = 1/2.** celerite's coefficient
  parametrisation carries a `1/sqrt(4Q^2-1)`, singular at critical damping;
  a 4096-chain warmup crosses Q = 1/2 routinely. Here the propagator is built
  from `cos(d dt)` and `sin(d dt)/d`, entire functions of `u = d^2 dt^2`
  alone, which continue analytically from underdamped through critical into
  overdamped. `d` never appears by itself.
* **Exact Matérn-3/2 and Matérn-5/2**, neither celerite-representable.
  Matérn-3/2 needs no code at all: it *is* the SHO at Q = 1/2.

Both show up in the parity table below as places where **celerite2** is the
approximate one.

## Layout

| module | role |
|---|---|
| `special.py` | cancellation-free scalar functions; true float64 `exp`/`sincos` |
| `terms.py` | kernel terms as SSMs — propagator, stationary covariance, process noise |
| `linalg.py` | tiny batched Cholesky / triangular solve, unrolled and differentiable |
| `elements.py` | affine-Gaussian scan-element algebra; chunked-scan **reduction** |
| `kalman.py` | sequential prediction-error filter — an independent oracle |
| `likelihood.py` | `GPLogLike`: the anvil engine contract |
| `priors.py` | parametrisation, default priors, periodogram initialiser |
| `oracle.py` | dense float64 Cholesky oracle, `predict`, celerite2 wrapper |
| `metal.py` | fused Metal scan kernel and its checkpointed backward — M3 and M4 |
| `adjoint.py` | hand-derived backward recursion — M4's specification |
| `anvil.py` | engine integration (lazy import; core stays mlx + numpy only) |

Kernel set: `RealTerm`, `SHOTerm`, `Matern32Term`, `Matern52Term`,
`RotationTerm` (two SHOs at P and P/2, the celerite2/exoplanet convention),
sums of terms, per-instrument jitters and mean offsets.

## The likelihood is a reduction, not a scan

An element is what one stretch of the series does to the state entering it —
an affine-Gaussian transition times the stretch's observation likelihood as a
function of that state. Composition is a Gaussian integral, and it is
associative.

The marginal likelihood needs only the **root** of a binary reduction tree,
never the prefix at every leaf: the first element has `A = J = eta = 0`, so
those stay zero all the way up, and **log L is the root element's
log-normaliser**. That halves the working set against a prefix scan and makes
the float32 accumulation of the log-determinant and the quadratic form a
pairwise tree.

A leaf's information matrix is **rank one**, so composing with a leaf
collapses under Sherman–Morrison — no Cholesky, no triangular solve, and the
log-normaliser reduces to the Kalman prediction-error form. `chunked_reduce`
exploits that: fold sub-blocks of 16 observations sequentially with the cheap
composition, then tree-reduce only the sub-block roots.

One algebra, three executors: run the same code float64-on-CPU and it is the
verification path; float32-on-GPU and it is the parity baseline; and the fused
Metal kernel executes the *same* composition on the same block boundaries —
its inner loop is precisely `combine_leaf` folded in registers.

Full derivation, including the closed-form combine: [docs/statistical-foundation.md](docs/statistical-foundation.md).

## Verification

Absolute log-likelihood (not differential — a wrong constant is invisible to
MCMC), N = 512, against **two** independent oracles:

| case | vs celerite2 | vs dense float64 Cholesky |
|---|---:|---:|
| SHO, Q ∈ {0.1, 0.4, 1/√2, 3, 30, 300} | < 5e-12 | < 8e-12 |
| Real, sum of terms, RotationTerm | < 6e-12 | < 1e-11 |
| SHO + jitter | 1e-13 | — |
| duplicate timestamps | 0 | — |
| **SHO at Q = 1/2** | **4.4e-04** | **2.5e-12** |
| **SHO at Q = 1/2 + 1e-6** | **2.6e-04** | **1.6e-12** |
| **Matérn-3/2** | **6.5e-03** | **1.5e-11** |
| **Matérn-5/2** | not representable | **7.5e-12** |

The four bold rows are corners where celerite2 itself is approximate: its
`SHOTerm` carries an `eps` to dodge the critical-damping singularity, and its
`Matern32Term` takes an explicit `eps` approximation parameter. A test asserts
that celerite2's Matérn-3/2 *converges to ours* as its `eps` shrinks
(6.5e-3 → 6.5e-7 → 1.9e-10), which is what fixes the direction of the
discrepancy.

Beyond parity, the suite pins (the full suite runs in ~4.5 minutes, ~2.5
with `-m 'not slow'`):

* the SSM identity `k(tau) = h Phi(tau) h^T` and the Lyapunov consistency
  `Q = I - Phi Phi^T` for all twelve term configurations;
* the composed element against a **brute-force numerical Gaussian integral**
  as a function of `(x_out, x_in)` — the only check that can see a wrong
  log-normaliser, which is otherwise a constant offset that every relative
  test passes;
* tree reduction == chunked scan == left fold == sequential Kalman filter,
  values *and* gradients, at every sub-block length;
* the **Metal kernel** against the dense float64 oracle and the MLX path, for
  every term type, every block length from 1 to 1024, duplicate timestamps,
  gaps, irregular cadence, a jitter and a mean model (`tests/test_metal.py`);
* the **backward recursion** against `mx.vjp` *and* float64 central
  differences, and its blocked form against its whole-series form
  (`tests/test_adjoint.py`) — the kernel cannot be checked against autodiff,
  which is exactly why it needs a specification that can;
* the rank-one `combine_leaf` against the general `combine`, to 1e-12;
* combine's identity element and associativity;
* the special functions against mpmath, *and* against the naive forms they
  replace;
* float32: value within 5.9e-4 of a dense float64 Cholesky at N = 1024;
  gradient within 0.14% of float64 central differences, uniformly across
  components (i.e. rounding, not a defect);
* `anvil.validate_precision` on the assembled target: **OK**, median
  float32 error 1.4e-4 against the ~1-unit Metropolis decision scale;
* **unbiasedness across realisations** (`tests/test_calibration.py`): the
  profile maximum of σ_GP over 12 fresh draws is **1.500 ± 0.135** against a
  truth of 1.5, with ρ and the jitter likewise unbiased. One recovery at one
  truth has almost no power against a subtle likelihood bias — a single
  draw's realised RMS scatters by tens of percent when the baseline holds
  only a handful of correlation times — so the tests average instead. This
  is the cheap version of M5's SBC work, not a substitute for it.

### Float32 error scaling, decomposed

`benchmarks/error_scaling.py` separates the log-determinant from the
quadratic form (evaluating at `y = 0` zeroes the latter exactly) and scans
the three axes that could compound:

| scan | range | worst absolute | worst **relative** |
|---|---|---:|---:|
| N | 1e3 → 3.2e4 | 1.5e-2 | 8.0e-07 |
| `rho/dt` | 2 → 4e4 (`rho/T` from 3e-4 to 5) | 2.3e-3 | 4.8e-07 |
| `sigma_GP/yerr` | 0.1 → 1000 | 3.4e-2 | 2.2e-06 |

The relative error is flat at a few times float32 eps everywhere: the
recursion does not compound, and what grows with N is `|logL|` itself.
Extrapolated, N = 1e5 gives ~7e-2 log-units (an order of magnitude under the
decision scale) and N = 1e6 gives ~0.7 — which is where `reanchor_every`, or
composing the per-block roots in float64, starts to matter.

One notable negative result: the brief predicted a float32-infeasible corner
at `sigma_GP/yerr` of 10–300 with a timescale far above the cadence. Across
that whole range the worst relative error measured is 4.3e-7, with no sign of
the innovation-variance cancellation — the cancellation-free process noise
appears to have removed the hazard. That is one slice, not the map; M3 still
owes the full feasibility survey.

## Performance: the fused Metal kernel

**The kernel is in, and differentiable** (`anvilgp/metal.py`). It walks a
block of observations with the running scan element **in registers**, so one
element per *block* reaches memory instead of one per point, and M4's
checkpointed backward sweep gives it its own VJP. Against the MLX path,
same problem, same answers:

| config | MLX path | **Metal** | speedup | peak MLX | peak Metal |
|---|---:|---:|---:|---:|---:|
| N=1e4, 1024 chains | 75.1 ms | **1.67 ms** | 45x | 1.20 GiB | 0.04 GiB |
| N=1e4, 4096 chains | 326 ms | **4.40 ms** | 74x | 3.74 GiB | 0.16 GiB |
| N=1e5, 1024 chains | 719 ms | **11.5 ms** | 63x | 0.82 GiB | 0.39 GiB |

(Minimum of 12 sustained evaluations, compiled, gapped cadence with three
distinct lags, `block=4096`, `metal_block` at its default, all rows in one
window on 2026-09-20. `benchmarks/verify_readme.py` re-checks every number
here; it gates on a measured stability probe rather than on the load
average, which is a poor proxy — a CPU-bound indexer can sit at 14 while
the GPU is steady to 0.6%.) The speedups are smaller than the
37-71x first reported here because the MLX path itself got faster in the
meantime, twice — see "Unroll the leaf reductions" and the sub_block change
in the history. The kernel is unchanged.)

Against celerite2 on the CPU, measured in the same session:

| config | vs 1 core | vs 12 cores |
|---|---:|---:|
| N=1e4, 1024 chains | **255x faster** | **35x faster** |
| N=1e5, 1024 chains | **376x faster** | **42x faster** |

**The >=100x gate is not yet cleared.** The brief is explicit that "the honest
>=100x denominator is the multiprocess one", and against a fully parallelised
12-core celerite2 we are **35-42x**, not 100x. The single-core column is
reported because it is the comparison most users' code actually makes (one
CPU process, one chain at a time), not because it clears the gate. What the
kernel did change: the 12-core pool was 2x *ahead* of us before it, and is now
20-30x behind. Closing the remaining factor is future work; see
"What is left" below.

The brief predicted "~10-50 ms forward at 1024 x 1e5"; the measurement is
16.4 ms, at the fast end of its own range.

End to end, `examples/gp_injection_recovery.py` (1000 points, 1024 chains,
5800 stretch steps) goes from **91 s to 15.7 s**, 226 -> **1,300 ESS/s**,
with the posteriors agreeing to three decimals. Run it either way:

```bash
.venv/bin/python examples/gp_injection_recovery.py           # MLX graph
METAL=1 .venv/bin/python examples/gp_injection_recovery.py   # fused kernel
```


### Gradients

M4 landed: the kernel carries its own VJP, so a gradient sampler can use it.

| config | MLX path | **Metal** | speedup | peak MLX | peak Metal |
|---|---:|---:|---:|---:|---:|
| value+grad, N=1e4, 1024 chains | 884 ms | **8.9 ms** | 99x | 6.10 GiB | 0.17 GiB |
| value+grad, N=1e5, 1024 chains | 15.4 s | **58 ms** | 264x | 34.5 GiB | 1.61 GiB |

The gradient matches float64 central differences to ~1e-7 on every
parameter including the jitter, and the two paths agree with each other to
~1e-6 relative.

End to end, ChEES-HMC on the injection-recovery problem (1500 points, 64
chains, 300 iterations) runs in **4.0 s on the kernel against 126 s on the
MLX graph**, and samples better for it: minimum ESS 239 against 94.

Three passes, all with the forward's launch shape because the backward
blocks are **independent**: the forward kernel emits one element per block;
one thread per chain folds those into each block's entry state and its
seed, in closed form; and one thread per (chain, block) replays its block
from checkpoints and sweeps back. The seed is the part that removes the
chain -- see [docs/m4-plan.md](docs/m4-plan.md).

### What the kernel does and does not do

It runs the composition, not the kernel-term algebra. `Phi` and `Q` arrive as
a **table indexed by lag**, built by the float64-parity code in `terms.py`.
Two reasons: the cancellation-free special functions are the most delicate
numerics in the package and are pinned against mpmath, so re-deriving them in
Metal Shading Language would put the thing most likely to be silently wrong
in the place hardest to test; and real photometry has one cadence plus one lag
per gap, so the table is a handful of rows that lives in cache.

The first observation has no predecessor, and the *table* says so rather than
the kernel special-casing it: row 0 holds `Phi = 0`, `Q = P_inf = I`, and
`lag_index[0] = 0`. `ell` is accumulated with Kahan compensation in-thread,
since a block is up to 4096 sequential adds with no tree to spread the error.

Limits, stated: m <= 4 (the running element is `3m^2+2m+1` floats in
registers — 17 at m = 2, 57 at m = 4, 209 at m = 8, which would spill);
one jitter, not per-instrument; and an all-distinct-lag cadence degenerates
the table to one row per point, which is correct but gives up the traffic
saving — the value costs 13.2 GiB at 1024 chains x N = 1e5 where a regular
cadence costs 0.4. The gradient handles it: its adjoint buffer holds one
slot per `(block, lag)` pair that occurs, so it is bounded by N rather than
by N x blocks, and that case runs at 1.3x the forward's peak. Larger m and
in-kernel propagators are the obvious follow-ons.

### The MLX reference path

Still the float64 verification path, the thing the kernel is proved against,
and the only path for m > 4 or per-instrument jitter. It is ~40x faster than its first working
version and beats a single-core CPU, though not a fully parallelised one.
M2 Max, float32 GPU, compiled forward, **minimum of 15** sustained
evaluations, `block=4096`, realistic (regular-cadence-with-gaps) times, all
rows in one window:

| config | first version | now | vs celerite2, 1 core | vs celerite2, 12 cores |
|---|---:|---:|---:|---:|
| N=1e4, 1024 chains, m=2 (SHO) | 3428 ms | **73.8 ms** | 5.8x faster | 1.3x slower |
| N=1e5, 1024 chains, m=2 | — | **693 ms** | 6.2x faster | 1.4x slower |
| N=1e5, 256 chains, m=2 | 8719 ms | **351 ms** | 3.1x faster | 2.9x slower |
| N=1e4, 4096 chains, m=2 | — | **326 ms** | 5.4x faster | 1.4x slower |
| N=1e4, 1024 chains, m=4 (rotation) | — | **276 ms** | — | — |
| N=1e4, 1024 chains, m=8 (4 SHO) | — | **942 ms** | — | — |

celerite2 fp64 on this machine, measured in the same session
(`benchmarks/bench_vs_celerite2.py`): 0.428 ms per evaluation at N=1e4 and
4.26 ms at N=1e5, so 1024 parameter vectors cost 426 ms and 4327 ms in a
single-core loop. A 12-core `multiprocessing` pool gets 7.2-8.9x of the
theoretical 12x (60-74% efficiency), giving 59 ms and 486 ms.

**So: yes, a fully parallelised CPU celerite2 is still ~1.4x faster than the
MLX reference path.** Against a single core we are ~6x faster. Both
comparisons are against exact float64 on the CPU versus float32 on the GPU
with a separate float64 verification path. The >=100x gate is Milestone 3's,
and the multiprocess figure is its denominator.

### A note on measuring this machine

Wall times here move by 2x with machine state, and it is *not* thermal
throttling. Diagnostics:

* 80 consecutive evaluations under sustained load: 110.5 ms, flat, min 107.5
  and max 112.5 — no warm-up ramp, no decay. So the steady state an MCMC run
  actually sees is the fast one. (That was the pre-unroll code, which then
  took 112 ms; the point is the flatness, not the absolute number.)
* Six isolated processes, rested, 45 s cooldowns, in one bad window: 218,
  225, 229, 232, 223, 233 ms — consistently 2x slow, and tight within the
  window.
* Same command in a quiet window: 109.7, 110.6, 110.4 ms.
* Throughout: AC power, low-power mode off, `pmset -g therm` reporting no
  thermal or performance warning.

The differences track background load (WebKit, iCloud, Spotlight), which only
ever adds time. So the benchmark reports the **minimum** alongside the median,
and the numbers above are from one quiet session. Treat a 2x discrepancy
against these as a busy machine before treating it as a regression -- that
mistake cost real time here.

Three changes got the 31x, in order of size:

1. **Never `mx.matmul` a tiny matrix — but do use it once they are not so
   tiny (~7x at m=2, 1.5x at m=8).** The composition does ~13 m-by-m
   products. A batched GEMM over (256, 512) two-by-two matrices measures
   4.1 ms, while the same product written as elementwise multiply-adds over
   full-size planes measures 0.62 ms: the GEMM is all per-matrix overhead at
   that size, and it is opaque to the fusion pass. After the switch,
   `mx.compile` went from buying 1.05x to buying 2.4x.

   The unrolled form is m^3 elementwise ops while `mx.matmul`'s cost is
   essentially **flat in m**, so the ranking inverts. Measured ratio
   (unrolled/GEMM, both compiled, batch (1024, 256)):

   | m | 2 | 4 | 6 | 8 | 10 |
   |---|---:|---:|---:|---:|---:|
   | ratio | 0.03 | 0.19 | 0.89 | 2.20 | 4.30 |

   `linalg.UNROLL_MAX_M = 6` puts the switch where the measurement does; the
   crossover sat between 6 and 8 at every batch shape tried. This is worth
   1.5x on a four-term kernel and nothing at all on a single SHO.
2. **Chunked scan instead of a plain binary tree (1.7x).** A leaf's
   information matrix is rank one, so composing *with a leaf* collapses under
   Sherman–Morrison: no Cholesky, no triangular solve, three matrix products
   instead of thirteen (`combine_leaf`). `chunked_reduce` folds sub-blocks of
   16 observations that way and tree-reduces only the sub-block roots, cutting
   general compositions per chain from `K` to `K/16`. It is also exactly the
   structure M3's kernel runs, so it de-risked the kernel too.

   The general composition costs ~13 m^3 against the fold's ~3 m^3, so the
   best sub-block climbs with the state dimension — measured 16 at m = 2 and
   4, 32 at m = 6, 64 at m = 8. `GPPolicy.sub_block=None` (the default) picks
   from `term.n_state` so this is not a tuning trap.
3. **Reuse Phi and Q across repeated lags (1.3x).** `dt` is *data*, and real
   photometry has one cadence plus one lag per gap. Grouping is done **per
   dtype**, which is the only way it stays exact: `arange(n)*0.02` has 15
   distinct float64 diffs from representation noise alone, and exactly one in
   float32. The production path groups the float32 lags it will actually
   evaluate, the verification path groups the float64 ones, and neither needs
   a tolerance.

### Multi-term kernels

A rotation term is m = 4, rotation plus granulation is m = 6, four SHOs is
m = 8 — and cost grows roughly as m^2.5, so the multi-term case needs its own
attention. Three things were measured, and two of them contradicted the
obvious guess:

**The leaf array is the memory.** A dense leaf holds `3m^2 + 2m + 1` numbers
per (chain, point); at 1024 chains x 4096 points x m = 10 that is 5 GiB
of leaf array, and ~12 GiB peak once the composition's intermediates are
counted.
Storing the *pieces* instead — `Phi` and `Q` as blocks, plus
`kappa = Q h^T/sqrt(S)`, `z = (h Phi)^T/sqrt(S)`, `g = y/sqrt(S)` — costs
`2 sum m_i^2 + 2m + 2`, and every component is recoverable
(`A = Phi - kappa z^T`, `C = Q - kappa kappa^T`, `J = z z^T`, `b = kappa g`,
`eta = g z`). `combine_leaf_blocked` substitutes those identities so `Phi`'s
zeros are never multiplied and `A` is never assembled. Measured, A/B in one
process on a rested machine, 1024 chains x N = 1e4:

| m | 2 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|
| blocked speedup | 1.01x | 0.80x | 1.16x | 1.11x | 1.44x |
| peak, dense | 1.16 | 2.48 | 4.57 | 7.67 | 11.93 GiB |
| peak, blocked | 1.01 | 1.58 | 1.89 | 2.03 | 2.36 GiB |

The speed effect is modest and thermally sensitive (the same A/B heat-soaked
put m = 8 at 2.36x); the **memory** effect is large, monotone and
reproducible — 5x at m = 10. Default is blocked for m >= 6 with two or more
blocks; set `GPPolicy.blocked=True` at m = 4 if memory rather than time is
what binds.

A caution worth recording: an *eager* microbenchmark of a single composition
ranked these the other way round (1.6x for blocked at m = 4, 0.6x at m = 8)
and would have led to exactly the wrong default. What the compact form buys
is traffic on the leaf array, which only appears in the compiled
whole-pipeline measurement.

**The unroll threshold is on work, not on size.** `linalg.matmul` unrolls
below `UNROLL_MAX_WORK = 6^3` multiply-adds. Keying it on `max(dimension)`
instead — the obvious first guess — sent the block products `(2x2) @ (2x8)`,
all of 32 multiply-adds, to `mx.matmul`, and made the block-structured path
2.6x *slower* than the dense one it was meant to beat.

**Block-diagonality does not reach the compositions.** The tempting claim is
"`m^3` instead of `sum m_i^3`", but a leaf's `A = Phi - k(h Phi)` already has
a rank-one correction that couples every block, and after one composition
`A`, `C` and `J` are genuinely dense. Profiling a four-SHO kernel puts only
11% of the runtime in the term matrices and 89% in the compositions. What was
worth taking there was small: `_block_diag` was folding pairwise, rewriting
the growing left-hand block once per term (48 ms for four 2x2 blocks), and now
places every block against the final width in one pass.

### Reproducing these numbers

```bash
.venv/bin/python benchmarks/bench_vs_celerite2.py --n-data 10000 --n-param 1024
.venv/bin/python benchmarks/bench_reference.py --n-data 10000 --n-chains 1024 \
    --block 4096 --cadence gapped --repeat 25
```

`bench_reference.py` also takes `--grad`, `--term {sho,rotation,sum2,sum3,sum4}`,
`--sub-block`, `--no-compile` and `--cadence {regular,gapped,random}`.

Note its defaults (256 chains, `block=256`, random cadence) are *not* the
target workload, so pass the flags above when quoting a number. The
`--block` default in particular is worth 3.2x on the gradient path.

The Metal rows come from the same script with `metal=True` in the policy;
all the kernel timings in this README are the minimum of 7 sustained
evaluations at `metal_block=1024` (N=1e4) and 4096 (N=1e5), compiled, on a
gapped cadence with three distinct lags, on an M2 Max. Read
[A note on measuring this machine](#a-note-on-measuring-this-machine)
before comparing against your own.

### Practical guidance

**Add chains before you subtract them.** At N = 1000 the compiled likelihood
costs 7.3 ms at 256 chains and 11.5 ms at 1024 — four times the chains for 1.6
times the time. Small problems are bound by op count, not by data, so the GPU
is idle most of the step and extra chains ride along nearly free. Use
`block = 4096` and the default `sub_block = 16` unless you are below
N ~ 2000, where a smaller `sub_block` shaves a few sequential stages.

**The `GPPolicy` knobs**, and when to touch them:

| field | default | change it when |
|---|---|---|
| `metal` | `False` | you want the fused kernel: ~45-75x on the forward, ~100x on the gradient. `m <= 4`, one jitter; the gradient also needs few distinct timestamp spacings (see "What is left"). Warns if it cannot be honoured. |
| `metal_block` | `None` (N/12, 1024-8192) | essentially never: the rule picks the measured best at N = 1e4, 3e4 and 1e5. |
| `block` | `1024` | raise to 4096 for speed (3.2x on the gradient path); lower it if the gradient's peak memory does not fit. Accuracy no longer depends on it. |
| `sub_block` | `None` (16 at m<=4) | below N ~ 2000, try 8. Above m = 6 the default already climbs to 32/64. |
| `eval_blocks` | `False` | you are memory-bound and can give up `mx.compile`. The `.hi` path sets it itself. |
| `blocked` | `None` (measured) | you have profiled both and disagree with `use_blocked_leaves`. |
| `hi_chain_block` / `hi_block` | 16 / 1024 | the float64 reference pass runs out of memory. |
| `s_floor_rel` | `1e-6` | essentially never: it defines what a positive-definiteness failure means. |

### What is left, and what it is worth

Both kernels landed (M3 forward, M4 backward); what follows is what is
still open, with what each is worth. Measurements are from the review of 2026-09-19 unless
marked as an estimate.

* ~~**The kernel gradient on irregular cadences.**~~ **Done.** Where
  every timestamp spacing is distinct the lag table degenerates to one row
  per observation, and two things scaled with N: MLX's backward through
  the term front end (75 GiB at 1024 chains x N = 1e5, against 0.009 GiB
  on a regular cadence) and the kernel's own adjoint buffer (39.7 GiB).
  The front end's backward is now split over the lag axis when the table
  is large, and the buffer holds one slot per `(block, lag)` pair that
  *occurs* rather than one per pair possible. That case needed ~115 GiB
  and was refused; it now runs at 17.5 GiB, which is 1.3x the forward's
  own peak. At 256 chains it went 31.3 GiB and 11.4 s to 4.4 GiB and
  1.5 s. Regular cadences are bit-identical and untouched. Reviewed
  adversarially afterwards (stage R2): the arithmetic held under ~400
  configurations, and the five defects found were all in the contracts
  between components — chiefly that assigning `metal_block` on a built
  instance left the kernel's slot map stale, giving a correct value with
  a wrong gradient; it is a property that rebuilds the map now. See
  [docs/lag-slots-plan.md](docs/lag-slots-plan.md); an earlier proposal
  here, per-parameter Jacobian tables, was measured and retracted.
* **m > 4 and per-instrument jitter in the kernel.** Both stay on the MLX
  gradient path, where a gradient costs 8.8 s at m = 6, 256 chains and
  N = 1e4.
* **Block count as an accuracy knob for one adjoint.** The backward seed
  amplifies block-element error through `eta_s - J_s mu`, which reaches
  only `dh` (the amplitude gradient). At the joint corner (`sigma/yerr =
  100`, `rho/dt = 1000`) that adjoint is off by 1.4e-4 at one block, 8e-3
  at 8 and 1.9e-2 at the default's 16, against 0.37 on the MLX path. Every
  other parameter stays under 1.1e-4. Rounding `metal_block` up rather than
  to the nearest power of two would halve the blocks and cost forward
  speed; not done.
* ~~**Kernel thread and residual layout (measured 1.5–1.9x).**~~ Landed
  with M4's stage B: flat chains-fastest grid, residual stored
  `(npts, n_chains)`, `ell` bit-identical.
* **Bit-reversed leaf permutation (measured, MLX path).** Storing each
  block's leaves in fold-step order makes every fold step a contiguous slab
  and every tree level a contiguous halving. Measured 85 -> 55 ms forward
  at N = 1e4 over the already-unrolled code, and 1.15x on the gradient.
  Deferred: it rewrites the reduction's index plumbing, which is where the
  `_pairs` workaround for MLX's strided-slice VJP bug lives, and M4
  replaces the path it optimises.
* **Compile the gradient by splitting the VJP at the term-matrix table.**
  `mx.compile` over the whole backward pass fails, but bisection shows the
  element algebra is not the cause — `chol`, `combine`, `combine_leaf` and
  the whole leaf → fold → tree chain each compile under `mx.vjp`, as does
  `(phi_table, q_table, h) → ell` end to end. The offending fused group is
  inside `terms._sho_matrices`. Running the front end eager and the rest
  compiled measured 1.15-1.25x on the gradient; worth more in a quiet
  window. M4 did not moot it: the over-fused group is in the front end
  both paths share, so this is still what would let ChEES run without
  `mx.disable_compile()`.
* **Per-block gradient checkpointing: measured, and not taken.**
  `mx.checkpoint` around each block's `leaf_inputs -> chunked_reduce` trades
  time for the leaf arrays the backward pass would otherwise hold. Measured
  at 512 chains x N = 3e4: peak 6.26 -> 2.61 GiB for 1559 -> 1901 ms, i.e.
  2.4x the memory for 22% of the time. `block` already gives a coarser
  memory lever at a similar price, and M4 replaces this path, so it is not
  worth a policy knob. Recorded because the obvious implementation is
  **wrong**: build the `get_mat` closure outside the checkpointed function
  and MLX treats the term matrices as constants, so the gradient silently
  loses the whole kernel-parameter path — measured 337 absolute against a
  correct gradient, with the *value* still bit-identical. Build it inside.
* **Symmetric storage for C and J.** 15 numbers instead of 17 at m = 2, 45
  instead of 57 at m = 4. Marginal in MLX, and immaterial in the kernel
  (block outputs are 0.05 GiB) — kept here only so nobody re-derives it.
* **Not worth doing:** `metal::fast` intrinsics buy nothing once the layout
  above is fixed (1.35 -> 1.19 ms) and cost up to 6e-2 log-units. A
  whole-series fold with a grouped tree measured *slower* (114–122 vs
  111 ms) and used 1.75 GiB more.

### End to end

`examples/gp_injection_recovery.py` fits an SHO kernel plus a jitter to 1000
points with **1024 chains**, 5800 stretch steps, in **15.7 s** on the fused
kernel (1,300 ESS/s) or **91 s** on the MLX graph (226 ESS/s), against
684 s for the same example before any of this work (at 256 chains). Both paths recover all four parameters,
and now agree to three decimals on every one:

| parameter | Metal | MLX | truth | z |
|---|---:|---:|---:|---:|
| σ_GP | 1.139 ± 0.285 | 1.138 ± 0.285 | 1.5 | −1.27 |
| ρ | 6.950 ± 0.480 | 6.951 ± 0.481 | 7.3 | −0.72 |
| Q | 6.14 ± 5.49 | 6.11 ± 5.67 | 8.0 | −0.33 |
| jitter | 0.1133 ± 0.0457 | 0.1133 ± 0.0456 | 0.12 | −0.15 |

**The jitter width in that table is the sampler's, not the likelihood's,
and it is wrong.** Stage R of M4 checked it. A float64 Laplace
approximation at the mode gives a half-width of 0.0107; ChEES-HMC on the
kernel, 1024 chains, converged (R-hat 1.0003, min ESS 386,000), gives
0.1113 ± 0.0110. The stretch move's ± 0.0457 is two of its 1024 chains:
they sit in a secondary mode (jitter 1.13, ρ shorter than the cadence, log
posterior 1,500 units below the bulk) that the stretch move cannot leave,
because its proposals reach at most halfway to the bulk and halfway is the
bottom of a 1,000-unit valley. The other 1,022 chains give ± 0.0109, and
the ρ width is inflated the same way (0.482 against ChEES's 0.376). The two
columns agree with each other because they are the same likelihood under
the same sampler with the same seed. Written up for the anvil side in
[docs/anvil-stretch-width.md](docs/anvil-stretch-width.md); quote the
ChEES posterior for the widths.

min ESS 20,422 (Metal) and 20,683 (MLX), max R-hat 1.037 on both. The MLX
column is worth noting: before the float32 work in the reduction it gave
Q = 6.51 ± 13.6 and min ESS 17,120 — a posterior width 2.5x the kernel's,
from rounding rather than from anything statistical. σ_GP sits low because *this draw's* realised
RMS is 1.118 against a process value of 1.518 — with ~5.5 correlation times in
the baseline a single realisation scatters that much, and the posterior is
correctly tracking the data. `tests/test_calibration.py` is what establishes
that the estimator itself is unbiased.

## Parametrisation and priors

Shipped as defaults, because each of these costs throughput rather than
announcing itself as an error:

* Sample **(log σ, log ρ, log Q)** per SHO term, never `(S0, ω0, Q)`: a
  high-Q posterior is an axis-aligned box in the first and a curved ridge in
  the second, and a diagonal preconditioner cannot help with a curved ridge.
  Granulation fixes `Q = 1/√2`.
* Every log-parameter is an **offset from a float64 host reference** taken
  from the data (RMS of y, median error, median cadence): `v = 0` is a
  sensible guess, which is where float32 has the most room and where the
  shared preconditioner wants to be centred.
* Default proper priors: `log σ_GP ~ N(log RMS(y), 1.5²)`,
  `log jitter ~ N(log median(yerr), 1²)`, `log ρ` uniform on
  `[2 Δt_med, T/2]`, `log Q ~ N(log 3, 1²)` truncated at ~500. **Prior
  propriety is a throughput feature**: with thousands of chains, a flat
  amplitude prior guarantees stray chains on the `σ → 0` ridge, and ChEES
  adapts one step size from the harmonic-mean acceptance, so those few chains
  drag it down for everyone.
* Sums of same-type terms get an **ordering reparametrisation**, applied
  automatically by `make_target`: each subsequent timescale is a ratio > 1 of
  the previous, so the k! permutation modes collapse by construction with no
  rejection and no discontinuity. Without it, R-hat across label-switched
  modes means nothing. (anvil's `Transform` has no ordered-vector transform;
  this keeps the fix local rather than requiring an engine change.)
* `priors.periodogram_period` seeds rotation kernels inside one harmonic —
  the legitimate use of spectral machinery, as an *initialiser*. The Whittle
  likelihood itself is a quasi-likelihood and fails the exactness rule.

Named identifiability ridges to expect, none of them bugs: σ_GP vs jitter as
ρ → cadence; GP vs baseline as ρ → T; GP vs transit depth when ρ ~ transit
duration; period harmonics in rotation kernels.

Geometry note, to calibrate fear: the marginalised GP has **no Neal's-funnel
latent hierarchy**. What remains is mild ridges, which ChEES's shared diagonal
preconditioner plus trajectory adaptation handle — conditional on the
parametrisation above.

## The float32 discipline

anvil's five rules carry over. What a GP adds is that the recursion itself has
hazards: forbidden process-noise differences, a log-determinant that must be
reference-subtracted, an `inf × 0` in the overdamped corner, and a
positive-definiteness failure mode that must be a *defined, finite* outcome.
Details and measurements: [docs/float32.md](docs/float32.md).

### Known limitation: compiled gradients

**M4 does not remove this, and I checked rather than assumed.** The fused
kernel's VJP is a `custom_function`, so `mx.compile` never sees the element
algebra's backward graph — but the failure was never there. It is in the
*front end*, `terms._sho_matrices`, which both paths share, so
`mx.disable_compile()` is still required for ChEES-HMC on either. What M4
changes is the cost of the gradient once it runs, not whether it compiles.

`mx.compile` over the **backward** pass of the whole likelihood fails with
*"Too many inputs/outputs fused in the Metal Compiled primitive"* for m ≥ 2.
MLX fuses a long dependent elementwise chain into a single Metal kernel whose
argument count exceeds its limit of 24 arrays per fused group. It is
structural, not size-dependent: identical at N = 128 and N = 1000, and absent
at m = 1. Forward-only compilation is fine at every size and every m.

**Correction (2026-09-19).** This section used to blame the unrolled
Cholesky. That is wrong, and it matters because it made the problem look
unfixable. Every piece of the element algebra compiles under `mx.vjp` when
taken on its own — `chol`, `combine`, `combine_leaf`, `leaf_elements`, and
leaf → fold → tree end to end at K = 1024, the last of these 1.5x faster
compiled than eager. Only the *whole* `__call__` fails, so the offending
group is in the front end (kernel parameters, term matrices, the gather, the
first-point mask, `_sanitize`) and not in the Cholesky at all. Splitting the
VJP at the term-matrix boundary should therefore recover most of it; that is
tracked under "What is left" rather than done. (An earlier version of this
paragraph expected M4 to make it moot. It did not; see above. Note also that
the failure is a process abort from inside Metal, not a Python exception,
so nothing can catch it.)

This matters because anvil's ChEES kernel compiles its leapfrog step, so
**ChEES-HMC cannot run on a GP target as-is**. The workaround is one line and
is verified (30 ChEES steps, 64 chains, zero divergences):

```python
import mlx.core as mx
mx.disable_compile()          # before anvil.run(ChEESHMC(...), ...)
```

The gradient-free stretch move is unaffected, and so is the forward path.
The fix is still the one-line workaround: the over-fused group is in
`terms._sho_matrices`, so splitting the VJP at the term-matrix table is
what would remove it, and that is tracked under "What is left".

### Silent MLX behaviours

Three more are worked around and pinned as regression tests
(`tests/test_mlx_gotchas.py`), because all three are silent:

1. **`mx.array(np_float64_array)` returns float32** unless the dtype is
   named. Downstream arithmetic promotes the result back, so a "float64 path"
   can carry float32 data with every dtype it reports looking correct.
2. **A single-element strided slice has a wrong VJP.** The backward pass
   fills `start:stop` rather than `start:stop:step`, so `x[0:2:2]` — which
   selects only index 0 — returns `[1, 1]` instead of `[1, 0]`. Values
   unaffected; gradients silently wrong. A reduction tree's last level is
   always `k = 2`, so this hit every gradient this package produced until
   `elements._pairs` replaced striding with reshape-and-index. Present in the
   current MLX release; a ready-to-file report is at
   [docs/mlx-issue-strided-slice-vjp.md](docs/mlx-issue-strided-slice-vjp.md).
3. **float64 `exp`/`sin`/`cos` are only float32-accurate** (`log`, `expm1`,
   `sqrt` are true float64), so the verification path needs a written-out
   exponential — and `exp(-x)` cannot be recovered from `1 + expm1(-x)`.

## Milestone status

* **M1 — algebra + reference path: complete.** SSM propagators via
  `expm1`/sinc forms, dtype-polymorphic tree-reduction reference, float64
  `exp` helper, float64 parity against celerite2 *and* a dense oracle
  including absolute log L, sums of terms, duplicate timestamps,
  Q ∈ {0.1, 1/√2, 1/2 ± ε, 10, 100, 300}. Plus, ahead of schedule: working
  float32 GPU path with correct gradients, the priors/parametrisation layer,
  and the anvil integration.
* **M2 — decision benchmark: mostly measured.** Done: the reference forward
  grid over chains x N x m, its memory, block and sub-block scaling
  (`benchmarks/bench_reference.py`); **both** celerite2 denominators, single
  core and 12-core multiprocess (`benchmarks/bench_vs_celerite2.py`); the
  decomposed float32 error scaling (`benchmarks/error_scaling.py`); a roofline
  placement showing the reference is ~40x off bandwidth and therefore op-count
  bound; and the compiled-gradient limit, a hard structural result rather than
  a tuning number. Still owed: the Design-A probe comparison, the
  `metal::fast` accuracy study, and the ChEES leapfrog cost at scale.
* **M3 — fused Metal blocked-scan kernel: pass 1 done.**
  45-74x over the MLX path and 35-42x ahead of a 12-core celerite2 pool,
  which is the brief's denominator, so the >=100x gate is **not** cleared
  (it is 255-376x against a single-core loop). `tests/test_metal.py` pins
  it against the dense float64 oracle and the MLX path across every term type, every block
  length, duplicate timestamps, gaps, irregular cadence, a jitter and a mean
  model. Still owed: m > 4, per-instrument jitter in-kernel, and the float32
  feasibility map; the thread-layout work landed with M4's stage B. The feasibility map now has evidence that
  it matters: amplitude and correlation length are **not separable**. At
  `sigma/yerr = 100` *and* `rho/dt = 1000` the MLX path is off by 1.5
  log-units where either axis alone stays under 0.2, while the kernel holds
  0.02. See `test_the_joint_high_amplitude_long_correlation_corner`.
* **M4 — checkpointed backward kernel: done.** The kernel carries its own
  VJP; see [Gradients](#gradients) for the numbers (99x at N = 1e4, 264x at
  N = 1e5, matching float64 central differences to ~1e-7) and
  [docs/m4-plan.md](docs/m4-plan.md) for the stages, what each measured,
  and the adversarial review that closed it. The design that shipped is
  not the one first sketched here. The backward blocks are **independent**:
  a block's backward seed has a closed form in the prefix state and the
  suffix element (`r = G^T (eta_s - J_s mu)`, `Pbar = (cot/2)(r r^T - Jt)`,
  where `G` and `Jt` are what `combine` already forms), so there is no
  chain of `(mbar, Pbar)` through the blocks and no affine-map composition.
  Three launches, all on the forward's grid: block elements; one thread per
  chain for prefix, suffix and seeds; one thread per (chain, block) that
  replays its block from checkpoints and sweeps it back.

  Two practical limits to know before relying on it. A gradient at the
  brief's realistic kernel (rotation + granulation, m = 6) is off the kernel
  and costs seconds per call at 256 chains and N = 1e4. And a ChEES chain
  started deep in a bounded parameter's prior tail never moves again — not
  because of anything here (the GP likelihood and its gradient stay finite
  well past the point where anvil's own transform returns `-inf`) but
  because the transform's Jacobian gradient reaches ~6e8 there against ~5e2
  for a healthy chain, and ChEES adapts one step size for the whole
  ensemble. Capping `max_leapfrog` does not help; an earlier version of this
  note claimed it was a Metal buffer limit, which was wrong. Full
  measurements and suggested fixes:
  [docs/anvil-chees-boundary.md](docs/anvil-chees-boundary.md). The healthy
  chains are unaffected, which is the part anvil-gp owes.

  What the adversarial review after it (stage R) found, all fixed and
  pinned by tests: a reachable point where the value was right and the
  gradient came back exactly zero, because the replay's covariance update
  was not PSD in float32 (it is now the Joseph form); the `s_floor_rel`
  knob honoured by the forward and not the backward (values that could
  bind are refused); the per-lag adjoint buffer unbounded on irregular
  cadences (refused over 2 GiB; see "What is left"); the kernel's Cholesky
  pivot floor built from the signed trace where the MLX one clamps it, so a
  non-PSD block gave NaN on one path and a finite value on the other;
  three tests that still passed with their subject deleted; and the
  stretch-move jitter width in the example table, which is two chains
  trapped in a secondary mode, the sampler's and not the likelihood's (see
  "End to end").

  Also recorded, because it is the tempting shortcut and it is wrong: never
  reconstruct the forward state by running the filter *backwards* through
  the propagator. That divides by `Phi`, which underflowed on the way
  forward at exactly the lags where it would now overflow — every gap in
  the data.
* **M5 — joint validation, SBC calibration, docs: started.** The unbiasedness
  check across realisations is in `tests/test_calibration.py`,
  `examples/gp_injection_recovery.py` is the end-to-end fiducial, and the
  brief's failure-region recovery test is in place (half an ensemble started
  in the infeasible corner, every chain out within 500 stretch steps).
  Still owed: SBC rank checks — the present check profiles one parameter
  with the others fixed at truth over 12 realisations, which tests
  unbiasedness of a point estimate and cannot see a mis-scaled posterior
  width — the adversarial geometry tests (zero-amplitude injection,
  timescale ≈ cadence), and the MetalPlanet hot-Jupiter joint fit.

## Install / test

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test]"        # mlx, numpy; celerite2, mpmath, scipy for tests
.venv/bin/pip install -e ../anvil         # for the engine integration only
.venv/bin/python -m pytest tests -q       # ~4.5 min; -m 'not slow' for ~2.5 min
.venv/bin/python examples/gp_injection_recovery.py
```

Requires macOS on Apple Silicon, Python ≥ 3.10, MLX ≥ 0.30. `import anvilgp`
needs only mlx and numpy; anvil is imported lazily inside `anvilgp.anvil`.

## Related work

Assessments of papers checked against this problem live in
[docs/related-work-notes.md](docs/related-work-notes.md), so the same ones are
not re-read from scratch. Currently: **LightGP** (arXiv 2605.17898) — a
dependency-free C++ GP library with Metal and CUDA backends, not applicable
here (unbatched, finite-difference gradients only, and its scalable paths are
CG and SKI which fail the exactness rule), but carrying one measurement worth
remembering: Accelerate's Cholesky on Apple's AMX coprocessor is 9–27x faster
than their Metal Cholesky for a *single* factorisation.

## Prior art

No GP implementation existed on MLX before this one. Foundations: Särkkä &
García-Fernández [1905.13002](https://arxiv.org/abs/1905.13002) (temporal
parallelisation); Corenflos, Zhao & Särkkä
[2102.09964](https://arxiv.org/abs/2102.09964) (parallel-in-time GP
regression); Foreman-Mackey et al.
[1703.09710](https://arxiv.org/abs/1703.09710) (celerite),
[1801.10156](https://arxiv.org/abs/1801.10156) (backprop); Kelly et al.
[1402.5978](https://arxiv.org/abs/1402.5978) (CARMA/SSM); Yaghoobi et al.
[2406.05188](https://arxiv.org/abs/2406.05188) (parallel square-root filters —
the pre-registered numerical fallback if float32 covariance-form composition
ever fails its gates); S+LEAF
[2201.02440](https://arxiv.org/abs/2201.02440) (integrated kernels, the v2
path for exposure-time smearing).
