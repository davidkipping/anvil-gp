# Plan: the kernel's block boundaries in the square-root information form

Status: agreed 2026-10-25; all six stages done 2026-10-25. The square-root boundary is the kernel's unconditional default (decision 2, revised: B).

## The problem

The fused kernel's backward starts every block from an *entry state*
(the filtered mean and covariance there) and a *seed* (the adjoint of the
state at its exit), both composed across blocks in the dense information
form. On bright, densely sampled data float32 gets both wrong: the
covariances through the conditioning of `(I + C J)^-1`, the means and
seeds through the cancellation in `eta - J mu`. The gradient is then
heavy-tailed across the posterior (worst 4-54 sigma), ChEES collapses its
step size, and on a 20-s, N = 4096, s/e = 100 problem it does not converge
at the default blocks. The value is affected too at short blocks. Measured
and diagnosed in docs/float32.md, section 7, step 2.

## The fix

Carry `C = U U^T` and `J = Z Z^T` as triangular factors (Yaghoobi,
Corenflos, Hassan and Sarkka, 2022), and build every block's elements
centred on the filter replayed from that block's own start state, with
the boundary mismatch moved into the previous block's output `b`. The
square-root form fixes the covariances (~5000x), centring fixes the means;
neither works alone. Reference: `tests/boundary_oracle.py` (the
prototype it came from is `benchmarks/boundary_prototype.py`).

Pipeline: today's pass 1 and prefix (approximate entry states) -> one
centred square-root pass per block, the replay fused with the fold ->
square-root prefix, suffix and seeds per chain -> the value replay and
the backward, unchanged.

## Decisions (2026-10-25)

1. **Convolved terms keep the dense path.** A `TermConvolution`'s `Q` is
   indefinite by design (the "formal" variant), so `L_Q` and `U` do not
   exist. A J-orthogonal QR could serve it; not in scope.
2. **Gate or ship at +25% value + gradient** on the README configurations
   (forward +50%). Over that, gate by state dimension or block length
   instead of paying it everywhere. **Revised after stage 5: ship
   everywhere (option B).** It came in at 1.4-1.8x; state dimension cannot
   gate it (m = 2 fails too), and a gate on the data's signal-to-noise was
   judged too likely to fail silently on a posterior it did not anticipate
   -- the failure it would let back in is ChEES not converging, while the
   cost is visible and bounded.
3. **The old boundary stays behind a policy switch for one release**, for
   A/B comparison and as the formal terms' path.

## Stages

| stage | what | estimate |
|---|---|---|
| 0 | Oracle and red tests | 0.5-1 day, **done** |
| 1 | Q factor table, a small Metal kernel (Cholesky, pivots clamped at 0, no floor) | 0.5 day, **done** |
| 2 | Centred square-root block pass, one thread per (chain, block); rank-1 leaf update | 2 days, **done** |
| 3 | Square-root prefix, suffix and seeds, one thread per chain; 2m x 2m `tria` | 1-1.5 days, **done** |
| 4 | Wire into `scan_loglike` forward and VJP, behind the switch; flips stage 0's expected failures | 0.5 day, **done** |
| 5 | Measure on a quiet machine: README timings, watchdog at m = 4, the exposure matrix (m = 2-4, gapped and 2-s cadences), then ChEES | 1 day + machine time, **done** |
| 6 | Docs and README | 0.5 day, **done** |

## Acceptance

* Boundary quantities against the float64 filter: entry means < 1e-3,
  covariances < 1e-4, seeds < 2.5e-3, relative
  (`tests/test_boundary.py`).
* Gradient < 0.5 posterior sigma at every one of 16 Laplace draws of the
  20-s problem, at 16 x 256 and 4 x 1024 (same file).
* ChEES on `t20s100`, 1024 chains, 300 warmup, default blocks: max R-hat
  <= 1.1 and >= 0.4 ESS per 1000 gradients (one block: 0.81-0.86;
  `benchmarks/chees_block_cost.py`).
* Value + gradient within the decision 2 threshold; no row of
  `benchmarks/floor_exposure.py` worse; full suite green.

## Stage 0: what it found (2026-10-25)

`tests/boundary_oracle.py` holds the float64 truth (a sequential filter's
boundary states, and their adjoints by one MLX float64 reverse sweep) and
the block-local centred square-root boundary in NumPy.
`tests/test_boundary.py` pins it on the 20-s problem (N = 4096, SHO +
SHO, s/e = 100, data seed 2, 16 Laplace draws), ~12 s for both block
lengths:

| | 16 x 256 | 4 x 1024 |
|---|---|---|
| kernel: entry means / covariances / seeds | 5.1e-2 / 2.1e-2 / 1.4e-3 | 4.5e-2 / 5.5e-3 / 9.1e-3 |
| float32 oracle | 3.1e-4 / 1.2e-5 / 5.2e-5 | 1.6e-4 / 1.0e-5 / 8.3e-4 |
| kernel gradient, worst per posterior sigma | **4.0** | **5.0** |
| oracle boundary through the kernel's float32 backward | 0.33 | 0.24 |

The float64 oracle reproduces the filter to 1e-12. The kernel's two tests
are `xfail(strict=True)`: stage 4 has to remove the markers.

**A correction to stage 1, found here.** The plan factored `Q` by float32
Cholesky with the relative floor `eps * trace` the composition kernels
use. At 20-s cadence `Q`'s eigenvalues run 2.5e-2 down to 1.7e-9, below
that floor (~3e-9), so the floor rewrites exactly the directions a
high-s/e filter depends on. It cost the seeds 20-50x (7.8e-4 / 2.0e-3
against 4.7e-5 / 3.9e-5) and the entry covariances 12x. A float32
Cholesky with each pivot clamped at zero and no floor matches a float64
eigendecomposition (4.8e-5 / 3.7e-5); so does a 1e-3 smaller floor. Stage
1 uses the clamp. Ruled out alongside: storing `Q` in float32 is not the
loss (a float64 eigendecomposition of the float32 table is as good).

## Stage 1: what it found (2026-10-25)

`metal.q_factor` factors every `(chain, lag)` entry of the `Q` table, one
thread each: Cholesky, each pivot clamped at zero, a non-positive pivot
zeroing its column, no floor. Not wired in yet, and not differentiated
(the boundary quantities only fix where each block starts).

* **Every direction of `Q` survives**, not just every entry. Across
  `RealTerm`, `SHOTerm` (under- and overdamped), Matern-3/2 and -5/2,
  `RotationTerm`, SHO + SHO and SHO + Matern-3/2, at lags 0, 2 s, 20 s,
  2 min, 30 min and 10 days, `u^T L L^T u / lambda` is within 3e-5 of 1
  for every eigenvector, though the eigenvalues span up to 22 decades
  (Matern-5/2 at 2 s). Entries to 1e-7 of `max |Q|`; the NumPy oracle's
  factor to 2e-7. The planned `eps * trace / m` floor inflated the
  smallest eigenvalue 1.7x at 20 s and 70x at 2 s.
* **The clamp holds at denser cadences.** In the oracle, the kernel-style
  factor matches a float64 eigendecomposition in every boundary quantity
  at 2 s (SHO + SHO, eigenvalues to 1e-12 of the largest), 20 s and
  2 min, m = 2 and 4.
* **Pinned** in `tests/test_boundary.py`: the direction test over the
  terms and lags above, rank-deficient and slightly indefinite `Q`
  (finite, `L L^T = Q`), batching, and the oracle on the kernel's factor
  meeting the stage 0 bounds on the 20-s problem. Mutation-checked: the
  old floor fails 7 of them, removing the clamp 8.

For stage 5, seen on the way. At 2-s cadence, SHO + SHO, s/e = 100, the
kernel's entry covariances are **57%** wrong at 16 x 256 (17% at
4 x 1024), against 2% at 20 s: the exposure grows fast with cadence. And
the oracle's entry means reach 1.2e-3 at 2 s, 4 x 1024, just over the
1e-3 bound, with either factor, so that is not the factor's doing.

## Stage 2: what it found (2026-10-25)

`metal.sqrt_blocks`, one thread per (chain, block): the replay from the
block's start state fused with the centred square-root leaf fold, and the
output moved to the next block's start state. Not wired in yet. The
algebra, in `_SRC_SQRT_BLOCKS`'s comment: the leaf's information is rank
one, so every `(I + C J)^-1` collapses to the scalar `alpha = 1 + |U^T
z|^2` that pass 1 already uses, and only two factor updates remain per
observation -- `Z` by a Givens rank-one update and `U` by one `m x 2m`
Householder `tria`. The leaf's `U2` and the update's `(I + p p^T)^-1/2`
are closed-form symmetric roots written without cancellation; no
`(1+m) x (1+m)` QR per leaf.

Against the float64 oracle from the same start states (relative to each
field's largest entry; float32 NumPy oracle in brackets):

| | A | b | U | eta | Z |
|---|---|---|---|---|---|
| 20 s, 16 x 256 | 6.5e-6 (5.6e-6) | 1.3e-4 (1.3e-4) | 1.5e-5 (4.5e-6) | 1.0e-5 (9.4e-6) | 3.5e-6 (3.6e-6) |
| 20 s, 4 x 1024 | 9.5e-6 (9.2e-6) | 4.3e-3 (2.0e-3) | 2.1e-5 (1.1e-5) | 1.7e-5 (1.8e-5) | 1.2e-5 (1.1e-5) |
| 2 s, 16 x 256 | 1.7e-5 (1.9e-5) | 8.4e-3 (1.1e-2) | 6.1e-5 (3.0e-5) | 5.9e-5 (5.7e-5) | 9.4e-7 (1.4e-6) |
| 2 min, 16 x 256 | 1.1e-5 (5.5e-6) | 7.1e-4 (5.8e-4) | 4.9e-6 (1.4e-6) | 3.3e-6 (3.7e-6) | 1.2e-6 (1.0e-6) |

(`b` is a centred correction, small against its own scale, so its
relative error reads large; what it feeds is below.) `U` is 2-3.5x the
NumPy oracle's. Not the algebra: the oracle on the kernel's exact algebra
is within 1.2x of LAPACK's, and on the kernel's exact Householder within
1.6x; the rest is Metal's float32 rounding (precise division changes
nothing, to the digit).

Boundary quantities from the kernel's blocks through the NumPy scan
(stage 3's job), entry means / covariances / seeds: 3.2e-4 / 1.8e-5 /
2.5e-4 at 20 s, 16 x 256, and 1.6e-4 / 2.7e-5 / 1.0e-3 at 4 x 1024, all
inside the bounds (the NumPy blocks: 1.5e-4 / 6.6e-6 / 1.2e-4 and 1.3e-4 /
1.2e-5 / 8.5e-4). The gradient through the kernel's own backward from
them is inside 0.5 sigma at all 16 draws. At 2 s the entry covariances
reach 1.9e-4 at 4 x 1024, over the 1e-4 bound (NumPy blocks 1.0e-4; the
dense kernel 17%) -- for stage 5.

Pinned in `tests/test_boundary.py`: each field no worse than 5x the
float32 oracle's error, block 0's `A`, `eta`, `Z` exactly zero (stage 3
may rely on it), the boundary bounds and the gradient on the 20-s
problem, per-instrument jitter (0, 1, 3), and a short last block.
Mutation-checked: dropping the leaf's jitter, the output shift, the `Z`
update, the centring, or the `(I + p p^T)^-1/2` factor each fails 2-10
of them.

Cost and the m = 4 watchdog: not measured (stage 5). The thread carries
the replay's state as well as the element, so register pressure at m = 4
is the risk.

## Stage 3: what it found (2026-10-25)

`metal.sqrt_scan`, one thread per chain over its block elements: a prefix
carrying `(b, U)` emits each block's entry state, and a backward suffix
carrying `(eta, Z)` forms each block's seed as it passes, so nothing but
the prefix is stored. Each composition takes one `2m x 2m` Householder
`tria` and triangular solves with `X11`, whose diagonal is >= 1. An
entry-only variant (no `cot`) drops the suffix, for the forward. Not
wired in yet.

The scan adds nothing measurable. Against a float64 scan of the same
kernel blocks it is within 1.1e-5 on the 20-s problem and 3.6e-5 at 2 s
(the NumPy float32 scan: 3.0e-6 and 9.4e-6), and against the exact
filter it reproduces the NumPy scan's numbers to two digits: entry means
/ covariances / seeds 3.3e-4 / 1.8e-5 / 2.6e-4 at 16 x 256 and 1.6e-4 /
2.7e-5 / 1.0e-3 at 4 x 1024 -- the block pass's error, carried through.

**The whole new boundary now runs on the GPU** (`q_factor`,
`sqrt_blocks`, `sqrt_scan`, from the dense kernel's entry states): inside
the bounds, and the kernel's own backward from it is inside 0.5 posterior
sigma at all 16 draws, at both block lengths. Pinned in
`tests/test_boundary.py`, with the structure the backward relies on
(block 0 enters at its start, the last seed is exactly zero, the
entry-only variant agrees bit for bit), per-chain independence, and a
single block. Mutation-checked: five defects in the scan (the prefix's
`U U^T eta` term, `X11^-T` swapped for `X11^-1`, the seed's `X22 X22^T`,
the suffix's `(I + J C)^-1`, its `Z` update) each fail 4-6 tests.

## Stage 4: what it found (2026-10-25)

`GPPolicy.metal_boundary`: `None` (default) resolves to `"sqrt"`, or to
`"dense"` for a convolved term; `"sqrt"` on a convolved term is refused at
construction. `scan_loglike(boundary=)` runs pass 1, the dense prefix for
the start states, then `q_factor`, `sqrt_blocks` and `sqrt_scan` -- in the
forward (entry-only scan) and in the VJP (with seeds); the same kernels on
the same inputs, so the backward replays from exactly the states the value
did. A single block skips the square-root pass (both settings give the
start state and a zero seed, and agree bit for bit).

On the 20-s problem, 16 Laplace draws, dense -> square-root:

| | 16 x 256 | 4 x 1024 |
|---|---|---|
| gradient, worst per posterior sigma | 4.0 -> 0.31 | 5.0 -> 0.34 |
| entry means / covariances / seeds | 5.1e-2 / 2.1e-2 / 1.4e-3 -> 4.6e-4 / 2.8e-5 / 1.8e-4 | 4.5e-2 / 5.5e-3 / 9.1e-3 -> 3.2e-4 / 3.1e-5 / 1.8e-3 |
| value, sd across the draws | 0.066 -> 0.020 | 0.038 -> 0.025 |

Over all 16 draws the seeds at 4 x 1024 reach 1.8e-3 against the
2.5e-3 bound (1.0e-3 at the 3 draws stage 3 measured): a thinner margin
than the other quantities, worth watching in stage 5's wider matrix.

Stage 0's expected failures now pass and have lost their markers; the
dense boundary's own tests are the strict expected failures now, which
pins that `"dense"` still is the old boundary. The boundary the default
path hands its backward is bit-identical to the stage 2-3 pipeline.
README and docs/float32.md say what changed, and the README's timing
tables are marked as predating it.

## Stage 5: what it found (2026-10-25)

### Cost: over the line, by 1.4-1.8x on value + gradient

`benchmarks/boundary_cost.py`, interleaved A/B, minimum of 3 rounds,
stability probe at 0.99x its quiet level (0.3% spread), no other GPU job:

| config | forward: dense -> sqrt | value + gradient: dense -> sqrt |
|---|---|---|
| m = 2, N = 1e4, 1024 chains | 2.68 -> 4.62 ms (1.72x) | 9.33 -> 13.13 ms (1.41x) |
| m = 2, N = 1e5, 1024 chains | 21.7 -> 39.7 ms (1.83x) | 66.8 -> 102.6 ms (1.54x) |
| m = 4, N = 1e4, 256 chains | 13.4 -> 33.8 ms (2.51x) | 60.4 -> 101.9 ms (1.69x) |
| m = 4, N = 1e5, 256 chains | 46.5 -> 127.9 ms (2.75x) | 220 -> 397 ms (1.80x) |
| m = 4, N = 4096 (16 x 256), 1024 chains, 20 s | 9.14 -> 23.0 ms (2.52x) | 42.0 -> 71.5 ms (1.70x) |

That is after one optimisation found here. Per kernel at m = 4, N = 1e4,
256 chains: pass 1 8.4 ms, the value replay 4.5, `sqrt_blocks` **36**,
`sqrt_scan` 2.2 (0.9 entry-only), `q_factor` 0.2, the backward 34. Of
`sqrt_blocks`' 36 ms, 22 were the `m x 2m` Householder `tria`: written
with triangular loop bounds and an early `continue`, the compiler did not
unroll it. Constant bounds with masks, and selects instead of branches
(also in the Givens update), took `sqrt_blocks` to 19.7 ms (m = 2: 2.85
-> 2.06; the short-block case 22 -> 12.9), bit-for-bit within float32
rounding of the old (6.6e-6 in `U U^T`), every test unchanged.
`#pragma unroll` adds nothing. What is left of it: the replay and leaf
4.8 ms, the fold's products 5.5, the Givens update ~3, the `tria` ~6.

The structure sets a floor on the ratio. The block pass is a second sweep
of the data at ~2.3x pass 1's per-step work (a replay plus a fold), and
the VJP recomputes the forward's boundary, as it always recomputed pass 1,
so it is paid twice per gradient. Decision 2's +25% is out of reach by
tuning.

### Watchdog: no kill

m = 4, value + gradient on the square-root boundary: N = 1e5 at 256,
512, 1024 and 4096 chains with 1024- and 2048-point blocks (2048 was
recorded as the edge for pass 1 alone), N = 8192-32768 with explicit
2048-8192-point blocks at up to 4096 chains: all finite, none killed.

### Where the dense boundary fails, and what the square-root one leaves

`benchmarks/boundary_exposure.py`: 16 Laplace draws, default blocks, the
worst gradient error per posterior sigma and the sd of the value error:

| setup | blocks | gradient: dense | sqrt | value sd: dense | sqrt |
|---|---|---|---|---|---|
| m=4 SHO+SHO, 20 s, s/e 100 | 16x256 | 3.16 | **0.44** | 0.064 | 0.015 |
| m=4 SHO+SHO, 20 s, s/e 30 | 16x256 | 0.14 | 0.12 | 0.005 | 0.004 |
| m=4 SHO+SHO, 20 s, s/e 10 | 16x256 | 0.013 | 0.015 | 0.002 | 0.001 |
| m=4 SHO+SHO, 2 s, s/e 100 | 16x256 | 6.78 | **8.55** | 0.192 | 0.155 |
| m=4 SHO+SHO, 2 s, s/e 30 | 16x256 | 0.98 | **0.57** | 0.019 | 0.017 |
| m=4 SHO+SHO, 2 min, s/e 100 | 16x256 | 0.30 | 0.05 | 0.003 | 0.004 |
| m=4 SHO+SHO, 30 min, s/e 100 | 16x256 | 0.001 | 0.001 | 0.001 | 0.001 |
| m=4 SHO+SHO, 20 s, 1-day gap, N=8192, s/e 100 | 16x512 | 6.86 | **0.89** | 0.053 | 0.027 |
| m=4 SHO+SHO, 2 min, N=16384, s/e 100 | 16x1024 | 0.09 | 0.21 | 0.008 | 0.007 |
| m=4 Rotation, 2 min, s/e 100 | 16x256 | 3.92 | **0.74** | 0.085 | 0.043 |
| m=3 SHO+Real, 20 s, s/e 100 | 16x256 | 0.04 | 0.02 | 0.003 | 0.001 |
| m=3 SHO+Real, 2 s, s/e 100 | 16x256 | 0.26 | 0.11 | 0.012 | 0.006 |
| m=2 SHO, 20 s, s/e 100 | 16x256 | 0.34 | 0.01 | 0.001 | 0.001 |
| m=2 SHO, 20 s, s/e 300 | 16x256 | **1.57** | 0.05 | 0.001 | 0.002 |
| m=2 SHO, 2 s, s/e 100 | 16x256 | 0.14 | 0.006 | 0.002 | 0.000 |
| m=2 SHO, 2 s, N=16384, s/e 300 | 16x1024 | **1.79** | 0.05 | 0.002 | 0.002 |

* **m = 2 is not safe on the dense boundary** (1.6-1.8 sigma at s/e 300),
  so gating by state dimension is ruled out.
* **Where the square-root boundary is above 0.5, the boundary is mostly
  not why.** Exact float64 entry states and seeds through the same
  float32 backward (worst per sigma; `tests/boundary_oracle.py`):

  | setup | exact boundary | sqrt | one block | MLX path |
  |---|---|---|---|---|
  | 20 s, s/e 100 | 0.31 | 0.44 | 0.19 | 0.74 |
  | 2 s, s/e 100 | **6.4** | 8.6 | 0.90 | 0.85 |
  | 2 s, s/e 30 | 0.58 | 0.57 | 0.11 | 0.04 |
  | Rotation, 2 min, s/e 100 | 0.49 | 0.74 | 0.08 | 0.77 |
  | 20 s gapped, N = 8192 | (graph too large) | 0.89 | 0.41 | 1.02 |

  At 2 s, s/e 100 even the exact boundary is 6.4 sigma off: the entry
  covariance's eigenvalues span ~12 decades there, and handing it to the
  backward as a dense float32 matrix -- and the seeds likewise -- is
  what fails, before any composition. One block avoids it because the
  replay builds its own covariance from the start. That floor belongs to
  the dense float32 backward, and fixing it means a square-root backward:
  outside this plan. Rotation at 2 min is the one row where the
  square-root boundary sits clearly above its floor (0.74 against 0.49).

### ChEES: converges at the default blocks, at the one-block rate

`benchmarks/chees_block_cost.py t20s100 0 {1,2} 1024 300 200 128` (the
problem docs/float32.md's table measured), square-root boundary, default
16 x 256 blocks; the earlier rows for comparison:

| boundary, blocks | step size | leapfrog / iter | max R-hat | ESS per 1000 gradients | wall |
|---|---|---|---|---|---|
| **sqrt, 16 x 256, seed 1** | 0.0225 | 35.6 | 1.116 | **0.81** | 11.5 min |
| **sqrt, 16 x 256, seed 2** | 0.0208 | 27.2 | 1.189 | **0.68** | 8.9 min |
| dense, 16 x 256, 300 warmup | 0.0078-0.0093 | 12 | 2.8-3.3 | -- | 3 min |
| dense, 16 x 256, 2000 warmup | 0.0042-0.0043 | 33-41 | 1.57-1.74 | 0.084-0.093 | 52-55 min |
| one block (`metal_block = 4096`) | 0.0225-0.0228 | 35-38 | 1.11 | 0.81-0.86 | 32-34 min |

* **ESS per gradient: met** (0.68-0.81 against >= 0.4; one block
  0.81-0.86), ~8x the dense boundary's, and the step size back at the
  one-block value. Minimum ESS per minute of wall time 424-512, against
  182-195 for the `metal_block = N` workaround (2.2-2.8x; the one-block
  runs were timed on a machine that was not quiet, so take the ratio as
  approximate).
* **R-hat <= 1.1: not met, and the criterion was wrong.** One block
  itself reaches only 1.11 at 300 warmup and 200 draws. Per parameter,
  seed 1 matches one block within 0.006 everywhere (worst: the rotation
  timescale, 1.116 against 1.110 / 1.108). Seed 2 is 1.19 on the same
  parameter: its adapted trajectory came out shorter (1.11 against
  1.58), so the slowest direction mixed less in 200 draws; its step size
  is 8% under one block's, which the residual gradient error may have
  helped along.
* The pooled posterior agrees with one block's to <= 0.022 sigma in
  every mean, widths within 2%.

### The decision this leaves (decision 2)

Over the +25% line, so by decision 2 the boundary should be gated rather
than paid everywhere. What the measurements say about each gate:

* **By state dimension: ruled out.** m = 2 fails on the dense boundary
  (1.6-1.8 sigma at s/e 300).
* **By block count: already done** where it is free (one block skips it).
* **By the data's signal-to-noise**, decided once at construction from
  `rms(y) / median(yerr)` (the ratio the kappa warning already computes):
  the dense boundary was within 0.14 sigma at every s/e <= 10 and 20-s s/e
  30 row, and failed from s/e 30 at 2 s and from s/e 100 at 20 s and
  2 min. A threshold near 20 would keep the dense boundary's speed on
  quiet stars, RV and sparse cadences, and send bright, dense data to the
  square-root one.
* **None**: keep the square-root boundary everywhere and pay 1.4-1.8x on
  every gradient, including the problems that never needed it.

## Stage 6 (2026-10-25)

Decision 2 resolved as B: no gate. README, docs/float32.md and
docs/benchmarks.md updated; the README's tables re-measured on a quiet
GPU (`bench_reference.py`, which takes `--boundary` now, and
`verify_readme.py`, whose claims now hold the new figures):

| | dense | square-root |
|---|---|---|
| forward, N = 1e4 x 1024 / 1e4 x 4096 / 1e5 x 1024 | 2.73 / 8.53 / 21.7 ms | 4.4 / 13.8 / 37.6 ms |
| value + gradient, N = 1e4 / 1e5, 1024 chains | 9.3 / 66.8 ms | 12.9 / 104 ms |
| peak, value + gradient, N = 1e4 / 1e5 | 0.13 / 1.23 GiB | 0.13 / 1.24 GiB |
| injection-recovery example, stretch | 10.2 s, 1,766 ESS/s | 14.7 s, 1,435 ESS/s |

Against celerite2's 12-core pool (its 2026-10-24 figures; the CPU was busy)
the kernel's forward is now ~13x, from 22-25x. docs/benchmarks.md is
marked as predating the change rather than regenerated, for the same
reason.

**Found on the way: a residual copy per kernel.** Inside `scan_loglike`
MLX handed every further `metal_kernel` that read the residuals its own
copy of them (0.38 GiB each at N = 1e5 x 1024 chains): the forward's peak
was 0.77 GiB on the dense boundary and 1.15 on the square-root one, the
gradient's 2.00 and 2.76. Making them contiguous once in the forward and
the VJP removes every copy, for both boundaries: 0.39 and 1.23-1.24 GiB,
time unchanged. The README's peak columns had predated the replayed value
and now match again.

**The watchdog, again.** The A/B's m = 4, N = 1e5, 256-chain gradient,
clean in the morning, was killed ("Impacting Interactivity") in the
afternoon on *both* boundaries and with or without the copy fix, while the
display was in use (screen sharing, a user at the machine). A pre-existing
hazard, recorded in the README.

### Open, for a later plan

* ~~Reuse the forward's boundary in the VJP.~~ Done; see below.
* **A square-root backward**, for the 2-s, s/e 100 floor that exact
  boundaries do not fix.
* **Regenerate docs/benchmarks.md** on a quiet CPU and GPU.

## Follow-up: the VJP reuses the forward's boundary (2026-10-25)

`scan_loglike`'s custom function now returns, after the log-likelihood,
what its forward computed for the boundary -- pass 1's elements (dense)
or the square-root blocks and their start states -- and the VJP takes
them from ``outputs`` and runs only the per-chain scan with seeds, instead
of re-running pass 1, the prefix, `q_factor` and `sqrt_blocks`.
`scan_loglike` returns the log-likelihood alone, so the extra outputs'
cotangents are zero and ignored. Values and gradients are bit-identical
to before on seven configurations (both boundaries, m = 2 and 4, one
block, a convolved term, three jitters); pinned by a test that counts pass
1 and the square-root pass at one call each per value + gradient, and the
custom function's first output is still pinned at one scalar per chain.

The first version gained far less than estimated at N = 1e5, and made the
dense boundary *slower* (66.8 -> 78.5 ms). Not the reuse: the forward was
not being recomputed (a forward made ~19 ms heavier left the gradient's
own evaluation unchanged), and passing the boundary in as a stop-gradient
input instead of through ``outputs`` measured the same. It was the VJP's
`mx.contiguous(rs)`, added the same day for memory: with the backward
kernel now the residuals' only reader in the VJP it copies nothing without
it, and with it the VJP cost ~21 ms more. Removed there, kept in the
forward; peaks unchanged.

`benchmarks/boundary_cost.py`, quiet GPU (it now records a watchdog kill
instead of aborting), value + gradient:

| config | dense (now) | sqrt (now) | sqrt before the reuse | old dense |
|---|---|---|---|---|
| m = 2, N = 1e4, 1024 chains | 8.22 ms | **10.02** | 13.13 | 9.33 |
| m = 2, N = 1e5, 1024 chains | 55.7 | **73.0** | 102.6 | 66.8 |
| m = 4, N = 1e4, 256 chains | 52.5 | **74.5** | 101.9 | 60.4 |
| m = 4, N = 4096 (16 x 256), 20 s | 37.4 | **53.5** | 71.5 | 42.0 |
| m = 4, N = 1e5, 256 chains | killed | killed | 397 | 220 |

Against the dense boundary as it was before this plan, the square-root one
now costs 1.07-1.09x on value + gradient at m = 2 and 1.23-1.27x at m = 4
-- inside decision 2's +25% at m = 2, at its edge at m = 4. The forward is
unchanged (1.7-1.8x at m = 2, 2.5-2.8x at m = 4): forward-only samplers
gain nothing from this. README timings (`bench_reference.py`): value +
gradient 9.9 ms at N = 1e4 and 75.9 ms at N = 1e5, 1024 chains. The
m = 4, N = 1e5 row was killed by the GPU watchdog on both boundaries,
as in the afternoon before this change, with the display in use.

## Follow-on plan (agreed 2026-10-25)

In execution order. Decision I is taken; VII is deferred until VI
reports.

I. **Flip the `metal` default -- agreed, after III.** `GPPolicy.metal`
   becomes `None`, meaning the fused kernel wherever it is eligible and
   the MLX path otherwise, with the existing warning and fallback. The
   kernel is now the more accurate path on every row measured (value and
   gradient), the only one whose gradient runs at N ~ 1e5, and 17-200x
   faster; the one way the flip could make things worse is the watchdog,
   so it waits for III. One line plus a docs and README pass; a one-time
   message says which path was chosen.

II. **Watchdog diagnosis.** One-hour budget, no estimate before it.
   Reproduce the m = 4, N = 1e5, 256-chain kill with the display busy;
   find which kernel dies, whether committing between passes
   (`mx.eval`) saves it, whether the watchdog responds to per-thread
   length or to total command-buffer time, and at what threshold. Output:
   a note here and a go/no-go for III.

III. **Watchdog guard.** Whatever II shows: separately committed passes,
   chain chunking at the likelihood level, a lower m = 4 block cap, or a
   combination. Pinned by a stress test under simulated display load.
   Estimated after II; expected half a day to one day.

IV. **The default MLX path's float32 floor** (docs/float32.md s7,
   step 3): the kernel's replay fix at sub-block scale -- fold sub-blocks,
   prefix, replay. After I the scope is m > 4, the only path left on the
   MLX reduction. 1-2 days. Acceptance: `floor_exposure.py`'s MLX rows at
   the kernel's level.

V. **Review remainder and a measurement pass**, half a day: the logit
   link's dead gradient past |v| ~ 88; `TermConvolution` gaps
   (`make_target` / `default_hypers` reject it, `auto_order_groups`
   returns nothing silently, `kappa()` omits the diagonal correction);
   stale docs and A13; the seed margin (1.8e-3 against 2.5e-3)
   re-measured on a second data draw.

VI. **Size the convolved-term boundary gap.** Convolved terms keep the
   dense boundary (decision 1), and that is not a corner: exposure-time
   integration is the right model for Kepler long-cadence and TESS 2-min
   fits, and the dense boundary failed for an unconvolved `RotationTerm`
   at 2-min cadence, s/e 100 (3.9 sigma). Measure how indefinite Q is at
   realistic lags for convolved SHO and Rotation terms and the dense
   boundary's gradient error on a convolved Rotation at that cadence;
   then estimate a hyperbolic (J-orthogonal) `tria`. Half a day of
   measurement; the fix is estimated, not started.

VII. **Resolve the forward cost against decision 3.** Deferred until VI.
   Either keep `metal_boundary="dense"` as a documented speed option for
   forward-only samplers (its value is still 5-30x better than the MLX
   path's) or remove it after one release as agreed and accept the
   forward's 1.7-2.8x. If the forward cost matters, the deterministic
   alternative is a block-length gate for the value only (the value's
   failure was short blocks only, measured); a data-dependent gate was
   rejected above.

VIII. **Regenerate docs/benchmarks.md** when the CPU is quiet. Under an
   hour.

Longer term, not scheduled: a square-root backward for the 2-s cadence
floor (several days, partly research); removing `metal_boundary="dense"`
per VII; the >= 100x gate against celerite2's 12-core pool, now ~13x on
the forward.

## II. Watchdog diagnosis (2026-10-25)

Reproduced: an m = 4 (SHO + SHO), N = 1e5 value + gradient, with the
display in use, is killed ("Impacting Interactivity") at 224 chains and
above with the automatic 1024-point blocks, on either boundary; 192
chains survive (245 ms). The same call ran in ~400 ms that morning.

**Ruled out:**

* *Total GPU time.* Six backward kernels in one evaluation (459 ms)
  survive; so does the same gradient at 512-point blocks for a whole
  second (1024 chains).
* *Any single kernel.* Each kernel of the gradient in its own evaluation
  survives (longest: the backward, 142 ms).
* *Command-buffer packing.* `MLX_MAX_OPS_PER_BUFFER` = 1, 2 or 4 is still
  killed.
* *Memory.* Peak 0.5-0.7 GiB, 62% of system memory free.

The value and the gradient evaluated separately (131 + 171 ms) survive;
together they are killed.

**The lever is per-thread length.** At 256 chains, N = 1e5: 256- and
512-point blocks survive (357, 324 ms), 1024 and 2048 are killed; at
1024 chains, 256 and 512 survive (1.00 s, 0.96 s). At the other state
dimensions' caps nothing is killed under the same load: m = 2 at 8192 and
m = 3 (Matern-5/2) at 2048, up to 4096 chains (0.26 s, 0.80 s), and m = 4
at 1024 at N = 1e4. Not established: what the watchdog measures exactly.
The fit to everything above is that the unit it cannot preempt is a
thread's (or threadgroup's) uninterrupted run, which at N = 1e5 is
stretched by the number of co-resident threadgroups, so per-thread work
matters and total work does not.

**A 512-point cap at m = 4 is nearly free**, and mostly a speed-up
(value + gradient, 1024 -> 512-point blocks): N = 1e4 75 -> 48 ms at 256
chains and 101 -> 113 ms (+12%) at 1024; N = 3e4, 1024 chains, 340 -> 293;
N = 1e5, 256 chains, ~400 -> 324 (not killed), and 1024 chains 1304 ->
958 (not killed). More blocks means more threads for a GPU that m = 4's
register load keeps under-occupied.

**Go for III**, scoped down: lower `_MAX_METAL_BLOCK_HI_M` (m >= 4) from
1024 to 512, re-time the README's m = 4 rows and the auto-block test that
pins the measured optimum, and decide whether an explicit `metal_block`
above the cap should warn. The stress test cannot rely on display load
being present; it will pin the cap and run the m = 4, N = 1e5 case as a
slow test that fails if killed. Estimate: half a day.

## III. Watchdog guard (2026-10-25)

The 512 cap II proposed was not enough once tested at the extremes (value +
gradient, m = 4, display in use): killed at 4096 chains at N = 1e5 and at
N = 3e4. In the hour between II and III the limit had also tightened -- at
1024-point blocks even the *forward* was killed at N = 1e5 x 1024 chains,
and value + gradient at N = 3e4 x 1024. 256-point blocks survived every
case measured:

| m = 4, value + gradient | 1024 | 512 | 256 |
|---|---|---|---|
| N = 1e4, 256 chains | 74.6 ms | 48.2 | **40.5** |
| N = 1e4, 1024 chains | 101 | 115 | 113 |
| N = 1e4, 4096 chains | killed (forward too) | 374 | 369 |
| N = 3e4, 4096 chains | -- | killed | 1081 |
| N = 1e5, 1024 chains | killed (forward too) | 929 | 978 |
| N = 1e5, 4096 chains | -- | killed | 3557 |

The forward at 256 is within -55% to +12% of 1024 where both ran. The
block is fixed at construction (the gradient's slot map depends on it)
while the chain count arrives per call, so the cap has to hold for the
largest chain counts: **m >= 4 is capped at 256**, the auto rule's floor.
m = 2 at 8192 and m = 3 at 2048 survived up to 4096 chains at N = 1e5
under the same load and are unchanged. An explicit `metal_block` above a
cap is still allowed but warns at construction.

Tests: the cap (m = 4 and 6 -> 256), the warning (and its absence at and
below the caps), and a slow value + gradient at m = 4, N = 1e5, 4096
chains -- which cannot fail on an idle machine whatever the cap, so the
cap is what is pinned. With III done, I (the `metal` default) is next.
