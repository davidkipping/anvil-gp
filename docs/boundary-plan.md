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

## I. The `metal` default flipped (2026-10-25)

`GPPolicy.metal` defaults to `None`: the fused kernel wherever
`metal_eligible()` says it can serve the problem, the MLX path otherwise.
`True` keeps its meaning (insist, warn if it cannot be honoured) and
`False` forces the MLX path. A `MetalDefaultNotice` (a `UserWarning`,
filterable) says once per process and outcome which path the default took
-- the scripts this changes never asked for either path, and their float32
numbers and timings move. `make_target` inherits the default, so anvil
users get the kernel. The slot map is no longer deferred for a default
instance the kernel could serve (`_may_use_metal`).

Every existing test and benchmark that built a `GPPolicy` without `metal`
was meant for the MLX path and is now pinned to `metal=False` (64 calls,
found by parsing the source, not by grep) -- including `verify_readme.py`'s
stability probe, whose quiet level was measured on the MLX path, and
`floor_exposure.py`'s MLX column. The `make_target` calls in the
calibration and prior tests are left on the default: they test the anvil
integration as a user gets it. New tests: the default picks the kernel
bit for bit as `metal=True` does, falls back silently (no `RuntimeWarning`)
at m = 6, notices once per process and outcome and not when a path was
asked for, and `make_target` gets the kernel.

IV's scope is now m > 4, the only users the MLX path still serves by
default.

## V. Review remainder and a measurement pass (2026-10-25)

**Fixed.**

* *The logit link's dead gradient.* `1 / (1 + exp(-v))` had a float32
  gradient of inf / inf = NaN past v ~ -89 (and 0 past ~ -44), which the
  guard zeroed: a healthy value, a dead gradient. Now written with
  `z = e^{-|v|} <= 1`, |v| taken by a `where` so the gradient at v = 0
  stays 1/4 (an `abs` gives 0 there), `exp_acc` so float64 stays exact
  (`mx.sigmoid` is float32-accurate in float64). Finite and accurate in
  both dtypes from -120 to 120; the test fails on the old form with the
  NaN.
* *`TermConvolution` usability.* `default_hypers` raised `TypeError` on
  one, so `make_target` could not take a convolved term, and
  `auto_order_groups` returned `[]` for a convolved sum -- k!
  label-switching modes without a word. Both unwrap now (the convolution
  keeps the wrapped term's parameters); `seed_period` reaches a convolved
  rotation through the same path.
* *`kappa()` and the convolution's zero-lag correction.* `kappa()` left
  `diag_correction` c out, and c runs from -2e5 r to +1e8 r over ordinary
  parameters. Worse, without it the numerator was the formal
  h P_inf h^T, which is negative for an overdamped SHO -- kappa as low as
  -1e8, and a negative kappa slips past a refusal that only refuses above
  `kappa_max`. With c in both halves (k0 + c, the exact convolved k(0);
  max(h Q h^T + c, 0) + r) kappa is positive everywhere swept, within
  1-1.3x of the unconvolved term's in typical cases, and equal to it as
  the exposure shrinks.
* *Docs.* float32.md rule 1 now names `TermConvolution` as its documented
  exception, with the measured cost (A13); float32.md's compiled-gradient
  section no longer opens "a hard MLX limit" and closes "cannot be caught"
  around the text saying it was fixed (A6); the README's `sub_block`
  default (16 -> 32, A4), test count (A9) and `s_floor_rel` row (A12).
  Already fixed before this pass: A1-A3, A5, A7, A15; A10 waits for VIII;
  A14 is a historical plan's figure.

**Measured: the boundary's margin on two more data draws.** The 20-s,
s/e = 100 problem with the data seed changed, 16 Laplace draws each:

| data seed | blocks | entry means | seeds | gradient | dense gradient | exact boundary |
|---|---|---|---|---|---|---|
| 2 (the tests') | 16 x 256 / 4 x 1024 | 3.0e-4 / 2.6e-4 | 1.3e-4 / 1.1e-3 | 0.35 / 0.24 | 4.0 / 5.0 | 0.28 / 0.18 |
| 3 | 16 x 256 / 4 x 1024 | 3.3e-4 / **1.7e-3** | 2.6e-4 / 2.1e-3 | **0.51 / 0.58** | 1.1 / 2.9 | 0.15 / 0.12 |
| 4 | 16 x 256 / 4 x 1024 | 1.8e-4 / 1.6e-4 | 2.7e-4 / 9.4e-4 | 0.30 / 0.19 | 2.7 / 7.9 | 0.42 / 0.22 |

All on the current code. Seed 2 reads 0.35 / 0.24 where stage 4 measured
0.31 / 0.34: stage 5's rewrite of the QR moved rounding at the 1e-6 level
(the VJP reuse was bit-identical), and a worst-of-16-draws statistic moves
by ~0.1 sigma with that alone. The float32 backward's own floor, from
exact boundaries, is itself 0.12-0.42 sigma depending on the draw.

The acceptance bounds were set on one draw and do not all hold on
another: seed 3 is over the 0.5 sigma gradient bound at both block
lengths and over the entry-mean bound at 4 x 1024. That residual is the
boundary's, not the backward's (exact boundaries give 0.12-0.15 there),
and it splits between the float32 algebra -- the NumPy oracle leaves 0.41
/ 0.34 -- and the Metal implementation's rounding, 1.2-1.7x on top. Since
III, m = 4 always runs 256-point blocks, so the default's worst case
measured is 0.51 sigma (dense: 1.1). Still 2-40x better than the dense
boundary on every draw; not at the floor. Candidates if it matters: a
compensated accumulation in `sqrt_blocks`' fold products, or a second
centring round (which bought nothing on seed 2).

## VI. Sizing the convolved-term boundary gap (2026-10-25)

**How indefinite.** Eigenvalues of a convolved term's `P_inf` and `Q(dt)`
(exposure = cadence), float64, min / max:

| case | delta / rho | `P_inf` | `Q(dt)` |
|---|---|---|---|
| TESS 2-min, rotation 3 d | 0.003 | +1.0 | +1.9e-13 |
| TESS 2-min, granulation SHO, rho 0.2 d | 0.007 | +1.0 | -1.3e-8 |
| TESS 2-min, granulation SHO, rho 1 h | 0.03 | +0.99 | -8.8e-6 |
| TESS 20-s, SHO + SHO | 0.0005 | +1.0 | -9.8e-12 |
| Kepler 30-min, rotation 10 d | 0.013 | +1.0 | +8.7e-11 |
| Kepler 30-min, granulation SHO, rho 1 d | 0.02 | +1.0 | -1.1e-6 |
| Kepler 30-min, granulation SHO, rho 3 h | 0.16 | +0.76 | -9.0e-3 |
| Kepler 30-min, granulation SHO, rho 15 min | 2.0 | all eigenvalues negative | all negative |
| Kepler 30-min, overdamped SHO, rho 1 h | 0.5 | all negative | all negative |

PSD to rounding at small delta / rho; ~1e-2 negative at 0.16; entirely
negative definite once the exposure approaches the timescale.

**What it costs, and what fixes it.** Gradient error per posterior sigma
(worst of 16 Laplace draws) and the value's sd across them; data drawn by
dense Cholesky from the exact convolved covariance; N = 4096, 16 x 256:

| case | dense (today) | exact boundaries | sqrt, Q clamped | one block | value sd |
|---|---|---|---|---|---|
| TESS 2-min, convolved Rotation, s/e 100 | **10.4** | 1.59 | 1.67 | 1.29 | 0.32 (one block 0.19) |
| same, s/e 30 | 0.43 | 0.10 | 0.49 | 0.08 | 0.019 |
| TESS 2-min, convolved SHO + SHO, s/e 100 | 0.66 | 0.59 | **10.0** | 0.60 | 0.013 |
| same, s/e 30 | 0.06 | 0.06 | 1.86 | 0.05 | 0.002 |
| Kepler 30-min, convolved SHO, rho 1 d, s/e 100 | 0.02 | 0.02 | 0.02 | 0.02 | 0.006 |
| Kepler 30-min, convolved SHO, rho 3 h, s/e 100 | 0.00 | 0.00 | 0.05 | 0.00 | 0.001 |
| *TESS 2-min, unconvolved Rotation, s/e 100* | *0.57 (sqrt)* | *0.25* | *0.36* | *0.10* | *0.018* |

* **The default is broken for a mainstream case:** a convolved rotation on
  a bright TESS 2-min star is 10 sigma off.
* **Clamping `Q`'s negative part is not a fix:** it matches the floor for
  the rotation and is 10 sigma off for SHO + SHO, whose `Q` is negative by
  only 1.3e-8 relative -- in exactly the small directions that matter.
* **The boundary is not the main problem.** With exact boundaries the
  convolved rotation is still 1.6 sigma off and its value varies by
  0.19-0.32, ~10x the unconvolved term's on the same data. That floor is
  the formal representation itself, and it contradicts the docs' old "no
  measurable cost on data drawn from the model", which held at the
  moderate signal-to-noise it was measured at (corrected in kernels.md,
  float32.md, statistical-foundation.md and the likelihood docstring).
* **A hyperbolic (J-orthogonal) `tria` is not recommended:** it could take
  10.4 to ~1.6 at best, it is numerically delicate as the negative part
  grows, and it cannot represent the regime where the formal model is
  entirely negative definite.

**Proposed instead: a positive-definite realisation.** Every spacing is
at least the exposure (already enforced), so the exposure average `y_k`
over `(t_k - delta, t_k]` is a linear function of the state at the
*previous* sample plus noise, jointly Gaussian with that step's process
noise: `x_k = Phi x_{k-1} + w_k`, `y_k = g^T x_{k-1} + e_k`, with
`[[Q, s], [s^T, sigma^2]]` the joint covariance -- positive semi-definite
by construction, and with `P_inf = I` (the true state, whitened). The
leaves keep their rank-one form,

    A = Phi - s g^T / sigma^2,   b = s y / sigma^2,
    C = Q - s s^T / sigma^2 (PSD),   J = g g^T / sigma^2,

so pass 1, the square-root boundary and the scans apply unchanged once the
lag table carries `g`, `s` and `sigma^2`; `Q` is the unconvolved term's
(cancellation-free); the first observation is a predecessor at
`t_1 - delta` drawn from the stationary state. What changes: the forward
replay (the observation reads the previous state) and the hand-written
backward. Estimate: a NumPy float32 prototype first (~1 day) to confirm it
removes the ~10x floor; the kernel and backward 3-5 days after that.

## VI, prototype: the floor is the front end, not the representation (2026-10-25)

Fable's review of VI asked for a float64-first prototype of the
positive-definite realisation and a warning in the meantime. The warning
landed (convolved term, data rms > 30 x the median error bar). The
prototype changed the diagnosis.

**The positive-definite realisation is right, and not needed for the
floor.** Built in NumPy (`g`, `s`, `sigma^2` from stationary covariances:
`g = (1/delta) int Phi^T h` over the window, `s = c - Phi g` with
`c = (1/delta) int_0^delta Phi h`, `sigma^2 = k_conv(0) - |g|^2 + r`, and
a filter update written as two covariances, PSD in any precision): in
float64 it reproduces `.hi` to 7e-10. Value spread across the 16 draws,
one block, float32 sequential filter, TESS 2-min convolved rotation at
s/e = 100:

| | value sd |
|---|---|
| formal form, float32 tables (today's) | 0.185 (the kernel's one block: 0.19) |
| positive-definite realisation, float32 filter, float64 tables | 0.035 |
| **formal form, float32 filter, float64 tables** | **0.021** |
| unconvolved term (kernel, one block) | 0.029 |

So the ~10x floor is the float32 front end computing the formal tables
-- `Q = P - Phi P Phi^T` by subtraction, A13 -- not the indefiniteness.
Through the real kernel, with the convolved term's `matrices`,
`p_inf_blocks` and `diag_correction` bridged through float64 (forward and
VJP) and the kernel unchanged:

| TESS 2-min, s/e 100 | float32 tables, dense (today) | float64 tables, dense | float64, exact boundary | **float64, sqrt with Q clamped** | float64, one block |
|---|---|---|---|---|---|
| convolved Rotation | 10.4 / 0.32 | 8.2 / 0.12 | 0.40 | **0.48** | 0.07 / 0.027 |
| convolved SHO + SHO | 0.66 / 0.013 | 1.10 / 0.003 | 0.03 | **0.06** | 0.03 / 0.002 |

(gradient per posterior sigma / value sd.) Accurate tables remove the
floor; the dense boundary is then the remaining failure, as for
unconvolved terms; and the square-root boundary with `Q`'s negative part
clamped -- 10 sigma on SHO + SHO with float32 tables -- is within ~1.2-2x
of the exact boundary once the tables are accurate. The earlier failure
was float32 garbage in the small directions, not the genuine 1e-8
negative part.

**What that makes the fix, and its limits.**

1. *Accurate convolved tables.* Either cancellation-free closed forms
   for the formal `Q`, `P_inf` and correction, or the lag table evaluated
   in float64 on the host (cheap on a regular cadence, a handful of rows
   per chain; costly when every spacing is distinct). Fixes the floor on
   one block and the value everywhere; the MLX path (1.34 sigma, value sd
   0.32 on this case) benefits too.
2. *The square-root boundary for convolved terms*, `Q` clamped by
   `q_factor`, the replay and backward staying on the exact formal path.
   Valid only where `Q`'s negative part is tiny (delta / rho <~ 0.03 in
   the sweep above: TESS 2-min and 20-s, Kepler rotation); at delta / rho
   >~ 0.5 the formal model is entirely negative definite and clamping
   cannot work. A real-type component's correction is negative and can
   exceed the white noise, so the leaf variance `|u|^2 + r + c` could go
   non-positive: needs a guard or the dense fallback.
3. *The positive-definite realisation* stays the general answer -- every
   regime, no clamping, no correction -- at the cost of a correlated-noise
   replay and backward (estimated 6-9 days with cancellation-free
   `g`, `s`, `sigma^2`).

Not measured yet: the float64 host bridge's cost; regimes beyond the
four cases above; the negative-correction cases.

## VI, built: accurate convolved tables and the square-root boundary (2026-10-25)

Option (a1), agreed after the prototype.

* **`likelihood._Float64FrontEnd`** wraps a convolved term; its `matrices`
  and `matrices_blocks` evaluate in float64 on the host when called with
  float32 parameters, forward and VJP (`_float64_bridge`), and round to
  float32. Only those two: bridging `p_inf_blocks` and `diag_correction`
  as well changed nothing measurable (one block, 0.08 -> 0.07 sigma) and
  doubled the cost. Cost, convolved rotation at N = 4096, regular or
  gapped TESS 2-min: forward unchanged, value + gradient 48.6 -> 57.2 ms
  at 1024 chains (+18%) and 23.2 -> 31.9 at 256 (+37%); unconvolved terms
  untouched.
* **The square-root boundary for convolved terms whose members are all
  `SHOTerm` or `RotationTerm`, while the spectral `||F delta||` at the
  reference point is <= 0.5** (`_convolution_takes_sqrt`), `Q`'s negative
  part clamped by `q_factor`; the replay and the backward stay formal and
  exact. Their exposure correction is non-negative at every one of ~15k
  points swept per type inside `||F delta|| <= 8`; `RealTerm`'s is negative
  everywhere and drives the leaf variance below zero (NaN), and Matern-5/2
  measured worse than dense (0.43 against 0.00 sigma), so every other
  member type keeps the dense boundary, and asking for `"sqrt"` there is
  refused. The fully negative-definite regime (delta / rho >~ 0.5) is
  outside `||F delta|| <= 8` and already refused.
* **The warning added the same day is gone**: no convolved case measured
  is bad any more, and its text (10 sigma) was no longer true.

Measured on the library's default path, N = 4096, 16 Laplace draws:

| s/e 100 | before | **now** | exact boundaries | one block |
|---|---|---|---|---|
| TESS 2-min convolved Rotation | 10.4 sigma / value sd 0.32 | **0.63 / 0.016** | 0.33 | 0.08 / 0.020 |
| TESS 2-min convolved SHO + SHO | 0.66 / 0.013 | **0.08 / 0.002** | 0.03 | 0.03 / 0.003 |
| Kepler 30-min convolved RealTerm, Matern-5/2 (dense: real-type) | 0.00 | 0.00 | 0.00 | 0.00 |

**Why `||F delta||` too.** The clamp is harmless only while `Q`'s
negative part is small, and that part grows steeply with the exposure
over the timescale (1e-8 relative at delta / rho = 0.007, 9e-3 at 0.16).
On the library's path, convolved SHO at s/e 100, value sd across the
draws with the square-root boundary against one block's: 0.002 / 0.000
at `||F delta||` = 0.25, 0.006 / 0.000 at 0.41, **0.027 / 0.000 at 0.99,
0.074 / 0.001 at 1.98** -- the gradient stays <= 0.05 sigma throughout.
The dense boundary measured 0.00 sigma / 0.001 at 0.41, 0.99 and 1.98,
and fails only at small `||F delta||` and high s/e (10 sigma at 0.007).
So the cut is 0.5 (`_SQRT_FDELTA`), at the reference point: a chain that
walks well past it keeps the boundary chosen at construction and its
value degrades as above (0.03 at ~1, 0.07 at ~2).

The rotation's 0.63 is the unconvolved rotation's level on the same kind
of data (0.57); the gap to 0.33 is the Metal rounding V measured. Pinned
in `tests/test_boundary.py`: the front end's float32 table equals the
float64 one rounded and its gradient matches float64; the convolved
rotation at s/e 100 within 1 posterior sigma and value sd 0.04; the
boundary's resolution and refusal by member type. The positive-definite
realisation stays on file for anything this does not reach.

## Follow-on step 1: loose ends of the convolved fix (2026-10-25)

**Matern terms join the square-root boundary.** Matern-5/2 had been left
on the dense boundary on a measurement at `||F delta||` ~ 2, outside the
guard that now exists, and Matern-3/2 had not been measured. Inside the
guard, TESS 2-min, s/e 100, gradient per posterior sigma:

| | dense (was the default) | square-root, `Q` clamped | exact | one block |
|---|---|---|---|---|
| SHO (rotation 3 d) + Matern-3/2 (rho 0.2 d) | **2.06** | 0.05 (wired: 0.19) | 0.06 | 0.03 |
| Matern-5/2, rho 1 d | **1.38** | 0.04 (wired: 0.03) | 0.10 | 0.00 |
| Matern-3/2, rho 1 d | 0.27 | 0.01 (wired: 0.02) | 0.01 | 0.00 |
| Matern-5/2, rho 0.2 d | 0.11 | 0.14 (wired: 0.14) | 0.00 | 0.00 |
| Matern-3/2, rho 0.2 d; Kepler 30-min Matern-3/2, rho 1 d | <= 0.02 | <= 0.00 | 0.00 | 0.00 |

What decides it is the leaf's white-noise part `r + c`, which its
symmetric root needs positive (the process part `h Q+ h^T + c` is >= 0
inside the guard for every type, `RealTerm` included, ~13-17k points
swept each). `c >= 0` for SHO, rotation and Matern-3/2; Matern-5/2's is
negative but >= -4.6e-8 k(0), so `r + c < 0` would need s/e ~ 5000;
`RealTerm`'s reaches -0.17 k(0) and stays on the dense boundary, which
measured fine on it. `sqrt_blocks` now floors the leaf's white noise at
`r_ref / 2` as a safety net -- it cannot bind for any unconvolved term
(jitter is a variance; bit-identical, checked) or for SHO / rotation /
Matern-3/2.

**The off-model figure, re-measured.** kernels.md's "50 to 100x worse
far off-model" was measured with the float32 table and its script was
not kept. A reconstruction (white noise against a long timescale,
N = 300, mean |float32 - float64| over 8 points): with the float32 table
a convolved SHO was 20-2800x the unconvolved term's error (2e-4 at
kappa ~ 90 up to 3.6e-2 at ~9700) and a convolved `RealTerm` 4-40x; with
the float64 table both are at the unconvolved term's level, 1e-5 to 1e-4,
kernel and MLX paths. The limitation is gone; docs, the `TermConvolution`
docstring and the oracle test's note corrected.
