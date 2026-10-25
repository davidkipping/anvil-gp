# Plan: the kernel's block boundaries in the square-root information form

Status: agreed 2026-10-25; stages 0-3 done 2026-10-25.

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
   instead of paying it everywhere.
3. **The old boundary stays behind a policy switch for one release**, for
   A/B comparison and as the formal terms' path.

## Stages

| stage | what | estimate |
|---|---|---|
| 0 | Oracle and red tests | 0.5-1 day, **done** |
| 1 | Q factor table, a small Metal kernel (Cholesky, pivots clamped at 0, no floor) | 0.5 day, **done** |
| 2 | Centred square-root block pass, one thread per (chain, block); rank-1 leaf update | 2 days, **done** |
| 3 | Square-root prefix, suffix and seeds, one thread per chain; 2m x 2m `tria` | 1-1.5 days, **done** |
| 4 | Wire into `scan_loglike` forward and VJP, behind the switch; flips stage 0's expected failures | 0.5 day |
| 5 | Measure on a quiet machine: README timings, watchdog at m = 4, the exposure matrix (m = 2-4, gapped and 2-s cadences), then ChEES | 1 day + machine time |
| 6 | Docs and README | 0.5 day |

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
