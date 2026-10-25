# Plan: the kernel's block boundaries in the square-root information form

Status: agreed 2026-10-25; stage 0 done 2026-10-25.

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
| 1 | Q factor table, a small Metal kernel (Cholesky, pivots clamped at 0, no floor) | 0.5 day |
| 2 | Centred square-root block pass, one thread per (chain, block); rank-1 leaf update | 2 days |
| 3 | Square-root prefix, suffix and seeds, one thread per chain; 2m x 2m `tria` | 1-1.5 days |
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
