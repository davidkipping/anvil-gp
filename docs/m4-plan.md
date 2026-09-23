# M4: the fused Metal backward kernel

Locked 2026-09-20, after three independent reviews of the design. Each
stage names the check that gates it; a stage is not done until that check
passes. Where a number appears it was measured, and the script is named.

## What the design is, and what it replaced

An earlier version of this plan chained the backward blocks: block *k*'s
reverse sweep needed the `(m_bar, P_bar)` that block *k+1* produced, and
the chain was to be broken by composing each block's backward sweep as an
**affine map** on that state, in three passes.

The affine claim is true. It was verified to float64 roundoff at m = 1..4,
at every span length tested, with and without the innovation-variance
clamp firing. **The design built on it was still wrong.** The chain is not
structural: a block's incoming `(m_bar, P_bar)` has a closed form in that
block's exit filter state and the *suffix* element of everything after it,

    r      = G^T (eta_s - J_s mu)          # `combine`'s own `gt_eta`
    m_bar  = cot * r
    P_bar  = (cot/2) (r r^T - Jt)          # `combine`'s own `jt`

with `G = (I + Sigma J_s)^-1` and `Jt = (I + J_s Sigma)^-1 J_s`. Both
pieces are already computed inside `elements.combine`. Verified against the
chained sweep at 1e-15 relative, m = 1..4, every block length tested.

So the backward blocks are **independent**, exactly as the forward blocks
are. Three passes collapse to one. Cost falls from ~8x a forward sweep to
~2.5x, and unlike the affine design it survives m = 4.

## Two constraints that came out of the reviews

**Block count is an accuracy knob, not just a parallelism knob.** The seed
amplifies the block elements' float32 error by ~1e3 through the
cancellation in `eta_s - J_s mu`. Four of the five parameter adjoints are
untouched; `dh` is not. Measured on-model at sigma_GP/yerr = 100,
rho/dt = 1000, N = 8192 (`scratchpad/f32seed.py`), against float64:

    blocks    dh, seed route    dh, autodiff today    ratio
       4          7.5e-3              2.9e-4           26x
       8          5.7e-3              2.9e-4           19x
      32          3.2e-2              2.9e-4          110x

Keep B near 12. That is also where the forward's occupancy saturates, so
the constraint is free -- but it must be written down, because the instinct
when tuning a kernel is to add blocks. Compensating the `J`/`eta`
accumulations only halves the damage (2.9e-3 with them exact), so it is not
the fix. For HMC this is efficiency, not correctness: acceptance uses the
value.

**The clamp must protect the same thing in both forms.** See stage 0.

## Stages

| # | Stage | Days | Model | Gate |
|---|---|---|---|---|
| 0 | Clamp the non-physical term, not the total | 0.5 | Opus | **DONE** |
| A | Split the log-determinant in the forward kernel | 0.5 | Opus | **DONE** -- 11x at the default block |
| B | Thread and residual layout | 1 | Opus | **DONE** -- bit-identical; 1.16-1.24x whole-path |
| C | Block reduction into the kernel | 1.5 | Opus | **DONE** -- scalar boundary; 1.10-1.68x |
| D | Suffix scan and closed-form seed, in MLX | 1.5 | Opus | **DONE** -- worst 2.0e-15 on random problems, 6.6e-14 on real terms with gaps and duplicates |
| E | Backward kernel, single block | 2 | Opus | **DONE** -- 1e-7 at k = 2 (arithmetic exact), < 1e-4 float32 over four terms x three cadences |
| F | Backward kernel, all blocks, checkpointed | 3 | Fable | **DONE** -- == E to 1.5e-7; == spec to 8.7e-5 at blocks up to 256 (9e-4 in the gapped m = 4 corner; the 2e-5 first written here was the duplicate-timestamp cadence alone); m = 4 runs at N = 1e5 |
| G | Wire up, measure, ChEES end to end | 1 | Opus | **DONE** -- gradient 99x at N=1e4, 264x at N=1e5; ChEES 4.0 s vs 126 s |
| R | Adversarial review: what did I not see | 0.5 | Fable | **DONE** -- see "Stage R, measured": one dead-gradient point, one un-differentiated knob, one unbounded buffer, one port mismatch, three vacuous tests, one wrong posterior width |

**Why C is a prerequisite and not an optimisation.** The custom function
currently wraps `scan_blocks`, which returns six element fields per block,
so its VJP would have to accept cotangents on all six. Moving the block
reduction into the kernel collapses the output to one scalar per chain,
which is exactly the cotangent `adjoint.py` was derived for. It also
happens to be worth 2.2 of the 3.0 ms at N = 1e4.

**Never** reconstruct forward state by running the filter backwards through
the propagator. That divides by `Phi`, which underflowed on the way forward
at exactly the lags where it would now overflow: every gap in the data.
Replay forward, always.

## Stage F notes, for stage G

**The seed scans had to move into the kernel, and did.** The plan put them
in stage G; measured at N = 1e5 the MLX `prefix_scan`/`suffix_scan`/
`block_seeds` cost 33 ms at m = 2 (against a 33 ms backward kernel) and
**1.44 s** at m = 4 (against 211 ms), and the 1.4 s version is also what
tripped the GPU watchdog in an earlier timing loop, not the kernel.
`block_seeds_kernel` does prefix, suffix and seeds in one launch, one
thread per chain: 0.75 ms at m = 2, 7.7 ms at m = 4.

Whole gradient path (forward elements -> seeds -> backward), production
size, one window:

    m = 2, N = 1e5, 1024 chains, 13 blocks:   44 ms   = 3.9x the forward
    m = 4, N = 1e5,  512 chains, 98 blocks:  270 ms   = 4.8x the forward

(Stage F's window.  The stage-G verifier measured 58 ms for the m = 2 row
in its own window, and that is the number in the Outcome table and the
README; the two windows differ, not the code.)

The backward kernel alone is 2.9x / 3.8x the forward, as designed.  The
per-lag accumulation is flush-on-change: a run of one lag accumulates in
registers and is written to the thread's own ``(chain, block, lag)`` slot
when the lag changes, so there is no runtime-indexed register array and no
cap on ``n_lags``.

**The ill-conditioned corner is named.** Rotation (m = 4) is where the
seed's float32 sensitivity shows: seed kernel and MLX seeds differ at 1e-4
(2e-5 on duplicates, 1e-4 across a 20-unit gap), and the end-to-end
adjoints at up to 9e-4 across the gap, against ~1e-6 for every other term
and cadence.  The tests hold m = 4 to a stated looser bound rather than
pretending it is not there.

Chain-blocking (splitting chains across launches) was measured and is not
needed for the watchdog at these sizes, and *hurts* when it is not needed
-- m = 3 at 4 launches of 128 chains took 255 ms against 68 ms in one.

## Stage E notes, for stage F

The kernel keeps every forward state of its block, `(m + m^2)` floats per
observation, so `MAX_BWD_BLOCK = 32`.  Stage F replaced that with
checkpoints and lifted the bound; stage R deleted the stage-E kernel as
dead code (git history has it).

It also writes `dphi`/`dq` **per observation**, which is fine for a test
and impossible at production N: `(n_chains, npts, m, m)` is 1.6 GB at
N = 1e5, 1024 chains, m = 2.  Stage F must accumulate them **per lag**
instead -- `(n_chains, n_blocks, n_lags, m, m)`, which is under a megabyte
at three lags -- and that is the change most likely to hit the register
budget, since the accumulator is indexed by a runtime lag id.

Measured agreement against the float64 specification, float32 kernel:

    k = 2, one block (nothing accumulates)      1e-7   -- arithmetic exact
    k = 24-30, four terms x three cadences    < 1e-4   -- float32 accumulation

The second is dominated by `dh`, which sums over every observation, and by
the gapped cadence, where the propagator underflows across the gap. For
scale, the MLX float32 gradient's own `dh` error at the high-amplitude
corner is ~2.9e-4.

## What failure looks like, stated in advance

* ~~Stage D's seed not matching at 1e-12 in float64 means the design is
  wrong.~~ **Cleared 2026-09-20**: worst 2.0e-15 over m = 1..4 and five
  block configurations, 6.6e-14 over five real terms crossed with regular,
  gapped and duplicate-timestamp cadences.  `independent_backward` in
  `anvilgp/adjoint.py` is the specification stages E and F port.
* Stage F unable to hold the register budget at m = 4 means a kernel
  gradient for m <= 2 with m = 3, 4 on the MLX path. That is still most
  real light curves.
* `dh` costing measurable HMC efficiency rather than just paper accuracy
  means fewer blocks, which the occupancy data says is free.

## Outcome, measured

|                                | MLX      | Metal    | speedup |
|--------------------------------|---------:|---------:|--------:|
| value+grad, N=1e4, 1024 chains |   884 ms |   8.9 ms |     99x |
| value+grad, N=1e5, 1024 chains |   15.4 s |    58 ms |    264x |
| peak memory, N=1e5             | 34.5 GiB | 1.61 GiB |      21x |
| ChEES, 300 iters, 64 chains    |    126 s |    4.0 s |     31x |

The gradient matches float64 central differences to ~1e-7 on every
parameter including the jitter, and the two paths agree to ~1e-6.

**One prediction in this plan was wrong.** It said M4 would remove the
need for `mx.disable_compile()`. It does not: the over-fused group is in
`terms._sho_matrices`, the front end both paths share, not in the element
algebra the kernel replaces. Tested, not assumed. Splitting the VJP at the
term-matrix table is what would remove it.

## Expected outcome, as locked, with what happened

A differentiable fused kernel: a gradient at ~2.5-3x the forward, so ~7-9 ms
at 1024 chains x N = 1e4 against 880 ms on the MLX path today. ChEES-HMC
becomes something to run rather than a documented workaround needing a
process-global `mx.disable_compile()`.

> **Measured: ~5x the forward** (8.9 ms against 1.67 ms at N = 1e4; 58
> against 11.5 ms at N = 1e5). The absolute target of 7-9 ms was met and
> the ratio was missed by 2x, because the forward got faster in the same
> stages (A-C) and the backward replays its block twice. ChEES runs, at
> 4.0 s against 126 s -- but still under `mx.disable_compile()`, which M4
> did not remove (see "Outcome, measured").

Secondary: the value comes from the backward's own replay, which should
close the joint high-amplitude long-correlation corner where the MLX path
is currently off by 1.5 log-units; and the kernel becomes the default for
both value and gradient at m <= 4, returning the MLX graph to being a
reference rather than a production path.

> **Neither delivered.** The value still comes from the forward's leaf
> form; the replay is the backward's alone, and the two are the same
> function to 1.5e-3 log-units in float32 (the kernel already held the
> joint corner to 0.02, so there was nothing to close). `GPPolicy.metal`
> still defaults to `False`; with the lag-buffer limit found in stage R
> (below) that default is a separate decision, not a formality.

**It does not close the >=100x gate.** That gate is the forward likelihood
against a 12-core celerite2 pool, where we are at 35-42x, and M4 does not
touch the forward.

## Stage R, measured

Two reviewers (an audit of the stages against their claims, and an attack
on the gradient's range and contract) plus my own checks.  Everything below
is fixed on `main` and pinned by a test that fails with the fix reverted.

**Three tests passed with their subject deleted.** Stage 0's clamp was
unpinned entirely: reverting it at all four sites left 279 tests green,
because the existing test's fixture never drove `h Q h^T` negative.
`test_the_differentiation_boundary_encloses_the_reduction` spied on
`scan_loglike`'s output, which is a scalar per chain whether or not the
reduction is inside the boundary.  `test_chees_runs_on_the_fused_kernel`
asserted only that the chains were finite and had moved, which a zero
gradient satisfies.  All three now assert the property: a negative-variance
fixture that NaNs without the clamp, the `custom_function`'s own output
shape, and ChEES acceptance above 0.5 with the jitter width in the Laplace
window.

**A healthy value with an exactly-zero gradient (HIGH).** At `sigma/yerr =
1e4`, `rho/dt = 1000`, `Q = 500` (the shipped prior's cap) the kernel value
is right to 2e-4 and its gradient was `[0, 0, 0]`: `backward_ckpt` returned
NaN and the input guard zeroed it, which is the right answer only when the
value was rejected.  Localised to the replay's first observation: with
`cp = I` and `h.h / r ~ 1e8`, `1 - h.h/(h.h + r)` rounds to 0 or a negative
ulp along `h` instead of `1e-8`, and an undamped propagator grows it until
the replay overflows -- block 0 only, every block length, every sub-block
length, and gone with `lag[0] = 1` or a rescaling of the data.  The forward
never forms this covariance (it carries the normalised rank-one leaf),
which is why the value survived.  Fixed with the Joseph-form update
`(I - K h) cp (I - K h)^T + rk K K^T`, PSD by construction and equal to the
spec's `cp - K ch^T` in exact arithmetic; the gradient at that point is now
within 0.4% of float64.  Two `m^3` products per replay step; the gradient
benchmark rows were re-measured afterwards.  Note the wider point: HMC with
a wrong gradient still targets the right distribution (leapfrog stays
volume-preserving and reversible for any position-dependent force), so a
dead lane costs acceptance, not correctness.

**`s_floor_rel` was differentiated by one path and not the other.** The
kernel forward applies it; `backward_ckpt` has no such parameter, so at a
value where it binds the gradient is wrong by a factor of 3 at 1e3 and 982
at 1e5.  Since stage 0 it cannot bind below 1, so `GPPolicy` now refuses
`s_floor_rel >= 1` with the reason.

**The per-lag adjoint buffer was unbounded.** `2 x chains x blocks x lags x
m^2` floats: 21 MB on a regular cadence and 2.2 GiB on a random one at 64
chains x N = 2e4, linear in chains, 42 GiB at production size.  Nothing
checked it.  The VJP now refuses over `MAX_ADJOINT_BUFFER_BYTES` (2 GiB) with the
arithmetic and the fix in the message.  The structural fix -- Jacobian
tables by lag, parameter gradient accumulated in-kernel -- is in the
README's "What is left".

**A port mismatch in the Cholesky pivot floor.** The reduce and seed
kernels floored the pivot at `eps * trace / m` with the *signed* trace,
where `linalg.diag_floor` clamps it at zero.  For a block whose `C` has
negative trace (a non-PSD model, which the stage-0 fixture builds) the
kernel's `sqrt(max(d, fl))` saw a negative floor and returned NaN, while
the MLX path returned a finite value.  Four sites, now `max(tr, 0)`.

**The example table's jitter width is wrong, and it is the stretch move.**
On the README example (1000 points, 1024 chains) a float64 Laplace
half-width at the mode is 0.0107; ChEES on the kernel, converged (R-hat
1.0003, min ESS 386,000), gives 0.0110; the stretch move gives 0.0457 at
4,000 warmup steps and 0.0457 at 16,000.  Broken down by chain: 1,022
chains give 0.0109 and two sit in a secondary "white-noise" mode (jitter
1.13, rho ~ 3 cadence steps, log posterior 1,500 below the bulk) with 2%
acceptance.  A scan along the line to the bulk shows a 1,000-unit valley,
and the stretch proposal reaches at most halfway, so no budget frees them.
Reported for anvil in `docs/anvil-stretch-width.md`.

**Attacked and held**: lane isolation (healthy lanes bit-identical beside
a dead one), mean model and per-instrument offsets (kernel == MLX to 1e-5;
both 1.3e-4 from float64, shared, so the front end), duplicates at N =
5000 (1e-7), compile behaviour (identical on both paths; `RealTerm`
compiles its gradient, `SHOTerm` aborts, confirming the front end),
memory at N = 1e5 (1.61 GiB at m = 2, matching the README; 2.59 at
m = 4), and the leaf-vs-filter clamp difference, which no configuration
in ~140 could make bind.  That last one is latent, not live.

**Housekeeping**: stage E's single-block kernel (`_SRC_BWD`,
`backward_blocks`, `MAX_BWD_BLOCK`, two tests) deleted as dead; the stage
F gate corrected from 2e-5 to 8.7e-5 (the smaller number was one cadence);
the 44 ms / 58 ms discrepancy above annotated as two windows; every
"forward-only until M4" in the tree rewritten.
