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
| E | Backward kernel, single block | 2 | Opus | matches `adjoint.py` on all five inputs; matches autodiff |
| F | Backward kernel, all blocks, checkpointed | 3 | Fable | == E at one block, == `adjoint.py` at every block count |
| G | Wire up, measure, ChEES end to end | 1 | Opus | gradient rows in `verify_readme.py`; joint corner improves |
| R | Adversarial review: what did I not see | 0.5 | Fable | -- |

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

## Expected outcome

A differentiable fused kernel: a gradient at ~2.5-3x the forward, so ~7-9 ms
at 1024 chains x N = 1e4 against 880 ms on the MLX path today. ChEES-HMC
becomes something to run rather than a documented workaround needing a
process-global `mx.disable_compile()`.

Secondary: the value comes from the backward's own replay, which should
close the joint high-amplitude long-correlation corner where the MLX path
is currently off by 1.5 log-units; and the kernel becomes the default for
both value and gradient at m <= 4, returning the MLX graph to being a
reference rather than a production path.

**It does not close the >=100x gate.** That gate is the forward likelihood
against a 12-core celerite2 pool, where we are at 20-37x, and M4 does not
touch the forward.
