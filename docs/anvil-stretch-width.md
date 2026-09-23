# The stretch move leaves chains in a secondary mode, and the ensemble width inherits it

Found 2026-09-21 in stage R of anvil-gp's M4. **This is an `anvil`
(sampler) finding, not an anvil-gp one**: the likelihood is the same on both
of anvil-gp's paths and under both samplers, and the two samplers disagree.

## What was seen

The README example: SHO kernel plus a jitter, 1000 points, 1024 chains,
`EnsembleKernel` (the stretch move), 4,000 warmup steps, 600 samples thinned
by 3. It reports jitter = 0.1133 ± 0.0457, and that width is in the README's
example table. Three checks on the same data and the same target:

| jitter | width | source |
|---|---:|---|
| 0.1130 | **0.0107** | float64 Laplace approximation (u-space Hessian at the mode) |
| 0.1113 | **0.0110** | ChEES-HMC on the kernel, 1024 chains, 1500 + 500 iterations, R-hat 1.0003, min ESS 386,000 |
| 0.1133 | 0.0457 | stretch move, 4,000 warmup |
| 0.1133 | 0.0457 | stretch move, 16,000 warmup |

Four times too wide, and not moving with four times the budget.

## Where the width comes from

Broken down by chain (`res.get_chain()` without flattening, converted to
natural units):

| | chains | per-chain mean jitter | within-chain std | log posterior (median) | acceptance |
|---|---:|---:|---:|---:|---:|
| bulk | 1022 | 0.107 – 0.116 (1st – 99th pct) | 0.0107 (median) | −736 | 0.59 (mean) |
| chains 861, 983 | 2 | 1.139, 1.132 | 0.010, 0.024 | −2248 | 0.02 |

The pooled width over the 1022 bulk chains is **0.0109**, matching Laplace
and ChEES. The between-chain standard deviation of the per-chain means is
0.0446 — that is the reported width, and it is two chains. The ρ width is
inflated the same way (0.482 reported; ChEES 0.376; Laplace 0.346).

The two chains sit in a distinct mode. Chain 861 ends at σ_GP = 0.47,
ρ = 0.164 d, Q = 150, jitter = 1.14; chain 983 at σ_GP = 0.08, ρ = 0.121 d,
Q = 4.4, jitter = 1.13. The bulk medians are σ_GP = 1.08, ρ = 6.94 d,
Q = 5.1, jitter = 0.11. That is the "white-noise" solution: the GP's
timescale shrunk to about three cadence steps, the jitter carrying the
variance. Its log posterior is 1,500 units below the main mode. They moved
37 and 48 times in 600 samples.

They did not start there. From `gt.u0(1024, seed=1, spread=0.3)` chain 861
began at σ_GP = 0.94, ρ = 0.47 d, jitter = 0.31 and chain 983 at σ_GP = 0.42,
ρ = 0.48 d, jitter = 0.18 — the short-ρ side of the initial cloud. They
walked into the mode during warmup; I did not trace the path.

## Why the stretch move cannot leave it

A one-dimensional scan of the target's log density along the straight line
in u-space from chain 861's final state (λ = 0) to the bulk median (λ = 1):

    λ     0.00   0.10   0.30   0.45   0.50   0.60   0.80   1.00
    logp  -2260  -2303  -2799  -3275  -3172  -2330  -924   -734

A valley 1,000 units deep between the two modes. The stretch proposal for
walker k from complement j is `y = x_j + z (x_k − x_j)` with
`z ∈ [1/a, a] = [0.5, 2]`, which in λ reaches `[−1, 0.5]`: at its very best
the proposal lands at the bottom of the valley (λ = 0.5, −3172), never on
the far side. Acceptance is `exp(−900)`. No budget changes that, which is
what the 16,000-warmup run shows. This is the textbook limitation of
affine-invariant ensemble moves on multimodal targets; what is new here is
that it happened on a four-parameter GP fit with a single injected signal,
at a rate of 2 in 1024, and that the run summary looked converged (max
R-hat 1.037, min ESS 20,000).

From the same 1024 starting points, no ChEES chain ended in the secondary
mode. HMC does not cross a barrier like this either; presumably the gradient
from those starts points at the main mode while the early ensemble
proposals, drawn from a broad initial cloud, do not. Not verified.

## What anvil could do

1. **Report it.** The run already has what is needed: per-chain acceptance
   (0.020 for the two chains against a 5th percentile of 0.571) and
   per-chain log probability (1,500 units below the ensemble median). A
   chain whose acceptance is a tenth of the ensemble's, or whose median log
   probability sits more than a few tens of units below the ensemble
   median, is trapped, and the flat-chain moments should not be quoted
   without saying so. R-hat 1.037 was the only tell.
2. **Warmup-only rescue.** During warmup, re-seed walkers whose log
   probability is far below the ensemble's from a random walker in the
   bulk (emcee users do this by hand). After warmup nothing may be touched.
3. **A move that can cross.** A differential-evolution or "walk" move, or
   tempering, reaches the far side; the stretch move by construction does
   not.

anvil-gp owes nothing here: both of its paths return the same likelihood at
every point on that scan, and its gradient is what let ChEES settle the
question in 80 s. The README's example table now says which width to
quote.
