# Reply: the three stuck ChEES chains

From anvil, answering `anvil-chees-stuck-chains.md`. Measured against
anvil 0.7.0 (`v0.7.0`) with your venv, which runs anvil from the source tree,
so you already have every change below.

**Short version.** Your H2 is right: the stored and the fresh
log-probability disagree because compiled and eager evaluation of your
likelihood round differently at the float32 feasibility edge. It is not a
ChEES bookkeeping bug. The triage then turned up something larger than the
three chains: **one chain stranded near that edge collapses ChEES's shared
step size for every chain**, and the root of that is a property of your
likelihood that only you can change. anvil has fixed the parts that were its
own; the rest is below, with what we recommend.

## What the triage found

Your steps, in order:

1. **H1 or H2.** At the three final positions, `mx.compile(target.log_prob)`
   gives about −17,000 and eager evaluation gives about −6.8e30. The
   kernel's own `log_prob_and_grad` agrees with eager. Re-evaluating the three
   alone, or inside a reversed 64-batch, gives the same eager value, so it is
   not batch order. **H2.**
2. **Why they do not move.** The shared step size had collapsed to 3.8e-8,
   so nothing moved. The run as a whole was not converged.
3. **Without anvil-gp.** On anvil's radius target, whose failure boundary is
   clean, the stored and the eager values disagree on 0 chains. The
   disagreement needs a rounding-sensitive edge like yours.
4. **Why `lost_chains` missed −17,000.** Not the three chains specifically:
   all 32 chains you started at `u = 30` sat near −17,500 against about −900
   for the other half. A 32/32 split puts the median between the two groups
   and inflates the robust spread.

One correction to the write-up: **"29 of 32 recover" means feasible, not
converged.** All 32 were stranded about 16,000 log-units below the healthy
chains, and the three you saw are the ones where the edge flips the eager
value to failure.

## The larger finding: one chain collapses the step size

We swept how many of 64 chains start at `u = 30` (300 warmup, seed 3):

| chains started in the failure region | final step size |
|---|---|
| 0 | 1.0e-2 |
| **1** | **7.3e-7** |
| 2 | 1.1e-6 |
| 4 | 2.0e-7 |
| 8 | 1.2e-7 |
| 16 | 1.1e-8 |

A single chain takes the run down by four orders of magnitude, and the
other 63 chains are stuck behind it.

**Why.** ChEES adapts one step size for all chains from the *harmonic mean*
of their acceptance, which deliberately lets the worst chain pull the step
down until it mixes. Tracing the one bad chain through warmup:

- at t = 1 it leaves the failure region and lands at a *finite* −18,000;
- at a step size of ~7e-4 every other chain accepts and it accepts nothing;
- by t = 100 the step size has shrunk to 2.5e-7, where it accepts again.

So that chain sits in a band near your float32 feasibility edge where the
density is rough enough that only steps of order 1e-7 are accepted, the same
band in which compiled and eager evaluation disagree. The harmonic mean
serves it faithfully, at the expense of everything else.

## What anvil changed

- **0.6.1:** the lost-chain check reads an eager re-evaluation at the final
  positions as well as the stored value, so a chain whose record is feasible
  while its position is not is no longer invisible. `warmup_trace` gains
  `log_prob_eager`; runs gain `extras["lost_chains"]`.
- **0.6.2:** `warmup_report` distinguishes chains at a failed evaluation from
  finite chains stranded far below the bulk, instead of blaming a failure
  value for both.
- **0.7.0:** chains at a non-finite log-density or at or below the −1e29
  failure floor are left out of the step-size statistic. One such chain used
  to drive the step size to exactly 0. Healthy runs are bit-identical.

**What 0.7.0 does not fix is your case.** Your bad chain is finite after its
first step, so it is a straggler by any definition anvil can apply from
outside, and the sweep above is unchanged by 0.7.0.

## Exposure

- **Measured: low, at 64 chains initialized sensibly.** Every run started
  near the reference point kept a step size of ~1e-2. The collapse needed a
  chain started deep in the failure region.
- **It cannot bias a posterior.** Adaptation only sets the step size, and the
  frozen step size used for sampling is valid at any value. The worst case is
  a wasted run.
- **It is visible.** A step size near 1e-7, chains that barely move, poor
  R-hat and ESS, and `warmup_report` naming the stranded chains.
- **Not measured: your production configuration.** Your own `_sanitize`
  docstring says "with 4096 chains, warmup *will* visit float32-infeasible
  corners". If even one of those chains ends warmup in the rough band, the
  table above says the whole run's step size goes with it. We tried 4096
  chains with your `u0` and could not finish one 300-iteration run inside a
  30-minute limit, so **this is the one measurement we are asking you for.**

## What we recommend, on your side

1. **Measure the production case.** One run at your real chain count and
   initialization; record `res.final_params["step_size"]`,
   `min(res.warmup_trace["step_size"])`, `res.extras["lost_chains"]` and
   `anvil.warmup_report(res)`. A step size within a factor of a few of a
   small clean run means you are not exposed. A collapse to ~1e-6 or below
   means you are.
2. **If exposed, the root fix is the edge itself.** A band where only 1e-7
   steps are accepted, and where compiled and eager evaluation disagree, is
   hard for any sampler. Possibilities, in your terms: extend the region
   where the likelihood is well-conditioned, or make the failure boundary
   sharp enough that chains are rejected outright rather than stranded on a
   rough slope. The second turns these chains into the case 0.7.0 already
   handles.
3. **Keep initializing near the reference** (`spread=0.05` in `u0`). That is
   why the measured exposure is low.

## What anvil could still do

Replace the harmonic mean with a statistic that stops honouring the very
worst few chains, for example one trimmed of the worst ~5%. That would make
anvil tolerate a straggler like yours, but it changes step-size adaptation on
every target and weakens the protection that legitimate stragglers, such as
Neal's funnel's neck, rely on. We have not done it. If your production
measurement shows real exposure that you cannot remove on your side, tell us
and we will gate it against your sweep, the funnel's divergences and the
healthy targets' ESS.
