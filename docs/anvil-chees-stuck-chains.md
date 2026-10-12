# Three ChEES chains whose recorded log-probability disagrees with their position

A **proposed investigation** for `anvil`, written 2026-10-11 against anvil
0.6.0 (`3758996`) and MLX 0.32.2. This is not a diagnosis. It records what
was seen, two competing explanations, and a short triage that would tell
them apart. **anvil-gp needs nothing here**: its likelihood stayed finite
and the healthy chains kept sampling, which is all anvil-gp promises.

> **Correction (2026-10-11), after anvil's reply.** An earlier version of
> this brief said 29 of 32 chains "recover". That counted a chain as
> recovered once its likelihood was finite, which is not convergence. Measured
> at the end of the run, **all 32** dead-started chains sit near −17,500 in
> recorded log-probability against about −900 for the healthy half, a gap of
> ~16,600; none is anywhere near the bulk. The three discussed here are only
> the ones where compiled and uncompiled evaluation disagree enough to flip
> the value to the failure value. See
> [`anvil-chees-stuck-chains-reply.md`](anvil-chees-stuck-chains-reply.md).

## Why it matters

Three ChEES chains are stuck, and **anvil's new lost-chain check cannot see
them**, because ChEES's own record of their log-probability is a feasible
value while their positions are not. The diagnostics added in 0.6.0
(`warmup_report`'s `LOST CHAINS`, `diagnose(..., log_prob=...)`) read that
record, so a chain whose record is wrong is invisible to them. If the record
is wrong in general rather than only here, the same gap applies to every
ChEES run.

## What was seen

The setup: anvil-gp's SHO target, 600 points, 64 ChEES chains
(`max_leapfrog=16`), 300 warmup and 50 samples, seed 3. Half the chains
start at `u = 30` in every coordinate, inside the region where the GP
likelihood cannot be evaluated in float32. The run is
[`repro/anvilgp_chees_stuck_chains.py`](repro/anvilgp_chees_stuck_chains.py),
which needs anvil-gp. 29 of the 32 dead-started chains reach a *finite*
likelihood; **none converge** (see the correction below). Three are still at
the failure value:

| | value |
|---|---|
| stuck chains | `0, 4, 25` |
| their positions, first two coordinates | (23.49, 28.71), (23.66, 29.60), (23.26, 29.06); largest coordinate 30.7 |
| **recorded log-probability** (`warmup_trace["log_prob"]`, and the first and last sampled draws) | **about −17,000** (−17204, −17357, −16992) |
| **target log-probability evaluated afresh at those final positions** | **about −6.8e30**, the failure value |
| `warmup_report` | `INCONCLUSIVE` -- not `LOST CHAINS` |
| `lost_chains(...)` on the recorded values | flags nothing |
| `n_divergent` | 12 |

Three things make this worth a look rather than a shrug:

- **The record and the position disagree by 26 orders of magnitude.** The
  end-of-warmup record and the first and last sampled draws all give about
  −17,000 for these chains, while the target says
  their position is a failed evaluation.
- **It is independent of anvil-gp's failure value.** anvil-gp reports a
  failed evaluation two ways: the old constant `−1e30`, and the current
  sloped value (see `docs/anvil-warmup-lost-walkers.md`). Both give the same
  three chains, the same positions and the same 12 divergences. That fits the
  record: inside the run these chains never registered as failed, so the
  failure value never came into play.
- **It predates 0.6.0.** The same count, largest coordinate and divergence
  count were seen on 0.5.0, so it is not caused by the new finite-position
  check.

The positions sit near the edge of float32 feasibility. On the all-equal
diagonal the likelihood first fails at `u = 25`, and these chains have first
coordinates of 23.3–23.7, with others between 28 and 31.

## Two competing explanations

**H1: bookkeeping.** ChEES updates the position and the stored
log-probability under different conditions. For example, one is written with
the full accept mask and the other with a mask missing a term. Then a chain
can move to a failed position while keeping the log-probability of a feasible
one. This would be a ChEES bug, independent of the target.

**H2: compiled and eager evaluation disagree at a float32 feasibility
edge.** The stored value comes from the kernel's compiled step, and my fresh
evaluation was eager. anvil's own `75d89f5` notes that compiled and eager
evaluation differ slightly (its example: `mx.random.normal` 6e-6 apart for
the same key). At a feasibility edge a slight difference can decide between
an ordinary value and a failed one. Then the target is not a single-valued
function of position near the edge, and the record is "right" only for the
compiled path. This would not be a ChEES logic bug, but it would still make
the diagnostics blind wherever it happens.

Neither has been tested. The hypothesis in the earlier brief -- that ChEES's
divergence veto treats a finite failure value as an ordinary state -- now
looks unlikely, since inside the run these chains do not seem to be at the
failure value at all.

## Proposed triage

Ordered so each step can end the investigation. About 15 to 30 minutes in
total.

**Step 1 -- H1 or H2.** At the three final positions, evaluate
`target.log_prob` three ways: eagerly, through `mx.compile(target.log_prob)`,
and through the same compiled function the kernel uses for its step.

- Compiled gives about −17,000 and eager gives about −6.8e30: **H2**. Go to
  step 3.
- All fresh evaluations agree at about −6.8e30: **H1**. Compare how
  `chees.py` writes the position and the log-probability on accept and on
  veto. With the finite-position term added in 0.6.0 the accept expression
  now appears in two places, so check both paths use the same mask. Go to
  step 2.
- All fresh evaluations agree at about −17,000: my eager check was wrong.
  Then the chains are simply in a bad but feasible spot. Go to step 2 to see
  why they do not move.

**Step 2 -- why the chains do not move.** From the same run:
`extras["divergent_per_chain"]` for chains 0, 4 and 25, the final step size
(`warmup_trace["step_size"][-1]`), and their acceptance. This separates
"every proposal is vetoed as a divergence" from "steps too small to travel".

**Step 3 -- does it reproduce without anvil-gp?** Run
[`repro/anvil_lost_walkers.py`](repro/anvil_lost_walkers.py) with
`anvil.ChEESHMC(target, max_leapfrog=16)` in place of the ensemble kernel.
That target fails at a clean radius, with no rounding-sensitive edge. If it
does not reproduce there, that supports H2, and a reproducer needs a target
whose failure boundary is itself rounding-sensitive. If it does reproduce,
anvil can own the whole investigation with no anvil-gp context.

**Step 4, only if H2 -- why `lost_chains` did not flag about −17,000
either.** Against a healthy bulk of a few hundred, a chain at −17,000 looks
far outside ten robust sigmas. The likely reason is that the recovering chains
spread the ensemble's log-probabilities widely at the end of warmup, which
inflates the median absolute deviation. Worth confirming, since it decides
whether the relative test can catch a stuck-but-finite chain at all.

## What each outcome would justify

| outcome | suggested response |
|---|---|
| H1 | **A fix.** A stored log-probability that disagrees with the stored position corrupts every later accept decision for that chain, as well as the diagnostics. |
| H2 | **No sampler fix, but close the diagnostic gap.** For example, have the diagnostics re-evaluate the log-probability at the final positions with the eager path, or flag chains whose stored and re-evaluated values disagree. Also worth documenting that a target can be multi-valued at a float32 feasibility edge. |
| Neither reproduces, or only with this artificial start | **Drop it.** All 32 chains start deep in the failure region, and real initialization starts near the reference point. Note that the 0.6.0 diagnostics can miss a chain whose record is feasible. |

## Not part of this brief

Everything in `docs/anvil-warmup-lost-walkers.md` is resolved by anvil 0.6.0
and verified from anvil-gp. This brief is the one thread it left open.
