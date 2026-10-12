# The production measurement anvil asked for

anvil's reply ([`anvil-chees-stuck-chains-reply.md`](anvil-chees-stuck-chains-reply.md))
found that one chain stranded near anvil-gp's float32 feasibility edge can
collapse ChEES's shared step size for every chain, and asked for one
measurement from this side: **is a production-scale run exposed?** This is
that measurement, from 2026-10-11, anvil 0.7.0, MLX 0.32.2, M2 Max.

## Answer

**At the default initialisation, no. At wide initialisation, yes, and
gradually rather than suddenly.** The step size is a smooth function of how
widely the chains start, from 0.29 at the default down to a collapse of
2.9e-7 at the wide end. Chain count (1024 or 4096) makes little difference.

**Later the same day:** the collapse at the wide end is one chain reaching
the float32 edge through the unbounded GP amplitude, and it is a trap: 3000
warmup steps do not clear it. The milder cells are slow warmup from a wide
start, which a longer warmup clears. See
[What strands the chains](#what-strands-the-chains-added-later-the-same-day).

## Setup

The tutorial problem: SHO kernel (truth 1.2, 5.0, 4.0), 2000 points at 0.02
spacing, error bars 0.15. `make_target(..., policy=GPPolicy(metal=True))`, so
the fused kernel. ChEES-HMC, `max_leapfrog=32`, 300 warmup and 300 samples.
Chains initialised by `gt.u0(n, spread=...)`, whose default `spread` is 0.05.
Three seeds per cell, each seed changing the initial positions and the
sampler's random stream.

| chains | start spread | final step size | min step during warmup | lost chains at end | divergences |
|---:|---:|---|---|---|---:|
| 1024 | **0.05 (default)** | 2.9e-01 to 2.9e-01 | 7.8e-03 to 7.9e-03 | 0 to 0 | 0 |
| 1024 | 0.3 | 3.0e-02 to 4.0e-02 | 3.5e-04 to 7.8e-03 | 0 to 1 | 0 |
| 1024 | 0.5 | 1.2e-02 to 1.7e-02 | 3.1e-04 to 3.4e-04 | 4 to 6 | 0 |
| 1024 | 0.7 | 1.1e-03 to 1.3e-03 | 4.9e-08 to 3.2e-04 | 41 to 44 | 0 |
| 1024 | 1.0 | **2.9e-07** to 9.4e-04 | 9.3e-08 to 1.1e-04 | 62 to 81 | 100 |
| 4096 | **0.05 (default)** | 1.5e-01 to 2.8e-01 | 7.4e-03 to 7.9e-03 | 0 to 0 | 0 |
| 4096 | 0.3 | 2.5e-02 to 5.9e-02 | 3.5e-04 to 7.5e-04 | 2 to 5 | 0 |

"Lost chains at end" is `res.extras["lost_chains"]`. Spreads 0.5 and
0.7 were not run at 4096 chains. 4096 chains at spread 1.0 was run once (seed
1): final step 7.6e-7, 286 lost, 96 divergences. The run time is not a
constraint: 4096 chains take 17 to 30 seconds on the fused kernel.

## What it shows

- **The default initialisation is not exposed.** At spread 0.05 the step
  size is 0.15 to 0.29 at both 1024 and 4096 chains, with no lost chains and
  no divergences, in all six runs. This is the documented, recommended way to
  start (`GPTarget.u0`), and the one `docs/tutorial.md` and the README use.
- **Exposure grows steadily with initial spread.** Against the
  default's 0.29, the final step size is about 8 times smaller at spread 0.3,
  about 20 times at 0.5, a few hundred times at 0.7, and between 300 times
  and a million times at 1.0. The number of chains anvil counts as lost goes
  from 0 or 1 to a handful at 0.5 and to dozens at 0.7 and above. I have
  not measured whether any chain *starts* somewhere the likelihood cannot be
  evaluated in those cells; the lost chains are ones found stranded at the
  end.
- **A step size that small is a slow run, not a wrong one.** At spreads 0.7
  and 1.0 acceptance reads 1.00 because the steps are too small to be
  rejected, which is the signature anvil's reply describes. As anvil says,
  adaptation only sets the step size, so this cannot bias a posterior.
  **Not measured: how much effective sample size is actually lost.** The
  step size says the trajectories are shorter; it does not say by how much
  the sampling efficiency drops, because ChEES also adapts trajectory length.
  Treat "slow" as established and "how slow" as open.
- **anvil's 30-minute timeout was the default evaluation path.** `GPPolicy`
  defaults to `metal=False`. On the same problem, 256 chains for 15
  iterations takes 10.6 s there against 0.3 s on the fused kernel, which
  extrapolates to about two hours for 4096 chains x 600 iterations. Measured
  on the fused kernel it takes 17 to 30 seconds.
- **The final step size is the one that governs sampling.** Adaptation ends
  with warmup and the sampling phase uses that frozen value, so the "final
  step size" column describes the whole run, while "min step during warmup"
  shows how far adaptation was dragged on the way.

## Something for anvil, from the same runs

`warmup_report` returned `LOST CHAINS` for runs in which the chains it named
were **not** lost. I inspected two: 4096 chains at spread 0.05 (seed 1), and
1024 chains at spread 0.3 (seed 1). In the first, one chain was flagged at the end of warmup: 37 robust sigmas below the median
(recorded log-probability -1274.6 against -1225.6), at a normal position
(`|u|` 1.24 against a median of 1.19), and not at the failure value. At the
end of sampling the same chain had recorded and eager values of -1224.9, in
the bulk. In the second, four chains were flagged at 36 to 78 sigmas below
the median and all were back within a few log-units of the bulk by the end
of the run, with `extras["lost_chains"]` correctly empty. The other cells in
the table were not inspected this way, and at spreads 0.5 and above some of
the chains counted as lost are genuinely stranded.

So these are chains still descending when warmup ended, not chains that "the
target cannot guide back", and the verdict's advice, "longer warmup will not
recover them", is wrong for them: they did recover, within the same run. It
is a separate matter from the original brief, whose lost chains were real,
and a fair reading is that a one-in-4096 straggler at a robust sigma of 1.3
is flagged correctly as unusual but described too confidently. This is an
observation for anvil to weigh, not a request.

## What strands the chains (added later the same day)

A first version of this section said there was nothing to fix. Taking the
collapsed run apart (1024 chains, `spread=1.0`, seed 1, final step 2.9e-7,
79 lost) found two separate things. Scripts were not kept; the numbers are
from the session.

**Most of the lost chains are slow, not broken.** 78 of the 79 sit at
ordinary positions (every `|u|` under 3), mostly with the jitter 7 to 71 in
natural units, 45 to 470 times the error bar: they mostly *started* there
(median starting jitter coordinate 1.62 against −0.06 for the rest). There
the likelihood is correct (float32 matches float64 to 1.3e-7 relative;
compiled and eager agree; point-to-point noise as in the bulk) but about
nine times steeper than the bulk in the run's own preconditioning. Over 8
leapfrog steps with the run's mass matrix, these 78 accept at steps of 1e-4
to 1e-2. They would come in on their own; at 3e-7 they cannot move.

**One chain causes the collapse, and it is at the float32 edge.** Chain 288
drifted during warmup to `u = (19.3, 53.7, 19.8, −1.0)`: the GP amplitude
2.5e13 times the error bar, `rho` and `Q` pinned at their upper bounds
(their sigmoids saturated). There the float32 log-likelihood is off from
float64 by **2.2 log-units**, and the chain accepts only steps of about
1e-7. The harmonic mean follows it, which freezes the other 78. A second
chain (170: amplitude 8.8e3 times the error bar, jitter at its floor, `rho`
and `Q` at their bounds, float32 off by 1.2) produced 75 of the run's 100
divergences without being counted as lost.

**It is a trap, not just slow.** The same seed with 3000 warmup ends at step
3.3e-7 with the same 79 chains lost and 116 divergences. A second seed with
3000 warmup was slowly recovering (step 1.5e-3, 26 lost, R-hat 1.51).

**A start inside the prior is only slow.** `GPTarget.u0` now takes
`units="prior"`, which counts `spread` in prior standard deviations per
column. In `u`, one unit is 3.4 prior sigmas for the default jitter, so
`spread=1.0` in `u` was a start far outside the prior. At 1024 chains:

| start | warmup | final step | lost | max R-hat | min bulk ESS |
|---|---:|---|---:|---|---:|
| `spread=1, units="prior"` (3 seeds) | 300 | 3.0e-3 to 4.0e-3 | 15 to 38 | 2.0 to 4.8 | 1076 to 1402 |
| `spread=2, units="prior"` (3 seeds) | 300 | 4.6e-5 to 9.8e-5 | 27 to 46 | 78 to 106 | 1028 |
| `spread=0.3` (3 seeds) | 300 | 3.0e-2 to 4.0e-2 | 0 to 1 | 1.07 to 1.13 | 5602 to 9582 |
| `spread=1, units="prior"` (2 seeds) | 1500 | 6.5e-3, 2.2e-2 | 7, 0 | 1.50, 1.03 | 1905, 29552 |
| `spread=1, units="prior"` (2 seeds) | 3000 | 8.9e-2, 2.6e-1 | 0, 0 | 1.003, 1.002 | 165589, 207189 |
| `spread=0.3` (2 seeds) | 1500 | 3.2e-1, 3.2e-1 | 0, 0 | 1.003, 1.003 | 184347, 185999 |
| `spread=1` (2 seeds) | 3000 | 3.3e-7, 1.5e-3 | 79, 26 | 111, 1.51 | 1028, 1888 |

At `spread=1, units="prior"` and 300 warmup, all 31 lost chains of seed 1
climb by gradient ascent to the same maximum as the healthy chains: there is
no secondary mode. They sit out along the long-timescale direction (`rho`
up to the T/2 bound, the amplitude 0.03 to 97) and are still travelling.

## What this means on our side

1. **The likelihood is right wherever float32 is in range.** Nothing to fix
   for the slow chains: they are a warmup-length matter, and the table above
   shows a 3000-step warmup clears a one-prior-sigma start.
2. **The trap was ours, and is fixed at the source (2026-10-12).** It needs
   one chain to reach a region where float32 is off by order one log-unit,
   here through an amplitude 1e13 times the error bar, about 20 prior sigmas
   out on an *unbounded* `log_gp_sigma`. The float32 feasibility map
   (docs/float32.md, section 6) showed the error is set by one number,
   `kappa`, and `GPPolicy.kappa_max` (default 1e6) now gives chains past it
   the failure value, which anvil 0.7.0 leaves out of the step-size
   statistic. The trapped seed: final step 4.7e-4 at 300 warmup (was
   2.9e-7) and 1.6e-3 at 3000 (was 3.3e-7), divergences 0 to 11 (were 100
   to 116). Its remaining lost chains all have kappa below 6e3: a wide start
   still travelling in, not a precision problem.
3. **`GPTarget.u0` documents all this,** and `units="prior"` makes an
   overdispersed start mean what it says.
4. **Our own tests that use `spread=0.3`** (the ChEES kernel test and the
   geometry tests) run 300 or fewer warmup steps from a start where, on this
   problem, a 300-step warmup ends with R-hat 1.07 to 1.13. They still pass
   their assertions, but they are not exercising converged sampling, and a
   reader should know that.
5. **The strict expected-failure test now passes** and is an ordinary test:
   with the guard, 8 of 16 chains stuck at `u = 15` leave the step size at
   1.6e-2 (control 2.4e-2); with `kappa_max=None` it is back to 3.1e-10.
