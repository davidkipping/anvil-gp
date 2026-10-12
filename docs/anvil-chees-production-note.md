# Note for anvil: the production measurement, and the fix

From anvil-gp, answering the one request in
[`anvil-chees-stuck-chains-reply.md`](anvil-chees-stuck-chains-reply.md).
Measured 2026-10-11 and 2026-10-12 on anvil 0.7.0, MLX 0.32.2, M2 Max; the
fix is anvil-gp `eb75c02`. Full numbers in
[`anvil-chees-stuck-chains-measurement.md`](anvil-chees-stuck-chains-measurement.md)
and [`float32.md`](float32.md), section 6.

**Short version.** Your diagnosis was right, the fix was ours, and it is
done. **No action is requested**, including the trimmed step-size statistic.
This closes the thread from our side.

## The measurement

- **The default start is not exposed.** `gt.u0(..., spread=0.05)` at 1024 and
  4096 chains: final step 0.15 to 0.29, no lost chains, no divergences, in
  all six runs. Your 30-minute timeout came from our default `metal=False`
  path; on the fused kernel 4096 chains x 600 iterations take 17 to 30
  seconds.
- **A wide start can collapse the step size, and one chain does it.** At
  `spread=1.0` (1024 chains, seed 1) the final step is 2.9e-7. One chain
  drifts to a GP amplitude 1e13 times the error bar, where our float32
  likelihood is off from float64 by 2.2 log-units and only steps of about
  1e-7 are accepted. The other 78 lost chains are ordinary: with the run's
  mass matrix they accept steps of 1e-4 to 1e-2, and are stuck only because
  the harmonic mean followed the one. It is a trap, not a slow run: 3000
  warmup steps leave it at 3.3e-7 with the same 79 lost.
- **A wide start inside the prior is only slow.** One prior sigma per
  parameter: after 300 warmup steps, final step 3e-3 and R-hat 2 to 5; after
  3000, step 0.09 to 0.26 and R-hat 1.002 to 1.003. In the seed we examined,
  every chain that was behind climbs to the same maximum as the rest, so
  there is no secondary mode.

## The fix

We mapped where our float32 likelihood goes wrong. The error is set by one
number, `kappa`: the GP's stationary variance over the variance a single
step adds, i.e. how nearly deterministic the state is between samples.
Amplitude alone does not set it (1e14 times the error bar is fine at a short
timescale). Where a posterior actually lives, errors reach 0.8 log-units
around `kappa = 1e6` and 0.6 to 14 near 3e6, and evaluations fail outright
past about 3e7.

`GPPolicy.kappa_max` (default 1e6) now gives any chain past it our failure
value, which 0.7.0 already leaves out of the step-size statistic. Values at
feasible points are unchanged bit for bit. Results:

| | before | after |
|---|---|---|
| trapped seed, final step, 300 warmup | 2.9e-7 | 4.7e-4 |
| trapped seed, final step, 3000 warmup | 3.3e-7 | 1.6e-3 |
| our test with 8 of 16 chains stuck at `u = 15` | 3.1e-10 | 1.6e-2 (control 2.4e-2) |

Turning the guard off brings the 3.1e-10 straight back.

**What you may still see.** A start far outside the prior still converges
slowly under ChEES: the chains still travelling in are finite stragglers on
a smooth slope, and the harmonic mean waits for them, as designed.
`u0(spread=1.0)`, where one unit is 3.4 prior sigmas for our jitter, still
has R-hat 1.4 to 3.7 after 3000 warmup steps with the guard on; none of its
remaining stragglers is near the float32 boundary. We document that as too
wide a start. It is not a request.

## Two observations, for you to weigh

- In the trapped run, chain 170 (also past the float32 boundary) produced 75
  of the run's 100 divergences. `warmup_report` named it as lost, but
  `extras["lost_chains"]` at the end of the run did not: its
  log-probability, −1757 against a bulk median of −1656, was not far enough
  out. Divergences on most iterations might be worth counting as a sign of a
  lost chain.
- `warmup_report` returned `LOST CHAINS` for chains that were still
  descending at the end of warmup and recovered within the same run
  (examples in the measurement doc). Its advice that "longer warmup will not
  recover them" was wrong for those chains.
