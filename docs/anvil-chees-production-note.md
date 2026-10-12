# Note for anvil: the production measurement, and what it found

**DRAFT, not yet sent.** Answers the one request in
[`anvil-chees-stuck-chains-reply.md`](anvil-chees-stuck-chains-reply.md).
Measured 2026-10-11 on anvil 0.7.0, MLX 0.32.2, M2 Max. Full numbers in
[`anvil-chees-stuck-chains-measurement.md`](anvil-chees-stuck-chains-measurement.md).

**No action requested.** This closes the thread from our side.

## The answer

- **The default start is not exposed.** `gt.u0(..., spread=0.05)` at 1024 and
  4096 chains: final step 0.15 to 0.29, no lost chains, no divergences, in
  all six runs. Your 30-minute timeout was our default `metal=False` path;
  the fused kernel runs 4096 chains x 600 iterations in 17 to 30 seconds.
- **Your diagnosis of the collapse was right.** At `spread=1.0` (1024 chains,
  seed 1) the step size goes to 2.9e-7. **One** chain causes it: it drifts
  to a GP amplitude 1e13 times the error bar, where our float32 likelihood
  is off from float64 by 2.2 log-units and only steps of about 1e-7 are
  accepted. The other 78 lost chains are ordinary: with the run's mass
  matrix they accept steps of 1e-4 to 1e-2, and are lost only because the
  harmonic mean followed the one. That run is a trap: 3000 warmup steps
  leave it at 3.3e-7 with the same 79 lost.
- **Wide starts inside the prior are only slow.** A start of one prior
  sigma per column at 300 warmup ends with step 3e-3 and R-hat 2 to 5. At
  3000 warmup it ends with step 0.09 to 0.26 and R-hat 1.003. The chains
  that were behind all climb to the same maximum as the rest: no secondary
  mode.

## What we did

The fix was ours, as you said: a sharp failure boundary. We mapped where
float32 goes wrong and found it is set by one number, `kappa`, the GP's
stationary variance over the variance one step adds, which measures how
nearly deterministic the state is between samples. Past `kappa ~ 1e6` the
float32 likelihood is off by a log-unit or more, and past ~3e7 it fails.
`GPPolicy.kappa_max` (default 1e6) now gives chains there our failure value,
which 0.7.0 already leaves out of the step-size statistic. On the trapped
seed the final step goes from 2.9e-7 to 4.7e-4 (300 warmup) and from 3.3e-7
to 1.6e-3 (3000 warmup), and our former strict-xfail test (8 of 16 chains
stuck) goes from 3.1e-10 to 1.6e-2. So **we are not asking for the trimmed
statistic.**

One thing you may see: a start far outside the prior still converges slowly
under ChEES, because the chains still travelling in are finite stragglers on
a smooth slope and the harmonic mean waits for them. That is the design
working. A start of one prior sigma converges with 3000 warmup steps (R-hat
1.003); `u0(spread=1.0)`, where one unit is 3.4 prior sigmas for our jitter, still
had R-hat 1.4 to 3.7 after 3000 with the guard on. We document the second as
too wide a start; it is not a request.

## Two observations, for you to weigh

- In the trapped run, a second chain at the edge (amplitude 8.8e3 times the
  error bar, `rho` and `Q` at their bounds, float32 off by 1.2) produced 75
  of the run's 100 divergences but was not in `extras["lost_chains"]`: its
  log-probability, −1600 against a bulk of about −1230, is near enough to the
  bulk. A chain that diverges on most iterations may be worth naming even
  when its value looks normal.
- `warmup_report` returned `LOST CHAINS` for chains that were still
  descending at the end of warmup and recovered within the same run (details
  in the measurement doc). Its advice that "longer warmup will not recover
  them" was wrong for those chains.
