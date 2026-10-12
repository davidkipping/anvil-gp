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

## What we are doing

The fix is ours, as you said: make the failure boundary sharp, so a chain
that reaches the float32-inaccurate region gets our failure value, which
0.7.0 already leaves out of the step-size statistic. That turns this case
into one you already handle, so **we are not asking for the trimmed
statistic**. We will tell you if the fix does not clear the trapped seed.

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
