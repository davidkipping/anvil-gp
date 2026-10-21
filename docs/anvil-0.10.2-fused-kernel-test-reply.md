# Reply: anvil 0.10.2 and `test_chees_runs_on_the_fused_kernel`

From anvil-gp, answering
[`anvil-0.10.2-fused-kernel-test-note.md`](anvil-0.10.2-fused-kernel-test-note.md).
Measured 2026-10-21 on anvil 0.10.2, MLX 0.32.2, M2 Max, on the test's own
configuration (256 chains, `spread=0.3`, 300 + 300, seeds 3 to 5).

**Short version.** Your measurements reproduce, and the suggested change is
made: the test now takes the median per-chain width and caps the number of
lost chains. **No action is requested.** One correction to the diagnosis:
the lost chains are not trapped, and not at the float32 boundary. They are
still climbing back when the run ends.

## What reproduces

Seed 3: width over all chains 0.0169, lost chains `[44, 110, 207, 215, 232]`,
as you found. Seeds 4 and 5: 0.0161 and 0.0187, with 10 and 9 lost.

## What the lost chains are

- **Not at the float32 boundary.** Every lost chain has `kappa` at most 2e3,
  against `GPPolicy.kappa_max` = 1e6 (docs/float32.md, section 6).
- **At or near saturated bounds.** Across the three seeds, 15 of the 24 lost
  chains have `rho` within 1 of its upper bound (`T/2` = 20, `u_rho` 4.7 to
  18.6), and 8 have the jitter at its floor (`u_jitter` −9.5 to −17; some
  have both). A few have neither, for example seed 5's chain 39 (`rho` 9.7,
  `Q` 71).
- **Coming back, not stuck.** Over the 300 sampling draws (seed 3), chain 232
  goes from `u_rho` = 15.3 to 3.4, chain 207 from 16.0 to 8.9, and chain 110
  from `u_jitter` = −15.3 to −1.9. With the run's mass matrix and step, the
  Jacobian's pull on a saturated coordinate moves a chain only a little per
  trajectory, so it takes longer than the run.
- **No secondary mode.** From their final positions, gradient ascent takes
  all five to the same maximum as the healthy chains.

So it is a warmup too short for the start, which we had already measured on
another problem: 300 warmup steps from `spread=0.3` leave R-hat at 1.07 to
1.13. The width over all chains measures how far the stragglers still are,
not the gradient the test is about.

## The change

The test now asserts:
- **the median per-chain jitter width is in 0.009 to 0.013:** measured
  0.0108 to 0.0109 on all three seeds, against the float64 Laplace 0.0107;
- **at most 26 of 256 chains are lost (10%):** measured 5 to 10;
- **acceptance is above 0.5:** unchanged.

A random walk still fails it, because per-chain widths stay small, and a
wrong gradient fails the acceptance check. We are not narrowing the
`log_jitter` box: most of these chains are out along `rho`, so it would not
remove them, and the bounds set the prior.

## Your reply to the production note

Received, thank you. One thing in it is out of date: our strict
expected-failure test already passes. The `kappa_max` guard made it pass, it
is an ordinary test now, and turning the guard off brings the failure back.
