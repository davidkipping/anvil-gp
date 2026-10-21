# anvil 0.10.2 and `test_chees_runs_on_the_fused_kernel`

*Untracked note from the anvil side, 2026-10-20. Not a change to anvil-gp.*

On anvil 0.10.2 `tests/test_calibration.py::test_chees_runs_on_the_fused_kernel`
fails its width check: jitter posterior sd 0.0169 against the bound 0.016
(seed 3; seeds 4 and 5 give 0.0161 and 0.0187). Everything else in the
suite passes (803/804). What is going on, measured on the test's own
configuration with an uncompiled run and a spying target:

- During warmup, chains are flung to |u| up to 1000 in every column
  (392 chain-evaluations past |u| = 88.7 in the 300 + 300 run).
- **anvil ≤ 0.10.1** returned a **non-finite gradient** at every one of
  those 392 evaluations (`Transform.to_model` evaluated `exp(u)` for every
  column and the discarded branch's inf back-propagated as NaN). Those
  proposals were rejected, which acted as an accidental barrier at
  |u| ≈ 88.7.
- **anvil 0.10.2** fixes that (the fix was needed: one chain started on a
  bound froze every chain's step size at 0). The gradient there is now
  finite, HMC explores those regions legitimately, and the width statistic
  changes.
- In **both** versions the run ends with **5 lost chains of 256**
  (`res.extras["lost_chains"]`: `[44, 110, 207, 215, 232]` new,
  `[44, 110, 207, 213, 215]` old), stuck far out where the sigmoid is
  saturated. The test's `jit_sd` is the std over *all* chains, lost ones
  included; 3–5 chains sitting far from the bulk are what put it at
  0.013–0.019 rather than the Laplace 0.0107. The bound 0.016 is marginal
  with either anvil (seed 5 fails on 0.10.1 too).

Suggested change on the anvil-gp side, if you agree: drop
`res.extras["lost_chains"]` before taking the width (the healthy chains
give ≈ 0.011), and/or assert on the median per-chain width. The lost
chains themselves are the float32 boundary trap anvil's docs describe
(a bounded parameter pinned past |u| ≈ 17 cannot come back in one
trajectory); if they matter for production, a narrower box on
`log_jitter` or a tighter `u0` spread removes them.
