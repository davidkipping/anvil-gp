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

## What this means on our side

1. **No engineering change now.** The default is not exposed, and the exposure
   elsewhere is graded, visible (tiny steps, near-1.00 acceptance, lost
   chains, a `warmup_report` that names them) and cannot bias a posterior.
   anvil's suggested root fix, a sharper failure boundary at the feasibility
   edge, remains the option if wide initialisation ever matters.
2. **Say so where users will see it.** `GPTarget.u0`'s docstring now states
   that the default spread is the tested one, and gives this table's
   headline: a spread of 0.3 or more costs step size, 0.7 or more can make
   the run unproductive.
3. **Our own tests that use `spread=0.3`** (the ChEES kernel test and the
   geometry tests) operate, on this problem, with a step size about ten
   times below the default; their own problems were not measured. They still pass their assertions, but they are not exercising
   the default regime, and a reader should know that.
4. **The strict expected-failure test stays.** It fails today because the
   stuck chains in it start at `u = 15`, deep in the exposure range
   (equivalent to the wide end of this table), and will pass if the edge is
   ever fixed.
