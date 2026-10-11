# Lost ensemble walkers are invisible to anvil's diagnostics

Found 2026-10-11 while adapting anvil-gp to anvil 0.5.0 (`75d89f5`, MLX
0.32.2). **These are recommendations for `anvil`, not requests on
anvil-gp's behalf**: anvil-gp has already fixed its own side (`3975960`), and
nothing here is needed for anvil-gp to work. They are aimed at every anvil
user whose target can fail.

> **Status (2026-10-11): items 1 and 2 implemented in anvil `3758996`
> (0.6.0); item 3 declined.** Verified from anvil-gp against the
> reproducer, with RuntimeWarnings escalated to errors:
>
> - Constant failure value: still 13/64 walkers lost, as expected -- no
>   sampler change can give a flat region a direction -- but **no coordinate
>   is non-finite** (they pin near float32's maximum, ~3.4e38, instead of
>   `inf`). `warmup_report` returns **`LOST CHAINS`**, names them, and says
>   longer warmup will not help; `diagnose(chain, log_prob=...)` names the
>   same 13 beside R-hat. No `inf/inf`.
> - Sloped control: 0 lost, healthy verdict, `diagnose(...).lost` empty.
> - anvil-gp's own target, the case that got a false `OK`: now
>   **`LOST CHAINS`, exactly 11**. With anvil-gp's current sloped failure
>   value: `OK`, 0 lost.
>
> The finite-position check went wider than asked: ensemble, RWM *and*
> ChEES. Item 3 was declined because re-seeding would also hide a target
> that fails over much of its prior, which a diagnostic should not do --
> a reasonable call, and the brief had already flagged it as optional for
> that reason. The ChEES observation under "Not in this brief" remains
> undiagnosed.

Three items, labelled by how strongly they are argued:

| | item | what it asks |
|---|---|---|
| **Needed** | [1](#1-needed-lost-walkers-pass-anvils-diagnostics) | Reject non-finite positions in the ensemble; give `warmup_report` and `diagnose` a lost-chain check |
| **Recommended** | [2](#2-recommended-tell-target-authors-what-to-return-on-failure) | Document what a target should return when its model fails |
| **Optional** | [3](#3-optional-re-seed-lost-walkers-during-warmup) | Re-seed lost walkers during warmup |

Everything below is reproducible with anvil alone:
[`repro/anvil_lost_walkers.py`](repro/anvil_lost_walkers.py) needs only anvil
and MLX.

## Background: how this surfaced

anvil `f97f583` made the bounded-parameter log-Jacobian finite. It had been
`log(sig) + log1p(-sig)`, which is exactly `-inf` for `|u| >= 18` because
`sigmoid(18)` is exactly 1.0 in float32. The change is **correct**, and it was
prompted by anvil-gp's own `docs/anvil-chees-boundary.md`.

It also broke one anvil-gp test, with no anvil-gp change at all. anvil-gp
reported a failed likelihood as the constant `-1e30`. A constant is a flat
region, not a rejection: a sampler accepts moves *within* it and drifts. The
old `-inf` Jacobian had hidden that, because every move between failed states
was `(-inf) - (-inf) = NaN` and so rejected, an accidental wall. With the
Jacobian finite, float32 absorbed it into the constant
(`-1e30 + (-120) == -1e30` exactly), the region went perfectly flat, and 11 of
32 stretch walkers started there diffused to `|u| ~ 1e11` and never came back.
Restoring the old Jacobian in-process, same anvil and seeds, gave 0/32 stuck,
so the cause is that change alone.

anvil-gp's fix was on its own side. What the episode exposed on anvil's side
is that **anvil never noticed**: its diagnostics did not report the lost
walkers, and one of them gave the all-clear.

## 1. Needed: lost walkers pass anvil's diagnostics

### What was seen

The reproducer is a 4-D standard normal whose model "fails" beyond a radius
of 25 and returns a constant `-1e30` there. Half of 64 walkers start at
`|u| = 30`, just inside the failure region. Run with `EnsembleKernel`, 400
warmup, 2,000 samples, seed 3. Then the same run with a failure value that
slopes back towards the origin, as a healthy control:

| | constant failure value | sloped failure value |
|---|---|---|
| lost walkers at the end | **13 / 64, at `|u|` = inf** | 0 / 64 |
| `warmup_report` | **`TOO SHORT`** -- "try at least 800 iterations" | `LONGER THAN NEEDED` |
| `diagnose`, max R-hat | 1.50 | 1.02 |
| lost-chain check proposed below | flags **exactly those 13** | flags 0 |

On anvil-gp's own target, the same failure gave a *different* wrong verdict:
**`warmup_report` returned `OK`**, "the spread settled by iteration 370,
comfortably inside the warmup budget", with 11 of 64 walkers at `|u| ~ 1e11`.
`diagnose` showed R-hat 2.16.

So which wrong answer you get depends on whether the lost walkers' spread
levels off (`OK`) or keeps growing (`TOO SHORT`). Neither verdict mentions
lost walkers. `TOO SHORT` is actively harmful advice here: longer warmup only
lets lost walkers drift further. R-hat does flag a problem, but says nothing
about what it is.

### Why

Three separate gaps, each with its location in anvil `75d89f5`:

- **The ensemble accepts non-finite positions.** `kernels/ensemble.py:156-158`
  decides acceptance from the log-probability alone,
  `accept = log(U) < log_ratio`. A stretch proposal from a walker far out
  overflows to `inf` in float32, but its log-probability is still the finite
  failure value, so it is accepted. In the reproducer, 13 walkers end at
  `|u| = inf`.
- **The warmup probe only checks relative stability.** It records
  `mx.std(u, axis=0)` (`engine.py:573`), and the settle test asks whether that
  spread stopped changing, within `max(settle_tol, 3 x noise)`, about 27% at
  64 chains. It never compares the spread with anything absolute, and never
  looks at log-probabilities. In anvil-gp's run, the spread levelled off at
  about 5.4e10, changing by +7%, -3%, +6%, +3% between probes, so it counted
  as settled. The healthy run settles at 0.2 and is also `OK`: two spreads
  2.7e11 apart, one verdict.
- **The settle test divides `inf` by `inf`.** Once positions are infinite, so
  is the spread, and `diagnostics.py:666` computes `sd / ref` as NaN
  (`RuntimeWarning: invalid value encountered in divide`). The report is then
  reasoning about NaNs.

### Recommendations

1. **Reject proposals with non-finite positions.** One condition at
   `ensemble.py:156`:

   ```python
   accept = (mx.log(mx.random.uniform(shape=(K,), key=k_acc)) < log_ratio) \
            & mx.all(mx.isfinite(prop), axis=-1)
   ```

   A position that has overflowed is never a valid state, whatever the target
   returns there. Worth applying to any other kernel that proposes positions
   without a gradient check, such as `RandomWalkMetropolis`.

2. **Give `warmup_report` a lost-chain check, run before any verdict.** Flag
   chains whose final log-probability sits far below the ensemble's, measured
   robustly so the lost chains cannot inflate the scale:

   ```python
   med = np.median(lp)
   mad = 1.4826 * np.median(np.abs(lp - med))
   lost = np.where(lp < med - 10.0 * max(mad, 1e-12))[0]
   ```

   In the reproducer this flags exactly the 13 lost walkers and nothing in
   the healthy run. When it fires, the verdict should say so, name the
   chains, and **not** suggest longer warmup. It is the ensemble counterpart
   of ChEES's existing `extras["divergent_per_chain"]`.

3. **Report the same count from `diagnose`,** next to R-hat, so a bad R-hat
   arrives with its likely cause.

   The threshold does not need fine tuning. A lost chain sits at the
   target's failure value, here `1e30`, against ensemble log-probabilities of
   order 1 to 1e5, so it is many orders of magnitude out rather than
   marginally. One real limitation: the check measures distance from the
   *median*, so it goes blind once half or more of the chains are lost,
   because the median is then a lost chain. A cheap backstop is to also
   report any chain whose log-probability is non-finite or below a fixed
   floor such as `-1e29`.

4. **Guard the settle test against non-finite spreads,** so a NaN cannot
   reach a verdict. With recommendation 1 in place this should not happen,
   but the diagnostic should not depend on that.

A good acceptance test is the reproducer itself: in constant mode,
`warmup_report` should neither say `OK` nor recommend more warmup.

## 2. Recommended: tell target authors what to return on failure

anvil gives no guidance on this. Its README, `docs/` and source have nothing
on what a target should return when its model cannot be evaluated. Yet both
known downstream users return large finite values: anvil-gp returned `-1e30`,
and Elome clamps every numerical hazard so that an unphysical geometry
returns an ordinary finite log-likelihood. This episode showed two hazards
worth documenting:

- **A constant failure value is a flat region.** Samplers accept moves within
  it, so walkers that land there drift instead of being rejected. The stretch
  move's acceptance on a flat density is `min(1, z^(d-1))`, so outward moves
  are always accepted.
- **A failure value much larger than the Jacobian swallows it.** This one is
  specific to anvil: `TransformedLogDensity` adds the target's value and the
  log-Jacobian together in float32, and float32 resolution near `1e30` is
  about `7e22`. So the Jacobian -- which correctly prefers smaller `|u|` --
  and any explicit prior are rounded away. The failure region then has no
  direction at all.

A suggested pattern, which is what anvil-gp now does: if the target returns a
finite failure value, make it **slope back towards feasibility**, make the
slope **relative** so float32 can resolve it at that magnitude, and
**`stop_gradient`** it, since its true derivative is of the same enormous
order and would wreck an HMC trajectory:

```python
failed = -1e30 * (1.0 + mx.stop_gradient(mx.log1p(mx.mean(v * v, axis=-1))))
```

Because the slope is only ever selected on failed evaluations, wherever the
model succeeds values and gradients are bit-identical to before. anvil-gp
verified this on both its evaluation paths and over a whole ChEES run of
25,600 draws.

The documentation should probably also mention the alternative of returning
`-inf`, which restores "moves between failed states are rejected" exactly,
and its costs under HMC. And `f97f583`'s changelog entry could note the
downstream consequence: its message says a run with no chains at the boundary
cannot change, which is true, but a target whose model fails past
`|u| ~ 18` loses the wall that the `-inf` had been providing.

## 3. Optional: re-seed lost walkers during warmup

During warmup only, replace any walker that the lost-chain check flags with
a copy of a randomly chosen live walker, plus a small jitter. Correctness is
unaffected, since warmup draws are discarded and adaptation is still free to
use the replaced positions.

The case for it: with recommendations 1 and 2 in place, a lost walker is
*reported*, but it is still lost, and the run still finishes with fewer
effective walkers than it started with. Re-seeding recovers them for any
target, including ones whose authors never read the documentation from item
2.

The case against, which is why this is optional: it can hide a broken
target. A model that fails over a large part of its prior would quietly lose
and regain walkers every warmup, and look fine. If adopted, it should be
logged in `extras` (how many walkers, which ones, at which iterations) and
reported by `warmup_report`, and it is a reasonable candidate to be off by
default. anvil-gp no longer needs it.

## Not in this brief

While testing anvil-gp's fix, 3 of 32 ChEES chains started in the failure
region stayed stuck at `|u| ~ 30.7`. This happened with both the constant and
the sloped failure value, with identical divergence counts, so anvil-gp's
change neither caused nor fixed it. The other 29 recovered, which is anvil's
own boundary work in `f97f583`. The cause of the remaining three has not been
diagnosed. One possibility is that ChEES's rule "a divergence vetoes a
proposal only from a finite state" treats a finite failure value as an
ordinary state, but that is a hypothesis, so it is left out until it is
understood.

## Reproducing

```bash
python docs/repro/anvil_lost_walkers.py            # constant failure value
python docs/repro/anvil_lost_walkers.py sloped     # the healthy control
```

Each run takes a few seconds on an M2 Max. The script prints how many walkers
were lost and where, `warmup_report`'s verdict, `diagnose`'s table, and what
the proposed lost-chain check flags.
