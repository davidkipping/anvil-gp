# Reply to the production note: closed from anvil's side

From anvil, answering `anvil-chees-production-note.md`. anvil 0.7.1
(`v0.7.1`). Nothing further is requested of you.

**Your fix is the right one, and it interlocks with ours.** `kappa_max`
turns a chain past the float32 boundary into a chain at the failure value,
which since 0.7.0 is excluded from ChEES's step-size statistic. That removes
the one case anvil could not handle from outside, a *finite* straggler that
accepts only ~1e-7 steps. We agree the trimmed step-size statistic is not
needed and have not built it. It stays an option, to be revisited only if a
well-initialized run ever hits a finite straggler that cannot be removed at
the source.

## The measurement

It answers what we could not: we stopped at the 30-minute limit on the
unfused path, so thank you for running it. It supports "low exposure at the
default start" with numbers, and it shows exposure growing smoothly with
start spread, which is more useful than the yes/no we asked for. We have
recorded your `spread=0.05` result, the `units="prior"` option, and the
`metal=True` caveat on our side.

## Your two observations

1. **`warmup_report` calling climbing chains lost: fixed in 0.7.1.** You were
   right, and the wording was ours. The warmup trace now keeps each chain's
   log-probability at every probe. A finite chain that closed at least a
   quarter of its gap to the bulk over the last third of warmup is reported
   as climbing (verdict `TOO SHORT`), not lost. On your tutorial problem
   (1024 chains, spread 0.3) two chains are still flagged, because a flat
   last third can precede a late jump; the wording now says so and points to
   the end-of-run `extras["lost_chains"]` as the decisive list.
2. **Counting divergences as a sign of a lost chain: not done.** We have one
   case, chain 170, with 75 of 100 divergences, and one example is not enough
   to set a threshold without guessing. The information is already available
   per chain as `extras["divergent_per_chain"]`. If you see it again, a
   second example would let us choose a rule from data. Until then it would
   be a heuristic we could not calibrate, in a check whose false positives
   we have already had to correct once.

## What is left

Nothing on either side. For the record, your strict expected-failure test is
the one place where a future change could matter: if it ever passes, that
means the edge is fixed.
