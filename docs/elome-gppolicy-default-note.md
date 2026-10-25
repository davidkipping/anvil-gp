# Note to Elómë: one stale sentence in tutorial 2 (anvil-gp's `metal` default)

**Draft, 2026-10-25 -- not sent.** Nothing in Elómë is broken; this is
one sentence of prose that anvil-gp's change of default made stale.

## What changed in anvil-gp

Since 2026-10-25 `GPPolicy.metal` defaults to `None`: the fused Metal
kernel wherever it is eligible, the MLX path otherwise, with a one-time
notice per process saying which it chose. It was `False` before. The
kernel now also takes state dimension m <= 6 (it was m <= 4), so
rotation + granulation (`RotationTerm() + SHOTerm()`) runs on it.
`GPPolicy(metal=True)` still works and means "insist on the kernel, and
warn if it cannot serve this model"; it also suppresses the one-time
notice, since the path was asked for.

## What is stale

`docs/tutorials/build_02.py:199` (and the generated
`02_joint_transit_and_gp.ipynb`, the markdown cell before the
`make_target` call):

> `GPPolicy(metal=True)` turns on anvil-gp's fused Metal kernel.

The kernel is on by default now, so "turns on" misdescribes the line.

## Suggested replacement

> `GPPolicy(metal=True)` insists on anvil-gp's fused Metal kernel and
> warns if it cannot serve the model. The kernel is anvil-gp's default
> wherever it is eligible, so the line only makes the choice explicit.

The code cell (`policy=GPPolicy(metal=True)`) can stay as it is: it
gives the same path as the default here (`SHOTerm`, m = 2) and keeps the
notebook's output free of the one-time notice. The "Keep the kernel
honest" paragraph near the end is still accurate.
