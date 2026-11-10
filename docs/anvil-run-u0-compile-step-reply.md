# Reply: `run()` with a NumPy `u0`, and `compile_step`

From anvil, answering
[`anvil-run-u0-compile-step-note.md`](anvil-run-u0-compile-step-note.md).
All three suggestions landed in anvil **0.12.0** and were refined in
**0.12.1** (`v0.12.1`, 2026-11-10). Nothing further is requested of you.
One point matters for your JWST user, under "The remedy for a gradient
sampler".

## 1. NumPy `u0`

`run` now accepts anything array-like, converted to float32, as
`run_mcmc` already did. Since 0.12.1:

- errors about the start name `u0`, for a non-2-D shape or input NumPy
  cannot convert;
- the column count is checked against `target.dim`, which is the
  likeliest mistake when a joint start is assembled by hand;
- a float64 `mx.array` is converted too, where it used to fail on the
  GPU.

A script that passed a start with the wrong number of columns used to run
or fail later with a broadcast error. It now gets an error up front.

## 2. `compile_step` on the kernel

`run`'s default is now `compile_step=None`, which keeps whatever
`kernel.compile_step` says. An explicit True or False to `run` still
wins. The value is written back to the kernel object, so after
`run(k, ..., compile_step=False)` a later `run(k, ...)` is uncompiled
too, until True is passed. This is documented. `compile_step` must be a
bool or None.

`EnsembleSampler.run_mcmc` gained a `compile_step` argument in 0.12.1, as
did `bridge_evidence`. Every entry point that can raise the hint below can
now follow it.

## 3. The error for a target that leaves MLX

The first compiled step now catches MLX's
`[eval] Attempting to eval an array during function transformations` and
re-raises it as a `ValueError`. The message names the step, the usual
causes (NumPy, `.item()`, `float()`, host control flow on array values)
and the remedy. It matches MLX's full message, so a user's own
`ValueError` that happens to start with `[eval]` is not relabelled. A
review checked that `.item()`, `float()`, `int()`, `np.array()`, NumPy
ufuncs, `bool()` in an `if`, `mx.eval`, `.tolist()` and `memoryview` each
get the message, under every kernel and under `optimize` and `smc`.

## The remedy for a gradient sampler

0.12.0's message told everyone to pass `compile_step=False`. That is right
for a gradient-free method and wrong for a gradient one.

Uncompiled, a `log_prob` that leaves MLX runs. But no gradient flows
through the part computed outside MLX, so ChEES and `optimize` see only
the Jacobian's and the priors' gradient, with no warning. On a
three-parameter Gaussian written in NumPy:

| Same density, sampled with ChEES | Adapted step size |
|---|---|
| written in `mlx.core` ops | 1.07 |
| written in NumPy, `compile_step=False` | 0.004 |

The draws were still from the right distribution, at random-walk
efficiency. 0.12.1's message now says this. `compile_step=False` is the
remedy for the ensemble kernel, random-walk Metropolis, `smc`,
`bridge_evidence` and `run_mcmc`. ChEES and `optimize` need the target in
`mlx.core` ops.

**For your JWST user,** whose evaluated part of the model runs in NumPy on
the CPU: with the ensemble kernel, `compile_step=False` is a complete
answer. With ChEES it would run but would not be HMC in any useful sense.
They should either port that part to MLX or stay on the ensemble kernel.
anvil-gp's tutorial already passes `u0` as an `mx.array` and uses the
ensemble kernel for its warm start, so nothing there needs to change.
