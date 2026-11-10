# Reply: held-out GP conditioning, now on the fused kernel

**DRAFT, not sent.** To the TOI-199 b JWST/NIRSpec team, answering
`anvilgp_masked_gp_request.md` and the field report
`anvilgp_masked_speed_gap.md`.

---

Thank you for the field report. Your timings were the measurement we had
left open, and they were right: on the MLX fallback a masked fit was far
too slow to be the default. That is fixed. The answers below follow your
§6 in order.

## 1. The kernel path (Phase 2): done

`held_out` now runs on the fused Metal kernel, value **and** gradient,
with a marginalised `linear_basis` (shared or per chain) and with
`noise_fn` noise models such as `error_scale`. Your call does not
change: the same boolean `(N,)` mask into `GPLogLike(..., held_out=mask)`
or `make_target(..., held_out=mask)`. Nothing in the parameter vector
changes either.

Measured on a quiet machine, at your shape: N = 2818 one-minute points,
398 (14%) held out in two windows, Matern-3/2, `noise.error_scale()` and
a per-chain quadratic baseline marginalised. Time per likelihood call
against the same likelihood unmasked on the kernel:

| | 128 chains | 1024 chains |
|---|---:|---:|
| value | 2.62 ms (1.7x) | 7.21 ms (2.0x) |
| value + gradient | 6.08 ms (2.0x) | 13.9 ms (2.0x) |

The unmasked kernel takes 1.57, 3.01, 3.57 and 6.88 ms for the same
four cells. The MLX fallback took 51 ms (value) and 342 ms (value +
gradient) at 128 chains. So masking now costs at most about twice the
unmasked likelihood, value or gradient, within the 1.5-2x you asked for.
(At 128 chains the unmasked times are near 1 ms, mostly launch latency,
and these ratios move by about 0.2 between runs.) These
are likelihood-only figures. Your step also evaluates the transit model,
which costs the same masked or not, so the whole-step ratio you see
should be lower.

How it works, briefly: the kept points are scanned forward and
time-reversed by the existing kernels (in every term's state, time
reversal is a sign flip, so the same lag tables serve both directions).
Each window is then walked from its two edges with the cadence's
propagator, and the two sides are joined at each held-out point. Your
§4 decomposition is answered by this as well. On the kernel, masking
costs about two passes over the data, and the joining adds little on
top. The number of windows makes little difference: four windows over
the same points cost about what two do, and one long window a little
more at small chain counts.

## 2. Accuracy: please re-run your reference check

The kernel path is more accurate in float32 than the MLX path you
validated. Against float64:

- **Value, at your scale:** 8e-5 log-units with a GP comparable to the
  noise. With slow systematics at 30x the noise it is 7.6e-4, against
  5e-2 on the MLX path.
- **Gradient, at your scale:** as accurate as the unmasked kernel's,
  1e-6 to 7e-6 relative.

The float64 `hi` path is unchanged and still matches the dense covariance
to ~1e-11. Your §2 comparison (`hi` within ~1e-6 of your celerite2
reference, float32 within ~1e-3 across draws) is the acceptance test we
would like you to run on the new path. We expect the float32 column to
tighten.

## 3. The optimiser excess

It was cost per evaluation, not iterations. On the same shape we counted
a median of 9-10 L-BFGS iterations per start masked against 10 unmasked,
all converged. The 32x you saw was the MLX path's value + gradient. On
the kernel it is now 2.3x.

## 4. Warm starts

Documented in the tutorial (docs/tutorial.md, "Holding a feature out of
the GP"). The masked target has the same parameters, so you can skip
`anvil.optimize` and start from your unmasked chain's last positions.
Convert them through model space, as the example does, so the start is
right even if the two targets' transforms differ. Keep a warm-up, because
the posterior moves once the feature stops being absorbed.

We passed on the two anvil issues in your first note, and both are fixed
in anvil 0.12.1. `run` now accepts a NumPy `u0`, and it no longer
overwrites a `compile_step` you set on the kernel. On an earlier anvil,
pass the start as an `mx.array`, as the tutorial does. anvil added one
caution: a log density that leaves MLX (NumPy on the CPU) runs uncompiled
with `compile_step=False`, but no gradient flows through the NumPy part.
That is a complete answer for the ensemble kernel but not for ChEES or
`anvil.optimize`, which would run with only the prior's gradient. With
`held_out` native, the CPU workaround is no longer needed. If the rest of
your model is in `mlx.core` ops, the gradient sampler and the optimiser
are open to the masked fit again.

## Two smaller things

- With `make_target(seed_period=True)`, the Lomb-Scargle that seeds a
  rotation period now runs on the kept points only. A held-out feature
  can no longer set the period reference.
- A convolved term (`TermConvolution`) still runs masked on the MLX path,
  with a warning that says why. It has no time reversal in the form the
  kernel needs. You use Matern-3/2, so this does not affect you.

The caveat in the original plan stands: correlated noise *inside* a
window is not modelled. Deep inside a window much longer than the
kernel's timescale, points are scored as white noise about the mean
model. To sample extra white noise there, label the held-out points as
their own instrument with a jitter.

The commit to pick up is `c5943fe`; docs/masked-gp-plan.md records the
design and every measurement.
