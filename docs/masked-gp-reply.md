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
| value | 3.00 ms (1.8x) | 8.22 ms (2.3x) |
| value + gradient | 6.85 ms (2.3x) | 19.7 ms (3.1x) |

The unmasked kernel takes 1.68, 3.00, 3.53 and 6.45 ms for the same
four cells. The MLX fallback took 51 ms (value) and 342 ms (value +
gradient) at 128 chains. For your ensemble runs, which need values only,
that is inside the 1.5-2x you asked for. Gradients are at 2.3x. These
are likelihood-only figures. Your step also evaluates the transit model,
which costs the same masked or not, so the whole-step ratio you see
should be lower.

How it works, briefly: the kept points are scanned forward and
time-reversed by the existing kernels (in every term's state, time
reversal is a sign flip, so the same lag tables serve both directions).
The two are joined at each held-out point by one further kernel. Your
§4 decomposition is answered by this as well. On the kernel, masking
costs about two passes over the data. The number of windows makes
little difference: two or four windows over the same points cost the
same, and one long window a little more.

## 2. Accuracy: please re-run your reference check

The kernel path is more accurate in float32 than the MLX path you
validated. Against float64:

- **Value, at your scale:** 8e-5 log-units with a GP comparable to the
  noise. With slow systematics at 30x the noise it is 2.4e-3, against
  5e-2 on the MLX path.
- **Gradient, at your scale:** as accurate as the unmasked kernel's,
  8e-7 to 9e-6 relative.

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
right even if the two targets' transforms differ, and pass the start to
`anvil.run` as an `mx.array` (a NumPy `u0` is the opaque dtype error your
first note reported against anvil). Keep a warm-up, because the posterior
moves once the feature stops being absorbed.

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

The commit to pick up is `b62ee94`; docs/masked-gp-plan.md records the
design and every measurement.
