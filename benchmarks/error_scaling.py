"""float32 error scaling, decomposed (M2 deliverable 3).

The design question is which axis the float32 error actually grows along.
celerite kernels are exponentially forgetting, so recursion error should not
compound with N -- the axes to worry about are ``tau/dt`` (conditioning of
the propagator) and ``tau/T`` (long-memory kernels that never forget within
the dataset).  The log-determinant and the quadratic form fail differently,
so a single number hides which one is in trouble; they separate cleanly by
evaluating at ``y = 0``, which zeroes every quadratic contribution (``b``,
``eta``, and all three ``ell`` cross terms vanish with it) and leaves exactly
``-1/2 sum log(S_n / yerr_n^2)``.

Measured on an M2 Max, SHO with Q = 3, block 1024:

    along N and tau/dt the RELATIVE error is flat at 3e-8 to 1.2e-6, a few
    times float32 eps.  Along sigma_GP/yerr it is NOT flat: see panel C.

So the recursion is not compounding: what grows with N is ``|logL|`` itself,
and the absolute error follows it.  The consequences are the useful part:

* At N = 1e5 (``|logL| ~ 6e4``) with a modest amplitude the worst-case
  absolute error extrapolates to ~7e-2 log-units -- an order of magnitude
  under the ~1-unit Metropolis decision scale.  That extrapolation holds
  along N and tau/dt only; along amplitude it does not (panel C).
* At N = 1e6 it reaches ~0.7, i.e. the scale of the decision itself.  That
  is where anvil's ``reanchor_every`` earns its keep, or where the per-block
  roots want composing in float64 on the CPU stream (the blocked structure
  makes that cheap: ~1e3 roots, not 1e6 leaves).
* Panel C, measured (N = 25000, block 1024, on model): the relative error
  grows from 8e-8 at ``sigma/yerr = 0.1`` to 6.5e-6 at 1000, and it is the
  *quadratic* half that grows (4.6e-1 absolute against 1.6e-4 for the
  log-determinant).  That is the ``g^2`` cancellation in the general
  composition, and it is why ``sub_block`` is chosen for accuracy.
* **The axes are not separable, and these panels cannot show that.** B
  scans ``rho/dt`` at a modest amplitude and C scans amplitude at
  ``rho = 100 dt``.  Their *product* is far worse than either: at
  ``sigma/yerr = 100`` AND ``rho/dt = 1000``, N = 32768, the MLX path is
  off by **1.5 log-units** and is not monotone in ``sub_block`` (64
  measures -4.5 against -1.5 at 32), while the fused kernel is at 0.02.
  ``validate_precision`` warns on the MLX path there and passes the
  kernel, so it is detected rather than silent;
  ``test_the_joint_high_amplitude_long_correlation_corner`` pins it.  The
  full 2-D ``(sigma/yerr, rho/dt)`` map is the M3 deliverable still owed,
  and this is the evidence that it is worth doing.
* Panel C is the amplitude axis, and it is the one that bit.  An earlier
  version of this script held the *data* at ``SIGMA = 1.0`` while scaling
  the model ``sigma``, which makes the model mismatched rather than
  high-amplitude: ``g = y/sqrt(S)`` then shrinks as sigma grows.  It
  reported a flat 4.3e-7 across the range and concluded the brief's
  predicted infeasible corner had been removed.  It had not been; the panel
  could not see it.  Drawing the data on model (``oracle.sample_ssm``)
  gives ``g^2 ~ 1e4`` at ``sigma/yerr = 100``, and the cancellation of
  three such terms inside the general composition is the dominant float32
  error in the package.  It is now bounded by the sub_block default rather
  than absent; see ``_auto_sub_block``.  This is still one slice, not the
  map -- M3 owes the full ``(sigma/yerr, timescale/cadence, Q)`` survey.

Run:  .venv/bin/python benchmarks/error_scaling.py
"""

import argparse

import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap, a64
from anvilgp.oracle import sample_ssm

p = argparse.ArgumentParser()
p.add_argument("--block", type=int, default=1024)
p.add_argument("--cadence", type=float, default=0.02)
p.add_argument("--n-amp", type=int, default=25_000,
               help="N for the on-model amplitude panel C")
args = p.parse_args()

SIGMA, QUAL, YERR = 1.0, 3.0, 0.3


def split_error(t, y, yerr, kp, block):
    """(logdet error, quadratic error, total error) in float32 log-units."""
    pm = ParamMap(kernel=[KernelParam(link="const", value=float(v))
                          for v in kp], n_dim=1)
    out = []
    for data in (np.zeros_like(y), y):          # y = 0 isolates the log-det
        gp = GPLogLike(None, t, None, data, yerr, ag.SHOTerm(), pm,
                       GPPolicy(block=block, hi_block=block))
        with mx.stream(mx.gpu):
            f32 = float(np.array(gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
        f64 = float(np.array(gp.hi(a64(np.zeros((1, 1)))))[0])
        out.append((f32, f64))
    (ld32, ld64), (t32, t64) = out
    return (abs(ld32 - ld64),
            abs((t32 - ld32) - (t64 - ld64)),
            abs(t32 - t64), t64)


def make(n, cadence, rho, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * cadence
    yerr = np.full(n, YERR)
    # Data drawn from the *right* process, so the residuals are typical of
    # what the sampler sees rather than mismatched-and-therefore-large.
    from anvilgp.oracle import dense_covariance
    if n <= 3000:
        cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, rho, QUAL]),
                               t, yerr)
        y = np.linalg.cholesky(cov) @ rng.standard_normal(n)
    else:
        y = SIGMA * np.sin(2 * np.pi * t / rho) + YERR * rng.standard_normal(n)
    return t, y, yerr


HDR = (f"{'|logL|':>11s} {'log-det err':>12s} {'quad err':>10s} "
       f"{'total':>10s} {'total/|logL|':>13s}")


def row(prefix, ld, qd, tot, ref):
    print(f"{prefix} {abs(ref):>11.1f} {ld:>12.2e} {qd:>10.2e} "
          f"{tot:>10.2e} {tot/max(abs(ref), 1e-30):>13.2e}")


print("A. error vs N at fixed tau/dt  (rho = 100 dt, so tau/T shrinks with N)")
print(f"{'N':>8s} {'T/rho':>7s} " + HDR)
rho = 100.0 * args.cadence
for n in (1000, 2000, 4000, 8000, 16000, 32000):
    t, y, yerr = make(n, args.cadence, rho)
    ld, qd, tot, ref = split_error(t, y, yerr, [SIGMA, rho, QUAL], args.block)
    row(f"{n:>8d} {n*args.cadence/rho:>7.1f}", ld, qd, tot, ref)

print("\nB. error vs tau/dt at fixed N = 8000  (the conditioning axis)")
print(f"{'rho/dt':>8s} {'rho/T':>8s} " + HDR)
n = 8000
for ratio in (2.0, 10.0, 100.0, 1000.0, 8000.0, 40000.0):
    rho = ratio * args.cadence
    t, y, yerr = make(n, args.cadence, rho)
    ld, qd, tot, ref = split_error(t, y, yerr, [SIGMA, rho, QUAL], args.block)
    row(f"{ratio:>8.0f} {rho/(n*args.cadence):>8.3f}", ld, qd, tot, ref)

# Panel C draws its data ON MODEL, at the amplitude being tested.  An
# earlier version held the data at SIGMA = 1.0 while scaling the model
# sigma, which makes the model *mismatched* rather than high-amplitude:
# the normalised innovation g = y/sqrt(S) then shrinks as sigma grows, so
# g^2 never exceeded ~10 and the panel reported a flat 4.3e-7 with "no sign
# of the innovation-variance cancellation".  On model, g^2 reaches ~1e4 at
# sigma/yerr = 100 and the cancellation in the general composition is the
# dominant float32 error in the whole package.
print("\nC. error vs sigma_GP/yerr, ON MODEL, at fixed N and rho = 100 dt")
print(f"{'s/yerr':>8s} " + HDR)
rho = 100.0 * args.cadence
n_c = args.n_amp
for ratio in (0.1, 1.0, 10.0, 100.0, 1000.0):
    sigma_c = ratio * YERR
    rng_c = np.random.default_rng(4)
    t = np.arange(n_c) * args.cadence
    y = sample_ssm(ag.SHOTerm(), [sigma_c, rho, QUAL], args.cadence, n_c,
                   rng_c) + YERR * rng_c.standard_normal(n_c)
    yerr = np.full(n_c, YERR)
    ld, qd, tot, ref = split_error(t, y, yerr, [sigma_c, rho, QUAL],
                                   args.block)
    row(f"{ratio:>8.1f}", ld, qd, tot, ref)
