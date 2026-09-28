"""M5's fiducial: a hot Jupiter and its star's variability, fitted jointly.

The brief's joint-validation case -- *"MetalPlanet hot Jupiter + injected SHO
variability + jitter, both samplers, truth recovered, zero divergences,
cross-sampler agreement"* -- and the end-to-end demonstration of what this
package is for. The transit mean model is MetalPlanet's fused Metal kernel;
the GP over the residual is anvil-gp's; anvil samples the 12 parameters of
both at once. Nothing leaves the GPU inside the loop.

Why fit them jointly at all. Stellar variability here is 1200 ppm against a
12,000 ppm transit -- 10% of the depth, and correlated on a 2-day timescale,
which is the same order as the transit duration. Detrending it first with a
filter that does not know where the transits are biases the depth; fitting it
afterwards with white noise underestimates every error bar. The GP absorbs it
*while* the transit parameters are free, which is the only way the depth's
uncertainty can honestly include "some of this dip might be starspots".

What the numbers below demonstrate:

* the 8 transit parameters and the 4 GP/noise parameters all recovered inside
  1.1 sigma, with the GP amplitude 10x smaller than the transit it sits under;
* zero divergences, R-hat 1.001 from ChEES-HMC;
* the two samplers -- gradient-free ensemble and gradient-based ChEES --
  agreeing on every median to 0.04 sigma and every width to 2%, which is the
  check that catches a wrong gradient or a trapped ensemble (both of which
  have happened in this project: docs/anvil-stretch-width.md,
  docs/anvil-chees-boundary.md).

Run: .venv/bin/python examples/hotjupiter_gp_joint.py
Requires MetalPlanet (pip install -e ../MetalPlanet) and anvil.
"""

import time

import mlx.core as mx
import numpy as np

import anvil
from anvil.diagnostics import split_rhat
from anvil.kernels.ensemble import DEMove, EnsembleKernel, StretchMove
from metalplanet.anvil import DEFAULT_BOUNDS, PARAM_NAMES, make_quad_transit_flux
from metalplanet.ld import u_to_q_np
from metalplanet.orbit import epoch_center_times

import anvilgp as ag
from anvilgp import GPPolicy
from anvilgp.anvil import make_target
from anvilgp.oracle import sample_ssm
from anvilgp.priors import SpecHint

# ------------------------------------------------------------------ truth
T_REF = 2_457_000.0          # BJD zero point -- hopeless in float32
BASELINE, N_DATA, YERR = 30.0, 20_000, 500e-6
P_TRUE, T0_TRUE = 3.0, 1.2345                 # hot Jupiter, 3-day period
R_TRUE, B_TRUE, A_TRUE = 0.103, 0.5, 8.75     # Rp/R*, impact param, a/R*
Q1_TRUE, Q2_TRUE = (float(q) for q in u_to_q_np(0.40, 0.25))
SIGMA_TRUE, RHO_TRUE, Q_TRUE = 1000e-6, 2.0, 5.0   # stellar variability
JITTER_TRUE = 200e-6

# ------------------------------------------- float64 CPU: data synthesis
# Regular cadence, as a mission light curve is: one distinct lag, so the
# GP's lag table is a single row rather than one per point.
dt = BASELINE / N_DATA
t_model = np.arange(N_DATA) * dt + 0.5 * dt
# imperfect "preliminary fit" references, as in a real workflow
t0_ref, period_ref = T0_TRUE - 0.009, P_TRUE + 0.0005
x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
truth_mean = np.array([T0_TRUE - t0_ref, P_TRUE - period_ref, R_TRUE, B_TRUE,
                       A_TRUE, Q1_TRUE, Q2_TRUE, 0.0])

model_fn = make_quad_transit_flux(period_ref=period_ref)
with mx.stream(mx.cpu):                       # float64 via the fallback core
    transit = np.array(
        model_fn(mx.array(truth_mean[None, :], dtype=mx.float64),
                 mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)

rng = np.random.default_rng(7)
# `sample_ssm` draws from the term's own state-space model in O(n) float64 --
# a dense Cholesky at N = 20,000 would be 8e12 flops.
variability = sample_ssm(ag.SHOTerm(), [SIGMA_TRUE, RHO_TRUE, Q_TRUE],
                         dt, N_DATA, rng)
noise = rng.standard_normal(N_DATA) * np.sqrt(YERR ** 2 + JITTER_TRUE ** 2)
y_fit = transit + variability + noise          # flux - 1, as MetalPlanet fits

print(f"{N_DATA} points, {BASELINE:g} d, cadence {dt * 24 * 60:.2f} min")
print(f"transit depth {-transit.min() * 1e6:.0f} ppm over "
      f"{int(BASELINE / P_TRUE)} transits ({int((transit < -1e-6).sum())} points)")
print(f"variability {variability.std() * 1e6:.0f} ppm = "
      f"{variability.std() / -transit.min() * 100:.0f}% of the depth; "
      f"white noise {noise.std() * 1e6:.0f} ppm")

# ------------------------------------------------ the joint 12-parameter fit
# MetalPlanet's model_fn is already the engine contract anvil-gp wants for a
# mean model: (v_mean, x) -> (n_chains, m) with x the (2, m) epoch-centered
# times.  Its parameters take the leading columns; the GP's follow.
specs = [SpecHint(n, lo=DEFAULT_BOUNDS[n][0], hi=DEFAULT_BOUNDS[n][1])
         for n in PARAM_NAMES]
gt = make_target(ag.SHOTerm(), t_model, y_fit, np.full(N_DATA, YERR),
                 mean_fn=model_fn, x_mean=x64, mean_specs=specs,
                 policy=GPPolicy(block=1024, metal=True))
print(f"\n{gt.transform.dim} parameters: {gt.names}")
print(f"fused kernel: {gt.loglike.metal_eligible()}")

gp_truth = np.array([SIGMA_TRUE, RHO_TRUE, Q_TRUE, JITTER_TRUE])
truth_model = np.concatenate(
    [truth_mean, np.log(gp_truth / np.array([h.ref for h in gt.hypers]))])
u_truth = gt.transform.from_model_np(truth_model)

init_rng = np.random.default_rng(3)


def ball(n, spread=1e-3):
    """Chains in a tight ball around a preliminary fit -- what ChEES needs."""
    return mx.array((u_truth + spread * init_rng.standard_normal(
        (n, gt.transform.dim))).astype(np.float32))


# The float32 trust check, before any sampling (MetalPlanet's discipline).
print("\n", anvil.validate_precision(gt.target, ball(32)), sep="")


def summarise(res, wall, tag):
    n_div = int(res.extras.get("n_divergent", 0))
    rhat = split_rhat(np.asarray(res.get_chain(), dtype=np.float64))
    flat = gt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    nat = gt.natural(flat)
    print(f"\n--- {tag}: {wall:.0f} s | divergences {n_div} | "
          f"accept {np.mean(res.accept_fraction):.2f} | "
          f"max R-hat {rhat.max():.3f}")
    print(f"{'parameter':>12s} {'median':>13s} {'sd':>11s} {'truth':>13s} {'z':>7s}")
    cols = {}
    for i, name in enumerate(PARAM_NAMES):
        v, truth = flat[:, i], truth_model[i]
        cols[name] = v
        print(f"{name:>12s} {np.median(v):13.6f} {v.std():11.6f} "
              f"{truth:13.6f} {(np.median(v) - truth) / v.std():+7.2f}")
    for name, truth in zip(("gp_sigma", "gp_rho", "gp_Q", "jitter"), gp_truth):
        v = nat[name]
        cols[name] = v
        print(f"{name:>12s} {np.median(v):13.6f} {v.std():11.6f} "
              f"{truth:13.6f} {(np.median(v) - truth) / v.std():+7.2f}")
    return cols


# ChEES decorrelates in a couple of iterations and wants many chains with few
# draws; the ensemble's autocorrelation is ~120 iterations and wants the
# opposite.  Configuring both the same way produces an unconverged showcase.
runs = {}
with mx.stream(mx.gpu):
    t0 = time.perf_counter()
    res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=128), gt.target,
                    ball(128), n_warmup=400, n_samples=400, seed=2,
                    progress=False)
    runs["ChEES-HMC"] = summarise(res, time.perf_counter() - t0,
                                  "ChEES-HMC, 128 chains, 400+400")

    t0 = time.perf_counter()
    res = anvil.run(EnsembleKernel(gt.target,
                                   moves=[(StretchMove(), 0.5), (DEMove(), 0.5)],
                                   seed=1),
                    gt.target, ball(128), n_warmup=500, n_samples=6000, seed=2,
                    progress=False)
    runs["ensemble"] = summarise(res, time.perf_counter() - t0,
                                 "Ensemble (stretch + DE), 128 walkers, 500+6000")

# ------------------------------------------------- cross-sampler agreement
print("\n--- cross-sampler agreement (the check a wrong gradient fails) ---")
print(f"{'parameter':>12s} {'|dmedian|/sd':>14s} {'sd ratio':>10s}")
a, b = runs["ChEES-HMC"], runs["ensemble"]
worst_shift = worst_width = 0.0
for name in a:
    sd = 0.5 * (a[name].std() + b[name].std())
    shift = abs(np.median(a[name]) - np.median(b[name])) / sd
    ratio = a[name].std() / b[name].std()
    worst_shift = max(worst_shift, shift)
    worst_width = max(worst_width, abs(np.log(ratio)))
    print(f"{name:>12s} {shift:14.3f} {ratio:10.3f}")
print(f"\nworst median shift {worst_shift:.3f} sigma; "
      f"worst width ratio {np.exp(worst_width):.3f}x")
