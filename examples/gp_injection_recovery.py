"""End-to-end: inject a quasi-periodic signal, recover its GP hyperparameters.

Demonstrates the whole stack on a problem small enough to run on the
*reference* path (a few minutes on an M2 Max: 5800 stretch steps over 1024
chains).  Note the chain count -- at this N the likelihood is bound by op
count rather than by data, so 1024 chains cost only ~1.6x what 256 do:

  1. float64 host world -- data synthesis through the dense covariance, the
     data-derived references every parameter is an offset from;
  2. float32 GPU world -- the chunked-scan reduction behind anvil's
     LogDensity contract, hyperparameters sampled jointly with a jitter;
  3. the precision harness, run before any sampling;
  4. sampling, and truth recovery in natural units.

The gradient-free stretch move is used deliberately: it is what this
example is about.  ChEES-HMC needs nothing extra since 2026-09-27 -- it
used to require a process-global `mx.disable_compile()`, and the one
function responsible is fixed; see docs/float32.md.

Run:  .venv/bin/python examples/gp_injection_recovery.py
"""

import os
import time

import numpy as np

import anvil
from anvil.diagnostics import ess_bulk, split_rhat
from anvil.kernels.ensemble import EnsembleKernel

import anvilgp as ag
from anvilgp.anvil import make_target
from anvilgp.likelihood import GPPolicy
from anvilgp.oracle import dense_covariance

# ------------------------------------------------------------ truth (fp64)
N_DATA = 1000
BASELINE = 40.0
SIGMA_TRUE, RHO_TRUE, Q_TRUE = 1.5, 7.3, 8.0
JITTER_TRUE = 0.12
YERR = 0.20
# SEED is chosen so the *realized* variance of this draw is close to the
# process variance.  With only ~5.5 correlation times in the baseline, a
# single draw's RMS scatters widely around sigma_GP, and the posterior
# correctly tracks the data rather than the truth -- see
# tests/test_calibration.py, which checks unbiasedness across realizations
# instead of trusting one.
#
# N_CHAINS is 1024 rather than 256 because at this N the likelihood is bound
# by op count, not by data: 4x the chains costs ~1.6x the time.
N_CHAINS, SEED = 1024, 20260917

rng = np.random.default_rng(SEED)
t = np.arange(N_DATA) * (BASELINE / N_DATA)   # regular cadence, as real data
yerr = np.full(N_DATA, YERR)

term = ag.SHOTerm()
truth = np.array([SIGMA_TRUE, RHO_TRUE, Q_TRUE])
cov = dense_covariance(term, truth, t, yerr, jitter=JITTER_TRUE)
y = np.linalg.cholesky(cov) @ rng.standard_normal(N_DATA)
print(f"injected SHO: sigma={SIGMA_TRUE}, rho={RHO_TRUE} d, Q={Q_TRUE}, "
      f"jitter={JITTER_TRUE} on {N_DATA} points with {YERR} errors\n")

# --------------------------------------------------------- target assembly
# METAL=1 in the environment runs the fused kernel instead of the MLX graph.
# Since M4 it is differentiable too, though the stretch move below needs no
# gradient.  Same answers either way -- that is what the parity tests pin --
# so this is purely a wall-clock switch.
USE_METAL = os.environ.get("METAL", "0") not in ("", "0", "false", "no")
gt = make_target(term, t, y, yerr,
                 policy=GPPolicy(block=1024, sub_block=16, metal=USE_METAL))
print("likelihood path   :", "fused Metal kernel" if USE_METAL else "MLX graph")
print("sampled parameters:", ", ".join(gt.names))
refs = [h.ref for h in gt.hypers]
print("host references   :", ", ".join(f"{r:.4g}" for r in refs))
print("(v = 0 means every scale parameter sits at its reference)\n")

u0 = gt.u0(N_CHAINS, seed=1, spread=0.3)

# ------------------------------------------------------- trust check first
print(anvil.validate_precision(gt.target, u0[:16]), "\n")

# ---------------------------------------------------------------- sampling
t0 = time.perf_counter()
res = anvil.run(EnsembleKernel(gt.target, seed=2), gt.target, u0,
                n_warmup=4000, n_samples=600, thin=3, seed=3)
wall = time.perf_counter() - t0
chain = res.get_chain()
ess, rhat = ess_bulk(chain), split_rhat(chain)
print(f"stretch: {wall:.1f} s | min ESS {ess.min():,.0f} "
      f"({ess.min()/wall:,.1f} ESS/s) | max R-hat {rhat.max():.4f}\n")

# --------------------------------------------------- posteriors, natural units
flat_u = res.get_chain(flat=True)
v_model = gt.transform.model_np(flat_u.astype(np.float64))
nat = gt.natural(v_model)
truth_nat = dict(gp_sigma=SIGMA_TRUE, gp_rho=RHO_TRUE, gp_Q=Q_TRUE,
                 jitter=JITTER_TRUE)
print(f"realized RMS(y) = {y.std():.3f} vs process "
      f"sqrt(sigma^2 + jitter^2 + yerr^2) = "
      f"{np.sqrt(SIGMA_TRUE**2 + JITTER_TRUE**2 + YERR**2):.3f}\n")
print(f"{'parameter':>12s} {'posterior':>26s} {'truth':>10s}  {'z':>6s}")
for name, draws in nat.items():
    mu, sd = draws.mean(), draws.std()
    tv = truth_nat[name]
    print(f"{name:>12s} {mu:>14.4f} +/- {sd:<9.4f} {tv:>10.4f}  "
          f"{(mu - tv) / sd:>+6.2f}")

# ------------------------------------------------------------------ figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("\nmatplotlib not installed; skipping figure")

from anvilgp.oracle import dense_predict

# Posterior-median GP prediction.  dense_predict is an O(N^3) float64 solve:
# the right tool once per figure, the wrong tool once per leapfrog step,
# which is why it lives in oracle.py and not in the likelihood.
med_nat = {k: float(np.median(v)) for k, v in nat.items()}
kp_med = np.array([med_nat["gp_sigma"], med_nat["gp_rho"], med_nat["gp_Q"]])
mean, var = dense_predict(term, kp_med, t, y, yerr,
                          jitter=med_nat["jitter"])

BLUE, INK, MUTED = "#2563eb", "#1f2937", "#9ca3af"
fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.2, 1.0]})
ax = axes[0, 0].get_gridspec()
for a in axes[0]:
    a.remove()
ax = fig.add_subplot(ax[0, :])
ax.errorbar(t, y, yerr=yerr, fmt=".", ms=3, color=MUTED, elinewidth=0.5,
            label="injected data")
ax.plot(t, mean, color=BLUE, lw=1.6, label="posterior-median GP prediction")
ax.fill_between(t, mean - np.sqrt(np.maximum(var, 0)),
                mean + np.sqrt(np.maximum(var, 0)), color=BLUE, alpha=0.2)
ax.set_xlabel("time [d]")
ax.set_ylabel("flux")
ax.set_title(f"anvil-gp -- SHO kernel recovered jointly with a jitter "
             f"({N_DATA} pts, {N_CHAINS} chains)", color=INK)
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.15)

for a, (name, tv) in zip(axes[1], [("gp_rho", RHO_TRUE), ("gp_Q", Q_TRUE)]):
    a.hist(nat[name], bins=60, density=True, histtype="step", lw=2,
           color=BLUE)
    a.axvline(tv, color=INK, ls="--", lw=1)
    a.set_xlabel(name)
    a.set_yticks([])
    a.grid(alpha=0.15)
axes[1, 0].set_ylabel("posterior density")

fig.savefig("examples/gp_injection_recovery.png", dpi=150)
print("\nfigure saved: examples/gp_injection_recovery.png")
