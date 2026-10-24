"""float32 error on realistic cadences: TESS, Kepler, simultaneous
instruments, RV (docs/float32.md, section 7).

Error = |float32 - float64 ``.hi``| in log-units at the true parameters,
data drawn at them, worst of two draws, for the default MLX path and the
fused kernel.  "s/e" is the variability amplitude over the per-point error.
Every evaluation runs with ``kappa_max=None``: the guard must not hide what
is being measured.  Accuracy, not timing: a busy GPU does not change the
numbers.

    python benchmarks/cadence_exposure.py
"""

import warnings

import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
from anvilgp.oracle import dense_covariance

N = 4096
TWO_PI = 2 * np.pi
GRAN_Q = 1 / np.sqrt(2)


def worst(label, t, term, kp, yerr, seeds=(1, 2)):
    pm = ParamMap(kernel=[KernelParam(link="const", value=float(x))
                          for x in kp], n_dim=1)
    chol = np.linalg.cholesky(dense_covariance(term, np.array(kp), t, yerr))
    kap, err = 0.0, {"mlx": 0.0, "metal": 0.0}
    for seed in seeds:
        y = chol @ np.random.default_rng(seed).standard_normal(len(t))
        ref = GPLogLike(None, t, None, y, yerr, term, pm,
                        GPPolicy(kappa_max=None))
        hi = float(np.array(ref.hi(mx.zeros((1, 1), dtype=mx.float64,
                                            stream=mx.cpu)))[0])
        kap = max(kap, float(np.array(ref.kappa(mx.zeros((1, 1)))).max()))
        for name, metal in (("mlx", False), ("metal", True)):
            gp = GPLogLike(None, t, None, y, yerr, term, pm,
                           GPPolicy(metal=metal, kappa_max=None))
            with mx.stream(mx.gpu):
                lo = float(np.array(gp(mx.zeros((1, 1),
                                                dtype=mx.float32)))[0])
            err[name] = max(err[name], abs(lo - hi))
    print(f"{label:58s} kappa {kap:8.0f}   mlx {err['mlx']:8.3f}"
          f"   metal {err['metal']:8.3f}", flush=True)


def main():
    warnings.simplefilter("ignore")
    sho2 = ag.SHOTerm() + ag.SHOTerm()
    sho = ag.SHOTerm()
    ye = np.full(N, 1e-3)

    # TESS: 2-min cadence, two ~3-day orbits with a 1-day gap
    tess = np.arange(N) * (2 / 1440)
    tess[N // 2:] += 1.0
    for se in (10, 30, 100):
        a = 1e-3 * se
        worst(f"TESS 2-min, rotation 3 d + granulation 0.2 d, s/e = {se}",
              tess, sho2, [a, 3 / TWO_PI, 5.0, 0.3 * a, 0.2, GRAN_Q], ye)

    # Kepler long cadence: 30 min, ~85 days
    kep = np.arange(N) * (0.5 / 24)
    for se in (10, 100):
        a = 1e-3 * se
        worst(f"Kepler 30-min, rotation 10 d + granulation 1 d, s/e = {se}",
              kep, sho2, [a, 10 / TWO_PI, 5.0, 0.3 * a, 1.0, GRAN_Q], ye)

    # two instruments observing together, clocks 1 s apart
    base = np.arange(N // 2) * (2 / 1440)
    duo = np.sort(np.concatenate([base, base + 1 / 86400]))
    for se in (10, 30, 100):
        a = 1e-3 * se
        worst(f"two instruments, simultaneous to 1 s, s/e = {se}",
              duo, sho2, [a, 3 / TWO_PI, 5.0, 0.3 * a, 0.2, GRAN_Q], ye)

    # RV: 300 nights in 3 years, 1-3 exposures within an hour each night
    r = np.random.default_rng(7)
    nights = np.sort(r.choice(1100, 300, replace=False)).astype(float)
    rv = np.sort(np.concatenate(
        [n + r.uniform(0, 1 / 24, r.integers(1, 4)) for n in nights]))
    for se in (3, 10, 30):
        worst(f"RV {len(rv)} points, 1-3 per night, rotation 25 d, s/e = {se}",
              rv, sho2, [float(se), 25 / TWO_PI, 5.0, 0.3 * se, 5.0, GRAN_Q],
              np.ones(len(rv)))

    # a single term, for contrast
    for se in (30, 100, 300):
        worst(f"single SHO, TESS 2-min, rotation 3 d, s/e = {se}",
              tess, sho, [1e-3 * se, 3 / TWO_PI, 5.0], ye)
    for se in (30, 100):
        worst(f"single SHO, two instruments, s/e = {se}",
              duo, sho, [1e-3 * se, 3 / TWO_PI, 5.0], ye)


if __name__ == "__main__":
    main()
