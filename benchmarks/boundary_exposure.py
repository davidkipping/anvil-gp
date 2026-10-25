"""Where the kernel's dense block boundary fails, and whether the square-root
one fixes it (docs/boundary-plan.md, stage 5).

For each setup: data drawn on-model, 16 draws from the float64 Laplace
posterior (`floor_exposure.posterior_draws`), and for each boundary the
worst float32 gradient error per posterior sigma against float64 central
differences, and the sd of the float32 value error across the draws.
Default blocks.  Accuracy only: a busy GPU does not change it.

    .venv/bin/python benchmarks/boundary_exposure.py           # every row
    .venv/bin/python benchmarks/boundary_exposure.py "2 s,m=3"  # by label
"""
import sys
import warnings

import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from floor_exposure import TWO_PI, GQ, hi, posterior_draws, sample  # noqa: E402

warnings.simplefilter("ignore")
mx.set_default_device(mx.cpu)
DAY = 86400.0
SHO2 = (lambda: ag.SHOTerm() + ag.SHOTerm(), [0.1, 3 / TWO_PI, 5.0, 0.03, 0.2, GQ])
SHO1 = (lambda: ag.SHOTerm(), [0.1, 3 / TWO_PI, 5.0])
SHO_REAL = (lambda: ag.SHOTerm() + ag.RealTerm(), [0.1, 3 / TWO_PI, 5.0, 0.03, 1 / 0.2])
ROT = (lambda: ag.RotationTerm(), [0.1, 3.0, 5.0, 1.0, 0.5])


def regular(n, sec):
    return np.arange(n) * sec / DAY


def gapped(n, sec):              # two TESS-like orbits with a 1-day gap
    t = np.arange(n) * sec / DAY
    t[n // 2:] += 1.0
    return t


ROWS = [  # label, term, t-builder, n, cadence (s), s/e
    ("m=4 SHO+SHO, 20 s, s/e 100", SHO2, regular, 4096, 20, 100),
    ("m=4 SHO+SHO, 20 s, s/e 30", SHO2, regular, 4096, 20, 30),
    ("m=4 SHO+SHO, 20 s, s/e 10", SHO2, regular, 4096, 20, 10),
    ("m=4 SHO+SHO, 2 s, s/e 100", SHO2, regular, 4096, 2, 100),
    ("m=4 SHO+SHO, 2 s, s/e 30", SHO2, regular, 4096, 2, 30),
    ("m=4 SHO+SHO, 2 min, s/e 100", SHO2, regular, 4096, 120, 100),
    ("m=4 SHO+SHO, 30 min, s/e 100", SHO2, regular, 4096, 1800, 100),
    ("m=4 SHO+SHO, 20 s gapped, N=8192, s/e 100", SHO2, gapped, 8192, 20, 100),
    ("m=4 SHO+SHO, 2 min, N=16384, s/e 100", SHO2, regular, 16384, 120, 100),
    ("m=4 Rotation, 2 min, s/e 100", ROT, regular, 4096, 120, 100),
    ("m=3 SHO+Real, 20 s, s/e 100", SHO_REAL, regular, 4096, 20, 100),
    ("m=3 SHO+Real, 2 s, s/e 100", SHO_REAL, regular, 4096, 2, 100),
    ("m=2 SHO, 20 s, s/e 100", SHO1, regular, 4096, 20, 100),
    ("m=2 SHO, 20 s, s/e 300", SHO1, regular, 4096, 20, 300),
    ("m=2 SHO, 2 s, s/e 100", SHO1, regular, 4096, 2, 100),
    ("m=2 SHO, 2 s, N=16384, s/e 300", SHO1, regular, 16384, 2, 300),
]
pick = sys.argv[1].split(",") if len(sys.argv) > 1 else None


def lo(gp, V):
    u = mx.array(V, dtype=mx.float32)
    with mx.stream(mx.gpu):
        g = mx.grad(lambda u: gp(u).sum())(u)
        f = gp(u)
        mx.eval(f, g)
    return np.array(f), np.array(g)


print(f"{'setup':44s} {'blocks':>9s} | {'grad sigma: dense':>17s} {'sqrt':>6s} | "
      f"{'value sd: dense':>15s} {'sqrt':>6s}")
for label, (mk, kp), tb, n, sec, se in ROWS:
    if pick and not any(p.strip() in label for p in pick):
        continue
    term = mk()
    t = tb(n, sec)
    yerr = np.full(n, kp[0] / se)
    y = sample(term, kp, t, yerr, np.random.default_rng(2))
    d = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=float(x))
                          for i, x in enumerate(kp)], n_dim=d)
    ref = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy(kappa_max=None))
    V, sd = posterior_draws(ref, d, 16)
    fh = hi(ref, V)
    P = np.repeat(V, 2 * d, axis=0)
    for k in range(len(V)):
        for i in range(d):
            P[k * 2 * d + 2 * i, i] += 1e-5
            P[k * 2 * d + 2 * i + 1, i] -= 1e-5
    ff = hi(ref, P).reshape(len(V), d, 2)
    G = (ff[..., 0] - ff[..., 1]) / 2e-5
    res = {}
    for mode in ("dense", "sqrt"):
        gp = GPLogLike(None, t, None, y, yerr, term, pm,
                       GPPolicy(metal=True, kappa_max=None, metal_boundary=mode))
        f, g = lo(gp, V)
        res[mode] = ((np.abs(g - G) * sd).max(), np.std(f - fh))
    nb = -(-n // gp.metal_block)
    print(f"{label:44s} {f'{nb}x{gp.metal_block}':>9s} | {res['dense'][0]:17.3f} "
          f"{res['sqrt'][0]:6.3f} | {res['dense'][1]:15.3f} {res['sqrt'][1]:6.3f}",
          flush=True)
