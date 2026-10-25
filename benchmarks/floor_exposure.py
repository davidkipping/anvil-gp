"""Exposure to the residual float32 floor (docs/float32.md, section 7),
measured where it matters to a sampler: across the posterior, at realistic N.

A constant offset in log L is invisible to MCMC.  What moves a posterior is
how the float32 error *varies* over it: the float32 target is the true one
times ``exp(e)``, so to first order no posterior mean shifts by more than
``sd(e)`` posterior standard deviations.  For each setup:

* data drawn on-model at the true parameters (O(N) state-space sampler,
  any cadence, gaps included);
* posterior covariance from a float64 finite-difference Hessian of ``.hi``
  at the truth, every width capped at 0.5 in log (weak directions are
  prior-bounded in a real fit);
* ``K`` draws from it; ``sd(e)`` and ``max|e|`` of float32 minus float64;
* the float32 gradient against a float64 central difference at ``KG`` of
  them, reported as ``|dg| * sd`` -- the log L error per posterior sigma of
  motion -- and as a worst relative error.

Every evaluation runs with ``kappa_max=None``.  One data draw per setup.
Accuracy, not timing: a busy GPU does not change the numbers.  The default
MLX gradient exceeds Metal's buffer-count limit at N ~ 1e5 (README, "What is
left"); those entries print ``nan``.  About 25 minutes on an M2 Max.

    python benchmarks/floor_exposure.py            # every row
    python benchmarks/floor_exposure.py "TESS 20-s,Kepler"
"""

import sys
import time
import warnings

import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap

TWO_PI = 2 * np.pi
GQ = 1 / np.sqrt(2)
K, KG = 32, 6


def sample(term, kp, t, yerr, rng):
    """On-model draw at any cadence, exact in float64."""
    lags = np.diff(t)
    uniq, inv = np.unique(lags, return_inverse=True)
    with mx.stream(mx.cpu):
        phi, q, h = term.matrices(mx.array([list(kp)], dtype=mx.float64),
                                  mx.array(uniq, dtype=mx.float64))
        mx.eval(phi, q, h)
    Phi = np.array(phi, dtype=np.float64)[0]
    Q = np.array(q, dtype=np.float64)[0]
    H = np.array(h, dtype=np.float64)[0]
    LQ = []
    for qq in Q:
        w, V = np.linalg.eigh(0.5 * (qq + qq.T))
        LQ.append(V * np.sqrt(np.maximum(w, 0)))
    m = Phi.shape[-1]
    x = rng.standard_normal(m)
    out = np.empty(len(t))
    out[0] = H @ x
    z = rng.standard_normal((len(t), m))
    for i in range(1, len(t)):
        j = inv[i - 1]
        x = Phi[j] @ x + LQ[j] @ z[i]
        out[i] = H @ x
    return out + yerr * rng.standard_normal(len(t))


def hi(gp, V):
    with mx.stream(mx.cpu):
        # dtype= matters: mx.array of a float64 ndarray is float32, which rounded
        # V +- 1e-5 before the float64 evaluation (<= 0.005 sigma in the FD
        # gradients measured at 20-s, N = 4096; values were unaffected)
        return np.array(gp.hi(mx.array(V, dtype=mx.float64), refuse=False))


def lo(gp, V, grad=False):
    outs, grads = [], []
    for c in range(0, len(V), 8):
        v = mx.array(V[c:c + 8], dtype=mx.float32)
        with mx.stream(mx.gpu):
            if grad:
                _, g = mx.value_and_grad(lambda u: gp(u).sum())(v)
                f = gp(v)
                mx.eval(f, g)
                grads.append(np.array(g))
            else:
                f = gp(v)
                mx.eval(f)
        outs.append(np.array(f))
    return np.concatenate(outs), (np.concatenate(grads) if grad else None)


def posterior_draws(ref, d, k=K):
    """``k`` draws from N(0, H^-1), H the float64 FD Hessian of ``.hi`` at
    v = 0, and the marginal widths; draw 0 is v = 0 itself."""
    hstep = 1e-3
    pts = [np.zeros(d)]
    for i in range(d):
        for j in range(i, d):
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = np.zeros(d)
                p[i] += si * hstep
                p[j] += sj * hstep
                pts.append(p)
    f = hi(ref, np.array(pts))
    Hm = np.zeros((d, d))
    c = 1
    for i in range(d):
        for j in range(i, d):
            fpp, fpm, fmp, fmm = f[c:c + 4]
            c += 4
            Hm[i, j] = Hm[j, i] = -(fpp - fpm - fmp + fmm) / (4 * hstep ** 2)
    w, U = np.linalg.eigh(Hm)
    w = np.maximum(w, 4.0)               # no width beyond 0.5 in log
    sd = np.sqrt(np.diag((U / w) @ U.T))
    V = (U / np.sqrt(w) @ np.random.default_rng(99).standard_normal((d, k))).T
    V[0] = 0.0
    return V, sd


def run(label, t, term, kp, yerr, seed=1):
    t0 = time.time()
    y = sample(term, kp, t, yerr, np.random.default_rng(seed))
    d = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=float(x))
                          for i, x in enumerate(kp)], n_dim=d)
    ref = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy(kappa_max=None))
    V, sd = posterior_draws(ref, d)
    fhi = hi(ref, V)
    kap = float(np.array(ref.kappa(mx.array(V, dtype=mx.float32))).max())
    row = f"{label:48s} N={len(t):6d} kappa={kap:7.0f} |"
    gh = 1e-5
    G = []
    for v in V[:KG]:
        P = np.repeat(v[None], 2 * d, 0)
        for i in range(d):
            P[2 * i, i] += gh
            P[2 * i + 1, i] -= gh
        ff = hi(ref, P)
        G.append((ff[0::2] - ff[1::2]) / (2 * gh))
    G = np.array(G)
    for name, metal in (("mlx", False), ("metal", True)):
        gp = GPLogLike(None, t, None, y, yerr, term, pm,
                       GPPolicy(metal=metal, kappa_max=None))
        e = lo(gp, V)[0] - fhi
        try:
            g32 = lo(gp, V[:KG], grad=True)[1]
        except RuntimeError:             # Metal buffer-count limit
            g32 = np.full_like(G, np.nan)
        gerr = np.abs(g32 - G) * sd
        rel = np.max(np.abs(g32 - G) / np.maximum(np.abs(G), 1e-30))
        row += (f" {name}: max|e| {np.abs(e).max():6.3f} sd(e) {e.std():6.3f}"
                f" dg*sd {gerr.max():6.3f} (rel {rel:6.1e}) |")
    print(row + f" {time.time() - t0:.0f}s", flush=True)


def rows():
    sho2 = ag.SHOTerm() + ag.SHOTerm()
    sho = ag.SHOTerm()

    def tess(cad_min):                   # two 13.2-d orbits, 1-d gap
        n = int(13.2 * 1440 / cad_min)
        a = np.arange(n) * cad_min / 1440
        return np.concatenate([a, a + 14.2])

    def rot_gran(a, rot, gran):
        return [a, rot / TWO_PI, 5.0, 0.3 * a, gran, GQ]

    kq = np.arange(int(89 * 48)) / 48    # Kepler long cadence, one quarter
    k16 = np.concatenate([kq + 90 * i for i in range(16)])
    r = np.random.default_rng(7)         # RV: 300 nights, 1-3 per night
    nights = np.sort(r.choice(1100, 300, replace=False)).astype(float)
    rv = np.sort(np.concatenate(
        [n + r.uniform(0, 1 / 24, r.integers(1, 4)) for n in nights]))
    t2, t20 = tess(2), tess(1 / 3)
    e3 = lambda t: np.full(len(t), 1e-3)  # noqa: E731
    out = []
    for se in (3, 10, 30, 100):
        out.append((f"TESS 2-min, SHO+SHO rot 3d, s/e={se}", t2, sho2,
                    rot_gran(1e-3 * se, 3, 0.2), e3(t2)))
    for se in (30, 100, 300):
        out.append((f"TESS 2-min, single SHO rot 3d, s/e={se}", t2, sho,
                    [1e-3 * se, 3 / TWO_PI, 5.0], e3(t2)))
    for se in (10, 30):
        out.append((f"TESS 20-s, SHO+SHO rot 3d, s/e={se}", t20, sho2,
                    rot_gran(1e-3 * se, 3, 0.2), e3(t20)))
    for se in (30, 100, 300):
        out.append((f"Kepler LC quarter, SHO+SHO rot 10d, s/e={se}", kq,
                    sho2, rot_gran(1e-3 * se, 10, 1.0), e3(kq)))
    for se in (30, 100):
        out.append((f"Kepler LC 16 quarters, SHO+SHO rot 10d, s/e={se}", k16,
                    sho2, rot_gran(1e-3 * se, 10, 1.0), e3(k16)))
    for se in (10, 30):
        out.append((f"RV 600 pts, SHO+SHO rot 25d, s/e={se}", rv, sho2,
                    [float(se), 25 / TWO_PI, 5.0, 0.3 * se, 5.0, GQ],
                    np.ones(len(rv))))
    # N and cadence separated: 20-s cadence at a 2-min sector's N, and
    # 2-min cadence at a 20-s sector's N
    a20, a2 = np.arange(19008) / 4320, np.arange(114048) / 720
    out.append(("split: 20-s cadence, N = 19008, s/e=30", a20, sho2,
                rot_gran(0.03, 3, 0.2), e3(a20)))
    out.append(("split: 2-min cadence, N = 114048, s/e=30", a2, sho2,
                rot_gran(0.03, 3, 0.2), e3(a2)))
    return out


def main(which="all"):
    warnings.simplefilter("ignore")
    for r in rows():
        if which == "all" or any(w in r[0] for w in which.split(",")):
            run(*r)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
