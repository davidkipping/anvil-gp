"""Cost of linear marginalisation (`GPLogLike(linear_basis=...)`) on the
kernel: polarisation (1 + k(k+1)/2 filters) against the multi-column
filter (`metal.scan_gram`), relative to the plain likelihood.

    .venv/bin/python benchmarks/marginal_cost.py [--n 10000] [--chains 256]

Prints the verify_readme stability probe first: numbers taken above 1.5x
its quiet level are contention, not the library (docs/benchmarks.md).
"""
import argparse
import time

import mlx.core as mx
import numpy as np

import anvilgp as ag
import anvilgp.likelihood as lk
from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=10_000)
p.add_argument("--chains", type=int, default=256)
p.add_argument("--p", type=int, nargs="+", default=[2, 5, 10])
p.add_argument("--repeat", type=int, default=10)
p.add_argument("--grad", action="store_true", help="also value + gradient")
a = p.parse_args()

PROBE_QUIET_MS = 73.3


def timed(fn, reps):
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def probe():
    rng = np.random.default_rng(0)
    n = 10_000
    t = np.arange(n) * (30.0 / n)
    gp = GPLogLike(None, t, None, rng.standard_normal(n) * 0.9,
                   np.full(n, 0.3), ag.SHOTerm(),
                   ParamMap.log([1.0, 4.0, 3.0]),
                   GPPolicy(block=4096, metal=False))
    u = mx.array(rng.standard_normal((1024, 3)).astype(np.float32) * 0.01)
    f = mx.compile(gp.__call__)
    with mx.stream(mx.gpu):
        return min(timed(lambda: f(u), 10) for _ in range(3))


def build(n, n_lin):
    rng = np.random.default_rng(1)
    t = np.arange(n) * (30.0 / n)
    x = 2 * t / t[-1] - 1
    leg = np.polynomial.legendre.legvander(x, n_lin - 1)        # (n, p)
    y = leg @ rng.standard_normal(n_lin) * 0.1 + rng.standard_normal(n) * 0.9
    basis_m = mx.array(leg.astype(np.float32))

    def basis(vm, x_, m0):
        return mx.broadcast_to(basis_m.astype(m0.dtype),
                               (m0.shape[0],) + basis_m.shape)
    kw = dict(kappa_max=None, metal=True)
    plain = GPLogLike(None, t, None, y, np.full(n, 0.3), ag.SHOTerm(),
                      ParamMap.log([1.0, 4.0, 3.0]), GPPolicy(**kw))
    marg = GPLogLike(None, t, None, y, np.full(n, 0.3), ag.SHOTerm(),
                     ParamMap.log([1.0, 4.0, 3.0]), GPPolicy(**kw),
                     linear_basis=basis)
    return plain, marg


def polarised(gp):
    def f(v):
        return lk._sanitize(gp._marginal(
            v, gp._raw_resid, gp._y32, gp._x_mean32, gp._noise_x32,
            gp._r_ref32)[0], gp._infeasible_pull(v))
    return f


base = probe()
print(f"probe {base:.1f} ms = {base / PROBE_QUIET_MS:.2f}x quiet"
      + ("   <-- contended" if base > 1.5 * PROBE_QUIET_MS else ""))
lk.MAX_GRAM_COLS = 99                   # measure every k on the kernel
rng = np.random.default_rng(0)
u = mx.array(rng.standard_normal((a.chains, 3)).astype(np.float32) * 0.01)
ones = mx.ones((a.chains,), dtype=mx.float32)
print(f"N = {a.n}, {a.chains} chains, SHO; ms (x plain)")
with mx.stream(mx.gpu):
    for n_lin in a.p:
        plain, marg = build(a.n, n_lin)
        t_p = timed(lambda: plain(u), a.repeat)
        t_g = timed(lambda: marg(u), a.repeat)
        pol = polarised(marg)
        t_o = timed(lambda: pol(u), a.repeat)
        line = (f"p = {n_lin:2d} (k = {n_lin + 1:2d})  forward: plain "
                f"{t_p:7.1f}  polarised {t_o:7.1f} ({t_o / t_p:4.1f}x)  "
                f"multi-column {t_g:7.1f} ({t_g / t_p:4.1f}x)")
        if a.grad:
            r = max(3, a.repeat // 3)
            g_p = timed(lambda: mx.vjp(plain, [u], [ones]), r)
            g_o = timed(lambda: mx.vjp(pol, [u], [ones]), r)
            g_g = timed(lambda: mx.vjp(marg, [u], [ones]), r)
            line += (f"\n{'':19s}v+grad:  plain {g_p:7.1f}  polarised "
                     f"{g_o:7.1f} ({g_o / g_p:4.1f}x)  multi-column "
                     f"{g_g:7.1f} ({g_g / g_p:4.1f}x)")
        print(line, flush=True)
