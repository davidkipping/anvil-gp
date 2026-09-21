"""Check the README's performance claims against the current code.

Every number in a README rots. Both code reviews of this package found
stale figures, and twice the stale figure was one a commit in the *same
session* had already invalidated. So: keep the claims in one machine-
readable table, and give anyone a single command that says which of them
still hold.

    .venv/bin/python benchmarks/verify_readme.py            # all of them
    .venv/bin/python benchmarks/verify_readme.py --quick    # skip N = 1e5

**Timings are gated on a measured stability check, not on the load
average.** Wall times on an M2 Max can swing 2x with background load, but
the load average is a poor proxy: a CPU-bound indexer can sit at 14 while
the GPU is perfectly steady. So this runs one cheap workload three times
first and reports the spread of the three minima. Above 10% it declines to
judge the timing rows; accuracy rows are load-independent and always run.
"""
import argparse
import os
import time

import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap
from anvilgp.oracle import sample_ssm

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="skip the N = 1e5 rows")
p.add_argument("--repeat", type=int, default=15)
p.add_argument("--tol", type=float, default=0.25,
               help="fractional slack on timing claims")
a = p.parse_args()

LOAD = os.getloadavg()[0]


def gapped(n):
    full = np.arange(int(n * 1.3)) * (30.0 / n)
    keep = np.ones(full.size, bool)
    for g in range(6):
        st = (g + 1) * full.size // 8
        keep[st:st + full.size // 40] = False
    return full[keep][:n]


def timed(fn, reps):
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def forward(n, n_chains, metal):
    rng = np.random.default_rng(0)
    t = gapped(n)
    y = rng.standard_normal(n) * 0.9
    gp = GPLogLike(None, t, None, y, np.full(n, 0.3), ag.SHOTerm(),
                   ParamMap.log([1.0, 4.0, 3.0]),
                   GPPolicy(block=4096, metal=metal))
    u = mx.array(rng.standard_normal((n_chains, 3)).astype(np.float32) * 0.01)
    f = mx.compile(gp.__call__)
    with mx.stream(mx.gpu):
        return timed(lambda: f(u), a.repeat)


def gradient(n, n_chains):
    rng = np.random.default_rng(0)
    t = gapped(n)
    y = rng.standard_normal(n) * 0.9
    gp = GPLogLike(None, t, None, y, np.full(n, 0.3), ag.SHOTerm(),
                   ParamMap.log([1.0, 4.0, 3.0]), GPPolicy(block=4096))
    u = mx.array(rng.standard_normal((n_chains, 3)).astype(np.float32) * 0.01)
    ones = mx.ones((n_chains,), dtype=mx.float32)
    with mx.stream(mx.gpu):
        return timed(lambda: mx.vjp(gp, [u], [ones]), max(5, a.repeat // 3))


def fp32_error(n, sig_over_yerr, metal, sub_block=None):
    """|float32 - float64| on ON-MODEL data: the honest precision axis."""
    rng = np.random.default_rng(0)
    dt, rho, qual = 1.0, 100.0, 3.0
    t = np.arange(n) * dt
    y = sample_ssm(ag.SHOTerm(), [sig_over_yerr, rho, qual], dt, n, rng)
    y = y + rng.standard_normal(n)
    yerr = np.full(n, 1.0)
    pm = ParamMap.const([sig_over_yerr, rho, qual])
    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                   GPPolicy(block=4096, metal=metal, sub_block=sub_block))
    with mx.stream(mx.cpu):
        hi = float(np.array(gp.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])
    with mx.stream(mx.gpu):
        lo = float(np.array(gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
    return abs(lo - hi)


#: (label, claimed, getter, kind).  `kind` "time" rows are skipped as
#: unverifiable when the machine is loaded; "acc" rows always run.
CLAIMS = [
    ("MLX forward, 1e4 x 1024", 73.8, lambda: forward(10_000, 1024, False), "time"),
    ("Metal forward, 1e4 x 1024", 2.9, lambda: forward(10_000, 1024, True), "time"),
    ("MLX value+grad, 1e4 x 1024", 880.0, lambda: gradient(10_000, 1024), "time"),
    ("fp32 err, N=25e3, sig/yerr=100 (MLX)", 0.06,
     lambda: fp32_error(25_000, 100.0, False), "acc"),
    ("fp32 err, N=25e3, sig/yerr=100 (Metal)", 0.005,
     lambda: fp32_error(25_000, 100.0, True), "acc"),
]
if not a.quick:
    CLAIMS += [
        ("MLX forward, 1e5 x 1024", 693.0, lambda: forward(100_000, 1024, False), "time"),
        ("Metal forward, 1e5 x 1024", 13.2, lambda: forward(100_000, 1024, True), "time"),
        ("fp32 err, N=1e5, sig/yerr=30 (MLX)", 0.20,
         lambda: fp32_error(100_000, 30.0, False), "acc"),
    ]

def stability():
    """Spread of three independent minima on one cheap workload."""
    rng = np.random.default_rng(0)
    t = gapped(10_000)
    y = rng.standard_normal(10_000) * 0.9
    gp = GPLogLike(None, t, None, y, np.full(10_000, 0.3), ag.SHOTerm(),
                   ParamMap.log([1.0, 4.0, 3.0]), GPPolicy(block=4096))
    u = mx.array(rng.standard_normal((1024, 3)).astype(np.float32) * 0.01)
    f = mx.compile(gp.__call__)
    with mx.stream(mx.gpu):
        mins = [timed(lambda: f(u), 10) for _ in range(3)]
    return min(mins), (max(mins) - min(mins)) / min(mins)


base, spread = stability()
print(f"load average (1 min): {LOAD:.1f}")
print(f"stability probe: {base:.1f} ms, spread across 3 rounds {spread:.1%}"
      + ("" if spread <= 0.10 else "   <-- too noisy to judge timings"))
print()

# Two interleaved passes, best of both. Conditions drift during a run, and
# a short row (the Metal kernel is ~13 ms at N = 1e5) is far more sensitive
# to one scheduling hiccup than a long one; a single pass reported 24.8 ms
# for a row that measures 13.8 when timed on its own.
results = {}
for _ in range(2):
    for label, claimed, fn, kind in CLAIMS:
        if kind == "time" and spread > 0.10:
            continue
        got = fn()
        prev = results.get(label)
        results[label] = got if prev is None else min(prev, got)

print(f"{'claim':>40s} {'README':>9s} {'measured':>9s}   verdict")
for label, claimed, fn, kind in CLAIMS:
    if label not in results:
        print(f"{label:>40s} {claimed:>9.1f} {'--':>9s}   skipped (too noisy)")
        continue
    got = results[label]
    ratio = got / claimed
    ok = ratio <= 1.0 + a.tol
    print(f"{label:>40s} {claimed:>9.3g} {got:>9.3g}   "
          f"{'ok' if ok else 'OVER'} ({ratio:.2f}x)")

print()
print("Accuracy rows are absolute log-unit errors and must not grow; timing")
print("rows are minima and allowed {:.0%} slack.".format(a.tol))
if spread > 0.10:
    print("Timings were skipped: the stability probe varied by "
          f"{spread:.0%}. Re-run when it is under 10%.")
