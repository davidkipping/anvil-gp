"""What the square-root block boundary costs (docs/boundary-plan.md, stage 5).

Interleaved A/B of ``GPPolicy(metal_boundary="dense")`` against the default
``"sqrt"`` on the README's kernel configurations plus the short-block m = 4
case the boundary was built for; forward (compiled) and value + gradient,
minimum over rounds.  Gated by the same stability probe as
``verify_readme.py``: timings from a contended machine are not printed.

    .venv/bin/python benchmarks/boundary_cost.py            # every row
    .venv/bin/python benchmarks/boundary_cost.py --quick    # skip N = 1e5
"""
import argparse
import time
import warnings

import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--repeat", type=int, default=12)
a = p.parse_args()
warnings.simplefilter("ignore")


def gapped(n):                      # verify_readme.py's cadence
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


SHO = (ag.SHOTerm(), [1.0, 4.0, 3.0])
SHO2 = (ag.SHOTerm() + ag.SHOTerm(), [1.0, 4.0, 3.0, 0.3, 0.5, 0.7])


def make(n, term_p, boundary, cadence="gapped", metal_block=None):
    term, pv = term_p
    rng = np.random.default_rng(0)
    t = gapped(n) if cadence == "gapped" else np.arange(n) / 4320
    gp = GPLogLike(None, t, None, rng.standard_normal(n) * 0.9,
                   np.full(n, 0.3), term, ParamMap.log(pv),
                   GPPolicy(block=4096, metal=True, metal_block=metal_block,
                            metal_boundary=boundary, kappa_max=None))
    return gp, len(pv)


def forward(gp, d, chains):
    u = mx.array(np.random.default_rng(1).standard_normal((chains, d))
                 .astype(np.float32) * 0.01)
    f = mx.compile(gp.__call__)
    with mx.stream(mx.gpu):
        return timed(lambda: f(u), a.repeat)


def gradient(gp, d, chains):
    u = mx.array(np.random.default_rng(1).standard_normal((chains, d))
                 .astype(np.float32) * 0.01)
    ones = mx.ones((chains,), dtype=mx.float32)
    with mx.stream(mx.gpu):
        return timed(lambda: mx.vjp(gp, [u], [ones]), max(5, a.repeat // 2))


ROWS = [  # label, n, term, chains, cadence
    ("m=2, N=1e4, 1024 chains", 10_000, SHO, 1024, "gapped"),
    ("m=4, N=1e4, 256 chains", 10_000, SHO2, 256, "gapped"),
    ("m=4, N=4096 (16 x 256), 1024 chains, 20 s", 4096, SHO2, 1024, "20s"),
]
if not a.quick:
    ROWS += [("m=2, N=1e5, 1024 chains", 100_000, SHO, 1024, "gapped"),
             ("m=4, N=1e5, 256 chains", 100_000, SHO2, 256, "gapped")]

# stability probe, as verify_readme.py
PROBE_QUIET_MS = 73.3
rng = np.random.default_rng(0)
gp0 = GPLogLike(None, gapped(10_000), None, rng.standard_normal(10_000) * 0.9,
                np.full(10_000, 0.3), ag.SHOTerm(), ParamMap.log([1.0, 4.0, 3.0]),
                GPPolicy(block=4096))
u0 = mx.array(rng.standard_normal((1024, 3)).astype(np.float32) * 0.01)
f0 = mx.compile(gp0.__call__)
with mx.stream(mx.gpu):
    mins = [timed(lambda: f0(u0), 10) for _ in range(3)]
spread = (max(mins) - min(mins)) / min(mins)
level = min(mins) / PROBE_QUIET_MS
print(f"stability probe {min(mins):.1f} ms ({level:.2f}x quiet), spread {spread:.1%}")
if spread > 0.10 or level > 1.5:
    raise SystemExit("machine too noisy or contended to time; not printing")

best = {}
for r in range(a.rounds):
    for label, n, tp, chains, cad in ROWS:
        for b in ("dense", "sqrt"):
            gp, d = make(n, tp, b, cad)
            for kind, fn in (("forward", forward), ("value+grad", gradient)):
                key = (label, kind, b)
                try:
                    got = fn(gp, d, chains)
                except RuntimeError as e:     # the GPU watchdog, typically
                    print(f"{label} {kind} {b}: {str(e)[:60]}", flush=True)
                    got = np.nan
                best[key] = np.nanmin([best.get(key, np.nan), got]) \
                    if not np.isnan(got) or key in best else got
            del gp
print(f"\n{'config':44s} {'':10s} {'dense':>9s} {'sqrt':>9s}  ratio")
for label, n, tp, chains, cad in ROWS:
    gp, _ = make(n, tp, "sqrt", cad)
    blk = gp.metal_block
    for kind in ("forward", "value+grad"):
        dd, ss = best[(label, kind, "dense")], best[(label, kind, "sqrt")]
        print(f"{label + f' [blk {blk}]':44s} {kind:10s} {dd:8.2f}  {ss:8.2f}  {ss / dd:5.2f}x")
