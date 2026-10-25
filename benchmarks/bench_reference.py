"""Wall-time and memory of the pure-MLX reference path (M2 baseline).

This is the *reference*, not the fast path: a tree reduction expressed as
MLX graph ops, with a Cholesky per composition.  Its numbers exist to (a)
size the float64 verification path, (b) give the fused Metal kernel of M3 a
denominator, and (c) show where the graph-op route runs out of road.

Defaults are the TARGET workload -- 1024 chains, N = 1e4, block 4096, a
gapped cadence -- not a small smoke test.  That matters: the old block=256
default made the value+gradient look like 2.7 s when the recommended
setting gives 0.85 s, and that wrong number reached the README.

Report the MINIMUM, not just the median: this machine's wall times swing
~2x with background process load (not thermal state), so a median can sit
entirely inside a slow window.  A/B old against new in the same window.
"""
import argparse, statistics, time
import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap

p = argparse.ArgumentParser()
p.add_argument("--n-data", type=int, default=10_000)
p.add_argument("--n-chains", type=int, default=1024)
p.add_argument("--block", type=int, default=4096)
p.add_argument("--term", default="sho")
p.add_argument("--grad", action="store_true")
p.add_argument("--repeat", type=int, default=5)
p.add_argument("--no-compile", action="store_true")
p.add_argument("--sub-block", type=int, default=None)
p.add_argument("--cadence", choices=("regular","gapped","random"), default="gapped")
p.add_argument("--metal", action="store_true",
               help="use the fused Metal kernel (value and gradient)")
p.add_argument("--metal-block", type=int, default=None)
p.add_argument("--boundary", choices=("dense", "sqrt"), default=None,
               help="the kernel's block boundary (default: the policy's)")
a = p.parse_args()

TERMS = {"sho": (ag.SHOTerm(), [1.0, 4.0, 3.0]),
         "rotation": (ag.RotationTerm(), [1.0, 7.0, 3.0, 1.0, 0.4]),
         "sum2": (ag.SHOTerm() + ag.SHOTerm(), [1.0, 4.0, 3.0, 0.5, 1.0, 2.0]),
         "sum3": (ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm(),
                  [1.0, 4.0, 3.0, 0.5, 1.0, 2.0, 0.3, 0.5, 5.0]),
         "sum4": (ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm(),
                  [1.0, 4.0, 3.0, 0.5, 1.0, 2.0,
                   0.3, 0.5, 5.0, 0.2, 0.25, 1.5])}
term, refs = TERMS[a.term]

rng = np.random.default_rng(0)
if a.cadence == "random":
    t = np.sort(rng.uniform(0.0, 30.0, a.n_data))
elif a.cadence == "regular":
    t = np.arange(a.n_data) * (30.0 / a.n_data)
else:                                    # regular cadence with data gaps
    full = np.arange(int(a.n_data * 1.3)) * (30.0 / a.n_data)
    keep = np.ones(full.size, bool)
    for g in range(6):
        st = (g + 1) * full.size // 8
        keep[st:st + full.size // 40] = False
    t = full[keep][:a.n_data]
yerr = np.full(a.n_data, 0.3)
y = rng.standard_normal(a.n_data) * 0.9
pm = ParamMap.log(refs)
pol = GPPolicy(block=a.block, sub_block=a.sub_block, metal=a.metal,
               metal_boundary=a.boundary,
               **({} if a.metal_block is None
                  else {"metal_block": a.metal_block}))
gp = GPLogLike(None, t, None, y, yerr, term, pm, pol)
if a.metal:
    ok, why = gp.metal_eligible()
    if not ok:
        raise SystemExit(f"--metal requested but unusable: {why}")
u = mx.array(rng.standard_normal((a.n_chains, len(refs))).astype(np.float32) * 0.01)

# mx.compile is worth ~2.4x on the forward path now that the combine is
# elementwise rather than tiny-GEMM (it was 1.05x before, because a batched
# GEMM is opaque to the fusion pass).  It cannot be used on the *backward*
# pass -- see docs/float32.md.
_fwd = gp.__call__ if a.no_compile else mx.compile(gp.__call__)


def once():
    if a.grad:
        v, g = mx.vjp(gp, [u], [mx.ones((a.n_chains,), dtype=mx.float32)])
        mx.eval(v, g)
    else:
        mx.eval(_fwd(u))

t0 = time.perf_counter(); once(); trace = time.perf_counter() - t0
mx.reset_peak_memory()
ts = []
for _ in range(a.repeat):
    t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
med = statistics.median(ts)
best = min(ts)
print(f"term={a.term} m={term.n_state} N={a.n_data} chains={a.n_chains} "
      f"block={a.block} sub={gp.sub_block} {a.cadence}({gp.n_distinct_lags} lags) "
      f"{'METAL ' if a.metal else ''}"
      f"{'value+grad' if a.grad else ('forward' if a.no_compile else 'fwd-compiled')}: "
      f"{med*1e3:.1f} ms (min {best*1e3:.1f}, first call {trace*1e3:.0f} ms) "
      f"peak {mx.get_peak_memory()/2**30:.2f} GiB")
