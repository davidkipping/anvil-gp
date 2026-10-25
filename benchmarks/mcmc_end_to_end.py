"""The worked example in docs/benchmarks.md: an ensemble MCMC run, end to end.

600 steps (300 warmup + 300 samples) of anvil's `EnsembleKernel` with 1024
walkers on the SHO kernel and the fused Metal kernel, at N = 1,000 and
100,000, on `bench_docs_tables.py`'s gapped cadence and white-noise data.
Minimum wall time over `--rounds` runs in one process (the first pays
compilation).  Also times one batched call at the half-width the red-black
move makes (512) against the full width (1024).

    .venv/bin/python benchmarks/mcmc_end_to_end.py

Run on a quiet machine (`benchmarks/verify_readme.py`'s probe first).
"""
import argparse
import time
import warnings

import mlx.core as mx
import numpy as np

import anvil
from anvil.kernels.ensemble import EnsembleKernel
import anvilgp as ag
from anvilgp import GPPolicy
from anvilgp.anvil import make_target

p = argparse.ArgumentParser()
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--sizes", default="1000,100000")
a = p.parse_args()
warnings.simplefilter("ignore")


def make_data(n, seed=7):                 # bench_docs_tables.make_data
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.02
    t[n // 2:] += 20.0
    return t, rng.standard_normal(n), np.full(n, 0.3)


def call_ms(gt, width, reps=15):
    u = gt.u0(width, seed=1)
    f = mx.compile(gt.target.log_prob)
    with mx.stream(mx.gpu):
        for _ in range(3):
            mx.eval(f(u))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mx.eval(f(u))
            ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


for n in (int(s) for s in a.sizes.split(",")):
    t, y, yerr = make_data(n)
    gt = make_target(ag.SHOTerm(), t, y, yerr,
                     policy=GPPolicy(block=4096, metal=True))
    walls = []
    for r in range(a.rounds):
        t0 = time.perf_counter()
        res = anvil.run(EnsembleKernel(gt.target, seed=2), gt.target,
                        gt.u0(1024, seed=1), n_warmup=300, n_samples=300,
                        seed=3, progress=False)
        mx.eval(mx.array(res.get_chain()[-1]))
        walls.append(time.perf_counter() - t0)
    print(f"N = {n:>7,}: end to end {min(walls):.2f} s "
          f"(rounds {', '.join(f'{w:.2f}' for w in walls)}); one call "
          f"512 wide {call_ms(gt, 512):.2f} ms, 1024 wide "
          f"{call_ms(gt, 1024):.2f} ms")
