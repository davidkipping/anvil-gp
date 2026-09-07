"""celerite2 CPU baselines: the denominator anvil-gp has to beat.

Two denominators, because they answer different questions:

* **single core** -- what one Python process doing a loop over parameter
  vectors costs. This is the honest comparison for a user who has not
  parallelised anything.
* **multiprocess over all cores** -- the honest comparison for a user who
  has. The brief is explicit that this is the denominator the >=100x gate
  is measured against, so it is the one that matters for M3's target.

Both time ``compute(t, yerr) + log_likelihood(y)`` per parameter vector,
which is exactly the work ``GPLogLike.__call__`` does for one row of ``v``.
Caveat to keep in view: celerite2 is float64 throughout; anvil-gp's
production path is float32 with a separate float64 verification path.

Run rested and in an isolated process -- sustained load on this SoC moves
wall times by 2-3x, and the CPU shares the thermal budget with the GPU.

    .venv/bin/python benchmarks/bench_vs_celerite2.py --n-data 10000 --n-param 1024
"""

import argparse
import multiprocessing as mp
import os
import statistics
import time

import numpy as np

_DATA = {}


def _init(t, y, yerr):
    _DATA["t"], _DATA["y"], _DATA["yerr"] = t, y, yerr


def _loglike_chunk(params):
    """log-likelihood for a chunk of (sigma, rho, Q) vectors."""
    import celerite2
    from celerite2 import terms as c2t
    t, y, yerr = _DATA["t"], _DATA["y"], _DATA["yerr"]
    out = np.empty(len(params))
    for i, (sigma, rho, q) in enumerate(params):
        gp = celerite2.GaussianProcess(
            c2t.SHOTerm(sigma=sigma, rho=rho, Q=q), mean=0.0)
        gp.compute(t, yerr=yerr)
        out[i] = gp.log_likelihood(y)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-data", type=int, default=10_000)
    p.add_argument("--n-param", type=int, default=1024)
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--n-proc", type=int, default=0,
                   help="0 = os.cpu_count()")
    a = p.parse_args()

    rng = np.random.default_rng(0)
    t = np.arange(a.n_data) * (30.0 / a.n_data)
    yerr = np.full(a.n_data, 0.3)
    y = rng.standard_normal(a.n_data) * 0.9
    params = np.column_stack([
        np.exp(rng.normal(0.0, 0.05, a.n_param)),
        4.0 * np.exp(rng.normal(0.0, 0.05, a.n_param)),
        3.0 * np.exp(rng.normal(0.0, 0.05, a.n_param))])

    _init(t, y, yerr)
    # one evaluation, to size the per-call cost
    ts = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        _loglike_chunk(params[:1])
        ts.append(time.perf_counter() - t0)
    per_call = statistics.median(ts)

    ts = []
    for _ in range(max(3, a.repeat // 2)):
        t0 = time.perf_counter()
        _loglike_chunk(params)
        ts.append(time.perf_counter() - t0)
    single = statistics.median(ts)

    n_proc = a.n_proc or os.cpu_count() or 1
    chunks = np.array_split(params, n_proc)
    with mp.Pool(n_proc, initializer=_init, initargs=(t, y, yerr)) as pool:
        pool.map(_loglike_chunk, [c[:1] for c in chunks])   # warm the workers
        ts = []
        for _ in range(max(3, a.repeat // 2)):
            t0 = time.perf_counter()
            pool.map(_loglike_chunk, chunks)
            ts.append(time.perf_counter() - t0)
        multi = statistics.median(ts)

    print(f"celerite2 fp64 CPU, SHO, N={a.n_data}, {a.n_param} parameter vectors")
    print(f"  per evaluation           : {per_call*1e3:8.3f} ms")
    print(f"  {a.n_param} in a loop (1 core) : {single*1e3:8.1f} ms")
    print(f"  {a.n_param} on {n_proc:2d} cores       : {multi*1e3:8.1f} ms "
          f"({single/multi:.1f}x speedup, {100*single/multi/n_proc:.0f}% efficiency)")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
