"""The comparison tables in `docs/benchmarks.md`, regenerated from scratch.

Emits markdown. Every number in the docs' comparison section comes from
this script, so a claim there can be rechecked with one command:

    .venv/bin/python benchmarks/bench_docs_tables.py > /tmp/tables.md

Three things are measured, because they answer different questions.

* **Speed against celerite2, as a function of how many parameter vectors
  you need at once.** This is the axis that decides whether anvil-gp helps
  you at all. celerite2 evaluates one vector at a time on the CPU; anvil-gp
  evaluates a whole ensemble on the GPU. At one vector the comparison
  favours celerite2 and the docs say so.
* **Speed by kernel**, since the state dimension m drives the kernel's cost
  (the running element is 3m^2+2m+1 floats in registers).
* **Accuracy against a float64 dense Cholesky**, for both libraries.
  anvil-gp's production path is float32; celerite2 is float64 throughout.
  The question is whether float32 costs anything that matters.

Run on a quiet machine: wall times on this SoC move 2x with background
load. `benchmarks/verify_readme.py` has a stability probe; use it first.
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

_DATA = {}
_POOL = {}          # (n_proc, kind, n) -> Pool, so it is built once


def _get_pool(n_proc, kind, t, y, yerr):
    """One warmed pool per (workers, kernel, N), reused across cells.

    `bench_vs_celerite2.py` forces the fork start method for the same
    reason: under spawn (macOS's default since 3.8) every worker re-imports
    celerite2 and numpy, which shows up as celerite2 being slow rather than
    as process startup.  Building a fresh pool per table cell put that cost
    in fifteen separate measurements.
    """
    key = (n_proc, kind, len(t))
    if key not in _POOL:
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        _POOL[key] = mp.Pool(n_proc, initializer=_init,
                             initargs=(t, y, yerr, kind))
    return _POOL[key]


def _init(t, y, yerr, kind):
    _DATA.update(t=t, y=y, yerr=yerr, kind=kind)


def _c2_term(kind, p):
    from celerite2 import terms as c2t
    if kind == "real":
        # anvilgp RealTerm is k(tau) = sigma^2 exp(-c tau), so the
        # second parameter is the rate itself, not a timescale.
        return c2t.RealTerm(a=p[0] ** 2, c=p[1])
    if kind == "sho":
        return c2t.SHOTerm(sigma=p[0], rho=p[1], Q=p[2])
    if kind == "matern32":
        return c2t.Matern32Term(sigma=p[0], rho=p[1])
    if kind == "rotation":
        return c2t.RotationTerm(sigma=p[0], period=p[1], Q0=p[2],
                                dQ=p[3], f=p[4])
    if kind == "sho2":
        return (c2t.SHOTerm(sigma=p[0], rho=p[1], Q=p[2])
                + c2t.SHOTerm(sigma=p[3], rho=p[4], Q=p[5]))
    raise KeyError(kind)


def _loglike_chunk(params):
    import celerite2
    t, y, yerr, kind = (_DATA["t"], _DATA["y"], _DATA["yerr"], _DATA["kind"])
    out = np.empty(len(params))
    for i, p in enumerate(params):
        gp = celerite2.GaussianProcess(_c2_term(kind, p), mean=0.0)
        gp.compute(t, yerr=yerr)
        out[i] = gp.log_likelihood(y)
    return out


# kind -> (anvilgp term factory, natural params, m, celerite2 support)
KERNELS = {
    "real":     (lambda ag: ag.RealTerm(),     [1.2, 3.0],                1, True),
    "sho":      (lambda ag: ag.SHOTerm(),      [1.3, 4.0, 3.0],           2, True),
    "matern32": (lambda ag: ag.Matern32Term(), [1.1, 2.5],                2, True),
    "matern52": (lambda ag: ag.Matern52Term(), [1.1, 2.5],                3, False),
    "rotation": (lambda ag: ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4], 4, True),
    "sho2":     (lambda ag: ag.SHOTerm() + ag.SHOTerm(),
                 [1.3, 4.0, 3.0, 0.6, 0.8, 1.5],                          4, True),
}


def make_data(n, seed=7, kind=None):
    """Timestamps, data and errors.  A regular grid with one 20-day gap,
    which is what photometry looks like -- and *not* the bare regular grid
    the README's milestone-3 figures use, which is why the ratios here are
    lower.  Measured in one window, the gap costs the kernel 1.7x at
    N = 1e4.

    With ``kind``, the data is drawn **on model** from that kernel.  For a
    precision test that is the honest input: off-model residuals are large
    and the quadratic form dominates, while on model the log-determinant
    and the quadratic form very nearly cancel, and what survives the
    cancellation is what a sampler actually sees (see `anvilgp/oracle.py`).
    Timing does not care, so it keeps the cheaper white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.02
    t[n // 2:] += 20.0
    yerr = np.full(n, 0.3)
    if kind is None:
        return t, rng.standard_normal(n), yerr
    import anvilgp as ag
    from anvilgp.oracle import sample_ssm
    factory, pars, _, _ = KERNELS[kind]
    y = sample_ssm(factory(ag), pars, 0.02, n, rng) + rng.standard_normal(n) * yerr
    return t, y, yerr


def time_anvilgp(kind, n, n_chains, metal, reps=5):
    import mlx.core as mx
    import anvilgp as ag
    from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap
    factory, pars, m, _ = KERNELS[kind]
    t, y, yerr = make_data(n)
    gp = GPLogLike(None, t, None, y, yerr, factory(ag), ParamMap.log(pars),
                   GPPolicy(block=4096, hi_block=4096, metal=metal))
    if metal and not gp.metal_eligible()[0]:
        return None
    u = mx.zeros((n_chains, len(pars)), dtype=mx.float32)
    with mx.stream(mx.gpu):
        mx.eval(gp(u))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mx.eval(gp(u))
            ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def time_celerite2(kind, n, n_chains, n_proc, reps=3, probe=8):
    """Per-vector cost x n_chains.  celerite2 is linear in the vector count
    and embarrassingly parallel, so timing a probe and scaling is both
    faster and kinder than running 1024 of them."""
    _, pars, _, ok = KERNELS[kind]
    if not ok:
        return None
    t, y, yerr = make_data(n)
    params = np.tile(np.array(pars), (probe, 1))
    if n_proc <= 1:
        _init(t, y, yerr, kind)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _loglike_chunk(params)
            ts.append((time.perf_counter() - t0) / probe)
        return min(ts) * n_chains * 1e3
    # Only split across as many workers as there is work for: with
    # n_chains = 1 and n_proc = 12, array_split yields one real chunk and
    # eleven empty ones, and the number that comes back is pool dispatch
    # overhead rather than celerite2 (which made the published 1-vector
    # "12 cores" cells, and their ratios, meaningless).
    n_use = max(1, min(n_proc, n_chains))
    chunks = [c for c in np.array_split(np.tile(np.array(pars), (n_chains, 1)),
                                        n_use) if len(c)]
    pool = _get_pool(n_use, kind, t, y, yerr)
    pool.map(_loglike_chunk, [c[:1] for c in chunks])          # warm up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pool.map(_loglike_chunk, chunks)
        ts.append(time.perf_counter() - t0)
    # The minimum, not the median `bench_vs_celerite2.py` reports: on this
    # machine background load only ever *adds* time, so the minimum is the
    # least contaminated estimate (see the benchmarking notes). The two
    # scripts therefore differ slightly even on identical inputs.
    return min(ts) * 1e3


def accuracy(kind, n):
    """|log L - float64 dense Cholesky| for both libraries, same data."""
    import mlx.core as mx
    import anvilgp as ag
    from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap
    from anvilgp.oracle import dense_loglike
    factory, pars, m, ok = KERNELS[kind]
    t, y, yerr = make_data(n, kind=kind)          # on model: see make_data
    term = factory(ag)
    ref = dense_loglike(term, np.array(pars), t, y, yerr)
    out = {"|logL|": abs(ref)}
    for tag, metal in (("mlx", False), ("metal", True)):
        gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(pars),
                       GPPolicy(block=4096, hi_block=4096, metal=metal))
        if metal and not gp.metal_eligible()[0]:
            continue
        u = mx.zeros((1, len(pars)), dtype=mx.float32)
        with mx.stream(mx.gpu):
            v = float(np.array(gp.log_prob_absolute(u))[0])
        out[tag] = abs(v - ref)
    if ok:
        import celerite2
        gp = celerite2.GaussianProcess(_c2_term(kind, pars), mean=0.0)
        gp.compute(t, yerr=yerr)
        out["celerite2"] = abs(gp.log_likelihood(y) - ref)
    return out


def _ratio(num, den):
    if not num or not den:
        return "—"
    r = num / den
    return f"**{r:.0f}x**" if r >= 10 else (f"{r:.1f}x" if r >= 1
                                            else f"{r:.2f}x")


def fmt(x, unit="ms"):
    if x is None:
        return "—"
    if unit == "ms":
        return f"{x:.2f}" if x < 10 else (f"{x:.0f}" if x < 10_000
                                          else f"{x/1000:.1f} s")
    return f"{x:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    NP = a.n_proc
    print(f"<!-- generated by benchmarks/bench_docs_tables.py, "
          f"{NP} CPU cores -->\n")

    # ---- table 1: the ensemble axis --------------------------------------
    print("### Speed against celerite2, by ensemble size\n")
    print(f"SHO kernel (3 hyperparameters), one log-likelihood per "
          f"parameter vector, milliseconds for the whole batch. celerite2 "
          f"columns are per-vector cost x the batch, single core and a "
          f"{NP}-core pool. anvil-gp runs at `block=4096`, not the shipped "
          f"default of 1024; `block` affects only the MLX column.\n")
    print("| N | vectors | anvil-gp kernel | anvil-gp MLX | celerite2 1 core |"
          f" celerite2 {NP} cores | kernel vs 1 core | kernel vs pool |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    sizes = [(1_000, [1, 64, 1024]), (10_000, [1, 64, 1024]),
             (100_000, [1, 64, 1024])]
    kernels = list(KERNELS)
    if a.quick:                       # smoke test: every code path, once
        sizes = [(10_000, [1, 1024])]
        kernels = ["sho", "matern52"]
    for n, chains in sizes:
        for c in chains:
            met = time_anvilgp("sho", n, c, True)
            mlx_ = time_anvilgp("sho", n, c, False)
            one = time_celerite2("sho", n, c, 1)
            pool = time_celerite2("sho", n, c, NP)
            r1 = _ratio(one, met)
            rp = _ratio(pool, met)
            print(f"| {n:,} | {c} | **{fmt(met)}** | {fmt(mlx_)} | {fmt(one)} "
                  f"| {fmt(pool)} | {r1} | {rp} |")

    # ---- table 2: by kernel ----------------------------------------------
    print("\n### Speed by kernel\n")
    print("N = 10,000, 1024 parameter vectors. `m` is the state dimension: "
          "the kernel holds `3m^2+2m+1` floats per thread, so cost grows "
          "steeply with it.\n")
    print(f"| kernel | m | anvil-gp kernel | anvil-gp MLX | celerite2 1 core "
          f"| celerite2 {NP} cores | kernel vs pool |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for kind in kernels:
        m = KERNELS[kind][2]
        met = time_anvilgp(kind, 10_000, 1024, True)
        mlx_ = time_anvilgp(kind, 10_000, 1024, False)
        one = time_celerite2(kind, 10_000, 1024, 1)
        pool = time_celerite2(kind, 10_000, 1024, NP)
        rp = _ratio(pool, met)
        print(f"| `{kind}` | {m} | **{fmt(met)}** | {fmt(mlx_)} | {fmt(one)} "
              f"| {fmt(pool)} | {rp} |")

    # ---- table 3: accuracy -----------------------------------------------
    print("\n### Accuracy against a float64 dense Cholesky\n")
    print("Absolute error in log L at N = 2,000, where an exact dense "
          "factorisation is affordable. anvil-gp's production path is "
          "float32; celerite2 is float64 throughout.\n")
    print("| kernel | m | \\|log L\\| | anvil-gp kernel | anvil-gp MLX "
          "| celerite2 |")
    print("|---|---:|---:|---:|---:|---:|")
    for kind in kernels:
        m = KERNELS[kind][2]
        acc = accuracy(kind, 2_000)
        print(f"| `{kind}` | {m} | {acc['|logL|']:.0f} "
              f"| {fmt(acc.get('metal'), 'e')} | {fmt(acc.get('mlx'), 'e')} "
              f"| {fmt(acc.get('celerite2'), 'e')} |")


if __name__ == "__main__":
    try:
        main()
    finally:
        for _p in _POOL.values():
            _p.terminate()
