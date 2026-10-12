"""The float32 feasibility map behind ``GPPolicy.kappa_max``.

Three measurements, all SHO, error = |float32 - float64 ``.hi``| in
log-units, a failed evaluation counted as infinite (docs/float32.md,
section 6):

1. fixed data, parameters anywhere -- where a wandering chain goes;
2. the same, at N = 2e3, 2e4 and 1e5 -- does the boundary move with N;
3. data drawn at each parameter setting -- where a posterior lives.

Every evaluation runs with ``kappa_max=None``: this measures what the guard
is for, so the guard must not hide it.  Accuracy, not timing: a busy GPU
does not change the numbers.

    python benchmarks/kappa_map.py
"""

import itertools

import mlx.core as mx
import numpy as np

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
from anvilgp.oracle import sample_ssm

YE, DT = 0.15, 0.02
PM = ParamMap(kernel=[KernelParam(link="log", index=i) for i in range(3)],
              n_dim=4, jitter=[KernelParam(link="log", index=3)])


def to_v(grid):
    """(sigma/yerr, rho/dt, Q, jitter/yerr) -> log natural units."""
    return np.log(grid * np.array([YE, DT, 1.0, YE]))


def tutorial_data(n):
    rng = np.random.default_rng(42)
    y = (sample_ssm(ag.SHOTerm(), [1.2, 5.0, 4.0], DT, n, rng)
         + rng.standard_normal(n) * YE)
    return np.arange(n) * DT, y


def errors(t, y, grid):
    """kappa and the error of both float32 paths at every grid row."""
    v = to_v(grid)
    out = {}
    for name, metal in (("kernel", True), ("MLX", False)):
        gp = GPLogLike(None, t, None, y, np.full(len(t), YE), ag.SHOTerm(),
                       PM, GPPolicy(metal=metal, kappa_max=None))
        with mx.stream(mx.gpu):
            out[name] = np.array(gp(mx.array(v.astype(np.float32))),
                                 np.float64)
    with mx.stream(mx.cpu):
        hi = np.array(gp.hi(mx.array(v)), np.float64)
    kappa = np.array(gp.kappa(mx.array(v.astype(np.float32))), np.float64)
    err = {k: np.where(x < -1e29, np.inf, np.abs(x - hi))
           for k, x in out.items()}
    return kappa, err


def binned(kappa, err, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (kappa >= lo) & (kappa < hi)
        if m.any():
            yield lo, hi, m


def part1():
    grid = np.array(list(itertools.product(
        10.0 ** np.arange(-2, 15), [2, 10, 50, 250, 1000.],
        [0.05, 0.5, 3, 30, 500.], [1e-3, 1e-1, 1, 10, 1e3])))
    kappa, err = errors(*tutorial_data(2000), grid)
    print(f"\n1. fixed tutorial data, N = 2000, {len(grid)} points")
    print(f"{'kappa':>17} {'n':>5} {'kernel max':>11} {'MLX max':>9}")
    for lo, hi, m in binned(kappa, err, 10.0 ** np.arange(-10, 11.0)):
        print(f"{lo:7.0e} - {hi:7.0e} {m.sum():5d} "
              f"{np.max(err['kernel'][m]):11.3g} {np.max(err['MLX'][m]):9.3g}")


def part2():
    grid = np.array(list(itertools.product(
        10.0 ** np.arange(1, 8.01, 0.25), [50., 250., 1000.],
        [0.5, 3., 30., 500.], [1e-3])))
    res = {n: errors(*tutorial_data(n), grid) for n in (2000, 20000, 100000)}
    kappa = res[2000][0]
    print("\n2. the same data at three lengths: max error per kappa bin")
    print(f"{'kappa':>17} {'n':>4} " + " ".join(
        f"{'kern ' + str(n):>11} {'MLX ' + str(n):>11}" for n in res))
    for lo, hi, m in binned(kappa, None, 10.0 ** np.arange(4, 9.6, 0.5)):
        print(f"{lo:7.1e} - {hi:7.1e} {m.sum():4d} " + " ".join(
            f"{np.max(e['kernel'][m]):11.3g} {np.max(e['MLX'][m]):11.3g}"
            for _, e in res.values()))


def part3():
    rows = []
    for sy, rd, q in itertools.product([10., 1e2, 1e3, 1e4, 1e5],
                                       [50., 250., 1000.], [0.5, 3., 30.]):
        rng = np.random.default_rng(int(sy + rd + 10 * q))
        n = 4000
        t = np.arange(n) * DT
        y = (sample_ssm(ag.SHOTerm(), [sy * YE, rd * DT, q], DT, n, rng)
             + rng.standard_normal(n) * YE)
        kappa, err = errors(t, y, np.array([[sy, rd, q, 1e-3]]))
        rows.append((kappa[0], sy, rd, q, err["kernel"][0], err["MLX"][0]))
    print("\n3. data drawn at each setting, N = 4000, one draw each")
    print(f"{'kappa':>9} {'sig/yerr':>8} {'rho/dt':>6} {'Q':>4} "
          f"{'kernel':>9} {'MLX':>9}")
    for r in sorted(rows):
        print(f"{r[0]:9.2e} {r[1]:8g} {r[2]:6g} {r[3]:4g} "
              f"{r[4]:9.3g} {r[5]:9.3g}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
