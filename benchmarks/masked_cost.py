"""Cost of held-out points (`GPLogLike(held_out=...)`) against the plain
likelihood, at N = 2818 one-minute points with 14% held out: the fused
kernel and the MLX path, value and value + gradient, 128 and 1024 chains,
plain (Matern-3/2 + jitter) and with an error-scale noise model and a
per-chain quadratic baseline marginalised; one, two and four windows
holding out the same 398 points.

    .venv/bin/python benchmarks/masked_cost.py

Prints the verify_readme stability probe first and last: numbers taken
above 1.5x its quiet level are contention, not the library
(docs/benchmarks.md).  Each figure is the minimum over 3 rounds of 20
calls after warm-up, compiled.  The MLX path with a basis at 1024 chains
takes tens of seconds a call.
"""
import sys, time, warnings, numpy as np, mlx.core as mx, anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap
from anvilgp.anvil import make_target
import anvilgp.noise as noise
warnings.simplefilter("ignore")

def timed(fn, reps=20, rounds=3):
    for _ in range(3):
        mx.eval(fn())
    best = []
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e3)
        best.append(min(ts))
    return min(best)

def probe():
    rng = np.random.default_rng(0); n = 10_000
    t = np.arange(n) * (30.0 / n)
    gp = GPLogLike(None, t, None, rng.standard_normal(n) * 0.9, np.full(n, 0.3), ag.SHOTerm(),
                   ParamMap.log([1.0, 4.0, 3.0]), GPPolicy(block=4096, metal=False))
    u = mx.array(rng.standard_normal((1024, 3)).astype(np.float32) * 0.01)
    f = mx.compile(gp.__call__)
    with mx.stream(mx.gpu):
        return min(timed(lambda: f(u), 10, 1) for _ in range(3))

N = 2818
t = np.arange(N) / 1440.0                       # days, one-minute cadence
tc = t.mean()
rng = np.random.default_rng(5)
yerr = np.full(N, 3e-4)
y = 1.0 + 3e-4 * rng.standard_normal(N) + 2e-4 * np.sin(2 * np.pi * t / 0.4)

def windows(k, frac=398 / 2818):
    """k windows of equal length, total ~frac of the data, spread over the middle."""
    w = frac * (t[-1] - t[0]) / k
    centres = tc + (np.arange(k) - (k - 1) / 2) * 3 * w
    return [(c - w / 2, c + w / 2) for c in centres]

def per_chain_basis(vm, x, m0):
    xc = (x - tc) / 0.5
    return mx.stack([mx.ones_like(m0), mx.broadcast_to(xc, m0.shape),
                     mx.broadcast_to(xc * xc, m0.shape)], axis=-1)

def target(cfg, held, metal):
    pol = GPPolicy(metal=metal)
    kw = dict(policy=pol, held_out=held)
    if cfg == "basis":
        kw.update(noise=noise.error_scale(), linear_basis=per_chain_basis)
    return make_target(ag.Matern32Term(), t, y, yerr, **kw)

def bench(gt, n_chains, grad):
    u = gt.u0(n_chains, seed=0, spread=0.1)
    v = mx.array(np.array(gt.transform.to_model(u)), dtype=mx.float32)
    ll = gt.loglike
    if grad:
        f = mx.compile(mx.value_and_grad(lambda w: ll(w).sum()))
    else:
        f = mx.compile(ll.__call__)
    with mx.stream(mx.gpu):
        return timed(lambda: f(v))

out = {"probe_before": probe()}
print(f"probe before: {out['probe_before']:.1f} ms (quiet 73.3)", flush=True)
rows = []
for cfg in ["plain", "basis"]:
    for n_chains in [128, 1024]:
        for grad in [False, True]:
            res = {}
            for lab, held, metal in [("A kernel unmasked", None, None),
                                     ("B MLX unmasked", None, False),
                                     ("C kernel masked 2w", windows(2), None),
                                     ("C' MLX masked 2w", windows(2), False),
                                     ("D kernel masked 1w", windows(1), None),
                                     ("D kernel masked 4w", windows(4), None)]:
                gt = target(cfg, held, metal)
                ok = gt.loglike._metal_ok
                ms = bench(gt, n_chains, grad)
                res[lab] = ms
                print(f"{cfg:5s} chains={n_chains:4d} {'v+g' if grad else 'val'} {lab:20s} metal={ok!s:5s} {ms:8.2f} ms", flush=True)
            a = res["A kernel unmasked"]
            print(f"   -> masked/unmasked kernel: 1w {res['D kernel masked 1w']/a:.2f}x  2w {res['C kernel masked 2w']/a:.2f}x  4w {res['D kernel masked 4w']/a:.2f}x ; MLX masked/kernel masked {res[chr(67)+chr(39)+' MLX masked 2w']/res['C kernel masked 2w']:.1f}x", flush=True)
out["probe_after"] = probe()
print(f"probe after: {out['probe_after']:.1f} ms (quiet 73.3)", flush=True)
