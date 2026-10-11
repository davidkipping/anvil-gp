"""The observation, as seen: ChEES chains started in anvil-gp's failure region.

Needs anvil-gp (it uses anvil-gp's likelihood and target). Half of 64 ChEES
chains start at u = 30 in every coordinate, inside the region where the GP
likelihood cannot be evaluated in float32. Run with anvil-gp's two failure
values: the old constant -1e30 and the current sloped one.

    python anvilgp_chees_stuck_chains.py

This reproduces the observation only. It deliberately does not do the
triage proposed in docs/anvil-chees-stuck-chains.md.
"""
import mlx.core as mx
import numpy as np

import anvil
import anvilgp as ag
from anvilgp.anvil import make_target
from anvilgp.likelihood import GPLogLike
from anvilgp.oracle import dense_covariance

SIGMA, RHO, QUAL, JIT, YERR, BASE = 1.5, 7.3, 8.0, 0.12, 0.20, 40.0
_sloped = GPLogLike._infeasible_pull


def run(constant: bool):
    # pull = None is exactly anvil-gp's previous constant failure value
    GPLogLike._infeasible_pull = ((lambda self, v: None) if constant
                                  else _sloped)
    try:
        rng = np.random.default_rng(5)
        n = 600
        t = np.sort(rng.uniform(0.0, BASE, n))
        yerr = np.full(n, YERR)
        cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                               yerr, jitter=JIT)
        y = np.linalg.cholesky(cov) @ rng.standard_normal(n)
        gt = make_target(ag.SHOTerm(), t, y, yerr)
        dim = gt.transform.dim
        u0 = rng.standard_normal((64, dim)).astype(np.float32) * 0.3
        u0[:32] = 30.0                       # inside the failure region
        with mx.stream(mx.gpu):
            res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=16),
                            gt.target, mx.array(u0), n_warmup=300,
                            n_samples=50, seed=3, progress=False)
            last = np.array(res.final_state["u"])
            ll = np.array(gt.loglike(gt.transform.to_model(mx.array(last))))
    finally:
        GPLogLike._infeasible_pull = _sloped
    stuck = np.where(ll[:32] <= -1e29)[0]
    far = np.abs(last[stuck].astype(np.float64)).max() if stuck.size else 0.0
    print(f"{'constant' if constant else 'sloped  '} failure value: "
          f"{stuck.size}/32 dead-started chains still dead {stuck.tolist()}, "
          f"max |u| {far:.3g}; healthy half all live "
          f"{bool(np.all(ll[32:] > -1e29))}; "
          f"n_divergent {res.extras.get('n_divergent')}; "
          f"warmup_report {anvil.warmup_report(res).verdict}")


print(f"anvil {anvil.__version__}")
run(constant=True)
run(constant=False)
