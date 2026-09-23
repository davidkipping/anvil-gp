"""Does the likelihood point at the truth on average?

One injection-recovery at one truth has almost no power against a subtle
likelihood bias -- a wrong normalisation, a mis-scaled process noise, an
off-by-one in the recursion can all leave a single fit looking fine, because
a single realisation's own scatter is larger than the bias.  These tests
average over realisations instead.

This is the cheap, fast version of the M5 calibration work (SBC rank checks
at N = 1e3 through the float64 reference path).  It is not a substitute for
it: unbiasedness of a point estimate is a weaker claim than correct posterior
coverage.  It is, however, enough to catch the class of bug that a single
recovery would miss, and it runs in seconds.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap, a64
from anvilgp.oracle import dense_covariance


N = 800
BASELINE = 40.0
SIGMA, RHO, QUAL, JIT, YERR = 1.5, 7.3, 8.0, 0.12, 0.20


def _profile_max(gp_factory, grid, n_real=12, seed0=1000):
    """MAP of one parameter over ``n_real`` fresh realisations of the data."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, BASELINE, N))
    yerr = np.full(N, YERR)
    cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                           yerr, jitter=JIT)
    chol = np.linalg.cholesky(cov)
    best = []
    for k in range(n_real):
        y = chol @ np.random.default_rng(seed0 + k).standard_normal(N)
        ll = np.array([float(np.array(
            gp_factory(t, y, yerr, g).hi_absolute(a64(np.zeros((1, 1)))))[0])
            for g in grid])
        best.append(grid[int(np.argmax(ll))])
    return np.array(best)


def _fixed(t, y, yerr, kp, jitter=JIT):
    pm = ParamMap(kernel=[KernelParam(link="const", value=float(v))
                          for v in kp], n_dim=1,
                  jitter=[KernelParam(link="const", value=float(jitter))])
    return GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                     GPPolicy(block=1024, hi_block=1024))


@pytest.mark.slow
def test_amplitude_is_recovered_without_bias():
    grid = np.linspace(0.8, 2.4, 17)
    best = _profile_max(lambda t, y, e, s: _fixed(t, y, e, [s, RHO, QUAL]),
                        grid)
    # the grid step is 0.1, so the mean cannot be resolved below ~0.03
    assert abs(best.mean() - SIGMA) < 0.15, best
    assert best.std() < 0.5, best


@pytest.mark.slow
def test_timescale_is_recovered_without_bias():
    grid = np.exp(np.linspace(math.log(3.0), math.log(18.0), 17))
    best = _profile_max(lambda t, y, e, r: _fixed(t, y, e, [SIGMA, r, QUAL]),
                        grid)
    assert abs(np.log(best.mean() / RHO)) < 0.25, best


@pytest.mark.slow
def test_jitter_is_recovered_without_bias():
    grid = np.exp(np.linspace(math.log(0.03), math.log(0.5), 17))
    best = _profile_max(
        lambda t, y, e, j: _fixed(t, y, e, [SIGMA, RHO, QUAL], jitter=j), grid)
    assert abs(np.log(best.mean() / JIT)) < 0.35, best


def test_a_single_realisation_is_not_evidence():
    """The point of the tests above, made explicit.

    The realised RMS of one draw scatters widely around the process sigma
    when the baseline holds only a handful of correlation times, so a single
    recovery landing 2-3 sigma from the truth is expected behaviour and says
    nothing about the likelihood.
    """
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, BASELINE, N))
    yerr = np.full(N, YERR)
    cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                           yerr, jitter=JIT)
    chol = np.linalg.cholesky(cov)
    rms = np.array([(chol @ np.random.default_rng(2000 + k)
                     .standard_normal(N)).std() for k in range(24)])
    expected = math.sqrt(SIGMA ** 2 + JIT ** 2 + YERR ** 2)
    assert 0.15 < rms.std() / expected < 0.8, rms.std() / expected
    assert abs(rms.mean() / expected - 1.0) < 0.25


@pytest.mark.slow
def test_chains_started_in_the_infeasible_region_recover():
    """The brief's requirement: warmup WILL visit float32-infeasible corners.

    Landing there must be survivable, not terminal.  Two things have to
    hold.  The GP log-likelihood itself must come back large-negative and
    **finite** -- that is anvil-gp's contract, and a NaN there would poison
    the whole ensemble's adaptation.  And a sampler started there must walk
    back out.

    (The *target's* log-probability may still be -inf at such a point,
    because anvil's Transform contributes a -inf Jacobian at the edge of a
    bounded prior.  That is a correct statement about the prior, not a
    failure of the likelihood, and a sampler rejects it cleanly.)

    Uses the stretch move, which needs no gradient and no
    ``mx.disable_compile()``.  Half the ensemble starts in the failure region so the rest
    has somewhere to pull it back from.
    """
    anvil = pytest.importorskip("anvil")
    from anvilgp.anvil import make_target

    rng = np.random.default_rng(5)
    n = 600
    t = np.sort(rng.uniform(0.0, BASELINE, n))
    yerr = np.full(n, YERR)
    cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                           yerr, jitter=JIT)
    y = np.linalg.cholesky(cov) @ rng.standard_normal(n)

    tgt = make_target(ag.SHOTerm(), t, y, yerr)
    dim = tgt.transform.dim

    n_chains, half = 64, 32
    u0 = rng.standard_normal((n_chains, dim)).astype(np.float32) * 0.3
    u0[:half] = 30.0                     # deep in the infeasible corner
    u = mx.array(u0)

    with mx.stream(mx.gpu):
        ll = np.array(tgt.loglike(tgt.transform.to_model(u)))
        assert np.all(np.isfinite(ll)), "the likelihood must never be NaN"
        assert np.all(ll[:half] <= -1e29), "the dead half should be rejected"
        assert np.all(ll[half:] > -1e29), "the live half should not be"

        from anvil.kernels.ensemble import EnsembleKernel
        res = anvil.run(EnsembleKernel(tgt.target, seed=2), tgt.target, u,
                        n_warmup=400, n_samples=100, seed=3)
        last = res.final_state["u"]
        final = np.array(tgt.loglike(tgt.transform.to_model(last)))

    assert np.all(np.isfinite(final))
    assert np.all(final > -1e29), (
        f"{int((final <= -1e29).sum())} chains never left the dead region")


@pytest.mark.slow
def test_a_stuck_chain_does_not_poison_the_ensemble_under_chees():
    """What anvil-gp owes a gradient sampler deep in the prior tail.

    The contract is not that every chain recovers -- that is the sampler's
    business, and ChEES cannot do it here for a reason that belongs to
    anvil's transform (see docs/anvil-chees-boundary.md).  The contract is
    that a chain out there returns a finite value and a finite gradient, so
    the *rest* of the ensemble keeps sampling.  A NaN would take all of it.

    Half the ensemble starts at u = 15, which is ~15 sigma into the prior:
    log-probability around -9000 against -800 for a healthy chain, and a
    transform Jacobian gradient of ~6e8 against ~5e2.
    """
    anvil = pytest.importorskip("anvil")
    from anvilgp.anvil import make_target

    rng = np.random.default_rng(5)
    n = 400
    t = np.sort(rng.uniform(0.0, BASELINE, n))
    yerr = np.full(n, YERR)
    cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                           yerr, jitter=JIT)
    y = np.linalg.cholesky(cov) @ rng.standard_normal(n)
    tgt = make_target(ag.SHOTerm(), t, y, yerr)
    dim = tgt.transform.dim

    # value and gradient stay finite right up to where the transform itself
    # saturates (float32 sigmoid hits exactly 1 near u = 18)
    with mx.stream(mx.gpu):
        for val in (5.0, 10.0, 15.0, 17.0):
            u = mx.array(np.full((2, dim), val, dtype=np.float32))
            lp = tgt.target.log_prob(u)
            _, g = mx.vjp(tgt.target.log_prob, [u],
                          [mx.ones((2,), dtype=mx.float32)])
            mx.eval(lp, g)
            assert bool(mx.all(mx.isfinite(lp))), val
            assert bool(mx.all(mx.isfinite(g[0]))), val

    n_chains, half = 16, 8
    u0 = rng.standard_normal((n_chains, dim)).astype(np.float32) * 0.3
    u0[:half] = 15.0

    mx.disable_compile()                 # required for ChEES on a GP target
    try:
        with mx.stream(mx.gpu):
            res = anvil.run(anvil.ChEESHMC(tgt.target, max_leapfrog=16),
                            tgt.target, mx.array(u0), n_warmup=60,
                            n_samples=20, seed=3, progress=False)
            last = np.array(res.final_state["u"])
            final = np.array(tgt.target.log_prob(mx.array(last)))
    finally:
        mx.enable_compile()

    assert np.all(np.isfinite(final)), "no chain may go non-finite"
    moved = np.abs(last - u0).max(axis=1)
    assert np.all(moved[half:] > 1e-3), (
        "the healthy half must keep sampling regardless of the stuck half")


@pytest.mark.slow
def test_chees_runs_on_the_fused_kernel():
    """M4's point: a gradient sampler can use the kernel -- and gets the
    posterior right.

    An earlier version asserted only that the chains were finite and had
    moved, which a zero gradient also satisfies (HMC degenerates to a
    random walk).  This one checks two things a wrong gradient fails.  The
    mean acceptance must be high: with a wrong gradient the leapfrog energy
    error is large and acceptance collapses.  And the jitter's posterior
    width must match the float64 Laplace width at the mode, 0.0107 on this
    problem -- which ChEES reproduces to 3% and which a random walk cannot
    find in the budget.  (The stretch move gives 0.046 here: two of its
    1024 chains trapped in a secondary mode; see docs/anvil-stretch-width.md.)

    `mx.disable_compile()` is still needed, for both paths: the over-fused
    group is in `terms._sho_matrices`, which the kernel does not replace.
    """
    anvil = pytest.importorskip("anvil")
    from anvilgp import GPPolicy
    from anvilgp.anvil import make_target

    rng = np.random.default_rng(20260917)
    n = 1000
    t = np.arange(n) * (BASELINE / n)
    yerr = np.full(n, YERR)
    cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                           yerr, jitter=JIT)
    y = np.linalg.cholesky(cov) @ rng.standard_normal(n)

    gt = make_target(ag.SHOTerm(), t, y, yerr,
                     policy=GPPolicy(block=1024, sub_block=16, metal=True))
    assert gt.target.supports_grad, "the kernel target must accept gradients"

    mx.disable_compile()
    try:
        with mx.stream(mx.gpu):
            res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=32),
                            gt.target, gt.u0(256, seed=1, spread=0.3),
                            n_warmup=300, n_samples=300, seed=3,
                            progress=False)
    finally:
        mx.enable_compile()

    acc = float(np.mean(res.accept_fraction))
    assert acc > 0.5, f"acceptance {acc:.2f}: the gradient is not right"
    flat = gt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    jit_sd = float(gt.natural(flat)["jitter"].std())
    assert 0.007 < jit_sd < 0.016, (
        f"jitter posterior width {jit_sd:.4f}; float64 Laplace gives 0.0107")
