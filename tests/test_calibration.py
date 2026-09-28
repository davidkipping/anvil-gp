"""Does the likelihood point at the truth on average?

One injection-recovery at one truth has almost no power against a subtle
likelihood bias -- a wrong normalisation, a mis-scaled process noise, an
off-by-one in the recursion can all leave a single fit looking fine, because
a single realisation's own scatter is larger than the bias.  These tests
average over realisations instead.

Two layers, and the difference between them matters.  The tests in the first
half take the profile *argmax* over realisations: unbiasedness of a point
estimate, cheap, and blind to the posterior's width -- a log-likelihood off by
a constant factor moves the mode not at all.  The SBC rank checks at the end
close that gap, and carry their own power test showing they do.
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

    Uses the stretch move, which needs no gradient.  Half the ensemble
    starts in the failure region so the rest has somewhere to pull it back
    from.
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

    # No mx.disable_compile() since 2026-09-27: anvil compiles its leapfrog
    # step and the backward pass now survives it (docs/float32.md).
    with mx.stream(mx.gpu):
        res = anvil.run(anvil.ChEESHMC(tgt.target, max_leapfrog=16),
                        tgt.target, mx.array(u0), n_warmup=60,
                        n_samples=20, seed=3, progress=False)
        last = np.array(res.final_state["u"])
        final = np.array(tgt.target.log_prob(mx.array(last)))

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

    Runs with compilation on: the fused-argument limit that used to force
    `mx.disable_compile()` was one function in `special.py` and is fixed.
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

    with mx.stream(mx.gpu):
        res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=32),
                        gt.target, gt.u0(256, seed=1, spread=0.3),
                        n_warmup=300, n_samples=300, seed=3,
                        progress=False)

    acc = float(np.mean(res.accept_fraction))
    assert acc > 0.5, f"acceptance {acc:.2f}: the gradient is not right"
    flat = gt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    jit_sd = float(gt.natural(flat)["jitter"].std())
    assert 0.007 < jit_sd < 0.016, (
        f"jitter posterior width {jit_sd:.4f}; float64 Laplace gives 0.0107")


# ---------------------------------------------------------------------------
# SBC rank checks (M5's owed calibration work)
# ---------------------------------------------------------------------------
#
# The tests above profile one parameter and keep the *argmax*.  That tests
# unbiasedness of a point estimate and is blind to the posterior's width: a
# likelihood off by a constant factor in its log -- a tempered reduction, a
# double-counted log-determinant, a mis-scaled process noise -- moves the mode
# not at all and the width a lot.  These tests look at the whole shape.
#
# The construction is simulation-based calibration in its conditional form.
# Draw `theta* ~ prior`, generate `y ~ p(y | theta*)`, then rank `theta_p*`
# inside the posterior `p(theta_p | y, theta_{-p} = theta*_{-p})`.  The other
# parameters being *known* makes that a one-dimensional posterior, which can
# be had exactly by quadrature -- so there is no MCMC in the loop and no
# sampler noise in the verdict.  Under a correct likelihood the probability
# integral transform `u = P(theta_p <= theta_p* | y)` is exactly Uniform(0, 1),
# and that is what is tested.
#
# Two things this deliberately does *not* do.  It does not re-derive the
# priors per simulation: `make_target` sets its references and bounds from the
# data (`priors.data_references`), so the prior moves with `y` and the joint
# `p(theta, y)` that SBC needs would not exist.  The hyperparameters are
# pinned once, from one reference dataset, and reused.  And it does not test
# anvil's sampler -- that is anvil's to test; this tests the likelihood
# anvil-gp owns.


def _sbc_hypers(t, y, yerr):
    """The pinned prior: derived once, then frozen across simulations."""
    from anvilgp.priors import default_hypers, jitter_hyper
    return default_hypers(ag.SHOTerm(), t, y, yerr), jitter_hyper(yerr)


def _support(h, n_sd=6.0):
    """The interval the 1-D quadrature runs over: the prior's support."""
    if h.prior is None:
        assert np.isfinite(h.lo) and np.isfinite(h.hi), h.name
        return h.lo, h.hi
    _, mu, sd = h.prior
    return max(h.lo, mu - n_sd * sd), min(h.hi, mu + n_sd * sd)


def _prior_logpdf(h, v):
    """Model-space log prior, up to a constant, on the support.

    Bounded-with-no-prior is uniform -- anvil's ``Transform`` supplies that
    density and its Jacobian exactly, so in model space it is flat.  A
    declared normal prior is a normal *truncated* to the bounds, which the
    quadrature range already imposes.
    """
    if h.prior is None:
        return np.zeros_like(v)
    _, mu, sd = h.prior
    return -0.5 * ((v - mu) / sd) ** 2


def _prior_draw(h, rng):
    if h.prior is None:
        return float(rng.uniform(h.lo, h.hi))
    _, mu, sd = h.prior
    for _ in range(1000):
        x = float(rng.normal(mu, sd))
        if h.lo <= x <= h.hi:
            return x
    raise RuntimeError(f"cannot draw {h.name} inside its bounds")


def _natural(hypers, v):
    """Model space -> natural units, the ``log`` link's ``ref * exp(v)``."""
    return np.array([h.ref * math.exp(vi) for h, vi in zip(hypers, v)])


def _conditional_cdf(gp, v_star, p, h, temper, n_coarse=97, n_fine=193):
    """``P(theta_p <= theta_p* | y, theta_{-p})`` by two-stage quadrature.

    **Two stages, because one is not enough and the failure would be silent.**
    The prior support for ``log_sigma`` is +/-9 in the log, while its posterior
    is ~0.3 wide, so a single grid over the support resolves the CDF with a
    handful of points -- and under the *tempered* likelihood of
    `test_the_sbc_check_detects_a_mis_scaled_posterior` the width halves again.
    A grid too coarse to resolve the posterior would make that test pass on
    quadrature error rather than on the miscalibration it is supposed to
    detect: a vacuous power test, which is worse than none.

    So: a coarse pass over the whole support for the location and scale, then a
    fine pass over +/-8 of those scales.  Mass outside is ~1e-15 under any
    calibration; a truth falling outside returns 0 or 1, which is the correct
    PIT there and exactly what drives detection.
    """
    def logpost(grid):
        vv = np.tile(v_star, (grid.size, 1))
        vv[:, p] = grid
        with mx.stream(mx.cpu):
            ll = np.array(gp.hi(a64(vv)), dtype=np.float64)
        return temper * ll + _prior_logpdf(h, grid)

    lo, hi = _support(h)
    coarse = np.linspace(lo, hi, n_coarse)
    lp = logpost(coarse)
    w = np.exp(lp - lp.max())
    mean = float(np.sum(w * coarse) / np.sum(w))
    var = float(np.sum(w * (coarse - mean) ** 2) / np.sum(w))
    sd = math.sqrt(max(var, 1e-12))

    f_lo, f_hi = max(lo, mean - 8.0 * sd), min(hi, mean + 8.0 * sd)
    if not f_hi > f_lo:
        f_lo, f_hi = lo, hi
    if v_star[p] <= f_lo:
        return 0.0
    if v_star[p] >= f_hi:
        return 1.0

    fine = np.linspace(f_lo, f_hi, n_fine)
    lp = logpost(fine)
    w = np.exp(lp - lp.max())
    cell = 0.5 * (w[1:] + w[:-1]) * np.diff(fine)
    cdf = np.concatenate([[0.0], np.cumsum(cell)])
    total = cdf[-1]
    assert total > 0.0 and np.isfinite(total), h.name
    return float(np.interp(v_star[p], fine, cdf / total))


def _pit_values(n_sim=48, n_data=300, temper=1.0, seed=4242):
    """PIT value per parameter per simulation; shape ``(n_sim, 4)``.

    ``temper`` scales the log-likelihood.  ``1.0`` is the real check; anything
    else is an injected miscalibration, which
    `test_the_sbc_check_detects_a_mis_scaled_posterior` uses to show the test
    has power.  Scaling the log is precisely the failure mode a point-estimate
    check cannot see: the mode does not move at all.
    """
    from anvilgp.priors import build_param_map

    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, BASELINE, n_data))
    yerr = np.full(n_data, YERR)
    # the prior is pinned here, once, from a throwaway reference realisation
    ref_cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                               yerr, jitter=JIT)
    y_ref = np.linalg.cholesky(ref_cov) @ rng.standard_normal(n_data)
    kern, jit = _sbc_hypers(t, y_ref, yerr)
    hypers = kern + [jit]
    pm, _, _ = build_param_map(kern, jitters=[jit])

    out = np.full((n_sim, len(hypers)), np.nan)
    for i in range(n_sim):
        v_star = np.array([_prior_draw(h, rng) for h in hypers])
        nat = _natural(hypers, v_star)
        # The generative model must be the one the likelihood evaluates, so
        # check *both* links rather than trust them.  A mismatch here --
        # generating with one jitter and inferring with another -- would not
        # fail loudly; it would quietly make every PIT value meaningless
        # while the test still reported a verdict.
        with mx.stream(mx.cpu):
            v1 = a64(v_star[None, :])
            kp = np.array(pm.kernel_params(v1), dtype=np.float64)[0]
            jv = float(np.array(pm.jitter[0].apply(v1), dtype=np.float64)[0, 0])
        assert np.allclose(kp, nat[:3], rtol=1e-12), (kp, nat[:3])
        assert abs(jv / nat[3] - 1.0) < 1e-12, (jv, nat[3])

        cov = dense_covariance(ag.SHOTerm(), nat[:3], t, yerr, jitter=nat[3])
        y = np.linalg.cholesky(cov) @ rng.standard_normal(n_data)
        gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                       GPPolicy(block=1024, hi_block=1024))

        for p, h in enumerate(hypers):
            out[i, p] = _conditional_cdf(gp, v_star, p, h, temper)
    return out, [h.name for h in hypers]


def _uniformity_p(u, n_bins=8):
    """Chi-squared p-value for ``u`` against Uniform(0, 1)."""
    from scipy import stats
    counts, _ = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    expect = len(u) / n_bins
    chi2 = float(np.sum((counts - expect) ** 2) / expect)
    return float(stats.chi2.sf(chi2, n_bins - 1)), chi2, counts


@pytest.mark.slow
def test_sbc_pit_values_are_uniform():
    """The owed SBC check: the PIT is Uniform(0, 1) for every parameter.

    This is the test that can see a mis-scaled posterior *width*, which every
    other calibration test in this file is blind to.  Run through the float64
    reference path (`hi`), so what it certifies is the likelihood's shape, not
    float32 rounding and not the sampler.

    Measured at 48 simulations, n = 300: chi-squared p-values 0.85 / 0.78 /
    0.28 / 0.23 for sigma / rho / Q / jitter, PIT standard deviations
    0.301-0.325 against the uniform 0.289, and flat bin counts.
    """
    pytest.importorskip("scipy")
    u, names = _pit_values()
    for p, name in enumerate(names):
        col = u[:, p]
        assert np.all((col >= 0.0) & (col <= 1.0)), name
        pval, chi2, counts = _uniformity_p(col)
        assert pval > 0.005, (
            f"{name}: PIT not uniform, chi2={chi2:.1f} p={pval:.2e} "
            f"counts={counts.tolist()}; mean={col.mean():.3f} "
            f"(0.5 expected), sd={col.std():.3f} (0.289 expected)")


@pytest.mark.slow
def test_the_sbc_check_detects_a_mis_scaled_posterior():
    """The SBC test must *fail* on a known-bad likelihood, or it proves nothing.

    Tempering the log-likelihood by 4 leaves every mode exactly where it was
    and divides each posterior width by about two -- the precise failure the
    profile-argmax tests above cannot see.  If this ever stops rejecting, the
    uniformity gate has lost its power and
    `test_sbc_pit_values_are_uniform` is decoration.

    Measured: p-values 7.5e-08 to 1.7e-12 across the four parameters, PIT
    standard deviations 0.376-0.419 against the uniform 0.289, and bin counts
    piled at both ends (e.g. ``[22, 2, 1, 2, 2, 2, 3, 14]``) -- the signature
    of an over-narrow posterior.  The PIT *means* stay at 0.43-0.50, which is
    the whole point: tempering does not move the mode at all, so a
    point-estimate check sees nothing.
    """
    pytest.importorskip("scipy")
    u, names = _pit_values(n_sim=48, temper=4.0)
    pvals = [_uniformity_p(u[:, p])[0] for p in range(len(names))]
    assert min(pvals) < 0.005, (
        f"a 4x tempered likelihood went undetected: p-values "
        f"{[f'{p:.3f}' for p in pvals]} against names {names}")
    # and the direction is the expected one: over-narrow posteriors push the
    # PIT towards the ends, so its spread exceeds the uniform 0.289
    worst = int(np.argmin(pvals))
    assert u[:, worst].std() > 0.30, u[:, worst].std()
