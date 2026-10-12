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

    Uses the stretch move, which needs no gradient.  Half the ensemble
    starts in the failure region so the rest has somewhere to pull it back
    from.

    **This test passed for the wrong reason until 2026-10-11.**  It used to
    say the *target's* log-probability "may still be -inf at such a point,
    because anvil's Transform contributes a -inf Jacobian at the edge of a
    bounded prior" -- and that -inf was what made it pass.  It made every
    move *within* the failure region ``(-inf) - (-inf) = NaN`` and so
    rejected, an accidental wall.  anvil f97f583 made the Jacobian finite
    (correctly), the constant -1e30 sentinel then absorbed it in float32,
    the region went flat, and 11 of the 32 dead walkers diffused to
    ``|u| ~ 1e11`` -- proven causal by restoring the old Jacobian
    in-process (0/32 stuck).  The fix is anvil-gp's: the failure value now
    slopes back towards the reference point (see `likelihood._sanitize` and
    the contract tests in `test_contract.py`).
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

    **What this test does and does not establish.**  It establishes the
    finiteness contract, which is all anvil-gp owes.  Its last assertion --
    that the healthy half "moved" -- compares against ``u0`` and so is
    satisfied in the first few warmup steps; it cannot see a sampler that
    has stopped.  It was passing while the stuck half collapsed ChEES's
    shared step size to 3e-10 and the healthy half moved EXACTLY zero across
    the whole sampling phase (anvil's reply,
    docs/anvil-chees-stuck-chains-reply.md; measured here 2026-10-11).
    Whether the healthy half really keeps sampling is tested separately, in
    `test_healthy_chains_keep_sampling_*` below.
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
    # Weak on purpose, and labelled so: distance from the START, which warmup
    # alone covers.  See the docstring and the tests below for the real check.
    moved = np.abs(last - u0).max(axis=1)
    assert np.all(moved[half:] > 1e-3), "healthy chains never left u0"


def _chees_sampling_health(n_stuck):
    """ChEES with ``n_stuck`` of 16 chains started at u = 15 (finite, deep).

    Returns ``(step_size, healthy_motion)``: the step size adaptation ended
    on, and the median distance a *healthy* chain travels across the 20
    **sampling** steps -- measured between draws, not from ``u0``, which is
    what makes it able to see a stalled sampler.
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
    u0 = rng.standard_normal((16, tgt.transform.dim)).astype(np.float32) * 0.3
    if n_stuck:
        u0[:n_stuck] = 15.0
    with mx.stream(mx.gpu):
        res = anvil.run(anvil.ChEESHMC(tgt.target, max_leapfrog=16),
                        tgt.target, mx.array(u0), n_warmup=60, n_samples=20,
                        seed=3, progress=False)
    chain = np.asarray(res.get_chain(), dtype=np.float64)   # sampling phase
    healthy = slice(n_stuck, None)
    motion = np.abs(chain[-1, healthy] - chain[0, healthy]).max(axis=1)
    return float(np.array(res.final_params["step_size"])), float(np.median(motion))


@pytest.mark.slow
def test_healthy_chains_keep_sampling_when_no_chain_is_stuck():
    """The control: the assertions below must PASS when nothing is stuck.

    Measured: step size 2.4e-2, healthy chains travel 0.35 across the 20
    sampling steps.  Without this, the xfail beside it could be failing for
    any reason; with it, the failure is attributable to the stuck chains.
    """
    step, motion = _chees_sampling_health(n_stuck=0)
    assert step > 1e-3, f"step size {step:.2e} with no chain stuck"
    assert motion > 1e-2, f"chains travelled {motion:.2e} across sampling"


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "ChEES adapts ONE step size from the harmonic mean of every chain's "
    "acceptance, so a single finite chain stranded near anvil-gp's float32 "
    "feasibility edge drags it down for all: measured 3.1e-10 with 8 of 16 "
    "stranded, healthy chains moving exactly 0 across sampling (control: "
    "2.4e-2 and 0.35).  anvil 0.7.0 excludes chains at the failure value "
    "from the statistic but cannot exclude a finite straggler.  See "
    "docs/anvil-chees-stuck-chains-reply.md.  strict: when this starts "
    "passing, the edge has been fixed -- delete the xfail."))
def test_healthy_chains_keep_sampling_when_some_chains_are_stuck():
    """What a user of anvil-gp actually wants, which does not hold today.

    The finiteness contract above is met.  This is the stronger property --
    that stranded chains do not stop the rest of the run -- and anvil's reply
    shows it fails for a reason that sits in anvil-gp's likelihood, not in
    ChEES: a band near the float32 feasibility edge where only steps of
    order 1e-7 are accepted, and where compiled and eager evaluation
    disagree.  It cannot bias a posterior (adaptation only sets the step
    size) but it can make a whole run slow.  The default initialisation is
    NOT exposed; the exposure is graded by initial spread and is measured in
    `docs/anvil-chees-stuck-chains-measurement.md`.  This test starts half
    the chains at ``u = 15``, deep in the exposed range.
    """
    step, motion = _chees_sampling_health(n_stuck=8)
    assert step > 1e-3, f"step size collapsed to {step:.2e}"
    assert motion > 1e-2, f"healthy chains travelled {motion:.2e} across sampling"


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


# ---------------------------------------------------------------------------
# Adversarial geometry (M5 (b))
# ---------------------------------------------------------------------------
#
# The brief asks for two injections, chosen not for the parameters they
# recover but for the posterior *shapes* they create -- the two shapes that
# break gradient samplers:
#
#   "zero/weak-amplitude injection (posterior piles against the bound, zero
#    divergences) and timescale ~= cadence injection (sigma_GP-jitter ridge
#    sampled correctly)."
#
# This is a different question from the SBC checks above.  Those ask whether
# the likelihood has the right shape on typical data; these ask whether the
# sampler survives the two atypical shapes.  A plateau (nothing to infer) and
# a ridge (two parameters trading off) are where HMC diverges, sticks, or
# reports false precision.

GEOM_WARM, GEOM_SAMP, GEOM_CHAINS = 600, 1200, 128


def _geometry_run(y, t, yerr, seed=3):
    """ChEES on `make_target`'s own priors; returns natural draws + diagnostics.

    ChEES rather than the stretch move because only an HMC kernel reports
    divergences at all (`anvil.kernels.chees` sets ``diverged`` from the
    energy error; the ensemble kernel has no such concept), and "zero
    divergences" is the brief's acceptance criterion.
    """
    anvil = pytest.importorskip("anvil")
    from anvil.diagnostics import split_rhat
    from anvilgp import GPPolicy
    from anvilgp.anvil import make_target

    gt = make_target(ag.SHOTerm(), t, y, yerr,
                     policy=GPPolicy(block=1024, metal=True))
    with mx.stream(mx.gpu):
        res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=32), gt.target,
                        gt.u0(GEOM_CHAINS, seed=1, spread=0.3),
                        n_warmup=GEOM_WARM, n_samples=GEOM_SAMP, seed=seed,
                        progress=False)
    rhat = split_rhat(np.asarray(res.get_chain(), dtype=np.float64))
    flat = gt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    return dict(gt=gt, res=res, rhat=rhat, model=flat, nat=gt.natural(flat),
                n_div=int(res.extras.get("n_divergent", 0)),
                accept=float(np.mean(res.accept_fraction)))


@pytest.mark.slow
def test_zero_amplitude_injection_is_a_survivable_plateau():
    """No GP signal and no jitter: nothing to infer, and it must not break.

    The hardest version of the brief's "zero/weak-amplitude" case -- pure
    white noise at the quoted error bars, so *both* extra scales are
    unidentified at once.  As ``sigma -> 0`` the kernel stops depending on
    ``rho`` and ``Q`` at all, and the posterior over them flattens onto the
    prior.  That plateau is where a gradient sampler wanders off or diverges.

    What correct behaviour looks like, and all of it is asserted: no
    divergences, nothing non-finite, both amplitudes pinned near zero, and
    the unidentified directions actually **explored** rather than stuck --
    ``rho`` covers 95% of its prior support here, which is what a flat
    posterior should look like and what a stuck chain would not.

    **R-hat on a plateau converges slowly, and that is not a defect.**
    Measured on this problem: 1.26 at 400 warmup / 400 samples, 1.036 at
    800/1600, 1.009 at 1500/3000 -- monotone towards 1, with zero divergences
    at every length.  A flat direction simply needs more samples to mix.  The
    gate below is 1.1 at 600/1200 (measured 1.032); if it ever fails, check
    whether it *falls* with a longer run before calling it a bug.
    """
    rng = np.random.default_rng(11)
    n = 600
    t = np.sort(rng.uniform(0.0, BASELINE, n))
    yerr = np.full(n, YERR)
    y = rng.standard_normal(n) * YERR          # no GP, no jitter

    out = _geometry_run(y, t, yerr)
    nat, gt = out["nat"], out["gt"]

    assert out["n_div"] == 0, f"{out['n_div']} divergences on the plateau"
    assert out["accept"] > 0.5, out["accept"]
    for k, v in nat.items():
        assert np.all(np.isfinite(v)), f"{k} went non-finite"
    assert out["rhat"].max() < 1.1, (
        f"R-hat {out['rhat'].max():.3f}; measured 1.032 here. Check it falls "
        f"with a longer run (1.009 at 1500/3000) before calling it a defect")

    # both amplitudes consistent with zero, against the quoted error bar
    s99 = float(np.percentile(nat["gp_sigma"], 99))
    j99 = float(np.percentile(nat["jitter"], 99))
    assert s99 < 0.6 * YERR, f"GP amplitude 99th pct {s99:.4f} vs yerr {YERR}"
    assert j99 < 0.8 * YERR, f"jitter 99th pct {j99:.4f} vs yerr {YERR}"

    # and the flat directions are explored, not stuck: rho spans its prior
    lo, hi = gt.hypers[1].lo, gt.hypers[1].hi
    log_rho = out["model"][:, 1]
    covered = (np.percentile(log_rho, 99) - np.percentile(log_rho, 1)) / (hi - lo)
    assert covered > 0.7, (
        f"log_rho covers only {covered:.2f} of its prior support; on a "
        f"plateau it should cover nearly all of it (measured 0.95)")


@pytest.mark.slow
def test_timescale_near_cadence_samples_the_sigma_jitter_ridge():
    """A GP too fast to resolve looks like white noise, and must say so.

    The brief's second geometry.  Injected at ``rho = 2 x cadence``, which is
    exactly the floor `priors._log_rho_bounds` puts on the timescale prior --
    below it the kernel is unresolved and degenerates into the jitter.  So
    this sits the truth *on* the boundary, which is the adversarial part: the
    posterior has to pile against that bound and, at the same time, trade GP
    amplitude against white jitter along a ridge, because a very short
    timescale and a larger jitter explain the same data.

    Sampling it "correctly" means three things, all asserted: the ridge is
    traversed rather than collapsed to one end (the two parameters come out
    anti-correlated); the *combination* is far better determined than either
    part, which is what makes it a ridge rather than two loose parameters;
    and the truth is inside the credible region despite neither parameter
    being individually sharp.  Measured here: correlation -0.41, and the
    coefficient of variation is 0.046 for ``sqrt(sigma^2 + jitter^2)``
    against 0.34 for the jitter alone -- a factor of 7.
    """
    rng = np.random.default_rng(11)
    n = 600
    t = np.sort(rng.uniform(0.0, BASELINE, n))
    yerr = np.full(n, YERR)
    cadence = float(np.median(np.diff(t)))
    rho_true = 2.0 * cadence               # the prior's own floor
    sigma_true, jit_true = 0.35, 0.05
    cov = dense_covariance(ag.SHOTerm(),
                           np.array([sigma_true, rho_true, 3.0]), t, yerr,
                           jitter=jit_true)
    y = np.linalg.cholesky(cov) @ rng.standard_normal(n)

    out = _geometry_run(y, t, yerr)
    nat = out["nat"]
    s, j = nat["gp_sigma"], nat["jitter"]

    assert out["n_div"] == 0, f"{out['n_div']} divergences on the ridge"
    for k, v in nat.items():
        assert np.all(np.isfinite(v)), f"{k} went non-finite"
    assert out["rhat"].max() < 1.1, out["rhat"]

    # the timescale piles against its lower bound
    ratio = float(np.median(nat["gp_rho"]) / rho_true)
    assert ratio < 1.15, (
        f"rho median is {ratio:.3f}x the prior floor; it should sit on it")

    # the ridge: anti-correlated, and the combination is the sharp direction
    corr = float(np.corrcoef(np.log(s), np.log(j))[0, 1])
    assert corr < -0.2, (
        f"corr(log sigma, log jitter) = {corr:+.3f}; without a ridge these "
        f"are independent, and a collapsed chain would show neither")
    cv = lambda x: float(x.std() / x.mean())
    total = np.sqrt(s ** 2 + j ** 2)
    assert cv(total) < 0.4 * cv(j), (
        f"total cv {cv(total):.4f} vs jitter cv {cv(j):.4f}: the combination "
        f"should be much better determined than the parts")
    assert cv(total) <= cv(s), (cv(total), cv(s))

    # and the truth survives, despite neither parameter being sharp alone
    for name, val, draws in (("sigma", sigma_true, s), ("jitter", jit_true, j)):
        lo, hi = np.percentile(draws, 1), np.percentile(draws, 99)
        assert lo < val < hi, f"{name} truth {val} outside [{lo:.4f},{hi:.4f}]"


@pytest.mark.slow
def test_the_geometry_signatures_are_absent_when_the_gp_is_resolved():
    """The power test for the two above: a resolved GP must show neither shape.

    Both geometry tests assert that a *signature* is present -- a timescale
    piled on its bound, an anti-correlated pair, a plateau covering its prior.
    Asserting presence is only meaningful if the signature can be absent, so
    this fits a well-resolved GP (``rho = 5``, fifty times the cadence, with a
    real amplitude) and checks that every one of those assertions would fail
    on it.  Measured: rho sits at 48x the floor rather than on it, the
    sigma-jitter correlation is -0.02 rather than -0.41, the combination is
    *worse* determined than the jitter rather than seven times better, the
    amplitude is 0.72 rather than under 0.12, and log_rho covers 5% of its
    prior rather than 95%.

    If this test ever fails it means a geometry signature has become
    unconditional -- the two tests above would then be passing on anything,
    and would be measuring nothing.
    """
    rng = np.random.default_rng(11)
    n = 600
    t = np.sort(rng.uniform(0.0, BASELINE, n))
    yerr = np.full(n, YERR)
    cadence = float(np.median(np.diff(t)))
    cov = dense_covariance(ag.SHOTerm(), np.array([0.35, 5.0, 3.0]), t, yerr,
                           jitter=0.05)
    y = np.linalg.cholesky(cov) @ rng.standard_normal(n)

    out = _geometry_run(y, t, yerr)
    nat, gt = out["nat"], out["gt"]
    s, j = nat["gp_sigma"], nat["jitter"]
    cv = lambda x: float(x.std() / x.mean())

    assert out["n_div"] == 0, out["n_div"]
    # not piled on the timescale floor
    assert float(np.median(nat["gp_rho"]) / (2.0 * cadence)) > 1.15
    # no ridge
    assert float(np.corrcoef(np.log(s), np.log(j))[0, 1]) > -0.2
    assert cv(np.sqrt(s ** 2 + j ** 2)) > 0.4 * cv(j)
    # a real amplitude, not one consistent with zero
    assert float(np.percentile(s, 99)) > 0.6 * YERR
    # and the timescale is identified, not a plateau
    lo, hi = gt.hypers[1].lo, gt.hypers[1].hi
    log_rho = out["model"][:, 1]
    covered = (np.percentile(log_rho, 99) - np.percentile(log_rho, 1)) / (hi - lo)
    assert covered < 0.7, covered


# ---------------------------------------------------------------------------
# The fiducial joint fit (M5 (a))
# ---------------------------------------------------------------------------
#
# The brief (written before MetalPlanet was renamed Elome):
# "MetalPlanet hot Jupiter + injected SHO variability + jitter,
# both samplers, truth recovered, zero divergences, cross-sampler agreement."
# `examples/hotjupiter_gp_joint.py` is the full-size version with the printed
# posterior; this is the same problem shrunk to test size.
#
# It is the only test here that exercises the whole stack at once -- a
# Elome transit as the mean model, an anvil-gp GP over the residual, and
# anvil sampling all twelve parameters jointly -- so it is the one that catches
# an integration break that every unit test passes through.

HJ_BASELINE, HJ_N, HJ_YERR = 30.0, 5000, 500e-6
HJ_TRANSIT = dict(period=3.0, t0=1.2345, r=0.103, b=0.5, a=8.75,
                  u1=0.40, u2=0.25)
HJ_GP = dict(sigma=1000e-6, rho=2.0, qual=5.0, jitter=200e-6)


@pytest.fixture(scope="module")
def hotjupiter_joint():
    """Data, target and truth for the joint fit; built once for both samplers."""
    pytest.importorskip("anvil")
    el = pytest.importorskip("elome.anvil")
    from elome.ld import u_to_q_np
    from elome.orbit import epoch_center_times

    from anvilgp import GPPolicy
    from anvilgp.anvil import make_target
    from anvilgp.oracle import sample_ssm
    from anvilgp.priors import SpecHint

    tr = HJ_TRANSIT
    q1, q2 = (float(q) for q in u_to_q_np(tr["u1"], tr["u2"]))
    dt = HJ_BASELINE / HJ_N
    t_model = np.arange(HJ_N) * dt + 0.5 * dt
    # references a preliminary fit would have got slightly wrong
    t0_ref, period_ref = tr["t0"] - 0.009, tr["period"] + 0.0005
    x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
    truth_mean = np.array([tr["t0"] - t0_ref, tr["period"] - period_ref,
                           tr["r"], tr["b"], tr["a"], q1, q2, 0.0])

    model_fn = el.make_quad_transit_flux(period_ref=period_ref)
    with mx.stream(mx.cpu):                    # float64 through the fallback
        transit = np.array(
            model_fn(mx.array(truth_mean[None, :], dtype=mx.float64),
                     mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)
    rng = np.random.default_rng(7)
    gp_draw = sample_ssm(ag.SHOTerm(),
                         [HJ_GP["sigma"], HJ_GP["rho"], HJ_GP["qual"]],
                         dt, HJ_N, rng)
    white = rng.standard_normal(HJ_N) * np.sqrt(HJ_YERR ** 2
                                                + HJ_GP["jitter"] ** 2)
    y_fit = transit + gp_draw + white

    specs = [SpecHint(n, lo=el.DEFAULT_BOUNDS[n][0], hi=el.DEFAULT_BOUNDS[n][1])
             for n in el.PARAM_NAMES]
    gt = make_target(ag.SHOTerm(), t_model, y_fit, np.full(HJ_N, HJ_YERR),
                     mean_fn=model_fn, x_mean=x64, mean_specs=specs,
                     policy=GPPolicy(block=1024, metal=True))
    gp_truth = np.array([HJ_GP["sigma"], HJ_GP["rho"], HJ_GP["qual"],
                         HJ_GP["jitter"]])
    truth_model = np.concatenate(
        [truth_mean, np.log(gp_truth / np.array([h.ref for h in gt.hypers]))])
    u_truth = gt.transform.from_model_np(truth_model)
    init = np.random.default_rng(3)

    def ball(n, spread=1e-3):
        return mx.array((u_truth + spread * init.standard_normal(
            (n, gt.transform.dim))).astype(np.float32))

    truth = dict(zip(el.PARAM_NAMES, truth_model[:8]))
    truth.update(zip(("gp_sigma", "gp_rho", "gp_Q", "jitter"), gp_truth))
    return dict(gt=gt, ball=ball, truth=truth, names=list(el.PARAM_NAMES),
                depth=float(-transit.min()), gp_sd=float(gp_draw.std()))


def _hj_draws(gt, res, names):
    """Posterior draws per parameter: mean model in model space, GP natural."""
    flat = gt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    nat = gt.natural(flat)
    cols = {n: flat[:, i] for i, n in enumerate(names)}
    cols.update({k: nat[k] for k in ("gp_sigma", "gp_rho", "gp_Q", "jitter")})
    return cols


@pytest.fixture(scope="module")
def hj_chees(hotjupiter_joint):
    """ChEES-HMC on the joint fit.

    400 warmup, not 250, and that is load-bearing: Elome's
    docs/sampler-integration.md reports ~1% divergences at 200 warmup
    iterations and exactly zero at 400-600, calling them a step-size
    adaptation artefact rather than a geometry problem.  Reproduced here on a
    *different* posterior -- twelve joint parameters rather than eight transit
    ones -- 1 divergence at 250, zero at 400 and at 600, with acceptance
    rising 0.84 -> 0.92.  If this test ever shows divergences, lengthen the
    warmup before suspecting the model.
    """
    anvil = pytest.importorskip("anvil")
    from anvil.diagnostics import split_rhat
    gt, ball = hotjupiter_joint["gt"], hotjupiter_joint["ball"]
    with mx.stream(mx.gpu):
        res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=128), gt.target,
                        ball(64), n_warmup=400, n_samples=200, seed=2,
                        progress=False)
    return dict(cols=_hj_draws(gt, res, hotjupiter_joint["names"]),
                n_div=int(res.extras.get("n_divergent", 0)),
                rhat=float(split_rhat(np.asarray(res.get_chain(),
                                                 dtype=np.float64)).max()),
                accept=float(np.mean(res.accept_fraction)))


@pytest.fixture(scope="module")
def hj_ensemble(hotjupiter_joint):
    """The gradient-free arm: stretch + DE, long chains.

    The two samplers want opposite budgets and configuring them alike produces
    an unconverged comparison.  The ensemble's autocorrelation on this family
    of problems is ~120 iterations (anvil's own hot-Jupiter example), so it
    gets 6000 draws where ChEES gets 200.
    """
    anvil = pytest.importorskip("anvil")
    from anvil.diagnostics import split_rhat
    from anvil.kernels.ensemble import DEMove, EnsembleKernel, StretchMove
    gt, ball = hotjupiter_joint["gt"], hotjupiter_joint["ball"]
    with mx.stream(mx.gpu):
        res = anvil.run(
            EnsembleKernel(gt.target,
                           moves=[(StretchMove(), 0.5), (DEMove(), 0.5)],
                           seed=1),
            gt.target, ball(128), n_warmup=500, n_samples=6000, seed=2,
            progress=False)
    return dict(cols=_hj_draws(gt, res, hotjupiter_joint["names"]),
                rhat=float(split_rhat(np.asarray(res.get_chain(),
                                                 dtype=np.float64)).max()))


@pytest.mark.slow
def test_the_joint_fit_is_well_posed(hotjupiter_joint):
    """The injection is the problem the brief asks for, not an easier one.

    Asserted because the whole test below is vacuous if the GP is negligible
    or the transit unconstrained: the variability has to be a real contaminant
    (~10% of the transit depth, correlated on the scale of the transit
    duration) and there have to be enough in-transit points to measure a
    depth.
    """
    gt = hotjupiter_joint["gt"]
    assert gt.transform.dim == 12, gt.names
    assert gt.loglike.metal_eligible()[0], gt.loglike.metal_eligible()
    depth, gp_sd = hotjupiter_joint["depth"], hotjupiter_joint["gp_sd"]
    assert 0.05 < gp_sd / depth < 0.3, (
        f"variability is {gp_sd / depth:.1%} of the depth; the point of the "
        f"fit is that it is a real contaminant")
    assert gp_sd > 1.5 * HJ_YERR, "variability must dominate the white noise"
    assert HJ_GP["rho"] > 5.0 * (HJ_BASELINE / HJ_N), "GP must be resolved"


@pytest.mark.slow
def test_the_joint_fit_recovers_truth_without_divergences(hotjupiter_joint,
                                                          hj_chees):
    """ChEES on twelve joint parameters: truth in, truth out, no divergences.

    Measured at this size (anvil 0.5.0): zero divergences, acceptance 0.92,
    max R-hat 1.015, and the worst parameter 2.24 sigma from truth (``gp_Q``,
    the least constrained -- posterior sd 1.0 here on a truth of 5).  The
    full-size version in ``examples/hotjupiter_gp_joint.py`` puts every one of
    the twelve within 1.1 sigma (worst 1.10).
    """
    cols, truth = hj_chees["cols"], hotjupiter_joint["truth"]
    assert hj_chees["n_div"] == 0, (
        f"{hj_chees['n_div']} divergences; lengthen the warmup before "
        f"suspecting the model (see the fixture docstring)")
    assert hj_chees["accept"] > 0.6, hj_chees["accept"]
    assert hj_chees["rhat"] < 1.05, hj_chees["rhat"]
    for name, draws in cols.items():
        assert np.all(np.isfinite(draws)), name
        z = (np.median(draws) - truth[name]) / draws.std()
        assert abs(z) < 3.0, (
            f"{name}: median {np.median(draws):.6g} vs truth "
            f"{truth[name]:.6g}, {z:+.2f} sigma")


@pytest.mark.slow
def test_the_two_samplers_agree_on_the_joint_posterior(hotjupiter_joint,
                                                      hj_chees, hj_ensemble):
    """Cross-sampler agreement -- the check a wrong gradient fails.

    ChEES uses the analytic gradient of both the transit kernel and the GP
    reduction; the ensemble uses no gradient at all.  If they agree on all
    twelve marginals, the gradient is right; if the gradient were wrong, HMC
    would still *run* -- it would sample a different distribution.  This
    project has been bitten from the other side too, by an ensemble whose
    trapped walkers inflated a width fourfold (docs/anvil-stretch-width.md),
    so the comparison guards both directions.

    Measured on anvil 0.5.0: worst median shift 0.070 sigma, worst width ratio
    1.26x at this size; 0.019 sigma and 1.013x at full size.  The width gate
    (1.4x) has the least margin of the assertions here, ~11%, and it is the
    ensemble's estimate that is noisy.  The ensemble's R-hat is the looser of
    the two (1.088 here against ChEES's 1.015) because the stretch
    move decorrelates in ~120 iterations rather than ~2, which is why its
    width estimate is the noisier one.
    """
    a, b = hj_chees["cols"], hj_ensemble["cols"]
    assert hj_ensemble["rhat"] < 1.15, hj_ensemble["rhat"]
    for name in a:
        sd = 0.5 * (a[name].std() + b[name].std())
        shift = abs(np.median(a[name]) - np.median(b[name])) / sd
        ratio = a[name].std() / b[name].std()
        assert shift < 0.25, (
            f"{name}: samplers disagree by {shift:.3f} sigma on the median")
        assert 1 / 1.4 < ratio < 1.4, (
            f"{name}: width ratio {ratio:.3f} between samplers")
