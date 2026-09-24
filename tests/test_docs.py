"""The claims in docs/ that are numbers, re-run.

Documentation rots quietly: a tutorial that no longer produces the value
it prints is worse than none, because a reader cannot tell which of the
two is wrong.  Everything pinned here is a figure a reader would act on.
"""

import numpy as np
import pytest

import mlx.core as mx

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, ParamMap
from anvilgp.oracle import dense_loglike, sample_ssm

c2 = pytest.importorskip("celerite2")
c2t = pytest.importorskip("celerite2.terms")
metal = pytest.importorskip("anvilgp.metal")

#: Every test here asserts a number the docs attribute to the *fused
#: kernel*.  Without this, a machine with no Metal gets a RuntimeWarning,
#: silently runs the MLX path, and validates figures against the wrong
#: one -- the two differ by 4x on matern32 in the docs' own table.
pytestmark = pytest.mark.skipif(not metal.metal_available(),
                                reason="needs mx.fast.metal_kernel")


def _tutorial_data():
    """docs/tutorial.md, "Some data"."""
    rng = np.random.default_rng(42)
    n = 2000
    t = np.arange(n) * 0.02
    yerr = np.full(n, 0.15)
    y = sample_ssm(ag.SHOTerm(), [1.2, 5.0, 4.0], 0.02, n, rng)
    return t, y + rng.standard_normal(n) * yerr, yerr


def test_the_tutorials_headline_agreement_with_celerite2():
    """docs/tutorial.md: "The two agree to 1.1e-3 on a log-likelihood of 734."

    The tutorial's whole claim to a celerite2 user is that this is the same
    likelihood.  If the gap grows past ~1e-2 the claim is no longer
    honest -- a Metropolis decision turns on differences of order 1, and
    the margin to that is what makes float32 acceptable.
    """
    t, y, yerr = _tutorial_data()
    gp2 = c2.GaussianProcess(c2t.SHOTerm(sigma=1.2, rho=5.0, Q=4.0), mean=0.0)
    gp2.compute(t, yerr=yerr)
    ll_c2 = gp2.log_likelihood(y)

    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
                   ParamMap.log([1.2, 5.0, 4.0]), GPPolicy(metal=True))
    with mx.stream(mx.gpu):
        ll = float(np.array(gp.log_prob_absolute(
            mx.zeros((1, 3), dtype=mx.float32)))[0])

    assert abs(ll_c2 - 733.62) < 0.5, f"tutorial prints 733.62, got {ll_c2}"
    assert abs(ll - ll_c2) < 1e-2, f"tutorial claims 1.1e-3, got {abs(ll-ll_c2):.2e}"


def test_the_batched_call_returns_one_value_per_vector():
    """docs/tutorial.md: "1024 log-likelihoods ... shape (1024,)", and the
    gradient "shape (1024, 3)"."""
    t, y, yerr = _tutorial_data()
    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
                   ParamMap.log([1.2, 5.0, 4.0]), GPPolicy(metal=True))
    rng = np.random.default_rng(0)
    u = mx.array(rng.standard_normal((1024, 3)).astype(np.float32) * 0.1)
    with mx.stream(mx.gpu):
        lls = np.array(gp(u))
        _, grad = mx.vjp(gp, [u], [mx.ones((1024,), dtype=mx.float32)])
        g = np.array(grad[0])
    assert lls.shape == (1024,) and np.all(np.isfinite(lls))
    assert g.shape == (1024, 3) and np.all(np.isfinite(g))


@pytest.mark.slow
def test_anvilgp_usually_but_not_always_beats_celerite2_on_matern32():
    """The one accuracy claim the docs make in anvil-gp's favour, pinned as
    the *aggregate* it actually is.

    celerite2 approximates Matern-3/2 rationally; anvil-gp uses the exact
    two-state model (the SHO at critical damping). That wins about three
    times in four, not always, because the two error sources are of
    similar size and which dominates depends on the parameters.

    This test has been wrong twice, both times because a story was fitted
    to too few points.  First it used off-model white noise, where the
    quadratic form dominates and anvil-gp appeared to win everywhere by
    65x (on-model data is the honest input; see
    `anvilgp/oracle.py::sample_ssm`).  Then it asserted a per-point winner
    monotone in rho/dt, fitted to four points; `sigma` matters too and the
    ordering is not monotone.  Hence a grid and an aggregate.
    """
    import itertools

    n, dt = 2000, 0.02
    wins_ag = 0
    errs_c2, errs_ag = [], []
    for sigma, rho, seed in itertools.product((0.5, 1.1, 2.0),
                                              (0.5, 1.0, 2.5, 10.0),
                                              (7, 11, 13)):
        rng = np.random.default_rng(seed)
        t = np.arange(n) * dt
        t[n // 2:] += 20.0
        yerr = np.full(n, 0.3)
        y = sample_ssm(ag.Matern32Term(), [sigma, rho], dt, n, rng) \
            + rng.standard_normal(n) * yerr

        ref = dense_loglike(ag.Matern32Term(), np.array([sigma, rho]),
                            t, y, yerr)
        gp2 = c2.GaussianProcess(c2t.Matern32Term(sigma=sigma, rho=rho),
                                 mean=0.0)
        gp2.compute(t, yerr=yerr)
        err_c2 = abs(gp2.log_likelihood(y) - ref)

        gp = GPLogLike(None, t, None, y, yerr, ag.Matern32Term(),
                       ParamMap.log([sigma, rho]), GPPolicy(metal=True))
        with mx.stream(mx.gpu):
            err_ag = abs(float(np.array(gp.log_prob_absolute(
                mx.zeros((1, 2), dtype=mx.float32)))[0]) - ref)

        errs_c2.append(err_c2)
        errs_ag.append(err_ag)
        wins_ag += err_ag < err_c2

    n_cfg = len(errs_ag)
    assert wins_ag >= 0.6 * n_cfg, (
        f"docs say anvil-gp wins ~27 of 36; got {wins_ag} of {n_cfg}")
    assert np.median(errs_ag) < np.median(errs_c2), (
        f"median: anvil-gp {np.median(errs_ag):.2e} vs "
        f"celerite2 {np.median(errs_c2):.2e}")
    assert max(errs_ag) < max(errs_c2), (
        f"worst case: anvil-gp {max(errs_ag):.2e} vs "
        f"celerite2 {max(errs_c2):.2e}")
    assert max(errs_ag) < 1e-2, f"anvil-gp worst {max(errs_ag):.2e}"


@pytest.mark.parametrize("term,m,eligible", [
    (ag.RealTerm(), 1, True),
    (ag.SHOTerm(), 2, True),
    (ag.Matern32Term(), 2, True),
    (ag.Matern52Term(), 3, True),
    (ag.RotationTerm(), 4, True),
    (ag.SHOTerm() + ag.SHOTerm(), 4, True),
    (ag.SHOTerm() + ag.Matern52Term(), 5, False),
    (ag.RotationTerm() + ag.SHOTerm(), 6, False),
])
def test_the_kernel_tables_state_dimensions_and_eligibility(term, m, eligible):
    """docs/kernels.md prints `m` for every kernel and a table of which
    sums stay under the fused kernel's limit.  A reader designs their model
    around these two columns."""
    assert term.n_state == m, f"docs say m = {m}, got {term.n_state}"
    t = np.arange(400) * 0.02
    gp = GPLogLike(None, t, None, np.zeros(400), np.full(400, 0.3), term,
                   ParamMap.log([1.0] * 2), GPPolicy(metal=False))
    gp.policy.metal = True
    ok, why = gp.metal_eligible()
    assert ok is eligible, f"docs say {eligible}, got {ok}: {why}"
    if not eligible:
        assert "register budget" in why, why
