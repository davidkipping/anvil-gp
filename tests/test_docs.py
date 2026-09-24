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


def test_anvilgp_is_the_more_accurate_of_the_two_on_matern32():
    """docs/kernels.md and docs/benchmarks.md make this the one accuracy
    claim in anvil-gp's favour, so it carries more weight than the rest and
    is worth pinning across the parameter range it is claimed over.

    celerite2 represents Matern-3/2 by a rational approximation; anvil-gp
    uses the exact two-state model (the SHO at critical damping).
    """
    rng = np.random.default_rng(7)
    n = 2000
    t = np.arange(n) * 0.02
    t[n // 2:] += 20.0
    y, yerr = rng.standard_normal(n), np.full(n, 0.3)

    for sigma, rho in ((1.1, 2.5), (1.0, 1.0), (0.5, 10.0), (2.0, 0.5)):
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

        assert err_ag < err_c2, (
            f"sigma={sigma} rho={rho}: anvil-gp {err_ag:.2e} vs "
            f"celerite2 {err_c2:.2e} -- the docs claim anvil-gp is better")
        assert err_ag < 1e-3, f"sigma={sigma} rho={rho}: {err_ag:.2e}"


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
