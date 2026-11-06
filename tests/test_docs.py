"""The claims in docs/ that are numbers, re-run.

Documentation rots quietly: a tutorial that no longer produces the value
it prints is worse than none, because a reader cannot tell which of the
two is wrong.  Everything pinned here is a figure a reader would act on.
"""

import warnings

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
    """docs/tutorial.md: "The two agree to 4e-5 on a log-likelihood of 734."

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
    assert abs(ll - ll_c2) < 1e-3, f"tutorial claims 4e-5, got {abs(ll-ll_c2):.2e}"


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


@pytest.mark.parametrize("term,m,n_par,eligible", [
    (ag.RealTerm(), 1, 2, True),
    (ag.SHOTerm(), 2, 3, True),
    (ag.Matern32Term(), 2, 2, True),
    (ag.Matern52Term(), 3, 2, True),
    (ag.RotationTerm(), 4, 5, True),
    (ag.ComplexTerm(), 2, 4, True),
    # kernels.md: "the state dimension is unchanged, so a convolved
    # RotationTerm is still m = 4 and still runs on the fused kernel".
    (ag.TermConvolution(ag.RotationTerm(), 0.01), 4, 5, True),
    (ag.TermConvolution(ag.SHOTerm(), 0.01), 2, 3, True),
    (ag.SHOTerm() + ag.SHOTerm(), 4, 6, True),
    (ag.SHOTerm() + ag.Matern52Term(), 5, 5, True),
    (ag.RotationTerm() + ag.SHOTerm(), 6, 8, True),
    (ag.RotationTerm() + ag.RotationTerm(), 8, 10, False),
])
def test_the_kernel_tables_state_dimensions_and_eligibility(term, m, n_par,
                                                            eligible):
    """docs/kernels.md prints `m` for every kernel and a table of which
    sums stay under the fused kernel's limit.  A reader designs their model
    around these two columns.

    The parameter map matches the term, and `GPPolicy(metal=True)` is
    passed at construction rather than flipped afterwards, so this
    exercises the real dispatch and the documented "warns and falls back"
    behaviour -- not just the predicate.  An earlier version gave every
    term a 2-parameter map and flipped the policy, which left `_metal_ok`
    stale and made the objects uncallable.
    """
    assert term.n_state == m, f"docs say m = {m}, got {term.n_state}"
    n = 400
    rng = np.random.default_rng(0)
    t = np.arange(n) * 0.02
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp = GPLogLike(None, t, None, rng.standard_normal(n),
                       np.full(n, 0.3), term, ParamMap.log([1.0] * n_par),
                       GPPolicy(metal=True))
    ok, why = gp.metal_eligible()
    assert ok is eligible, f"docs say {eligible}, got {ok}: {why}"
    assert gp._metal_ok is eligible

    if eligible:
        assert not [w for w in caught if "cannot be honoured" in str(w.message)]
    else:
        assert "register budget" in why, why
        # docs/kernels.md: "warns and falls back", not fails
        assert [w for w in caught if "cannot be honoured" in str(w.message)], (
            "kernels.md promises a warning on fallback")

    # and it is actually callable on whichever path it chose
    u = mx.zeros((2, n_par), dtype=mx.float32)
    with mx.stream(mx.gpu):
        v = gp(u)
        mx.eval(v)
    assert bool(mx.all(mx.isfinite(v)))


def _exposure_data():
    """docs/tutorial.md, "Long-cadence data: exposure time".

    `sample_ssm` cannot draw from a convolved process -- it assumes
    `P_inf = I` and clips negative eigenvalues of `Q`, and a convolved `Q`
    has genuinely negative directions.  So the on-model draw is the dense
    one, from the exact piecewise covariance.
    """
    n, cad = 2000, 0.0204
    t = np.arange(n) * cad
    yerr = np.full(n, 0.15)
    p = [1.2, 0.1, 4.0]
    cv = ag.TermConvolution(ag.SHOTerm(), cad)
    tau = np.abs(t[:, None] - t[None, :]).ravel()
    k = np.asarray(cv.value_np(np.array(p), tau)).reshape(n, n)
    w, v = np.linalg.eigh(k)
    rng = np.random.default_rng(42)
    y = ((v * np.sqrt(np.maximum(w, 0.0))) @ rng.standard_normal(n)
         + rng.standard_normal(n) * yerr)
    return t, y, yerr, p, cad, cv


def test_the_tutorials_exposure_time_table():
    """Every figure in the tutorial's exposure-time table, re-run.

    Including the last row, which is the point of the section: the same
    term *without* the convolution, at the same true parameters, is 44
    log-units worse.  If that shrinks, the example has stopped
    demonstrating anything and the prose needs rewriting, not the number.
    """
    t, y, yerr, p, cad, cv = _exposure_data()
    pm = ParamMap.log(p)
    conv = GPLogLike(None, t, None, y, yerr, cv, pm, GPPolicy(metal=True))
    plain = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                      GPPolicy(metal=True))
    with mx.stream(mx.cpu):
        u = np.zeros((1, 3))
        ours = float(np.array(conv.hi_absolute(mx.array(u, dtype=mx.float64)))[0])
        naive = float(np.array(plain.hi_absolute(mx.array(u, dtype=mx.float64)))[0])
    ref = c2.GaussianProcess(
        c2t.TermConvolution(c2t.SHOTerm(sigma=p[0], rho=p[1], Q=p[2]), cad),
        mean=0.0)
    ref.compute(t, yerr=yerr)
    c2v = ref.log_likelihood(y)
    with mx.stream(mx.gpu):
        f32 = float(np.array(conv.log_prob_absolute(
            mx.zeros((4, 3), dtype=mx.float32)))[0])

    assert abs(ours - (-1982.2663)) < 5e-4, ours
    assert abs(ours - c2v) / abs(c2v) < 1e-13, f"{ours} vs {c2v}"
    assert abs(f32 - c2v) / abs(c2v) < 5e-7, f"{f32} vs {c2v}"
    assert abs(naive - (-2026.1469)) < 5e-4, naive
    assert naive - ours < -40.0, f"penalty shrank to {naive - ours:+.1f}"
    # "suppresses the zero-lag variance by 12%"
    ratio = (float(np.asarray(cv.value_np(np.array(p), np.array([0.0])))[0])
             / float(np.asarray(ag.SHOTerm().value_np(np.array(p),
                                                      np.array([0.0])))[0]))
    assert abs(ratio - 0.8781) < 1e-4, ratio


def test_the_tutorials_held_out_section():
    """docs/tutorial.md, "Holding a feature out of the GP": the parameter
    list, `predict` at a held-out time is ``E[f_M | r_S]`` (finite, from
    S), and the warm start through model space runs (few steps here)."""
    anvil = pytest.importorskip("anvil")
    from anvil import EnsembleKernel
    from anvilgp.anvil import make_target
    t, y, yerr = _tutorial_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = make_target(ag.SHOTerm(), t, y, yerr, held_out=[(18.0, 20.0)])
        plain = make_target(ag.SHOTerm(), t, y, yerr)
    assert gt.names == plain.names == ['log_gp_sigma', 'log_gp_rho',
                                       'log_gp_Q', 'log_jitter']
    assert gt.loglike._metal_ok
    res0 = anvil.run(EnsembleKernel(plain.target, seed=2), plain.target,
                     plain.u0(64, seed=1), n_warmup=50, n_samples=20,
                     seed=3, progress=False)
    v_last = plain.transform.model_np(
        np.asarray(res0.get_chain()[-1], dtype=np.float64))
    u_start = mx.array(gt.transform.from_model_np(v_last), dtype=mx.float32)
    res = anvil.run(EnsembleKernel(gt.target, seed=2), gt.target, u_start,
                    n_warmup=20, n_samples=20, seed=4, progress=False)
    chain = np.asarray(res.get_chain())
    assert chain.shape[1:] == (64, 4) and np.all(np.isfinite(chain))
    mean, var = gt.predict(v_last[:3], np.array([18.5, 19.0, 19.5]))
    assert mean.shape == (3, 3) and np.all(np.isfinite(mean))


def test_the_tutorials_feature_sections():
    """docs/tutorial.md, "Beyond the plain likelihood": the parameter lists
    it prints, the shapes, and "``Y^T X`` agrees with ``G`` to 3e-6".  The
    sampled figures (beta, mu) are not pinned; the draws here are
    reference-centred rather than posterior ones."""
    pytest.importorskip("anvil")
    from anvilgp import noise
    from anvilgp.anvil import make_target
    t, y, yerr = _tutorial_data()
    n, tc = t.size, float(t.mean())

    def baseline(v_mean, x, m0):
        return mx.stack([mx.ones_like(x), x - tc], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gn = make_target(ag.SHOTerm(), t, y, yerr,
                         noise=noise.scale_and_jitter(yerr))
        gt = make_target(ag.SHOTerm(), t, y + 3.0 + 0.05 * (t - tc), yerr,
                         linear_basis=baseline)
    assert gn.names == ['log_gp_sigma', 'log_gp_rho', 'log_gp_Q',
                        'log_noise_scale', 'log_noise_jitter']
    assert gt.names == ['log_gp_sigma', 'log_gp_rho', 'log_gp_Q',
                        'log_jitter']
    # "raw times (... a BJD) give exactly the same float32 value as the
    # centred form": the GP sees only time differences, so shifting t to a
    # BJD changes nothing else
    u = gt.u0(4, seed=1)
    for t0 in (0.0, 2459000.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = make_target(ag.SHOTerm(), t + t0,
                              y + 3.0 + 0.05 * (t - tc), yerr,
                              linear_basis=lambda vm, x, m0: mx.stack(
                                  [mx.ones_like(x), x], -1))
        with mx.stream(mx.gpu):
            np.testing.assert_allclose(np.array(raw.target.log_prob(u)),
                                       np.array(gt.target.log_prob(u)),
                                       rtol=1e-6)
    vs = np.array(gt.transform.to_model(gt.u0(4, seed=1))).astype(np.float64)
    beta, cov = gt.linear_posterior(vs)
    assert beta.shape == (4, 2) and cov.shape == (4, 2, 2)
    mean, var = gt.predict(vs, np.linspace(-2.0, 42.0, 500))
    assert mean.shape == var.shape == (4, 500)
    Y = np.stack([np.ones(n), t - tc], 1)
    with mx.stream(mx.gpu):
        G = np.array(gt.loglike.gram(mx.array(vs, dtype=mx.float32), Y))
    X = gt.apply_inverse(vs, Y)
    assert G.shape == (4, 2, 2) and X.shape == (4, n, 2)
    err = np.abs(np.einsum("nk,cnl->ckl", Y, X) - G).max() / np.abs(G).max()
    assert err < 3e-5, f"tutorial says 3e-6, got {err:.1e}"
