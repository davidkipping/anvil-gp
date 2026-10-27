"""Linear mean parameters integrated out: `GPLogLike(linear_basis=...)`.

The marginal comes from the Gram matrix ``[r Phi]^T K^-1 [r Phi]``, by
polarisation of the existing likelihood (`GPLogLike._marginal`).  Checked
against an independent route -- quadrature over beta of the *plain*
likelihood -- and against dense float64 formulas.
"""
import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
from anvilgp.oracle import dense_covariance, sample_ssm

N, CAD = 500, 120 / 86400
KP = [1e-3, 0.3, 1 / math.sqrt(2)]
TERM = ag.SHOTerm()


def _setup(seed=7, depth=0.01, beta=(0.02, 0.3)):
    rng = np.random.default_rng(seed)
    t = np.arange(N) * CAD
    tc = float(t.mean())
    yerr = np.full(N, 1e-4)
    shape = np.exp(-0.5 * ((t - tc) / (0.05 * np.ptp(t))) ** 2)
    m0 = -depth * shape
    phi = np.stack([1 + m0, (1 + m0) * (t - tc)], 1)
    y = (m0 + phi @ np.asarray(beta) + sample_ssm(TERM, KP, CAD, N, rng)
         + yerr * rng.standard_normal(N))
    return t, tc, y, yerr, shape


def _model(t, tc, shape):
    """A dip of free depth (the mean parameter, so m0 and Phi move with v)
    and the baseline (1 + m0)(F + G (t - tc)), linear in (F, G)."""
    shape_m = mx.array(shape)

    def mean_fn(vm, x):
        return -vm[:, :1] * shape_m.astype(vm.dtype)[None, :]

    def basis(vm, x, m0):
        return mx.stack([1 + m0, (1 + m0) * (x - tc)], axis=-1)
    return mean_fn, basis


def _gp(metal, prior=None, **kw):
    t, tc, y, yerr, shape = _setup(**kw)
    mean_fn, basis = _model(t, tc, shape)
    pm = ParamMap(kernel=[KernelParam(link="log", index=1 + i, ref=r)
                          for i, r in enumerate(KP)], n_dim=4,
                  mean=slice(0, 1))
    gp = GPLogLike(mean_fn, t, t, y, yerr, TERM, pm,
                   GPPolicy(metal=metal, kappa_max=None),
                   linear_basis=basis, linear_prior=prior)
    return gp, (t, tc, y, yerr, shape)


def _hi(gp, rows):
    with mx.stream(mx.cpu):
        return (np.array(gp.hi(mx.array(rows, dtype=mx.float64),
                               refuse=False)) + gp.log_norm_const)


def _dense_parts(row, data):
    t, tc, y, yerr, shape = data
    kp = np.asarray(KP) * np.exp(row[1:])
    m0 = -row[0] * shape
    phi = np.stack([1 + m0, (1 + m0) * (t - tc)], 1)
    K = dense_covariance(TERM, kp, t, yerr)
    return K, y - m0, phi


ROWS = np.array([[0.01, 0.0, 0.0, 0.0], [0.008, 0.1, -0.2, 0.05]])


def test_the_marginal_is_the_quadrature_of_the_plain_likelihood():
    """The acceptance test asked for: integrate the plain likelihood over
    beta numerically.  It is exactly Gaussian in beta, so Gauss-Hermite
    quadrature around the GLS estimate is exact to rounding; the marginal
    must match it to 1e-8 relative (p = 2, N = 500)."""
    gp, data = _gp(False)
    t, tc, y, yerr, shape = data
    beta_hat, cov = gp.linear_posterior(mx.array(ROWS))
    plain = GPLogLike(None, t, None, y, yerr, TERM, ParamMap.log(KP),
                      GPPolicy(metal=False, kappa_max=None))
    z, w = np.polynomial.hermite.hermgauss(5)
    for c, row in enumerate(ROWS):
        L = np.linalg.cholesky(cov[c])
        m0 = -row[0] * shape
        terms = []
        for i in range(5):
            for j in range(5):
                zz = np.array([z[i], z[j]])
                b = beta_hat[c] + math.sqrt(2) * L @ zz
                resid = y - m0 - np.stack([1 + m0, (1 + m0) * (t - tc)], 1) @ b
                p = GPLogLike(None, t, None, resid, yerr, TERM,
                              ParamMap.log(KP), GPPolicy(metal=False))
                with mx.stream(mx.cpu):
                    ll = float(np.array(p.hi(mx.array(row[None, 1:],
                                                      dtype=mx.float64)))[0])
                terms.append(math.log(w[i] * w[j]) + ll + p.log_norm_const
                             + zz @ zz)
        lz = (np.log(np.linalg.det(L)) + math.log(2.0)
              + np.logaddexp.reduce(terms))
        got = _hi(gp, row[None])[0]
        assert abs(got - lz) < 1e-8 * abs(lz), (got, lz)


def test_a_gaussian_prior_matches_the_dense_marginal():
    """With beta ~ N(mu, S): r ~ N(Phi mu, K + Phi S Phi^T)."""
    mu, S = np.array([0.0, 0.2]), np.diag([0.05 ** 2, 0.5 ** 2])
    gp, data = _gp(False, prior=(mu, S))
    got = _hi(gp, ROWS)
    for c, row in enumerate(ROWS):
        K, r, phi = _dense_parts(row, data)
        cov = K + phi @ S @ phi.T
        d = r - phi @ mu
        sign, logdet = np.linalg.slogdet(cov)
        ref = -0.5 * (d @ np.linalg.solve(cov, d) + logdet
                      + N * math.log(2 * math.pi))
        assert abs(got[c] - ref) < 1e-8 * abs(ref)


@pytest.mark.parametrize("prior", [None, "gauss"], ids=["flat", "gauss"])
def test_linear_posterior_matches_dense(prior):
    mu, S = np.array([0.0, 0.2]), np.diag([0.05 ** 2, 0.5 ** 2])
    gp, data = _gp(False, prior=None if prior is None else (mu, S))
    mean, cov = gp.linear_posterior(mx.array(ROWS))
    for c, row in enumerate(ROWS):
        K, r, phi = _dense_parts(row, data)
        C = phi.T @ np.linalg.solve(K, phi)
        b = np.linalg.solve(C, phi.T @ np.linalg.solve(K, r))
        if prior is None:
            want_m, want_c = b, np.linalg.inv(C)
        else:
            want_c = np.linalg.inv(C + np.linalg.inv(S))
            want_m = want_c @ (C @ b + np.linalg.solve(S, mu))
        np.testing.assert_allclose(mean[c], want_m, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(cov[c], want_c, rtol=1e-7, atol=1e-18)


def test_the_marginal_ignores_the_basis_amplitude():
    """Flat prior: adding ``Phi b`` to the data for any b leaves the
    marginal unchanged -- the property the float32 centring relies on.
    Only for the model's own basis: row 0 has the dip depth the data were
    built with.  (A huge shift, so float64 cancellation sets the 1e-8.)"""
    a, _ = _gp(False, beta=(0.02, 0.3))
    b, _ = _gp(False, beta=(5.0, -40.0))
    np.testing.assert_allclose(_hi(a, ROWS[:1]), _hi(b, ROWS[:1]), rtol=1e-8)


@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_float32_value_and_gradient(metal):
    """float32 against float64, value and gradient, including the mean
    parameter, which moves the residual and the basis (through m0)."""
    gp, _ = _gp(metal)
    assert gp._metal_ok is metal
    hi = _hi(gp, ROWS)
    with mx.stream(mx.gpu):
        u = mx.array(ROWS, dtype=mx.float32)
        lo = np.array(gp(u)) + gp.log_norm_const
        g = np.array(mx.grad(lambda u: gp(u).sum())(u))
    G = np.zeros_like(ROWS)
    for i in range(4):
        h = 1e-7 if i == 0 else 1e-6
        dp = np.zeros(4)
        dp[i] = h
        G[:, i] = (_hi(gp, ROWS + dp) - _hi(gp, ROWS - dp)) / (2 * h)
    assert np.abs(lo - hi).max() < 5e-3
    # The MLX path's float32 is the less accurate one (its value here is
    # off by ~5e-3 against the kernel's ~1e-3); its mean-parameter
    # derivative measured 1.4e-4 relative.
    tol = 1e-4 if metal else 1e-3
    assert np.abs(g - G).max() / np.abs(G).max() < tol


@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_a_collinear_basis_and_a_refused_chain_fail_cleanly(metal):
    """Under the flat prior a rank-deficient basis is an improper model:
    the failure value, on both routes and in float64."""
    t, tc, y, yerr, shape = _setup()
    pm = ParamMap.log(KP)

    def twice(vm, x, m0):
        col = mx.ones_like(m0)
        return mx.stack([col, 2.0 * col], axis=-1)     # rank 1
    gp = GPLogLike(None, t, None, y, yerr, TERM, pm,
                   GPPolicy(metal=metal, kappa_max=None), linear_basis=twice)
    with mx.stream(mx.gpu):
        out = np.array(gp(mx.zeros((1, 3))))
    assert out[0] <= -1e29
    assert np.array(_hi(gp, np.zeros((1, 3))))[0] <= -1e29

    def ok(vm, x, m0):
        return mx.stack([mx.ones_like(m0),
                         mx.broadcast_to(x - tc, m0.shape)], axis=-1)
    gp = GPLogLike(None, t, None, y, yerr, TERM, pm, GPPolicy(metal=True),
                   linear_basis=ok)
    v = mx.array([[0.0, 0.0, 0.0], [20.0, 30.0, 0.0]])  # 2nd: kappa >> max
    with mx.stream(mx.gpu):
        out = np.array(gp(v))
        g = np.array(mx.grad(lambda u: gp(u).sum())(v))
    assert np.isfinite(out[0]) and out[0] > -1e20 and out[1] <= -1e29
    assert np.all(np.isfinite(g))


def _singular_basis(kind, t, tc):
    """Rank-deficient bases that a proper prior makes well defined."""
    inst = (np.arange(t.size) % 3 == 0).astype(np.float64)
    if kind == "zero-column":
        cols = np.stack([np.ones_like(t), np.zeros_like(t), t - tc], 1)
    else:                       # a global level beside one offset each
        cols = np.stack([np.ones_like(t), inst, 1.0 - inst], 1)
    return cols


@pytest.mark.parametrize("kind", ["zero-column", "level+offsets"])
def test_a_gaussian_prior_makes_a_singular_basis_well_defined(kind):
    """``C = Phi^T K^-1 Phi`` singular, the prior proper: the marginal is
    the dense ``N(Phi mu, K + Phi S Phi^T)`` on every path, the posterior
    the dense one, and the float32 gradient that of float64.  Until
    2026-10-26 the Gaussian prior was formed through ``C^-1``: the zero
    column got the failure value, and the level + offsets case a finite
    WRONG value (240 log-units on the kernel, 3.7 in float64).  A zero
    column is exactly zero in float32 too, so float32 resolves it; exact
    collinearity, since 2026-10-27, because a shared basis is
    orthonormalised in float64 (`GPLogLike._cache_shared_basis`)."""
    rng = np.random.default_rng(11)
    t = np.arange(N) * CAD
    tc, yerr, shape = float(t.mean()), np.full(N, 1e-4), None
    cols = _singular_basis(kind, t, tc)
    y = (cols @ np.array([0.01, 0.003, -0.002])
         + sample_ssm(TERM, KP, CAD, N, rng) + yerr * rng.standard_normal(N))
    mu, S = np.array([0.0, 0.0, 0.0]), np.diag([0.02, 0.01, 0.01]) ** 2

    def basis(vm, x, m0):
        return mx.array(cols, dtype=m0.dtype)
    ref = []
    for row in ROWS[:, 1:]:
        K = dense_covariance(TERM, np.asarray(KP) * np.exp(row), t, yerr)
        cov = K + cols @ S @ cols.T
        d = y - cols @ mu
        ref.append(-0.5 * (d @ np.linalg.solve(cov, d)
                           + np.linalg.slogdet(cov)[1]
                           + N * math.log(2 * math.pi)))
    ref = np.array(ref)
    for metal in (False, True):
        gp = GPLogLike(None, t, None, y, yerr, TERM, ParamMap.log(KP),
                       GPPolicy(metal=metal, kappa_max=None),
                       linear_basis=basis, linear_prior=(mu, S))
        with mx.stream(mx.gpu):
            u = mx.array(ROWS[:, 1:], dtype=mx.float32)
            lo = np.array(gp(u)).astype(np.float64) + gp.log_norm_const
            g = np.array(mx.grad(lambda u: gp(u).sum())(u))
        hi = _hi(gp, ROWS[:, 1:])
        np.testing.assert_allclose(hi, ref, rtol=1e-10)
        # Exactly collinear (level + offsets) was 4.6 log-units off on the
        # kernel in float32, then refused; a shared basis is now
        # orthonormalised in float64 first, so the redundant direction is
        # an exactly zero column the prior carries.
        tol = 2e-3 if kind == "zero-column" else 1e-2
        assert np.abs(lo - ref).max() < tol, (metal, np.abs(lo - ref))
        G = np.zeros_like(ROWS[:, 1:])
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = 1e-6
            G[:, i] = (_hi(gp, ROWS[:, 1:] + dp)
                       - _hi(gp, ROWS[:, 1:] - dp)) / 2e-6
        assert np.abs(g - G).max() / np.abs(G).max() < 1e-3, metal
    mean, pcov = gp.linear_posterior(mx.array(ROWS[:, 1:]))
    for c, row in enumerate(ROWS[:, 1:]):
        K = dense_covariance(TERM, np.asarray(KP) * np.exp(row), t, yerr)
        C = cols.T @ np.linalg.solve(K, cols)
        want_c = np.linalg.inv(C + np.linalg.inv(S))
        want_m = want_c @ (cols.T @ np.linalg.solve(K, y)
                           + np.linalg.solve(S, mu))
        np.testing.assert_allclose(mean[c], want_m, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(pcov[c], want_c, rtol=1e-8, atol=1e-16)


def test_a_bad_linear_prior_is_refused():
    t, tc, y, yerr, shape = _setup()

    def basis(vm, x, m0):
        return mx.ones_like(m0)[..., None]
    for prior, match in [((np.zeros(1), -np.eye(1)), "positive definite"),
                         ((np.zeros(2), np.eye(3)), "linear_prior must"),
                         ((np.zeros(2), np.eye(2)), "p = 1")]:
        with pytest.raises(ValueError, match=match):
            gp = GPLogLike(None, t, None, y, yerr, TERM, ParamMap.log(KP),
                           GPPolicy(metal=False), linear_basis=basis,
                           linear_prior=prior)
            _hi(gp, np.zeros((1, 3)))


@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_a_shared_basis_and_a_per_chain_one(metal):
    """The same basis returned as (N, p) -- cached, orthonormalised in
    float64 (`GPLogLike._cache_shared_basis`) and on the kernel a separate,
    chain-stride-0 input with no cotangent (`metal._col_layout`) -- and per
    chain, (n, N, p), evaluated every time: the same model, each within its
    route's float32 floor of float64 in value and gradient.  (They take
    different routes, so they agree with each other only to that floor:
    measured 3.0e-4 / 1.4e-3 in value on the kernel, 2.5e-2 / 1.7e-2 on the
    MLX path, the gradients 2-3e-6 and 1.2-1.6e-4.)"""
    t, tc, y, yerr, shape = _setup()

    def shared(vm, x, m0):
        return mx.stack([mx.ones_like(x), x - tc], axis=-1)

    def per_chain(vm, x, m0):
        return mx.broadcast_to(shared(vm, x, m0), m0.shape + (2,))
    rows = ROWS[:, 1:]
    his = []
    for basis in (shared, per_chain):
        gp = GPLogLike(None, t, None, y, yerr, TERM, ParamMap.log(KP),
                       GPPolicy(metal=metal, kappa_max=None),
                       linear_basis=basis)
        assert (gp._basis_q64 is not None) is (basis is shared)
        with mx.stream(mx.gpu):
            u = mx.array(rows, dtype=mx.float32)
            lo = np.array(gp(u)).astype(np.float64) + gp.log_norm_const
            g = np.array(mx.grad(lambda u: gp(u).sum())(u))
        hi = _hi(gp, rows)
        his.append(hi)
        G = np.zeros_like(rows)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = 1e-6
            G[:, i] = (_hi(gp, rows + dp) - _hi(gp, rows - dp)) / 2e-6
        assert np.abs(lo - hi).max() < (5e-3 if metal else 5e-2)
        assert np.abs(g - G).max() / np.abs(G).max() < (1e-4 if metal
                                                        else 1e-3)
    # float64: the same model, the orthonormalisation exact
    np.testing.assert_allclose(his[0], his[1], rtol=1e-10)


@pytest.mark.parametrize("t0", [3000.0, 2459000.0], ids=["t~3e3", "BJD"])
@pytest.mark.parametrize("prior", [None, "gauss"], ids=["flat", "gauss"])
def test_a_raw_time_baseline(t0, prior):
    """``[1, t]`` in raw times, shared: ``1 - rho^2`` is 3.7e-6 at t ~ 3000
    and 1e-11 for a BJD, which float32 refused (and before 2026-10-26 got
    0.16-5.8 log-units wrong).  Orthonormalised in float64 at
    construction, it is now the model with ``[1, t - tc]`` exactly (the
    map has unit determinant; a Gaussian prior is mapped to match), on
    every path, gradient included, and its posterior is in raw units.  A
    per-chain raw-time basis is evaluated in float32 every time and is
    still refused."""
    if prior == "gauss" and t0 > 1e4:
        pytest.skip("no float64 reference: a prior on the raw intercept (the "
                    "level at t = 0) is a centred prior of correlation 1 - "
                    "1e-12, which the reference model cannot invert")
    t, tc0, y, yerr, shape = _setup()
    tr = t0 + t
    tc = float(tr.mean())
    # broad in raw coordinates, as a raw-time prior would be: intercept at
    # t = 0 within 100, slope within 1 per day
    mu, S = np.array([0.0, 0.0]), np.diag([100.0, 1.0]) ** 2
    # the same prior on the centred coefficients: (a + b tc, b)
    a_inv = np.array([[1.0, tc], [0.0, 1.0]])
    lp_raw = None if prior is None else (mu, S)
    lp_c = None if prior is None else (a_inv @ mu, a_inv @ S @ a_inv.T)
    raw = GPLogLike(None, tr, None, y, yerr, TERM, ParamMap.log(KP),
                    GPPolicy(metal=True, kappa_max=None), linear_prior=lp_raw,
                    linear_basis=lambda vm, x, m0: mx.stack(
                        [mx.ones_like(x), x], -1))
    cen = GPLogLike(None, tr, None, y, yerr, TERM, ParamMap.log(KP),
                    GPPolicy(metal=True, kappa_max=None), linear_prior=lp_c,
                    linear_basis=lambda vm, x, m0: mx.stack(
                        [mx.ones_like(x), x - tc], -1))
    rows = ROWS[:, 1:]
    ref = _hi(cen, rows)
    # float64 carries a BJD to ~2.5e-10 d, over this fixture's 0.69-d span
    # ~1e-9 of the slope column, in any parametrisation: measured 4e-9
    np.testing.assert_allclose(_hi(raw, rows), ref,
                               rtol=1e-10 if t0 < 1e4 else 1e-8)
    for metal in (True, False):
        raw.policy.metal = metal
        raw._metal_ok = metal
        with mx.stream(mx.gpu):
            u = mx.array(rows, dtype=mx.float32)
            lo = np.array(raw(u)).astype(np.float64) + raw.log_norm_const
            g = np.array(mx.grad(lambda u: raw(u).sum())(u))
        assert np.abs(lo - ref).max() < (5e-3 if metal else 5e-2), metal
        G = np.zeros_like(rows)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = 1e-6
            G[:, i] = (_hi(cen, rows + dp) - _hi(cen, rows - dp)) / 2e-6
        assert np.abs(g - G).max() / np.abs(G).max() < (1e-4 if metal
                                                        else 1e-3), metal
    b_raw, c_raw = raw.linear_posterior(rows)
    b_cen, c_cen = cen.linear_posterior(rows)
    # beta_raw = A beta_cen, A = [[1, -tc], [0, 1]]
    A = np.linalg.inv(a_inv)
    np.testing.assert_allclose(b_raw, b_cen @ A.T, rtol=1e-6,
                               atol=1e-9 * np.abs(b_raw).max())
    np.testing.assert_allclose(c_raw, A @ c_cen @ A.T, rtol=1e-6,
                               atol=1e-9 * np.abs(c_raw).max())
    if t0 == 3000.0 and prior is None:
        per_chain = GPLogLike(
            None, tr, None, y, yerr, TERM, ParamMap.log(KP),
            GPPolicy(metal=True, kappa_max=None),
            linear_basis=lambda vm, x, m0: mx.stack(
                [mx.ones_like(m0), mx.broadcast_to(x, m0.shape)], -1))
        with mx.stream(mx.gpu):
            assert np.all(np.array(per_chain(mx.array(
                rows, dtype=mx.float32))) <= -1e29)


def test_make_target_marginalises():
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    from anvilgp.priors import SpecHint
    t, tc, y, yerr, shape = _setup()
    mean_fn, basis = _model(t, tc, shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = make_target(TERM, t, y, yerr, mean_fn=mean_fn, x_mean=t,
                         mean_specs=[SpecHint("depth", lo=0.0, hi=0.05)],
                         linear_basis=basis)
    u0 = gt.u0(6, seed=1)
    lp = gt.target.log_prob(u0)
    g = mx.grad(lambda u: gt.target.log_prob(u).sum())(u0)
    assert np.all(np.isfinite(np.array(lp))) and np.all(np.isfinite(np.array(g)))
    mean, cov = gt.linear_posterior(np.array(gt.transform.to_model(u0)))
    assert mean.shape == (6, 2) and cov.shape == (6, 2, 2)
    assert np.all(np.abs(mean[:, 0] - 0.02) < 0.02)


@pytest.mark.slow
def test_marginalised_and_sampled_posteriors_agree():
    """The calibration check: sampling (depth, GP) with beta integrated
    out gives the same posterior as sampling beta too under a wide flat
    box.  anvil's ensemble sampler on both; posterior means of the shared
    parameters within 0.2 posterior sd (well over the Monte Carlo error at
    this length)."""
    anvil = pytest.importorskip("anvil")
    from anvil.kernels.ensemble import EnsembleKernel
    from anvilgp.anvil import make_target
    from anvilgp.priors import SpecHint
    t, tc, y, yerr, shape = _setup(seed=11)
    t, y, yerr, shape = t[::2], y[::2], yerr[::2] * 30, shape[::2]
    mean_fn, basis = _model(t, tc, shape)
    shape_m = mx.array(shape)
    tt = mx.array(t - tc)

    def full_mean(vm, x):                      # depth, F, G all sampled
        m0 = -vm[:, :1] * shape_m.astype(vm.dtype)[None, :]
        return m0 + (1 + m0) * (vm[:, 1:2] + vm[:, 2:3] * tt.astype(vm.dtype))
    depth = SpecHint("depth", lo=0.0, hi=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        marg = make_target(TERM, t, y, yerr, mean_fn=mean_fn, x_mean=t,
                           mean_specs=[depth], linear_basis=basis)
        samp = make_target(TERM, t, y, yerr, mean_fn=full_mean, x_mean=t,
                           mean_specs=[depth, SpecHint("F", lo=-1.0, hi=1.0),
                                       SpecHint("G", lo=-10.0, hi=10.0)])
    out = {}
    for name, gt in (("marg", marg), ("samp", samp)):
        u0 = gt.u0(128, seed=1)
        if name == "samp":                     # start beta near its GLS value
            b, _ = marg.linear_posterior(np.array(
                marg.transform.to_model(marg.u0(1, seed=1))))
            v0 = np.array(gt.transform.to_model(u0))
            v0[:, 1:3] = b[0] + 1e-3 * np.random.default_rng(2).standard_normal((128, 2))
            u0 = mx.array(gt.transform.from_model_np(v0).astype(np.float32))
        res = anvil.run(EnsembleKernel(gt.target, seed=2), gt.target, u0,
                        n_warmup=1500, n_samples=1500, seed=3, progress=False)
        v = np.array(gt.transform.model_np(
            res.get_chain(flat=True).astype(np.float64)))
        cols = [0] + [gt.names.index(k) for k in
                      ("log_gp_sigma", "log_gp_rho", "log_jitter")]
        out[name] = v[:, cols]
    m, s = out["marg"], out["samp"]
    sd = np.sqrt(0.5 * (m.var(0) + s.var(0)))
    assert np.all(np.abs(m.mean(0) - s.mean(0)) < 0.2 * sd), (
        (m.mean(0) - s.mean(0)) / sd)


# ---- the multi-column filter on the kernel (`metal.scan_gram`) -------------

def _cols64(row, data):
    """Columns the size `_marginal` gives them: a centred residual and a
    scaled basis."""
    K, r, phi = _dense_parts(row, data)
    rc = r - phi @ np.linalg.lstsq(phi, r, rcond=None)[0]
    return K, np.column_stack([rc, phi * 1e-2])


@pytest.mark.parametrize("boundary", ["dense", "sqrt"])
@pytest.mark.parametrize("blk", [512, 64], ids=["one-block", "blocks"])
def test_the_multi_column_filter_is_the_dense_gram(boundary, blk):
    """``Z^T K^-1 Z`` and ``-1/2 log|K|`` (relative to ``r_ref``) from one
    filter carrying three columns, against dense float64 -- one block, and
    eight through either boundary, whose entry means are per column."""
    from anvilgp.metal import scan_gram
    gp, data = _gp(True)
    gp.metal_block = blk
    yerr = data[3]
    parts = [_cols64(row, data) for row in ROWS]
    cols = mx.array(np.stack([Z.T for _, Z in parts]).astype(np.float32))
    pt, qt, h, rr, jit2, _ = gp._metal_inputs(mx.array(ROWS, dtype=mx.float32),
                                             cols)
    ld, G = scan_gram(pt, qt, gp._metal_lag, gp._metal_inst, h, rr,
                      gp._r_ref32, jit2, gp.policy.s_floor_rel, blk,
                      boundary=boundary)
    ld, G = np.array(ld), np.array(G)
    for c, (K, Z) in enumerate(parts):
        want = Z.T @ np.linalg.solve(K, Z)
        scale = np.sqrt(np.outer(np.diag(want), np.diag(want)))
        assert np.abs(G[c] - want).max() / scale.max() < 2e-5
        np.testing.assert_array_equal(G[c], G[c].T)
        ld_want = -0.5 * (np.linalg.slogdet(K)[1] - np.sum(np.log(yerr ** 2)))
        assert abs(ld[c] - ld_want) < 2e-4


def _scale_jitter(vn, d):
    from anvilgp.special import exp_acc
    return (exp_acc(vn[:, :1]) * d.yerr) ** 2 + exp_acc(2.0 * vn[:, 1:2])


_GRAM_CASES = {
    "sho": (TERM, KP, "none"),
    "sho-jitter": (TERM, KP, "jitter"),
    "sho2-noise_fn": (TERM + ag.SHOTerm(), KP + [5e-4, 0.1, 2.0], "noise_fn"),
    "rot+sho": (ag.RotationTerm() + ag.SHOTerm(),
                [1e-3, 0.3, 3.0, 1.0, 0.4, 5e-4, 0.1, 0.7], "none"),
}


def _gp_case(name, metal_block=None):
    term, kp, noise = _GRAM_CASES[name]
    t, tc, y, yerr, shape = _setup()
    mean_fn, basis = _model(t, tc, shape)
    k = len(kp)
    kern = [KernelParam(link="log", index=1 + i, ref=r)
            for i, r in enumerate(kp)]
    extra = {"jitter": 1, "noise_fn": 2}.get(noise, 0)
    pm = ParamMap(kernel=kern, n_dim=1 + k + extra, mean=slice(0, 1),
                  jitter=([KernelParam(link="log", index=1 + k, ref=1e-4)]
                          if noise == "jitter" else []),
                  noise=(slice(1 + k, 3 + k) if noise == "noise_fn"
                         else None))
    gp = GPLogLike(mean_fn, t, t, y, yerr, term, pm,
                   GPPolicy(metal=True, kappa_max=None), linear_basis=basis,
                   noise_fn=_scale_jitter if noise == "noise_fn" else None)
    if metal_block is not None:
        gp.metal_block = metal_block
    rows = np.zeros((2, pm.n_dim))
    rows[:, 0] = [0.01, 0.008]
    rows[1, 1:1 + k] = 0.05 * np.cos(np.arange(k))
    if noise == "noise_fn":
        rows[:, 1 + k:] = [[0.0, -9.0], [0.1, -8.5]]
    return gp, rows


def _polarised(gp, v):
    return gp._marginal(v, gp._raw_resid, gp._y32, gp._x_mean32,
                        gp._noise_x32, gp._r_ref32)[0]


@pytest.mark.parametrize("blk", [None, 64], ids=["auto", "blocks"])
@pytest.mark.parametrize("name", list(_GRAM_CASES))
def test_the_kernel_gram_is_the_polarised_marginal(name, blk):
    """The marginal on the kernel, by one multi-column filter, against the
    float64 path and against the polarisation it replaces: as close to
    float64 as polarisation is, or closer, for m = 2, 4 and 6 and every
    noise variant.  And the gradient (`scan_gram`'s own VJP) matches
    polarisation's to float32 rounding, the algebra downstream evaluated at
    each route's own Gram matrix."""
    gp, rows = _gp_case(name, blk)
    hi = np.array(gp.hi(mx.array(rows, dtype=mx.float64), refuse=False))
    with mx.stream(mx.gpu):
        v = mx.array(rows, dtype=mx.float32)
        new = np.array(gp._raw_call(v))
        old = np.array(_polarised(gp, v))
        g_new = np.array(mx.grad(lambda u: gp._raw_call(u).sum())(v))
        g_old = np.array(mx.grad(lambda u: _polarised(gp, u).sum())(v))
    assert np.abs(new - hi).max() < max(2e-3, 1.5 * np.abs(old - hi).max())
    assert np.abs(g_new - g_old).max() < 1e-4 * np.abs(g_old).max()


def test_past_the_column_cap_the_kernel_polarises(monkeypatch):
    import anvilgp.likelihood as lk
    from anvilgp import metal
    gp, rows = _gp_case("sho")
    v = mx.array(rows, dtype=mx.float32)
    want = np.array(_polarised(gp, v))
    monkeypatch.setattr(lk, "MAX_GRAM_COLS", 2)          # k = 3 here

    def boom(*a, **k):
        raise AssertionError("scan_gram called past the cap")
    monkeypatch.setattr(metal, "scan_gram", boom)
    np.testing.assert_array_equal(np.array(gp._raw_call(v)), want)


@pytest.mark.parametrize("shared", [False, True], ids=["per-chain", "shared"])
@pytest.mark.parametrize("p", [1, 3, 10])
def test_the_normal_equations_kernel(p, shared):
    """The centring fit's reduction kernel against float64, for a basis per
    chain and one shared by every chain (read once, not broadcast)."""
    from anvilgp.metal import normal_equations
    rng = np.random.default_rng(p)
    n, N = 5, 1777                       # not a multiple of the threadgroup
    phi = rng.standard_normal((1 if shared else n, p, N))
    r, w = rng.standard_normal((n, N)), rng.uniform(0.5, 2.0, N)
    with mx.stream(mx.gpu):
        A, b = normal_equations(mx.array(phi.astype(np.float32)),
                                mx.array(r.astype(np.float32)),
                                mx.array(w.astype(np.float32)))
    P = np.broadcast_to(phi, (n, p, N))
    A_ref = np.einsum("cat,t,cbt->cab", P, w, P)
    b_ref = np.einsum("cat,t,ct->ca", P, w, r)
    np.testing.assert_allclose(np.array(A), A_ref, rtol=0,
                               atol=1e-5 * np.abs(A_ref).max())
    np.testing.assert_allclose(np.array(b), b_ref, rtol=0,
                               atol=1e-5 * np.abs(b_ref).max())


@pytest.mark.parametrize("p", [1, 2, 5, 12])
def test_the_small_inverse_and_its_vjp(p):
    """`linalg.inv_logdet_fast`: the kernel's inverse and log-determinant
    against float64, and the analytic VJP against differentiating the
    unrolled elimination it replaces.  Indefinite matrices included (the
    pivoting is what makes those work); singular gives non-finite."""
    from anvilgp.linalg import inv_logdet, inv_logdet_fast
    rng = np.random.default_rng(p)
    x = rng.standard_normal((7, p, p)) + 2 * np.eye(p)
    wt = mx.array(rng.standard_normal((p, p)).astype(np.float32))

    def loss(fn):
        return lambda z: ((fn(z)[0] * wt).sum() + 0.3 * fn(z)[1].sum())
    with mx.stream(mx.gpu):
        xm = mx.array(x.astype(np.float32))
        inv, ld = inv_logdet_fast(xm)
        g = np.array(mx.grad(loss(inv_logdet_fast))(xm))
        g_ref = np.array(mx.grad(loss(inv_logdet))(xm))
        sing = inv_logdet_fast(mx.ones((2, p, p)) if p > 1
                               else mx.zeros((2, 1, 1)))
    np.testing.assert_allclose(np.array(inv), np.linalg.inv(x), rtol=1e-4,
                               atol=1e-5)
    np.testing.assert_allclose(np.array(ld), np.linalg.slogdet(x)[1],
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(g, g_ref, rtol=1e-3,
                               atol=1e-4 * np.abs(g_ref).max())
    assert not np.all(np.isfinite(np.array(sing[0])))


_VJP_CASES = [
    # (term, kernel refs, noise, metal_block, boundary)
    ("sho", "none", None, None),
    ("sho", "jitter", 64, "sqrt"),
    ("sho", "noise_fn", 64, "dense"),
    ("sho2", "noise_fn", 64, "sqrt"),
    ("rot+sho", "jitter", 64, "sqrt"),
    ("conv", "none", None, None),
    ("conv", "jitter", 64, "dense"),
]
_VJP_TERMS = {
    "sho": (TERM, [1.0, 2.0, 3.0]),
    "sho2": (TERM + ag.SHOTerm(), [1.0, 2.0, 3.0, 0.5, 0.7, 1.5]),
    "rot+sho": (ag.RotationTerm() + ag.SHOTerm(),
                [1.0, 4.0, 3.0, 1.0, 0.4, 0.3, 0.5, 0.7]),
    "conv": (ag.TermConvolution(TERM, 0.01), [1.0, 2.0, 3.0]),
}


@pytest.mark.parametrize("name,noise,blk,boundary", _VJP_CASES,
                         ids=["-".join(map(str, c)) for c in _VJP_CASES])
def test_the_multi_column_vjp_is_the_polarised_one(name, noise, blk,
                                                    boundary):
    """`scan_gram`'s VJP -- the multi-column backward from generalised block
    seeds -- against the same functional, ``c_ld * ld + sum W * G`` with
    random cotangents, built by polarisation through `scan_loglike`'s
    verified backward: gradients with respect to the parameters and to the
    columns, for m = 2, 4, 6 and a convolved term, every noise variant, one
    block and many through either boundary.  Measured 1e-8-8e-7 relative;
    gated at 1e-5."""
    from anvilgp.metal import scan_gram
    term, kp = _VJP_TERMS[name]
    rng = np.random.default_rng(0)
    n, N, k = 3, 600, 3
    t = (np.arange(N) * 0.02 if name == "conv"
         else np.sort(rng.uniform(0, 20, N)))
    yerr = rng.uniform(0.2, 0.4, N)
    nk = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(kp)],
                  n_dim=nk + {"jitter": 1, "noise_fn": 2}.get(noise, 0),
                  jitter=([KernelParam(link="log", index=nk, ref=0.05)]
                          if noise == "jitter" else []),
                  noise=slice(nk, nk + 2) if noise == "noise_fn" else None)
    gp = GPLogLike(None, t, None, rng.standard_normal(N), yerr, term, pm,
                   GPPolicy(metal=True, kappa_max=None),
                   noise_fn=_scale_jitter if noise == "noise_fn" else None)
    if blk is not None:
        gp.metal_block = blk
    if boundary is not None:
        gp._metal_boundary = boundary
    v = 0.1 * rng.standard_normal((n, pm.n_dim))
    if noise == "noise_fn":
        v[:, nk:] = [0.0, -3.0]
    v = mx.array(v.astype(np.float32))
    cols = mx.array(rng.standard_normal((n, k, N)).astype(np.float32))
    W = mx.array(rng.standard_normal((n, k, k)).astype(np.float32))
    cl = mx.array(rng.standard_normal(n).astype(np.float32))
    slots = (gp._metal_slot_id, gp._metal_slot_start, gp.n_metal_slots,
             gp._metal_fold_plan)

    def via_gram(v, cols):
        pt, qt, h, r, j2, _ = gp._metal_inputs(v, cols)
        ld, G = scan_gram(pt, qt, gp._metal_lag, gp._metal_inst, h, r,
                          gp._r_ref32, j2, gp.policy.s_floor_rel,
                          gp.metal_block, slots=slots, formal=gp._formal,
                          boundary=gp._metal_boundary)
        return mx.sum(cl * ld) + mx.sum(W * G)

    def via_polar(v, cols):
        ell0, G, _ = gp._gram_polarised(
            v, lambda vv, rr: gp._evaluate_metal(vv, rr)[0], cols)
        return mx.sum(cl * ell0) + mx.sum(W * G)
    with mx.stream(mx.gpu):
        got = mx.grad(via_gram, argnums=(0, 1))(v, cols)
        want = mx.grad(via_polar, argnums=(0, 1))(v, cols)
    for a, b in zip(got, want):
        a, b = np.array(a), np.array(b)
        assert np.abs(a - b).max() < 1e-5 * np.abs(b).max()


def test_the_multi_column_gradient_in_the_hard_regime():
    """Bright, long-timescale data over 16 blocks -- where the single
    column's closed-form seed loses most to float32 (docs/m4-plan.md) --
    the marginal's kernel gradient against float64 finite differences.
    The multi-column seeds take ``r`` from a linear suffix of the actual
    innovations, not ``eta_s - J_s mu``.  Measured 2.1e-6 relative
    (polarisation 3.7e-6)."""
    from anvilgp.oracle import sample_ssm as ssm
    N, kp = 8192, [100.0, 1000.0, 3.0]
    rng = np.random.default_rng(0)
    t = np.arange(N, dtype=float)
    y = ssm(TERM, kp, 1.0, N, rng) + rng.standard_normal(N) + 5.0 + 1e-3 * t
    tt = mx.array((t - t.mean()) / t.std())

    def basis(vm, x, m0):
        b = mx.stack([mx.ones_like(tt), tt], -1).astype(m0.dtype)
        return mx.broadcast_to(b, (m0.shape[0], N, 2))
    gp = GPLogLike(None, t, None, y, np.ones(N), TERM, ParamMap.log(kp),
                   GPPolicy(metal=True, kappa_max=None), linear_basis=basis)
    gp.metal_block = 512
    rows = np.array([[0.0, 0.0, 0.0], [0.05, -0.1, 0.1]])
    with mx.stream(mx.gpu):
        g = np.array(mx.grad(lambda u: gp(u).sum())(
            mx.array(rows, dtype=mx.float32)))
    G = np.zeros_like(rows)
    for i in range(3):
        d = np.zeros(3)
        d[i] = 1e-5
        with mx.stream(mx.cpu):
            hp = np.array(gp.hi(mx.array(rows + d, dtype=mx.float64),
                                refuse=False))
            hm = np.array(gp.hi(mx.array(rows - d, dtype=mx.float64),
                                refuse=False))
        G[:, i] = (hp - hm) / 2e-5
    assert np.abs(g - G).max() < 2e-5 * np.abs(G).max()


def test_linear_posterior_is_chunked_over_chains(monkeypatch):
    """Chunked over chains by `predict._CHUNK_BYTES` (in one batch the old
    polarised route asked for ~440 GiB for a whole posterior at N = 2000):
    one chain a chunk gives the same answer as one chunk."""
    import anvilgp.predict as pr
    gp, _ = _gp(False)
    rows = np.tile(ROWS, (3, 1))
    whole = gp.linear_posterior(rows)
    monkeypatch.setattr(pr, "_CHUNK_BYTES", 1)
    tiled = gp.linear_posterior(rows)
    for a, b in zip(whole, tiled):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_linear_posterior_of_a_singular_basis():
    """Flat prior: no posterior, NaN for that chain (the others are
    untouched).  Gaussian prior: the dense one (covered above)."""
    t, tc, y, yerr, shape = _setup()

    def twice(vm, x, m0):
        col = mx.ones_like(m0)
        return mx.stack([col, 2.0 * col], axis=-1)     # rank 1
    gp = GPLogLike(None, t, None, y, yerr, TERM, ParamMap.log(KP),
                   GPPolicy(metal=False), linear_basis=twice)
    mean, cov = gp.linear_posterior(np.zeros((2, 3)))
    assert np.all(np.isnan(mean)) and np.all(np.isnan(cov))
