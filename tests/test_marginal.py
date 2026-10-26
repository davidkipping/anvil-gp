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


def test_a_collinear_basis_and_a_refused_chain_fail_cleanly():
    t, tc, y, yerr, shape = _setup()
    pm = ParamMap.log(KP)

    def twice(vm, x, m0):
        col = mx.ones_like(m0)
        return mx.stack([col, 2.0 * col], axis=-1)     # rank 1
    gp = GPLogLike(None, t, None, y, yerr, TERM, pm,
                   GPPolicy(metal=False, kappa_max=None), linear_basis=twice)
    with mx.stream(mx.gpu):
        out = np.array(gp(mx.zeros((1, 3))))
    assert out[0] <= -1e29

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
    noise variant.  And the gradient is polarisation's own -- the VJP
    differentiates it until the multi-column backward exists -- up to the
    cotangent, which the algebra downstream evaluates at the kernel's Gram
    matrix rather than polarisation's (~1e-5 relative)."""
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
