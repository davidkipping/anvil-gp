"""Held-out ("masked") conditioning: `GPLogLike(held_out=...)`.

The GP is conditioned on S and applied to the held-out M through its
two-sided conditional mean, M scored with its white noise alone
(docs/masked-gp-plan.md).  Every case against ``Sigma``, the joint
Gaussian that defines it, built densely (`oracle.dense_masked_covariance`).
"""
import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
from anvilgp.metal import metal_available
from anvilgp.oracle import (dense_covariance, dense_loglike,
                            dense_masked_covariance, dense_masked_loglike,
                            sample_ssm)
from anvilgp.special import exp_acc

N = 300
_TERMS = {
    "real": (ag.RealTerm(), [1.0, 0.7]),
    "sho": (ag.SHOTerm(), [1.0, 2.0, 3.0]),
    "matern52": (ag.Matern52Term(), [1.0, 1.5]),
    "rot+sho": (ag.RotationTerm() + ag.SHOTerm(),
                [1.0, 4.0, 3.0, 1.0, 0.4, 0.3, 0.5, 0.7]),
}


def _data(term, kp, t=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 20, N)) if t is None else t
    yerr = rng.uniform(0.1, 0.3, t.size)
    grid = np.linspace(t.min(), t.max(), 4000)
    y = (np.interp(t, grid, sample_ssm(term, kp, grid[1] - grid[0],
                                       grid.size, rng))
         + yerr * rng.standard_normal(t.size))
    return t, y, yerr


def _layout(name, t):
    """Interior; leading plus two interior; trailing; a single point; both
    ends held out, with one kept point alone between two windows."""
    i = np.arange(t.size)
    return {"interior": (t > 8) & (t < 9.5),
            "lead+2": (t < 1) | ((t > 5) & (t < 6)) | ((t > 12) & (t < 12.2)),
            "trail": t > 18.5,
            "single": i == t.size // 2,
            "edges": np.isin(i, np.r_[0:5, 120:140, 141:160,
                                      t.size - 5:t.size])}[name]


_LAYOUTS = ["interior", "lead+2", "trail", "single", "edges"]


def _rows(n_dim, n_kernel):
    rows = np.zeros((2, n_dim))
    rows[1, :n_kernel] = 0.1 * np.random.default_rng(2).standard_normal(
        n_kernel)
    return rows


def _hi(gp, rows):
    with mx.stream(mx.cpu):
        return np.array(gp.hi_absolute(mx.array(rows, dtype=mx.float64),
                                       refuse=False))


def _gp(term, kp, t, y, yerr, mask, **kw):
    return GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                     GPPolicy(metal=False, kappa_max=None), held_out=mask,
                     **kw)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("name", list(_TERMS))
def test_the_masked_likelihood_is_the_dense_sigma(name, layout):
    """float64 against ``log N(r; 0, Sigma)``, m = 1 to 6: measured
    1e-13 to 2e-11 absolute."""
    term, kp = _TERMS[name]
    t, y, yerr = _data(term, kp)
    mask = _layout(layout, t)
    gp = _gp(term, kp, t, y, yerr, mask)
    rows = _rows(len(kp), len(kp))
    got = _hi(gp, rows)
    for c, row in enumerate(rows):
        ref = dense_masked_loglike(term, np.asarray(kp) * np.exp(row), t, y,
                                   yerr ** 2, mask)
        assert abs(got[c] - ref) < 1e-10 * abs(ref)


def test_windows_are_the_same_mask():
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    a = _gp(term, kp, t, y, yerr, (t >= 5) & (t <= 6) | (t >= 9) & (t <= 9.5))
    b = _gp(term, kp, t, y, yerr, [(5.0, 6.0), (9.0, 9.5)])
    rows = _rows(3, 3)
    np.testing.assert_array_equal(_hi(a, rows), _hi(b, rows))


def test_ties_across_a_window_edge():
    """Duplicate timestamps, one of a pair held out and its twin kept, at
    both edges of a window (a held-out point at lag 0 from the edge)."""
    term, kp = _TERMS["sho"]
    base = np.linspace(0, 10, 150)
    t = np.sort(np.concatenate([base, base]))
    t, y, yerr = _data(term, kp, t=t)
    mask = (t > 4) & (t < 5)
    lo, hi = np.flatnonzero(mask)[[0, -1]]
    mask[lo - 1] = True                 # the twin of the first kept point
    mask[hi] = False                    # keep the twin of the last
    gp = _gp(term, kp, t, y, yerr, mask)
    rows = _rows(3, 3)
    got = _hi(gp, rows)
    for c, row in enumerate(rows):
        ref = dense_masked_loglike(term, np.asarray(kp) * np.exp(row), t, y,
                                   yerr ** 2, mask)
        assert abs(got[c] - ref) < 1e-10 * abs(ref)


def test_a_convolved_term():
    """A `TermConvolution`: its zero-lag correction is part of the GP on
    S and absent from M's white noise, as in ``Sigma``."""
    width = 0.04
    term, kp = ag.TermConvolution(ag.SHOTerm(), width), [1.0, 2.0, 3.0]
    t = np.arange(N) * 0.07
    t, y, yerr = _data(ag.SHOTerm(), kp, t=t)
    mask = ((t > 5) & (t < 6.5)) | (t > 19.5)
    gp = _gp(term, kp, t, y, yerr, mask)
    rows = _rows(3, 3)
    got = _hi(gp, rows)
    for c, row in enumerate(rows):
        ref = dense_masked_loglike(term, np.asarray(kp) * np.exp(row), t, y,
                                   yerr ** 2, mask)
        assert abs(got[c] - ref) < 1e-9 * abs(ref)


def test_an_empty_mask_is_the_unmasked_likelihood():
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    gp = _gp(term, kp, t, y, yerr, np.zeros(N, dtype=bool))
    assert gp._held is None
    ref = _gp(term, kp, t, y, yerr, None)
    rows = _rows(3, 3)
    np.testing.assert_array_equal(_hi(gp, rows), _hi(ref, rows))


def test_scoring_m_by_the_predictive_would_be_the_full_gp():
    """Why M is scored white: with the predictive covariance added, the two
    terms are ``p(r_S) p(r_M | r_S) = p(r)``, the unmasked likelihood --
    shown densely, since it is not an option."""
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    m = _layout("interior", t)
    s = ~m
    K = dense_covariance(term, kp, t, yerr)
    kss, kms = K[np.ix_(s, s)], K[np.ix_(m, s)]
    mean = kms @ np.linalg.solve(kss, y[s])
    cov = K[np.ix_(m, m)] - kms @ np.linalg.solve(kss, kms.T)

    def logn(x, c):
        lo = np.linalg.cholesky(c)
        a = np.linalg.solve(lo, x)
        return (-0.5 * a @ a - np.sum(np.log(np.diag(lo)))
                - 0.5 * x.size * math.log(2 * math.pi))
    full = logn(y[s], kss) + logn(y[m] - mean, cov)
    assert abs(full - dense_loglike(term, kp, t, y, yerr)) < 1e-9
    white = logn(y[s], kss) + logn(y[m] - mean, np.diag(yerr[m] ** 2))
    assert abs(white - dense_masked_loglike(term, kp, t, y, yerr ** 2,
                                            m)) < 1e-9


def _noise(vn, d):
    return (exp_acc(vn[:, :1]) * d.yerr) ** 2 + exp_acc(2.0 * vn[:, 1:2])


def _full_model(mask, policy=None):
    """A mean model, two instruments' offsets and a `noise_fn` -- each
    evaluated on all the data, M included."""
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    inst = (np.arange(N) % 2).astype(np.int32)
    y = y + 0.05 * t + 0.4 * inst
    k = len(kp)

    def mean_fn(vm, x):
        return vm[:, :1] * x
    pm = ParamMap(kernel=[KernelParam(link="log", index=1 + i, ref=r)
                          for i, r in enumerate(kp)],
                  n_dim=1 + k + 2 + 2, mean=slice(0, 1),
                  offset=[KernelParam(link="linear", index=1 + k, ref=0.0),
                          KernelParam(link="linear", index=2 + k, ref=0.0)],
                  instrument=inst, noise=slice(3 + k, 5 + k))
    gp = GPLogLike(mean_fn, t, t, y, yerr, term, pm,
                   policy or GPPolicy(metal=False, kappa_max=None),
                   noise_fn=_noise, held_out=mask)
    rows = np.array([[0.05, 0.0, 0.0, 0.0, 0.3, 0.7, 0.1, -2.0],
                     [0.06, 0.1, -0.1, 0.05, 0.2, 0.75, -0.2, -1.5]])

    def parts(row):
        r = y - row[0] * t - np.where(inst == 1, row[5], row[4])
        w = (np.exp(row[6]) * yerr) ** 2 + np.exp(2 * row[7])
        return np.asarray(kp) * np.exp(row[1:4]), r, w
    return gp, rows, parts, (term, t)


def test_mean_model_offsets_and_noise_fn():
    t = _data(*_TERMS["sho"])[0]
    mask = _layout("lead+2", t)
    gp, rows, parts, (term, t) = _full_model(mask)
    got = _hi(gp, rows)
    for c, row in enumerate(rows):
        p, r, w = parts(row)
        ref = dense_masked_loglike(term, p, t, r, w, mask)
        assert abs(got[c] - ref) < 1e-10 * abs(ref)


@pytest.mark.parametrize("name", ["sho", "rot+sho"])
def test_float32_value_and_gradient(name):
    """The float32 MLX path against float64, value and gradient (central
    differences of `hi`).  Measured: values 4e-7 relative at m = 2 (a few
    float32 ulps of the value), 3.6e-6 at m = 6 with a leading window,
    against 2e-7 unmasked (see docs/masked-gp-plan.md, Phase 1);
    gradients 1.4e-6 and 6.4e-6 relative."""
    term, kp = _TERMS[name]
    t, y, yerr = _data(term, kp)
    gp = _gp(term, kp, t, y, yerr, _layout("lead+2", t))
    rows = _rows(len(kp), len(kp))
    with mx.stream(mx.gpu):
        v32 = mx.array(rows, dtype=mx.float32)
        val = np.array(gp(v32))
        grad = np.array(mx.grad(lambda v: gp(v).sum())(v32))

    def hi(r):
        with mx.stream(mx.cpu):
            return np.array(gp.hi(mx.array(r, dtype=mx.float64)))
    ref = hi(rows)
    tol = 2e-6 if name == "sho" else 1e-5
    assert np.all(np.abs(val - ref) < tol * np.abs(ref))
    fd = np.zeros_like(rows)
    for i in range(rows.shape[1]):
        d = np.zeros_like(rows)
        d[:, i] = 1e-6
        fd[:, i] = (hi(rows + d) - hi(rows - d)) / 2e-6
    assert np.all(np.abs(grad - fd).max(1) < 3e-5 * np.abs(fd).max(1))


@pytest.mark.parametrize("path", ["mlx", "kernel"])
@pytest.mark.parametrize("case", ["white", "slow"])
def test_float32_at_scale_tracks_the_unmasked_path(case, path):
    """N = 2800 at a one-minute cadence, the transit plus ingress and
    egress held out (169 points): where masking is appropriate -- the GP's
    interpolation into the window comparable to the white noise -- the
    masked float32 error is the unmasked path's.  Measured on these rows:
    "white" (sigma = yerr, rho = 1 hr) 4.0e-4 against 1.4e-4; "slow" (sigma =
    30 yerr, rho = 30 hr) 5.1e-2 against 6.5e-2.  Where the GP inside the
    window is large and left unmodelled (sigma = 300 yerr, rho = 10 hr:
    residuals there of 96 sigma, an M term of -2e5) the fusion inside the
    window loses the information gain, ~1e5, and the error is 88 log-units
    against 0.9 -- the regime the class docstring's caveat excludes, and
    not one a sampler stays in.

    On the kernel (Phase 2): "white" 8.2e-5 against 1.0e-4 unmasked;
    "slow" 2.4e-3 against 1.6e-4.  There the held-out term carries it:
    every held-out residual comes from the same two window-edge states,
    whose float32 error (3e-5 in a prediction of a GP 30 times the noise,
    a few ulps) adds coherently over the 169 points.  Still 20x better
    than the MLX path, so the kernel's bound is its own."""
    if path == "kernel" and not metal_available():
        pytest.skip("needs the fused kernel")
    sig, rho = {"white": (1.0, 1 / 24), "slow": (30.0, 30 / 24)}[case]
    n, cad = 2800, 1 / 1440
    t = np.arange(n) * cad
    tc, half = t.mean(), 1.4 / 24
    kp = [sig, rho, 1 / math.sqrt(2)]
    rng = np.random.default_rng(1)
    yerr = np.ones(n)
    y = (sample_ssm(ag.SHOTerm(), kp, cad, n, rng)
         + yerr * rng.standard_normal(n))
    rows = np.zeros((4, 3))
    rows[1:] = 0.2 * np.random.default_rng(2).standard_normal((3, 3))
    err = {}
    for held in (None, [(tc - half, tc + half)]):
        gp = (_gp(ag.SHOTerm(), kp, t, y, yerr, held) if path == "mlx" else
              GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
                        ParamMap.log(kp), GPPolicy(kappa_max=None),
                        held_out=held))
        assert gp._metal_ok == (path == "kernel")
        with mx.stream(mx.gpu):
            lo = np.array(gp(mx.array(rows, dtype=mx.float32)))
        with mx.stream(mx.cpu):
            hi = np.array(gp.hi(mx.array(rows, dtype=mx.float64)))
        err[held is None] = np.abs(lo - hi).max()
    if path == "mlx":
        assert err[False] < 3.0 * err[True] + 1e-3
    else:
        assert err[False] < 1e-2


# -- on the kernel (docs/masked-gp-plan.md, Phase 2) ------------------------

_on_kernel = pytest.mark.skipif(not metal_available(),
                                reason="needs the fused kernel")


def _kgp(term, kp, t, y, yerr, mask, block=None, **kw):
    gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(kappa_max=None, metal_block=block),
                   held_out=mask, **kw)
    assert gp._metal_ok, gp._metal_why
    return gp


def _lo_hi(gp, rows):
    with mx.stream(mx.gpu):
        lo = np.array(gp(mx.array(rows, dtype=mx.float32)), dtype=np.float64)
    with mx.stream(mx.cpu):
        hi = np.array(gp.hi(mx.array(rows, dtype=mx.float64), refuse=False))
    return lo, hi


@pytest.mark.parametrize(
    "term", [ag.RealTerm(), ag.SHOTerm(), ag.Matern32Term(),
             ag.Matern52Term(), ag.RotationTerm(), ag.ComplexTerm(),
             ag.SHOTerm() + ag.RealTerm() + ag.Matern52Term()],
    ids=lambda t: type(t).__name__)
def test_time_reversal_is_a_sign_flip(term):
    """`Term.reversal_signs`, which the kernel's backward series rests on:
    ``Phi^T = S Phi S``, ``I - Phi^T Phi = S Q S`` and ``h S = h`` at
    random parameters (an SHO under- and over-damped) and lags."""
    sg = np.array(term.reversal_signs(), dtype=np.float64)
    big_s, m = np.diag(sg), sg.size
    rng = np.random.default_rng(5)
    lags = np.array([0.0, 1e-3, 0.03, 0.4, 2.0, 9.0])
    for _ in range(4):
        p = np.exp(rng.uniform(-1.0, 1.5, term.n_params))
        with mx.stream(mx.cpu):
            phi, q, h = term.matrices(mx.array(p[None], dtype=mx.float64),
                                      mx.array(lags, dtype=mx.float64))
        phi, q, h = np.array(phi[0]), np.array(q[0]), np.array(h[0])
        for f, g in zip(phi, q):
            np.testing.assert_allclose(f.T, big_s @ f @ big_s, atol=1e-12)
            np.testing.assert_allclose(np.eye(m) - f.T @ f,
                                       big_s @ g @ big_s, atol=1e-11)
        np.testing.assert_array_equal(h * sg, h)
    assert ag.TermConvolution(ag.SHOTerm(), 0.01).reversal_signs() is None


@_on_kernel
@pytest.mark.parametrize("block", [None, 16], ids=["auto", "block16"])
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("name", list(_TERMS))
def test_the_kernel_value_under_the_mask(name, layout, block):
    """The kernel's float32 value against float64: measured 7e-6 to
    1.4e-4 absolute, against 2.6e-5 to 6.4e-4 on the MLX path.  Blocks of
    16 put many block boundaries inside every segment."""
    term, kp = _TERMS[name]
    t, y, yerr = _data(term, kp)
    gp = _kgp(term, kp, t, y, yerr, _layout(layout, t), block)
    lo, hi = _lo_hi(gp, _rows(len(kp), len(kp)))
    assert np.abs(lo - hi).max() < 5e-4


@_on_kernel
@pytest.mark.parametrize("name", ["sho", "rot+sho"])
def test_the_kernel_gradient_is_the_mlx_paths(name):
    """Until the kernel has the held-out gradient (Phase 2c) its VJP is
    the MLX path's (`GPLogLike._evaluate_metal_masked`): measured equal to
    8e-8 and 1.2e-7 relative."""
    term, kp = _TERMS[name]
    t, y, yerr = _data(term, kp)
    mask = _layout("lead+2", t)
    v = mx.array(_rows(len(kp), len(kp)), dtype=mx.float32)
    grads = []
    for gp in (_kgp(term, kp, t, y, yerr, mask),
               _gp(term, kp, t, y, yerr, mask)):
        with mx.stream(mx.gpu):
            grads.append(np.array(mx.grad(lambda w: gp(w).sum())(v)))
    assert np.abs(grads[0] - grads[1]).max() < 1e-5 * np.abs(grads[1]).max()


@_on_kernel
def test_the_full_model_on_the_kernel():
    """Mean model, per-instrument offsets and a per-datum `noise_fn` with
    the kernel's value: measured 1e-4 against float64."""
    t = _data(*_TERMS["sho"])[0]
    gp, rows, _, _ = _full_model(_layout("lead+2", t),
                                 GPPolicy(kappa_max=None))
    assert gp._metal_ok
    lo, hi = _lo_hi(gp, rows)
    assert np.abs(lo - hi).max() < 3e-4


@_on_kernel
def test_the_kernel_holds_a_large_gp_inside_the_window():
    """The regime `test_float32_at_scale_tracks_the_unmasked_path` calls
    out for the MLX path -- a large GP left unmodelled inside a long
    window (sigma = 300 yerr, rho = 10 hr, the whole transit held out) --
    on the kernel, whose window fusion is in covariance form
    (`GPLogLike._held_out_mean`) rather than the MLX path's information
    form.  Measured on these rows: 0.54 log-units against 88 on the MLX
    path (and 9.5e-4 unmasked; the M term is -2e5 here)."""
    n, cad = 2800, 1 / 1440
    t = np.arange(n) * cad
    tc = t.mean()
    kp = [300.0, 10 / 24, 1 / math.sqrt(2)]
    rng = np.random.default_rng(1)
    yerr = np.ones(n)
    y = (sample_ssm(ag.SHOTerm(), kp, cad, n, rng)
         + yerr * rng.standard_normal(n))
    gp = _kgp(ag.SHOTerm(), kp, t, y, yerr, [(tc - 1.4 / 24, tc + 1.4 / 24)])
    rows = np.zeros((4, 3))
    rows[1:] = 0.2 * np.random.default_rng(2).standard_normal((3, 3))
    lo, hi = _lo_hi(gp, rows)
    assert np.abs(lo - hi).max() < 3.0


def _basis_setup(shared, prior):
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    y = y + 2.0 + 0.3 * (t - 10.0)
    mask = _layout("lead+2", t)
    tc = float(t.mean())
    if shared:
        basis = lambda vm, x, m0: mx.stack(                 # noqa: E731
            [mx.ones_like(x), x - tc], axis=-1)
    else:
        def basis(vm, x, m0):
            return mx.stack([mx.ones_like(m0),
                             mx.broadcast_to(x - tc, m0.shape)], axis=-1)
    lp = (None if prior is None
          else (np.array([1.5, 0.2]), np.diag([0.5 ** 2, 0.1 ** 2])))
    gp = _gp(term, kp, t, y, yerr, mask, linear_basis=basis, linear_prior=lp)
    phi = np.stack([np.ones_like(t), t - tc], 1)
    return gp, term, kp, t, y, yerr, mask, phi, lp


def _dense_marginal(sig, r, phi, lp):
    """Flat prior: the restricted likelihood under ``Sigma``; Gaussian:
    ``log N(r; Phi mu, Sigma + Phi S Phi^T)``.  Plus the posterior."""
    si_r, si_phi = np.linalg.solve(sig, r), np.linalg.solve(sig, phi)
    c = phi.T @ si_phi
    d = phi.T @ si_r
    n, p = phi.shape
    if lp is None:
        cov = np.linalg.inv(c)
        mean = cov @ d
        val = (-0.5 * (r @ si_r - d @ mean)
               - 0.5 * np.linalg.slogdet(sig)[1] - 0.5 * np.linalg.slogdet(c)[1]
               - 0.5 * (n - p) * math.log(2 * math.pi))
    else:
        mu, s = lp
        tot = sig + phi @ s @ phi.T
        e = r - phi @ mu
        val = (-0.5 * e @ np.linalg.solve(tot, e)
               - 0.5 * np.linalg.slogdet(tot)[1]
               - 0.5 * n * math.log(2 * math.pi))
        prec = np.linalg.inv(s)
        cov = np.linalg.inv(c + prec)
        mean = cov @ (d + prec @ mu)
    return val, mean, cov


@pytest.mark.parametrize("prior", [None, "gauss"], ids=["flat", "gauss"])
@pytest.mark.parametrize("shared", [False, True], ids=["per-chain", "shared"])
def test_a_marginalised_basis_under_the_mask(shared, prior):
    """The marginal and `linear_posterior` against dense formulas with
    ``Sigma`` for ``K``."""
    gp, term, kp, t, y, yerr, mask, phi, lp = _basis_setup(shared, prior)
    rows = _rows(3, 3)
    got = _hi(gp, rows)
    b, cb = gp.linear_posterior(rows)
    for c, row in enumerate(rows):
        sig = dense_masked_covariance(term, np.asarray(kp) * np.exp(row), t,
                                      yerr ** 2, mask)
        val, mean, cov = _dense_marginal(sig, y, phi, lp)
        assert abs(got[c] - val) < 1e-10 * abs(val)
        np.testing.assert_allclose(b[c], mean, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(cb[c], cov, rtol=1e-9, atol=1e-14)


def _dense_s_conditional(term, p, t, r, w, mask, tp):
    s = ~mask
    K = dense_covariance(term, p, t[s], np.sqrt(w[s]))
    ks = np.asarray(term.value_np(p, np.abs(tp[:, None] - t[s][None, :])
                                  .ravel())).reshape(tp.size, s.sum())
    k0 = term.value_np(p, np.zeros(1))[0]
    mean = ks @ np.linalg.solve(K, r[s])
    var = k0 - np.sum(ks * np.linalg.solve(K, ks.T).T, 1)
    return mean, var, ks @ np.linalg.solve(K, np.eye(s.sum()))


@_on_kernel
@pytest.mark.parametrize("prior", [None, "gauss"], ids=["flat", "gauss"])
@pytest.mark.parametrize("shared", [False, True], ids=["per-chain", "shared"])
def test_a_marginalised_basis_on_the_kernel(shared, prior):
    """The masked Gram on the kernel (`GPLogLike._gram_metal_masked`):
    value against float64, measured 1.5e-5 to 5e-5 (2e-4 to 4.5e-4 on the
    MLX path); gradient against central differences of `hi`, 2e-5 to
    7e-5 relative."""
    gp, term, kp, t, y, yerr, mask, phi, lp = _basis_setup(shared, prior)
    gk = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(kappa_max=None), held_out=mask,
                   linear_basis=gp.linear_basis, linear_prior=lp)
    assert gk._metal_ok
    rows = _rows(3, 3)
    lo, hi = _lo_hi(gk, rows)
    assert np.abs(lo - hi).max() < 3e-4
    with mx.stream(mx.gpu):
        grad = np.array(mx.grad(lambda w: gk(w).sum())(
            mx.array(rows, dtype=mx.float32)))
    fd = np.zeros_like(rows)
    for i in range(3):
        d = np.zeros_like(rows)
        d[:, i] = 1e-5
        fd[:, i] = (_lo_hi(gk, rows + d)[1] - _lo_hi(gk, rows - d)[1]) / 2e-5
    assert np.abs(grad - fd).max() < 3e-4 * np.abs(fd).max()


def test_predict_is_conditioned_on_s():
    """No basis: the dense conditional on S, at new points and at M (where
    the mean is the ``P r_S`` the likelihood applies)."""
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    mask = _layout("lead+2", t)
    gp = _gp(term, kp, t, y, yerr, mask)
    tp = np.concatenate([np.linspace(-2, 22, 41), t[mask]])
    rows = _rows(3, 3)
    mean, var = gp.predict(rows, tp)
    for c, row in enumerate(rows):
        p = np.asarray(kp) * np.exp(row)
        dm, dv, _ = _dense_s_conditional(term, p, t, y, yerr ** 2, mask, tp)
        np.testing.assert_allclose(mean[c], dm, rtol=0,
                                   atol=1e-10 * np.abs(dm).max())
        np.testing.assert_allclose(var[c], dv, rtol=0,
                                   atol=1e-10 * np.abs(dv).max())


@pytest.mark.parametrize("prior", [None, "gauss"], ids=["flat", "gauss"])
def test_predict_with_a_basis_under_the_mask(prior):
    """``P_* (r_S - Phi_S b)`` and ``var + u C_b u^T`` with ``u = P_*
    Phi_S``, ``(b, C_b)`` the posterior under ``Sigma``."""
    gp, term, kp, t, y, yerr, mask, phi, lp = _basis_setup(False, prior)
    tp = np.concatenate([np.linspace(-2, 22, 41), t[mask]])
    rows = _rows(3, 3)
    mean, var = gp.predict(rows, tp)
    for c, row in enumerate(rows):
        p = np.asarray(kp) * np.exp(row)
        sig = dense_masked_covariance(term, p, t, yerr ** 2, mask)
        _, b, cb = _dense_marginal(sig, y, phi, lp)
        _, dv, op = _dense_s_conditional(term, p, t, y, yerr ** 2, mask, tp)
        u = op @ phi[~mask]
        want_m = op @ (y[~mask] - phi[~mask] @ b)
        want_v = dv + np.einsum("mp,pq,mq->m", u, cb, u)
        np.testing.assert_allclose(mean[c], want_m, rtol=0,
                                   atol=1e-9 * np.abs(want_m).max())
        np.testing.assert_allclose(var[c], want_v, rtol=0,
                                   atol=1e-9 * np.abs(want_v).max())


def test_gram_is_the_masked_quadratic_form():
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    mask = _layout("interior", t)
    gp = _gp(term, kp, t, y, yerr, mask)
    rows = _rows(3, 3)
    Y = np.random.default_rng(7).standard_normal((2, N, 3))
    with mx.stream(mx.cpu):
        G = np.array(gp.gram(mx.array(rows, dtype=mx.float64),
                             mx.array(Y, dtype=mx.float64)))
    for c, row in enumerate(rows):
        sig = dense_masked_covariance(term, np.asarray(kp) * np.exp(row), t,
                                      yerr ** 2, mask)
        want = Y[c].T @ np.linalg.solve(sig, Y[c])
        np.testing.assert_allclose(G[c], want, rtol=1e-10,
                                   atol=1e-10 * np.abs(want).max())


@_on_kernel
def test_gram_on_the_kernel():
    """`gram` in float32 with the kernel's masked value: measured 7e-8
    relative to the float64 Gram."""
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    gp = _kgp(term, kp, t, y, yerr, _layout("lead+2", t))
    rows = _rows(3, 3)
    Y = np.random.default_rng(7).standard_normal((2, N, 3))
    with mx.stream(mx.cpu):
        want = np.array(gp.gram(mx.array(rows, dtype=mx.float64),
                                mx.array(Y, dtype=mx.float64)))
    with mx.stream(mx.gpu):
        got = np.array(gp.gram(mx.array(rows, dtype=mx.float32),
                               mx.array(Y, dtype=mx.float32)))
    assert np.abs(got - want).max() < 1e-5 * np.abs(want).max()


def test_bad_masks_and_unsupported_calls_are_refused():
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    with pytest.raises(ValueError, match="shape"):
        _gp(term, kp, t, y, yerr, np.zeros(N - 1, dtype=bool))
    with pytest.raises(ValueError, match="boolean mask"):
        _gp(term, kp, t, y, yerr, np.zeros(N, dtype=np.int32))
    with pytest.raises(ValueError, match="every point"):
        _gp(term, kp, t, y, yerr, np.ones(N, dtype=bool))
    with pytest.raises(ValueError, match="no data point"):
        _gp(term, kp, t, y, yerr, [(5.0, 6.0), (30.0, 31.0)])
    gp = _gp(term, kp, t, y, yerr, _layout("interior", t))
    with pytest.raises(ValueError, match="held-out"):
        gp.apply_inverse(np.zeros((1, 3)), y)
    conv = ag.TermConvolution(term, 1e-5)
    with pytest.warns(RuntimeWarning, match="time reversal"):
        gp = GPLogLike(None, t, None, y, yerr, conv, ParamMap.log(kp),
                       GPPolicy(metal=True, kappa_max=None),
                       held_out=_layout("interior", t))
    assert not gp._metal_ok


def test_make_target_passes_held_out():
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    term, kp = _TERMS["sho"]
    t, y, yerr = _data(term, kp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = make_target(term, t, y, yerr)
        gt = make_target(term, t, y, yerr, held_out=[(8.0, 9.5)])
    assert gt.names == plain.names
    assert gt.loglike._held is not None
    v = np.array(gt.transform.to_model(gt.u0(3, seed=0)))
    with mx.stream(mx.gpu):
        out = np.array(gt.model_log_prob(mx.array(v)))
    assert np.all(np.isfinite(out)) and np.all(out > -1e20)
    if gt.loglike._metal_ok:          # a jitter, on the kernel
        lo, hi = _lo_hi(gt.loglike, v)
        assert np.abs(lo - hi).max() < 3e-4


@pytest.mark.slow
def test_chees_samples_a_masked_target():
    """A gradient sampler on a masked target, as
    `test_calibration.test_chees_runs_on_the_fused_kernel` checks the
    kernel: a wrong gradient collapses the acceptance, and a random walk
    cannot find the jitter's posterior width in this budget.  The float64
    Laplace width of ``log_jitter`` at the mode is 0.138 on this problem;
    measured 0.154 (median per-chain), acceptance 0.89, 5 of 128 chains
    lost.  (At N = 1000, 256 chains and 300 + 300 steps: 0.094 against
    0.089, acceptance 0.96, none lost.)  Those were on the MLX path; since
    Phase 2b the value comes from the kernel and the gradient from the MLX
    path, and the same bounds hold."""
    anvil = pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    sigma, rho, qual, jit, err = 1.5, 7.3, 8.0, 0.12, 0.20
    rng = np.random.default_rng(20260917)
    n = 500
    t = np.arange(n) * (40.0 / n)
    yerr = np.full(n, err)
    cov = dense_covariance(ag.SHOTerm(), np.array([sigma, rho, qual]), t,
                           yerr, jitter=jit)
    y = np.linalg.cholesky(cov) @ rng.standard_normal(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = make_target(ag.SHOTerm(), t, y, yerr,
                         held_out=[(18.0, 20.0), (30.0, 30.5)])
    with mx.stream(mx.gpu):
        res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=16),
                        gt.target, gt.u0(128, seed=1, spread=0.3),
                        n_warmup=200, n_samples=200, seed=3, progress=False)
    acc = float(np.mean(res.accept_fraction))
    assert acc > 0.5, f"acceptance {acc:.2f}: the gradient is not right"
    assert len(res.extras["lost_chains"]) <= 13
    chain = np.asarray(res.get_chain(), dtype=np.float64)
    vm = gt.transform.model_np(chain.reshape(-1, chain.shape[-1]))
    j = vm.reshape(chain.shape)[:, :, gt.names.index("log_jitter")]
    sd = float(np.median(j.std(axis=0)))
    assert 0.11 < sd < 0.17, f"median per-chain log_jitter width {sd:.3f}"
