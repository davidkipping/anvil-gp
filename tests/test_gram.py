"""The public primitives: `GPLogLike.gram` (``Y^T K^-1 Y``, differentiable)
and `GPLogLike.apply_inverse` (``K^-1 Y``, float64), against dense float64
solves of the same covariance."""
import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
from anvilgp.oracle import dense_covariance, sample_ssm
from anvilgp.special import exp_acc

KP = [1.0, 2.0, 3.0]
N = 300


def _noise(vn, d):
    return (exp_acc(vn[:, :1]) * d.yerr) ** 2 + exp_acc(2.0 * vn[:, 1:2])


def _gp(metal, noise=False, term=None, kp=KP, t=None, seed=0):
    term = ag.SHOTerm() if term is None else term
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 20, N)) if t is None else t
    yerr = rng.uniform(0.1, 0.3, t.size)
    grid = np.linspace(t.min(), t.max(), 3000)
    y = (np.interp(t, grid, sample_ssm(term, kp, grid[1] - grid[0],
                                       grid.size, rng))
         + yerr * rng.standard_normal(t.size))
    k = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(kp)],
                  n_dim=k + (2 if noise else 0),
                  noise=slice(k, k + 2) if noise else None)
    gp = GPLogLike(None, t, None, y, yerr, term, pm,
                   GPPolicy(metal=metal, kappa_max=None),
                   noise_fn=_noise if noise else None)
    return gp, (term, t, yerr)


def _rows(n_dim, noise):
    rows = np.zeros((2, n_dim))
    rows[1] = 0.1 * np.random.default_rng(5).standard_normal(n_dim)
    if noise:
        rows[:, -2:] = [[0.1, -2.5], [-0.2, -2.0]]
    return rows


def _K(data, kp, row, noise):
    term, t, yerr = data
    k = len(kp)
    p = np.asarray(kp) * np.exp(row[:k])
    if noise:
        yerr = np.sqrt((np.exp(row[k]) * yerr) ** 2 + np.exp(2 * row[k + 1]))
    return dense_covariance(term, p, t, yerr)


def _Y(n=2, k=3, seed=7):
    return np.random.default_rng(seed).standard_normal((n, N, k))


@pytest.mark.parametrize("noise", [False, True], ids=["yerr", "noise_fn"])
@pytest.mark.parametrize("path", ["kernel", "mlx", "float64"])
def test_gram_is_the_dense_quadratic_form(path, noise):
    """Every path against ``Y^T K^-1 Y`` from a dense float64 solve, for
    per-chain columns.  Measured 5-7e-8 (kernel: the filter accumulates
    the Gram matrix directly), 2-3e-7 (MLX: polarisation, differences of
    likelihoods, which degrade further for large or nearly parallel
    columns) and 5-7e-15 (float64)."""
    gp, data = _gp(path == "kernel", noise)
    rows = _rows(gp.param_map.n_dim, noise)
    Y = _Y()
    dtype = mx.float64 if path == "float64" else mx.float32
    stream = mx.cpu if path == "float64" else mx.gpu
    with mx.stream(stream):
        G = np.array(gp.gram(mx.array(rows, dtype=dtype),
                             mx.array(Y, dtype=dtype)))
    tol = {"kernel": 1e-6, "mlx": 5e-6, "float64": 1e-12}[path]
    for c, row in enumerate(rows):
        K = _K(data, KP, row, noise)
        want = Y[c].T @ np.linalg.solve(K, Y[c])
        scale = np.sqrt(np.outer(np.diag(want), np.diag(want)))
        assert np.abs(G[c] - want).max() < tol * scale.max()


def test_gram_shapes_and_shared_columns():
    gp, _ = _gp(True)
    v = mx.zeros((3, 3))
    y1 = np.random.default_rng(1).standard_normal(N)
    with mx.stream(mx.gpu):
        g1 = gp.gram(v, y1)
        g2 = gp.gram(v, np.stack([y1, 2 * y1], 1))
    assert g1.shape == (3, 1, 1) and g2.shape == (3, 2, 2)
    np.testing.assert_allclose(np.array(g2)[:, 1, 1], 4 * np.array(g1)[:, 0, 0],
                               rtol=1e-5)
    with pytest.raises(ValueError, match="Y must be"):
        gp.gram(v, np.zeros(N + 1))


@pytest.mark.parametrize("path", ["kernel", "mlx"])
def test_gram_gradients(path):
    """d/dv of ``sum W * G`` against float64 finite differences of the
    dense form, and d/dY against the analytic ``K^-1 Y (W + W^T)``."""
    gp, data = _gp(path == "kernel", noise=True)
    rows = _rows(gp.param_map.n_dim, True)
    Y = _Y()
    W = np.random.default_rng(9).standard_normal((2, 3, 3))

    def f(v, y):
        return mx.sum(mx.array(W.astype(np.float32)) * gp.gram(v, y))
    with mx.stream(mx.gpu):
        gv, gy = mx.grad(f, argnums=(0, 1))(
            mx.array(rows, dtype=mx.float32), mx.array(Y, dtype=mx.float32))
    gv, gy = np.array(gv), np.array(gy)

    def dense(row, c):
        K = _K(data, KP, row, True)
        return np.sum(W[c] * (Y[c].T @ np.linalg.solve(K, Y[c])))
    for c, row in enumerate(rows):
        fd = np.zeros(row.size)
        for i in range(row.size):
            d = np.zeros(row.size)
            d[i] = 1e-6
            fd[i] = (dense(row + d, c) - dense(row - d, c)) / 2e-6
        tol = 1e-4 if path == "kernel" else 3e-3
        assert np.abs(gv[c] - fd).max() < tol * np.abs(fd).max()
        K = _K(data, KP, row, True)
        want_y = np.linalg.solve(K, Y[c]) @ (W[c] + W[c].T)
        assert np.abs(gy[c] - want_y).max() < tol * np.abs(want_y).max()


@pytest.mark.parametrize("path", ["kernel", "mlx", "float64"])
def test_gram_refuses_what_the_likelihood_refuses(path):
    """Every path, float64 included: until 2026-10-26 the float64 route
    skipped the refusals `hi` applies and returned a finite Gram for a
    chain at kappa 5.5e18."""
    gp, _ = _gp(path == "kernel")
    gp.policy.kappa_max = 1e6
    rows = [[0.0, 0.0, 0.0], [20.0, 30.0, 0.0]]          # 2nd: kappa >> max
    dtype = mx.float64 if path == "float64" else mx.float32
    with mx.stream(mx.cpu if path == "float64" else mx.gpu):
        G = np.array(gp.gram(mx.array(rows, dtype=dtype),
                             mx.array(_Y(n=2), dtype=dtype)))
    assert np.all(np.isfinite(G[0])) and np.all(np.isnan(G[1]))


@pytest.mark.parametrize("case", ["sho", "noise_fn", "rot+sho", "ties",
                                  "conv"])
def test_apply_inverse_is_the_dense_solve(case):
    """``K^-1 Y`` from the smoother against ``np.linalg.solve``, including a
    sum of terms (m = 6), exact duplicate timestamps and a convolved
    term (its zero-lag excess is part of K)."""
    term, kp, t, noise = ag.SHOTerm(), KP, None, False
    if case == "noise_fn":
        noise = True
    elif case == "rot+sho":
        term = ag.RotationTerm() + ag.SHOTerm()
        kp = [1.0, 4.0, 3.0, 1.0, 0.4, 0.3, 0.5, 0.7]
    elif case == "ties":
        t = np.sort(np.concatenate([np.linspace(0, 10, 100)] * 2
                                   + [np.linspace(0, 10, 100)[:N - 200]]))
    elif case == "conv":
        term, t = ag.TermConvolution(ag.SHOTerm(), 0.04), np.arange(N) * 0.07
    gp, data = _gp(False, noise, term=term, kp=kp, t=t)
    rows = _rows(gp.param_map.n_dim, noise)
    Y = _Y()
    X = gp.apply_inverse(rows, Y)
    x1 = gp.apply_inverse(rows, Y[0, :, 0])
    assert X.shape == (2, N, 3) and x1.shape == (2, N)
    for c, row in enumerate(rows):
        want = np.linalg.solve(_K(data, kp, row, noise), Y[c])
        np.testing.assert_allclose(X[c], want, rtol=0,
                                   atol=1e-10 * np.abs(want).max())
    want1 = np.linalg.solve(_K(data, kp, rows[1], noise), Y[0, :, 0])
    np.testing.assert_allclose(x1[1], want1, rtol=0,
                               atol=1e-10 * np.abs(want1).max())


def test_the_two_primitives_agree():
    """``Y^T (K^-1 Y)`` from `apply_inverse` is `gram` in float64."""
    gp, _ = _gp(False, noise=True)
    rows = _rows(gp.param_map.n_dim, True)
    Y = _Y()
    X = gp.apply_inverse(rows, Y)
    with mx.stream(mx.cpu):
        G = np.array(gp.gram(mx.array(rows, dtype=mx.float64),
                             mx.array(Y, dtype=mx.float64)))
    np.testing.assert_allclose(np.einsum("cnk,cnl->ckl", Y, X), G,
                               rtol=1e-10, atol=1e-10)
