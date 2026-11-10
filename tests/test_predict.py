"""`GPLogLike.predict`: the GP component's posterior at new inputs.

Every case against the dense float64 conditional, built independently
from the terms' own covariance functions (`oracle.dense_predict`, or the
dense formulas written out where it has no argument for the feature).
"""
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
from anvilgp.oracle import dense_covariance, dense_predict, sample_ssm
from anvilgp.special import exp_acc

_TERMS = [
    ("real", ag.RealTerm(), [1.0, 0.7]),
    ("sho", ag.SHOTerm(), [1.0, 2.0, 3.0]),
    ("matern52", ag.Matern52Term(), [1.0, 1.5]),
    ("rot+sho", ag.RotationTerm() + ag.SHOTerm(),
     [1.0, 4.0, 3.0, 1.0, 0.4, 0.3, 0.5, 0.7]),
]


def _data(term, kp, n=300, seed=0, t=None):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 20, n)) if t is None else t
    yerr = rng.uniform(0.1, 0.3, t.size)
    grid = np.linspace(t.min(), t.max(), 4000)
    y = (np.interp(t, grid, sample_ssm(term, kp, grid[1] - grid[0],
                                       grid.size, rng))
         + yerr * rng.standard_normal(t.size))
    return t, y, yerr


def _t_pred(t):
    """Interior, both ends outside the data, on data points, unsorted."""
    tp = np.concatenate([np.linspace(t[0] - 3, t[-1] + 3, 61), t[::37],
                         [t[5], t[5]]])
    return np.random.default_rng(1).permutation(tp)


def _rows(kp, n_extra=0):
    rng = np.random.default_rng(2)
    rows = np.zeros((2, len(kp) + n_extra))
    rows[1, :len(kp)] = 0.1 * rng.standard_normal(len(kp))
    return rows


def _close(got, want, tol=1e-10):
    scale = np.abs(want).max()
    assert np.abs(got - want).max() <= tol * scale, (
        np.abs(got - want).max() / scale)


@pytest.mark.parametrize("name,term,kp", _TERMS, ids=[c[0] for c in _TERMS])
@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_predict_is_the_dense_conditional(name, term, kp, metal):
    """Mean and variance at new points, on data points (twice), and
    outside the data, for m = 1 to 6; the likelihood's path is
    irrelevant (predict is float64 on the host either way)."""
    t, y, yerr = _data(term, kp)
    gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(metal=metal, kappa_max=None))
    tp = _t_pred(t)
    rows = _rows(kp)
    mean, var = gp.predict(rows, tp)
    assert mean.shape == var.shape == (2, tp.size)
    for c, row in enumerate(rows):
        dm, dv = dense_predict(term, np.asarray(kp) * np.exp(row), t, y,
                               yerr, tp)
        _close(mean[c], dm)
        _close(var[c], dv)
    m_only, none = gp.predict(rows, tp, return_var=False)
    assert none is None
    np.testing.assert_array_equal(m_only, mean)


def test_duplicate_timestamps():
    """Exact ties in the data (dt = 0, where the RTS form's P^-1 is
    singular) and a prediction at the tied time."""
    term, kp = ag.SHOTerm(), [1.0, 2.0, 3.0]
    t = np.sort(np.concatenate([np.linspace(0, 10, 120)] * 2
                               + [np.linspace(0, 10, 7)]))
    t, y, yerr = _data(term, kp, t=t)
    gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(metal=False))
    tp = np.array([t[10], 3.3, t[-1], 11.0])
    mean, var = gp.predict(np.zeros((1, 3)), tp)
    dm, dv = dense_predict(term, kp, t, y, yerr, tp)
    _close(mean[0], dm)
    _close(var[0], dv)


def test_mean_model_offsets_and_noise_fn():
    """The residual after `mean_fn` and per-instrument offsets, and a
    heteroscedastic `noise_fn` (error scale + jitter): the GP component
    is the dense conditional of that residual under that noise."""
    term, kp = ag.SHOTerm(), [1.0, 2.0, 3.0]
    t, y, yerr = _data(term, kp, n=250)
    inst = (np.arange(t.size) % 2).astype(np.int32)
    y = y + 0.3 + 0.05 * t + 0.4 * inst
    k = len(kp)

    def mean_fn(vm, x):
        return vm[:, :1] * x

    def noise_fn(vn, d):
        return (exp_acc(vn[:, :1]) * d.yerr) ** 2 + exp_acc(2.0 * vn[:, 1:2])
    # [slope | kernel (3) | offset per instrument (2) | log s, log j]
    pm = ParamMap(kernel=[KernelParam(link="log", index=1 + i, ref=r)
                          for i, r in enumerate(kp)],
                  n_dim=1 + k + 2 + 2, mean=slice(0, 1),
                  offset=[KernelParam(link="linear", index=1 + k, ref=0.0),
                          KernelParam(link="linear", index=2 + k, ref=0.0)],
                  instrument=inst, noise=slice(3 + k, 5 + k))
    gp = GPLogLike(mean_fn, t, t, y, yerr, term, pm,
                   GPPolicy(metal=False, kappa_max=None), noise_fn=noise_fn)
    rows = np.array([[0.05, 0.0, 0.0, 0.0, 0.3, 0.7, 0.1, -2.0],
                     [0.06, 0.1, -0.1, 0.05, 0.2, 0.75, -0.2, -1.5]])
    tp = _t_pred(t)
    mean, var = gp.predict(rows, tp)
    for c, row in enumerate(rows):
        r = y - row[0] * t - np.where(inst == 1, row[5], row[4])
        s, j = np.exp(row[6]), np.exp(row[7])
        dm, dv = dense_predict(term, np.asarray(kp) * np.exp(row[1:4]), t,
                               r, np.sqrt((s * yerr) ** 2 + j ** 2), tp)
        _close(mean[c], dm)
        _close(var[c], dv)


@pytest.mark.parametrize("prior", [None, "gauss"], ids=["flat", "gauss"])
def test_a_marginalised_linear_basis_is_integrated_out(prior):
    """With beta integrated over its posterior N(b, C_beta), the GP
    component's posterior is mean ``K* K^-1 (r - Phi b)`` and variance
    ``K** - K* K^-1 K*^T`` plus ``u C_beta u^T`` with ``u = K* K^-1
    Phi`` -- written out densely."""
    term, kp = ag.SHOTerm(), [1.0, 2.0, 3.0]
    t, y, yerr = _data(term, kp, n=250)
    y = y + 2.0 + 0.3 * t
    tc = float(t.mean())

    def basis(vm, x, m0):
        return mx.stack([mx.ones_like(m0),
                         mx.broadcast_to(x - tc, m0.shape)], axis=-1)
    lp = (None if prior is None
          else (np.array([1.5, 0.2]), np.diag([0.5 ** 2, 0.1 ** 2])))
    gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(metal=False, kappa_max=None),
                   linear_basis=basis, linear_prior=lp)
    rows = _rows(kp)
    tp = _t_pred(t)
    mean, var = gp.predict(rows, tp)
    b, c_beta = gp.linear_posterior(mx.array(rows, dtype=mx.float64))
    phi = np.stack([np.ones_like(t), t - tc], 1)
    for c, row in enumerate(rows):
        p = np.asarray(kp) * np.exp(row)
        K = dense_covariance(term, p, t, yerr)
        ks = term.value_np(p, np.abs(tp[:, None] - t[None, :]).ravel())
        ks = np.asarray(ks).reshape(tp.size, t.size)
        k0 = term.value_np(p, np.zeros(1))[0]
        want_m = ks @ np.linalg.solve(K, y - phi @ b[c])
        u = ks @ np.linalg.solve(K, phi)
        want_v = (k0 - np.sum(ks * np.linalg.solve(K, ks.T).T, 1)
                  + np.einsum("mp,pq,mq->m", u, c_beta[c], u))
        _close(mean[c], want_m)
        _close(var[c], want_v)


def test_a_convolved_term_predicts_the_exposure_averaged_process():
    """A TermConvolution's dense covariance (its `value_np`, the convolved
    kernel, zero-lag excess included) on a regular cadence, prediction
    points at least the exposure width from everything; closer is
    refused, since the state-space form does not hold there."""
    width = 0.04
    term, kp = ag.TermConvolution(ag.SHOTerm(), width), [1.0, 2.0, 3.0]
    t = np.arange(200) * 0.1
    t, y, yerr = _data(ag.SHOTerm(), kp, t=t)
    gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(metal=False, kappa_max=None))
    tp = np.concatenate([t[:-1:9] + 0.05, [-1.0, 25.0]])
    mean, var = gp.predict(np.zeros((1, 3)), tp)
    dm, dv = dense_predict(term, kp, t, y, yerr, tp)
    _close(mean[0], dm, 1e-9)
    _close(var[0], dv, 1e-9)
    with pytest.raises(ValueError, match="exposure width"):
        gp.predict(np.zeros((1, 3)), [t[3] + 0.01])
    with pytest.raises(ValueError, match="exposure width"):
        gp.predict(np.zeros((1, 3)), [t[3]])


def test_many_chains_in_chunks(monkeypatch):
    """Chunked over chains (forced small here) gives the same answer."""
    import anvilgp.predict as pr
    term, kp = ag.SHOTerm(), [1.0, 2.0, 3.0]
    t, y, yerr = _data(term, kp, n=150)
    gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(metal=False))
    rows = 0.1 * np.random.default_rng(3).standard_normal((7, 3))
    tp = _t_pred(t)
    want = gp.predict(rows, tp)
    monkeypatch.setattr(pr, "_CHUNK_BYTES", 1)          # one chain a chunk
    got = gp.predict(rows, tp)
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(got[1], want[1], rtol=0, atol=1e-13)


def test_make_target_passes_predict_through():
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    term, kp = ag.SHOTerm(), [1.0, 2.0, 3.0]
    t, y, yerr = _data(term, kp, n=150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = make_target(term, t, y, yerr)
    v = np.array(gt.transform.to_model(gt.u0(3, seed=0))).astype(np.float64)
    mean, var = gt.predict(v, np.linspace(0, 20, 11))
    assert mean.shape == var.shape == (3, 11)
    assert np.all(np.isfinite(mean)) and np.all(var > 0)


def test_bad_inputs_are_refused():
    term, kp = ag.SHOTerm(), [1.0, 2.0, 3.0]
    t, y, yerr = _data(term, kp, n=50)
    gp = GPLogLike(None, t, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(metal=False))
    with pytest.raises(ValueError, match="n_chains"):
        gp.predict(np.zeros((2, 4)), [1.0])
    with pytest.raises(ValueError, match="finite"):
        gp.predict(np.zeros((1, 3)), [1.0, np.nan])


def test_predict_at_no_times_is_empty():
    """Used to raise ``KeyError: 'mean'``."""
    rng = np.random.default_rng(0)
    t = np.arange(50) * 0.1
    gp = GPLogLike(None, t, None, rng.standard_normal(50), np.full(50, 0.1),
                   ag.SHOTerm(), ParamMap.log([1.0, 2.0, 3.0]),
                   GPPolicy(metal=False))
    mean, var = gp.predict(np.zeros((3, 3)), [])
    assert mean.shape == var.shape == (3, 0)
    assert gp.predict(np.zeros((3, 3)), [], return_var=False)[1] is None
