"""The general noise model: `GPLogLike(noise_fn=...)` and `anvilgp.noise`.

``noise_fn`` gives the total per-datum noise variance (celerite2's
``diag``).  On the fused kernel it travels as a per-datum extra variance
``(npts, n_chains, 1)`` with a per-datum adjoint (`metal._jitter_src`);
on the MLX and float64 paths it replaces ``yerr^2 + jitter^2`` in the
leaves.  Everything is checked against the dense float64 oracle.
"""
import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
from anvilgp.noise import NoiseData
from anvilgp.oracle import dense_loglike, sample_ssm
from anvilgp.special import exp_acc


def _data(term, kp, n=1200, seed=3, s=0.8, j=0.05):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 40, n))
    yerr = rng.uniform(0.2, 0.4, n)
    grid = np.linspace(0, 40, 4000)
    y = (np.interp(t, grid, sample_ssm(term, kp, grid[1], grid.size, rng))
         + np.sqrt((s * yerr) ** 2 + j ** 2) * rng.standard_normal(n))
    return t, y, yerr


def _scale_jitter(vn, d):
    """(s yerr)^2 + j^2 from (log s, log j), float64-accurate."""
    return (exp_acc(vn[:, :1]) * d.yerr) ** 2 + exp_acc(2.0 * vn[:, 1:2])


def _gp(term, kp, t, y, yerr, metal, noise_fn=_scale_jitter, n_noise=2,
        **pol):
    k = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(kp)],
                  n_dim=k + n_noise,
                  noise=slice(k, k + n_noise) if n_noise else None)
    return GPLogLike(None, t, None, y, yerr, term, pm,
                     GPPolicy(metal=metal, kappa_max=None, **pol),
                     noise_fn=noise_fn)


def _dense(term, kp, t, y, yerr, row):
    k = len(kp)
    p = np.asarray(kp) * np.exp(row[:k])
    s, j = np.exp(row[k]), np.exp(row[k + 1])
    return dense_loglike(term, p, t, y, np.sqrt((s * yerr) ** 2 + j ** 2))


_TERMS = [("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
          ("sho2", ag.SHOTerm() + ag.SHOTerm(), [1.3, 4.0, 3.0, 0.4, 1.0, 2.0]),
          ("rot+sho", ag.RotationTerm() + ag.SHOTerm(),
           [1.0, 6.0, 3.0, 1.0, 0.4, 0.3, 0.5, 0.7])]


@pytest.mark.parametrize("name,term,kp", _TERMS, ids=[c[0] for c in _TERMS])
@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_noise_fn_matches_the_dense_oracle(name, term, kp, metal):
    """Value on every path against dense float64; the float32 gradient
    against float64 central differences of `.hi`.  The noise columns move
    the variance below and above yerr^2 (s = 0.6 to 1.4)."""
    t, y, yerr = _data(term, kp)
    gp = _gp(term, kp, t, y, yerr, metal)
    assert gp._metal_ok is metal
    k = len(kp)
    rows = np.zeros((3, k + 2))
    rows[0, :k] = 0.1
    rows[:, k] = np.log([0.6, 1.4, 0.8])
    rows[:, k + 1] = np.log([0.1, 0.02, 0.05])
    ref = np.array([_dense(term, kp, t, y, yerr, r) for r in rows])
    with mx.stream(mx.gpu):
        u = mx.array(rows, dtype=mx.float32)
        lo = np.array(gp(u)) + gp.log_norm_const
        g = np.array(mx.grad(lambda u: gp(u).sum())(u))
    with mx.stream(mx.cpu):
        def hi(X):
            return np.array(gp.hi(mx.array(X, dtype=mx.float64),
                                  refuse=False))
        assert np.abs(hi(rows) + gp.log_norm_const - ref).max() < 1e-8
        h, G = 1e-6, np.zeros_like(rows)
        for i in range(rows.shape[1]):
            dp = np.zeros(rows.shape[1])
            dp[i] = h
            G[:, i] = (hi(rows + dp) - hi(rows - dp)) / (2 * h)
    assert np.abs(lo - ref).max() < 2e-3
    assert np.abs(g - G).max() / np.abs(G).max() < 1e-4


def test_an_error_scale_is_a_reparametrisation():
    """(s yerr)^2 is the likelihood of the data, mean and kernel amplitude
    all divided by s, with the Jacobian -N ln s: checked in float64 on the
    reference path, where nothing is approximate."""
    term, kp = ag.SHOTerm(), [1.3, 4.0, 3.0]
    t, y, yerr = _data(term, kp, n=600)
    s = 1.7
    gp = _gp(term, kp, t, y, yerr, False,
             noise_fn=lambda vn, d: (s * d.yerr) ** 2, n_noise=0)
    scaled = GPLogLike(None, t, None, y / s, yerr, term,
                       ParamMap.log([kp[0] / s, kp[1], kp[2]]),
                       GPPolicy(metal=False, kappa_max=None))
    with mx.stream(mx.cpu):
        z = mx.zeros((1, 3), dtype=mx.float64)
        a = float(np.array(gp.hi(z))[0]) + gp.log_norm_const
        b = float(np.array(scaled.hi(z))[0]) + scaled.log_norm_const
    assert abs(a - (b - t.size * math.log(s))) < 1e-8 * abs(a)


def test_small_scales_are_not_floored():
    """An error scale below 1 puts the noise variance under yerr^2, where
    the square-root block pass floored the leaf variance at r_ref / 2
    until noise_fn arrived.  At s = 0.3, over several kernel blocks, the
    kernel still matches float64."""
    term, kp = ag.SHOTerm() + ag.SHOTerm(), [1.3, 4.0, 3.0, 0.4, 1.0, 2.0]
    t, y, yerr = _data(term, kp, n=2048, s=0.3, j=0.0)
    gp = _gp(term, kp, t, y, yerr, True, metal_block=256,
             noise_fn=lambda vn, d: (0.3 * d.yerr) ** 2, n_noise=0)
    assert gp.metal_block == 256 and gp._metal_boundary == "sqrt"
    with mx.stream(mx.gpu):
        lo = float(np.array(gp(mx.zeros((1, 6))))[0]) + gp.log_norm_const
    ref = dense_loglike(term, np.asarray(kp), t, y, 0.3 * yerr)
    assert abs(lo - ref) < 5e-3, lo - ref


def test_a_convolved_term_adds_its_correction_to_the_noise():
    """A TermConvolution's zero-lag correction rides in the per-datum
    variance like it rides in the jitter; kernel against `.hi`."""
    term = ag.TermConvolution(ag.SHOTerm(), 0.02)
    kp = [1.3, 4.0, 3.0]
    rng = np.random.default_rng(3)
    t = np.arange(800) * 0.05                    # spacings >= the exposure
    yerr = rng.uniform(0.2, 0.4, t.size)
    y = (sample_ssm(ag.SHOTerm(), kp, 0.05, t.size, rng)
         + yerr * rng.standard_normal(t.size))
    gps = [_gp(term, kp, t, y, yerr, m) for m in (False, True)]
    row = np.array([[0.0, 0.0, 0.0, np.log(0.9), np.log(0.05)]])
    with mx.stream(mx.cpu):
        hi = float(np.array(gps[0].hi(mx.array(row, dtype=mx.float64)))[0])
    for gp in gps:
        with mx.stream(mx.gpu):
            lo = float(np.array(gp(mx.array(row, dtype=mx.float32)))[0])
        assert abs(lo - hi) < 2e-3


def test_per_instrument_noise_needs_labels_and_uses_them():
    term, kp = ag.SHOTerm(), [1.3, 4.0, 3.0]
    t, y, yerr = _data(term, kp, n=900)
    inst = (np.arange(t.size) % 3).astype(np.int32)
    nm = ag.noise.scale_and_jitter(yerr, n_instruments=3)
    k = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(kp)],
                  n_dim=k + 6, noise=slice(k, k + 6), instrument=inst)
    row = np.zeros((1, k + 6))
    row[0, k:k + 3] = np.log([0.7, 1.0, 1.5])
    row[0, k + 3:] = np.log([0.01, 0.05, 0.1])
    s = np.array([0.7, 1.0, 1.5])[inst]
    j = np.array([0.01, 0.05, 0.1])[inst]
    ref = dense_loglike(term, np.asarray(kp), t, y,
                        np.sqrt((s * yerr) ** 2 + j ** 2))
    for metal in (False, True):
        gp = GPLogLike(None, t, None, y, yerr, term, pm,
                       GPPolicy(metal=metal, kappa_max=None), noise_fn=nm.fn)
        with mx.stream(mx.gpu):
            lo = float(np.array(gp(mx.array(row, dtype=mx.float32)))[0])
        assert abs(lo + gp.log_norm_const - ref) < 2e-3
    pm.instrument = None
    with pytest.raises(ValueError, match="instrument"):
        GPLogLike(None, t, None, y, yerr, term, pm,
                  GPPolicy(metal=False), noise_fn=nm.fn)(
                      mx.array(row, dtype=mx.float32))


def test_noise_fn_refuses_ambiguous_setups():
    term, kp = ag.SHOTerm(), [1.3, 4.0, 3.0]
    t, y, yerr = _data(term, kp, n=200)
    jit = ParamMap(kernel=ParamMap.log(kp).kernel, n_dim=4,
                   jitter=[KernelParam(link="log", index=3, ref=0.1)])
    with pytest.raises(ValueError, match="cannot be combined"):
        GPLogLike(None, t, None, y, yerr, term, jit, noise_fn=_scale_jitter)
    inert = ParamMap(kernel=ParamMap.log(kp).kernel, n_dim=5,
                     noise=slice(3, 5))
    with pytest.raises(ValueError, match="inert"):
        GPLogLike(None, t, None, y, yerr, term, inert)
    from anvilgp.anvil import make_target
    pytest.importorskip("anvil")
    with pytest.raises(ValueError, match="n_jitter"):
        make_target(term, t, y, yerr, noise=ag.noise.error_scale(),
                    n_jitter=1)


def test_kappa_reads_the_noise_model():
    """The conditioning guard divides by the noise variance, so a smaller
    error scale must raise kappa -- by s^-2 when h Q h^T is negligible."""
    term, kp = ag.SHOTerm(), [1.3, 40.0, 3.0]          # long timescale
    t = np.arange(2000) * 0.01
    y = np.random.default_rng(0).standard_normal(t.size)
    yerr = np.full(t.size, 0.3)
    gp = _gp(term, kp, t, y, yerr, True, noise_fn=lambda vn, d:
             (exp_acc(vn[:, :1]) * d.yerr) ** 2, n_noise=1)
    v = mx.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, math.log(0.1)]],
                 dtype=mx.float32)
    k = np.array(gp.kappa(v))
    assert np.all(np.isfinite(k)) and k[1] > 50 * k[0]


def test_make_target_with_a_noise_model():
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    term, kp = ag.SHOTerm(), [1.3, 4.0, 3.0]
    t, y, yerr = _data(term, kp, n=600)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = make_target(term, t, y, yerr,
                         noise=ag.noise.scale_and_jitter(yerr))
    assert gt.names[-2:] == ["log_noise_scale", "log_noise_jitter"]
    assert gt.param_map.jitter == [] and gt.param_map.noise == slice(3, 5)
    u0 = gt.u0(8, seed=1)
    assert np.allclose(gt.u_ref()[-2:], 0.0)
    lp = gt.target.log_prob(u0)
    g = mx.grad(lambda u: gt.target.log_prob(u).sum())(u0)
    assert np.all(np.isfinite(np.array(lp))) and np.all(np.isfinite(np.array(g)))
    nat = gt.natural(np.array(gt.transform.to_model(u0)))
    assert {"noise_scale", "noise_jitter", "gp_sigma"} <= set(nat)


@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_a_non_time_input_with_ties(metal):
    """Any sorted coordinate is valid input, not only time: an irregular
    one with exact ties (here an angle, rounded to whole degrees so a third
    of the spacings are zero), value and gradient against float64."""
    rng = np.random.default_rng(5)
    x = np.sort(np.round(rng.uniform(0, 360, 600)))
    assert np.sum(np.diff(x) == 0) > 150
    y = rng.standard_normal(x.size) * 0.002
    yerr = np.full(x.size, 0.001)
    term, kp = ag.Matern32Term(), [0.003, 60.0]
    gp = GPLogLike(None, x, None, y, yerr, term, ParamMap.log(kp),
                   GPPolicy(metal=metal, kappa_max=None))
    assert gp._metal_ok is metal
    ref = dense_loglike(term, np.asarray(kp), x, y, yerr)
    with mx.stream(mx.gpu):
        u = mx.zeros((1, 2))
        lo = float(np.array(gp(u))[0]) + gp.log_norm_const
        g = np.array(mx.grad(lambda u: gp(u).sum())(u))[0]
    with mx.stream(mx.cpu):
        def hi(p):
            return float(np.array(gp.hi(mx.array([p], dtype=mx.float64)))[0])
        G = [(hi([1e-6 * (i == 0), 1e-6 * (i == 1)])
              - hi([-1e-6 * (i == 0), -1e-6 * (i == 1)])) / 2e-6
             for i in range(2)]
    assert abs(lo - ref) < 2e-3
    assert np.abs(g - G).max() / np.abs(G).max() < 1e-4


def test_noise_data_is_sliced_with_the_block():
    """The reference path calls noise_fn one block at a time; a
    heteroscedastic model that reads the abscissa must see the slice that
    matches its yerr.  Noise growing with x, MLX path at a short block,
    against the dense oracle."""
    term, kp = ag.SHOTerm(), [1.3, 4.0, 3.0]
    t, y, yerr = _data(term, kp, n=1000)
    gp = _gp(term, kp, t, y, yerr, False, block=128, n_noise=0,
             noise_fn=lambda vn, d: d.yerr ** 2 * (1.0 + d.x / 40.0))
    ref = dense_loglike(term, np.asarray(kp), t, y,
                        yerr * np.sqrt(1.0 + t / 40.0))
    with mx.stream(mx.cpu):
        a = float(np.array(gp.hi(mx.zeros((1, 3), dtype=mx.float64)))[0])
    assert abs(a + gp.log_norm_const - ref) < 1e-8 * abs(ref)


def test_noise_data_fields():
    d = NoiseData(mx.ones(3), mx.zeros(3), None)
    assert d.inst is None and d.yerr.shape == (3,)
