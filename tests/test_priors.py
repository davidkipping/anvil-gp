"""Defaults, the ordering reparametrisation, and the periodogram initialiser."""

import math

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp import priors
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap


rng = np.random.default_rng(3)
N = 400
T = np.sort(rng.uniform(0.0, 40.0, N))
YERR = np.full(N, 0.2)
Y = 1.5 * np.sin(2 * np.pi * T / 7.3) + rng.standard_normal(N) * 0.2


def test_default_hypers_cover_every_term_parameter():
    for term in (ag.SHOTerm(), ag.RealTerm(), ag.Matern32Term(),
                 ag.Matern52Term(), ag.RotationTerm(),
                 ag.SHOTerm() + ag.RealTerm()):
        hs = priors.default_hypers(term, T, Y, YERR)
        assert len(hs) == term.n_params, type(term).__name__


def test_log_rho_bounds_span_resolvable_timescales_only():
    """Below 2 dt_med the kernel is unresolved and degenerate with the
    jitter; above T/2 it is a baseline offset."""
    hs = priors.default_hypers(ag.SHOTerm(), T, Y, YERR)
    rho = hs[1]
    ref = priors.data_references(T, Y, YERR)
    assert math.isclose(rho.ref * math.exp(rho.lo), 2.0 * ref["cadence"],
                        rel_tol=1e-9)
    assert math.isclose(rho.ref * math.exp(rho.hi), 0.5 * ref["baseline"],
                        rel_tol=1e-9)


def test_amplitude_reference_is_the_data_rms():
    hs = priors.default_hypers(ag.SHOTerm(), T, Y, YERR)
    assert math.isclose(hs[0].ref, float(np.std(Y)), rel_tol=1e-12)
    assert hs[0].prior == ("normal", 0.0, 1.5)


def test_build_param_map_lays_out_v_in_the_documented_order():
    hs = priors.default_hypers(ag.SHOTerm(), T, Y, YERR)
    jit = [priors.jitter_hyper(YERR)]
    off = [priors.offset_hyper(Y)]
    pm, specs, names = priors.build_param_map(hs, n_mean=2, jitters=jit,
                                              offsets=off)
    assert pm.n_dim == 2 + 3 + 1 + 1
    assert [k.index for k in pm.kernel] == [2, 3, 4]
    assert [k.index for k in pm.jitter] == [5]
    assert [k.index for k in pm.offset] == [6]
    assert names[0].endswith("sigma") and names[-1] == "offset"
    assert len(specs) == 5


def test_order_groups_make_timescales_strictly_increasing():
    """Permutation multimodality is killed by construction, not by rejection.

    A two-SHO sum has 2! identical modes; the reparametrisation makes the
    second period a ratio > 1 of the first, so the map is smooth and no
    proposal is ever rejected for being out of order.
    """
    kernel = [KernelParam(link="log", index=i, ref=r) for i, r in
              enumerate([1.0, 2.0, 3.0, 1.0, 1.0, 3.0])]
    pm = ParamMap(kernel=kernel, n_dim=6, order_groups=[[1, 4]])
    v = mx.array(rng.standard_normal((64, 6)).astype(np.float32) * 2.0)
    kp = np.array(pm.kernel_params(v))
    assert np.all(kp[:, 4] > kp[:, 1]), "rho_1 must exceed rho_0 always"
    # and the unconstrained columns are untouched
    plain = ParamMap(kernel=kernel, n_dim=6)
    assert np.allclose(np.array(plain.kernel_params(v))[:, [0, 2, 3, 5]],
                       kp[:, [0, 2, 3, 5]])


def test_ordering_is_differentiable_and_finite():
    kernel = [KernelParam(link="log", index=i, ref=1.0) for i in range(6)]
    pm = ParamMap(kernel=kernel, n_dim=6, order_groups=[[1, 4]])
    v = mx.zeros((4, 6), dtype=mx.float32)
    _, g = mx.vjp(lambda z: mx.sum(pm.kernel_params(z), axis=-1), [v],
                  [mx.ones((4,), dtype=mx.float32)])
    assert bool(mx.all(mx.isfinite(g[0])))


def test_log_prior_is_a_proper_normalised_density():
    hs = priors.default_hypers(ag.SHOTerm(), T, Y, YERR)
    lp = priors.make_log_prior(hs, jitters=[priors.jitter_hyper(YERR)])
    v = mx.zeros((5, 4), dtype=mx.float32)
    val = np.array(lp(v))
    # at v = 0 every Gaussian sits at its mean: sum of -log(sd sqrt(2 pi))
    expect = -(math.log(1.5) + math.log(1.0) + math.log(1.0)
               + 1.5 * math.log(2 * math.pi))
    assert np.allclose(val, expect, atol=1e-5)
    assert np.ptp(val) == 0.0


def test_periodogram_finds_the_injected_period():
    p = priors.periodogram_period(T, Y)
    assert abs(p - 7.3) / 7.3 < 0.02


def test_defaults_drive_an_end_to_end_likelihood():
    """The whole point: defaults that produce a working target with no
    hand-tuning."""
    hs = priors.default_hypers(ag.SHOTerm(), T, Y, YERR)
    jit = [priors.jitter_hyper(YERR)]
    pm, specs, names = priors.build_param_map(hs, jitters=jit)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(block=128))
    with mx.stream(mx.gpu):
        u = mx.zeros((16, pm.n_dim), dtype=mx.float32)
        out = gp(u)
        val, grad = mx.vjp(gp, [u], [mx.ones((16,), dtype=mx.float32)])
        mx.eval(out, val, grad)
    assert bool(mx.all(mx.isfinite(out)))
    assert bool(mx.all(mx.isfinite(grad[0])))
    hi = np.array(gp.hi(mx.zeros((2, pm.n_dim), dtype=mx.float64)))
    assert abs(hi[0] - float(np.array(out)[0])) < 0.05


def test_auto_order_groups_finds_same_type_families():
    assert priors.auto_order_groups(ag.SHOTerm()) == []
    assert priors.auto_order_groups(ag.SHOTerm() + ag.RealTerm()) == []
    # two SHOs (3 params each): their rho columns are positions 1 and 4
    assert priors.auto_order_groups(ag.SHOTerm() + ag.SHOTerm()) == [[1, 4]]
    # three SHOs plus a Real: the Real is alone, so only the SHO family
    term = ag.SHOTerm() + ag.RealTerm() + ag.SHOTerm() + ag.SHOTerm()
    assert priors.auto_order_groups(term) == [[1, 6, 9]]


def test_make_target_orders_a_two_sho_sum_by_default():
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    gt = make_target(ag.SHOTerm() + ag.SHOTerm(), T, Y, YERR)
    assert gt.param_map.order_groups == [[1, 4]]
    v = mx.array(np.random.default_rng(0).standard_normal(
        (128, gt.param_map.n_dim)).astype(np.float32) * 1.5)
    kp = np.array(gt.param_map.kernel_params(v))
    assert np.all(kp[:, 4] > kp[:, 1])


@pytest.mark.parametrize("v_hi", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_ordering_ratio_does_not_depend_on_the_time_unit(v_hi):
    """The same data in days, hours and seconds must give the same physics.

    The ordered member used to be ``x_lo * (1 + x_hi)`` with ``x_hi`` a
    timescale in the user's unit, so a pure number was added to a
    dimensionful one.  The implied ratio was 1.4 in days and 3.9e4 in
    seconds for identical data: an accidental, invisible prior change on
    every ordered term after the first.
    """
    term = ag.SHOTerm() + ag.SHOTerm()
    n = 500
    r = np.random.default_rng(0)
    y = r.standard_normal(n)

    ratios, scaled = [], []
    for scale in (1.0, 24.0, 86400.0):          # days, hours, seconds
        t = np.arange(n) * 0.02 * scale
        yerr = np.full(n, 0.1)
        hyp = priors.default_hypers(term, t, y, yerr)
        pm, _, _ = priors.build_param_map(
            hyp, order_groups=priors.auto_order_groups(term))
        v = np.zeros((1, pm.n_dim))
        v[0, 4] = v_hi                          # the ordered rho
        kp = np.array(pm.kernel_params(mx.array(v, dtype=mx.float64)))[0]
        rho1, rho2 = float(kp[1]), float(kp[4])
        assert rho2 > rho1, "ordering must hold in every unit"
        ratios.append(rho2 / rho1)
        scaled.append((rho1 / scale, rho2 / scale))

    assert max(ratios) - min(ratios) < 1e-12 * max(ratios), ratios
    # and both timescales are equivariant: the same physical durations
    for a, b in zip(scaled[1:], scaled[:-1]):
        assert abs(a[0] - b[0]) < 1e-9 * b[0]
        assert abs(a[1] - b[1]) < 1e-9 * b[1]


def test_constant_data_does_not_pin_the_amplitude_at_zero():
    """A flat-lined column must not silently give a white-noise GP.

    `rms` is 0 for constant y, and a zero reference through the log link
    makes sigma exactly 0 for every chain and every step -- the GP is gone,
    with nothing in the output to say so.
    """
    t = np.linspace(0.0, 10.0, 200)
    y = np.full(200, 3.0)
    yerr = np.full(200, 0.05)
    hyp = priors.default_hypers(ag.SHOTerm(), t, y, yerr)
    sigma = hyp[0]
    assert sigma.ref > 0.0
    assert sigma.ref == pytest.approx(0.05)


def test_hi_prior_is_evaluated_at_float64():
    """The .hi path is the reference; it must not inherit float32 rounding."""
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target

    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 20.0, 256))
    y = rng.standard_normal(256)
    yerr = np.full(256, 0.3)
    tgt = make_target(ag.SHOTerm(), t, y, yerr)

    dim = tgt.transform.dim
    u = mx.array(rng.standard_normal((2, dim)) * 0.3, dtype=mx.float64)
    with mx.stream(mx.cpu):
        hi = tgt.target.log_prob_hi(u)
        assert hi.dtype == mx.float64
        assert bool(mx.all(mx.isfinite(hi)))


def test_ordering_refuses_a_non_log_link():
    """The ratio is dimensionless only through a log link.

    With any other link the `1 +` adds a pure number to a timescale, which
    is the unit-dependent bug the log path had. Refuse rather than repeat
    it silently for a hand-built map.
    """
    kernel = [KernelParam(link="const", value=v)
              for v in (1.0, 4.0, 3.0, 1.0, 8.0, 3.0)]
    pm = ParamMap(kernel=kernel, n_dim=1, order_groups=[[1, 4]])
    with pytest.raises(ValueError, match="require 'log'"):
        pm.kernel_params(mx.zeros((2, 1), dtype=mx.float32))


def test_zero_baseline_is_refused():
    """One epoch, or all timestamps equal: no timescale to put a prior on."""
    t = np.zeros(5)
    y = np.arange(5.0)
    yerr = np.full(5, 0.1)
    with pytest.raises(ValueError, match="zero time"):
        priors.default_hypers(ag.SHOTerm(), t, y, yerr)


def test_u0_spread_in_u_is_unchanged():
    """The default draws exactly what it always drew: spread * N(0, 1)."""
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    gt = make_target(ag.SHOTerm(), T, Y, YERR)
    z = np.random.default_rng(7).standard_normal((64, gt.transform.dim))
    np.testing.assert_array_equal(np.array(gt.u0(64, seed=7, spread=0.3)),
                                  (0.3 * z).astype(np.float32))


def test_u0_prior_units_put_one_prior_sigma_in_every_column():
    """With units='prior', spread 1 maps to one prior standard deviation in
    model space for every column, whatever its transform.  Checked through
    the transform itself: a small step along each column, scaled back up,
    must land at the prior width."""
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    gt = make_target(ag.SHOTerm(), T, Y, YERR)
    eps = 1e-4
    u = np.array(gt.u0(1, seed=0, spread=eps, units="prior"), dtype=np.float64)
    z = np.random.default_rng(0).standard_normal(gt.transform.dim)
    v = gt.transform.model_np(u[0]) - gt.transform.model_np(np.zeros_like(u[0]))
    sd_v = v / (eps * z)
    expected = []
    for h, s in zip(gt.hypers, gt.transform.specs):
        expected.append(h.prior[2] if h.prior is not None
                        else (s.hi - s.lo) / math.sqrt(12.0))
    np.testing.assert_allclose(sd_v, expected, rtol=1e-3)
    # The default jitter is the column the option exists for: one unit of u
    # is 3.4 prior sigmas there, so spread 1 in u is far outside the prior.
    j = gt.names.index("log_jitter")
    assert 0.28 < gt._prior_sd_in_u()[j] < 0.30


def test_u0_prior_units_leave_mean_columns_in_u():
    """The mean model's priors are its own business, so its columns keep
    spread in u."""
    pytest.importorskip("anvil")
    from anvilgp.anvil import make_target
    gt = make_target(ag.SHOTerm(), T, Y, YERR,
                     mean_fn=lambda v, x: mx.broadcast_to(v[:, :1], (v.shape[0], x.shape[-1])),
                     x_mean=T, mean_specs=[priors.SpecHint("m0", scale=1.0)])
    sd = gt._prior_sd_in_u()
    assert sd[0] == 1.0 and len(sd) == gt.transform.dim
    with pytest.raises(ValueError, match="units"):
        gt.u0(4, units="sigma")
