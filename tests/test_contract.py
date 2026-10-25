"""The engine contract: mean models, per-instrument nuisances, tiling, compile.

These are the surfaces a real dataset touches on day one, as opposed to the
mathematics the other test modules pin down.
"""

import numpy as np
import mlx.core as mx
import pytest

import anvilgp as ag
from anvilgp.likelihood import (GPLogLike, GPPolicy, KernelParam, ParamMap,
                                a64)
from anvilgp.oracle import dense_loglike


N = 384
_rng = np.random.default_rng(21)
T = np.sort(_rng.uniform(0.0, 15.0, N))
YERR = np.full(N, 0.25)
Y = _rng.standard_normal(N) * 0.7
KP = [1.2, 3.0, 4.0]


def _const(vals):
    return [KernelParam(link="const", value=float(v)) for v in vals]


def test_mean_model_is_subtracted_and_differentiable():
    """A linear mean model, so the residual is checkable in closed form and
    the gradient has an analytic target: d logL/d(mean) = (K+Sigma)^-1 r."""
    slope_ref = 0.3

    def mean_fn(v_mean, x):
        return v_mean[:, 0:1] * x[None, :]

    pm = ParamMap(kernel=_const(KP), n_dim=1, mean=slice(0, 1))
    gp = GPLogLike(mean_fn, T, T, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(metal=False, block=128, hi_block=128))
    v = a64([[slope_ref]])
    got = float(np.array(gp.hi_absolute(v))[0])
    ref = dense_loglike(ag.SHOTerm(), np.array(KP), T, Y - slope_ref * T, YERR)
    assert abs(got - ref) < 1e-9 * abs(ref)

    with mx.stream(mx.gpu):
        u = mx.array(np.array([[slope_ref]], dtype=np.float32))
        _, g = mx.vjp(gp, [u], [mx.ones((1,), dtype=mx.float32)])
        mx.eval(g)
    ad = float(np.array(g[0])[0, 0])
    h = 1e-5
    fd = (float(np.array(gp.hi(a64([[slope_ref + h]])))[0])
          - float(np.array(gp.hi(a64([[slope_ref - h]])))[0])
          ) / (2 * h)
    assert abs(ad - fd) < 5e-3 * max(1.0, abs(fd))


def test_x_mean_is_a_separate_abscissa_from_t():
    """t drives the GP's lag structure; x_mean drives the mean model, and for
    Elome it is a (2, m) stack of epoch-centred times and orbit numbers.
    Conflating them is a silent wrong answer, so the API keeps them apart."""
    x_mean = np.stack([T * 2.0, np.floor(T)])          # (2, N), unlike t

    def mean_fn(v_mean, x):
        return v_mean[:, 0:1] * x[0][None, :] + v_mean[:, 1:2] * x[1][None, :]

    pm = ParamMap(kernel=_const(KP), n_dim=2, mean=slice(0, 2))
    gp = GPLogLike(mean_fn, T, x_mean, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(metal=False, block=128, hi_block=128))
    v = a64([[0.05, -0.02]])
    resid = Y - (0.05 * x_mean[0] + (-0.02) * x_mean[1])
    ref = dense_loglike(ag.SHOTerm(), np.array(KP), T, resid, YERR)
    assert abs(float(np.array(gp.hi_absolute(v))[0]) - ref) < 1e-9 * abs(ref)


def test_per_instrument_jitters_and_offsets():
    """yerr alone is not the observation diagonal once a jitter is sampled,
    and two instruments do not share one."""
    inst = (T > 7.5).astype(np.int32)
    jit = np.array([0.30, 0.10])
    off = np.array([0.20, -0.15])
    pm = ParamMap(
        kernel=_const(KP), n_dim=1,
        jitter=[KernelParam(link="const", value=float(j)) for j in jit],
        offset=[KernelParam(link="const", value=float(o)) for o in off],
        instrument=inst)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(metal=False, block=128, hi_block=128))
    got = float(np.array(gp.hi_absolute(a64(np.zeros((1, 1)))))[0])

    # dense reference with a heteroscedastic diagonal
    from anvilgp.oracle import dense_covariance
    cov = dense_covariance(ag.SHOTerm(), np.array(KP), T, YERR)
    cov[np.diag_indices(N)] += jit[inst] ** 2
    r = Y - off[inst]
    lo = np.linalg.cholesky(cov)
    alpha = np.linalg.solve(lo.T, np.linalg.solve(lo, r))
    ref = float(-0.5 * r @ alpha - np.sum(np.log(np.diag(lo)))
                - 0.5 * N * np.log(2 * np.pi))
    assert abs(got - ref) < 1e-9 * abs(ref)


def test_multi_instrument_requires_an_instrument_index():
    pm = ParamMap(kernel=_const(KP), n_dim=1,
                  jitter=[KernelParam(link="const", value=0.1),
                          KernelParam(link="const", value=0.2)])
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm, GPPolicy(metal=False, block=128))
    with pytest.raises(ValueError, match="instrument"):
        gp.hi(a64(np.zeros((1, 1))))


def test_hi_tiles_over_chains_not_data():
    """A GP factorisation is global-sequential over data, so the float64 path
    can only tile chains.  The tile size must not change the answer."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(KP)], n_dim=3)
    v = a64(_rng.standard_normal((37, 3)) * 0.1)
    outs = []
    for tile in (1, 4, 16, 64):
        gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                       GPPolicy(metal=False, hi_block=128, hi_chain_block=tile))
        outs.append(np.array(gp.hi(v)))
    for o in outs[1:]:
        assert np.max(np.abs(o - outs[0])) < 1e-9 * np.max(np.abs(outs[0]))
    assert outs[0].shape == (37,)


def test_chains_do_not_interact():
    """anvil's summed-gradient trick is only valid because the Jacobian is
    block-diagonal in chains."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(KP)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm, GPPolicy(metal=False, block=128))
    with mx.stream(mx.gpu):
        u = mx.array(_rng.standard_normal((8, 3)).astype(np.float32) * 0.2)
        full = np.array(gp(u))
        alone = np.array(gp(u[3:4]))
    assert abs(full[3] - alone[0]) < 1e-4 * max(1.0, abs(alone[0]))


def test_is_mx_compile_safe():
    """Fixed iteration counts, no data-dependent control flow, no .item():
    the whole call has to survive mx.compile and give the same answer."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(KP)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(metal=False, block=128, eval_blocks=False))
    compiled = mx.compile(gp.__call__)
    with mx.stream(mx.gpu):
        u = mx.array(_rng.standard_normal((16, 3)).astype(np.float32) * 0.2)
        eager, comp = np.array(gp(u)), np.array(compiled(u))
    assert np.max(np.abs(eager - comp)) < 1e-3 * max(1.0, np.abs(eager).max())


def test_log_norm_const_is_the_only_thing_left_out():
    """The hyperparameter-dependent part of log|K| must be inside the graph;
    only the parameter-independent constant is held on the host."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(KP)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm, GPPolicy(metal=False, block=128))
    v = a64([[0.0, 0.0, 0.0], [0.4, -0.3, 0.6]])
    offset = np.array(gp.hi(v))
    absolute = np.array(gp.hi_absolute(v))
    assert np.allclose(absolute - offset, gp.log_norm_const)
    # the two states must differ by more than the constant: log|K| moved
    assert abs((absolute[1] - absolute[0]) - (offset[1] - offset[0])) < 1e-9
    assert abs(absolute[1] - absolute[0]) > 1.0


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_tiny_datasets(n):
    """N below one block, and N = 1 where there is no transition at all."""
    t = np.sort(_rng.uniform(0.0, 2.0, n))
    y = _rng.standard_normal(n)
    yerr = np.full(n, 0.4)
    pm = ParamMap(kernel=_const(KP), n_dim=1)
    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                   GPPolicy(metal=False, block=64, hi_block=64))
    got = float(np.array(gp.hi_absolute(a64(np.zeros((1, 1)))))[0])
    ref = dense_loglike(ag.SHOTerm(), np.array(KP), t, y, yerr)
    assert abs(got - ref) < 1e-9 * max(1.0, abs(ref))


def test_yerr_must_be_positive():
    bad = YERR.copy()
    bad[3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        GPLogLike(None, T, None, Y, bad, ag.SHOTerm(),
                  ParamMap(kernel=_const(KP), n_dim=1))


def test_ragged_final_block_is_padded_not_truncated():
    """N not a multiple of the block size must give the same answer: the tree
    pads with composition identities rather than dropping points."""
    n = 300
    t = np.sort(_rng.uniform(0.0, 12.0, n))
    y = _rng.standard_normal(n)
    yerr = np.full(n, 0.3)
    pm = ParamMap(kernel=_const(KP), n_dim=1)
    ref = dense_loglike(ag.SHOTerm(), np.array(KP), t, y, yerr)
    for block in (32, 64, 128, 256, 512):
        gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                       GPPolicy(metal=False, block=block, hi_block=block))
        got = float(np.array(gp.hi_absolute(a64(np.zeros((1, 1)))))[0])
        assert abs(got - ref) < 1e-9 * abs(ref), block


def test_cadence_reuse_is_exact_and_engages_only_when_it_pays():
    """Phi and Q depend on dt, which for real photometry takes a handful of
    distinct values.  Evaluating them once per distinct lag must give exactly
    the same answer as once per point."""
    n = 512
    reg = np.arange(n) * 0.02                       # one cadence
    rnd = np.sort(_rng.uniform(0.0, n * 0.02, n))   # all lags distinct
    pm = ParamMap(kernel=_const(KP), n_dim=1)
    y = _rng.standard_normal(n)
    yerr = np.full(n, 0.3)

    gp_reg = GPLogLike(None, reg, None, y, yerr, ag.SHOTerm(), pm,
                       GPPolicy(metal=False, block=256, hi_block=256))
    gp_rnd = GPLogLike(None, rnd, None, y, yerr, ag.SHOTerm(), pm,
                       GPPolicy(metal=False, block=256, hi_block=256))
    # A regular cadence is *one* lag in float32 even though float64
    # representation noise splits it into a dozen -- plus the dt = 0
    # sentinel that stands in for the first point's absent predecessor.
    assert gp_reg.n_distinct_lags == 2
    assert gp_reg._uniq32 is not None and gp_rnd._uniq32 is None

    for gp, t in ((gp_reg, reg), (gp_rnd, rnd)):
        got = float(np.array(gp.hi_absolute(a64(np.zeros((1, 1)))))[0])
        ref = dense_loglike(ag.SHOTerm(), np.array(KP), t, y, yerr)
        assert abs(got - ref) < 1e-9 * abs(ref)

    # and the reuse path must not perturb the float32 answer either
    with mx.stream(mx.gpu):
        f32 = float(np.array(gp_reg(mx.zeros((1, 1), dtype=mx.float32)))[0])
    f64 = float(np.array(gp_reg.hi(a64(np.zeros((1, 1)))))[0])
    assert abs(f32 - f64) < 0.05


@pytest.mark.parametrize("blocked", [False, True])
def test_blocked_leaves_give_the_same_likelihood(blocked):
    """The compact leaf representation must be invisible in the answer."""
    term = ag.SHOTerm() + ag.SHOTerm()
    kp = [1.2, 3.0, 4.0, 0.6, 1.1, 0.9]
    pm = ParamMap(kernel=_const(kp), n_dim=1)
    gp = GPLogLike(None, T, None, Y, YERR, term, pm,
                   GPPolicy(metal=False, block=128, hi_block=128, blocked=blocked))
    assert gp.blocked is blocked
    got = float(np.array(gp.hi_absolute(a64(np.zeros((1, 1)))))[0])
    ref = dense_loglike(term, np.array(kp), T, Y, YERR)
    assert abs(got - ref) < 1e-9 * abs(ref)
    with mx.stream(mx.gpu):
        f32 = np.array(gp(mx.zeros((2, 1), dtype=mx.float32)), dtype=np.float64)
    assert abs(f32[0] + gp.log_norm_const - ref) < 0.05


def test_blocked_default_follows_the_block_structure():
    kp2 = [1.2, 3.0, 4.0]
    pm2 = ParamMap(kernel=_const(kp2), n_dim=1)
    single = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm2, GPPolicy(metal=False))
    assert not single.blocked                      # one block, m = 2

    two = ag.SHOTerm() + ag.SHOTerm()               # m = 4: dense is faster
    pm4 = ParamMap(kernel=_const([1.0] * 6), n_dim=1)
    assert not GPLogLike(None, T, None, Y, YERR, two, pm4, GPPolicy(metal=False)).blocked

    four = ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    pm8 = ParamMap(kernel=_const([1.0] * 12), n_dim=1)
    assert GPLogLike(None, T, None, Y, YERR, four, pm8, GPPolicy(metal=False)).blocked


def test_blocked_gradients_are_finite_and_match_the_dense_path():
    term = ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    refs = [1.0, 4.0, 3.0, 0.5, 1.0, 2.0, 0.3, 0.5, 5.0, 0.2, 0.25, 1.5]
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(refs)], n_dim=len(refs))
    grads = []
    for blocked in (False, True):
        gp = GPLogLike(None, T, None, Y, YERR, term, pm,
                       GPPolicy(metal=False, block=128, blocked=blocked))
        with mx.stream(mx.gpu):
            u = mx.zeros((4, len(refs)), dtype=mx.float32)
            _, g = mx.vjp(gp, [u], [mx.ones((4,), dtype=mx.float32)])
            mx.eval(g)
        grads.append(np.array(g[0])[0])
        assert np.all(np.isfinite(grads[-1]))
    scale = max(1.0, np.abs(grads[0]).max())
    assert np.max(np.abs(grads[0] - grads[1])) < 2e-3 * scale


# -- input validation ------------------------------------------------------
#
# Every one of these used to be accepted and produce a wrong answer quietly:
# a NaN in the data made every chain -1e30 (a likelihood flat in every
# direction, which a sampler explores forever), and a half-specified mean
# model fitted the GP to y itself, which looks exactly like a converged
# GP-only fit rather than an error.


def _gp(**kw):
    pm = kw.pop("pm", ParamMap(kernel=_const(KP), n_dim=1))
    args = dict(mean_fn=None, t=T, x_mean=None, y=Y, yerr=YERR,
                kernel_spec=ag.SHOTerm(), param_map=pm)
    args.update(kw)
    return GPLogLike(args["mean_fn"], args["t"], args["x_mean"], args["y"],
                     args["yerr"], args["kernel_spec"], args["param_map"],
                     GPPolicy(metal=False, block=64))


@pytest.mark.parametrize("field", ["t", "y", "yerr"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_data_is_refused(field, bad):
    arr = {"t": T.copy(), "y": Y.copy(), "yerr": YERR.copy()}
    arr[field][7] = bad
    with pytest.raises(ValueError, match="non-finite"):
        _gp(**arr)


def test_empty_data_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        _gp(t=np.array([]), y=np.array([]), yerr=np.array([]))


def test_half_specified_mean_model_is_refused():
    """The failure this prevents is a transit fit that silently has no
    transit in it."""
    lin = ParamMap(kernel=_const(KP), n_dim=2, mean=slice(0, 1))
    plain = ParamMap(kernel=_const(KP), n_dim=1)

    def mean_fn(p, x):
        return p[:, 0:1] * x

    # mean_fn but no slice
    with pytest.raises(ValueError, match="silently ignored"):
        _gp(mean_fn=mean_fn, x_mean=T, pm=plain)
    # slice but no mean_fn
    with pytest.raises(ValueError, match="silently ignored"):
        _gp(pm=lin)
    # both, but nothing to evaluate them on
    with pytest.raises(ValueError, match="x_mean"):
        _gp(mean_fn=mean_fn, pm=lin)
    # all three: fine
    _gp(mean_fn=mean_fn, x_mean=T, pm=lin)


def test_wrong_parameter_width_is_refused():
    gp = _gp(pm=ParamMap(kernel=[KernelParam(link="log", index=i, ref=1.0)
                                 for i in range(3)], n_dim=3))
    with mx.stream(mx.gpu):
        gp(mx.zeros((4, 3), dtype=mx.float32))           # right
        with pytest.raises(ValueError, match="n_dim"):
            gp(mx.zeros((4, 5), dtype=mx.float32))       # too wide
        with pytest.raises(ValueError, match="n_dim"):
            gp(mx.zeros((4, 2), dtype=mx.float32))       # too narrow
        with pytest.raises(ValueError, match="n_chains"):
            gp(mx.zeros((3,), dtype=mx.float32))         # not 2-D


def test_numpy_scalars_in_kernel_params_do_not_poison_the_graph():
    """np.float64 refs used to fail deep inside mx.concatenate."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=np.float64(r))
                          for i, r in enumerate([1.0, 3.0, 4.0])], n_dim=3)
    gp = _gp(pm=pm)
    with mx.stream(mx.gpu):
        out = np.array(gp(mx.zeros((2, 3), dtype=mx.float32)))
    assert np.all(np.isfinite(out))


def test_param_map_constructors_match_the_long_form():
    """`ParamMap.const` / `.log` are sugar; they must be exactly sugar."""
    vals = (1.0, 4.0, 3.0)
    short = ParamMap.const(vals, jitter=0.1)
    long = ParamMap(kernel=[KernelParam(link="const", value=v) for v in vals],
                    n_dim=1,
                    jitter=[KernelParam(link="const", value=0.1)])
    v = mx.zeros((3, 1), dtype=mx.float32)
    assert np.array_equal(np.array(short.kernel_params(v)),
                          np.array(long.kernel_params(v)))
    assert np.array_equal(np.array(short.jitter_var(v, None)),
                          np.array(long.jitter_var(v, None)))

    short_l, long_l = ParamMap.log(vals), ParamMap(
        kernel=[KernelParam(link="log", index=i, ref=r)
                for i, r in enumerate(vals)], n_dim=3)
    u = mx.array(_rng.standard_normal((3, 3)).astype(np.float32))
    assert short_l.n_dim == long_l.n_dim == 3
    assert np.array_equal(np.array(short_l.kernel_params(u)),
                          np.array(long_l.kernel_params(u)))


def test_yerr_that_underflows_in_float32_is_refused():
    """yerr^2 is what enters the graph; below ~1e-19 it squares to zero.

    Every innovation variance then collapses and the likelihood is -1e30
    for every chain: a flat posterior with no error reported. This is the
    same silent-flat-likelihood failure the non-finite checks prevent.
    """
    with pytest.raises(ValueError, match="rescale|float32 zero"):
        _gp(yerr=np.full(N, 1e-20))
    _gp(yerr=np.full(N, 1e-15))          # fine


# -- the failure value ------------------------------------------------------
#
# A failed (float32-infeasible) evaluation returns a large-negative finite
# value.  It used to be the constant -1e30, and a constant is a *flat*
# region, not a rejection: a sampler accepts moves within it and drifts.
# That was masked until anvil made its bounded log-Jacobian finite (anvil
# f97f583; it had been exactly -inf past |u| ~ 18, an accidental wall), after
# which stretch walkers started there diffused to |u| ~ 1e11 and never came
# back.  The value now slopes back towards the reference point.  These pin
# the four properties that make that safe.

def _failing_gp(metal=False):
    rng = np.random.default_rng(5)
    n = 300
    t = np.sort(rng.uniform(0.0, 40.0, n))
    yerr = np.full(n, 0.2)
    y = rng.standard_normal(n)
    return GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
                     ParamMap.log([1.5, 7.3, 8.0]), GPPolicy(metal=metal))


def test_a_failed_likelihood_slopes_back_towards_the_reference():
    """Finite, never above -1e30, strictly better closer in, zero gradient.

    * **finite** -- a NaN or -inf would poison an ensemble's adaptation;
    * **at or below -1e30** -- it must never beat a real likelihood, and
      every ``<= -1e29`` check in the package still has to mean "failed";
    * **strictly better closer to v = 0** -- the whole point: a slope a
      sampler can follow home, rather than a plateau it wanders;
    * **exactly zero gradient** -- the slope is ``-1e30 * pull``, whose true
      derivative is of order 1e30 and would wreck an HMC trajectory, so it is
      ``stop_gradient``-ed and acts through values alone.
    """
    gp = _failing_gp()
    ks = np.array([45.0, 60.0, 100.0, 1e3, 1e6], dtype=np.float32)
    v = mx.array(np.repeat(ks[:, None], 3, axis=1))
    with mx.stream(mx.gpu):
        val = gp(v)
        _, grad = mx.vjp(gp, [v], [mx.ones((len(ks),), dtype=mx.float32)])
        mx.eval(val, grad)
    val, grad = np.array(val), np.array(grad[0])
    assert np.all(val <= -1e29), f"these points are meant to fail: {val}"
    assert np.all(np.isfinite(val)), val
    assert np.all(val <= -1e30), val
    assert np.all(np.diff(val) < 0), (
        f"not strictly better closer in -- a flat failure region is back: {val}")
    assert np.all(grad == 0.0), f"failed lanes must carry no gradient: {grad}"


@pytest.mark.parametrize("metal", [False, True])
def test_the_sloped_failure_value_is_invisible_on_feasible_points(metal):
    """Bit-identical values and gradients wherever the likelihood succeeds.

    The property the change was verified against, pinned: the slope is only
    ever *selected* on failed lanes, so a chain that never enters the
    infeasible region cannot see it, and no posterior can move.  Compared
    against ``pull = None``, which is exactly the previous constant sentinel.
    """
    gp = _failing_gp(metal=metal)
    v = mx.array(np.random.default_rng(1).standard_normal((128, 3))
                 .astype(np.float32))
    one = mx.ones((128,), dtype=mx.float32)

    def run():
        with mx.stream(mx.gpu):
            val = gp(v)
            _, g = mx.vjp(gp, [v], [one])
            mx.eval(val, g)
        return np.array(val), np.array(g[0])

    new_v, new_g = run()
    real = GPLogLike._infeasible_pull
    try:
        GPLogLike._infeasible_pull = lambda self, vv: None
        old_v, old_g = run()
    finally:
        GPLogLike._infeasible_pull = real
    ok = old_v > -1e29
    assert ok.sum() > 100, f"want mostly feasible points, got {ok.sum()}"
    assert np.array_equal(new_v[ok], old_v[ok])
    assert np.array_equal(new_g[ok], old_g[ok])


def test_the_pull_ignores_mean_model_columns():
    """Only the GP's own columns are pulled towards zero.

    Their reference point is ``v = 0`` by construction.  A mean model's is
    not -- a transit's ``a/R* = 0`` is not even inside its bounds -- so its
    columns must not enter the distance.
    """
    gp = GPLogLike(lambda vm, x: mx.zeros((vm.shape[0], x.shape[-1])),
                   np.arange(50.0), np.arange(50.0), np.zeros(50),
                   np.full(50, 0.2), ag.SHOTerm(),
                   ParamMap(kernel=[KernelParam(link="log", index=i + 2)
                                    for i in range(3)],
                            n_dim=5, mean=slice(0, 2)),
                   GPPolicy(metal=False))
    assert np.array(gp._pull_cols).tolist() == [2, 3, 4]


# --- the kappa guard ---------------------------------------------------------
# A chain whose float32 conditioning number is past GPPolicy.kappa_max gets
# the failure value instead of an evaluation float32 cannot do.  One such
# chain used to collapse ChEES's shared step size for every chain; see
# docs/float32.md.

def _kappa_np(term, kp, dt, r):
    """float64 reference: k(0) / (h Q h^T + r) for P_inf = I terms."""
    with mx.stream(mx.cpu):
        _, q, h = [np.array(o, np.float64) for o in term.matrices(
            mx.array(np.asarray(kp, np.float64)),
            mx.array(np.full((len(kp), 1), dt)))]
    h = h.reshape(len(kp), -1)
    q = q.reshape(len(kp), h.shape[1], h.shape[1])
    return (np.einsum("ni,ni->n", h, h)
            / (np.einsum("ni,nij,nj->n", h, q, h) + r))


def _guard_gp(metal=False, **policy):
    rng = np.random.default_rng(11)
    n = 400
    t = np.arange(n) * 0.02
    yerr = np.full(n, 0.15)
    y = rng.standard_normal(n)
    return GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
                     ParamMap.log([0.15, 5.0, 4.0]),
                     GPPolicy(metal=metal, **policy))


def test_kappa_matches_the_float64_formula():
    gp = _guard_gp()
    v = np.random.default_rng(2).uniform([-2, -3, -3], [25, 2, 4], (64, 3))
    kp = np.array([0.15, 5.0, 4.0]) * np.exp(v)
    want = _kappa_np(ag.SHOTerm(), kp, 0.02, 0.15 ** 2)
    got = np.array(gp.kappa(mx.array(v.astype(np.float32))), np.float64)
    np.testing.assert_allclose(got, want, rtol=1e-3)
    assert want.min() < 1e2 and want.max() > 1e8, "want both sides covered"


@pytest.mark.parametrize("term,p", [
    (ag.SHOTerm() + ag.Matern32Term(), [1.0, 5.0, 4.0, 0.5, 2.0]),
    (ag.RotationTerm(), [1.0, 5.0, 3.0, 1.0, 0.5]),
    (ag.Matern52Term(), [1.0, 5.0]),
])
def test_kappa_is_finite_and_grows_with_amplitude_for_every_term(term, p):
    """Blocked terms sum their blocks; amplitude raises k(0) and h Q h^T
    together, so kappa climbs to k(0)/(hQh^T) as the white noise drops out."""
    n = 200
    t = np.arange(n) * 0.02
    gp = GPLogLike(None, t, None, np.zeros(n), np.full(n, 0.15), term,
                   ParamMap.log(p), GPPolicy(metal=False))
    amp = np.zeros((6, len(p)), np.float32)
    amp[:, 0] = np.linspace(-4, 10, 6)
    k = np.array(gp.kappa(mx.array(amp)))
    assert np.all(np.isfinite(k)) and np.all(k > 0)
    assert np.all(np.diff(k) >= 0), k


def test_kappa_uses_the_convolved_stationary_covariance():
    """TermConvolution's P_inf is not the identity; kappa must use it."""
    from anvilgp.terms import TermConvolution
    n = 200
    t = np.arange(n) * 0.02
    term = TermConvolution(ag.Matern52Term(), 0.01)
    gp = GPLogLike(None, t, None, np.zeros(n), np.full(n, 0.15), term,
                   ParamMap.log([1.0, 5.0]), GPPolicy(metal=False))
    v = mx.zeros((1, 2), dtype=mx.float32)
    kp = gp.param_map.kernel_params(v)
    _, (q,), h = term.matrices_blocks(kp, mx.full((1, 1), 0.02))
    (p_inf,) = term.p_inf_blocks(kp)
    h = np.array(h, np.float64)[0]
    k0 = h @ np.array(p_inf, np.float64)[0] @ h
    hqh = h @ np.array(q, np.float64)[0, 0] @ h
    np.testing.assert_allclose(float(np.array(gp.kappa(v))[0]),
                               k0 / (max(hqh, 0.0) + 0.15 ** 2), rtol=1e-4)


def test_kappa_takes_the_smallest_white_noise_over_instruments():
    n = 200
    t = np.arange(n) * 0.02
    inst = np.arange(n) % 2
    yerr = np.where(inst == 0, 0.1, 1.0)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate([1.0, 5.0, 4.0])],
                  n_dim=5, instrument=inst,
                  jitter=[KernelParam(link="log", index=3, ref=1.0),
                          KernelParam(link="log", index=4, ref=1.0)])
    gp = GPLogLike(None, t, None, np.zeros(n), yerr, ag.SHOTerm(), pm,
                   GPPolicy(metal=False))
    v = np.array([[0, 0, 0, np.log(0.3), np.log(0.01)],
                  [0, 0, 0, np.log(0.01), np.log(0.01)]], np.float32)
    r = np.array([min(0.1 ** 2 + 0.3 ** 2, 1.0 + 1e-4),
                  min(0.1 ** 2 + 1e-4, 1.0 + 1e-4)])
    want = _kappa_np(ag.SHOTerm(), [[1.0, 5.0, 4.0]] * 2, 0.02, r)
    np.testing.assert_allclose(np.array(gp.kappa(mx.array(v))), want,
                               rtol=1e-4)


@pytest.mark.parametrize("metal", [False, True])
def test_the_kappa_guard_refuses_past_kappa_max_and_nothing_else(metal):
    """Refused lanes get the failure value and a zero gradient; every other
    lane is bit-identical in value to ``kappa_max=None``, and in gradient
    to rounding (see ``GPLogLike._refuse_ill_conditioned``).  And the guard
    is what refused them: some were finite without it."""
    v = mx.array(np.random.default_rng(4)
                 .uniform([-2, -3, -3], [18, 2, 4], (128, 3))
                 .astype(np.float32))
    one = mx.ones((128,), dtype=mx.float32)
    out = {}
    for kmax in (1e6, None):
        gp = _guard_gp(metal=metal, kappa_max=kmax)
        with mx.stream(mx.gpu):
            val = gp(v)
            _, g = mx.vjp(gp, [v], [one])
            mx.eval(val, g)
        out[kmax] = (np.array(val), np.array(g[0]))
    k = np.array(_guard_gp().kappa(v))
    refused = k > 1e6
    assert 10 < refused.sum() < 118, f"want both sides: {refused.sum()}"
    (gv, gg), (nv, ng) = out[1e6], out[None]
    assert np.all(gv[refused] <= -1e29)
    assert np.all(gg[refused] == 0.0)
    assert np.array_equal(gv[~refused], nv[~refused])
    if metal:
        # kappa reads the kernel's lag table in-graph, which moves MLX's
        # backward by up to 4.2e-6 relative (measured); both gradients
        # match a float64 finite difference equally well.
        np.testing.assert_allclose(gg[~refused], ng[~refused], rtol=2e-5,
                                   atol=1e-4)
    else:
        # The MLX path's gradient is not bit-reproducible run to run, guard
        # or no guard: two identical kappa_max=None calls differ by up to
        # 1.0e-5 relative here (measured over 6 runs), and the guarded
        # against the unguarded by 1.3e-5 -- the same noise.
        np.testing.assert_allclose(gg[~refused], ng[~refused], rtol=1e-4,
                                   atol=1e-4)
    assert np.any(nv[refused] > -1e29), "the guard refused nothing new"


@pytest.mark.parametrize("term,p", [
    (ag.SHOTerm(), [0.15, 5.0, 4.0]),
    (ag.SHOTerm() + ag.Matern32Term(), [1.0, 5.0, 4.0, 0.5, 2.0]),
])
def test_kappa_read_from_the_kernel_lag_table_matches_the_rebuilt_one(term, p):
    """The kernel path reads h Q h^T from its own lag table at the median
    lag instead of rebuilding the term's matrices (the rebuild cost ~1.1 ms
    a call, +66% on a 2000 x 256 eager value).  Same number, to float32
    rounding, over a range wide enough to cross kappa_max."""
    n = 400
    t = np.arange(n) * 0.02
    gp = GPLogLike(None, t, None, np.random.default_rng(0).standard_normal(n),
                   np.full(n, 0.15), term, ParamMap.log(p),
                   GPPolicy(metal=True))
    assert gp._kappa_row is not None
    lo = [-2.0] + [-1.0] * (len(p) - 1)
    hi = [18.0] + [1.0] * (len(p) - 1)
    v = mx.array(np.random.default_rng(1).uniform(lo, hi, (64, len(p)))
                 .astype(np.float32))
    with mx.stream(mx.gpu):
        _, tables = gp._evaluate_metal(v)
        got = np.array(gp.kappa(v, tables), np.float64)
        want = np.array(gp.kappa(v), np.float64)
    assert want.min() < 1e3 and want.max() > 1e6
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_the_float64_path_refuses_what_the_float32_path_refuses():
    """``.hi`` is the same target in float64.  anvil's ``reanchor_every``
    writes it into the chains' stored ``log_prob``, so an unguarded ``.hi``
    handed a refused chain a real likelihood again (and passed a float64
    NaN straight through).  Same refusals, same failure value -- to float32
    rounding of the ``1e30`` sentinel -- and ``refuse=False`` still gives
    the raw float64 value for measuring float32 error past the guard."""
    v = mx.array(np.random.default_rng(4)
                 .uniform([-2, -3, -3], [18, 2, 4], (64, 3))
                 .astype(np.float32))
    gp = _guard_gp(kappa_max=1e6)
    refused = np.array(gp.kappa(v)) > 1e6
    assert 5 < refused.sum() < 59, refused.sum()
    with mx.stream(mx.gpu):
        lo = np.array(gp(v), np.float64)
    hi = np.array(gp.hi(v), np.float64)
    raw = np.array(gp.hi(v, refuse=False), np.float64)
    assert np.all(hi[refused] <= -1e29)
    np.testing.assert_allclose(hi[refused], lo[refused], rtol=1e-6)
    assert np.all(np.isfinite(raw)) and np.all(raw[refused] > -1e29)
    assert np.array_equal(hi[~refused], raw[~refused])
    # a non-finite float64 value is sanitised too
    nan_gp = _guard_gp(kappa_max=None)
    bad = mx.array(np.array([[0.0, 0.0, float("nan")]], np.float32))
    assert float(np.array(nan_gp.hi(bad))[0]) <= -1e29
    assert np.isnan(float(np.array(nan_gp.hi(bad, refuse=False))[0]))


def test_construction_warns_when_the_data_could_reach_kappa_max():
    n = 200
    t = np.arange(n) * 0.02
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="kappa_max"):
        GPLogLike(None, t, None, 3000.0 * rng.standard_normal(n),
                  np.full(n, 1.0), ag.SHOTerm(), ParamMap.log([1, 1, 1]),
                  GPPolicy(metal=False))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GPLogLike(None, t, None, 5.0 * rng.standard_normal(n),
                  np.full(n, 1.0), ag.SHOTerm(), ParamMap.log([1, 1, 1]),
                  GPPolicy(metal=False))
        GPLogLike(None, t, None, 3000.0 * rng.standard_normal(n),
                  np.full(n, 1.0), ag.SHOTerm(), ParamMap.log([1, 1, 1]),
                  GPPolicy(metal=False, kappa_max=None))


def test_kappa_max_must_be_positive():
    with pytest.raises(ValueError, match="kappa_max"):
        GPPolicy(metal=False, kappa_max=0.0)
