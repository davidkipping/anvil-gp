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

# float64 arrays cannot be *created* on the GPU stream, so the module default
# is the CPU and the float32 paths opt into the GPU explicitly.
mx.set_default_device(mx.cpu)

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
                   GPPolicy(block=128, hi_block=128))
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
    MetalPlanet it is a (2, m) stack of epoch-centred times and orbit numbers.
    Conflating them is a silent wrong answer, so the API keeps them apart."""
    x_mean = np.stack([T * 2.0, np.floor(T)])          # (2, N), unlike t

    def mean_fn(v_mean, x):
        return v_mean[:, 0:1] * x[0][None, :] + v_mean[:, 1:2] * x[1][None, :]

    pm = ParamMap(kernel=_const(KP), n_dim=2, mean=slice(0, 2))
    gp = GPLogLike(mean_fn, T, x_mean, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(block=128, hi_block=128))
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
                   GPPolicy(block=128, hi_block=128))
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
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm, GPPolicy(block=128))
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
                       GPPolicy(hi_block=128, hi_chain_block=tile))
        outs.append(np.array(gp.hi(v)))
    for o in outs[1:]:
        assert np.max(np.abs(o - outs[0])) < 1e-9 * np.max(np.abs(outs[0]))
    assert outs[0].shape == (37,)


def test_chains_do_not_interact():
    """anvil's summed-gradient trick is only valid because the Jacobian is
    block-diagonal in chains."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(KP)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm, GPPolicy(block=128))
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
                   GPPolicy(block=128, eval_blocks=False))
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
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm, GPPolicy(block=128))
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
                   GPPolicy(block=64, hi_block=64))
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
                       GPPolicy(block=block, hi_block=block))
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
                       GPPolicy(block=256, hi_block=256))
    gp_rnd = GPLogLike(None, rnd, None, y, yerr, ag.SHOTerm(), pm,
                       GPPolicy(block=256, hi_block=256))
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
                   GPPolicy(block=128, hi_block=128, blocked=blocked))
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
    single = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm2, GPPolicy())
    assert not single.blocked                      # one block, m = 2

    two = ag.SHOTerm() + ag.SHOTerm()               # m = 4: dense is faster
    pm4 = ParamMap(kernel=_const([1.0] * 6), n_dim=1)
    assert not GPLogLike(None, T, None, Y, YERR, two, pm4, GPPolicy()).blocked

    four = ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    pm8 = ParamMap(kernel=_const([1.0] * 12), n_dim=1)
    assert GPLogLike(None, T, None, Y, YERR, four, pm8, GPPolicy()).blocked


def test_blocked_gradients_are_finite_and_match_the_dense_path():
    term = ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    refs = [1.0, 4.0, 3.0, 0.5, 1.0, 2.0, 0.3, 0.5, 5.0, 0.2, 0.25, 1.5]
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(refs)], n_dim=len(refs))
    grads = []
    for blocked in (False, True):
        gp = GPLogLike(None, T, None, Y, YERR, term, pm,
                       GPPolicy(block=128, blocked=blocked))
        with mx.stream(mx.gpu):
            u = mx.zeros((4, len(refs)), dtype=mx.float32)
            _, g = mx.vjp(gp, [u], [mx.ones((4,), dtype=mx.float32)])
            mx.eval(g)
        grads.append(np.array(g[0])[0])
        assert np.all(np.isfinite(grads[-1]))
    scale = max(1.0, np.abs(grads[0]).max())
    assert np.max(np.abs(grads[0] - grads[1])) < 2e-3 * scale
