"""Absolute log-likelihood parity against celerite2 and a dense fp64 oracle.

Absolute, not differential: a wrong additive constant is invisible to MCMC
and to every delta-based check, and would only show up as a wrong Bayesian
evidence years later.

Where the two oracles disagree, the dense Cholesky wins -- celerite2's
SHOTerm carries an ``eps`` to dodge the ``1/sqrt(4Q^2-1)`` singularity at
Q = 1/2, and its Matern32Term is an explicit *approximation* (Matern-3/2 is
not celerite-representable at all).  Those are exactly the corners the
state-space route was chosen to handle, so the test asserts against the
dense oracle there and records celerite2's deficit rather than chasing it.
"""

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.kalman import sequential_loglike
from anvilgp.likelihood import (GPLogLike, GPPolicy, KernelParam, ParamMap,
                                a64)
from anvilgp.oracle import celerite2_loglike, dense_loglike, sample_ssm

import celerite2.terms as c2t   # a declared test dep: never skip M1 parity

N = 512
_rng = np.random.default_rng(7)
T = np.sort(_rng.uniform(0.0, 20.0, N))
YERR = np.full(N, 0.3)
Y = _rng.standard_normal(N) * 0.9

SIG, RHO = 1.3, 4.0


def make(term, kp, t=T, y=Y, yerr=YERR, jitter=None, block=64):
    kernel = [KernelParam(link="const", value=float(x)) for x in kp]
    jit = [KernelParam(link="const", value=float(jitter))] if jitter else []
    pm = ParamMap(kernel=kernel, n_dim=1, jitter=jit)
    return GPLogLike(None, t, None, y, yerr, term, pm,
                     GPPolicy(block=block, hi_block=block))


def ours(term, kp, **kw):
    gp = make(term, kp, **kw)
    return float(np.array(gp.hi_absolute(mx.zeros((1, 1), dtype=mx.float64)))[0])


C2_CASES = [
    ("sho_Q0.1", ag.SHOTerm(), [SIG, RHO, 0.1],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=0.1)),
    ("sho_Q0.4", ag.SHOTerm(), [SIG, RHO, 0.4],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=0.4)),
    ("sho_granulation", ag.SHOTerm(), [SIG, RHO, 2 ** -0.5],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=2 ** -0.5)),
    ("sho_Q3", ag.SHOTerm(), [SIG, RHO, 3.0],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=3.0)),
    ("sho_Q30", ag.SHOTerm(), [SIG, RHO, 30.0],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=30.0)),
    ("sho_Q300", ag.SHOTerm(), [SIG, RHO, 300.0],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=300.0)),
    ("real", ag.RealTerm(), [0.8, 0.5], c2t.RealTerm(a=0.64, c=0.5)),
    ("sum", ag.SHOTerm() + ag.RealTerm(), [SIG, RHO, 3.0, 0.8, 0.5],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=3.0) + c2t.RealTerm(a=0.64, c=0.5)),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4],
     c2t.RotationTerm(sigma=1.5, period=7.0, Q0=3.0, dQ=1.0, f=0.4)),
    ("rotation_lowQ", ag.RotationTerm(), [0.8, 3.1, 0.2, 0.05, 0.9],
     c2t.RotationTerm(sigma=0.8, period=3.1, Q0=0.2, dQ=0.05, f=0.9)),
    ("rotation_highQ", ag.RotationTerm(), [2.2, 12.5, 40.0, 20.0, 0.05],
     c2t.RotationTerm(sigma=2.2, period=12.5, Q0=40.0, dQ=20.0, f=0.05)),
    ("sum3", ag.SHOTerm() + ag.SHOTerm() + ag.RealTerm(),
     [SIG, RHO, 3.0, 0.6, 1.1, 0.9, 0.8, 0.5],
     c2t.SHOTerm(sigma=SIG, rho=RHO, Q=3.0)
     + c2t.SHOTerm(sigma=0.6, rho=1.1, Q=0.9) + c2t.RealTerm(a=0.64, c=0.5)),
]


@pytest.mark.parametrize("name,term,kp,c2", C2_CASES,
                         ids=[c[0] for c in C2_CASES])
def test_absolute_loglike_matches_celerite2(name, term, kp, c2):
    got = ours(term, kp)
    ref = celerite2_loglike(c2, T, Y, YERR)
    assert abs(got - ref) < 1e-9 * max(1.0, abs(ref))


@pytest.mark.parametrize("name,term,kp,c2", C2_CASES,
                         ids=[c[0] for c in C2_CASES])
def test_absolute_loglike_matches_dense_cholesky(name, term, kp, c2):
    got = ours(term, kp)
    ref = dense_loglike(term, np.array(kp), T, Y, YERR)
    assert abs(got - ref) < 1e-9 * max(1.0, abs(ref))


@pytest.mark.parametrize("kp,term", [
    ([SIG, RHO, 0.5], ag.SHOTerm()),
    ([SIG, RHO, 0.5 + 1e-6], ag.SHOTerm()),
    ([1.1, 2.5], ag.Matern32Term()),
    ([1.1, 2.5], ag.Matern52Term()),
])
def test_corners_where_celerite2_itself_degrades(kp, term):
    """We match the dense oracle where celerite2 cannot."""
    got = ours(term, kp)
    ref = dense_loglike(term, np.array(kp), T, Y, YERR)
    assert abs(got - ref) < 1e-9 * max(1.0, abs(ref))


def test_celerite2_matern32_converges_to_ours_as_its_eps_shrinks():
    """Pins the interpretation of the 6e-3 discrepancy at celerite2's default
    eps: it is celerite2's approximation error, not ours."""
    got = ours(ag.Matern32Term(), [1.1, 2.5])
    errs = [abs(got - celerite2_loglike(
        c2t.Matern32Term(sigma=1.1, rho=2.5, eps=e), T, Y, YERR))
        for e in (1e-2, 1e-4, 1e-6)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-8


def test_jitter_enters_the_observation_variance():
    got = ours(ag.SHOTerm(), [SIG, RHO, 3.0], jitter=0.45)
    ref = celerite2_loglike(c2t.SHOTerm(sigma=SIG, rho=RHO, Q=3.0),
                            T, Y, YERR, jitter=0.45)
    assert abs(got - ref) < 1e-9 * abs(ref)


def test_duplicate_timestamps_are_legal():
    t = T.copy()
    t[10] = t[9]
    t[300] = t[299]
    t = np.sort(t)
    got = ours(ag.SHOTerm(), [SIG, RHO, 3.0], t=t)
    ref = celerite2_loglike(c2t.SHOTerm(sigma=SIG, rho=RHO, Q=3.0), t, Y, YERR)
    assert abs(got - ref) < 1e-9 * abs(ref)


def test_unsorted_input_raises():
    t = T.copy()
    t[5], t[6] = t[6], t[5]
    with pytest.raises(ValueError, match="sorted"):
        make(ag.SHOTerm(), [SIG, RHO, 3.0], t=t)


@pytest.mark.parametrize("block", [16, 64, 256, 512, 1024])
def test_answer_is_independent_of_the_block_size(block):
    """The block loop is a memory/graph-size knob, never a numerical one."""
    ref = ours(ag.SHOTerm(), [SIG, RHO, 3.0], block=512)
    got = ours(ag.SHOTerm(), [SIG, RHO, 3.0], block=block)
    assert abs(got - ref) < 1e-9 * abs(ref)


def test_tree_reduction_matches_the_sequential_filter():
    """Two independent recursions over identical leaf inputs."""
    gp = make(ag.SHOTerm(), [SIG, RHO, 3.0], block=512)
    v = mx.zeros((1, 1), dtype=mx.float64)
    args = gp.leaf_inputs(v, a64(gp._dt64), a64(gp._y64),
                          a64(gp._yerr64 ** 2), None, None, 0, N)
    seq = float(np.array(sequential_loglike(*args))[0]) + gp.log_norm_const
    assert abs(seq - ours(ag.SHOTerm(), [SIG, RHO, 3.0])) < 1e-9 * abs(seq)


def test_float32_gpu_path_agrees_with_float64():
    """The production path, measured against its own verification path on the
    scale that matters -- the ~1-unit Metropolis accept decision."""
    gp = make(ag.SHOTerm(), [SIG, RHO, 3.0], block=256)
    with mx.stream(mx.gpu):
        v32 = mx.zeros((4, 1), dtype=mx.float32)
        got = np.array(gp(v32), dtype=np.float64)
    ref = float(np.array(gp.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])
    assert np.max(np.abs(got - ref)) < 0.05
    assert np.ptp(got) == 0.0, "identical chains must give identical answers"


def test_positive_definiteness_failure_is_finite_not_nan():
    """An infeasible corner must reject the chain, not poison the ensemble.

    sigma_GP/yerr ~ 1e4 with a timescale far above the cadence is the
    documented float32-infeasible region.
    """
    pm = ParamMap(kernel=[KernelParam(link="log", index=0, ref=1e4),
                          KernelParam(link="log", index=1, ref=1e6),
                          KernelParam(link="log", index=2, ref=1e3)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, np.full(N, 1e-4), ag.SHOTerm(), pm,
                   GPPolicy(block=64))
    with mx.stream(mx.gpu):
        u = mx.zeros((4, 3), dtype=mx.float32)
        out = gp(u)
        val, grad = mx.vjp(gp, [u], [mx.ones((4,), dtype=mx.float32)])
        mx.eval(out, val, grad)
    assert bool(mx.all(mx.isfinite(out))), "log-likelihood must stay finite"
    # the brief asks for a zero gradient there, not merely a finite value.
    # This assertion was missing: the test computed `grad` and never looked
    # at it, and it was NaN.
    assert bool(mx.all(mx.isfinite(grad[0]))), "gradient must stay finite"


FAILURE_POINTS = [
    ("all-huge", [30.0, 30.0, 30.0]),
    ("sigma-huge", [30.0, 0.0, 0.0]),
    ("all-tiny", [-30.0, -30.0, -30.0]),
    ("rho-huge", [0.0, 30.0, 0.0]),
    ("Q-huge", [0.0, 0.0, 30.0]),
]


@pytest.mark.parametrize("name,point", FAILURE_POINTS,
                         ids=[c[0] for c in FAILURE_POINTS])
def test_infeasible_corners_have_finite_gradients(name, point):
    """Warmup visits these; a NaN there poisons the whole ensemble.

    Sanitising the value does not sanitise the gradient.  `_sanitize` routes
    a zero cotangent to the failed lanes, but the graph above it has already
    produced an infinite derivative, and 0 * inf is NaN however the zero
    arrives.  Two of these five points returned NaN gradients -- one of them
    with a perfectly finite value -- until the guard was added.
    """
    pm = ParamMap(kernel=[KernelParam(link="log", index=0, ref=1e4),
                          KernelParam(link="log", index=1, ref=1e6),
                          KernelParam(link="log", index=2, ref=1e3)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, np.full(N, 1e-4), ag.SHOTerm(), pm,
                   GPPolicy(block=64))
    u = mx.array(np.array([point], dtype=np.float32))
    with mx.stream(mx.gpu):
        out = gp(u)
        _, grad = mx.vjp(gp, [u], [mx.ones((1,), dtype=mx.float32)])
        mx.eval(out, grad)
    assert bool(mx.all(mx.isfinite(out)))
    g = np.array(grad[0])
    assert np.all(np.isfinite(g)), f"{name}: {g}"
    # and where the value itself was rejected, the gradient is exactly zero
    if float(np.array(out)[0]) <= -1e29:
        assert np.all(g == 0.0), f"{name}: rejected lane must have zero grad"


def test_the_gradient_guard_leaves_healthy_lanes_untouched():
    """The guard must be invisible wherever nothing failed."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate((1.0, 4.0, 3.0))], n_dim=3)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(block=128))
    u = mx.array(np.array([[0.2, -0.3, 0.5], [0.0, 0.0, 0.0]],
                          dtype=np.float32))
    with mx.stream(mx.gpu):
        _, guarded = mx.vjp(gp, [u], [mx.ones((2,), dtype=mx.float32)])
        _, plain = mx.vjp(gp._raw_call, [u], [mx.ones((2,), dtype=mx.float32)])
        mx.eval(guarded, plain)
    assert np.array_equal(np.array(guarded[0]), np.array(plain[0]))


def test_one_failed_chain_does_not_zero_its_neighbours():
    """The mask is per lane, not per batch."""
    pm = ParamMap(kernel=[KernelParam(link="log", index=0, ref=1e4),
                          KernelParam(link="log", index=1, ref=1e6),
                          KernelParam(link="log", index=2, ref=1e3)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, np.full(N, 1e-4), ag.SHOTerm(), pm,
                   GPPolicy(block=64))
    # The healthy lane sits at sigma = 1e-3 (10 x yerr), rho = 1, Q = 3.
    # It used to be u = 0 (sigma/yerr = 1e8, rho = 1e6), which evaluated to
    # something finite but is far past GPPolicy.kappa_max and now refused.
    healthy = np.log([1e-3 / 1e4, 1.0 / 1e6, 3.0 / 1e3])
    u = mx.array(np.array([[30.0, 30.0, 30.0], healthy], dtype=np.float32))
    assert float(np.array(gp.kappa(u))[1]) < 1e3
    with mx.stream(mx.gpu):
        out = gp(u)
        _, grad = mx.vjp(gp, [u], [mx.ones((2,), dtype=mx.float32)])
        mx.eval(out, grad)
    g = np.array(grad[0])
    assert np.all(g[0] == 0.0), "the failed chain is zeroed"
    assert np.any(g[1] != 0.0), "the healthy chain is not"


@pytest.mark.slow
def test_overdamped_float32_loglike_tracks_float64_at_scale():
    """The term-level fp32 loss of the overdamped corner, at N where it bites.

    A 3e-4 relative error in Q11 is invisible at N = 512 and worth 4.1
    log-likelihood units at N = 16384 -- hundreds of times the scale a
    Metropolis decision runs on.  The conditions that expose it: Q well
    below 1/2, a lag of order ten timescales (so each point is nearly
    independent and the process-noise term carries the whole prediction),
    and off-model data, since on model the per-point errors partly cancel
    and warmup chains are not on model.
    """
    import math

    n, qual, w0dt = 16384, 0.05, 10.0
    rho = 1.0
    dt = w0dt / (2.0 * math.pi / rho)
    t = np.arange(n) * dt
    rng = np.random.default_rng(11)
    y = rng.standard_normal(n)          # white: deliberately off-model
    yerr = np.full(n, 0.1)

    gp = make(ag.SHOTerm(), [1.0, rho, qual], t=t, y=y, yerr=yerr, block=1024)
    hi = float(np.array(gp.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])
    # on the GPU, which is where production runs: MLX's CPU float32 backend
    # carries enough extra precision to hide this one entirely.
    with mx.stream(mx.gpu):
        lo = float(np.array(gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
    assert abs(lo - hi) < 0.05, f"fp32 off by {lo - hi:.3f} log-units"


@pytest.mark.slow
def test_block_size_is_not_a_float32_precision_knob():
    """``block`` trades memory for graph size.  It must not trade accuracy.

    The per-block roots used to be folded left to right, accumulating
    ``ell`` sequentially in float32 at magnitude |log L| ~ 1e5: 0.40
    log-units of error at block 256 against 0.012 for a pairwise tree.  The
    existing block-independence test runs in float64, where the fold is
    harmless, so nothing caught it.
    """
    n = 100_000
    r = np.random.default_rng(5)
    t = np.sort(r.uniform(0.0, 2000.0, n))
    y = r.standard_normal(n)
    yerr = np.full(n, 0.3)

    errs = []
    for blk in (256, 1024, 4096):
        gp = make(ag.SHOTerm(), [1.0, 4.0, 3.0], t=t, y=y, yerr=yerr,
                  block=blk)
        hi = float(np.array(gp.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])
        with mx.stream(mx.gpu):
            lo = float(np.array(gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
        errs.append(lo - hi)

    assert max(abs(e) for e in errs) < 0.05, errs
    # and the choice of block must not move the answer at all
    assert max(errs) - min(errs) < 1e-3, errs


def test_the_blocked_leaf_path_does_not_make_sub_block_a_precision_knob():
    """The compact block-structured leaf (the default at m >= 6) must sum
    ``ell`` the way the dense leaf does.  It folded it sequentially, which
    at m = 6, N = 16384, sub_block 1024 drifted -1.02 log-units while the
    dense path sat at -0.012 for every sub-block (fixed 2026-10-23).
    Compared path against path in float32, so no float64 reference is
    needed: regular cadence, where both paths are otherwise accurate."""
    n = 4096
    term = ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    kp = [0.1, 0.3, 3.0, 0.05, 0.6, 1.0, 0.03, 1.0, 0.7]
    y = sample_ssm(term, kp, 0.1, n, np.random.default_rng(1))
    y = y + 0.001 * np.random.default_rng(2).standard_normal(n)
    t = np.arange(n) * 0.1
    pm = ParamMap(kernel=[KernelParam(link="const", value=x) for x in kp],
                  n_dim=1)
    vals = {}
    for blocked in (False, True):
        for sb in (32, 256):
            gp = GPLogLike(None, t, None, y, np.full(n, 0.001), term, pm,
                           GPPolicy(sub_block=sb, blocked=blocked))
            with mx.stream(mx.gpu):
                vals[blocked, sb] = float(np.array(
                    gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
    ref = vals[False, 32]
    assert max(abs(v - ref) for v in vals.values()) < 5e-3, vals


@pytest.fixture(scope="module", params=["near-duplicates", "uniform"])
def near_duplicate_problem(request):
    """SHO + SHO at sigma/yerr = 100, drawn on-model, on a regular cadence
    with 20 observations moved to 1e-4 cadences after their predecessor, or
    on a uniform-random one (whose closest pairs do the same thing)."""
    from anvilgp.oracle import dense_covariance
    n, dt = 4096, 400 / 4096
    r = np.random.default_rng(101)
    if request.param == "uniform":
        t = np.sort(r.uniform(0.0, 400.0, n))
    else:
        t = np.arange(n) * dt
        idx = np.sort(r.choice(np.arange(10, n - 10), 20, replace=False))
        t[idx] = t[idx - 1] + 1e-4 * dt
        t = np.sort(t)
    term = ag.SHOTerm() + ag.SHOTerm()
    kp = [0.1, 0.3, 3.0, 0.05, 0.6, 1.0]
    yerr = np.full(n, 0.001)
    y = np.linalg.cholesky(dense_covariance(term, np.array(kp), t, yerr)) \
        @ np.random.default_rng(1).standard_normal(n)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=x)
                          for i, x in enumerate(kp)], n_dim=len(kp))
    v0 = np.array([[0.05, -0.03, 0.02, -0.04, 0.03, 0.01]])
    gp = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy())
    h = 1e-5
    vs = np.repeat(v0, 1 + 2 * len(kp), 0)
    for i in range(len(kp)):
        vs[1 + 2 * i, i] += h
        vs[2 + 2 * i, i] -= h
    with mx.stream(mx.cpu):
        f = np.array(gp.hi(mx.array(vs).astype(mx.float64)))
    return (t, y, yerr, term, pm, v0, f[0],
            (f[1::2] - f[2::2]) / (2 * h))


@pytest.mark.parametrize("policy", [
    GPPolicy(), GPPolicy(sub_block=1), GPPolicy(metal=True)],
    ids=["default", "pure-tree", "metal"])
def test_near_duplicate_timestamps_cost_float32_nothing(
        near_duplicate_problem, policy):
    """Two observations very close together make the next block's
    information matrix ~sigma^2/r = 1e4, and the general composition used
    to form its log-normaliser as three ~1e4 cross terms cancelling to
    O(1), through a ``G = I - M J2`` whose own error was ~1e-3.  Before the
    fix (2026-10-24), value error and worst relative gradient error:

        cadence           default         pure tree       metal
        near-duplicates   -3.8   53%      +18.3  260%     -0.002  0.08%
        uniform           -2.9  110%      +58.9  270%     -1.3    2.9%

    and the kappa guard (3.7, at the median lag) saw none of it.  Written
    in the innovation and through the Cholesky factor (`elements.combine`),
    every cell is within 0.007 and 0.4%."""
    from anvilgp.metal import metal_available
    if policy.metal and not metal_available():
        pytest.skip("no Metal device")
    t, y, yerr, term, pm, v0, hi, fd = near_duplicate_problem
    gp = GPLogLike(None, t, None, y, yerr, term, pm, policy)
    with mx.stream(mx.gpu):
        v = mx.array(v0, dtype=mx.float32)
        lo = float(np.array(gp(v))[0])
        grad = np.array(mx.grad(lambda u: mx.sum(gp(u)))(v))[0]
    assert abs(lo - hi) < 0.05, lo - hi
    np.testing.assert_allclose(grad, fd, rtol=1e-2)


def test_float64_path_is_float64_accurate_through_a_log_link():
    """``.hi`` is the reference everything else is measured against.

    Every parity test in this file uses ``link="const"``, which never
    touches an exponential.  Through a log link the path used ``mx.exp``,
    whose float64 implementation in MLX is only float32-accurate (~7e-7
    relative): 2.0e-6 log-units at N = 2048, 4.8e-6 at N = 8192, so about
    sqrt(N) and ~1e-5 at N = 1e5.  Small in absolute terms, but ``.hi`` is
    the reference every precision claim here is measured against, and it
    should not be the limiting term.  ``exp_acc`` is the accurate one, and
    is the native op in float32 so the production path is untouched.
    """
    n = 2048
    r = np.random.default_rng(3)
    t = np.sort(r.uniform(0.0, 100.0, n))
    y = r.standard_normal(n)
    yerr = np.full(n, 0.2)
    refs = (1.0, 4.0, 3.0)
    v = np.array([[0.3, -0.4, 0.7]])

    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=ref)
                          for i, ref in enumerate(refs)], n_dim=3)
    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                   GPPolicy(block=1024, hi_block=1024))
    got = float(np.array(gp.hi_absolute(a64(v)))[0])

    natural = np.array([ref * np.exp(x) for ref, x in zip(refs, v[0])])
    want = dense_loglike(ag.SHOTerm(), natural, t, y, yerr)
    assert abs(got - want) < 1e-8, f"{got - want:+.3e} at |logL| = {abs(want):.0f}"


def test_guard_does_not_re_run_the_forward():
    """The guard sits on the input, so it must add no graph to the forward.

    An earlier version wrapped the whole call and its VJP re-ran the graph
    it was guarding, costing one extra forward on every gradient. That is
    invisible to a correctness test, so pin the structure: the guarded
    forward must be the raw forward, node for node.
    """
    pm = ParamMap.log([1.0, 4.0, 3.0])
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(block=128))
    u = mx.array(np.zeros((2, 3), dtype=np.float32))
    with mx.stream(mx.gpu):
        a = np.array(gp(u))
        b = np.array(gp._raw_call(u))
    assert np.array_equal(a, b), "the guard must be an identity on the value"


def test_the_guard_is_an_identity_on_the_value():
    """The guard sits on the input, so it must add nothing to the forward.

    An earlier version wrapped the whole call and its VJP re-ran the graph
    it was guarding, costing one extra forward on every gradient. That is
    invisible to a correctness test, so pin the structure instead: the
    guarded value must be the raw value, bit for bit.
    """
    pm = ParamMap.log([1.0, 4.0, 3.0])
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(block=128))
    u = mx.array(np.zeros((2, 3), dtype=np.float32))
    with mx.stream(mx.gpu):
        a = np.array(gp(u))
        b = np.array(gp._raw_call(u))
    assert np.array_equal(a, b)


@pytest.mark.slow
@pytest.mark.parametrize("sig_over_yerr", [10.0, 100.0])
def test_float32_holds_up_at_high_gp_amplitude(sig_over_yerr):
    """The precision that matters is at high GP amplitude, on-model.

    ``g = y / sqrt(S)`` is the normalised innovation; the leaf's log-
    normaliser carries ``-g^2/2`` and every composition carries terms of the
    same size that cancel.  At sigma_GP/yerr = 100 on-model data those terms
    are ~1e4, and what survives the cancellation in float32 reaches the
    Metropolis decision scale.  The published error-scaling table missed
    this because its data amplitude was held fixed while the model sigma
    scaled, so ``g^2`` never exceeded ~10.

    This is the *differentiable* path, so it is what a gradient sampler
    would use.  The fused Metal kernel is checked at the same point.
    """
    n, dt, rho, qual = 25_000, 1.0, 100.0, 3.0
    rng = np.random.default_rng(0)
    t = np.arange(n) * dt
    y = sample_ssm(ag.SHOTerm(), [sig_over_yerr, rho, qual], dt, n, rng)
    y = y + rng.standard_normal(n)              # yerr = 1
    yerr = np.full(n, 1.0)
    pm = ParamMap.const([sig_over_yerr, rho, qual])

    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                   GPPolicy(block=4096))
    hi = float(np.array(gp.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])
    with mx.stream(mx.gpu):
        lo = float(np.array(gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
    assert abs(lo - hi) < 0.1, (
        f"MLX float32 off by {lo - hi:+.3f} log-units at "
        f"sigma/yerr = {sig_over_yerr:g}, |logL| = {abs(hi):.0f}")


@pytest.mark.slow
def test_the_joint_high_amplitude_long_correlation_corner():
    """Amplitude and correlation length together, which neither axis alone
    reaches.

    The published error table scans amplitude at a fixed rho/dt and rho/dt
    at a modest amplitude. The *product* is far worse than either: at
    sigma_GP/yerr = 100 AND rho/dt = 1000, N = 32768, the MLX path is off
    by ~1.5 log-units even after the reduction's float32 work, and the
    error is not monotone in `sub_block` (64 measures worse than 32), so it
    is not simple accumulation.

    The fused kernel is fine there (~0.02), and `anvil.validate_precision`
    warns on the MLX path and passes the kernel, so this is a *detected*
    limitation rather than a silent wrong answer. It is a named deliverable
    of M4, whose backward kernel makes the kernel path differentiable and
    so removes the reason to use the MLX path at this size at all.

    The test pins what is true today: the kernel is accurate here, and the
    MLX path is not. If the MLX number improves, tighten it.
    """
    n, dt, qual, sig, rho = 32_768, 0.02, 3.0, 100.0, 1000.0 * 0.02
    rng = np.random.default_rng(0)
    t = np.arange(n) * dt
    y = sample_ssm(ag.SHOTerm(), [sig, rho, qual], dt, n, rng)
    y = y + rng.standard_normal(n)
    yerr = np.full(n, 1.0)
    pm = ParamMap.const([sig, rho, qual])

    ref = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                    GPPolicy(block=4096))
    hi = float(np.array(ref.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])

    kern = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                     GPPolicy(block=4096, metal=True))
    with mx.stream(mx.gpu):
        k32 = float(np.array(kern(mx.zeros((1, 1), dtype=mx.float32)))[0])
        m32 = float(np.array(ref(mx.zeros((1, 1), dtype=mx.float32)))[0])

    assert abs(k32 - hi) < 0.15, (
        f"the kernel must hold this corner; got {k32 - hi:+.3f}")
    # Documented, not asserted-good: if this ever drops below 0.3 the MLX
    # path has been fixed and this guard should be tightened.
    assert abs(m32 - hi) < 5.0, f"MLX unexpectedly far off: {m32 - hi:+.3f}"


def test_the_variance_floor_is_refused_where_it_could_bind():
    """Stage R: `s_floor_rel` is applied by the kernel's forward and not by
    its backward, so at a value where it binds the two disagree (relative
    gradient error 3.3 at s_floor_rel = 1e3).  Below 1 it provably never
    binds -- the innovation variance is at least r_ref since stage 0 -- so
    the policy refuses the range where it would matter.
    """
    GPPolicy(s_floor_rel=0.0)
    GPPolicy(s_floor_rel=0.999)
    for bad in (1.0, 10.0, -1e-3):
        with pytest.raises(ValueError, match="s_floor_rel"):
            GPPolicy(s_floor_rel=bad)


# ---------------------------------------------------------------------------
# exposure-time integration, end to end (M5 stage 3c)
# ---------------------------------------------------------------------------

CONV_CASES = [
    ("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
    ("real", ag.RealTerm(), [1.2, 0.7]),
    ("complex", ag.ComplexTerm(), [1.5, 0.4, 0.6, 1.1]),
    ("sho+real", ag.SHOTerm() + ag.RealTerm(), [1.3, 4.0, 3.0, 1.2, 0.7]),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4]),
]


def _conv_c2(name, p):
    if name == "sho":
        return c2t.SHOTerm(sigma=p[0], rho=p[1], Q=p[2])
    if name == "real":
        return c2t.RealTerm(a=p[0] ** 2, c=p[1])
    if name == "complex":
        return c2t.ComplexTerm(a=p[0], b=p[1], c=p[2], d=p[3])
    if name == "sho+real":
        return (c2t.SHOTerm(sigma=p[0], rho=p[1], Q=p[2])
                + c2t.RealTerm(a=p[3] ** 2, c=p[4]))
    return c2t.RotationTerm(sigma=p[0], period=p[1], Q0=p[2], dQ=p[3], f=p[4])


def _conv_setup(term, p, delta, n=500, spacing=0.05, metal=True, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * spacing
    y = rng.standard_normal(n)
    yerr = np.full(n, 0.3)
    pm = ParamMap.log(list(p))
    gp = GPLogLike(None, t, None, y, yerr, ag.TermConvolution(term, delta), pm,
                   GPPolicy(block=256, hi_block=256, metal=metal))
    return gp, t, y, yerr, pm


@pytest.mark.parametrize("name,term,p", CONV_CASES, ids=[c[0] for c in CONV_CASES])
@pytest.mark.parametrize("delta", [1e-2, 0.04])
@pytest.mark.parametrize("at", ["reference", "off-reference"])
def test_convolved_loglike_matches_celerite2(name, term, p, delta, at):
    """Absolute float64 parity with ``celerite2.terms.TermConvolution``.

    **Off the reference point as well as on it**, which is the whole point
    of the second parameter: at ``u = 0`` the parameter map returns exactly
    its reference values, so a quantity frozen at construction -- or one
    whose *parameter dependence* is wrong while its value is right -- passes
    an on-reference value test.  That is not hypothetical: the convolution's
    amplitudes were once computed in a cancelling form that was accurate at
    the reference point and 9% wrong in the gradient.
    """
    gp, t, y, yerr, pm = _conv_setup(term, p, delta)
    u = (np.zeros(len(p)) if at == "reference"
         else np.linspace(0.3, -0.25, len(p)))
    with mx.stream(mx.cpu):
        pv = np.array(pm.kernel_params(a64(u[None, :])), np.float64)[0]
        ours = float(np.array(gp.hi_absolute(a64(u[None, :])))[0])
    ref = celerite2_loglike(
        c2t.TermConvolution(_conv_c2(name, pv), delta), t, y, yerr)
    assert abs(ours - ref) / abs(ref) < 1e-11, f"{name} delta={delta} {at}"


@pytest.mark.parametrize("name,term,p", CONV_CASES, ids=[c[0] for c in CONV_CASES])
def test_the_convolved_gradient_flows_through_the_diagonal_correction(name, term, p):
    """The gradient, against float64 central differences of ``hi``.

    The gate that matters for this stage, and the one that caught the bug
    the stage shipped with.  A constant added to the diagonal is easy to get
    *value*-correct and gradient-wrong two ways: by detaching it, or by
    computing it in a form whose derivative has lost its digits.  The second
    is what happened -- and note the sweep over ``delta``: at ``delta =
    1e-2`` the error was 7%, at ``1e-3`` it was 247x, and at ``1e-4`` the
    analytic gradient was exactly zero.  A single-``delta`` test at the
    largest width would have called that 7% float32 noise.
    """
    k = len(p)
    for delta in (1e-4, 1e-3, 1e-2):
        gp, *_ = _conv_setup(term, p, delta, metal=False)
        u = mx.array(np.zeros((2, k)), dtype=mx.float32)
        with mx.stream(mx.gpu):
            _, grad = mx.vjp(gp, [u], [mx.ones((2,), dtype=mx.float32)])
            mx.eval(grad)
        got = np.array(grad[0], np.float64)[0]
        fd = np.zeros(k)
        step = 1e-5
        with mx.stream(mx.cpu):
            for i in range(k):
                up, dn = np.zeros((1, k)), np.zeros((1, k))
                up[0, i] += step
                dn[0, i] -= step
                fd[i] = (float(np.array(gp.hi(a64(up)))[0])
                         - float(np.array(gp.hi(a64(dn)))[0])) / (2 * step)
        rel = float(np.max(np.abs(got - fd)) / np.max(np.abs(fd)))
        assert rel < 1e-4, f"{name} delta={delta}: {rel:.2e}"


@pytest.mark.parametrize("name,term,p", CONV_CASES, ids=[c[0] for c in CONV_CASES])
def test_the_convolved_kernel_path_matches_the_reference_path(name, term, p):
    """Fused Metal kernel against ``GPPolicy(metal=False)``, value and gradient.

    The correction reaches the kernel through stage 2's additive diagonal
    slot, on a different route from the reference path's ``r_var``, so it
    needs its own gate.
    """
    k = len(p)
    gk, *_ = _conv_setup(term, p, 1e-2, metal=True)
    gm, *_ = _conv_setup(term, p, 1e-2, metal=False)
    assert gk.metal_eligible()[0], f"{name} fell back: {gk.metal_eligible()[1]}"
    u = mx.array(np.tile(np.linspace(0.2, -0.15, k), (4, 1)), dtype=mx.float32)
    out = []
    for gp in (gk, gm):
        with mx.stream(mx.gpu):
            val, grad = mx.vjp(gp, [u], [mx.ones((4,), dtype=mx.float32)])
            val = gp(u)
            mx.eval(val, grad)
        out.append((float(val[0]), np.array(grad[0], np.float64)[0]))
    (vk, gkv), (vm, gmv) = out
    assert abs(vk - vm) / abs(vm) < 1e-5, f"{name} value: {vk} vs {vm}"
    rel = float(np.max(np.abs(gkv - gmv)) / np.max(np.abs(gmv)))
    assert rel < 1e-3, f"{name} gradient: {rel:.2e}"


@pytest.mark.parametrize("cadence", ["dupes", "alternating"])
def test_convolution_refuses_a_spacing_below_the_exposure_width(cadence):
    """Duplicate and simultaneous timestamps must raise, naming the spacing.

    Inside the exposure width the convolved kernel is not the exponential
    sum the state-space form evaluates; at zero lag, for a term with a real
    component, the observed matrix goes indefinite and the innovation
    variance negative.  This library *supports* duplicate timestamps, so
    the case is reachable and has to be refused rather than trusted.
    """
    rng = np.random.default_rng(1)
    n = 200
    if cadence == "dupes":
        t = np.sort(np.concatenate([rng.uniform(0.0, 8.0, n // 2)] * 2))
    else:
        t = np.sort(np.concatenate([np.arange(n // 2) * 0.3] * 2))
    y, yerr = rng.standard_normal(n), np.full(n, 0.3)
    with pytest.raises(ValueError, match="exposure width"):
        GPLogLike(None, t, None, y, yerr,
                  ag.TermConvolution(ag.SHOTerm(), 0.05),
                  ParamMap.log([1.3, 4.0, 3.0]), GPPolicy())
    # and the same cadence is fine without the convolution
    GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
              ParamMap.log([1.3, 4.0, 3.0]), GPPolicy())


@pytest.mark.parametrize("qual", [0.5, 0.4, 0.1])
def test_an_overdamped_convolved_sho_now_evaluates(qual):
    """``Q <= 1/2`` used to be refused at construction; now it runs.

    The refusal existed because the coefficient route could not represent the
    negative amplitude in the overdamped decomposition, and the alternative
    was silent: the non-finite guard returned -1e30 with an exactly zero
    gradient, a flat plateau. The state-space route needs no decomposition,
    so the parameters are ordinary ones -- checked against the dense float64
    oracle, and the gradient checked finite, since a flat plateau is what
    this replaced.
    """
    rng = np.random.default_rng(2)
    n = 200
    delta = 0.02
    t = np.arange(n) * 0.05
    y, yerr = rng.standard_normal(n), np.full(n, 0.3)
    p = [1.3, 4.0, qual]
    cv = ag.TermConvolution(ag.SHOTerm(), delta)
    gp = GPLogLike(None, t, None, y, yerr, cv, ParamMap.log(p),
                   GPPolicy(block=256, hi_block=256, metal=False))
    with mx.stream(mx.cpu):
        hi = float(np.array(gp.hi_absolute(a64(np.zeros((1, 3)))))[0])
    ref = dense_loglike(cv, np.asarray(p, dtype=np.float64), t, y, yerr)
    assert abs(hi - ref) / abs(ref) < 1e-10, f"Q={qual}: {hi} vs {ref}"

    u = mx.zeros((2, 3), dtype=mx.float32)
    with mx.stream(mx.gpu):
        val = gp(u)
        _, grad = mx.vjp(gp, [u], [mx.ones((2,), dtype=mx.float32)])
        mx.eval(val, grad)
    g = np.array(grad[0], np.float64)
    assert np.all(np.isfinite(g)), f"Q={qual}: non-finite gradient"
    assert float(np.max(np.abs(g))) > 1e-6, (
        f"Q={qual}: zero gradient -- the flat -1e30 plateau is back")
    assert float(val[0]) > -1e29, f"Q={qual}: sanitised value"


CONV_NEW = [
    ("matern32", ag.Matern32Term(), [1.1, 2.5]),
    ("matern52", ag.Matern52Term(), [1.1, 2.5]),
    ("sho-overdamped", ag.SHOTerm(), [1.3, 4.0, 0.1]),
]


@pytest.mark.parametrize("name,term,p", CONV_NEW, ids=[c[0] for c in CONV_NEW])
@pytest.mark.parametrize("delta", [1e-3, 1e-2, 0.1])
def test_the_newly_convolvable_terms_match_the_dense_oracle(name, term, p, delta):
    """The three shapes the coefficient route could not reach, end to end.

    Against the dense float64 Cholesky, which is the only oracle available:
    celerite2 has no Matern-5/2, only approximates Matern-3/2, and cannot
    convolve an overdamped SHO either.

    A convolved kernel whose exponential-sum continuation is not positive
    definite *cannot* have a PSD process noise -- if it could, that
    continuation would be a valid covariance -- so ``Q`` here is indefinite
    (min eigenvalue -1e-12 to -3e-2 relative, growing with ``delta``).  The
    algebra is exact anyway on the formal path, which drops the positivity
    clamps for exactly this reason, so the tolerance is the 1e-10 every
    other convolved term holds.  (It was 1e-5 while the clamps were on, and
    this grid's large error bars hid how far they could move the answer --
    see `test_a_convolved_term_is_exact_or_refused_never_wrong`.)
    """
    rng = np.random.default_rng(11)
    n = 200
    t = np.arange(n) * delta * 1.8
    y, yerr = rng.standard_normal(n), np.full(n, 0.3)
    cv = ag.TermConvolution(term, delta)
    gp = GPLogLike(None, t, None, y, yerr, cv, ParamMap.log(list(p)),
                   GPPolicy(block=256, hi_block=256, metal=False))
    with mx.stream(mx.cpu):
        hi = float(np.array(gp.hi_absolute(a64(np.zeros((1, len(p))))))[0])
    ref = dense_loglike(cv, np.asarray(p, dtype=np.float64), t, y, yerr)
    assert abs(hi - ref) / abs(ref) < 1e-10, f"{name} delta={delta}"


#: (term, natural parameters in units of (sigma, delta, 1) -- a RealTerm's
#: rate in units of 1/delta -- and yerr).  Every point here once came back
#: wrong, by the amount noted, and was then refused until the reduction
#: stopped factoring the formal model's indefinite covariances.  Spacing =
#: exposure, the tightest cadence the convolution allows.
_FORMAL_GRID = [
    ("real-tau10-snr30", ag.RealTerm(), (1.0, 0.1), 3.3e-5),     # +36
    ("real-tau100-snr100", ag.RealTerm(), (1.0, 0.01), 1e-5),    # +500
    ("sho-r2-Q0.71", ag.SHOTerm(), (1.0, 2.0, 0.7071), 3e-4),     # -10
    ("sho-r3-Q0.5", ag.SHOTerm(), (1.0, 3.0, 0.5), 3e-4),         # +0.57
    ("sho-r4-Q0.3", ag.SHOTerm(), (1.0, 4.0, 0.3), 3e-3),         # -4.7
    ("sho-r3-Q3", ag.SHOTerm(), (1.0, 3.0, 3.0), 3e-4),           # control
    ("matern32-r1", ag.Matern32Term(), (1.0, 1.0), 3e-4),         # +0.24
    ("matern52-r3-snr100", ag.Matern52Term(), (1.0, 3.0), 1e-5),
    ("matern52-r10", ag.Matern52Term(), (1.0, 10.0), 3e-4),       # +0.46
    ("matern52-r20", ag.Matern52Term(), (1.0, 20.0), 3e-4),       # +0.09
]


@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_a_convolved_term_matches_the_dense_oracle_where_it_used_to_fail(metal):
    """Every path matches the dense oracle on the convolution's hard cases.

    A convolved term's state-space form is only *formally* Gaussian: for a
    ``RealTerm`` above signal-to-noise ~10, a low-``Q`` SHO or a Matern
    within a few exposures, its ``P_inf``, ``Q``, the filtered covariances
    and the leaf fold's ``alpha`` are indefinite or negative, not by
    rounding.  The algebra is exact anyway -- the formal model reproduces
    the data's covariance -- so long as nothing assumes positivity.  Three
    guards did: the ``h Q h^T`` clamp, the replay's predicted-variance
    clamp, and the Cholesky pivot floor in the block composition, each
    silently substituting a different number (up to 500 log-units, some too
    *high*, which attracts chains).  The formal path drops the clamps, takes
    ``log|alpha|``, and composes blocks through a pivoted direct inverse.

    The assertion is exactness, not refusal: ``.hi`` within 1e-9 of the
    oracle and the float32 path within float32 noise, at every point.  The
    data are drawn from the model, as a posterior's are.  (Far off-model --
    white noise against a long timescale -- a convolved term's float32 path
    is 50-100x less accurate than the plain term's at the same ``kappa``,
    from the cancellation in ``Q = P - Phi P Phi^T``; that is a separate,
    older issue, recorded in docs/kernels.md.)
    """
    n, dl = 300, 0.02
    t = np.arange(n) * dl
    rng = np.random.default_rng(23)
    for name, base, p_units, ye in _FORMAL_GRID:
        unit = ([1e-3, 1.0 / dl] if isinstance(base, ag.RealTerm)
                else [1e-3, dl, 1.0])
        p = np.array(p_units, dtype=np.float64) * np.array(unit[:len(p_units)])
        cv = ag.TermConvolution(base, dl)
        yerr = np.full(n, ye)
        kmat = cv.value_np(p, np.abs(t[:, None] - t[None, :]))
        y = np.linalg.cholesky(kmat + np.diag(yerr ** 2)) @ rng.standard_normal(n)
        gp = GPLogLike(None, t, None, y, yerr, cv, ParamMap.log(list(p)),
                       GPPolicy(metal=metal, kappa_max=None))
        u = np.zeros((1, len(p)))
        ref = dense_loglike(cv, p, t, y, yerr)
        with mx.stream(mx.cpu):
            hi = float(np.array(gp.hi_absolute(a64(u)))[0])
        lo = float(np.array(gp.log_prob_absolute(
            mx.array(u, dtype=mx.float32)))[0])
        assert abs(hi - ref) < 1e-9 * abs(ref), f"{name}: .hi {hi} vs {ref}"
        tol = 2e-6 * abs(ref) + 5e-3
        assert abs(lo - ref) < tol, f"{name}: float32 {lo} vs {ref}"


@pytest.mark.parametrize("case", ["matern52-r10", "real-snr30"])
def test_a_formal_kernel_gradient_matches_float64(case):
    """The kernel's formal variant differentiates the right function.

    Its backward replays the sequential filter (whose clamp the variant also
    drops) and its seeds come from the same direct-inverse composition as
    the forward; a mismatch anywhere would give the gradient of a different
    function.  Checked against float64 central differences of ``.hi``, at
    two points the old code got wrong: a convolved Matern-5/2 at ``rho`` =
    10 exposures (variance clamp) and a convolved ``RealTerm`` at
    signal-to-noise 30 (indefinite composition), off the reference point.
    """
    n, dl = 400, 0.02
    t = np.arange(n) * dl
    rng = np.random.default_rng(5)
    if case == "matern52-r10":
        cv, p, ye = ag.TermConvolution(ag.Matern52Term(), dl), [1e-3, 10 * dl], 3e-4
    else:
        cv, p, ye = ag.TermConvolution(ag.RealTerm(), dl), [1e-3, 0.1 / dl], 3.3e-5
    yerr = np.full(n, ye)
    kmat = cv.value_np(np.array(p), np.abs(t[:, None] - t[None, :]))
    y = np.linalg.cholesky(kmat + np.diag(yerr ** 2)) @ rng.standard_normal(n)
    pm = ParamMap.log(p)
    gk = GPLogLike(None, t, None, y, yerr, cv, pm, GPPolicy(metal=True))
    gm = GPLogLike(None, t, None, y, yerr, cv, pm, GPPolicy(metal=False))
    assert gk.metal_eligible()[0]
    u0 = np.array([[0.05, -0.1]])
    u = mx.array(u0, dtype=mx.float32)
    grads = [np.array(mx.grad(lambda w, g=g: mx.sum(g(w)))(u), np.float64)[0]
             for g in (gk, gm)]
    fd, h = np.zeros(2), 3e-4
    with mx.stream(mx.cpu):
        for i in range(2):
            e = np.zeros((1, 2))
            e[0, i] = h
            fd[i] = (float(np.array(gm.hi(a64(u0 + e)))[0])
                     - float(np.array(gm.hi(a64(u0 - e)))[0])) / (2 * h)
    for label, g in zip(("kernel", "mlx"), grads):
        rel = np.abs(g - fd).max() / np.abs(fd).max()
        assert rel < 1e-3, f"{case} {label} vs float64 FD: {rel:.2e}"


def test_a_convolved_term_cannot_join_a_sum():
    """``TermConvolution(a) + b`` was silently the wrong kernel; now it raises.

    The sum kept ``P_inf = I`` and no zero-lag correction for every member,
    and the likelihood looked for the exposure width on the term it was
    given rather than inside it -- so the convolved member was scored as
    neither itself nor its base term (0.04 to 5.3 log-units off, on all
    three paths).  One detector integrates the whole signal, so the
    supported form, ``TermConvolution(a + b)``, is also the physical one.
    """
    cv = ag.TermConvolution(ag.SHOTerm(), 0.02)
    for build in (lambda: cv + ag.SHOTerm(), lambda: ag.SHOTerm() + cv,
                  lambda: ag.TermSum(cv, ag.Matern32Term())):
        with pytest.raises(TypeError, match="convolve the sum"):
            build()
    ok = ag.TermConvolution(ag.SHOTerm() + ag.SHOTerm(), 0.02)
    assert ok.n_state == 4


@pytest.mark.parametrize("metal", [False, True], ids=["mlx", "kernel"])
def test_a_chain_outside_the_series_range_is_refused(metal):
    """``||F delta|| <= MAX_FDELTA`` holds per chain, not just at ``v = 0``.

    Construction checks the reference point only, and a chain can walk
    anywhere: at ``rho = 2 dt``, ``Q = 0.05`` -- inside the default prior box
    -- the norm is 63, the series has not converged and the value was 432
    log-units wrong in float64.  Here the reference is well inside the bound
    (``c = 0.005``, norm 0.1) and the second chain is outside it (``c =
    0.6``, norm 12).

    The guard is checked directly as well as end to end, so the test shows
    that the guard is what fires.
    """
    rng = np.random.default_rng(3)
    n = 100
    t = np.arange(n) * 20.0
    y, yerr = rng.standard_normal(n), np.full(n, 0.3)
    cv = ag.TermConvolution(ag.RealTerm(), 20.0)
    p = [1.2, 0.005]
    gp = GPLogLike(None, t, None, y, yerr, cv, ParamMap.log(p),
                   GPPolicy(metal=metal))
    u = mx.array([[0.0, 0.0], [0.0, float(np.log(120.0))]], dtype=mx.float32)
    guarded = np.array(gp._refuse_outside_series(
        u, mx.zeros((2,), dtype=mx.float32)))
    assert guarded[0] == 0.0 and np.isnan(guarded[1]), guarded
    out = np.array(gp.log_prob_absolute(u), np.float64)
    ref = dense_loglike(cv, np.array(p), t, y, yerr)
    assert abs(out[0] - ref) < 1e-3, (out[0], ref)
    assert out[1] < -1e29, out


def test_convolution_refuses_too_wide_an_exposure_for_the_series():
    """``||F delta|| > MAX_FDELTA`` is refused, not silently degraded.

    The matrix functions are series in ``(F delta)^2``; the coefficients are
    positive but ``F delta`` is not, so far enough out the alternating powers
    cost float32 digits. Unlike the amplitude cancellation this design
    replaced, no algebraic rewrite removes it -- so it is a bound, and the
    bound is enforced.
    """
    from anvilgp import terms as _t
    rng = np.random.default_rng(3)
    n = 100
    t = np.arange(n) * 20.0
    y, yerr = rng.standard_normal(n), np.full(n, 0.3)
    # c = 1 per unit time, delta = 20 -> ||F delta|| = 20 > 8
    with pytest.raises(ValueError, match="F delta"):
        GPLogLike(None, t, None, y, yerr,
                  ag.TermConvolution(ag.RealTerm(), 20.0),
                  ParamMap.log([1.2, 1.0]), GPPolicy())
    # and just inside the bound is accepted
    GPLogLike(None, t, None, y, yerr,
              ag.TermConvolution(ag.RealTerm(), 20.0),
              ParamMap.log([1.2, 0.2]), GPPolicy())
    assert _t.MAX_FDELTA == 8.0


def test_the_convolved_diagonal_correction_is_not_dead():
    """Pin that the correction *moves* the likelihood and its gradient.

    Two failure modes a parity test can miss: dropping the correction
    entirely (it is small), and detaching it from the parameter vector.  So
    compare against the same model with the correction forced to zero: both
    the value and the ``sigma`` gradient must differ by more than float32
    noise.
    """
    term = ag.RealTerm()          # a corner, so the correction is O(delta)
    p = [1.2, 0.7]
    gp, *_ = _conv_setup(term, p, 0.04, metal=False)
    u = mx.array(np.zeros((2, 2)), dtype=mx.float32)

    def val_and_grad(g):
        with mx.stream(mx.gpu):
            v = g(u)
            _, gr = mx.vjp(g, [u], [mx.ones((2,), dtype=mx.float32)])
            mx.eval(v, gr)
        return float(v[0]), np.array(gr[0], np.float64)[0]

    v_on, g_on = val_and_grad(gp)
    original = type(gp.term).diag_correction
    try:
        type(gp.term).diag_correction = lambda self, pp: None
        v_off, g_off = val_and_grad(gp)
    finally:
        type(gp.term).diag_correction = original
    assert abs(v_on - v_off) / abs(v_on) > 1e-5, (
        f"the correction does not move the value: {v_on} vs {v_off}")
    assert float(np.max(np.abs(g_on - g_off))
                 / np.max(np.abs(g_on))) > 1e-4, (
        "the correction does not move the gradient -- it is detached")
