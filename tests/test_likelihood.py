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
from anvilgp.oracle import celerite2_loglike, dense_loglike

mx.set_default_device(mx.cpu)
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
    u = mx.array(np.array([[30.0, 30.0, 30.0], [0.0, 0.0, 0.0]],
                          dtype=np.float32))
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


def test_float64_path_is_float64_accurate_through_a_log_link():
    """``.hi`` is the reference everything else is measured against.

    Every parity test in this file uses ``link="const"``, which never
    touches an exponential.  Through a log link the path used ``mx.exp``,
    whose float64 implementation in MLX is only float32-accurate (~7e-7
    relative): 2.0e-6 log-units at N = 2048, which scales to ~1e-2 at
    N = 1e5 -- the same size as the float32 error ``.hi`` exists to
    measure.  ``exp_acc`` is the accurate one, and is the native op in
    float32 so the production path is untouched.
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
