"""The fused Metal scan kernel (M3 pass 1).

The kernel is a *reimplementation* of the composition in Metal Shading
Language, so it gets adjudicated the same way the MLX path was: against the
dense float64 Cholesky, and against the MLX path itself, which is already
pinned to celerite2.  Agreement has to hold for every term type, every block
length, and with a jitter and a mean model in play -- a kernel that is right
only for the configuration it was written against is not much use.
"""

import numpy as np
import mlx.core as mx
import pytest

import anvilgp as ag
from anvilgp.likelihood import (GPLogLike, GPPolicy, KernelParam,
                                ParamMap, a64)
from anvilgp.oracle import dense_loglike

metal = pytest.importorskip("anvilgp.metal")
pytestmark = pytest.mark.skipif(not metal.metal_available(),
                                reason="mx.fast.metal_kernel unavailable")

N = 384
_rng = np.random.default_rng(11)
T = np.arange(N) * 0.05                     # regular cadence, as real data
YERR = np.full(N, 0.3)
Y = _rng.standard_normal(N) * 0.9

TERMS = [
    ("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
    ("sho_critical", ag.SHOTerm(), [1.3, 4.0, 0.5]),
    ("sho_overdamped", ag.SHOTerm(), [1.3, 4.0, 0.1]),
    ("sho_highQ", ag.SHOTerm(), [1.3, 4.0, 100.0]),
    ("matern32", ag.Matern32Term(), [1.1, 2.5]),
    ("matern52", ag.Matern52Term(), [1.1, 2.5]),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4]),
    ("real", ag.RealTerm(), [0.8, 0.5]),
]


def _build(term, kp, metal_on, blk=128, jitter=None, t=T, y=Y, yerr=YERR):
    kernel = [KernelParam(link="const", value=float(x)) for x in kp]
    jit = [KernelParam(link="const", value=float(jitter))] if jitter else []
    pm = ParamMap(kernel=kernel, n_dim=1, jitter=jit)
    return GPLogLike(None, t, None, y, yerr, term, pm,
                     GPPolicy(block=1024, hi_block=1024, metal=metal_on,
                              metal_block=blk))


def _f32(gp, n_chains=3):
    with mx.stream(mx.gpu):
        out = np.array(gp(mx.zeros((n_chains, 1), dtype=mx.float32)),
                       dtype=np.float64)
    return out


@pytest.mark.parametrize("name,term,kp", TERMS, ids=[c[0] for c in TERMS])
def test_kernel_matches_the_mlx_path_and_the_dense_oracle(name, term, kp):
    got = _f32(_build(term, kp, True))
    ref = _f32(_build(term, kp, False))
    gp = _build(term, kp, False)
    dense = dense_loglike(term, np.array(kp), T, Y, YERR) - gp.log_norm_const
    assert np.ptp(got) == 0.0, "identical chains must give identical answers"
    assert abs(got[0] - ref[0]) < 0.02, (got[0], ref[0])
    assert abs(got[0] - dense) < 0.05, (got[0], dense)


@pytest.mark.parametrize("blk", [1, 7, 16, 64, 128, 512, 1024])
def test_block_length_is_not_a_numerical_knob(blk):
    """Every thread length must give the same answer, including blk = 1
    (one observation per thread) and blk > N (one thread for everything)."""
    ref = _f32(_build(ag.SHOTerm(), [1.3, 4.0, 3.0], False))[0]
    got = _f32(_build(ag.SHOTerm(), [1.3, 4.0, 3.0], True, blk=blk))[0]
    assert abs(got - ref) < 0.02, (blk, got, ref)


def test_jitter_enters_the_kernels_observation_variance():
    got = _f32(_build(ag.SHOTerm(), [1.3, 4.0, 3.0], True, jitter=0.45))[0]
    gp = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], False, jitter=0.45)
    dense = dense_loglike(ag.SHOTerm(), np.array([1.3, 4.0, 3.0]), T, Y, YERR,
                          jitter=0.45) - gp.log_norm_const
    assert abs(got - dense) < 0.05


def test_duplicate_timestamps_and_gaps():
    """dt = 0 and large gaps both go through the lag table, so the kernel
    needs no special case -- but the table had better be built right."""
    t = T.copy()
    t[10] = t[9]                       # duplicate
    t[200:] += 3.0                     # gap
    t = np.sort(t)
    got = _f32(_build(ag.SHOTerm(), [1.3, 4.0, 3.0], True, t=t))[0]
    gp = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], False, t=t)
    dense = dense_loglike(ag.SHOTerm(), np.array([1.3, 4.0, 3.0]), t, Y,
                          YERR) - gp.log_norm_const
    assert abs(got - dense) < 0.05


def test_irregular_cadence_still_correct():
    """Every lag distinct: the table degenerates to one row per point. Slow,
    but it must not be wrong."""
    t = np.sort(_rng.uniform(0.0, 20.0, N))
    got = _f32(_build(ag.SHOTerm(), [1.3, 4.0, 3.0], True, t=t))[0]
    gp = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], False, t=t)
    dense = dense_loglike(ag.SHOTerm(), np.array([1.3, 4.0, 3.0]), t, Y,
                          YERR) - gp.log_norm_const
    assert abs(got - dense) < 0.05


def test_mean_model_is_subtracted_before_the_kernel():
    slope = 0.3

    def mean_fn(v_mean, x):
        return v_mean[:, 0:1] * x[None, :]

    outs = []
    for metal_on in (False, True):
        pm = ParamMap(kernel=[KernelParam(link="const", value=v)
                              for v in (1.3, 4.0, 3.0)], n_dim=1,
                      mean=slice(0, 1))
        gp = GPLogLike(mean_fn, T, T, Y, YERR, ag.SHOTerm(), pm,
                       GPPolicy(block=1024, metal=metal_on, metal_block=128))
        with mx.stream(mx.gpu):
            outs.append(float(np.array(
                gp(mx.array([[slope]], dtype=mx.float32)))[0]))
    ref = dense_loglike(ag.SHOTerm(), np.array([1.3, 4.0, 3.0]), T,
                        Y - slope * T, YERR)
    gp0 = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], False)
    assert abs(outs[1] - outs[0]) < 0.02
    assert abs(outs[1] - (ref - gp0.log_norm_const)) < 0.05


def test_chains_stay_independent_under_the_kernel():
    refs = [1.0, 4.0, 3.0]
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(refs)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(metal=True, metal_block=64))
    with mx.stream(mx.gpu):
        u = mx.array(_rng.standard_normal((8, 3)).astype(np.float32) * 0.2)
        full = np.array(gp(u))
        alone = np.array(gp(u[5:6]))
    assert abs(full[5] - alone[0]) < 1e-3 * max(1.0, abs(alone[0]))


def test_eligibility_is_reported_not_silently_ignored():
    """A configuration the kernel cannot serve must say so, so nobody
    concludes it is on when it is not.

    Asking for the kernel and quietly getting the MLX path is a ~40x
    performance surprise with no signal at all, so it warns at construction
    as well as answering `metal_eligible()`.
    """
    four = ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    pm = ParamMap(kernel=[KernelParam(link="const", value=1.0)] * 12, n_dim=1)
    with pytest.warns(RuntimeWarning, match="cannot be honoured"):
        gp = GPLogLike(None, T, None, Y, YERR, four, pm, GPPolicy(metal=True))
    ok, why = gp.metal_eligible()
    assert not ok and "register budget" in why

    off = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], False)
    ok, why = off.metal_eligible()
    assert not ok and "off" in why

    on = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], True)
    assert on.metal_eligible()[0]


def test_psd_failure_stays_finite_under_the_kernel():
    pm = ParamMap(kernel=[KernelParam(link="log", index=0, ref=1e4),
                          KernelParam(link="log", index=1, ref=1e6),
                          KernelParam(link="log", index=2, ref=1e3)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, np.full(N, 1e-4), ag.SHOTerm(), pm,
                   GPPolicy(metal=True, metal_block=64))
    with mx.stream(mx.gpu):
        out = gp(mx.zeros((4, 3), dtype=mx.float32))
        mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)))


def test_per_instrument_offsets_under_the_kernel():
    """The one reachable Metal branch nothing covered.

    ``_evaluate_metal`` subtracts per-instrument offsets on its own code
    path (it does not go through ``leaf_inputs``), so a bug there would not
    show up in any MLX test.  Per-instrument *jitter* is refused by the
    kernel; offsets are not, and must match.
    """
    inst = (np.arange(N) % 3).astype(np.int32)
    kernel = [KernelParam(link="const", value=v) for v in (1.3, 4.0, 3.0)]
    offs = [KernelParam(link="linear", index=i, ref=0.0, scale=1.0)
            for i in range(3)]

    def build(metal_on):
        pm = ParamMap(kernel=kernel, n_dim=3, offset=offs, instrument=inst,
                      jitter=[KernelParam(link="const", value=0.05)])
        return GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                         GPPolicy(block=1024, hi_block=1024, metal=metal_on,
                                  metal_block=128))

    gp_m, gp_x = build(True), build(False)
    assert gp_m.metal_eligible()[0]
    u = mx.array(np.array([[0.3, -0.2, 0.1], [0.0, 0.0, 0.0]],
                          dtype=np.float32))
    with mx.stream(mx.gpu):
        got = np.array(gp_m(u))
        want = np.array(gp_x(u))
    assert np.max(np.abs(got - want)) < 0.02, (got, want)
    # and the offsets actually do something, so this is not a null test
    with mx.stream(mx.gpu):
        zero = np.array(gp_m(mx.zeros((1, 3), dtype=mx.float32)))
    assert abs(float(zero[0]) - float(got[0])) > 1.0


@pytest.mark.parametrize("n_data,want", [(10_000, 1024), (30_000, 2048),
                                         (100_000, 8192)])
def test_auto_metal_block_matches_the_measured_optimum(n_data, want):
    """The default must track N, not sit at a constant.

    Measured on an M2 Max at 1024 chains, whole compiled path, interleaved:
    the best block is 1024 at N = 1e4, 2048 at 3e4 and 8192 at 1e5. A fixed
    1024 was 1.26x and 1.29x off at the two larger sizes.
    """
    from anvilgp.likelihood import _auto_metal_block
    assert _auto_metal_block(n_data, 2) == want


@pytest.mark.parametrize("n_state,cap", [(2, 8192), (3, 2048), (4, 1024)])
def test_auto_metal_block_caps_long_blocks_at_high_state_dimension(n_state,
                                                                   cap):
    """A thread walks its whole block, so block length is a time limit.

    Per-observation work grows like m^3, and macOS kills a command buffer
    that runs too long. At N = 1e5 and 512 chains the uncapped rule chose
    8192 for every m; at m = 4 that is not slow, it is
    "[METAL] Command buffer execution failed: Impacting Interactivity",
    and 4096 fails too while 2048 failed in one run and not another. The
    cap keeps every state dimension inside the watchdog with margin.
    """
    from anvilgp.likelihood import _auto_metal_block
    assert _auto_metal_block(1_000_000, n_state) == cap
    assert _auto_metal_block(10_000, n_state) <= cap


@pytest.mark.slow
def test_the_kernel_survives_a_large_high_state_problem():
    """The configuration that the uncapped block rule killed."""
    n, n_chains = 100_000, 512
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 300.0, n))
    y = rng.standard_normal(n) * 0.9
    refs = [1.0, 7.0, 3.0, 1.0, 0.4]
    gp = GPLogLike(None, t, None, y, np.full(n, 0.3), ag.RotationTerm(),
                   ParamMap.log(refs), GPPolicy(block=4096, metal=True))
    assert gp.metal_block <= 1024, "m = 4 must not get a long block"
    u = mx.array(rng.standard_normal((n_chains, 5)).astype(np.float32) * 0.01)
    with mx.stream(mx.gpu):
        out = np.array(gp(u))
    assert np.all(np.isfinite(out))


def test_metal_block_is_resolved_on_the_instance():
    """`GPPolicy.metal_block=None` must become a concrete value."""
    auto = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(),
                     ParamMap.const([1.3, 4.0, 3.0]), GPPolicy(metal=True))
    assert isinstance(auto.metal_block, int) and 256 <= auto.metal_block <= 8192
    fixed = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(),
                      ParamMap.const([1.3, 4.0, 3.0]),
                      GPPolicy(metal=True, metal_block=512))
    assert fixed.metal_block == 512


@pytest.mark.slow
@pytest.mark.parametrize("sig_over_yerr", [30.0, 100.0])
def test_the_kernel_holds_high_amplitude_on_model(sig_over_yerr):
    """The kernel's own g^2 round trip, the counterpart of the MLX fix.

    The kernel formed `ell2 = -(log(S/r) + g^2)/2` and the composition
    immediately added `+g^2/2` back. On-model at sigma_GP/yerr = 100 that
    round-trips a quantity near 1e4 against a log-determinant of order 1,
    and float32 loses four digits at every observation. Carrying the
    log-determinant half separately is exact and measured 11x better at the
    default block: -0.018 -> +0.002.
    """
    from anvilgp.oracle import sample_ssm

    n, dt, qual, rho = 25_000, 1.0, 3.0, 100.0
    rng = np.random.default_rng(0)
    t = np.arange(n) * dt
    y = sample_ssm(ag.SHOTerm(), [sig_over_yerr, rho, qual], dt, n, rng)
    y = y + rng.standard_normal(n)
    yerr = np.full(n, 1.0)
    pm = ParamMap.const([sig_over_yerr, rho, qual])

    ref = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                    GPPolicy(block=4096))
    hi = float(np.array(ref.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])
    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                   GPPolicy(block=4096, metal=True))
    with mx.stream(mx.gpu):
        lo = float(np.array(gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
    assert abs(lo - hi) < 0.02, (
        f"kernel off by {lo - hi:+.4f} at sigma/yerr = {sig_over_yerr:g}")


def test_the_differentiation_boundary_encloses_the_reduction():
    """M4 requirement: the custom_function's OUTPUT is one scalar per chain.

    If it returned the six element fields per block instead, its VJP would
    have to accept cotangents on all six -- not the problem `adjoint.py`
    solves.  An earlier version of this test spied on `scan_loglike`'s
    output, which is one scalar per chain whether or not the reduction is
    inside the boundary; it passed with the reduction moved back outside.
    This one captures what the decorated function itself returns.
    """
    import anvilgp.metal as metal_mod

    seen = {}
    real_cf = mx.custom_function

    def spy_cf(fn):
        def wrapped(*a, **kw):
            out = fn(*a, **kw)
            seen["custom_function_output"] = (
                out.shape if hasattr(out, "shape") else [o.shape for o in out])
            return out
        wrapped.__name__ = fn.__name__
        return real_cf(wrapped)

    n_chains = 6
    metal_mod.mx.custom_function = spy_cf
    try:
        gp = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], True)
        with mx.stream(mx.gpu):
            mx.eval(gp(mx.zeros((n_chains, 1), dtype=mx.float32)))
    finally:
        metal_mod.mx.custom_function = real_cf

    assert seen["custom_function_output"] == (n_chains,), (
        f"the custom_function must return (n_chains,); it returned "
        f"{seen['custom_function_output']}")



@pytest.mark.parametrize("name,term,kp", TERMS, ids=[c[0] for c in TERMS])
@pytest.mark.parametrize("blk", [64, 256])
def test_in_kernel_fold_matches_the_mlx_tree(name, term, kp, blk):
    """Pass 2 is a kernel now; it must fold exactly what the tree folded.

    The fold carries only (b, C, ell). It can because the first
    observation has no predecessor, so the first block's element has
    A = J = eta = 0 and `combine` preserves all three -- A_out = A2 G A1,
    eta_out = A1^T gt_eta + eta1 and J_out = A1^T jt A1 + J1 each vanish
    when the accumulator's do. If that ever stops holding, this test is
    what catches it.
    """
    from anvilgp.elements import tree_reduce
    from anvilgp.metal import reduce_blocks, scan_blocks

    gp = _build(term, kp, True, blk=blk)
    n_chains = 4
    v = mx.array(_rng.standard_normal((n_chains, 1)).astype(np.float32) * 0.0)
    m = term.n_state
    phi, q, h = term.matrices(gp.param_map.kernel_params(v),
                              gp._metal_uniq_dt)
    pt = mx.concatenate([mx.zeros((n_chains, 1, m, m), dtype=v.dtype), phi],
                        axis=1)
    qt = mx.concatenate(
        [mx.broadcast_to(mx.eye(m, dtype=v.dtype), (n_chains, 1, m, m)), q],
        axis=1)
    r = mx.broadcast_to(gp._y32[:, None], (gp.n_data, n_chains))
    j2 = mx.zeros((n_chains, 0), dtype=v.dtype)

    with mx.stream(mx.gpu):
        el = scan_blocks(pt, qt, gp._metal_lag, h, r, gp._r_ref32, j2,
                         gp.policy.s_floor_rel, blk)
        tree = np.array(tree_reduce(el).ell, dtype=np.float64)
        kern = np.array(reduce_blocks(el), dtype=np.float64)

    assert el.b.shape[1] > 1, "the fixture must have several blocks"
    scale = max(np.max(np.abs(tree)), 1.0)
    assert np.max(np.abs(tree - kern)) / scale < 1e-5, (tree, kern)


# -- stage E: the reverse sweep in the kernel ------------------------------

_BWD_TERMS = [
    ("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
    ("matern32", ag.Matern32Term(), [1.1, 2.5]),
    ("matern52", ag.Matern52Term(), [1.1, 2.5]),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4]),
]


def _fwd_setup(term, p, k, blk, n_chains=3, cadence="regular", seed=1):
    """Lag table, block elements and the float64 reference adjoints."""
    from anvilgp.adjoint import filter_backward, filter_forward
    from anvilgp.metal import scan_blocks

    rng = np.random.default_rng(seed)
    if cadence == "gapped":
        t = np.sort(rng.uniform(0.0, 8.0, k))
        t[k // 2:] += 20.0
    elif cadence == "dupes":
        t = np.sort(np.concatenate([rng.uniform(0.0, 8.0, (k + 1) // 2)] * 2))
        t = t[:k]
    elif cadence == "distinct":
        # Every spacing its own lag, so the lag table degenerates to one
        # row per observation -- what makes the backward's per-lag buffer
        # N-sized.  Ground-based photometry looks like this.  The mean
        # spacing is "regular"'s 0.3, deliberately: jittering the spacings
        # is then the *only* thing that differs from "regular", and the
        # comparison isolates the degenerate table.  (Drawing the times
        # uniformly over a short span instead would also shrink dt/rho to
        # ~1e-3, which is its own ill-conditioning and degrades the
        # regular cadence nearly as much -- measured 7.4e-5 on `dh` at
        # m = 4.  Two effects, one fixture; not what this tests.)
        t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.2, 0.4, k - 1))])
    elif cadence == "alternating":
        # two distinct lags (0 and the cadence) and a run length of one:
        # the flush-on-change accumulator flushes at every observation, the
        # most it ever can.  Two instruments observing simultaneously.
        t = np.sort(np.concatenate([np.arange((k + 1) // 2) * 0.3] * 2))[:k]
    else:
        t = np.arange(k) * 0.3
    dt = np.concatenate([[0.0], np.diff(t)])
    uniq, inv = np.unique(dt.astype(np.float32), return_inverse=True)
    lag_np = inv.astype(np.int32) + 1
    lag_np[0] = 0
    lag = mx.array(lag_np)

    m = term.n_state
    P = mx.array(np.tile(np.array(p, np.float32), (n_chains, 1)))
    phi_u, q_u, h = term.matrices(P, mx.array(uniq.astype(np.float32)))
    pt = mx.concatenate([mx.zeros((n_chains, 1, m, m), dtype=mx.float32),
                         phi_u], axis=1)
    qt = mx.concatenate([mx.broadcast_to(mx.eye(m, dtype=mx.float32),
                                         (n_chains, 1, m, m)), q_u], axis=1)
    y = mx.array(rng.standard_normal(k).astype(np.float32))
    rrv = mx.full((k,), 0.09, dtype=mx.float32)
    yv = mx.broadcast_to(y, (n_chains, k))
    rv = mx.broadcast_to(rrv, (n_chains, k))
    cot = mx.array(rng.standard_normal(n_chains).astype(np.float32))
    resid = mx.swapaxes(yv, 0, 1)
    j2 = mx.zeros((n_chains, 0), dtype=mx.float32)

    phi, q = mx.take(pt, lag, axis=1), mx.take(qt, lag, axis=1)
    states, _ = filter_forward(phi, q, h, yv, rv, rrv)
    want = filter_backward(phi, q, h, yv, rv, rrv, states, cot)
    with mx.stream(mx.gpu):
        el = scan_blocks(pt, qt, lag, h, resid, rrv, j2, 1e-6, blk)
        mx.eval(el)
    return dict(pt=pt, qt=qt, lag=lag, lag_np=lag_np, h=h, resid=resid,
                rrv=rrv, j2=j2, cot=cot, el=el, want=want, blk=blk,
                n_lags=pt.shape[1])


def _per_lag(x_obs, lag_np, n_lags):
    x = np.array(x_obs, dtype=np.float64)
    out = np.zeros((x.shape[0], n_lags) + x.shape[2:])
    for i in range(x.shape[1]):
        out[:, lag_np[i]] += x[:, i]
    return out


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-20))


def _e2e_errors(c, seeds):
    from anvilgp.metal import backward_ckpt
    with mx.stream(mx.gpu):
        o = backward_ckpt(c["pt"], c["qt"], c["lag"], c["h"], c["resid"],
                          c["rrv"], c["j2"], c["cot"], *seeds, c["blk"])
        mx.eval(o)
    w = c["want"]
    return [_rel(_per_lag(w[0], c["lag_np"], c["n_lags"]), mx.sum(o[0], axis=1)),
            _rel(_per_lag(w[1], c["lag_np"], c["n_lags"]), mx.sum(o[1], axis=1)),
            _rel(w[2], mx.sum(o[2], axis=1)),
            _rel(w[3], mx.swapaxes(o[3], 0, 1)),
            _rel(np.sum(np.array(w[4], np.float64), axis=1), mx.sum(o[4], axis=1))]


@pytest.mark.parametrize("name,term,p", _BWD_TERMS,
                         ids=[c[0] for c in _BWD_TERMS])
@pytest.mark.parametrize("cadence", ["regular", "gapped", "dupes",
                                    "distinct", "alternating"])
@pytest.mark.parametrize("k,blk", [(64, 32), (300, 100), (1000, 256)])
def test_checkpointed_backward_matches_the_specification(
        name, term, p, cadence, k, blk):
    """Stage F: long blocks, checkpointed, per-lag adjoints -- vs adjoint.py.

    Blocks of 100 and 256 are past stage E's cap of 32, so this is the
    checkpoint scheme under test. The seeds here come from the MLX scans so
    that the sweep is isolated from the seed kernel (tested separately).

    Tolerance: float32 against a float64 spec. The gapped cadence at m = 4
    is the ill-conditioned corner -- a 20-unit gap underflows the
    propagator and the seed's cancellation amplifies the block elements'
    float32 error -- and it is held to a looser bound and named as such.
    """
    from anvilgp.adjoint import block_seeds
    from anvilgp.elements import Element, prefix_scan, suffix_scan

    c = _fwd_setup(term, p, k, blk, cadence=cadence)
    el, n = c["el"], c["cot"].shape[0]
    m = term.n_state
    pre, suf = prefix_scan(el), suffix_scan(el)
    nb = el.b.shape[1]
    ent_m = mx.concatenate([mx.zeros((n, 1, m), dtype=mx.float32),
                            pre.b[:, :-1]], axis=1)
    ent_C = mx.concatenate([mx.zeros((n, 1, m, m), dtype=mx.float32),
                            pre.C[:, :-1]], axis=1)
    sm, sc = block_seeds(pre.b, pre.C,
                         Element(*(getattr(suf, f)[:, 1:]
                                   for f in Element._fields)),
                         mx.broadcast_to(c["cot"][:, None], (n, nb)))
    errs = _e2e_errors(c, (ent_m, ent_C, sm, sc))
    tol = 2e-3 if (cadence == "gapped" and m >= 4) else 1e-4
    for nm, e in zip(["phi", "q", "h", "y", "r_var"], errs):
        assert e < tol, f"{name}/{cadence} k={k} blk={blk} {nm}: {e:.2e}"


@pytest.mark.parametrize("name,term,p", _BWD_TERMS,
                         ids=[c[0] for c in _BWD_TERMS])
@pytest.mark.parametrize("cadence", ["regular", "gapped", "dupes",
                                    "distinct", "alternating"])
def test_seed_kernel_matches_the_mlx_scans(name, term, p, cadence):
    """The in-kernel prefix, suffix and seeds against the MLX versions.

    Both are float32 on the same block elements, so this is arithmetic
    order only. The MLX scans it replaces cost 33 ms at m = 2 and 1.4 s at
    m = 4 for N = 1e5; the kernel is under 8 ms at either.
    """
    from anvilgp.adjoint import block_seeds
    from anvilgp.elements import Element, prefix_scan, suffix_scan
    from anvilgp.metal import block_seeds_kernel

    c = _fwd_setup(term, p, 300, 64, cadence=cadence)
    el, n, m = c["el"], c["cot"].shape[0], term.n_state
    with mx.stream(mx.gpu):
        k_ent_m, k_ent_C, k_sm, k_sC = block_seeds_kernel(el, c["cot"])
        mx.eval(k_ent_m, k_ent_C, k_sm, k_sC)
    pre, suf = prefix_scan(el), suffix_scan(el)
    nb = el.b.shape[1]
    r_ent_m = mx.concatenate([mx.zeros((n, 1, m), dtype=mx.float32),
                              pre.b[:, :-1]], axis=1)
    r_ent_C = mx.concatenate([mx.zeros((n, 1, m, m), dtype=mx.float32),
                              pre.C[:, :-1]], axis=1)
    r_sm, r_sC = block_seeds(pre.b, pre.C,
                             Element(*(getattr(suf, f)[:, 1:]
                                       for f in Element._fields)),
                             mx.broadcast_to(c["cot"][:, None], (n, nb)))
    # m = 4 is the ill-conditioned dimension: 1e-4 in general, 1e-3 across
    # a gap, 1e-5 for everything else.  Measured, not chosen.
    tol = 1e-5 if m < 4 else (1e-3 if cadence == "gapped" else 1e-4)
    for nm, a, b in (("ent_m", r_ent_m, k_ent_m), ("ent_C", r_ent_C, k_ent_C),
                     ("seed_m", r_sm, k_sm), ("seed_C", r_sC, k_sC)):
        assert _rel(a, b) < tol, f"{name}/{cadence} {nm}: {_rel(a, b):.2e}"


@pytest.mark.parametrize("name,term,p", _BWD_TERMS,
                         ids=[c[0] for c in _BWD_TERMS])
def test_full_kernel_gradient_path_matches_the_specification(name, term, p):
    """Forward kernel -> seed kernel -> backward kernel, no MLX in between.

    This is the composition stage G wires into the VJP.
    """
    from anvilgp.metal import block_seeds_kernel

    c = _fwd_setup(term, p, 1000, 128, cadence="dupes")
    with mx.stream(mx.gpu):
        seeds = block_seeds_kernel(c["el"], c["cot"])
        mx.eval(*seeds)
    errs = _e2e_errors(c, seeds)
    for nm, e in zip(["phi", "q", "h", "y", "r_var"], errs):
        assert e < 2e-4, f"{name} {nm}: {e:.2e}"


@pytest.mark.parametrize("name,term,p", _BWD_TERMS,
                         ids=[c[0] for c in _BWD_TERMS])
def test_the_kernel_is_differentiable(name, term, p):
    """M4: the kernel carries its own VJP, and it agrees with the MLX path.

    Both are float32 on the same problem, so this is the arithmetic
    comparison a user actually cares about: swapping `metal` on or off must
    not move the gradient.
    """
    n = 1500
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 40.0, n))
    y = rng.standard_normal(n) * 0.9
    yerr = np.full(n, 0.3)
    pm = ParamMap.log(p)
    u = mx.array(rng.standard_normal((4, len(p))).astype(np.float32) * 0.1)
    ones = mx.ones((4,), dtype=mx.float32)

    got = {}
    for metal in (False, True):
        gp = GPLogLike(None, t, None, y, yerr, term, pm,
                       GPPolicy(block=1024, metal=metal))
        with mx.stream(mx.gpu):
            v, g = mx.vjp(gp, [u], [ones])
            mx.eval(v, g)
        got[metal] = np.array(g[0])

    a, b = got[False], got[True]
    assert np.all(np.isfinite(b))
    rel = np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-20)
    assert rel < 1e-4, f"{name}: metal vs MLX gradient {rel:.2e}"


def test_kernel_gradient_matches_float64_central_differences():
    """The independent check: no shared code with the MLX path at all."""
    n = 600
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 30.0, n))
    y = rng.standard_normal(n) * 0.9
    yerr = np.full(n, 0.3)
    refs = [1.0, 4.0, 3.0]
    kern = [KernelParam(link="log", index=i, ref=r)
            for i, r in enumerate(refs)]
    pm = ParamMap(kernel=kern, n_dim=4,
                  jitter=[KernelParam(link="log", index=3, ref=0.1)])
    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                   GPPolicy(block=256, metal=True, hi_block=256))
    v0 = rng.standard_normal(4) * 0.15
    with mx.stream(mx.gpu):
        _, g = mx.vjp(gp, [mx.array(v0[None].astype(np.float32))],
                      [mx.ones((1,), dtype=mx.float32)])
        mx.eval(g)
    got = np.array(g[0])[0]

    fd = np.zeros(4)
    for i in range(4):
        h = 1e-5 * max(1.0, abs(v0[i]))
        up, dn = v0.copy(), v0.copy()
        up[i] += h
        dn[i] -= h
        with mx.stream(mx.cpu):
            fd[i] = (float(np.array(gp.hi(a64(up[None])))[0])
                     - float(np.array(gp.hi(a64(dn[None])))[0])) / (2 * h)
    rel = np.max(np.abs(got - fd)) / max(np.max(np.abs(fd)), 1e-20)
    assert rel < 1e-5, f"{rel:.2e}\n kernel {got}\n fd     {fd}"


def test_the_kernel_clamps_its_variance_term():
    """Stage 0 in the kernel: fails if `sq = max(sq, 0)` is removed.

    Same fixture as the adjoint test: the leaf variance term is driven to
    -2e-3 against an r_var of 1e-3, so the unclamped innovation variance is
    negative and the log of it NaN.  The lag table has one row, `bad_q`,
    and every observation but the first points at it.
    """
    from anvilgp.metal import scan_loglike

    n_chains, k, m = 2, 12, 2
    bad_q = mx.array(np.tile(-1e-3 * np.eye(m, dtype=np.float32),
                             (n_chains, 1, 1, 1)))
    pt = mx.concatenate([mx.zeros((n_chains, 1, m, m), dtype=mx.float32),
                         mx.zeros((n_chains, 1, m, m), dtype=mx.float32)],
                        axis=1)
    qt = mx.concatenate([mx.broadcast_to(mx.eye(m, dtype=mx.float32),
                                         (n_chains, 1, m, m)), bad_q], axis=1)
    lag = mx.array(np.array([0] + [1] * (k - 1), dtype=np.int32))
    h = mx.ones((n_chains, m), dtype=mx.float32)
    resid = mx.array(np.random.default_rng(0).standard_normal((k, n_chains))
                     .astype(np.float32))
    r_ref = mx.full((k,), 1e-3, dtype=mx.float32)
    j2 = mx.zeros((n_chains, 0), dtype=mx.float32)
    with mx.stream(mx.gpu):
        out = scan_loglike(pt, qt, lag, h, resid, r_ref, j2, 0.0, 8)
        mx.eval(out)
    assert bool(mx.all(mx.isfinite(out))), np.array(out)


def test_a_healthy_value_never_comes_with_a_dead_gradient():
    """Stage R: the converse of the guard's contract, on the kernel path.

    The guard zeroes a lane's gradient when the backward returns NaN.  That
    is right when the value was rejected and silently wrong when it was
    not.  At this point -- sigma/yerr = 1e4, rho/dt = 1000, Q = 500, all
    inside the shipped priors' bounds except the unbounded log sigma -- the
    kernel value is right to 2e-4 and its gradient came back exactly zero,
    because the first observation's covariance update ``I - h h^T/(h.h + r)``
    rounds to a negative ulp along ``h`` when ``h.h/r ~ 1e8`` and an
    undamped propagator grew that ulp until the replay overflowed.  The
    Joseph-form update fixed it.  Fails with the plain update restored.
    """
    from anvilgp.likelihood import GPLogLike, ParamMap, a64
    from anvilgp.oracle import sample_ssm

    dt, n, yerr = 0.02, 2048, 1e-3
    t = np.arange(n) * dt
    rng = np.random.default_rng(41)
    y = (sample_ssm(ag.SHOTerm(), [30 * yerr, 100 * dt, 3.0], dt, n, rng)
         + rng.standard_normal(n) * yerr)
    pm = ParamMap.log([1e4 * yerr, 1000 * dt, 500.0])
    gp = GPLogLike(None, t, None, y, np.full(n, yerr), ag.SHOTerm(), pm,
                   GPPolicy(block=1024, hi_block=1024, metal=True))
    assert gp.metal_eligible()[0]
    u = mx.zeros((1, 3), dtype=mx.float32)
    with mx.stream(mx.gpu):
        val = gp(u)
        _, grad = mx.vjp(gp, [u], [mx.ones((1,), dtype=mx.float32)])
        mx.eval(val, grad)
    v0 = np.zeros((1, 3))
    with mx.stream(mx.cpu):
        f64 = float(np.array(gp.hi(a64(v0)))[0])
        fd = np.zeros(3)
        for i in range(3):
            up, dn = v0.copy(), v0.copy()
            up[0, i] += 1e-4
            dn[0, i] -= 1e-4
            fd[i] = (float(np.array(gp.hi(a64(up)))[0])
                     - float(np.array(gp.hi(a64(dn)))[0])) / 2e-4
    v = float(np.array(val)[0])
    assert abs(v - f64) / abs(f64) < 1e-3, "the value is healthy here"
    g = np.array(grad[0])[0].astype(np.float64)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0), f"dead gradient {g}"
    rel = np.max(np.abs(g - fd) / np.maximum(np.abs(fd), 1e-9))
    assert rel < 1e-2, f"kernel {g} vs float64 {fd}: rel {rel:.2e}"


def test_the_kernel_gradient_refuses_an_unaffordable_lag_slab(monkeypatch):
    """Stage R: the per-lag adjoint buffer is N-sized on an irregular
    cadence and used to allocate silently -- 2.2 GiB at N = 2e4 and 64
    chains, linear in chains.  The VJP now refuses over a budget, with the
    arithmetic and the fix in the message, and the forward is unaffected.
    """
    import anvilgp.metal as metal_mod

    from anvilgp.likelihood import GPLogLike, ParamMap

    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 50.0, 400))          # every lag distinct
    gp = GPLogLike(None, t, None, rng.standard_normal(400), np.full(400, 0.3),
                   ag.SHOTerm(), ParamMap.log([1.3, 4.0, 3.0]),
                   GPPolicy(block=1024, hi_block=1024, metal=True,
                            metal_block=64))
    assert gp.metal_eligible()[0] and gp.n_distinct_lags > 300
    u = mx.zeros((4, 3), dtype=mx.float32)
    with mx.stream(mx.gpu):
        mx.eval(gp(u))                                # the forward is fine
    monkeypatch.setattr(metal_mod, "MAX_LAG_SLAB_BYTES", 1 << 16)
    with pytest.raises(ValueError, match="distinct lags.*metal=False"):
        with mx.stream(mx.gpu):
            _, g = mx.vjp(gp, [u], [mx.ones((4,), dtype=mx.float32)])
            mx.eval(g)


_CHUNK_TERMS = [("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
                ("matern52", ag.Matern52Term(), [1.1, 2.5]),
                ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4])]


def _distinct_gp(term, pars, n=3000, n_chains=8):
    """A cadence whose every spacing is its own lag, so the table is N rows."""
    from anvilgp.likelihood import GPLogLike, ParamMap
    rng = np.random.default_rng(7)
    t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.2, 0.4, n - 1))])
    gp = GPLogLike(None, t, None, rng.standard_normal(n), np.full(n, 0.3), term,
                   ParamMap.log(pars),
                   GPPolicy(block=1024, hi_block=1024, metal=True))
    return gp, mx.zeros((n_chains, len(pars)), dtype=mx.float32)


def _grad_at_budget(gp, u, budget):
    import anvilgp.likelihood as L
    old = L._FRONT_END_BUDGET_BYTES
    L._FRONT_END_BUDGET_BYTES = budget
    try:
        with mx.stream(mx.gpu):
            _, g = mx.vjp(gp, [u], [mx.ones((u.shape[0],), dtype=mx.float32)])
            mx.eval(g)
        return np.array(g[0], np.float64)
    finally:
        L._FRONT_END_BUDGET_BYTES = old


@pytest.mark.parametrize("name,term,pars", _CHUNK_TERMS,
                         ids=[c[0] for c in _CHUNK_TERMS])
def test_chunking_the_front_end_does_not_change_the_gradient(name, term, pars):
    """Stage A: splitting the term front end's backward over the lag axis.

    The budget is lowered so the split actually happens at test size -- at
    the shipped budget this problem is one chunk, and a test that never
    enters the branch would pass with the branch deleted.

    This is also what pins **`h`'s cotangent being charged to one chunk**.
    `h` does not depend on the lag, so charging it to every chunk
    multiplies that part of the gradient by the chunk count: 9e-2 relative
    when I first wrote it, against the 1e-7 below.
    """
    import anvilgp.likelihood as L

    gp, u = _distinct_gp(term, pars)
    n_lags = gp._metal_uniq_dt.shape[0]
    old = L._FRONT_END_BUDGET_BYTES
    L._FRONT_END_BUDGET_BYTES = 1 << 22
    try:
        n_chunk = L._front_end_chunks(u.shape[0], n_lags, term.n_state)
    finally:
        L._FRONT_END_BUDGET_BYTES = old
    assert n_chunk > 1, "the fixture must exercise the chunked branch"

    chunked = _grad_at_budget(gp, u, 1 << 22)
    whole = _grad_at_budget(gp, u, 1 << 60)
    rel = np.max(np.abs(chunked - whole)) / max(np.max(np.abs(whole)), 1e-30)
    assert rel < 1e-6, f"{name}: {n_chunk} chunks change the gradient by {rel:.2e}"


def test_a_regular_cadence_never_chunks_the_front_end():
    """The condition, not just the arithmetic.

    Chunking a two-row lag table costs ~1.8x on the front end's backward to
    save memory that is already 0.009 GiB at 1024 chains x N = 1e5.  Every
    cadence with a handful of distinct spacings -- which is all space
    photometry, however large N is -- must take the unchunked path.
    """
    from anvilgp.likelihood import _front_end_chunks

    for n_chains in (64, 256, 1024, 4096):
        for n_lags in (2, 3, 4, 20):
            for m in (1, 2, 3, 4):
                assert _front_end_chunks(n_chains, n_lags, m) == 1, (
                    f"chunked a {n_lags}-row table at {n_chains} chains, m={m}")
    # and it does engage where the table is N-sized
    assert _front_end_chunks(256, 100_001, 2) > 1
    assert _front_end_chunks(1024, 100_001, 2) > 1


@pytest.mark.slow
def test_chunking_the_front_end_lowers_the_gradients_peak_memory():
    """Stage A's point, and a guard against the eval that undid it.

    Forcing each chunk to completion with `mx.eval` inside the loop looks
    like the way to free it and does the opposite -- it materialises the
    upstream graph, the backward kernel and its per-lag slab, and pins it
    for the whole loop.  Measured 3.6x *worse* than not chunking (7.76 GiB
    against 2.17) where leaving it lazy gives 0.97.  This test fails if
    that eval comes back.
    """
    gp, u = _distinct_gp(ag.SHOTerm(), [1.3, 4.0, 3.0], n=20_000, n_chains=64)

    def peak(budget):
        mx.clear_cache()
        mx.reset_peak_memory()
        _grad_at_budget(gp, u, budget)
        return mx.get_peak_memory() / 2**30

    whole = peak(1 << 60)
    chunked = peak(1 << 22)
    assert chunked < 0.75 * whole, (
        f"chunked peak {chunked:.3f} GiB vs unchunked {whole:.3f} GiB")


def _lag_of(t):
    dt = np.concatenate([[0.0], np.diff(t)]).astype(np.float32)
    uniq, inv = np.unique(dt, return_inverse=True)
    lag = inv.astype(np.int32) + 1
    lag[0] = 0
    return lag


_SLOT_CADENCES = [
    # name, builder, expected slots at N = 1e5, blk = 8192 (13 blocks)
    ("regular + 1 gap", lambda n: np.concatenate(
        [np.arange(n // 2) * 0.02, np.arange(n // 2) * 0.02 + 20.0]), 15),
    ("regular, 20 gaps", lambda n: np.arange(n) * 0.02
     + np.repeat(np.arange(20) * 5.0, n // 20), 26),
    ("alternating dupes", lambda n: np.sort(
        np.concatenate([np.arange(n // 2) * 0.02] * 2)), 27),
    ("two-cadence mix", lambda n: np.sort(np.concatenate(
        [np.arange(n // 2) * 0.02, np.arange(n // 2) * 0.05])), 53),
    ("all distinct", lambda n: np.sort(
        np.random.default_rng(0).uniform(0.0, 2000.0, n)), 99_992),
]


@pytest.mark.parametrize("name,build,want", _SLOT_CADENCES,
                         ids=[c[0] for c in _SLOT_CADENCES])
def test_the_slot_map_is_one_slot_per_block_lag_pair_that_occurs(name, build,
                                                                 want):
    """Stage B: the index map the backward's adjoint buffer will use.

    The buffer had one slot per ``(chain, block, lag)``, nearly all of it
    never written: a block can touch at most ``blk_len`` lags and usually
    touches one or two.  One slot per pair that *occurs* is 2600x smaller
    on a regular cadence with a gap and 13x smaller when every timestamp
    spacing is distinct, where the old shape is 40 GiB at 1024 chains.

    The counts are pinned because they are the whole argument for the
    change; a map that quietly produced more slots than pairs would still
    satisfy every invariant below.
    """
    from anvilgp.likelihood import _slot_map

    n, blk = 100_000, 8192
    lag = _lag_of(build(n))
    slot_id, slot_start, slot_lag = _slot_map(lag, blk)
    n_blocks = -(-n // blk)
    n_slots = int(slot_lag.size)

    assert n_slots == want, f"{name}: {n_slots} slots, expected {want}"
    assert n_slots <= min(n_blocks * (int(lag.max()) + 1), n)

    # every observation lands in exactly one slot, and that slot's lag is
    # the observation's lag
    assert slot_id.shape == lag.shape
    assert int(slot_id.min()) == 0 and int(slot_id.max()) == n_slots - 1
    assert np.array_equal(slot_lag[slot_id], lag), f"{name}: slot lag mismatch"

    # each block owns a contiguous slot range and writes only inside it --
    # the kernel thread zeroes exactly this range
    assert slot_start[0] == 0 and slot_start[-1] == n_slots
    assert np.all(np.diff(slot_start) >= 0)
    blk_of = np.arange(n) // blk
    for b in range(n_blocks):
        used = np.unique(slot_id[blk_of == b])
        assert used.min() >= slot_start[b] and used.max() < slot_start[b + 1], (
            f"{name}: block {b} writes outside its range")
    # and no slot is shared between blocks
    owner = np.zeros(n_slots, dtype=np.int64) - 1
    owner[slot_id] = blk_of
    for b in range(n_blocks):
        assert np.all(owner[slot_start[b]:slot_start[b + 1]] == b), (
            f"{name}: a slot in block {b}'s range belongs to another block")


def test_the_slot_map_follows_the_resolved_block_length():
    """A map built for a different block length misattributes every
    adjoint -- and the *value* stays correct, because the forward never
    reads it.  So it is built from `self.metal_block`, after that is
    resolved, not from the policy field that may be None.
    """
    from anvilgp.likelihood import GPLogLike, ParamMap, _slot_map

    rng = np.random.default_rng(4)
    n = 5000
    t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.2, 0.4, n - 1))])
    for blk in (None, 256, 1024):
        gp = GPLogLike(None, t, None, rng.standard_normal(n), np.full(n, 0.3),
                       ag.SHOTerm(), ParamMap.log([1.3, 4.0, 3.0]),
                       GPPolicy(block=1024, hi_block=1024, metal=True,
                                metal_block=blk))
        want = _slot_map(np.array(gp._metal_lag), gp.metal_block)
        for got, exp in zip((gp._metal_slot_id, gp._metal_slot_start,
                             gp._metal_slot_lag), want):
            assert np.array_equal(np.array(got), exp), f"blk={blk}"
        assert gp.n_metal_slots == int(want[2].size)
        # and the map covers every observation of every block
        assert np.array(gp._metal_slot_id).size == n
