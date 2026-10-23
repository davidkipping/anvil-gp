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
        el = scan_blocks(pt, qt, gp._metal_lag, gp._metal_inst, h, r,
                         gp._r_ref32, j2,
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
        el = scan_blocks(pt, qt, lag, mx.zeros((k,), dtype=mx.int32), h,
                         resid, rrv, j2, 1e-6, blk)
        mx.eval(el)
    from anvilgp.likelihood import _slot_map
    sid, sstart, slag = _slot_map(lag_np, blk)
    inst = mx.zeros((k,), dtype=mx.int32)
    return dict(pt=pt, qt=qt, lag=lag, lag_np=lag_np, h=h, resid=resid,
                rrv=rrv, j2=j2, cot=cot, el=el, want=want, blk=blk,
                n_lags=pt.shape[1], inst=inst, slot_id=mx.array(sid),
                slot_start=mx.array(sstart), slot_lag=slag)


def _per_lag(x_obs, lag_np, n_lags):
    x = np.array(x_obs, dtype=np.float64)
    out = np.zeros((x.shape[0], n_lags) + x.shape[2:])
    for i in range(x.shape[1]):
        out[:, lag_np[i]] += x[:, i]
    return out


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-20))


def _fold_slots(x_slots, slot_lag, n_lags):
    """(n_chains, n_slots, m, m) -> (n_chains, n_lags, m, m).

    The blocks are already summed: distinct blocks own disjoint slots, so
    this only regroups.  Repeated `slot_lag` entries are the several blocks
    that used one lag.
    """
    x = np.array(x_slots, dtype=np.float64)
    out = np.zeros((x.shape[0], n_lags) + x.shape[2:])
    np.add.at(out, (slice(None), slot_lag), x)
    return out


def _e2e_errors(c, seeds):
    from anvilgp.metal import backward_ckpt
    with mx.stream(mx.gpu):
        o = backward_ckpt(c["pt"], c["qt"], c["lag"], c["inst"], c["h"],
                          c["resid"],
                          c["rrv"], c["j2"], c["cot"], *seeds,
                          c["slot_id"], c["slot_start"], c["slot_lag"].size,
                          c["blk"])
        mx.eval(o)
    w, sl, nl = c["want"], c["slot_lag"], c["n_lags"]
    return [_rel(_per_lag(w[0], c["lag_np"], nl), _fold_slots(o[0], sl, nl)),
            _rel(_per_lag(w[1], c["lag_np"], nl), _fold_slots(o[1], sl, nl)),
            _rel(w[2], mx.sum(o[2], axis=1)),
            _rel(w[3], mx.swapaxes(o[3], 0, 1)),
            # `drv` is (n_chains, n_blocks, n_jit) since per-instrument
            # jitter landed; with one jitter (or none) the total r_var
            # adjoint is the sum over both trailing axes.
            _rel(np.sum(np.array(w[4], np.float64), axis=1),
                 mx.sum(o[4], axis=(1, 2)))]


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
@pytest.mark.parametrize("cadence", ["gapped", "random"])
def test_the_kernel_is_differentiable(name, term, p, cadence):
    """M4: the kernel carries its own VJP, and it agrees with the MLX path.

    Both are float32 on the same problem, so this is the arithmetic
    comparison a user actually cares about: swapping `metal` on or off must
    not move the gradient.
    """
    n = 1500
    rng = np.random.default_rng(3)
    # Both cadences, deliberately.  On a random one every spacing is its
    # own lag and any one lag carries ~1/1500 of the gradient, so a
    # per-lag mis-routing worth 6e-2 on the gapped cadence reads as 3e-4
    # here and passed this test as first written (stage R2).  The gapped
    # cadence is also every README number and every space light curve.
    if cadence == "gapped":
        t = np.arange(n) * (40.0 / n)
        t[n // 2:] += 20.0
    else:
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


@pytest.mark.parametrize("cadence", ["gapped", "random"])
def test_kernel_gradient_matches_float64_central_differences(cadence):
    """The independent check: no shared code with the MLX path at all."""
    n = 600
    rng = np.random.default_rng(4)
    if cadence == "gapped":            # see test_the_kernel_is_differentiable
        t = np.arange(n) * (30.0 / n)
        t[n // 2:] += 20.0
    else:
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
        out = scan_loglike(pt, qt, lag, mx.zeros((k,), dtype=mx.int32), h,
                           resid, r_ref, j2, 0.0, 8,
                           slots=_slots_for(k, 8, lag))
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

    Run with ``kappa_max=None``: this point's kappa is far past the default
    guard, which now refuses it before the kernel runs (and "right to
    2e-4" relative is ~0.4 log-units here, as the kappa map predicts).  The
    test pins the kernel's own behaviour, which the guard must not hide.
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
                   GPPolicy(block=1024, hi_block=1024, metal=True,
                             kappa_max=None))
    assert gp.metal_eligible()[0]
    u = mx.zeros((1, 3), dtype=mx.float32)
    assert float(np.array(gp.kappa(u))[0]) > 1e6
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


def test_the_kernel_gradient_refuses_an_unaffordable_adjoint_buffer(
        monkeypatch):
    """The fence is still there, and still names the cadence.

    It is a sanity check now rather than the wall it was: the buffer is one
    slot per (block, lag) pair that occurs, which is bounded by N however
    irregular the timestamps, where a slot per pair that *could* occur was
    N x blocks.  See the test below for what that changes.
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
    monkeypatch.setattr(metal_mod, "MAX_ADJOINT_BUFFER_BYTES", 1 << 12)
    with pytest.raises(ValueError, match="slots.*metal=False"):
        with mx.stream(mx.gpu):
            _, g = mx.vjp(gp, [u], [mx.ones((4,), dtype=mx.float32)])
            mx.eval(g)


@pytest.mark.slow
def test_the_slot_buffer_admits_a_cadence_the_per_lag_buffer_refused():
    """Stage C's point: a problem that used to be refused now runs.

    512 chains over 20,000 points with every timestamp spacing distinct.
    One slot per (block, lag) *possible* is 3.05 GiB there, past the 2 GiB
    fence this shipped with before stage C, so the gradient raised; one
    slot per pair that occurs is the block count smaller and it runs.

    Deliberately not the production shape.  At 1024 chains and N = 1e5 the
    same comparison is 39.7 GiB against 3.05, inside a 17.5 GiB gradient
    whose forward alone is 13.2 -- measured, and recorded in
    docs/lag-slots-plan.md.  Asserting it here needs ~20 GiB free and made
    this test fail once and pass on retry, which is worse than not having
    it: a flaky test teaches people to re-run rather than to look.
    """
    from anvilgp.likelihood import GPLogLike, ParamMap

    #: the fence this shipped with before stage C, when the buffer had a
    #: slot per (block, lag) *possible*
    was_refused_above = 2 << 30

    n, n_chains, m = 20_000, 512, 2
    rng = np.random.default_rng(11)
    t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.2, 0.4, n - 1))])
    gp = GPLogLike(None, t, None, rng.standard_normal(n), np.full(n, 0.3),
                   ag.SHOTerm(), ParamMap.log([1.3, 4.0, 3.0]),
                   GPPolicy(block=1024, hi_block=1024, metal=True))
    n_blocks = -(-n // gp.metal_block)
    n_lags = gp.n_distinct_lags + 1

    was = 2 * n_chains * n_blocks * n_lags * m * m * 4
    now = 2 * n_chains * gp.n_metal_slots * m * m * 4
    assert was > was_refused_above, (
        f"fixture no longer reproduces the old refusal: {was / 2**30:.2f} GiB")
    assert now < was_refused_above
    assert gp.n_metal_slots < n_blocks * n_lags / 5

    u = mx.zeros((n_chains, 3), dtype=mx.float32)
    with mx.stream(mx.gpu):
        _, g = mx.vjp(gp, [u], [mx.ones((n_chains,), dtype=mx.float32)])
        mx.eval(g)
    g = np.array(g[0])
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)
    # every chain is the same point, so every row is the same gradient
    assert np.max(np.abs(g - g[0])) / max(np.max(np.abs(g[0])), 1e-30) < 1e-5


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


def _slots_for(npts, blk, lag):
    """The slot map + fold plan `scan_loglike` needs, for a direct call."""
    from anvilgp.likelihood import _slot_map, _lag_fold_plan
    lag_np = np.array(lag)
    sid, sstart, slag = _slot_map(lag_np, blk)
    kind, plan = _lag_fold_plan(slag, int(lag_np.max()) + 1)
    fold = ((kind, mx.array(plan)) if kind == "dense"
            else (kind, [(mx.array(a), mx.array(b)) for a, b in plan]))
    return (mx.array(sid), mx.array(sstart), int(slag.size), fold)


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


@pytest.mark.parametrize("cadence", ["regular", "distinct"])
def test_the_per_lag_fold_is_reproducible(cadence):
    """Two identical gradient calls must agree bit for bit.

    Folding slots back into a per-lag table is a scatter, and MLX's
    scatter-add is **not** reproducible when indices repeat -- the
    accumulation order across threads is not fixed.  A lag repeats across
    blocks on every regular cadence, so the naive fold would have cost the
    gradient its reproducibility on the common case.  The fold goes
    through a padded `(n_lags, max_rank)` table and one `mx.sum` instead,
    which is reproducible.  This fails if the naive scatter comes back.
    """
    from anvilgp.likelihood import GPLogLike, ParamMap

    rng = np.random.default_rng(7)
    n = 6000
    t = (np.arange(n) * 0.02 if cadence == "regular"
         else np.concatenate([[0.0], np.cumsum(rng.uniform(0.015, 0.025,
                                                           n - 1))]))
    gp = GPLogLike(None, t, None, rng.standard_normal(n), np.full(n, 0.3),
                   ag.SHOTerm(), ParamMap.log([1.3, 4.0, 3.0]),
                   GPPolicy(block=1024, hi_block=1024, metal=True,
                            metal_block=512))
    kind, plan = gp._metal_fold_plan
    if cadence == "regular":            # the case with repeated lags
        assert kind == "dense" and plan.shape[1] > 1, (
            "fixture must have a lag shared across blocks")
    u = mx.zeros((32, 3), dtype=mx.float32)
    one = mx.ones((32,), dtype=mx.float32)
    outs = []
    for _ in range(4):
        with mx.stream(mx.gpu):
            _, g = mx.vjp(gp, [u], [one])
            mx.eval(g)
        outs.append(np.array(g[0]))
    for i, o in enumerate(outs[1:], 1):
        assert np.array_equal(outs[0], o), (
            f"{cadence}: call {i} differs by "
            f"{np.max(np.abs(o - outs[0])):.3e}")


@pytest.mark.parametrize("name,build,_want", _SLOT_CADENCES,
                         ids=[c[0] for c in _SLOT_CADENCES])
def test_the_fold_plan_places_every_slot_exactly_once(name, build, _want):
    """The fold must be a permutation, not a resampling.

    A table that dropped a slot would lose that block's contribution to
    one lag's adjoint -- a wrong gradient with a correct value, which no
    value test can see.  One that repeated a slot would double it.
    """
    from anvilgp.likelihood import _slot_map, _lag_fold_plan

    n, blk = 100_000, 8192
    lag = _lag_of(build(n))
    slot_id, _, slot_lag = _slot_map(lag, blk)
    n_lags, n_slots = int(lag.max()) + 1, int(slot_lag.size)
    kind, plan = _lag_fold_plan(slot_lag, n_lags)

    if kind == "dense":
        assert plan.shape[0] == n_lags
        flat = plan.ravel()
        used = flat[flat != n_slots]
        # every slot exactly once, and each in its own lag's row
        assert np.array_equal(np.sort(used), np.arange(n_slots)), name
        rows = np.repeat(np.arange(n_lags), plan.shape[1])[flat != n_slots]
        assert np.array_equal(rows, slot_lag[used]), f"{name}: wrong row"
    else:
        used = np.concatenate([g[0] for g in plan])
        assert np.array_equal(np.sort(used), np.arange(n_slots)), name
        for sel, lags_of in plan:
            assert np.array_equal(slot_lag[sel], lags_of), name
            assert len(np.unique(lags_of)) == len(lags_of), (
                f"{name}: a group repeats a lag, so its scatter races")


def test_assigning_metal_block_rebuilds_the_slot_map():
    """Stage R2's finding.  The map is indexed by block, so it is only
    valid for the block length it was built with; assigning the attribute
    used to leave it stale, and the failure was silent or a hang.
    """
    from anvilgp.likelihood import GPLogLike, ParamMap

    rng = np.random.default_rng(3)
    n = 3000
    t = np.arange(n) * 0.3
    t[n // 2:] += 20.0
    y, ye = rng.standard_normal(n), np.full(n, 0.3)

    def build(blk):
        return GPLogLike(None, t, None, y, ye, ag.SHOTerm(),
                         ParamMap.log([1.3, 4.0, 3.0]),
                         GPPolicy(block=1024, hi_block=1024, metal=True,
                                  metal_block=blk))

    def grad(gp):
        u = mx.array([[0.1, -0.2, 0.05]], dtype=mx.float32)
        with mx.stream(mx.gpu):
            _, g = mx.vjp(gp, [u], [mx.ones((1,), dtype=mx.float32)])
            mx.eval(g)
        return np.array(g[0])[0].astype(np.float64)

    gp = build(256)
    for blk in (1024, 128, 4096):
        gp.metal_block = blk
        assert gp._metal_slot_start.shape[0] - 1 == -(-n // blk)
        want = grad(build(blk))
        got = grad(gp)
        rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert rel < 1e-6, f"reassigned to {blk}: {rel:.2e}"
    with pytest.raises(ValueError, match="metal_block"):
        gp.metal_block = 0
    with pytest.raises(ValueError, match="metal_block"):
        GPPolicy(metal_block=0)


def test_the_launch_refuses_a_map_it_did_not_build():
    """Defence in depth behind the property: bypassing the setter must be
    loud, because the alternative is a silently wrong gradient."""
    from anvilgp.likelihood import GPLogLike, ParamMap

    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(),
                   ParamMap.log([1.3, 4.0, 3.0]),
                   GPPolicy(metal=True, metal_block=256))
    gp._metal_block = 512                       # not the property
    with pytest.raises(RuntimeError, match="slot map"):
        with mx.stream(mx.gpu):
            mx.eval(gp(mx.zeros((2, 3), dtype=mx.float32)))


def test_the_fold_picks_groups_when_the_dense_table_would_dwarf_the_buffer():
    """Stage R2: choosing dense on `max_rank` alone sent a mostly-regular
    cadence with thousands of odd spacings to a gather 10x its buffer."""
    from anvilgp.likelihood import _slot_map, _lag_fold_plan

    n, blk = 20_000, 2048
    rng = np.random.default_rng(5)
    regular = np.arange(n) * 0.02
    odd = regular.copy()
    idx = rng.choice(n, 9000, replace=False)
    odd[idx] += rng.uniform(0.001, 0.009, idx.size)
    odd = np.sort(odd)
    for name, t, want in (("regular", regular, "dense"),
                          ("mostly regular", odd, "groups"),
                          ("distinct", np.sort(rng.uniform(0, 400.0, n)),
                           "groups")):
        lag = _lag_of(t)
        _, _, slag = _slot_map(lag, blk)
        kind, plan = _lag_fold_plan(slag, int(lag.max()) + 1)
        assert kind == want, f"{name}: {kind}"
        if kind == "dense":
            assert plan.size <= 4 * slag.size


def test_the_front_end_chunk_count_honours_its_budget():
    """Stage R2: a 64-lags-per-chunk floor returned 1 chunk with the
    estimate 3x over budget above ~1365 chains at m = 4."""
    from anvilgp.likelihood import (_front_end_chunks, _FRONT_END_RETENTION,
                                    _FRONT_END_BUDGET_BYTES)
    for n_chains, n_lags, m in ((4096, 64, 4), (4096, 100, 2), (4096, 200, 4)):
        n = _front_end_chunks(n_chains, n_lags, m)
        est = _FRONT_END_RETENTION * 2 * n_chains * n_lags * m * m * 4
        assert est / n <= 1.05 * _FRONT_END_BUDGET_BYTES, (
            f"{n_chains} x {n_lags} x m={m}: {n} chunks, "
            f"{est / n / 2**20:.0f} MB each")
        assert n <= n_lags


def test_a_single_observation_folds_on_either_branch(monkeypatch):
    """Stage R2: the fold's row count came from `lag.max() + 1`, which is 1
    at N = 1 where the table has 2 rows; the dense branch then raised.
    Unreachable as shipped (N = 1 has one rank), so force the branch."""
    import anvilgp.likelihood as L
    from anvilgp.likelihood import GPLogLike, ParamMap

    monkeypatch.setattr(L, "_MAX_FOLD_RANKS", 0)
    gp = GPLogLike(None, np.array([0.0]), None, np.array([0.3]),
                   np.array([0.2]), ag.SHOTerm(), ParamMap.log([1.3, 4.0, 3.0]),
                   GPPolicy(block=1024, hi_block=1024, metal=True))
    assert gp._metal_fold_plan[0] == "dense"
    u = mx.zeros((2, 3), dtype=mx.float32)
    with mx.stream(mx.gpu):
        v = gp(u)
        _, g = mx.vjp(gp, [u], [mx.ones((2,), dtype=mx.float32)])
        mx.eval(v, g)
    assert bool(mx.all(mx.isfinite(v))) and bool(mx.all(mx.isfinite(g[0])))


def test_the_kernel_refuses_wide_index_dtypes():
    """Metal reads the index arrays as 4-byte ints; an int64 array would
    index garbage with the value still right."""
    from anvilgp.metal import backward_ckpt

    c = _fwd_setup(ag.SHOTerm(), [1.3, 4.0, 3.0], 64, 32)
    seeds = (mx.zeros((3, 2, 2)), mx.zeros((3, 2, 2, 2)),
             mx.zeros((3, 2, 2)), mx.zeros((3, 2, 2, 2)))
    with pytest.raises(TypeError, match="int32"):
        backward_ckpt(c["pt"], c["qt"], c["lag"], c["inst"], c["h"],
                      c["resid"], c["rrv"],
                      c["j2"], c["cot"], *seeds,
                      c["slot_id"].astype(mx.int64), c["slot_start"],
                      c["slot_lag"].size, c["blk"])


def test_the_slot_map_is_deferred_until_the_kernel_is_asked_for():
    """Building it costs an np.unique over N and three device arrays, and
    only the Metal backward reads it -- so a float64-only or MLX-path
    instance should not pay.  Deferred, not skipped: flipping
    `policy.metal` afterwards must still give the same gradient.
    """
    from anvilgp.likelihood import GPLogLike, ParamMap

    rng = np.random.default_rng(1)
    n = 2000
    t = np.arange(n) * 0.02
    t[n // 2:] += 20.0
    y, ye = rng.standard_normal(n), np.full(n, 0.3)

    def build(metal):
        return GPLogLike(None, t, None, y, ye, ag.SHOTerm(),
                         ParamMap.log([1.3, 4.0, 3.0]), GPPolicy(metal=metal))

    lazy = build(False)
    assert lazy._metal_slot_id is None and lazy.n_metal_slots == 0
    lazy.policy.metal = True
    lazy._metal_ok, lazy._metal_why = lazy.metal_eligible()

    eager = build(True)
    assert eager._metal_slot_id is not None

    u = mx.zeros((4, 3), dtype=mx.float32)
    one = mx.ones((4,), dtype=mx.float32)

    def vg(gp):
        with mx.stream(mx.gpu):
            v = gp(u)
            _, g = mx.vjp(gp, [u], [one])
            mx.eval(v, g)
        return np.array(v, np.float64), np.array(g[0], np.float64)

    v1, g1 = vg(lazy)
    v2, g2 = vg(eager)
    assert lazy.n_metal_slots == eager.n_metal_slots
    assert np.array_equal(v1, v2), "deferred build changed the value"
    assert np.array_equal(g1, g2), "deferred build changed the gradient"


def test_the_fold_gather_is_charged_to_the_adjoint_budget(monkeypatch):
    """The dense fold gathers (n_chains, n_lags, max_rank, m, m) on top of
    the slot buffer, and the fold plan only bounds that at 4x the slots --
    so a configuration inside the fence could still OOM in the fold.  The
    fence counts it now.  Fails if only the buffer is charged.
    """
    import anvilgp.metal as metal_mod
    from anvilgp.likelihood import GPLogLike, ParamMap

    n, n_chains, m = 4000, 8, 2
    t = np.arange(n) * 0.02                      # regular -> dense fold
    rng = np.random.default_rng(2)
    gp = GPLogLike(None, t, None, rng.standard_normal(n), np.full(n, 0.3),
                   ag.SHOTerm(), ParamMap.log([1.3, 4.0, 3.0]),
                   GPPolicy(block=1024, hi_block=1024, metal=True,
                            metal_block=64))
    kind, plan = gp._metal_fold_plan
    assert kind == "dense", "fixture must exercise the dense branch"

    buf = 2 * n_chains * gp.n_metal_slots * m * m * 4
    gather = 2 * n_chains * int(plan.shape[0]) * int(plan.shape[1]) * m * m * 4
    assert gather > buf, "fixture must have a gather larger than the buffer"

    # a budget that admits the buffer but not buffer + gather
    monkeypatch.setattr(metal_mod, "MAX_ADJOINT_BUFFER_BYTES",
                        buf + gather // 2)
    u = mx.zeros((n_chains, 3), dtype=mx.float32)
    with pytest.raises(ValueError, match="gather"):
        with mx.stream(mx.gpu):
            _, g = mx.vjp(gp, [u], [mx.ones((n_chains,), dtype=mx.float32)])
            mx.eval(g)


def _jitter_gp(n_jit, metal, n=1200, jits=(0.03, 0.09, 0.2, 0.05),
               inst=None, seed=4):
    """A GP with `n_jit` per-instrument jitters, labels round-robin by default."""
    from anvilgp.likelihood import GPLogLike, KernelParam, ParamMap
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.05
    if inst is None:
        inst = None if n_jit == 1 else (np.arange(n) % n_jit).astype(np.int32)
    pm = ParamMap(
        kernel=[KernelParam(link="log", index=i, ref=r)
                for i, r in enumerate((1.3, 4.0, 3.0))],
        jitter=[KernelParam(link="log", index=3 + j, ref=jits[j % len(jits)])
                for j in range(n_jit)],
        n_dim=3 + n_jit, instrument=inst)
    return GPLogLike(None, t, None, rng.standard_normal(n), np.full(n, 0.3),
                     ag.SHOTerm(), pm,
                     GPPolicy(block=1024, hi_block=1024, metal=metal))


def _vg(gp, n_par, n_chains=4):
    u = mx.zeros((n_chains, n_par), dtype=mx.float32)
    with mx.stream(mx.gpu):
        v = gp(u)
        _, g = mx.vjp(gp, [u], [mx.ones((n_chains,), dtype=mx.float32)])
        mx.eval(v, g)
    return np.array(v, np.float64), np.array(g[0], np.float64)


@pytest.mark.parametrize("n_jit", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("cadence", ["regular", "gapped", "dupes", "distinct",
                                     "alternating"])
def test_per_instrument_jitter_matches_the_mlx_path(n_jit, cadence):
    """M5 stage 2: the kernel used to refuse more than one jitter.

    The jitter sits inside the innovation variance, so it reaches the value
    *and* every adjoint -- this compares both against the reference path,
    which the suite already trusts for per-instrument jitter.
    """
    from anvilgp.likelihood import GPLogLike, KernelParam, ParamMap
    n = 1200
    rng = np.random.default_rng(4)
    if cadence == "regular":
        t = np.arange(n) * 0.3
    elif cadence == "gapped":
        t = np.arange(n) * 0.3
        t[n // 2:] += 20.0
    elif cadence == "dupes":
        t = np.sort(np.concatenate([np.arange((n + 1) // 2) * 0.3] * 2))[:n]
    elif cadence == "alternating":
        t = np.sort(np.concatenate([np.arange((n + 1) // 2) * 0.3] * 2))[:n]
    else:
        t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.2, 0.4, n - 1))])
    inst = None if n_jit == 1 else (np.arange(n) % n_jit).astype(np.int32)
    jits = (0.03, 0.09, 0.2, 0.05)
    y = rng.standard_normal(n)          # once: both paths must see one dataset

    def build(metal):
        pm = ParamMap(
            kernel=[KernelParam(link="log", index=i, ref=r)
                    for i, r in enumerate((1.3, 4.0, 3.0))],
            jitter=[KernelParam(link="log", index=3 + j,
                                ref=jits[j % len(jits)])
                    for j in range(n_jit)],
            n_dim=3 + n_jit, instrument=inst)
        return GPLogLike(None, t, None, y, np.full(n, 0.3), ag.SHOTerm(), pm,
                         GPPolicy(block=1024, hi_block=1024, metal=metal))

    kern, ref = build(True), build(False)
    assert kern.metal_eligible()[0], kern.metal_eligible()[1]
    vk, gk = _vg(kern, 3 + n_jit)
    vm, gm = _vg(ref, 3 + n_jit)
    assert np.max(np.abs(vk - vm)) < 1e-2, f"value: {vk[0]} vs {vm[0]}"
    rel = np.max(np.abs(gk - gm)) / max(np.max(np.abs(gm)), 1e-30)
    assert rel < 1e-5, f"gradient rel {rel:.2e}\n kernel {gk[0]}\n mlx {gm[0]}"


def test_each_instruments_jitter_gets_its_own_gradient():
    """The trap the plan named, and it is sharper than a value check.

    With anvil's own priors every instrument shares a jitter reference, so
    at `v = 0` all jitters are numerically equal, permuting instrument
    labels leaves the log-likelihood **bit-identical**, and only the
    gradient moves.  A value-only test therefore passes on a kernel that
    attributes every observation to instrument 0.

    Three assertions, in increasing sharpness:
      1. with *distinct* jitters, a permutation changes the kernel's answer
         at all -- it is reading `inst`;
      2. the kernel agrees with the reference path under *both* labelings --
         it is reading `inst` *correctly*, which (1) alone does not show;
      3. with *equal* jitters the value is bit-identical across the
         permutation while the gradient still moves, which is the trap.

    Note what is *not* asserted: that an instrument left alone by the
    permutation keeps its gradient.  It does not, and should not -- the
    jitters enter one shared filter, so changing which observations carry
    which variance changes every later observation's contribution.  An
    instrument untouched by the swap moved 0.2% here.
    """
    n, n_jit = 1200, 3
    lab = (np.arange(n) % n_jit).astype(np.int32)
    swapped = lab.copy()
    swapped[lab == 0], swapped[lab == 1] = 1, 0

    # (1) and (2): distinct jitters
    out = {}
    for name, L in (("base", lab), ("swap", swapped)):
        k = _jitter_gp(n_jit, True, inst=L)
        r = _jitter_gp(n_jit, False, inst=L)
        assert k.metal_eligible()[0]
        out[name] = (_vg(k, 3 + n_jit), _vg(r, 3 + n_jit))

    (vk0, gk0), (vm0, gm0) = out["base"]
    (vk1, gk1), (vm1, gm1) = out["swap"]

    assert np.max(np.abs(gk0[0][3:] - gk1[0][3:])) > 1.0, (
        f"permuting instrument labels left the kernel's jitter gradients "
        f"unchanged ({gk0[0][3:]} vs {gk1[0][3:]}): it is not reading `inst`")
    for nm, (vk, gk, vm, gm) in (("base", (vk0, gk0, vm0, gm0)),
                                 ("swap", (vk1, gk1, vm1, gm1))):
        assert np.max(np.abs(vk - vm)) < 1e-2, f"{nm}: {vk[0]} vs {vm[0]}"
        rel = np.max(np.abs(gk - gm)) / max(np.max(np.abs(gm)), 1e-30)
        assert rel < 1e-5, f"{nm}: gradient rel {rel:.2e}"
    assert np.all(np.abs(gk0[0][3:]) > 1e-6), f"a jitter has no gradient"

    # (3) the trap: equal jitter references, as anvil's priors produce
    eq = (0.07, 0.07, 0.07, 0.07)
    vke, gke = _vg(_jitter_gp(n_jit, True, inst=lab, jits=eq), 3 + n_jit)
    vks, gks = _vg(_jitter_gp(n_jit, True, inst=swapped, jits=eq), 3 + n_jit)
    assert np.array_equal(vke, vks), (
        "with equal jitters the value must be identical under a permutation "
        "-- if it is not, this fixture no longer demonstrates the trap")
    assert np.max(np.abs(gke[0][3:] - gks[0][3:])) > 1.0, (
        f"with equal jitters only the gradient can reveal a wrong "
        f"instrument index, and it did not move: {gke[0][3:]}")


def test_more_jitters_than_the_kernel_holds_is_refused_not_wrong():
    """`MAX_JITTER` sizes a per-thread register array, so it is a real
    budget.  Above it the MLX path is correct and eligibility says so."""
    import warnings
    from anvilgp.metal import MAX_JITTER
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp = _jitter_gp(MAX_JITTER + 1, True)
    ok, why = gp.metal_eligible()
    assert not ok and "exceeds the kernel" in why, why
    # and the user is told, rather than silently getting the slow path
    assert [w for w in caught if "cannot be honoured" in str(w.message)]
    # and it still computes, on the reference path
    v, g = _vg(gp, 3 + MAX_JITTER + 1)
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(g))


def test_per_instrument_jitter_needs_instrument_labels():
    """Several jitters and no labels is a user error the kernel cannot
    guess its way out of, and `jitter_var` would raise deep inside the
    graph.  Eligibility reports it up front instead."""
    gp = _jitter_gp(3, True)
    # _jitter_gp supplies labels for n_jit > 1; strip them to make the case
    gp.param_map.instrument = None
    ok, why = gp.metal_eligible()
    assert not ok and "param_map.instrument" in why, why


def test_instrument_labels_are_range_checked():
    """`mx.take` is silent on an out-of-range label and the kernel would
    read the next chain's jitter row, so the host refuses at construction."""
    n = 400
    bad = np.zeros(n, dtype=np.int32)
    bad[5] = 7                                   # only 3 instruments declared
    with pytest.raises(ValueError, match="instrument"):
        _jitter_gp(3, True, n=n, inst=bad)
    short = np.zeros(n - 1, dtype=np.int32)
    with pytest.raises(ValueError, match="one label per observation"):
        _jitter_gp(3, True, n=n, inst=short)


def test_the_formal_variant_changes_only_the_positivity_guards():
    """The convolved-term variant drops the clamps; nothing else moves.

    Every other term runs the non-formal source, which must be the code it
    always was -- when the variant was introduced it was compared against
    the previous source with comments and whitespace stripped, and found
    identical.  This pins the substitution itself: no placeholder survives
    in either variant, the non-formal one still carries every clamp, and the
    formal one carries none of them and refuses a negative pivot instead.
    """
    tokens = ("LEAF_VARIANCE", "STRICT_PIVOT", "HCH_CLAMPED",
              "RK_CLAMP_TERM", "CLAMPED_TEST")
    fwd = metal._jitter_src(metal._SRC, 1)
    bwd = metal._jitter_src(
        metal._SRC_BWD_CKPT.replace("FWD_STEP", metal._FWD_STEP), 1)
    red = metal._SRC_REDUCE
    clamps = {
        "fwd": ["sq = max(sq, 0.0f);", "sq = max(sq, sfloor * rr);"],
        "bwd": ["max(hch, 0.0f) + r_ref[gi]", "max(-hch, 0.0f)",
                "bool clamped = hch < 0.0f;"],
        "red": [],
    }
    for name, src in (("fwd", fwd), ("bwd", bwd), ("red", red)):
        plain = metal._formal_src(src, False)
        formal = metal._formal_src(src, True)
        for tok in tokens:
            assert tok not in plain and tok not in formal, (name, tok)
        for c in clamps[name]:
            assert c in plain, (name, c)
            assert c not in formal, (name, c)
        assert "NAN" not in plain, name
    assert "L1[j][j] = NAN;" in metal._formal_src(red, True)
    assert "bool clamped = false;" in metal._formal_src(bwd, True)
