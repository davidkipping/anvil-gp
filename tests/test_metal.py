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


@pytest.mark.parametrize("n_state,cap", [(2, 8192), (3, 2048), (4, 256),
                                         (5, 128), (6, 64), (8, 64)])
def test_auto_metal_block_caps_long_blocks_at_high_state_dimension(n_state,
                                                                   cap):
    """A thread walks its whole block, so block length is a time limit.

    Per-observation work grows like m^3, and macOS kills a command buffer
    that runs too long. At N = 1e5 and 512 chains the uncapped rule chose
    8192 for every m; at m = 4 that is not slow, it is
    "[METAL] Command buffer execution failed: Impacting Interactivity",
    and 4096 fails too while 2048 failed in one run and not another. With
    the display in use the limit is stricter: m = 4 at 1024 was killed from
    ~224 chains at N = 1e5 and at 4096 chains at N = 1e4, and 512 at 4096
    chains, so m = 4 is capped at 256 (docs/boundary-plan.md, III).  At
    N = 1e5 m = 5 at 256 and m = 6 at 128 are killed from 512 chains, so
    they are capped at 128 and 64, which survive 4096 chains (follow-on
    step 2).
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
    assert gp.metal_block <= 256, "m = 4 must not get a long block"
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
                    GPPolicy(metal=False, block=4096))
    hi = float(np.array(ref.hi(mx.zeros((1, 1), dtype=mx.float64)))[0])
    gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                   GPPolicy(block=4096, metal=True))
    with mx.stream(mx.gpu):
        lo = float(np.array(gp(mx.zeros((1, 1), dtype=mx.float32)))[0])
    assert abs(lo - hi) < 0.02, (
        f"kernel off by {lo - hi:+.4f} at sigma/yerr = {sig_over_yerr:g}")


def test_the_value_does_not_carry_the_dense_information_floor():
    """Bright, densely sampled data: a TESS 20-s cadence, a 3-day rotation
    plus granulation, s/e = 100.  Each block element stores its
    log-normaliser for a zero incoming state, ~-1/2 sum g^2 over the block,
    and composing blocks cancels those against cross terms of the same
    size; the kernel took its value from that fold and was off by 0.45
    here (5.9 on a full 20-s sector at s/e = 30).  The value now comes from
    the filter replayed through each block from its entry state, every term
    an O(1) innovation: 0.055.  docs/float32.md, section 7.
    """
    from anvilgp.oracle import sample_ssm

    n, dt = 4096, 1.0 / 4320                # 20 s, in days
    term = ag.SHOTerm() + ag.SHOTerm()
    kp = [0.1, 3 / (2 * np.pi), 5.0, 0.03, 0.2, 1 / np.sqrt(2)]
    rng = np.random.default_rng(1)
    t = np.arange(n) * dt
    yerr = np.full(n, 1e-3)
    y = sample_ssm(term, kp, dt, n, rng) + yerr * rng.standard_normal(n)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=x)
                          for i, x in enumerate(kp)], n_dim=len(kp))
    v = np.random.default_rng(3).normal(0.0, 0.01, (8, len(kp)))
    v[0] = 0.0
    ref = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy(metal=False, kappa_max=None))
    with mx.stream(mx.cpu):
        hi = np.array(ref.hi(mx.array(v).astype(mx.float64)))
    gp = GPLogLike(None, t, None, y, yerr, term, pm,
                   GPPolicy(metal=True, kappa_max=None))
    with mx.stream(mx.gpu):
        lo = np.array(gp(mx.array(v, dtype=mx.float32)))
    err = np.abs(lo - hi)
    assert err.max() < 0.15, f"kernel off by up to {err.max():.3f}"


def test_the_differentiation_boundary_encloses_the_reduction():
    """M4 requirement: the custom_function's DIFFERENTIATED output is one
    scalar per chain.

    If it returned the six element fields per block instead, its VJP would
    have to accept cotangents on all six -- not the problem `adjoint.py`
    solves.  An earlier version of this test spied on `scan_loglike`'s
    output, which is one scalar per chain whether or not the reduction is
    inside the boundary; it passed with the reduction moved back outside.
    This one captures what the decorated function itself returns.

    Since 2026-10-25 it also returns the block boundary's working arrays,
    so the VJP can reuse them instead of recomputing pass 1; those are
    never differentiated (`scan_loglike` returns the first output alone, so
    their cotangents are zero) and the VJP ignores them.  The first output
    must still be the per-chain log-likelihood, and `scan_loglike` must
    return nothing else.
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
            out = gp(mx.zeros((n_chains, 1), dtype=mx.float32))
            mx.eval(out)
    finally:
        metal_mod.mx.custom_function = real_cf

    got = seen["custom_function_output"]
    assert isinstance(got, list) and got[0] == (n_chains,), (
        f"the custom_function's first output must be (n_chains,); it "
        f"returned {got}")
    assert all(len(sh) >= 2 and sh[0] == n_chains for sh in got[1:]), got
    assert out.shape == (n_chains,), out.shape



@pytest.mark.parametrize("name,term,kp", TERMS, ids=[c[0] for c in TERMS])
def test_pass_one_without_its_ell_changes_nothing_else(name, term, kp):
    """The value and the backward run pass 1 with its blocks' own
    log-normalisers stripped (`metal._ell_src`); the five fields the prefix
    and the seeds read must be bit-identical to the full kernel's, and the
    stripped `ell` zero."""
    from anvilgp.metal import scan_blocks

    gp = _build(term, kp, True, blk=64)
    n_chains = 3
    v = mx.zeros((n_chains, 1), dtype=mx.float32)
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
        full = scan_blocks(pt, qt, gp._metal_lag, gp._metal_inst, h, r,
                           gp._r_ref32, j2, gp.policy.s_floor_rel, 64)
        bare = scan_blocks(pt, qt, gp._metal_lag, gp._metal_inst, h, r,
                           gp._r_ref32, j2, gp.policy.s_floor_rel, 64,
                           with_ell=False)
        mx.eval(full, bare)
    for f in ("A", "b", "C", "eta", "J"):
        assert np.array_equal(np.array(getattr(full, f)),
                              np.array(getattr(bare, f))), f
    assert not np.any(np.array(bare.ell)) and np.any(np.array(full.ell))


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
        tiers, groups = plan if kind == "hybrid" else ([], plan)
        used = [g[0] for g in groups]
        for sel, lags_of in groups:
            assert np.array_equal(slot_lag[sel], lags_of), name
            assert np.unique(lags_of).size == lags_of.size, name
        for lags_t, table in tiers:
            assert np.unique(lags_t).size == lags_t.size, name
            keep = table != n_slots
            used.append(table[keep])
            rows = np.repeat(lags_t, table.shape[1])[keep.ravel()]
            assert np.array_equal(rows, slot_lag[table[keep]]), name
        used = np.concatenate(used)
        assert np.array_equal(np.sort(used), np.arange(n_slots)), name


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
        GPPolicy(metal=False, metal_block=0)


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


def test_more_jitters_than_the_old_cap_run_on_the_kernel():
    """Until 2026-10-25 the kernel kept one jitter adjoint per instrument
    in registers and refused more than eight (`MAX_JITTER`).  Several
    jitters now go per datum (`metal._jitter_src`), so there is no cap:
    twelve run on the kernel and match the MLX path, value and gradient."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp = _jitter_gp(12, True)
    ok, why = gp.metal_eligible()
    assert ok and gp._metal_ok, why
    assert not [w for w in caught if "cannot be honoured" in str(w.message)]
    vk, gk = _vg(gp, 3 + 12)
    vm, gm = _vg(_jitter_gp(12, False), 3 + 12)
    assert np.max(np.abs(vk - vm)) < 1e-2
    rel = np.max(np.abs(gk - gm)) / np.max(np.abs(gm))
    assert rel < 1e-5, f"gradient rel {rel:.2e}"


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


def test_the_formal_variant_changes_only_the_positivity_assumptions():
    """The convolved-term variant drops what assumes positivity; nothing else.

    Every other term runs the non-formal source, which must be the code it
    always was -- when the variant was introduced it was compared against
    the previous source with comments and whitespace stripped, and found
    identical.  This pins the substitution itself: no placeholder survives
    in either variant, the non-formal one still carries every clamp and
    Cholesky factor, and the formal one carries none of them and composes
    through the pivoted inverse instead.
    """
    tokens = ("LEAF_VARIANCE", "REDUCE_CORE", "REDUCE_MD", "LOGDET_ACC",
              "COMBINE_FACTOR", "COMBINE_RR", "LOG_INV_A", "HCH_CLAMPED", "RK_CLAMP_TERM", "CLAMPED_TEST",
              "C1N", "FWD_ELL", "FWD_STEP", "COMBINE_CORE")
    srcs = {
        "fwd": metal._jitter_src(metal._SRC, 1),
        "bwd": metal._jitter_src(
            metal._SRC_BWD_CKPT.replace("FWD_STEP", metal._fwd_step()), 1),
        "red": metal._SRC_REDUCE,
        "seeds": metal._SRC_SEEDS.replace("COMBINE_CORE", metal._COMBINE_CORE),
        "entry": (metal._SEEDS_HEAD + metal._SRC_PREFIX)
        .replace("COMBINE_CORE", metal._COMBINE_CORE),
        "replay": metal._jitter_src(
            metal._SRC_REPLAY.replace("FWD_STEP", metal._fwd_step(True)), 1),
    }
    only_plain = {
        "fwd": ["sq = max(sq, 0.0f);", "sq = max(sq, sfloor * rr);",
                "log(inv_a)"],
        "bwd": ["max(hch, 0.0f) + r_ref[gi]", "max(-hch, 0.0f)",
                "bool clamped = hch < 0.0f;"],
        "red": ["L1[j][j] = metal::precise::sqrt(max(d, fl));",
                "log(Ls[i][i])"],
        "seeds": ["L1[j][j] = metal::precise::sqrt(max(d, flc));"],
        "entry": ["L1[j][j] = metal::precise::sqrt(max(d, flc));"],
        "replay": ["max(hch, 0.0f) + r_ref[gi]", "max(-hch, 0.0f)"],
    }
    only_formal = {
        "fwd": ["log(fabs(inv_a))"],
        "bwd": ["bool clamped = false;"],
        "red": ["ldx += metal::precise::log(fabs(dk));", "0.5f * ldx"],
        "seeds": ["ldx += metal::precise::log(fabs(dk));"],
        "entry": ["ldx += metal::precise::log(fabs(dk));"],
        "replay": ["float sv = hch + r_ref[gi]"],
    }
    for name, src in srcs.items():
        plain = metal._formal_src(src, False)
        formal = metal._formal_src(src, True)
        for tok in tokens:
            assert tok not in plain and tok not in formal, (name, tok)
        for c in only_plain[name]:
            assert c in plain and c not in formal, (name, c)
        for c in only_formal[name]:
            assert c in formal and c not in plain, (name, c)


def test_an_explicit_block_above_the_cap_warns():
    """Allowed, but the GPU watchdog's trigger: say so at construction."""
    n = 3000
    t = np.arange(n) * 0.01
    y = np.random.default_rng(0).standard_normal(n)
    pm = ParamMap.log([1.0, 4.0, 3.0, 0.3, 0.5, 0.7])
    term = ag.SHOTerm() + ag.SHOTerm()
    with pytest.warns(UserWarning, match="GPU watchdog"):
        GPLogLike(None, t, None, y, np.full(n, 0.3), term, pm,
                  GPPolicy(metal=True, metal_block=1024))
    import warnings as _w
    with _w.catch_warnings():
        _w.filterwarnings("error", message=".*GPU watchdog.*")
        GPLogLike(None, t, None, y, np.full(n, 0.3), term, pm,
                  GPPolicy(metal=True, metal_block=256))
        GPLogLike(None, t, None, y, np.full(n, 0.3), term, pm,
                  GPPolicy(metal=True))
        GPLogLike(None, t, None, y, np.full(n, 0.3), ag.SHOTerm(),
                  ParamMap.log([1.0, 4.0, 3.0]),
                  GPPolicy(metal=True, metal_block=2048))


@pytest.mark.slow
def test_the_kernel_gradient_survives_many_chains_at_high_state():
    """The configuration the 1024 and 512 caps were killed in with the
    display in use: m = 4, N = 1e5, 4096 chains, value + gradient.  The
    watchdog's limit depends on display load, so this can pass at a bad
    cap on an idle machine; the cap itself is pinned above."""
    n, n_chains = 100_000, 4096
    rng = np.random.default_rng(0)
    t = np.arange(n) * (30.0 / n)
    gp = GPLogLike(None, t, None, rng.standard_normal(n) * 0.9,
                   np.full(n, 0.3), ag.SHOTerm() + ag.SHOTerm(),
                   ParamMap.log([1.0, 4.0, 3.0, 0.3, 0.5, 0.7]),
                   GPPolicy(block=4096, metal=True, kappa_max=None))
    assert gp.metal_block == 256
    u = mx.array(rng.standard_normal((n_chains, 6)).astype(np.float32) * 0.01)
    with mx.stream(mx.gpu):
        v, g = mx.vjp(gp, [u], [mx.ones((n_chains,), dtype=mx.float32)])
        mx.eval(v, g)
    assert np.all(np.isfinite(np.array(v))) and np.all(np.isfinite(np.array(g[0])))


# ------------------------------- the default: the kernel where eligible
def _default_build(term, p, policy):
    n = 800
    t = np.arange(n) * 0.05
    y = np.random.default_rng(0).standard_normal(n)
    return GPLogLike(None, t, None, y, np.full(n, 0.3), term,
                     ParamMap.log(p), policy)


def test_the_default_policy_uses_the_kernel_where_eligible():
    """`GPPolicy.metal` is None by default since 2026-10-25: the kernel
    when it can serve the problem, bit for bit what `metal=True` gives;
    the MLX path otherwise (m > 6), without the warning `metal=True`
    gives there."""
    import warnings as _w
    from anvilgp.likelihood import MetalDefaultNotice
    assert GPPolicy().metal is None
    u = mx.array(np.random.default_rng(1).normal(0, 0.1, (3, 3))
                 .astype(np.float32))
    with _w.catch_warnings():
        _w.simplefilter("ignore", MetalDefaultNotice)
        dflt = _default_build(ag.SHOTerm(), [1.3, 4.0, 3.0], GPPolicy())
        asked = _default_build(ag.SHOTerm(), [1.3, 4.0, 3.0],
                               GPPolicy(metal=True))
        assert dflt._metal_ok and asked._metal_ok
        with mx.stream(mx.gpu):
            got = [(np.array(gp(u)),
                    np.array(mx.grad(lambda u: gp(u).sum())(u)))
                   for gp in (dflt, asked)]
        np.testing.assert_array_equal(got[0][0], got[1][0])
        np.testing.assert_array_equal(got[0][1], got[1][1])

        big = ag.RotationTerm() + ag.RotationTerm()     # m = 8
        _w.simplefilter("error", RuntimeWarning)
        gp8 = _default_build(big, [1.3, 4.0, 3.0, 1.0, 0.4] * 2, GPPolicy())
        assert not gp8._metal_ok
    with pytest.warns(RuntimeWarning, match="cannot be honoured"):
        _default_build(big, [1.3, 4.0, 3.0, 1.0, 0.4] * 2, GPPolicy(metal=True))


def test_the_default_says_once_per_process_which_path_it_chose(monkeypatch):
    """The default moved under scripts that never asked for either path,
    so it says which it took -- once per process and outcome, not per
    instance -- and says nothing when the path was asked for."""
    import warnings as _w
    import anvilgp.likelihood as L
    monkeypatch.setattr(L, "_METAL_NOTICES", set())
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        for _ in range(3):
            _default_build(ag.SHOTerm(), [1.3, 4.0, 3.0], GPPolicy())
        _default_build(ag.SHOTerm(), [1.3, 4.0, 3.0], GPPolicy(metal=False))
        _default_build(ag.SHOTerm(), [1.3, 4.0, 3.0], GPPolicy(metal=True))
        for _ in range(2):
            _default_build(ag.RotationTerm() + ag.RotationTerm(),
                           [1.3, 4.0, 3.0, 1.0, 0.4] * 2, GPPolicy())
    notes = [str(r.message) for r in rec
             if issubclass(r.category, L.MetalDefaultNotice)]
    assert len(notes) == 2, notes
    assert "fused Metal kernel" in notes[0] and "MLX path" in notes[1]


def test_make_target_gets_the_kernel_by_default():
    pytest.importorskip("anvil")
    import warnings as _w
    from anvilgp.anvil import make_target
    n = 400
    t = np.arange(n) * 0.05
    y = np.random.default_rng(0).standard_normal(n)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        gt = make_target(ag.SHOTerm(), t, y, np.full(n, 0.3))
    assert gt.loglike._metal_ok


# ---------------------------------------- m = 5 and 6 (follow-on step 2)
def _grad_sigma_vs_float64(term, kp, dt_s, se, n=2048, n_draws=4):
    """Worst kernel gradient error, in Laplace posterior sigma, over
    `n_draws` posterior draws, against a float64 central difference; and
    the value's worst error.  On-model data at s/e = `se`."""
    from anvilgp.oracle import sample_ssm
    rng = np.random.default_rng(2)
    dt = dt_s / 86400
    t = np.arange(n) * dt
    y0 = sample_ssm(term, kp, dt, n, rng)
    yerr = np.full(n, np.std(y0) / se)
    y = y0 + yerr * rng.standard_normal(n)
    d = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=x)
                          for i, x in enumerate(kp)], n_dim=d)
    ref = GPLogLike(None, t, None, y, yerr, term, pm,
                    GPPolicy(metal=False, kappa_max=None))

    def hi(V):
        with mx.stream(mx.cpu):
            return np.array(ref.hi(mx.array(V, dtype=mx.float64),
                                   refuse=False))

    hs, pts = 1e-3, [np.zeros(d)]
    for i in range(d):
        for j in range(i, d):
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = np.zeros(d)
                p[i] += si * hs
                p[j] += sj * hs
                pts.append(p)
    f, H, c = hi(np.array(pts)), np.zeros((d, d)), 1
    for i in range(d):
        for j in range(i, d):
            fpp, fpm, fmp, fmm = f[c:c + 4]
            c += 4
            H[i, j] = H[j, i] = -(fpp - fpm - fmp + fmm) / (4 * hs ** 2)
    w, U = np.linalg.eigh(H)
    w = np.maximum(w, 4.0)
    sd = np.sqrt(np.diag((U / w) @ U.T))
    V = (U / np.sqrt(w) @ np.random.default_rng(99)
         .standard_normal((d, n_draws))).T
    gh = 1e-5
    P = np.repeat(V, 2 * d, axis=0)
    for k in range(n_draws):
        for i in range(d):
            P[k * 2 * d + 2 * i, i] += gh
            P[k * 2 * d + 2 * i + 1, i] -= gh
    ff = hi(P).reshape(n_draws, d, 2)
    G = (ff[..., 0] - ff[..., 1]) / (2 * gh)

    gp = GPLogLike(None, t, None, y, yerr, term, pm,
                   GPPolicy(metal=True, kappa_max=None))
    from anvilgp.likelihood import _MAX_METAL_BLOCK, _MAX_METAL_BLOCK_HI_M
    assert gp._metal_ok and gp.metal_block == _MAX_METAL_BLOCK.get(
        term.n_state, _MAX_METAL_BLOCK_HI_M)
    with mx.stream(mx.gpu):
        u = mx.array(V, dtype=mx.float32)
        g = np.array(mx.grad(lambda u: gp(u).sum())(u))
        lo = np.array(gp(u))
    return (np.abs(g - G) * sd).max(), np.abs(lo - hi(V)).max()


@pytest.mark.parametrize("name,term,kp,dt_s", [
    ("rotation+sho, 2 min", ag.RotationTerm() + ag.SHOTerm(),
     [1.0, 3.0, 3.0, 1.0, 0.4, 0.3, 0.2, 1 / np.sqrt(2)], 120.0),
    ("3 x sho, 20 s", ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm(),
     [0.1, 3 / (2 * np.pi), 5.0, 0.03, 0.2, 1 / np.sqrt(2),
      0.05, 0.05, 1 / np.sqrt(2)], 20.0),
    ("sho+matern52, 2 min", ag.SHOTerm() + ag.Matern52Term(),
     [0.1, 3 / (2 * np.pi), 5.0, 0.03, 0.5], 120.0),
])
def test_the_kernel_at_m5_and_m6_against_float64(name, term, kp, dt_s):
    """m = 5 and 6 joined the kernel on 2026-10-25 (`MAX_STATE = 6`) at
    128- and 64-point blocks.  Measured at N = 4096 and 16 draws: 0.12
    sigma (rotation + SHO, 2 min), 0.12 (3 x SHO, 20 s), 0.07 (SHO +
    Matern-5/2, 2 min).  Not here, because the fast-cadence floor owns
    them: SHO + Matern-5/2 at 20 s is 0.6-0.67 (the MLX path 1.42), and
    rotation + SHO at 20 s is 6.1 (the MLX path 1.07 with 2 of 16 draws
    refused, and no gradient at all from N ~ 16384); rotation alone at
    m = 4 is worse (README, "Still open")."""
    e, v = _grad_sigma_vs_float64(term, kp, dt_s, se=100.0)
    assert e < 0.5, f"{name}: gradient {e:.3f} posterior sigma"
    assert v < 0.2, f"{name}: value off by {v:.3f}"


def test_m6_per_instrument_jitter_matches_the_mlx_path():
    """The jitter accumulators share the thread's registers with the m = 6
    element, which already spills; value and gradient against the MLX path
    with three instruments."""
    n, n_jit = 1200, 3
    rng = np.random.default_rng(4)
    t = np.arange(n) * 0.05
    inst = (np.arange(n) % n_jit).astype(np.int32)
    y = rng.standard_normal(n)
    kp = (1.0, 3.0, 3.0, 1.0, 0.4, 0.3, 0.5, 0.7)

    def build(metal):
        pm = ParamMap(
            kernel=[KernelParam(link="log", index=i, ref=r)
                    for i, r in enumerate(kp)],
            jitter=[KernelParam(link="log", index=len(kp) + j, ref=r)
                    for j, r in enumerate((0.03, 0.09, 0.2))],
            n_dim=len(kp) + n_jit, instrument=inst)
        return GPLogLike(None, t, None, y, np.full(n, 0.3),
                         ag.RotationTerm() + ag.SHOTerm(), pm,
                         GPPolicy(block=1024, hi_block=1024, metal=metal))

    kern, ref = build(True), build(False)
    assert kern.metal_eligible()[0], kern.metal_eligible()[1]
    vk, gk = _vg(kern, len(kp) + n_jit)
    vm, gm = _vg(ref, len(kp) + n_jit)
    assert np.max(np.abs(vk - vm)) < 1e-2, f"value: {vk[0]} vs {vm[0]}"
    rel = np.max(np.abs(gk - gm)) / max(np.max(np.abs(gm)), 1e-30)
    assert rel < 1e-4, f"gradient rel {rel:.2e}\n kernel {gk[0]}\n mlx {gm[0]}"


@pytest.mark.slow
def test_the_kernel_gradient_survives_many_chains_at_m6():
    """The m = 6 cap's stress case: N = 1e5, 4096 chains, value + gradient,
    with the eight per-instrument jitters whose accumulators share the
    spilled registers.  128-point blocks were killed here from 512 chains
    with the display in use; 64 survived (37 s an evaluation)."""
    n, n_chains, n_jit = 100_000, 4096, 8
    rng = np.random.default_rng(0)
    t = np.arange(n) * (30.0 / n)
    kp = [1.0, 3.0, 3.0, 1.0, 0.4, 0.3, 0.5, 0.7]
    pm = ParamMap(
        kernel=[KernelParam(link="log", index=i, ref=r)
                for i, r in enumerate(kp)],
        jitter=[KernelParam(link="log", index=len(kp) + j, ref=0.05)
                for j in range(n_jit)],
        n_dim=len(kp) + n_jit,
        instrument=(np.arange(n) % n_jit).astype(np.int32))
    gp = GPLogLike(None, t, None, rng.standard_normal(n) * 0.9,
                   np.full(n, 0.3), ag.RotationTerm() + ag.SHOTerm(), pm,
                   GPPolicy(block=4096, metal=True, kappa_max=None))
    assert gp._metal_ok and gp.metal_block == 64
    u = mx.array(rng.standard_normal((n_chains, pm.n_dim))
                 .astype(np.float32) * 0.01)
    with mx.stream(mx.gpu):
        v, g = mx.vjp(gp, [u], [mx.ones((n_chains,), dtype=mx.float32)])
        mx.eval(v, g)
    assert np.all(np.isfinite(np.array(v))) and np.all(np.isfinite(np.array(g[0])))


def test_a_deep_ragged_fold_is_tiered_not_one_scatter_per_block():
    """Hundreds of short blocks on a nearly regular cadence -- a held-out
    fit with one window per transit -- stack a few lags in every block and
    many lags in one each.  One group per rank was a scatter pair per
    block, 201 + 201 + 400 of them per gradient at 200 windows (2026-11-10
    review, E3).  The hybrid plan folds the deep lags densely in tiers
    and leaves at most `_MAX_FOLD_RANKS` groups; it is the same sum, and
    reproducible."""
    from anvilgp.likelihood import (_fold_plan_mx, _lag_fold_plan,
                                    _MAX_FOLD_RANKS)
    from anvilgp.metal import _fold_slots
    rng = np.random.default_rng(0)
    n_blocks, n_lags = 300, 420
    slot_lag = []
    for b in range(n_blocks):
        lags = {1, 2} | set(rng.choice(np.arange(3, 12), 3, replace=False))
        lags |= {12 + b, 12 + (b * 7) % 400}
        slot_lag.extend(sorted(lags))
    slot_lag = np.array(slot_lag, dtype=np.int32)
    kind, plan = _lag_fold_plan(slot_lag, n_lags)
    assert kind == "hybrid"
    tiers, groups = plan
    assert len(groups) <= _MAX_FOLD_RANKS and len(tiers) <= 10
    for _, table in tiers:
        assert table.size < 2 * (table != slot_lag.size).sum() + table.shape[0]
    n, m = 3, 2
    d = rng.standard_normal((n, slot_lag.size, m, m)).astype(np.float32)
    want = np.zeros((n, n_lags, m, m))
    np.add.at(want, (slice(None), slot_lag), d.astype(np.float64))
    fold = _fold_plan_mx(kind, plan)
    outs = []
    for _ in range(3):
        a, b = _fold_slots(mx.array(d), mx.array(2 * d), fold, n, n_lags, m)
        outs.append((np.array(a), np.array(b)))
    np.testing.assert_allclose(outs[0][0], want, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(outs[0][1], 2 * want, rtol=2e-5, atol=2e-5)
    assert all(np.array_equal(o[0], outs[0][0]) for o in outs[1:])


# ---- the forward's replay writes the backward's checkpoints ---------------

def _ckpt_case(term, n_par, basis, mask, flag, n=1200):
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 30, n))
    y, yerr = rng.standard_normal(n), np.full(n, 0.3)
    kw = {}
    if basis:
        tc = (t - 15.0) / 15.0
        kw["linear_basis"] = lambda vm, x, m0: mx.stack(
            [mx.ones_like(m0), mx.broadcast_to(mx.array(
                tc.astype(np.float32)), m0.shape)], axis=-1)
    if mask:
        kw["held_out"] = [(9.0, 11.0), (20.0, 20.5)]
    gp = GPLogLike(None, t, None, y, yerr, term,
                   ParamMap.log([1.0] * n_par),
                   GPPolicy(metal=True, metal_block=128, kappa_max=None,
                            forward_checkpoints=flag), **kw)
    v = mx.array(0.1 * rng.standard_normal((8, n_par)).astype(np.float32))
    val, grad = mx.value_and_grad(lambda x: mx.sum(gp(x)))(v), None
    return np.array(gp(v)), np.array(val[1])


@pytest.mark.parametrize("mask", [False, True], ids=["plain", "held_out"])
@pytest.mark.parametrize("basis", [False, True], ids=["one", "columns"])
@pytest.mark.parametrize("m", [2, 4])
def test_checkpoints_from_the_replay_are_the_backwards_own(m, basis, mask):
    """`forward_checkpoints`: the same filter from the same entry states,
    so the value is untouched and the gradient is the backward's own to
    rounding (the kernels are separate compilations)."""
    term = ag.SHOTerm() if m == 2 else ag.SHOTerm() + ag.SHOTerm()
    v0, g0 = _ckpt_case(term, 3 * m // 2, basis, mask, False)
    v1, g1 = _ckpt_case(term, 3 * m // 2, basis, mask, True)
    np.testing.assert_array_equal(v0, v1)
    assert np.all(np.isfinite(g0))
    np.testing.assert_allclose(g1, g0, rtol=2e-4,
                               atol=2e-5 * np.abs(g0).max())


def test_forward_checkpoints_by_default_only_where_they_are_small():
    from anvilgp.metal import forward_checkpoints as fc
    # m <= 3: no larger than the residual at any size
    assert fc(None, 2, 0, 100_000, 4096, 4096)
    assert fc(None, 3, 0, 100_000, 4096, 1024)
    # m = 4 and 6: 2.5 and 5.25 of it, so only inside the byte budget
    assert fc(None, 4, 0, 10_000, 1024, 256)
    assert not fc(None, 4, 0, 100_000, 1024, 256)
    assert not fc(None, 6, 0, 100_000, 1024, 64)
    assert fc(True, 6, 0, 100_000, 1024, 64)
    assert not fc(False, 2, 0, 1000, 8, 256)
    with pytest.raises(ValueError, match="forward_checkpoints"):
        GPPolicy(forward_checkpoints=1)


# ---- the chunked front end emits the kernel's tables directly -------------

_TABLE_TERMS = _CHUNK_TERMS + [
    ("convolved", ag.TermConvolution(ag.SHOTerm(), 0.05), [1.3, 4.0, 3.0])]


def _front_end_at_budget(gp, u, budget):
    import anvilgp.likelihood as L
    old = L._FRONT_END_BUDGET_BYTES
    L._FRONT_END_BUDGET_BYTES = budget
    try:
        with mx.stream(mx.gpu):
            out = gp._metal_front_end(u)
            val, grad = mx.value_and_grad(lambda x: mx.sum(gp(x)))(u)
            mx.eval(out, val, grad)
        return [np.array(x) for x in out], float(val), np.array(grad)
    finally:
        L._FRONT_END_BUDGET_BYTES = old


@pytest.mark.parametrize("name,term,pars", _TABLE_TERMS,
                         ids=[c[0] for c in _TABLE_TERMS])
def test_the_chunked_front_end_gives_the_same_tables(name, term, pars):
    """`_metal_front_end`: chunked, the tables are concatenated from the
    chunks with row 0 as one more piece (the peak on an all-distinct
    cadence is the term algebra's, 9.9 -> 6.3 tables at N = 20000 x 1024
    chains).  Bit for bit the unchunked tables, and the same gradient --
    a convolved term's row 0 of ``Q`` is its own stationary covariance,
    which depends on the parameters."""
    gp, u = _distinct_gp(term, pars)
    u = u + 0.05 * mx.array(np.random.default_rng(1).standard_normal(
        u.shape).astype(np.float32))
    whole, v0, g0 = _front_end_at_budget(gp, u, 1 << 60)
    gp._metal_matrices_fn = None
    chunk, v1, g1 = _front_end_at_budget(gp, u, 1 << 18)
    assert gp._metal_matrices_fn is not None, "the chunked branch must run"
    assert gp._metal_matrices_fn[0][2], "and its forward must be chunked"
    for a, b in zip(whole, chunk):
        np.testing.assert_array_equal(a, b)
    assert v0 == v1
    np.testing.assert_allclose(g1, g0, rtol=2e-5,
                               atol=2e-6 * np.abs(g0).max())


def test_many_front_end_chunks_compile():
    """The chunks' gradients are stacked and summed.  A running ``acc +
    g`` is one fused kernel under `mx.compile`, and for a sum of terms at
    a few hundred chunks it had more arguments than Metal allows: "Too
    many inputs/outputs fused" on SHO + SHO at N = 20000 x 1024 chains
    (235 chunks), which anvil's compiled step would hit."""
    import anvilgp.likelihood as L
    gp, u = _distinct_gp(ag.SHOTerm() + ag.SHOTerm(),
                         [1.3, 4.0, 3.0, 0.7, 1.0, 2.0], n_chains=64)
    whole = _grad_at_budget(gp, u, 1 << 60)
    old, old_cap = L._FRONT_END_BUDGET_BYTES, L._FRONT_END_MAX_CHUNKS
    L._FRONT_END_BUDGET_BYTES = 4_000_000
    L._FRONT_END_MAX_CHUNKS = 10 ** 6     # the count this crashed at
    try:
        assert L._front_end_chunks(u.shape[0], gp._metal_uniq_dt.shape[0],
                                   4) > 200
        f = mx.compile(mx.value_and_grad(lambda x: mx.sum(gp(x))))
        with mx.stream(mx.gpu):
            g = np.array(f(u)[1], np.float64)
    finally:
        L._FRONT_END_BUDGET_BYTES, L._FRONT_END_MAX_CHUNKS = old, old_cap
    assert np.max(np.abs(g - whole)) < 1e-5 * np.max(np.abs(whole))


def test_auto_metal_block_is_shorter_with_a_row_per_observation():
    """An all-distinct cadence at the regular cadence's 8192 was killed by
    the GPU watchdog at N = 1e5 x 1024 chains (`_auto_metal_block`)."""
    from anvilgp.likelihood import _auto_metal_block
    assert _auto_metal_block(100_000, 2) == 8192
    assert _auto_metal_block(100_000, 2, n_lags=14) == 8192
    assert _auto_metal_block(100_000, 2, n_lags=99_591) == 256
    assert _auto_metal_block(100_000, 4, n_lags=99_591) == 256
    # 1025 lags survived 8192 (120 value and 100 gradient calls); 1281
    # was killed once in two runs, and every longer block from 1538 on
    assert _auto_metal_block(100_000, 2, n_lags=1024) == 8192
    assert _auto_metal_block(100_000, 2, n_lags=1281) == 256
    assert _auto_metal_block(50_000, 2, n_lags=2049) == 256
    # where the regular rule's block is short already it stays
    assert _auto_metal_block(20_000, 2, n_lags=2001) == 2048
    assert _auto_metal_block(20_000, 2, n_lags=19_980) == 256
    gp, _ = _distinct_gp(ag.SHOTerm(), [1.3, 4.0, 3.0], n=60_000)
    assert gp._metal_uniq_dt.shape[0] > 4096 and gp.metal_block == 256


def test_the_front_end_chunk_count_is_capped():
    """Metal allows 499,000 live buffers and every chunk's backward is up
    to a thousand small arrays: the budget alone asks for 1228 chunks at
    SHO + SHO, N = 1e5 x 1024 chains, and value + gradient died with
    "Resource limit (499000) exceeded" between 448 and 1024 chunks,
    depending on the term (`_FRONT_END_MAX_CHUNKS`).  The heaviest term
    measured, at the cap, on a table forced to need more."""
    import anvilgp.likelihood as L
    assert L._front_end_chunks(1024, 100_001, 4) == L._FRONT_END_MAX_CHUNKS
    assert L._front_end_chunks(1024, 100_001, 6) == L._FRONT_END_MAX_CHUNKS
    term = ag.TermConvolution(ag.RotationTerm() + ag.SHOTerm(), 0.05)
    gp, u = _distinct_gp(term, [1.5, 7.0, 3.0, 1.0, 0.4, 1.3, 4.0, 3.0],
                         n_chains=4)
    old = L._FRONT_END_BUDGET_BYTES
    L._FRONT_END_BUDGET_BYTES = 1 << 12
    try:
        assert L._front_end_chunks(4, gp._metal_uniq_dt.shape[0],
                                   6) == L._FRONT_END_MAX_CHUNKS
        f = mx.compile(mx.value_and_grad(lambda x: mx.sum(gp(x))))
        with mx.stream(mx.gpu):
            val, g = f(u)
            mx.eval(val, g)
    finally:
        L._FRONT_END_BUDGET_BYTES = old
    assert np.all(np.isfinite(np.array(g)))
