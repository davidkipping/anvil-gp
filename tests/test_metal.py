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
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
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


def test_differentiating_the_kernel_says_what_to_do_instead():
    """Forward-only until M4 -- but the error has to name the fix, not leave
    the user with MLX's '[Primitive::vjp] Not implemented for CustomKernel'."""
    refs = [1.0, 4.0, 3.0]
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                          for i, r in enumerate(refs)], n_dim=3)
    gp = GPLogLike(None, T, None, Y, YERR, ag.SHOTerm(), pm,
                   GPPolicy(metal=True, metal_block=64))
    with mx.stream(mx.gpu):
        u = mx.zeros((2, 3), dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="forward-only until M4"):
            mx.eval(mx.vjp(gp, [u], [mx.ones((2,), dtype=mx.float32)])[1][0])
        # the forward path still works after the failed differentiation
        assert bool(mx.all(mx.isfinite(gp(u))))


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
    """An M4 requirement, not tidiness: pin it so a refactor cannot undo it.

    If the `custom_function` wrapped the kernel alone, its outputs would be
    six element fields per block and its VJP would have to accept cotangents
    on all six. That is a harder problem than the one `adjoint.py` solves.
    Wrapping the reduction too makes the output one scalar per chain, and a
    scalar cotangent is what `filter_backward` is derived for.
    """
    import anvilgp.metal as metal_mod

    seen = {}
    real = metal_mod.scan_loglike

    def spy(*a, **kw):
        out = real(*a, **kw)
        seen["shape"] = out.shape
        return out

    n_chains = 6
    metal_mod.scan_loglike = spy
    try:
        gp = _build(ag.SHOTerm(), [1.3, 4.0, 3.0], True)
        with mx.stream(mx.gpu):
            mx.eval(gp(mx.zeros((n_chains, 1), dtype=mx.float32)))
    finally:
        metal_mod.scan_loglike = real

    assert seen["shape"] == (n_chains,), (
        f"the boundary must return one scalar per chain; got {seen['shape']}")
