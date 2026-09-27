"""The cancellation-free special functions, against mpmath and against the
naive forms they replace."""

import math

import mlx.core as mx
import numpy as np
import pytest

from anvilgp import special as sp

D = mx.float64


def _f64(x):
    return mx.array(np.asarray(x, dtype=np.float64), dtype=mx.float64)


@pytest.fixture(scope="module")
def mpm():
    import mpmath as mpm        # declared test dep: never skip silently
    mpm.mp.dps = 40
    return mpm


U = np.concatenate([-np.logspace(2, -14, 40), [0.0], np.logspace(-14, 4, 40)])


def test_sinc_gen_matches_mpmath(mpm):
    def ref(x):
        if x == 0:
            return 1.0
        s = mpm.sqrt(abs(mpm.mpf(float(x))))
        return float(mpm.sin(s) / s if x > 0 else mpm.sinh(s) / s)

    got = np.array(sp.sinc_gen(_f64(U)))
    exp = np.array([ref(x) for x in U])
    # Measured against the function's own envelope 1/sqrt(u), not against the
    # value: relative accuracy *at* a zero of sin(sqrt u) is unattainable for
    # anyone, because the argument sqrt(u) carries rounding of its own and the
    # derivative there is maximal.  The envelope is the honest scale.
    env = np.maximum(np.abs(exp), 1.0 / np.sqrt(np.maximum(np.abs(U), 1.0)))
    assert np.max(np.abs(got - exp) / env) < 1e-13


def test_cosm1_gen_matches_mpmath(mpm):
    def ref(x):
        s = mpm.sqrt(abs(mpm.mpf(float(x))))
        return float(mpm.cos(s) - 1 if x >= 0 else mpm.cosh(s) - 1)

    got = np.array(sp.cosm1_gen(_f64(U)))
    exp = np.array([ref(x) for x in U])
    scale = np.maximum(np.abs(exp), 1e-300)
    assert np.max(np.abs(got - exp) / scale) < 1e-14


def test_sincm1_matches_mpmath(mpm):
    def ref(x):
        if x == 0:
            return 1.0 / 6.0
        s = mpm.sqrt(abs(mpm.mpf(float(x))))
        sinc = mpm.sin(s) / s if x > 0 else mpm.sinh(s) / s
        return float((1 - sinc) / mpm.mpf(float(x)))

    got = np.array(sp.sincm1(_f64(U)))
    exp = np.array([ref(x) for x in U])
    assert np.max(np.abs(got - exp) / np.abs(exp)) < 1e-14


def test_gamma3_matches_mpmath_and_beats_the_naive_form(mpm):
    """The one function the float32 process noise stands on.

    Also records *why* it exists: the direct expression it replaces is the
    ``P_inf - Phi P_inf Phi^T`` cancellation, which loses everything below
    ``z ~ 1e-5`` even in float64.
    """
    z = np.concatenate([np.logspace(-12, 3, 60), [0.0]])
    exp = np.array([float(mpm.gammainc(3, 0, mpm.mpf(float(x)),
                                       regularized=True)) for x in z])
    got = np.array(sp.gamma3(_f64(z)))
    scale = np.maximum(np.abs(exp), 1e-300)
    assert np.max(np.abs(got - exp) / scale) < 1e-14

    naive = 1.0 - np.exp(-z) * (1.0 + z + 0.5 * z * z)
    naive_err = np.max(np.abs(naive - exp) / scale)
    assert naive_err > 1.0, "the naive form is supposed to be catastrophic"


def test_exp_acc_is_true_fp64_where_mlx_exp_is_not():
    """MLX 0.32's float64 ``exp`` is only float32-accurate.

    ``E = exp(-2 c dt)`` cannot be recovered as ``1 + expm1(-2 c dt)``
    either, so the verification path needs a written-out exponential.
    """
    x = np.linspace(-600.0, 600.0, 1201)
    ref = np.exp(x.astype(np.longdouble)).astype(np.float64)
    acc = np.array(sp.exp_acc(_f64(x)))
    native = np.array(mx.exp(_f64(x)))
    rel_acc = np.max(np.abs(acc - ref) / ref)
    rel_native = np.max(np.abs(native - ref) / ref)
    assert rel_acc < 1e-15
    assert rel_native > 1e-8, "MLX fp64 exp is expected to be fp32-accurate"


def test_sincos_is_true_fp64_where_mlx_sin_is_not():
    x = np.linspace(-1e4, 1e4, 4001)
    s, c = sp.sincos(_f64(x))
    assert np.max(np.abs(np.array(s) - np.sin(x))) < 1e-14
    assert np.max(np.abs(np.array(c) - np.cos(x))) < 1e-14
    assert np.max(np.abs(np.array(mx.sin(_f64(x))) - np.sin(x))) > 1e-9


def test_damped_cos_sinc_never_overflows_in_the_overdamped_corner():
    """``e^-a cosh(phi)`` is bounded by 1; forming the factors separately
    gives ``inf * 0 = NaN`` and loses the chain.

    Swept over a real overdamped SHO (c > |d|, so a > phi) at lags long
    enough that ``cosh(phi)`` alone overflows float64 many times over.
    """
    w0, q = 1.0, 0.05                       # heavily overdamped
    c = w0 / (2 * q)
    absd = math.sqrt(c * c - w0 * w0)
    dt = np.logspace(0, 4, 40)
    a = _f64(c * dt)
    u = _f64(-(absd * dt) ** 2)
    lo = _f64(w0 * w0 * dt / (c + absd))
    hi = _f64((c + absd) * dt)
    ec, es = sp.damped_cos_sinc(a, u, lo, hi)
    assert bool(mx.all(mx.isfinite(ec))) and bool(mx.all(mx.isfinite(es)))
    assert float(mx.max(mx.abs(ec))) <= 1.0 + 1e-12
    # the naive factorisation is what this replaces
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(-np.array(a, dtype=np.float64)) * np.cosh(absd * dt)
    assert not np.all(np.isfinite(naive))


@pytest.mark.parametrize("fn,x0", [
    (sp.sinc_gen, 0.3), (sp.sinc_gen, 3.0), (sp.sinc_gen, -5.0),
    (sp.cos_gen, 0.3), (sp.cos_gen, -5.0),
    (sp.sincm1, 0.3), (sp.sincm1, 3.0),
    (sp.gamma3, 0.5), (sp.gamma3, 3.0),
    (sp.exp_acc, 0.5), (sp.exp_acc, -3.0),
])
def test_vjps_match_central_differences(fn, x0):
    x = _f64([x0])
    _, g = mx.vjp(fn, [x], [mx.ones((1,), dtype=D)])
    ad = float(np.array(g[0])[0])
    h = 1e-6
    fd = (float(np.array(fn(_f64([x0 + h])))[0])
          - float(np.array(fn(_f64([x0 - h])))[0])) / (2 * h)
    assert abs(ad - fd) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_lower_gamma_reg_matches_mpmath(n):
    """The general-n case, against arbitrary precision.

    `gamma3` is the n = 2 case of this function; `terms.py` used to carry a
    second copy for n up to 4, with its own series constants and its own
    Horner loop.  One implementation, one oracle.
    """
    import mpmath as mpm

    z = np.concatenate([[0.0], np.logspace(-12, 2.5, 120)])
    got = np.array(sp.lower_gamma_reg(n, mx.array(z, dtype=mx.float64)),
                   dtype=np.float64)
    with mpm.workdps(50):
        want = np.array([float(mpm.gammainc(n + 1, 0, mpm.mpf(v),
                                            regularized=True)) for v in z])
    err = np.abs(got - want) / np.maximum(np.abs(want), 1e-300)
    assert np.max(err) < 1e-13, (n, float(np.max(err)))


def test_gamma3_is_the_n2_case():
    z = mx.array(np.concatenate([[0.0], np.logspace(-14, 3, 200)]),
                 dtype=mx.float64)
    assert float(mx.max(mx.abs(sp.gamma3(z) - sp.lower_gamma_reg(2, z)))) == 0.0


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_lower_gamma_is_finite_at_huge_arguments(n):
    """The direct branch used to overflow float32 into a NaN.

    `sum_j z^j/j!` overflows around z ~ 1e8 at n = 5 while `exp(-z)` has
    long since underflowed, giving inf * 0 where the answer is simply 1.
    Reachable only through an absurd lag, but the failure mode is a NaN
    likelihood, so it is worth one clamp.
    """
    zs = mx.array(np.array([1e2, 1e6, 1e8, 1e12, 1e20, 1e30]),
                  dtype=mx.float32)
    got = np.array(sp.lower_gamma_reg(n, zs), dtype=np.float64)
    assert np.all(np.isfinite(got)), got
    assert np.allclose(got, 1.0, atol=1e-6)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64])
def test_sincm1_series_is_bit_identical_on_its_stated_domain(dtype):
    """`sincm1_series` exists only to drop a branch that cannot be taken,
    so on that branch's domain it must agree to the *bit*, not to a
    tolerance.  A tolerance here would let a real numerical change hide.

    The domain is the closed interval `|u| <= _U_SERIES`, and the boundary
    is included on purpose: `_sho_matrices` calls it with `4u` where
    `|u| <= 0.5`, so `|argument|` reaches exactly `_U_SERIES`, and
    `sincm1`'s own branch test is `mx.abs(u) <= _U_SERIES`.
    """
    lo, hi = -sp._U_SERIES, sp._U_SERIES
    grid = np.concatenate([
        np.linspace(lo, hi, 20_001),
        np.array([lo, hi, 0.0, -0.0, np.nextafter(hi, 0.0),
                  np.nextafter(lo, 0.0)]),
    ])
    u = mx.array(grid.astype(np.float64), dtype=dtype)
    with mx.stream(mx.cpu):
        full = np.array(sp.sincm1(u))
        ser = np.array(sp.sincm1_series(u))
    view = np.uint32 if dtype == mx.float32 else np.uint64
    differing = int(np.sum(full.view(view) != ser.view(view)))
    assert differing == 0, (
        f"{differing} of {u.size} values differ in {dtype} -- "
        f"sincm1_series is only sound because it is bit-identical here")


def test_sincm1_series_eventually_diverges_so_the_restriction_is_real():
    """`sincm1` keeps its other branch for a reason, and this pins where.

    Worth knowing how much margin the `|u| <= _U_SERIES` restriction has:
    with 12 terms the series is bit-identical at u = 2, agrees to 1e-16 at
    u = 6 and to 6e-12 at u = 20, and only fails past |u| ~ 60 (1.3e-5),
    becoming useless at 200.  So `_U_SERIES = 2` is conservative by well
    over an order of magnitude, and `sincm1_series` at the call sites (where
    |u| <= 2) is not a marginal call.  It is still a restriction: at u = 200
    the series is wrong by 65x.
    """
    with mx.stream(mx.cpu):
        near = mx.array(np.array([6.0, 20.0, -6.0, -20.0]), dtype=D)
        assert np.max(np.abs(np.array(sp.sincm1_series(near))
                             - np.array(sp.sincm1(near)))) < 1e-10, (
            "the series should still be excellent just outside the threshold")
        far = mx.array(np.array([200.0, -200.0]), dtype=D)
        full = np.array(sp.sincm1(far))
        rel = np.max(np.abs(np.array(sp.sincm1_series(far)) - full)
                     / np.abs(full))
    assert rel > 1e-3, (
        f"the series still agrees to {rel:.1e} at |u| = 200; if _N_TRIG grew, "
        f"`sincm1`'s branch may no longer be needed at all")


def test_gamma3_shifted_beats_the_naive_form_where_it_matters(mpm):
    """`1 - e^{-z}(1 + y + y^2/2)` taken literally is catastrophic at small
    `z`; the identity used instead is `gamma3(z) + e^{-z} delta (1 + z -
    delta/2)`, exact and cancellation-free because `delta` is passed rather
    than formed as `z - y`."""
    from anvilgp import special as sp
    cases = [(1e-6, 0.0), (1e-6, 1e-7), (1e-4, 1e-5), (1e-3, 1e-3),
             (1e-2, 1e-3), (0.1, 0.05), (1.0, 0.5), (5.0, 2.0)]
    worst_new = worst_naive = 0.0
    for z, delta in cases:
        exact = 1 - mpm.e ** (-mpm.mpf(z)) * (
            1 + mpm.mpf(z - delta) + mpm.mpf(z - delta) ** 2 / 2)
        zz = mx.array([z], dtype=mx.float32)
        dd = mx.array([delta], dtype=mx.float32)
        with mx.stream(mx.cpu):
            new = float(np.array(sp.gamma3_shifted(zz, dd))[0])
            y = zz - dd
            naive = float(np.array(1.0 - sp.exp_acc(-zz)
                                   * (1.0 + y + 0.5 * y * y))[0])
        e = abs(float(exact))
        worst_new = max(worst_new, abs(new - float(exact)) / max(e, 1e-300))
        worst_naive = max(worst_naive,
                          abs(naive - float(exact)) / max(e, 1e-300))
    assert worst_new < 1e-6, f"the identity form is off by {worst_new:.1e}"
    assert worst_naive > 1e-3, (
        f"the naive form is only off by {worst_naive:.1e}; if float32 got "
        f"better this test has stopped demonstrating anything")


# ---------------------------------------------------------------------------
# the exposure-time convolution's two cancelling combinations (M5 stage 3c)
# ---------------------------------------------------------------------------

#: spans both sides of the |w|^2 <= 4 series/closed-form switch.
Z_CONV = np.concatenate([np.logspace(-9, -0.001, 30), [1.0, 1.9, 2.0, 2.1],
                         np.logspace(0.35, 1.6, 12)])


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
def test_coshm1_sinhm_over_z2_matches_mpmath(mpm, dtype):
    """``(cosh z - 1)/z^2`` and ``(sinh z - z)/z^2`` to working precision.

    These carry :class:`~anvilgp.terms.TermConvolution`'s *real* components.
    The tolerance is per-dtype eps, not a loose one: the whole point of the
    pair is that neither loses digits as ``z -> 0``, where both numerators
    cancel to nothing.
    """
    for z in Z_CONV:
        # Two regimes, two tolerances, because the worst case is *not* where
        # the danger is: in float32 the largest error over this range is
        # 1.7e-6 at z = 40, which is `exp`'s own relative error on
        # cosh(40) ~ 1e17 in the closed-form branch.  The small-z end --
        # what the helper exists for -- holds 1e-7.
        if dtype == mx.float64:
            tol = 4e-15
        else:
            tol = 3e-7 if z <= 2.0 else 3e-6
        got_c, got_s = sp.coshm1_sinhm_over_z2(mx.array([z], dtype=dtype))
        Z = mpm.mpf(float(z))
        ref_c = (mpm.cosh(Z) - 1) / Z ** 2
        ref_s = (mpm.sinh(Z) - Z) / Z ** 2
        assert abs(float(got_c[0]) / float(ref_c) - 1) < tol, f"cosh form, z={z}"
        assert abs(float(got_s[0]) / float(ref_s) - 1) < tol, f"sinh form, z={z}"


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
def test_coshm1_sinhm_complex_matches_mpmath(mpm, dtype):
    """``cosh w - 1`` and ``sinh w - w`` for ``w = x + i y``, both parts."""
    tol = 4e-15 if dtype == mx.float64 else 3e-7
    for x in (1e-9, 1e-5, 1e-2, 0.4, 1.4, 2.5, 4.0):
        for y in (0.0, 1e-9, 1e-5, 1e-2, 0.4, 1.4, 2.5, 7.0):
            (ct, st), (sr, si) = sp.coshm1_sinhm_complex(
                mx.array([x], dtype=dtype), mx.array([y], dtype=dtype))
            w = mpm.mpc(float(x), float(y))
            r1, r2 = mpm.cosh(w) - 1, mpm.sinh(w) - w
            scale = float(max(abs(r1), abs(r2)))
            got = np.array([float(ct[0]), float(st[0]), float(sr[0]), float(si[0])])
            ref = np.array([float(r1.real), float(r1.imag),
                            float(r2.real), float(r2.imag)])
            err = float(np.max(np.abs(got - ref)) / scale)
            assert err < tol, f"x={x} y={y}: {err:.2e}"


def test_the_naive_convolution_forms_are_the_ones_that_fail(mpm):
    """Why these helpers exist, pinned as a *comparison*.

    ``(cosh z - 1)/z^2`` written literally in float32 is worthless below
    ``z ~ 1e-3`` -- and that is the normal regime, a 30-minute exposure
    against a multi-day timescale.  Dividing by ``z^2`` afterwards cannot
    restore digits the subtraction already destroyed.  This is not a
    hypothetical: written that way, the convolved term's *value* still
    looked right at the reference point while its gradient was 9% wrong,
    and at ``delta = 1e-3`` the gradient was wrong by 247x.
    """
    z = 1e-4
    ref = float((mpm.cosh(mpm.mpf(z)) - 1) / mpm.mpf(z) ** 2)
    zz = mx.array([z], dtype=mx.float32)
    naive = float(np.array((mx.cosh(zz) - 1.0) / (zz * zz))[0])
    ours = float(np.array(sp.coshm1_sinhm_over_z2(zz)[0])[0])
    assert abs(naive / ref - 1) > 0.01, (
        "the naive form is supposed to be broken here; if it is not, this "
        "test has lost its point")
    assert abs(ours / ref - 1) < 3e-7
