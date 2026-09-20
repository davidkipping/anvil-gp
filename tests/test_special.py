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
