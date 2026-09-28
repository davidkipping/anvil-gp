"""Term algebra: does the state-space model reproduce the kernel it claims?

Two identities pin every term down without reference to any external code:

1. ``k(tau) = h Phi(tau) P_inf h^T = h Phi(tau) h^T`` (P_inf = I by
   construction) must equal the term's own covariance function.
2. ``Q(dt) = P_inf - Phi(dt) P_inf Phi(dt)^T = I - Phi Phi^T`` -- the
   Lyapunov consistency of a stationary process.  Note this is the *forbidden*
   expression: it is fine as a test at benign parameters, and catastrophic as
   an implementation, which the float32 test below demonstrates.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag

D = mx.float64

DT = np.concatenate([[0.0], np.logspace(-9, 2, 30)])

CASES = [
    ("real", ag.RealTerm(), [1.3, 0.7]),
    ("sho_lowQ", ag.SHOTerm(), [1.7, 2.0, 0.1]),
    ("sho_critical_minus", ag.SHOTerm(), [1.0, 1.0, 0.5 - 1e-9]),
    ("sho_critical", ag.SHOTerm(), [1.0, 1.0, 0.5]),
    ("sho_critical_plus", ag.SHOTerm(), [1.0, 1.0, 0.5 + 1e-9]),
    ("sho_granulation", ag.SHOTerm(), [1.0, 0.5, 2 ** -0.5]),
    ("sho_Q10", ag.SHOTerm(), [2.0, 3.0, 10.0]),
    ("sho_Q100", ag.SHOTerm(), [0.5, 7.0, 100.0]),
    ("matern32", ag.Matern32Term(), [1.1, 2.2]),
    ("matern52", ag.Matern52Term(), [1.1, 2.2]),
    ("rotation", ag.RotationTerm(), [1.5, 10.0, 3.0, 1.0, 0.4]),
    ("sum", ag.SHOTerm() + ag.RealTerm() + ag.Matern32Term(),
     [2.0, 3.0, 10.0, 1.3, 0.7, 1.1, 2.2]),
]


def _mats(term, p, dt=DT):
    P = mx.array(np.array([p], dtype=np.float64), dtype=D)
    phi, q, h = term.matrices(P, mx.array(dt, dtype=D))
    return np.array(phi)[0], np.array(q)[0], np.array(h)[0]


@pytest.mark.parametrize("name,term,p", CASES, ids=[c[0] for c in CASES])
def test_ssm_reproduces_the_covariance_function(name, term, p):
    phi, _, h = _mats(term, p)
    k_ssm = np.einsum("i,kij,j->k", h, phi, h)
    k_ref = term.value_np(np.array(p), DT)
    tol = 1e-11 if "critical_minus" in name or "critical_plus" in name else 1e-13
    # near critical damping the *reference* is the ill-conditioned one: the
    # textbook cosh/sinh form carries 1/d with d -> 0.  That asymmetry is the
    # whole argument for the state-space route.
    assert np.max(np.abs(k_ssm - k_ref)) / np.abs(k_ref).max() < tol


@pytest.mark.parametrize("name,term,p", CASES, ids=[c[0] for c in CASES])
def test_process_noise_satisfies_lyapunov(name, term, p):
    phi, q, _ = _mats(term, p)
    m = phi.shape[-1]
    lyap = np.eye(m)[None] - phi @ np.swapaxes(phi, -1, -2)
    assert np.max(np.abs(q - lyap)) < 1e-12


@pytest.mark.parametrize("name,term,p", CASES, ids=[c[0] for c in CASES])
def test_process_noise_is_psd(name, term, p):
    _, q, _ = _mats(term, p)
    for k in range(q.shape[0]):
        assert np.linalg.eigvalsh(q[k]).min() > -1e-13


@pytest.mark.parametrize("name,term,p", CASES, ids=[c[0] for c in CASES])
def test_duplicate_timestamps_give_identity_and_zero(name, term, p):
    """dt = 0 is legal input (celerite2 accepts it) and must be exact."""
    phi, q, _ = _mats(term, p, np.array([0.0]))
    m = phi.shape[-1]
    assert np.max(np.abs(phi[0] - np.eye(m))) < 1e-15
    assert np.max(np.abs(q[0])) < 1e-15


@pytest.mark.parametrize("name,term,p", CASES, ids=[c[0] for c in CASES])
def test_no_nan_in_any_regime_including_float32(name, term, p):
    """Long lags, dense cadence, both dtypes -- the corners a 4096-chain
    warmup actually visits."""
    dt = np.concatenate([[0.0], np.logspace(-10, 4, 60)])
    for dtype in (mx.float64, mx.float32):
        P = mx.array(np.array([p], dtype=np.float64), dtype=dtype)
        phi, q, h = term.matrices(P, mx.array(dt.astype(np.float32)
                                              if dtype == mx.float32 else dt,
                                              dtype=dtype))
        for arr in (phi, q, h):
            assert bool(mx.all(mx.isfinite(arr))), f"{name} {dtype}"


def test_matern32_is_the_sho_at_critical_damping():
    """Not a coincidence to be tested, a design claim to be enforced:
    Matern-3/2 IS the SHO at Q = 1/2, so it needs no separate propagator."""
    sigma, rho = 1.4, 3.3
    m32 = _mats(ag.Matern32Term(), [sigma, rho])
    rho_sho = 2.0 * math.pi * rho / math.sqrt(3.0)
    sho = _mats(ag.SHOTerm(), [sigma, rho_sho, 0.5])
    for a, b in zip(m32, sho):
        assert np.max(np.abs(a - b)) < 1e-14


def test_closed_form_process_noise_beats_the_forbidden_expression_in_fp32():
    """The measurement behind the 'never use P_inf - Phi P_inf Phi^T' rule.

    At omega_0 dt = 1e-3 the difference form is 225% wrong in float32; at
    1e-4 and below it returns exactly zero.  The closed form holds ~1e-7
    relative all the way down.
    """
    term = ag.SHOTerm()
    p = [1.0, 1.0, 3.0]
    for dt_val in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        ref = _mats(term, p, np.array([dt_val]))[1][0, 0, 0]
        P32 = mx.array(np.array([p]), dtype=mx.float32)
        T32 = mx.array(np.array([dt_val]), dtype=mx.float32)
        phi32, q32, _ = term.matrices(P32, T32)
        ours = float(np.array(q32)[0, 0, 0, 0])
        direct = float(np.array(
            mx.eye(2, dtype=mx.float32)
            - mx.matmul(phi32, mx.swapaxes(phi32, -1, -2)))[0, 0, 0, 0])
        err_ours = abs(ours - ref) / ref
        err_direct = abs(direct - ref) / ref
        assert err_ours < 1e-5, (dt_val, err_ours)
        # by dt = 1e-4 the difference form has no significant digits left;
        # below that it returns either 0 or the float32 granule 2^-24.
        assert err_direct > 1000.0 * max(err_ours, 1e-8), (dt_val, err_direct)


def test_matern_terms_match_the_bessel_form():
    """Independent check on ``value_np`` itself.

    Everything else validates the SSM *against* ``value_np``, and the dense
    oracle is built *from* it, so a bug there would be invisible to both.
    celerite2 cannot adjudicate either -- it has no exact Matern.  The
    general Matern family formula via the modified Bessel function K_nu is
    the outside opinion.
    """
    import scipy.special as sp_special    # declared test dep
    tau = np.logspace(-6, 1.5, 60)
    sigma, rho = 1.3, 2.7
    for nu, term in ((1.5, ag.Matern32Term()), (2.5, ag.Matern52Term())):
        x = math.sqrt(2.0 * nu) * tau / rho
        bessel = (sigma ** 2 * 2.0 ** (1.0 - nu) / math.gamma(nu)
                  * x ** nu * sp_special.kv(nu, x))
        got = term.value_np(np.array([sigma, rho]), tau)
        assert np.max(np.abs(got - bessel)) / sigma ** 2 < 1e-12, nu


# -- gradient finiteness ---------------------------------------------------
#
# A term can have exactly-correct values and a NaN gradient, and no value
# test can see it.  That is not hypothetical: every Matern-3/2 kernel had a
# NaN gradient w.r.t. rho until 2026-09-19, because Matern32Term fixes
# Q = 1/2, which makes `dsq` in `_sho_matrices` exactly zero, and the
# derivative of sqrt at 0 is infinite even when its cotangent is zero.
#
# So: differentiate every term at its nominal parameters and demand the
# gradient be finite.  Q = 1/2 exactly is included deliberately -- it is the
# measure-zero point a sampler hits only if it is unlucky or if a term puts
# it there by construction, which is exactly what Matern-3/2 does.

GRAD_CASES = CASES + [
    ("sho_critical_exact", ag.SHOTerm(), [1.0, 1.0, 0.5]),
    ("matern32_short", ag.Matern32Term(), [1.0, 0.3]),
    ("matern32_long", ag.Matern32Term(), [1.0, 5.0]),
    ("sum_with_matern32", ag.Matern32Term() + ag.SHOTerm(),
     [1.0, 1.0, 1.0, 2.0, 3.0]),
]


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64])
@pytest.mark.parametrize("name,term,p", GRAD_CASES,
                         ids=[c[0] for c in GRAD_CASES])
def test_term_matrices_have_finite_gradients(name, term, p, dtype):
    """d(Phi, Q, h)/d(params) must be finite at nominal parameters."""
    dt = mx.array(np.array([0.0, 0.01, 0.1, 1.0, 10.0]), dtype=dtype)
    P = mx.array(np.array([p], dtype=np.float64), dtype=dtype)

    def f(pp):
        phi, q, h = term.matrices(pp, dt)
        return mx.sum(phi) + mx.sum(q) + mx.sum(h)

    _, grad = mx.vjp(f, [P], [mx.array(1.0, dtype=dtype)])
    g = np.array(grad[0], dtype=np.float64)
    assert np.all(np.isfinite(g)), f"non-finite gradient for {name}: {g}"


def test_critical_damping_gradient_equals_its_one_sided_limits():
    """Q = 1/2 is not a special case, so its gradient must not be one.

    The value is smooth through critical damping by construction; the
    gradient must be too.  Approaching from either side pins it.
    """
    dt = mx.array(np.array([0.0, 0.05, 0.5, 2.0, 20.0]), dtype=D)

    def g(qual):
        P = mx.array(np.array([[1.0, 1.0, qual]]), dtype=D)

        def f(pp):
            phi, q, h = ag.SHOTerm().matrices(pp, dt)
            return mx.sum(phi) + mx.sum(q * q)

        _, grad = mx.vjp(f, [P], [mx.array(1.0, dtype=D)])
        return np.array(grad[0], dtype=np.float64).ravel()

    at = g(0.5)
    below, above = g(0.5 - 1e-7), g(0.5 + 1e-7)
    assert np.all(np.isfinite(at))
    # the one-sided limits agree with each other, and with the value at 1/2
    assert np.max(np.abs(below - above)) < 1e-5
    assert np.max(np.abs(at - 0.5 * (below + above))) < 1e-5


@pytest.mark.parametrize("qual", [0.05, 0.1, 0.2])
def test_overdamped_process_noise_is_accurate_in_float32(qual):
    """Q11 must not lose digits in the overdamped corner.

    `ee = e^{-2a}` used to be recovered from `ec^2 + u es^2`, which for
    u < 0 is cosh^2 - sinh^2.  The value of E is exponentially small there,
    but its *error* is ~eps phi^2 es^2 and gets multiplied by 2a^2, so
    float32 lost up to 3e-4 relative at Q = 0.05.  The default log_Q prior
    reaches Q = 0.05, so this was inside the sampled region.
    """
    term = ag.SHOTerm()
    omega0 = 1.0
    for w0dt in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        dt = np.array([w0dt / omega0])
        ref = _mats(term, [1.0, omega0, qual], dt)[1][0, 0, 0]
        P32 = mx.array(np.array([[1.0, omega0, qual]]), dtype=mx.float32)
        T32 = mx.array(dt, dtype=mx.float32)
        got = float(np.array(term.matrices(P32, T32)[1])[0, 0, 0, 0])
        assert abs(got - ref) / abs(ref) < 1e-6, (qual, w0dt, got, ref)


# ---------------------------------------------------------------------------
# M5 stage 3: the general complex term
# ---------------------------------------------------------------------------

def _complex_cases():
    """(a, b, c, d, name), spanning the corners the parametrisation can hit."""
    return [
        (1.69, 0.2857, 0.2618, 1.5488, "SHO-equivalent"),
        (2.0, 0.0, 0.5, 1.0, "b = 0 (pure cosine)"),
        (1.0, 0.3, 0.4, 1e-4, "d -> 0 (real limit)"),
        (1.0, 0.05, 2.0, 0.1, "c >> d (overdamped-ish)"),
        (1.0, -0.4, 0.5, 1.0, "b < 0"),
        (1.0, 0.4999, 0.5, 1.0, "just inside |b/a| <= c/d"),
        (3.0, 0.02, 0.05, 3.0, "c << d (barely damped)"),
    ]


@pytest.mark.parametrize("a,b,c,d,name", _complex_cases(),
                         ids=[k[4] for k in _complex_cases()])
def test_complex_term_matches_celerite2(a, b, c, d, name):
    """The general celerite term, against the library that defines it."""
    c2t = pytest.importorskip("celerite2.terms")
    tau = np.concatenate([[0.0], np.logspace(-6, 2, 80)])
    mine = ag.ComplexTerm().value_np(np.array([a, b, c, d]), tau)
    ref = np.asarray(c2t.ComplexTerm(a=a, b=b, c=c, d=d).get_value(tau))
    rel = np.max(np.abs(mine - ref)) / max(np.max(np.abs(ref)), 1e-300)
    assert rel < 1e-13, f"{name}: {rel:.2e}"


@pytest.mark.parametrize("a,b,c,d,name", _complex_cases(),
                         ids=[k[4] for k in _complex_cases()])
def test_the_complex_state_space_form_reproduces_its_own_kernel(a, b, c, d,
                                                                name):
    """`k(tau) = h Phi(tau) P_inf h^T` with `P_inf = I`.

    This is stage 3a's central claim, and the review proved the alternative
    wrong: a general complex term does **not** need a general stationary
    covariance.  One extra free rate in the SHO basis reaches it --
    `beta = b d / a`, `s = sqrt(beta^2 + d^2)`, `g = c - beta` -- and the
    identity `P_inf = I` is kept.
    """
    tau = np.array([0.0, 1e-5, 0.01, 0.3, 1.0, 5.0, 30.0])
    p = mx.array([[a, b, c, d]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        phi, q, h = ag.ComplexTerm().matrices(p, mx.array(tau, dtype=mx.float64))
        mx.eval(phi, q, h)
    hv = np.array(h)[0]
    k = np.array([hv @ np.array(phi)[0, i] @ hv for i in range(len(tau))])
    ref = ag.ComplexTerm().value_np(np.array([a, b, c, d]), tau)
    rel = np.max(np.abs(k - ref)) / max(np.max(np.abs(ref)), 1e-300)
    assert rel < 1e-13, f"{name}: {rel:.2e}"
    # and Q = I - Phi Phi^T, the stationarity that P_inf = I asserts
    for i in range(len(tau)):
        ph = np.array(phi)[0, i]
        want = np.eye(2) - ph @ ph.T
        got = np.array(q)[0, i]
        assert np.max(np.abs(got - want)) < 1e-12, f"{name} lag {tau[i]}"


def test_the_sho_is_the_complex_term_with_the_celerite_amplitude_ratio():
    """`SHOTerm` has `b/a = c/d` exactly, so it is `ComplexTerm` at `g = 0`.

    Pinned because it is what makes stage 3a's refactor safe: the SHO now
    *runs through* the general path, and if the two ever disagreed the
    general form would be wrong.
    """
    c2t = pytest.importorskip("celerite2.terms")
    tau = np.concatenate([[0.0], np.logspace(-5, 2, 60)])
    for sigma, rho, qual in ((1.3, 4.0, 3.0), (1.1, 2.0, 10.0),
                             (0.7, 6.0, 0.8), (2.0, 1.0, 100.0)):
        _, _, aa, bb, cc, dd = c2t.SHOTerm(
            sigma=sigma, rho=rho, Q=qual).get_coefficients()
        a, b, c, d = float(aa[0]), float(bb[0]), float(cc[0]), float(dd[0])
        assert abs(b / a - c / d) < 1e-12, "the SHO's ratio is not c/d"
        sho = ag.SHOTerm().value_np(np.array([sigma, rho, qual]), tau)
        cpx = ag.ComplexTerm().value_np(np.array([a, b, c, d]), tau)
        rel = np.max(np.abs(sho - cpx)) / np.max(np.abs(sho))
        assert rel < 1e-13, f"sigma={sigma} rho={rho} Q={qual}: {rel:.2e}"


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64])
def test_gamma3_shifted_is_gamma3_when_the_rates_agree(dtype):
    """The SHO is `delta = 0`, and it must be **bit**-identical there --
    anything else would perturb every existing term."""
    from anvilgp import special as sp
    z = mx.array(np.concatenate([[0.0], np.logspace(-7, 1.5, 1000)]),
                 dtype=dtype)
    with mx.stream(mx.cpu):
        a = np.array(sp.gamma3(z))
        b = np.array(sp.gamma3_shifted(z, mx.zeros_like(z)))
    view = np.uint32 if dtype == mx.float32 else np.uint64
    assert int(np.sum(a.view(view) != b.view(view))) == 0


# ---------------------------------------------------------------------------
# exposure-time integration (M5 stage 3c)
# ---------------------------------------------------------------------------

#: terms with exact celerite coefficients, so wrappable.  SHO is underdamped
#: only -- see `test_convolution_needs_an_underdamped_sho`.
CONVOLVABLE = [
    ("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
    ("real", ag.RealTerm(), [1.2, 0.7]),
    ("complex", ag.ComplexTerm(), [1.5, 0.4, 0.6, 1.1]),
    ("sho+real", ag.SHOTerm() + ag.RealTerm(), [1.3, 4.0, 3.0, 1.2, 0.7]),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4]),
]


def _f64p(p):
    return mx.array(np.asarray(p, dtype=np.float64)[None, :], dtype=mx.float64)


@pytest.mark.parametrize("name,term,p", CONVOLVABLE, ids=[c[0] for c in CONVOLVABLE])
@pytest.mark.parametrize("delta", [1e-4, 1e-2, 0.2])
def test_convolved_kernel_at_zero_lag_matches_a_quadrature(name, term, p, delta):
    """``k_conv(0) = (2/d^2) int_0^d (d - u) k(u) du``, by mpmath quadrature.

    The reference the plan reserves for the small-``delta`` end, where
    celerite2's own two float64 routes disagree by 5.7e-12 and a parity gate
    would mostly be testing celerite2.  A quadrature of the *base* kernel
    assumes nothing about the closed form being tested.
    """
    mpm = pytest.importorskip("mpmath")
    mpm.mp.dps = 40
    cv = ag.TermConvolution(term, delta)
    got = float(np.asarray(cv.value_np(np.asarray(p, dtype=np.float64),
                                       np.array([0.0])))[0])
    d = mpm.mpf(delta)
    ref = (2 / d ** 2) * mpm.quad(
        lambda u: (d - u) * mpm.mpf(float(term.value_np(
            np.asarray(p, dtype=np.float64), np.array([float(u)]))[0])),
        [0, d])
    assert abs(got / float(ref) - 1) < 1e-10, f"{name} delta={delta}"


@pytest.mark.parametrize("name,term,p", CONVOLVABLE, ids=[c[0] for c in CONVOLVABLE])
def test_convolution_recovers_the_base_term_as_delta_goes_to_zero(name, term, p):
    """``delta -> 0`` is the identity, and the limit must be *reached*.

    The cancellation-free forms exist for this: the amplitudes go as
    ``(cosh z - 1)/z^2 -> 1/2`` and the correction as ``-z/6 -> 0``, both
    finite limits the literal difference-of-ones form cannot reach -- it
    returns 0/0 noise, and at ``delta = 1e-8`` it returned NaN.

    Compared at lags ``>= delta`` only, which is not a dodge but the
    supported domain: :class:`~anvilgp.likelihood.GPLogLike` refuses any
    cadence with a spacing below the exposure width, so no shorter lag is
    ever evaluated.  Zero lag is genuinely *not* the base term -- it differs
    at first order in ``delta`` -- and
    `test_the_correction_is_delta_over_three_times_the_kernel_slope` pins
    that separately.
    """
    tau = np.logspace(-3, 1.5, 40)
    base = np.asarray(term.value_np(np.asarray(p, dtype=np.float64), tau))
    for delta in (1e-8, 1e-6):
        cv = ag.TermConvolution(term, delta)
        got = np.asarray(cv.value_np(np.asarray(p, dtype=np.float64), tau))
        assert np.all(np.isfinite(got)), f"{name} delta={delta}: non-finite"
        rel = np.max(np.abs(got - base)) / np.max(np.abs(base))
        assert rel < 1e-12, f"{name} delta={delta}: {rel:.2e}"


@pytest.mark.parametrize("name,term,p", CONVOLVABLE, ids=[c[0] for c in CONVOLVABLE])
def test_the_correction_is_delta_over_three_times_the_kernel_slope(name, term, p):
    """``corr = (delta/3) k'(0+)`` at leading order -- an independent check.

    The zero-lag correction is the one piece of the convolution that is not
    a kernel, so it gets a reference that does not come from the same closed
    form: the kink.  Integrating a boxcar over a kernel whose even extension
    has a corner at the origin misses ``(delta/3) k'(0+)``, and this holds to
    six digits for every term with a corner.

    It also explains the **sign**, which the class documents as
    indefinite.  ``k'(0+) = -1/2 h Q_c h^T <= 0`` for any stationary model
    with PSD process noise, so wherever the corner exists the correction is
    *negative* -- it removes variance.  A term with ``b/a = c/d`` (the SHO,
    and so `RotationTerm`) has no corner: ``k'(0+) = 0``, the first-order
    term vanishes, and what is left is positive and ``O(delta^3)``.  Both
    signs occur, but at different orders -- not, as the class docstring once
    implied, as two cases of one leading term.
    """
    pp = np.asarray(p, dtype=np.float64)
    h = 1e-7
    v = np.asarray(term.value_np(pp, np.array([0.0, h, 2 * h])))
    slope = float((-3 * v[0] + 4 * v[1] - v[2]) / (2 * h))
    smooth = abs(slope) < 1e-6 * float(v[0])
    for delta in (1e-6, 1e-4):
        with mx.stream(mx.cpu):
            corr = float(np.array(ag.TermConvolution(term, delta)
                                  .diag_correction(_f64p(p)))[0, 0])
        if smooth:
            # no corner: the O(delta) term is absent, and the remainder is
            # positive and cubic -- so it is far *below* the corner law.
            assert corr > 0.0, f"{name} delta={delta}: {corr:.3e}"
            assert corr < 1e-3 * delta * float(v[0]), f"{name} delta={delta}"
        else:
            assert abs(corr / (delta / 3.0 * slope) - 1) < 1e-4, \
                f"{name} delta={delta}: corr {corr:.6e}"
            assert corr < 0.0, f"{name}: a corner removes variance"


@pytest.mark.parametrize("name,term,p", [
    ("matern32", ag.Matern32Term(), [1.1, 2.5]),
    ("matern52", ag.Matern52Term(), [1.1, 2.5]),
])
def test_convolution_supports_the_matern_terms(name, term, p):
    """Matern-3/2 and Matern-5/2 *can* be convolved -- the inverted test.

    Both were refused while the convolution went through celerite
    coefficients, which a repeated root (Matern-3/2) and a triple one
    (Matern-5/2) do not have.  The state-space route is a matrix function of
    the generator, ``2(cosh(F delta) - I)/(F delta)^2``, which is an entire
    function and so is indifferent to root multiplicity.  Nothing is
    decomposed, so there is nothing to decompose badly.

    Checked against a quadrature of the definition, not against celerite2 --
    celerite2 has no Matern-5/2 at all, and only an *approximation* of
    Matern-3/2, so this is a capability it does not have.
    """
    mpm = pytest.importorskip("mpmath")
    mpm.mp.dps = 30
    for delta in (1e-3, 1e-2, 0.2):
        cv = ag.TermConvolution(term, delta)
        assert cv.n_state == term.n_state, "the convolution must not raise m"
        taus = np.array([0.0, delta, 2.0 * delta, 0.7, 2.0])
        got = np.asarray(cv.value_np(np.asarray(p, dtype=np.float64), taus))
        d = mpm.mpf(delta)
        for tau, g in zip(taus, got):
            ref = mpm.quad(
                lambda u: (d - abs(u)) * mpm.mpf(float(np.asarray(
                    term.value_np(np.asarray(p, dtype=np.float64),
                                  np.array([abs(float(tau + u))])))[0])),
                [-d, -min(tau, delta), 0, d]) / d ** 2
            assert abs(g / float(ref) - 1) < 1e-10, \
                f"{name} delta={delta} tau={tau}"


@pytest.mark.parametrize("qual", [0.1, 0.5, 0.7])
def test_convolution_supports_an_overdamped_sho(qual):
    """``Q <= 1/2`` is convolvable now -- the other inverted test.

    Below critical damping the SHO is two real exponentials, one with a
    *negative* amplitude, which is why the coefficient route refused it. The
    generator does not care: ``F = [[0, w0], [-w0, -2c]]`` is regular for
    every ``Q``, nothing takes a root of ``d^2 < 0``, and no amplitude is
    ever formed.
    """
    mpm = pytest.importorskip("mpmath")
    mpm.mp.dps = 30
    p = np.array([1.3, 4.0, qual])
    term = ag.SHOTerm()
    for delta in (1e-2, 0.2):
        cv = ag.TermConvolution(term, delta)
        taus = np.array([0.0, delta, 3.0 * delta, 1.5])
        got = np.asarray(cv.value_np(p, taus))
        assert np.all(np.isfinite(got)), f"Q={qual} delta={delta}"
        d = mpm.mpf(delta)
        for tau, g in zip(taus, got):
            ref = mpm.quad(
                lambda u: (d - abs(u)) * mpm.mpf(float(np.asarray(
                    term.value_np(p, np.array([abs(float(tau + u))])))[0])),
                [-d, -min(tau, delta), 0, d]) / d ** 2
            assert abs(g / float(ref) - 1) < 1e-10, \
                f"Q={qual} delta={delta} tau={tau}"


TERMS_WITH_GENERATOR = [
    ("real", ag.RealTerm(), [1.2, 0.7]),
    ("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
    ("sho-critical", ag.SHOTerm(), [1.3, 4.0, 0.5]),
    ("sho-overdamped", ag.SHOTerm(), [1.3, 4.0, 0.1]),
    ("sho-high-q", ag.SHOTerm(), [1.3, 4.0, 100.0]),
    ("complex", ag.ComplexTerm(), [1.5, 0.4, 0.6, 1.1]),
    ("matern32", ag.Matern32Term(), [1.1, 2.5]),
    ("matern52", ag.Matern52Term(), [1.1, 2.5]),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4]),
    ("sho+real", ag.SHOTerm() + ag.RealTerm(), [1.3, 4.0, 3.0, 1.2, 0.7]),
    ("m52+m32", ag.Matern52Term() + ag.Matern32Term(), [1.1, 2.5, 1.0, 3.0]),
]


@pytest.mark.parametrize("name,term,p", TERMS_WITH_GENERATOR,
                         ids=[c[0] for c in TERMS_WITH_GENERATOR])
def test_the_generator_exponentiates_to_the_propagator(name, term, p):
    """``expm(F dt) == Phi(dt)``, block by block.

    The invariant the whole state-space convolution rests on: ``F`` has to be
    in the *same* non-dimensionalised basis the term's own ``Phi`` uses, or
    the matrix functions built from it describe a different process. Checked
    against ``scipy.linalg.logm`` of the propagator the term actually
    returns, which is independent of how ``generator`` computes it.
    """
    logm = pytest.importorskip("scipy.linalg").logm
    pp = _f64p(p)
    for dt in (0.013, 0.21):
        with mx.stream(mx.cpu):
            phis, _, _ = term.matrices_blocks(
                pp, mx.array([dt], dtype=mx.float64))
            gens = term.generator(pp)
        assert len(gens) == len(phis), f"{name}: generator/block count differ"
        for ph, gen in zip(phis, gens):
            got = np.asarray(gen, dtype=np.float64)[0]
            ref = np.real(logm(np.asarray(ph, dtype=np.float64)[0, 0]) / dt)
            scale = max(float(np.max(np.abs(got))), 1e-300)
            err = float(np.max(np.abs(got - ref))) / scale
            assert err < 1e-13, f"{name} dt={dt}: {err:.2e}"


@pytest.mark.parametrize("name,term,p", TERMS_WITH_GENERATOR,
                         ids=[c[0] for c in TERMS_WITH_GENERATOR])
def test_the_convolved_stationary_covariance_reproduces_the_kernel(name, term, p):
    """``h Phi(tau) P_conv h^T == k_conv(tau)``, and ``P_conv`` is a covariance.

    The state-space claim in one test: ``P_conv`` is symmetric, PSD, and its
    only job is to satisfy ``P h^T = M h^T``.  ``M`` itself is *not*
    symmetric -- asserted here too, because if it ever became symmetric the
    rank-one construction would be unnecessary and this design would be
    over-built.
    """
    from anvilgp.terms import _block_diag, _conv_matrix_factors
    delta = 0.02
    cv = ag.TermConvolution(term, delta)
    pp = _f64p(p)
    taus = np.array([delta, 2.5 * delta, 0.4, 1.7])
    with mx.stream(mx.cpu):
        phi, _, h = cv.matrices(pp, mx.array(taus, dtype=mx.float64))
        blocks = cv.p_inf_blocks(pp)
        pin = _block_diag(*blocks) if len(blocks) > 1 else blocks[0]
        pin = np.asarray(pin, dtype=np.float64)[0]
        ph = np.asarray(phi, dtype=np.float64)[0]
        hv = np.asarray(h, dtype=np.float64)[0]
        asym = max(
            float(np.max(np.abs(np.asarray(_conv_matrix_factors(f, delta)[0])[0]
                                - np.asarray(_conv_matrix_factors(f, delta)[0])[0].T)))
            for f in term.generator(pp))
    assert np.max(np.abs(pin - pin.T)) < 1e-13, f"{name}: P_conv not symmetric"
    assert np.linalg.eigvalsh(pin).min() > -1e-12, f"{name}: P_conv not PSD"
    if term.n_state > 1:
        assert asym > 1e-9, (
            f"{name}: M came out symmetric, so the rank-one construction is "
            f"unnecessary -- revisit `_conv_p_inf`")
    ref = np.asarray(cv.value_np(np.asarray(p, dtype=np.float64), taus))
    got = np.array([float(hv @ ph[i] @ pin @ hv) for i in range(len(taus))])
    rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    assert rel < 1e-12, f"{name}: {rel:.2e}"


def test_the_sho_coefficients_are_non_finite_below_critical_damping():
    """At ``Q <= 1/2`` the SHO is two real exponentials, one *negative*.

    Still true of the *coefficients*, and pinned because it is the reason the
    convolution does not use them: `SHOTerm.celerite_coefficients` floors
    ``d`` at zero, so ``b = a c / d`` is non-finite below critical damping.
    Nothing on the evaluation path calls it any more --
    `test_convolution_supports_an_overdamped_sho` convolves exactly these
    parameters successfully -- so this documents a dead end, not a limit.
    """
    with mx.stream(mx.cpu):
        cv = ag.TermConvolution(ag.SHOTerm(), 0.01)
        ok = cv.celerite_coefficients(_f64p([1.3, 4.0, 3.0]))
        assert np.all(np.isfinite(np.asarray(ok[3])))
        for qual in (0.5, 0.4, 0.1):
            bad = cv.celerite_coefficients(_f64p([1.3, 4.0, qual]))
            assert not np.all(np.isfinite(np.asarray(bad[3]))), f"Q={qual}"
    # RotationTerm cannot reach it: both modes have Q >= 1/2 + Q0 > 1/2.
    with mx.stream(mx.cpu):
        rot = ag.TermConvolution(ag.RotationTerm(), 0.01)
        for q0, dq in ((1e-6, 1e-6), (3.0, 1.0), (0.01, 0.01)):
            co = rot.celerite_coefficients(_f64p([1.5, 7.0, q0, dq, 0.4]))
            assert all(np.all(np.isfinite(np.asarray(x))) for x in co), \
                f"Q0={q0} dQ={dq}"


@pytest.mark.parametrize("name,term,p", CONVOLVABLE, ids=[c[0] for c in CONVOLVABLE])
@pytest.mark.parametrize("delta", [1e-2, 0.2])
def test_convolved_coefficients_match_celerite2(name, term, p, delta):
    """Rescaled amplitudes and the zero-lag correction, against celerite2.

    celerite2 exposes the convolved coefficients directly, so this pins the
    closed form independently of the filter.

    **Not below ``delta = 1e-2``**, and that is not a tolerance dodge: down
    there celerite2 is the inaccurate one.  See
    `test_we_beat_celerite2_on_the_convolved_amplitudes_at_small_delta`;
    the small-``delta`` end is gated against an mpmath quadrature instead.
    """
    c2t = pytest.importorskip("celerite2.terms")
    base = _celerite2_equivalent(c2t, name, np.asarray(p, dtype=np.float64))
    if base is None:
        pytest.skip(f"no celerite2 equivalent for {name}")
    conv = c2t.TermConvolution(base, delta)
    ar_r, cr_r, a_r, b_r, c_r, d_r = [np.asarray(x, dtype=np.float64)
                                      for x in conv.get_coefficients()]
    with mx.stream(mx.cpu):
        ar, cr, a, b, c, d = [np.asarray(x)[0] for x in
                              ag.TermConvolution(term, delta)
                              .celerite_coefficients(_f64p(p))]
    for got, ref, lab in ((ar, ar_r, "ar"), (cr, cr_r, "cr"), (a, a_r, "a"),
                          (b, b_r, "b"), (c, c_r, "c"), (d, d_r, "d")):
        assert got.size == ref.size, f"{name} {lab}: {got.size} vs {ref.size}"
        if got.size:
            scale = max(float(np.max(np.abs(ref))), 1e-300)
            err = float(np.max(np.abs(np.sort(got) - np.sort(ref)))) / scale
            assert err < 1e-11, f"{name} delta={delta} {lab}: {err:.2e}"


def _celerite2_equivalent(c2t, name, p):
    """The same kernel in celerite2's parametrisation, or None."""
    if name == "sho":
        return c2t.SHOTerm(sigma=p[0], rho=p[1], Q=p[2])
    if name == "real":
        return c2t.RealTerm(a=p[0] ** 2, c=p[1])
    if name == "complex":
        return c2t.ComplexTerm(a=p[0], b=p[1], c=p[2], d=p[3])
    if name == "sho+real":
        return (c2t.SHOTerm(sigma=p[0], rho=p[1], Q=p[2])
                + c2t.RealTerm(a=p[3] ** 2, c=p[4]))
    if name == "rotation":
        return c2t.RotationTerm(sigma=p[0], period=p[1], Q0=p[2], dQ=p[3],
                                f=p[4])
    return None


def test_we_beat_celerite2_on_the_convolved_amplitudes_at_small_delta():
    """At ``delta = 1e-3`` anvil-gp is exact and celerite2 is 2e-10 wrong.

    The convolved amplitude of a real term is ``2 a (cosh z - 1)/z^2`` with
    ``z = c delta``, and celerite2 evaluates that numerator as written.  In
    float64 at ``z = 7e-4`` the subtraction costs ~6 digits.  anvil-gp uses
    the series (`coshm1_sinhm_over_z2`), so it stays at one ulp.

    This is why `test_convolved_coefficients_match_celerite2` stops at
    ``delta = 1e-2``: below it, a parity gate measures celerite2's error.
    Recorded as a *directional* assertion -- ours closer to the truth than
    theirs by a wide margin -- rather than a fixed tolerance.
    """
    mpm = pytest.importorskip("mpmath")
    c2t = pytest.importorskip("celerite2.terms")
    mpm.mp.dps = 50
    sigma, c, delta = 1.2, 0.7, 1e-3
    amp = sigma ** 2
    z = mpm.mpf(c) * mpm.mpf(delta)
    exact = float(2 * mpm.mpf(amp) * (mpm.cosh(z) - 1) / z ** 2)
    with mx.stream(mx.cpu):
        ours = float(np.asarray(
            ag.TermConvolution(ag.RealTerm(), delta)
            .celerite_coefficients(_f64p([sigma, c]))[0])[0, 0])
    theirs = float(np.asarray(c2t.TermConvolution(
        c2t.RealTerm(a=amp, c=c), delta).get_coefficients()[0])[0])
    err_ours = abs(ours / exact - 1)
    err_theirs = abs(theirs / exact - 1)
    assert err_ours < 1e-15, f"ours {err_ours:.2e}"
    assert err_theirs > 100 * err_ours, (
        f"celerite2 {err_theirs:.2e} vs ours {err_ours:.2e} -- if celerite2 "
        f"has been fixed upstream, this test has served its purpose")


# ---------------------------------------------------------------------------
# celerite coefficient interop (what `OriginalCeleriteTerm` actually is)
# ---------------------------------------------------------------------------

C2_ROUNDTRIP = [
    ("sho", dict(sigma=1.3, rho=4.0, Q=3.0), "SHOTerm", 2, 4),
    ("sho-high-q", dict(sigma=0.7, rho=11.0, Q=80.0), "SHOTerm", 2, 4),
    ("real", dict(a=1.44, c=0.7), "RealTerm", 1, 2),
    ("complex", dict(a=1.5, b=0.4, c=0.6, d=1.1), "ComplexTerm", 2, 4),
    ("rotation", dict(sigma=1.5, period=7.0, Q0=3.0, dQ=1.0, f=0.4),
     "RotationTerm", 4, 8),
]


@pytest.mark.parametrize("name,kw,cls,m,npar", C2_ROUNDTRIP,
                         ids=[c[0] for c in C2_ROUNDTRIP])
def test_from_celerite_coefficients_round_trips_celerite2(name, kw, cls, m, npar):
    """celerite's coefficients in, the same kernel out.

    This is the interop route that replaces the ``OriginalCeleriteTerm`` the
    docs used to promise.  celerite2's class of that name is **not** a
    reparametrisation of `ComplexTerm`: reading its source, it wraps a term
    from the *legacy* ``celerite`` package and passes
    ``get_all_coefficients()`` through unchanged.  So the coefficients are
    already in this convention, there is no factor to convert, and anything
    either celerite version can express goes through this function.

    Exact agreement is expected, not approximate: both sides evaluate the same
    closed form from the same numbers.
    """
    c2t = pytest.importorskip("celerite2.terms")
    c2term = getattr(c2t, cls)(**kw)
    term, p = ag.from_celerite_coefficients(*c2term.get_coefficients())
    assert term.n_state == m, f"{name}: m = {term.n_state}, expected {m}"
    assert p.size == npar, f"{name}: {p.size} parameters, expected {npar}"
    tau = np.concatenate([[0.0], np.logspace(-3, 1.5, 60)])
    ours = np.asarray(term.value_np(p, tau))
    ref = np.asarray(c2term.get_value(tau))
    rel = float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref)))
    assert rel < 1e-14, f"{name}: {rel:.2e}"


def test_from_celerite_coefficients_handles_a_sum():
    """A mixed real+complex set becomes a `TermSum` with the blocks in order."""
    c2t = pytest.importorskip("celerite2.terms")
    c2term = (c2t.SHOTerm(sigma=1.3, rho=4.0, Q=3.0)
              + c2t.RealTerm(a=1.44, c=0.7))
    term, p = ag.from_celerite_coefficients(*c2term.get_coefficients())
    assert term.n_state == 3
    # real blocks first, then the complex pairs -- the order this builds them
    assert term.block_sizes == (1, 2), term.block_sizes
    tau = np.concatenate([[0.0], np.logspace(-3, 1.5, 60)])
    rel = float(np.max(np.abs(np.asarray(term.value_np(p, tau))
                              - np.asarray(c2term.get_value(tau))))
                / np.max(np.abs(np.asarray(c2term.get_value(tau)))))
    assert rel < 1e-14, rel


def test_from_celerite_coefficients_refuses_an_overdamped_sho():
    """The negative real amplitude has no ``P_inf = I`` representation.

    An SHO below critical damping is two real exponentials whose amplitudes
    have opposite signs.  That is the *coefficient* route's limitation, not
    the package's -- `SHOTerm` carries the same kernel as one 2-state block at
    any ``Q``, and the message says so.
    """
    c2t = pytest.importorskip("celerite2.terms")
    co = c2t.SHOTerm(sigma=1.3, rho=4.0, Q=0.4).get_coefficients()
    with pytest.raises(ValueError, match="use SHOTerm directly"):
        ag.from_celerite_coefficients(*co)
    # and the kernel really is available the other way
    native = ag.SHOTerm().value_np(np.array([1.3, 4.0, 0.4]),
                                   np.array([0.0, 0.5, 2.0]))
    ref = c2t.SHOTerm(sigma=1.3, rho=4.0, Q=0.4).get_value(
        np.array([0.0, 0.5, 2.0]))
    assert np.max(np.abs(np.asarray(native) - np.asarray(ref))) < 1e-12


@pytest.mark.parametrize("kw,match", [
    (dict(a=[1.0], b=[9.0], c=[0.5], d=[1.0]), "not a valid covariance"),
    (dict(a=[1.0], b=[0.1], c=[0.5], d=[0.0]), "repeated root"),
    (dict(a=[-1.0], b=[0.0], c=[0.5], d=[1.0]), r"a\[0\] is -1"),
    (dict(a_real=[1.0], c_real=[1.0, 2.0]), "differ in length"),
    (dict(a=[1.0], b=[0.1], c=[0.5]), "differ in length"),
    (dict(), "no coefficients"),
])
def test_from_celerite_coefficients_refuses_invalid_sets(kw, match):
    """Each refusal names a condition the rest of the package documents.

    ``b/a <= c/d`` is ``k'(0+) <= 0``, the PSD condition on the implied
    process noise; ``d = 0`` is a repeated root and so a Matern-3/2 rather
    than a complex pair.
    """
    with pytest.raises(ValueError, match=match):
        ag.from_celerite_coefficients(**kw)


def test_from_celerite_coefficients_accepts_the_sho_boundary():
    """``b/a == c/d`` exactly is the SHO, and must not be refused.

    The check carries a relative tolerance for this: the SHO sits *on* the
    boundary, and a round trip through celerite2's float64 coefficients lands
    a few ulp either side of it.
    """
    c2t = pytest.importorskip("celerite2.terms")
    for qual in (0.51, 1.0, 3.0, 300.0):
        co = c2t.SHOTerm(sigma=1.1, rho=5.0, Q=qual).get_coefficients()
        _, _, a, b, c, d = [np.atleast_1d(np.asarray(x)) for x in co]
        assert abs(b[0] / a[0] - c[0] / d[0]) < 1e-9, "not the SHO boundary"
        ag.from_celerite_coefficients(*co)          # must not raise
