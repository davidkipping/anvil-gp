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

mx.set_default_device(mx.cpu)
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
    sp_special = pytest.importorskip("scipy.special")
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
