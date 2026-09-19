"""Kernel terms as linear Gaussian state-space models.

Every celerite-class kernel is exactly a linear Gaussian SSM (the CARMA
correspondence), with a closed-form stationary covariance for
initialisation.  A term supplies four things:

    h        observation row vector          (m,)      static
    P_inf    stationary state covariance     (m, m)
    Phi(dt)  transition (matrix exponential) (m, m)
    Q(dt)    process noise                   (m, m)

and the Kalman prediction-error decomposition over those reproduces the
celerite semiseparable Cholesky's marginal likelihood exactly -- same
log-determinant, same quadratic form.  The SSM route is chosen for two
concrete reasons, not for taste:

1. **It is smooth through Q = 1/2.**  celerite's coefficient
   parametrisation carries a ``1/sqrt(4 Q^2 - 1)`` that blows up at
   critical damping.  A 4096-chain warmup crosses Q = 1/2 routinely, and a
   float32 sampler cannot survive a pole in its own likelihood.  Here the
   propagator is built from ``cos(d dt)`` and ``sin(d dt)/d``, which are
   entire functions of ``u = d^2 dt^2`` and continue analytically from
   underdamped (u > 0) through critical (u = 0) into overdamped (u < 0).
2. **Matern-3/2 and Matern-5/2 come free**, and neither is
   celerite-representable.  Matern-3/2 is not even a special case that
   needs code: it *is* the SHO at Q = 1/2 exactly, with
   ``omega_0 = sqrt(3)/rho``.

Non-dimensionalised state
-------------------------
Terms hand back a state that has been rescaled so that ``P_inf = I`` and
the amplitude lives in ``h`` instead (for one SHO term, ``h = [sigma, 0]``).
Without this the SHO's natural state ``(x, xdot)`` has
``P_inf = sigma^2 diag(1, omega_0^2)``, an ``omega_0^2`` scale disparity
that would put a Cholesky pivot floor of one component at the rounding
scale of the *other*.  With it, every covariance in the recursion is O(1)
and dimensionless, and the whole conditioning story reduces to the one
axis that actually matters (dt versus the damping and oscillation times).

Cancellation
------------
``Q = P_inf - Phi P_inf Phi^T`` is forbidden: at ``omega_0 dt << 1`` the
three surviving terms cancel to ``O((omega_0 dt)^3)`` from ``O(omega_0 dt)``
pieces, which in float32 leaves nothing.  The closed forms below are the
algebraically rearranged versions in which every piece is computed by a
function that is regular at the origin (:mod:`anvilgp.special`).  The
identity that makes this work is

    Q11/sigma^2 = E * [expm1(2a) - 2 p (C + p)]
                = gamma3(2a) + E * [8 a u sincm1(4u)
                                    + 2 a^2 u sincm1(u) (1 + sinc(u))]

with ``a = c dt``, ``u = d^2 dt^2``, ``E = exp(-2a)``, ``p = a sinc(u)``.
The second line has no subtraction of nearly equal quantities anywhere, and
``gamma3`` is bounded in [0, 1), so the ``expm1(2a)`` of the first line --
which overflows at long lag inside a product that does not -- never forms.

The two bracketed terms each get a second, algebraically identical form for
large ``|u|``:  ``E 2a (1 - sinc(4u))`` and ``E 2a^2 (1 - sinc(u)^2)``
become differences of the damped products, which is overflow-free where the
sinc arguments are not, and where the brackets are O(1) so the subtraction
costs nothing.  The switch is at ``|u| = 1/2``, well inside both branches'
comfortable range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .linalg import matmul as _mm_small
from .special import (
    damped_cos_sinc,
    exp_acc,
    gamma3,
    sinc_gen,
    sincm1,
)

__all__ = [
    "Term", "TermSum", "RealTerm", "SHOTerm",
    "Matern32Term", "Matern52Term", "RotationTerm",
]

_TWO_PI = 2.0 * math.pi


class Term:
    """Base class: an SSM kernel term over a non-dimensionalised state.

    Subclasses define ``n_state``, ``param_names`` (the natural-unit
    parameters this term consumes, in order), and :meth:`matrices`.

    Shapes.  ``p`` is ``(n_chains, len(param_names))`` in natural units
    (already exponentiated out of any log parametrisation); ``dt`` is
    ``(K,)`` or ``(n_chains, K)``.  :meth:`matrices` returns

        phi : (n_chains, K, m, m)
        q   : (n_chains, K, m, m)
        h   : (n_chains, m)

    with ``P_inf = I_m`` implied.  ``dt = 0`` (duplicate timestamps, which
    celerite2 accepts and so must we) must give ``phi = I``, ``q = 0``
    exactly, with no NaN in the gradient.
    """

    n_state: int = 0
    param_names: tuple[str, ...] = ()

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def matrices(self, p: mx.array, dt: mx.array):
        raise NotImplementedError

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Sizes of Phi's and Q's diagonal blocks.  One block by default."""
        return (self.n_state,)

    def matrices_blocks(self, p: mx.array, dt: mx.array):
        """``(phi_blocks, q_blocks, h)`` -- the block-diagonal structure kept.

        The default splits what :meth:`matrices` assembled, which is pointless
        for a single block and is overridden by :class:`TermSum`, the only
        term that has more than one.  Keeping the blocks apart lets the leaf
        be stored compactly (``2 sum m_i^2 + 2m + 2`` numbers instead of
        ``3m^2 + 2m + 1``) and lets the composition multiply by Phi without
        touching its zeros.
        """
        phi, q, h = self.matrices(p, dt)
        return [phi], [q], h

    def __add__(self, other: "Term") -> "TermSum":
        left = self.terms if isinstance(self, TermSum) else (self,)
        right = other.terms if isinstance(other, TermSum) else (other,)
        return TermSum(*left, *right)

    # -- host-side, float64: the covariance function itself ---------------
    def value_np(self, p: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """k(tau) in float64 on the host, for oracles and documentation."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# real / exponential term
# ---------------------------------------------------------------------------


@dataclass
class RealTerm(Term):
    """``k(tau) = sigma^2 exp(-c tau)``.  One state dimension.

    Parameters (natural units): ``sigma``, ``c``.
    """

    n_state: int = field(default=1, init=False)
    param_names: tuple[str, ...] = field(default=("sigma", "c"), init=False)

    def matrices(self, p: mx.array, dt: mx.array):
        sigma = p[..., 0:1]
        c = p[..., 1:2]
        if dt.ndim == 1:
            dt = dt[None, :]
        a = c * dt                                   # (n, K)
        e = exp_acc(-a)
        phi = e[..., None, None]
        # Q = 1 - e^{-2a} in the unit-variance state; -expm1 is exact.
        q = (-mx.expm1(-2.0 * a))[..., None, None]
        h = mx.concatenate([sigma], axis=-1)
        return phi, q, h

    def value_np(self, p, tau):
        sigma, c = float(p[0]), float(p[1])
        return sigma ** 2 * np.exp(-c * np.abs(tau))


# ---------------------------------------------------------------------------
# SHO term (and Matern-3/2, which is the same object at Q = 1/2)
# ---------------------------------------------------------------------------


def _sho_matrices(sigma, omega0, qual, dt):
    """Propagator and process noise for one SHO block, unit stationary variance.

    ``sigma`` only enters ``h``; the returned ``phi``/``q`` are amplitude-free.
    All of ``a``, ``u``, ``w`` are dimensionless combinations of ``dt`` with
    the two natural rates, so nothing here has units to lose.
    """
    two_q = 2.0 * qual
    c = omega0 / two_q                            # damping rate
    # d^2 = omega0^2 (1 - 1/(4Q^2)); factored as (2Q-1)(2Q+1)/(4Q^2) so the
    # difference does not cancel as Q -> 1/2.  It vanishes *exactly* there,
    # which is the point: d never appears alone, only through u = d^2 dt^2.
    dsq = omega0 * omega0 * ((two_q - 1.0) * (two_q + 1.0)) / (two_q * two_q)
    # |d|, used *only* by the overdamped branch (u < 0) below.  The argument
    # is masked to 1 wherever dsq >= 0 rather than passed as |dsq|, because
    # d(sqrt)/dx is infinite at 0 and MLX propagates that through a zero
    # cotangent as 0 * inf = NaN.  dsq is exactly 0 at Q = 1/2 -- which
    # Matern32Term sets on every chain -- so |dsq| gave every Matern-3/2
    # gradient a NaN while its value stayed correct.  The mask is free: the
    # lanes it changes are exactly the lanes damped_cos_sinc discards.
    absd = mx.sqrt(mx.where(dsq < 0.0, -dsq, mx.ones_like(dsq)))

    a = c * dt                                    # (n, K), >= 0
    u = dsq * dt * dt                             # (n, K), any sign
    w = omega0 * dt
    # Overdamped decay rates a -/+ phi.  c - |d| = omega_0^2/(c + |d|):
    # the literal subtraction loses log10(1/(2Q^2)) digits as Q -> 0.
    cpd = c + absd
    lo = omega0 * omega0 * dt / cpd
    hi = cpd * dt

    # e^-a cos(d dt) and e^-a sin(d dt)/(d dt), formed as products so the
    # overdamped inf * 0 never arises.
    ec, es = damped_cos_sinc(a, u, lo, hi)

    # Phi in the non-dimensionalised state x' = diag(1, 1/omega_0) x / sigma:
    #   Phi' = [[ec + a es,  w es], [-w es,  ec - a es]],  det = e^{-2a}.
    phi = mx.stack([
        mx.stack([ec + a * es, w * es], axis=-1),
        mx.stack([-w * es, ec - a * es], axis=-1),
    ], axis=-2)

    # E = e^{-2a}, recovered from the two damped products by the identity
    # cos^2 + d^2 S^2 = 1 -- no second exponential, and exact for u >= 0
    # where it is a sum of positives.  For u < 0 it *is* a cancellation
    # (cosh^2 - sinh^2), but harmlessly so: in that regime E is exponentially
    # smaller than the es*ec and es*es it is added to below, so its error is
    # a negligible addend rather than the result.  Checked by the Lyapunov
    # test at Q = 0.1 out to omega_0 dt = 100.
    ee = ec * ec + u * es * es
    a2 = a * a
    # Q11 = gamma3(2a) + E 2a (1 - sinc(4u)) + E 2a^2 (1 - sinc(u)^2).
    # Two equivalent forms for each bracket: the series form has no
    # cancellation as u -> 0 but needs bounded sinc arguments; the
    # difference-of-products form is overflow-free at large |u|, where the
    # brackets are O(1) and the subtraction is harmless.
    u_small = mx.abs(u) <= 0.5
    us = mx.where(u_small, u, mx.zeros_like(u))
    ta = mx.where(u_small,
                  ee * 8.0 * a * us * sincm1(4.0 * us),
                  2.0 * a * (ee - es * ec))
    tb = mx.where(u_small,
                  ee * 2.0 * a2 * us * sincm1(us) * (1.0 + sinc_gen(us)),
                  2.0 * a2 * (ee - es * es))
    q11 = gamma3(2.0 * a) + ta + tb
    q22 = (-mx.expm1(-2.0 * a)) + 2.0 * a * es * ec - 2.0 * a2 * es * es
    q12 = 2.0 * a * w * es * es
    q = mx.stack([
        mx.stack([q11, q12], axis=-1),
        mx.stack([q12, q22], axis=-1),
    ], axis=-2)

    zero = mx.zeros_like(sigma)
    h = mx.concatenate([sigma, zero], axis=-1)
    return phi, q, h


@dataclass
class SHOTerm(Term):
    """Stochastically-driven damped harmonic oscillator.

    ``k(tau) = sigma^2 e^{-c tau} [cos(d tau) + (c/d) sin(d tau)]`` with
    ``c = omega_0/(2Q)``, ``d^2 = omega_0^2 - c^2`` -- the celerite2 SHOTerm,
    with ``sigma^2`` the stationary variance and ``rho = 2 pi / omega_0``.

    Parameters (natural units): ``sigma``, ``rho``, ``Q``.

    Sample ``(log sigma, log rho, log Q)``, never ``(S0, omega0, Q)``: at
    high Q the posterior is axis-aligned in the former and a curved ridge in
    the latter.  For granulation, fix ``Q = 1/sqrt(2)`` and sample
    ``(sigma, rho)``.
    """

    n_state: int = field(default=2, init=False)
    param_names: tuple[str, ...] = field(default=("sigma", "rho", "Q"),
                                         init=False)

    def matrices(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        sigma = p[..., 0:1]
        omega0 = _TWO_PI / p[..., 1:2]
        qual = p[..., 2:3]
        return _sho_matrices(sigma, omega0, qual, dt)

    def value_np(self, p, tau):
        sigma, rho, q = (float(v) for v in p[:3])
        tau = np.abs(np.asarray(tau, dtype=np.float64))
        w0 = _TWO_PI / rho
        c = w0 / (2.0 * q)
        dsq = w0 * w0 - c * c
        if dsq > 0:
            d = np.sqrt(dsq)
            return sigma ** 2 * np.exp(-c * tau) * (
                np.cos(d * tau) + (c / d) * np.sin(d * tau))
        if dsq == 0:
            return sigma ** 2 * np.exp(-c * tau) * (1.0 + c * tau)
        # Overdamped: the cosh/sinh form overflows long before the kernel
        # itself becomes small.  Split into the two real exponentials, whose
        # rates are c -/+ d = w0^2/(c+d) and c+d -- both positive, so both
        # branches decay and nothing overflows.
        d = np.sqrt(-dsq)
        rate_slow = w0 * w0 / (c + d)
        rate_fast = c + d
        return 0.5 * sigma ** 2 * (
            (1.0 + c / d) * np.exp(-rate_slow * tau)
            + (1.0 - c / d) * np.exp(-rate_fast * tau))


@dataclass
class Matern32Term(Term):
    """``k(tau) = sigma^2 (1 + sqrt3 tau/rho) exp(-sqrt3 tau/rho)``.

    Not celerite-representable, and needs no new code: it is exactly the
    SHO at ``Q = 1/2`` with ``omega_0 = sqrt(3)/rho``.  This is the payoff
    of building the propagator to be regular at critical damping.

    Parameters (natural units): ``sigma``, ``rho``.
    """

    n_state: int = field(default=2, init=False)
    param_names: tuple[str, ...] = field(default=("sigma", "rho"), init=False)

    def matrices(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        sigma = p[..., 0:1]
        omega0 = math.sqrt(3.0) / p[..., 1:2]
        qual = mx.full(omega0.shape, 0.5, dtype=omega0.dtype)
        return _sho_matrices(sigma, omega0, qual, dt)

    def value_np(self, p, tau):
        sigma, rho = float(p[0]), float(p[1])
        x = math.sqrt(3.0) * np.abs(np.asarray(tau, dtype=np.float64)) / rho
        return sigma ** 2 * (1.0 + x) * np.exp(-x)


# ---------------------------------------------------------------------------
# Matern-5/2: a triple root, m = 3
# ---------------------------------------------------------------------------


@dataclass
class Matern52Term(Term):
    """``k(tau) = sigma^2 (1 + x + x^2/3) e^{-x}``, ``x = sqrt5 tau/rho``.

    Three state dimensions ``(y, ydot, yddot)`` with a triple characteristic
    root at ``-lambda``, so the propagator is ``e^{-lambda dt}`` times a
    quadratic and the process noise is built from lower incomplete gammas.
    Not celerite-representable.

    Parameters (natural units): ``sigma``, ``rho``.

    v1 note: m = 3 breaks the block-diagonal 2x2 structure the fused Metal
    kernel exploits, so this term is served by the reference path only.
    """

    n_state: int = field(default=3, init=False)
    param_names: tuple[str, ...] = field(default=("sigma", "rho"), init=False)

    def matrices(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        sigma = p[..., 0:1]
        lam = math.sqrt(5.0) / p[..., 1:2]
        return _matern52_matrices(sigma, lam, dt)

    def value_np(self, p, tau):
        sigma, rho = float(p[0]), float(p[1])
        x = math.sqrt(5.0) * np.abs(np.asarray(tau, dtype=np.float64)) / rho
        return sigma ** 2 * (1.0 + x + x * x / 3.0) * np.exp(-x)


# lower incomplete gamma P(n+1, z) = 1 - e^{-z} sum_{j<=n} z^j/j!, as a
# series for small z (cancellation-free) and directly for large z.  The n = 2
# case is anvilgp.special.gamma3; the Matern-5/2 process noise needs n up to 4.
_GAMMA_SERIES_N = 24
_GAMMA_Z_SWITCH = 2.0


def _lower_gamma_reg(n: int, z: mx.array) -> mx.array:
    """``P(n+1, z) = 1 - e^{-z} sum_{j<=n} z^j/j!``, cancellation-free.

    Series ``z^{n+1} sum_k (-z)^k / (k! (n+1+k) n!)`` below the switch; the
    n = 2 case is :func:`anvilgp.special.gamma3`, and the same identity in
    the same form -- this is the one function that stands between the
    process noise and a total loss of significance at ``lambda dt << 1``.
    """
    coeffs = [(-1.0) ** k / (math.factorial(k) * (n + 1 + k)
                             * math.factorial(n))
              for k in range(_GAMMA_SERIES_N)]
    small = z <= _GAMMA_Z_SWITCH
    zs = mx.where(small, z, mx.zeros_like(z))
    acc = mx.array(coeffs[-1], dtype=z.dtype) + 0.0 * zs
    for c in reversed(coeffs[:-1]):
        acc = acc * zs + c
    ser = acc
    for _ in range(n + 1):
        ser = ser * zs
    zb = mx.where(small, mx.ones_like(z), z)
    term = mx.ones_like(zb)
    tot = term
    for j in range(1, n + 1):
        term = term * zb / j
        tot = tot + term
    direct = 1.0 - exp_acc(-zb) * tot
    return mx.where(small, ser, direct)


# Whitening of the lambda-scaled Matern-5/2 state.  In the scaled frame
# z = diag(1, 1/l, 1/l^2) y / sigma the stationary covariance is the *constant*
# matrix W = [[1,0,-1/3],[0,1/3,0],[-1/3,0,1]] -- lambda- and sigma-free, but
# not the identity.  One more constant similarity by its Cholesky factor makes
# P_inf = I for this term too, so the element algebra can assume P_inf = I
# uniformly across the kernel set.  h is unchanged: the first row of LW is
# e_1, so h LW = [sigma, 0, 0] still.
_M52_LW = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0 / math.sqrt(3.0), 0.0],
                    [-1.0 / 3.0, 0.0, 2.0 * math.sqrt(2.0) / 3.0]])
_M52_LW_INV = np.linalg.inv(_M52_LW)
_m52_const: dict = {}


def _m52_consts(dtype):
    if dtype not in _m52_const:
        _m52_const[dtype] = (
            mx.array(_M52_LW.astype(np.float64), dtype=dtype),
            mx.array(_M52_LW_INV.astype(np.float64), dtype=dtype),
            mx.array(_M52_LW_INV.T.copy().astype(np.float64), dtype=dtype),
        )
    return _m52_const[dtype]


def _matern52_matrices(sigma, lam, dt):
    """Matern-5/2 propagator and process noise in the whitened state.

    The natural state is ``(y, ydot, yddot)`` with
    ``P_inf = sigma^2 [[1,0,-l^2/3],[0,l^2/3,0],[-l^2/3,0,l^4]]``: not
    diagonal, and carrying an ``l^4`` scale disparity that would make any
    scale-aware Cholesky floor meaningless.  Two constant similarities fix
    both -- ``diag(1, 1/l, 1/l^2)/sigma`` removes the units, the Cholesky of
    what remains removes the correlation.
    """
    tau = lam * dt                                   # (n, K)
    e = exp_acc(-tau)
    t2 = tau * tau
    # Phi in the lambda-scaled state: Phi_scaled[i][j] = Phi[i][j] l^{j-i}.
    phi_s = mx.stack([
        mx.stack([e * (1.0 + tau + 0.5 * t2),
                  e * tau * (1.0 + tau),
                  e * 0.5 * t2], axis=-1),
        mx.stack([-e * 0.5 * t2,
                  e * (1.0 + tau - t2),
                  e * tau * (1.0 - 0.5 * tau)], axis=-1),
        mx.stack([e * tau * (0.5 * tau - 1.0),
                  e * tau * (tau - 3.0),
                  e * (1.0 - 2.0 * tau + 0.5 * t2)], axis=-1),
    ], axis=-2)

    # Q = P_inf - Phi P_inf Phi^T is forbidden here for the same reason as
    # for the SHO.  Integrate instead:
    #   Q_ij = q_c int_0^dt g_i(s) g_j(s) ds,  g = Phi(s) e_3,
    # whose integrands are polynomials times e^{-2 lambda s}, so every entry
    # is a combination of regularised lower incomplete gammas in z = 2 lambda
    # dt.  In the scaled frame q_c/(sigma^2 lambda^5) = 16/3 exactly (from
    # Var(y) = q_c int_0^inf (e^{-ls} s^2/2)^2 ds = 3 q_c/(16 l^5)).
    z = 2.0 * tau
    mom = [_lower_gamma_reg(n, z) * (math.factorial(n) / 2.0 ** (n + 1))
           for n in range(6)]

    def poly_int(ca, cb):
        """int_0^tau e^{-2x} A(x) B(x) dx for polynomials A, B in x."""
        out = None
        for i, va in enumerate(ca):
            if va == 0.0:
                continue
            for j, vb in enumerate(cb):
                if vb == 0.0:
                    continue
                term = mom[i + j] * (va * vb)
                out = term if out is None else out + term
        return out

    g1 = [0.0, 0.0, 0.5]           # x^2/2
    g2 = [0.0, 1.0, -0.5]          # x - x^2/2
    g3 = [1.0, -2.0, 0.5]          # 1 - 2x + x^2/2
    s = 16.0 / 3.0
    q11, q12, q13 = (s * poly_int(g1, g1), s * poly_int(g1, g2),
                     s * poly_int(g1, g3))
    q22, q23, q33 = (s * poly_int(g2, g2), s * poly_int(g2, g3),
                     s * poly_int(g3, g3))
    q_s = mx.stack([
        mx.stack([q11, q12, q13], axis=-1),
        mx.stack([q12, q22, q23], axis=-1),
        mx.stack([q13, q23, q33], axis=-1),
    ], axis=-2)

    lw, lwi, lwit = _m52_consts(dt.dtype)
    phi = _mm_small(lwi, _mm_small(phi_s, lw))
    q = _mm_small(lwi, _mm_small(q_s, lwit))

    zero = mx.zeros_like(sigma)
    h = mx.concatenate([sigma, zero, zero], axis=-1)
    return phi, q, h


# ---------------------------------------------------------------------------
# composites
# ---------------------------------------------------------------------------


@dataclass
class RotationTerm(Term):
    """Two SHOs at ``P`` and ``P/2`` -- the celerite2/exoplanet rotation kernel.

    Parameters (natural units): ``sigma``, ``period``, ``Q0``, ``dQ``, ``f``.
    The primary mode has ``Q1 = 1/2 + Q0 + dQ`` and *damped* period
    ``period``; the secondary has ``Q2 = 1/2 + Q0`` and damped period
    ``period/2``; ``f`` is the secondary's fraction of the total variance.

    Sample ``(log sigma, log P, log Q0, log dQ, logit f)``.  ``Q0, dQ > 0``
    keeps both modes strictly underdamped, which is the physical regime for
    stellar rotation; nothing here needs the Q = 1/2 crossing, but it costs
    nothing to be regular there.
    """

    n_state: int = field(default=4, init=False)
    param_names: tuple[str, ...] = field(
        default=("sigma", "period", "Q0", "dQ", "f"), init=False)

    def matrices(self, p: mx.array, dt: mx.array):
        phis, qs, h = self.matrices_blocks(p, dt)
        return _block_diag(*phis), _block_diag(*qs), h

    def matrices_blocks(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        sigma = p[..., 0:1]
        period = p[..., 1:2]
        q0 = p[..., 2:3]
        dq = p[..., 3:4]
        f = p[..., 4:5]

        amp = sigma * sigma / (1.0 + f)
        q1 = 0.5 + q0 + dq
        q2 = 0.5 + q0
        # omega_0 chosen so the *damped* period matches: w0 eta = 2 pi / P.
        w1 = 4.0 * math.pi * q1 / (period * mx.sqrt(mx.maximum(
            (2.0 * q1 - 1.0) * (2.0 * q1 + 1.0), mx.zeros_like(q1) + 1e-30)))
        w2 = 8.0 * math.pi * q2 / (period * mx.sqrt(mx.maximum(
            (2.0 * q2 - 1.0) * (2.0 * q2 + 1.0), mx.zeros_like(q2) + 1e-30)))
        s1 = mx.sqrt(amp)
        s2 = mx.sqrt(f * amp)

        phi1, qq1, h1 = _sho_matrices(s1, w1, q1, dt)
        phi2, qq2, h2 = _sho_matrices(s2, w2, q2, dt)
        return [phi1, phi2], [qq1, qq2], mx.concatenate([h1, h2], axis=-1)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return (2, 2)

    def value_np(self, p, tau):
        sigma, period, q0, dq, f = (float(v) for v in p[:5])
        amp = sigma ** 2 / (1.0 + f)
        q1, q2 = 0.5 + q0 + dq, 0.5 + q0
        w1 = 4.0 * math.pi * q1 / (period * math.sqrt(4.0 * q1 ** 2 - 1.0))
        w2 = 8.0 * math.pi * q2 / (period * math.sqrt(4.0 * q2 ** 2 - 1.0))
        sho = SHOTerm()
        return (sho.value_np(np.array([math.sqrt(amp), _TWO_PI / w1, q1]), tau)
                + sho.value_np(np.array([math.sqrt(f * amp), _TWO_PI / w2, q2]),
                               tau))


def _block_diag(*blocks: mx.array) -> mx.array:
    """Block-diagonal assembly of batched square matrices, in one pass.

    Folding pairwise (``bd(bd(bd(a, b), c), d)``) rewrites the growing
    left-hand block once per step, so a k-term sum copies O(k^2) worth of
    mostly-zero matrix.  Measured at (1024, 4096) batch: 48 ms for four 2x2
    blocks pairwise.  Placing every block against the final width once costs
    O(k) and needs no intermediates.
    """
    if len(blocks) == 1:
        return blocks[0]
    sizes = [b.shape[-1] for b in blocks]
    m = sum(sizes)
    batch = blocks[0].shape[:-2]
    dtype = blocks[0].dtype
    rows, off = [], 0
    for b, mi in zip(blocks, sizes):
        parts = []
        if off:
            parts.append(mx.zeros(batch + (mi, off), dtype=dtype))
        parts.append(b)
        if m - off - mi:
            parts.append(mx.zeros(batch + (mi, m - off - mi), dtype=dtype))
        rows.append(mx.concatenate(parts, axis=-1) if len(parts) > 1 else b)
        off += mi
    return mx.concatenate(rows, axis=-2)


@dataclass
class TermSum(Term):
    """Sum of independent terms: block-diagonal Phi/Q, concatenated h.

    The observed process is the sum of the components, so ``h`` is the
    concatenation of the component observation vectors and ``P_inf`` stays
    the identity.

    For sums of *same-type* terms, impose frequency ordering on the sampled
    parameters to kill permutation multimodality -- otherwise a k-term sum
    has k! identical posterior modes and every convergence diagnostic
    becomes meaningless.
    """

    terms: tuple[Term, ...]

    def __init__(self, *terms: Term):
        self.terms = tuple(terms)
        self.n_state = sum(t.n_state for t in terms)
        names: list[str] = []
        for i, t in enumerate(terms):
            names.extend(f"{type(t).__name__[:-4].lower()}{i}:{n}"
                         for n in t.param_names)
        self.param_names = tuple(names)

    def matrices(self, p: mx.array, dt: mx.array):
        phis, qs, h = self.matrices_blocks(p, dt)
        return _block_diag(*phis), _block_diag(*qs), h

    def matrices_blocks(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        phis, qs, hs = [], [], []
        off = 0
        for t in self.terms:
            k = t.n_params
            ph, qq, hh = t.matrices_blocks(p[..., off:off + k], dt)
            phis.extend(ph)
            qs.extend(qq)
            hs.append(hh)
            off += k
        return phis, qs, mx.concatenate(hs, axis=-1)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        out: list[int] = []
        for t in self.terms:
            out.extend(getattr(t, "block_sizes", (t.n_state,)))
        return tuple(out)

    def value_np(self, p, tau):
        out = np.zeros_like(np.asarray(tau, dtype=np.float64))
        off = 0
        for t in self.terms:
            k = t.n_params
            out = out + t.value_np(np.asarray(p)[off:off + k], tau)
            off += k
        return out
