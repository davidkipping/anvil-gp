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
    lower_gamma_reg,
    sinc_gen,
    gamma3_shifted,
    sincos,
    sincm1_series,
    coshm1_sinhm_over_z2, coshm1_sinhm_complex,
)

__all__ = [
    "Term", "TermSum", "RealTerm", "SHOTerm", "ComplexTerm",
    "Matern32Term", "Matern52Term", "RotationTerm", "TermConvolution",
    "from_celerite_coefficients", "MAX_FDELTA",
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

    def celerite_coefficients(self, p: mx.array):
        """``(a_real, c_real, a, b, c, d)``, each ``(n_chains, k)``.

        The celerite decomposition ``k(tau) = sum a_r e^{-c_r tau} + sum
        e^{-c tau}(a cos d tau + b sin d tau)``.  Only
        :class:`TermConvolution` needs it, because exposure-time integration
        is defined by a closed form in these coefficients.

        **Not every term has them**, which is not a gap in the
        implementation: :class:`Matern32Term` is a *repeated* root (``d = 0``
        exactly, which is what makes it exact here and approximate in
        celerite2) and :class:`Matern52Term` a triple one, so neither is a
        sum of distinct exponentials or a complex pair.  Those raise.
        """
        raise TypeError(
            f"{type(self).__name__} has no exact celerite coefficients, so it "
            f"cannot be wrapped in TermConvolution; see its docstring")

    def diag_correction(self, p: mx.array):
        """Constant to add to every observation's variance, ``(n_chains, 1)``.

        ``None`` for every term but :class:`TermConvolution`, whose
        convolution is *not* a sum of exponentials below the exposure width
        and so needs a correction at zero lag.  It is a function of the
        sampled parameters, so it carries a gradient.
        """
        return None

    def generator(self, p: mx.array):
        """The continuous-time generators ``F``, one per diagonal block.

        ``Phi(dt) = expm(F dt)`` in the *same* non-dimensionalised basis
        :meth:`matrices_blocks` uses, each of shape ``(n_chains, m_b, m_b)``.

        Only :class:`TermConvolution` needs this, and it needs it because
        the exposure-time convolution is a matrix function of ``F`` --
        ``2(cosh(F delta) - I)/(F delta)^2`` -- which is defined for a
        *repeated* characteristic root and for a negative-amplitude
        decomposition alike.  The celerite coefficients are defined for
        neither, which is why the coefficient route excluded the Matern
        terms and the overdamped SHO and this route does not.

        Every implementation here is verified against
        ``logm(Phi(dt))/dt`` from the term's own propagator, to 9e-15 or
        better.
        """
        raise TypeError(
            f"{type(self).__name__} does not provide a generator, so it "
            f"cannot be wrapped in TermConvolution")

    def p_inf_blocks(self, p: mx.array, dtype=None):
        """Stationary state covariance, one per block.  ``None`` means ``I``.

        ``None`` is not the same as returning identities: it tells the
        callers that this term keeps the package-wide ``P_inf = I``
        non-dimensionalisation, so row 0 of the lag table stays a constant
        and no per-chain matrix has to be built.  Only
        :class:`TermConvolution` overrides it.
        """
        return None

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
        return _real_matrices(p[..., 0:1], p[..., 1:2], dt)

    def generator(self, p: mx.array):
        return [-p[..., 1:2][..., None]]

    def value_np(self, p, tau):
        sigma, c = float(p[0]), float(p[1])
        return sigma ** 2 * np.exp(-c * np.abs(tau))

    def celerite_coefficients(self, p: mx.array):
        sigma, c = p[..., 0:1], p[..., 1:2]
        empty = sigma[..., :0]
        return sigma * sigma, c, empty, empty, empty, empty


# ---------------------------------------------------------------------------
# SHO term (and Matern-3/2, which is the same object at Q = 1/2)
# ---------------------------------------------------------------------------


def _real_matrices(sigma, c, dt):
    """Propagator and process noise for one real block, unit variance, m = 1.

    Factored out of :class:`RealTerm` so :class:`TermConvolution` can build
    the real components of a convolved term from the same code.
    """
    if dt.ndim == 1:
        dt = dt[None, :]
    a = c * dt                                   # (n, K)
    phi = exp_acc(-a)[..., None, None]
    # Q = 1 - e^{-2a} in the unit-variance state; -expm1 is exact.
    q = (-mx.expm1(-2.0 * a))[..., None, None]
    return phi, q, mx.concatenate([sigma], axis=-1)


def _pair_generator(g, s, c):
    """``F = [[-g, s], [-s, -(2c - g)]]`` -- the 2-state block's generator.

    Exactly the matrix :func:`_complex_matrices` documents and realises, so
    ``expm(F dt)`` is the ``phi`` it returns.  Built here rather than there
    because the convolution needs ``F`` itself, not its exponential.
    """
    row0 = mx.concatenate([-g, s], axis=-1)
    row1 = mx.concatenate([-s, -(2.0 * c - g)], axis=-1)
    return mx.stack([row0, row1], axis=-2)


def _sho_generator(omega0, qual):
    """The SHO's generator: :func:`_pair_generator` at ``beta = c``, ``g = 0``.

    ``F = [[0, omega_0], [-omega_0, -2c]]`` with ``c = omega_0/(2Q)``, whose
    eigenvalues are ``-c +/- i d`` and which is **regular at and below
    ``Q = 1/2``** -- no ``d`` appears, so nothing divides by zero and
    nothing takes a root of a negative number.  That regularity is the
    whole reason the state-space route can convolve an overdamped SHO and
    the coefficient route cannot.
    """
    zero = mx.zeros_like(omega0)
    return _pair_generator(zero, omega0, omega0 / (2.0 * qual))


def _complex_matrices(sigma, c, beta, s, g, dsq, wsq, dt):
    """Propagator and process noise for one 2-state block, ``P_inf = I``.

    The general celerite-class complex term, of which the SHO is the case
    ``beta = c`` (hence ``g = 0``) and ``s = omega_0``.  ``_sho_matrices``
    below is exactly that call, and is bit-identical to what it was before
    this generalisation.

    ``k(tau) = a e^{-c tau} cos(d tau) + b e^{-c tau} sin(d tau)`` is realised
    by ``h = (sqrt(a), 0)``, ``P_inf = I`` and

        F = [[-g, s], [-s, -(2c - g)]],   beta = b d / a,
        g = c - beta,                     s = sqrt(beta^2 + d^2),

    whose eigenvalues are ``-c +/- i d`` for every ``beta`` -- the rates are
    the SHO's, only the amplitude ratio moves.  Then

        Phi = [[ec + b_ es,  s_ es], [-s_ es,  ec - b_ es]]
        Q   = I - Phi Phi^T,  which reduces to
              Q11 = (1-E) - 2 b_ ec es - 2 b_^2 es^2
              Q22 = (1-E) + 2 b_ ec es - 2 b_^2 es^2
              Q12 = 2 b_ s_ es^2

    with ``b_ = beta dt``, ``s_ = s dt``, ``E = e^{-2a}``.  Verified against
    ``expm(F dt)`` and ``I - Phi Phi^T`` to 4e-16, and the kernel against
    ``celerite2.terms.ComplexTerm`` to 8e-15, over 120 random terms x 5 lags.

    **Why so many arguments.** ``dsq`` (``= d^2``) and ``wsq`` (``= c^2+d^2``)
    are passed rather than derived, because deriving them here would
    reintroduce exactly the cancellations each caller already avoids in its
    own parametrisation: the SHO gets ``d^2`` through the factored
    ``(2Q-1)(2Q+1)/(4Q^2)`` so it vanishes *exactly* at ``Q = 1/2``, and
    ``wsq`` is ``omega_0^2`` outright.  ``g`` likewise: ``Q11``'s correction
    needs ``c - beta``, which every caller knows directly, and forming it
    here would cancel as the term approaches the SHO.

    ``sigma`` only enters ``h``; the returned ``phi``/``q`` are
    amplitude-free.
    """
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
    b_ = beta * dt                                # Phi's diagonal rate
    s_ = s * dt                                   # Phi's off-diagonal scale
    # Overdamped decay rates a -/+ phi.  c - |d| = omega_0^2/(c + |d|):
    # the literal subtraction loses log10(1/(2Q^2)) digits as Q -> 0.
    cpd = c + absd
    lo = wsq * dt / cpd
    hi = cpd * dt

    # e^-a cos(d dt) and e^-a sin(d dt)/(d dt), formed as products so the
    # overdamped inf * 0 never arises.
    ec, es = damped_cos_sinc(a, u, lo, hi)

    # Phi in the non-dimensionalised state x' = diag(1, 1/omega_0) x / sigma:
    #   Phi' = [[ec + b_ es,  s_ es], [-s_ es,  ec - b_ es]], det = e^{-2a}.
    phi = mx.stack([
        mx.stack([ec + b_ * es, s_ * es], axis=-1),
        mx.stack([-s_ * es, ec - b_ * es], axis=-1),
    ], axis=-2)

    # E = e^{-2a}.  Taken directly, not recovered from the identity
    # cos^2 + d^2 S^2 = 1.
    #
    # That identity is exact for u >= 0, where it is a sum of positives, and
    # it was used here to save an exponential.  For u < 0 it is
    # cosh^2 - sinh^2, and the old comment argued the cancellation was
    # harmless because E is then exponentially small next to the es*ec and
    # es*es it is added to.  That argument is about the *value* of E and not
    # about the *error* in computing it: the absolute error is ~eps phi^2
    # es^2, and `tb` below multiplies it by 2a^2.  Measured in float32 at
    # Q = 0.05: 3.0e-4 relative in Q11, which is 4.1 log-likelihood units at
    # N = 16384 -- hundreds of times a Metropolis decision.  The default
    # log_Q prior reaches Q = 0.05, so this is inside the sampled region.
    #
    # One extra exponential per lag, on a table of distinct lags, is nothing.
    ee = exp_acc(-2.0 * a)
    b2 = b_ * b_
    # Q11 = gamma3(2a) + E 2a (1 - sinc(4u)) + E 2a^2 (1 - sinc(u)^2).
    # Two equivalent forms for each bracket: the series form has no
    # cancellation as u -> 0 but needs bounded sinc arguments; the
    # difference-of-products form is overflow-free at large |u|, where the
    # brackets are O(1) and the subtraction is harmless.
    u_small = mx.abs(u) <= 0.5
    us = mx.where(u_small, u, mx.zeros_like(u))
    # `sincm1_series`, not `sincm1`: `|us| <= 0.5` bounds both arguments by
    # `_U_SERIES`, so `sincm1`'s other branch is unreachable here, and that
    # branch is what makes any `mx.compile` over this graph exceed MLX's
    # fused-argument limit.  Bit-identical on this domain.
    ta = mx.where(u_small,
                  ee * 8.0 * b_ * us * sincm1_series(4.0 * us),
                  2.0 * b_ * (ee - es * ec))
    tb = mx.where(u_small,
                  ee * 2.0 * b2 * us * sincm1_series(us) * (1.0 + sinc_gen(us)),
                  2.0 * b2 * (ee - es * es))
    # `gamma3_shifted(2a, 2g dt)`, not `gamma3(2a)`: the polynomial belongs
    # to `beta` while the exponential belongs to `c`.  `g dt` is passed, not
    # formed as `a - b_`, so the one cancellation is avoided.  At g = 0 this
    # is `gamma3(2a)` bit for bit.
    q11 = gamma3_shifted(2.0 * a, 2.0 * g * dt) + ta + tb
    q22 = (-mx.expm1(-2.0 * a)) + 2.0 * b_ * es * ec - 2.0 * b2 * es * es
    q12 = 2.0 * b_ * s_ * es * es
    q = mx.stack([
        mx.stack([q11, q12], axis=-1),
        mx.stack([q12, q22], axis=-1),
    ], axis=-2)

    zero = mx.zeros_like(sigma)
    h = mx.concatenate([sigma, zero], axis=-1)
    return phi, q, h


def _sho_matrices(sigma, omega0, qual, dt):
    """Propagator and process noise for one SHO block, unit stationary variance.

    The ``beta = c`` case of :func:`_complex_matrices`, where the amplitude
    ratio is the SHO's ``b/a = c/d`` and so ``g = c - beta = 0``.  Kept as
    its own entry point because the two rate combinations it forms --
    ``d^2`` through the factored ``(2Q-1)(2Q+1)/(4Q^2)``, and
    ``omega_0^2`` -- are what keep the propagator regular at ``Q = 1/2``,
    and they are expressed in *this* parametrisation, not the general one.
    """
    two_q = 2.0 * qual
    c = omega0 / two_q                            # damping rate
    # d^2 = omega0^2 (1 - 1/(4Q^2)); factored as (2Q-1)(2Q+1)/(4Q^2) so the
    # difference does not cancel as Q -> 1/2.  It vanishes *exactly* there,
    # which is the point: d never appears alone, only through u = d^2 dt^2.
    dsq = omega0 * omega0 * ((two_q - 1.0) * (two_q + 1.0)) / (two_q * two_q)
    return _complex_matrices(sigma, c, c, omega0, mx.zeros_like(c), dsq,
                             omega0 * omega0, dt)


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

    def generator(self, p: mx.array):
        return [_sho_generator(_TWO_PI / p[..., 1:2], p[..., 2:3])]

    def celerite_coefficients(self, p: mx.array):
        """One complex pair, **valid only for ``Q > 1/2``**.

        Below critical damping the SHO is two *real* exponentials, one with
        a **negative** amplitude (celerite2 returns exactly that), and a
        negative amplitude is the one case the ``P_inf = I`` construction
        cannot carry.  ``d`` is floored at zero here, so ``b = a c / d``
        divides by zero and the convolved term becomes non-finite.

        **This cannot be caught at call time.** An earlier version of this
        docstring claimed it was; it never was, and the consequence is not
        a NaN the caller sees but the likelihood's own non-finite guard
        returning ``-1e30`` with an exactly **zero gradient** -- flat in
        every direction, the failure mode that guard exists to warn about.
        So the check is at *construction*:
        :class:`~anvilgp.likelihood.GPLogLike` evaluates a convolved term at
        the parameter map's reference point and refuses a non-finite one.
        A sampler can still walk here, so a convolved ``SHOTerm`` needs a
        prior that keeps ``Q > 1/2``; :class:`RotationTerm` is bounded away
        from it by construction and has no such requirement.
        """
        sigma = p[..., 0:1]
        omega0 = _TWO_PI / p[..., 1:2]
        qual = p[..., 2:3]
        c = omega0 / (2.0 * qual)
        dsq = omega0 * omega0 - c * c
        d = mx.sqrt(mx.maximum(dsq, mx.zeros_like(dsq)))
        a = sigma * sigma
        empty = a[..., :0]
        return empty, empty, a, a * c / d, c, d

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

    def generator(self, p: mx.array):
        omega0 = math.sqrt(3.0) / p[..., 1:2]
        return [_sho_generator(
            omega0, mx.full(omega0.shape, 0.5, dtype=omega0.dtype))]

    def value_np(self, p, tau):
        sigma, rho = float(p[0]), float(p[1])
        x = math.sqrt(3.0) * np.abs(np.asarray(tau, dtype=np.float64)) / rho
        return sigma ** 2 * (1.0 + x) * np.exp(-x)


# ---------------------------------------------------------------------------
# The general complex celerite term, m = 2
# ---------------------------------------------------------------------------


@dataclass
class ComplexTerm(Term):
    """``k(tau) = e^{-c tau} (a cos(d tau) + b sin(d tau))``.  Two states.

    The general celerite-class term, matching
    ``celerite2.terms.ComplexTerm(a, b, c, d)``.  :class:`SHOTerm` is the
    special case ``b/a = c/d`` in a physical parametrisation; this is the raw
    form, and it is what exposure-time integration needs, because convolving
    a kernel with a boxcar keeps ``c`` and ``d`` and moves ``b/a``.

    Parameters (natural units): ``a``, ``b``, ``c``, ``d``.  ``a > 0`` is the
    variance at zero lag, ``c > 0`` the decay rate, ``d`` the angular
    frequency, ``b`` the sine amplitude.

    **Positive-definiteness is the caller's business, as in celerite2.**
    ``k`` is a valid covariance only for ``|b/a| <= c/d``; the state-space
    form's process noise is ``diag(2g, 2(2c-g))`` with ``g = c - b d / a``,
    whose positivity is that same condition.  Outside it the filter still
    runs -- it is linear algebra on the *observed* matrix, which can stay
    positive definite -- and :class:`TermConvolution` relies on that.  A
    sampler exploring ``|b/a| > c/d`` is asking for a non-GP and will get
    whatever the filter makes of it, so bound ``b`` if you sample it freely.
    """

    n_state: int = field(default=2, init=False)
    param_names: tuple[str, ...] = field(
        default=("a", "b", "c", "d"), init=False)

    def matrices(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        a, b = p[..., 0:1], p[..., 1:2]
        c, d = p[..., 2:3], p[..., 3:4]
        beta = b * d / a
        return _complex_matrices(mx.sqrt(a), c, beta, mx.sqrt(beta * beta
                                                              + d * d),
                                 c - beta, d * d, c * c + d * d, dt)

    def generator(self, p: mx.array):
        a, b = p[..., 0:1], p[..., 1:2]
        c, d = p[..., 2:3], p[..., 3:4]
        beta = b * d / a
        return [_pair_generator(c - beta, mx.sqrt(beta * beta + d * d), c)]

    def value_np(self, p, tau):
        a, b, c, d = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        t = np.abs(np.asarray(tau, dtype=np.float64))
        return np.exp(-c * t) * (a * np.cos(d * t) + b * np.sin(d * t))

    def celerite_coefficients(self, p: mx.array):
        a, b = p[..., 0:1], p[..., 1:2]
        c, d = p[..., 2:3], p[..., 3:4]
        empty = a[..., :0]
        return empty, empty, a, b, c, d


def from_celerite_coefficients(a_real=(), c_real=(), a=(), b=(), c=(), d=()):
    """Build a term and its parameter vector from celerite coefficients.

    The celerite decomposition is
    ``k(tau) = sum a_r e^{-c_r tau} + sum e^{-c tau}(a cos d tau + b sin d tau)``,
    and this turns one into a :class:`RealTerm` / :class:`ComplexTerm` sum.
    Returns ``(term, params)`` -- two objects, because a term here carries no
    parameters of its own; they are supplied per evaluation.

    **What this is for.** It is the interop route, and it replaces what the
    docs used to promise as an ``OriginalCeleriteTerm``.  celerite2's class of
    that name is *not* a reparametrisation of :class:`ComplexTerm` (an earlier
    version of `docs/kernels.md` said it was, and that was simply wrong): it
    is a thin wrapper that takes a term object from the **legacy ``celerite``
    package** and passes ``get_all_coefficients()`` straight through,
    unchanged.  So there is no new parametrisation to implement and no factor
    to convert -- the coefficients are already in this convention.  Anything
    either celerite version can express, feed here:

    >>> import celerite2.terms as c2t                       # doctest: +SKIP
    >>> co = c2t.SHOTerm(sigma=1.3, rho=4.0, Q=3.0).get_coefficients()
    >>> term, p = from_celerite_coefficients(*co)           # doctest: +SKIP

    **Not every coefficient set is representable**, and the refusals are the
    same ones the rest of the package documents rather than new limitations:

    * ``a_r > 0`` and ``a > 0``.  The ``P_inf = I`` construction carries the
      amplitude in ``h`` as its square root, so a negative amplitude has no
      representation -- the obstruction that keeps an overdamped ``SHOTerm``
      out of the *coefficient* route (its convolution goes through the
      generator instead, and has no such trouble).
    * ``b/a <= c/d``, i.e. ``k'(0+) = -1/2 h Q_c h^T <= 0``, which is exactly
      the condition for the implied process noise to be PSD.  Equality is the
      SHO.
    * ``d > 0`` for a complex pair.  ``d = 0`` is a repeated root, not a
      complex pair; it is a Matern-3/2 (or a doubled real term), and the
      coefficient form cannot express it -- use :class:`Matern32Term`.
    """
    ar = np.atleast_1d(np.asarray(a_real, dtype=np.float64)).ravel()
    cr = np.atleast_1d(np.asarray(c_real, dtype=np.float64)).ravel()
    ac = np.atleast_1d(np.asarray(a, dtype=np.float64)).ravel()
    bc = np.atleast_1d(np.asarray(b, dtype=np.float64)).ravel()
    cc = np.atleast_1d(np.asarray(c, dtype=np.float64)).ravel()
    dc = np.atleast_1d(np.asarray(d, dtype=np.float64)).ravel()
    if ar.size != cr.size:
        raise ValueError(f"a_real and c_real differ in length: "
                         f"{ar.size} vs {cr.size}")
    if not (ac.size == bc.size == cc.size == dc.size):
        raise ValueError(
            f"the complex coefficients differ in length: a={ac.size}, "
            f"b={bc.size}, c={cc.size}, d={dc.size}")
    if ar.size + ac.size == 0:
        raise ValueError("no coefficients given")

    terms: list[Term] = []
    params: list[float] = []
    for i in range(ar.size):
        if not ar[i] > 0.0:
            raise ValueError(
                f"real amplitude {i} is {ar[i]:g}; the P_inf = I construction "
                f"carries sqrt(a_r) in h, so a non-positive amplitude has no "
                f"representation here. The usual source is an SHO below "
                f"critical damping, whose two real amplitudes have opposite "
                f"signs -- use SHOTerm directly, which represents it as one "
                f"2-state block and is regular at every Q")
        if not cr[i] > 0.0:
            raise ValueError(f"real rate {i} is {cr[i]:g}, must be > 0")
        terms.append(RealTerm())
        params.extend([math.sqrt(float(ar[i])), float(cr[i])])
    for i in range(ac.size):
        if not ac[i] > 0.0:
            raise ValueError(
                f"complex amplitude a[{i}] is {ac[i]:g}, must be > 0")
        if not cc[i] > 0.0:
            raise ValueError(f"complex rate c[{i}] is {cc[i]:g}, must be > 0")
        if not dc[i] > 0.0:
            raise ValueError(
                f"complex frequency d[{i}] is {dc[i]:g}. A zero frequency is a "
                f"repeated root, not a complex pair -- that is Matern32Term, "
                f"which the coefficient form cannot express")
        # b/a <= c/d, with a relative tolerance because the SHO sits exactly
        # on the boundary and round-trips through float64 land just outside it
        if bc[i] * dc[i] > ac[i] * cc[i] * (1.0 + 1e-9):
            raise ValueError(
                f"pair {i} has b/a = {bc[i] / ac[i]:.6g} > c/d = "
                f"{cc[i] / dc[i]:.6g}, so k'(0+) > 0 and the implied process "
                f"noise is not PSD: this is not a valid covariance")
        terms.append(ComplexTerm())
        params.extend([float(ac[i]), float(bc[i]), float(cc[i]), float(dc[i])])

    term = terms[0] if len(terms) == 1 else TermSum(*terms)
    return term, np.array(params, dtype=np.float64)


# ---------------------------------------------------------------------------
# Exposure-time integration
# ---------------------------------------------------------------------------


#: Series terms in ``Y = (F delta)^2`` for the convolution's matrix
#: functions.  Both series have *all-positive* coefficients, so nothing
#: cancels inside them; the switch point is on ``||F delta||``, validated in
#: `tests/test_terms.py`.
_N_MATCONV = 20
_MATCONV_M = [2.0 / math.factorial(2 * j + 2) for j in range(_N_MATCONV)]
_MATCONV_D = [1.0 / math.factorial(2 * j + 3) for j in range(_N_MATCONV)]
#: Above this ``||F delta||`` the series loses float32 digits to the
#: alternating signs of ``X^k`` (the coefficients are positive, ``X`` is
#: not).  `GPLogLike` refuses beyond it rather than degrade silently.
MAX_FDELTA = 8.0


def _mat_horner(y, coeffs, eye):
    """``sum_j coeffs[j] Y^j`` for a batch of square matrices ``Y``."""
    acc = coeffs[-1] * eye
    for c in reversed(coeffs[:-1]):
        acc = acc @ y + c * eye
    return acc


def _conv_matrix_factors(f, delta):
    """``(M, D)`` for one block: the amplitude map and the zero-lag defect.

    With ``X = F delta``,

        M = 2 (cosh X - I) X^-2          rescales the amplitudes
        D = 2 (sinh X - X) X^-2 = W - M  the zero-lag defect,

    where ``W = 2 (e^X - I - X) X^-2`` is the *true* stationary covariance of
    the exposure-averaged state.  ``M`` is what the exponential sum sees at
    lags ``>= delta``; ``W`` is what zero lag sees; their difference is the
    diagonal correction, and in the scalar case ``D`` reduces to exactly the
    ``2a(z - sinh z)/z^2`` the coefficient route used.

    Both are **entire** functions of ``X``, evaluated as series in ``X^2``
    with positive coefficients.  That is the whole point of this route: a
    repeated or triple characteristic root is no obstacle to a matrix
    function, while the celerite coefficients it would need do not exist,
    and a negative-amplitude decomposition never has to be formed.
    """
    x = f * delta
    y = x @ x
    m = x.shape[-1]
    eye = mx.eye(m, dtype=x.dtype)
    amp = _mat_horner(y, _MATCONV_M, eye)
    defect = 2.0 * (x @ _mat_horner(y, _MATCONV_D, eye))
    return amp, defect


def _conv_p_inf(amp, h):
    """The convolved block's stationary covariance, ``(n, m, m)``.

    Only ``P h^T`` is observable: ``k_conv(tau) = h Phi(tau) (M h^T)``, so
    *any* symmetric ``P`` with ``P h^T = M h^T`` gives the same kernel.
    ``M`` itself is not symmetric (measured 3e-4 to 1e-2 relative asymmetry)
    and no change of basis can make it so, since it transforms as
    ``T M T^T``.  So build one:

        v = M h^T,   s = h M h^T,   lambda = s / h h^T,
        P = v v^T / s + lambda (I - h^T h / h h^T).

    The second term is ``lambda`` times the projector onto ``h``'s orthogonal
    complement, which ``h^T`` annihilates -- so it is free, and it is *not*
    optional.  ``v v^T / s`` alone satisfies the constraint and is PSD, but
    it is **rank one**, and the element algebra takes a Cholesky of exactly
    this matrix: `anvilgp.linalg.chol` then floors a singular pivot and the
    likelihood comes out ~1e-3 wrong, while the sequential filter -- which
    never factors it -- stays exact.  That asymmetry is what identified it.

    ``lambda = s / h h^T`` rather than any positive constant because it makes
    the construction *equivariant*: ``M = c I`` gives ``P = c I`` exactly, so
    ``delta -> 0`` recovers the package-wide ``P_inf = I`` rather than
    merely approaching it.
    """
    v = (amp @ h[..., None])[..., 0]                  # (n, m)
    sc = mx.sum(h * v, axis=-1, keepdims=True)        # (n, 1)
    hh = mx.sum(h * h, axis=-1, keepdims=True)        # (n, 1)
    eye = mx.eye(h.shape[-1], dtype=h.dtype)
    proj = eye - (h[..., :, None] * h[..., None, :]) / hh[..., None]
    return (v[..., :, None] * v[..., None, :] / sc[..., None]
            + (sc / hh)[..., None] * proj)


class TermConvolution(Term):
    """``term`` convolved with a boxcar of width ``delta`` -- exposure time.

    A detector integrating for ``delta`` does not sample the process, it
    samples its running mean, and for a 30-minute cadence against an hours
    signal that materially smooths the kernel.  Equivalent to
    ``celerite2.terms.TermConvolution``.

    **What the convolution does, and why it fits.** It leaves the *rates*
    untouched and rescales the *amplitudes*, and beyond the exposure width
    the convolved kernel is exactly a sum of exponentials with the new
    amplitudes (verified against celerite2 to 3e-16).  So it is a
    celerite-class term with the same state dimension -- but not the same
    *parametrisation*: the rescaling breaks the SHO's ``b/a = c/d``, which
    is why this needs :class:`ComplexTerm` and could not be written before
    it.

    **Below the exposure width it is not that sum**, and the difference at
    zero lag is a constant added to the variance -- :meth:`diag_correction`.
    That is the only piece that is not a kernel, and it is a function of
    the sampled parameters, so it carries a gradient.  Note it is
    **sign-indefinite**: positive for oscillatory components, negative for
    real ones (which *removes* variance).

    **Every spacing must be at least ``delta``** -- including zero, so
    duplicate timestamps are refused.  At a lag inside the exposure window
    the exponential sum is simply the wrong function there, and for a term
    with a real component the observed matrix goes indefinite: measured
    min eigenvalue -3.0e-2 at ``delta = 0.1``, which makes the filter's
    innovation variance negative and the likelihood NaN.  Physically the
    rule is automatic for one instrument, since you cannot sample faster
    than you expose; two instruments observing simultaneously with
    different exposure times is the same zero-spacing case and is equally
    refused.  :class:`~anvilgp.likelihood.GPLogLike` checks it at
    construction.

    **Limits.** ``term`` must have exact celerite coefficients, so
    :class:`Matern32Term` (a repeated root) and :class:`Matern52Term` (a
    triple one) cannot be wrapped -- their convolutions *are* state-space
    models of the same order, with the same rate and a
    polynomial-times-exponential form (measured to 1.6e-15), but reaching
    them needs the ``m = 3`` propagator generalised the way
    :func:`_complex_matrices` generalised ``m = 2``.  An underdamped
    ``SHOTerm`` is fine; below ``Q = 1/2`` it decomposes into two real
    exponentials one of which has a *negative* amplitude, which the
    ``P_inf = I`` construction cannot carry, and that is refused too.
    """

    def __init__(self, term: Term, delta: float):
        self.term = term
        self.delta = float(delta)
        if self.delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {self.delta}")
        # probe the wrapped term now, so an unsupported one fails here
        # rather than inside a sampler.  The generator, *not* the celerite
        # coefficients: this route is a matrix function of `F`, and needing
        # coefficients is exactly what excluded the Matern terms.
        with mx.stream(mx.cpu):
            term.generator(mx.zeros((1, term.n_params), dtype=mx.float64))
        self.n_state = term.n_state
        self.param_names = term.param_names

    @property
    def n_params(self) -> int:
        return self.term.n_params

    def _convolved(self, p: mx.array):
        """The rescaled coefficients and the diagonal correction.

        Closed form, transcribed from celerite2 and verified against it to
        2e-16 on the amplitudes and 1.7e-5 on the correction (where
        celerite2's own float64 route cancels).
        """
        ar, cr, a, b, c, d = self.term.celerite_coefficients(p)
        dl = self.delta
        corr = None

        if ar.shape[-1]:
            # (cosh z - 1)/z^2 and (sinh z - z)/z^2 come back already
            # divided, from series: written as a subtraction they lose every
            # digit as z -> 0, and dividing by z^2 afterwards cannot put
            # them back.  See `coshm1_sinhm_over_z2`.
            chm1, shm = coshm1_sinhm_over_z2(cr * dl)
            ar2 = 2.0 * ar * chm1
            corr = mx.sum(-2.0 * ar * shm, axis=-1, keepdims=True)
        else:
            ar2 = ar

        if a.shape[-1]:
            c2pd2 = c * c + d * d
            # (c-d)(c+d), not c^2 - d^2: the two are equal in exact
            # arithmetic and the product keeps its digits when c ~ d.
            cmd = (c - d) * (c + d)
            C1 = a * cmd + 2.0 * b * c * d
            C2 = b * cmd - 2.0 * a * c * d
            norm = (dl * c2pd2) ** 2
            # cosh(w) - 1 and sinh(w) - w for w = (c + i d) delta, from
            # series in w^2.  Both combinations below cancel to nothing as
            # delta -> 0 if written directly; `coshm1_sinhm_complex` also
            # uses `exp_acc`/`sincos`, since MLX's float64 exp/sin/cos are
            # only float32-accurate (a known trap here -- see `exp_acc`).
            (ct, st), (sr, si) = coshm1_sinhm_complex(c * dl, d * dl)
            a2 = 2.0 * (C1 * ct - C2 * st) / norm
            b2 = 2.0 * (C2 * ct + C1 * st) / norm
            # The correction's numerator is
            #   2 (C2 Im sinh w - C1 Re sinh w + (ac + bd) delta (c^2+d^2)),
            # and `C2 d - C1 c = -(c^2+d^2)(ac + bd)` *exactly*, so the
            # O(delta) part cancels against that last term and the truth is
            # O(delta^3).  Subtracting w inside `sinhm` does that
            # cancellation analytically; as written above it was done in
            # float32 and destroyed the correction -- the value stayed
            # plausible and the gradient came out 9% wrong.
            dc = mx.sum(2.0 * (C2 * si - C1 * sr) / norm,
                        axis=-1, keepdims=True)
            corr = dc if corr is None else corr + dc
        else:
            a2, b2 = a, b

        return ar2, cr, a2, b2, c, d, corr

    def matrices(self, p: mx.array, dt: mx.array):
        phis, qs, h = self.matrices_blocks(p, dt)
        return _block_diag(*phis), _block_diag(*qs), h

    def matrices_blocks(self, p: mx.array, dt: mx.array):
        """The base term's ``Phi`` and ``h``, with ``Q`` rebuilt on ``P_conv``.

        **The propagator does not change.** The convolution leaves the rates
        alone, so ``Phi`` comes straight from the wrapped term -- already
        verified, already regular at critical damping, already correct for a
        repeated root. Only the stationary covariance moves, from ``I`` to
        :func:`_conv_p_inf`, and ``Q(dt) = P - Phi P Phi^T`` follows.

        Nothing here decomposes the kernel into exponentials, which is why
        this route covers the Matern terms and the overdamped SHO that the
        celerite-coefficient route could not.
        """
        if dt.ndim == 1:
            dt = dt[None, :]
        phis, _, h = self.term.matrices_blocks(p, dt)
        qs = []
        for ph, pin in zip(phis, self.p_inf_blocks(p)):
            pin = pin[..., None, :, :]                 # (n, 1, m, m)
            qs.append(pin - ph @ pin @ mx.swapaxes(ph, -1, -2))
        return phis, qs, h

    def p_inf_blocks(self, p: mx.array, dtype=None):
        out = []
        for f, h in zip(self.term.generator(p), self._h_blocks(p)):
            amp, _ = _conv_matrix_factors(f, self.delta)
            out.append(_conv_p_inf(amp, h))
        return out

    def _h_blocks(self, p: mx.array):
        """``h`` split per diagonal block, matching :meth:`generator`."""
        h = self.term.matrices_blocks(
            p, mx.zeros((1,), dtype=p.dtype))[2]
        out, off = [], 0
        for k in self.term.block_sizes:
            out.append(h[..., off:off + k])
            off += k
        return out

    def celerite_coefficients(self, p: mx.array):
        ar, cr, a, b, c, d, _ = self._convolved(p)
        return ar, cr, a, b, c, d

    def diag_correction(self, p: mx.array):
        """``sum_b h_b (W_b - M_b) h_b^T`` -- the zero-lag defect.

        The state-space form of what the coefficient route computed term by
        term, and identical to it where both apply. Defined for every term
        with a generator, so it no longer needs celerite coefficients.
        """
        tot = None
        for f, h in zip(self.term.generator(p), self._h_blocks(p)):
            _, defect = _conv_matrix_factors(f, self.delta)
            part = mx.sum(h * (defect @ h[..., None])[..., 0],
                          axis=-1, keepdims=True)
            tot = part if tot is None else tot + part
        return tot

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return self.term.block_sizes

    def value_np(self, p, tau):
        """The *exact* convolved kernel, piecewise -- not the exponential sum.

        Three regimes, because the convolution is only an exponential sum
        outside the exposure width:

        * ``tau >= delta``: ``h Phi(tau) M h^T``, closed form and exact.
        * ``tau == 0``: ``h W h^T``, the true variance of the
          exposure-averaged state, as ``h M h^T`` plus the correction.
        * ``0 < tau < delta``: Gauss-Legendre quadrature of
          ``(1/delta^2) int (delta - |u|) k(|tau + u|) du``, panelled at the
          integrand's two corners (``u = -tau`` and ``u = 0``) so each panel
          is analytic.  **A likelihood never reaches this band** --
          :class:`~anvilgp.likelihood.GPLogLike` refuses any spacing below
          ``delta`` -- but an oracle asked for an arbitrary lag has to
          answer, and answering by the exponential sum there would be wrong.
        """
        pm = mx.array(np.asarray(p, dtype=np.float64)[None, :],
                      dtype=mx.float64)
        t = np.abs(np.asarray(tau, dtype=np.float64))
        out = np.zeros_like(t)
        with mx.stream(mx.cpu):
            amp = [np.asarray(a, dtype=np.float64)[0] for a in
                   (_conv_matrix_factors(f, self.delta)[0]
                    for f in self.term.generator(pm))]
            hb = [np.asarray(x, dtype=np.float64)[0]
                  for x in self._h_blocks(pm)]
            corr = float(np.asarray(self.diag_correction(pm))[0, 0])
            far = t >= self.delta
            if far.any():
                phis, _, _ = self.term.matrices_blocks(
                    pm, mx.array(t[far], dtype=mx.float64))
                blocks = [np.asarray(x, dtype=np.float64)[0] for x in phis]
                acc = np.zeros(int(far.sum()))
                for ph, mm, h in zip(blocks, amp, hb):
                    acc += np.einsum("i,kij,jl,l->k", h, ph, mm, h)
                out[far] = acc
        zero = t < 1e-300
        if zero.any():
            s_sum = sum(float(h @ mm @ h) for mm, h in zip(amp, hb))
            out[zero] = s_sum + corr
        mid = (~far) & (~zero)
        if mid.any():
            out[mid] = self._value_np_quadrature(p, t[mid])
        return out

    def _value_np_quadrature(self, p, t, n_gl: int = 48):
        """``(1/delta^2) int_{-d}^{d} (delta - |u|) k(|tau + u|) du``.

        Panelled at ``u = -tau`` and ``u = 0``, the only two points where the
        integrand is not analytic.  Used for ``0 < tau < delta`` alone.
        """
        x, w = np.polynomial.legendre.leggauss(n_gl)
        dl = self.delta
        out = np.empty_like(t)
        for i, tau in enumerate(t):
            total = 0.0
            for lo, hi in ((-dl, -tau), (-tau, 0.0), (0.0, dl)):
                if hi <= lo:
                    continue
                mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
                u = mid + half * x
                kv = np.asarray(self.term.value_np(
                    np.asarray(p, dtype=np.float64), np.abs(tau + u)),
                    dtype=np.float64)
                total += half * np.sum(w * (dl - np.abs(u)) * kv)
            out[i] = total / (dl * dl)
        return out


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

    Served by the fused Metal kernel like every other term with m <= 4: the
    kernel runs the composition with a dense m x m element and never
    exploited a 2x2 block structure, so m = 3 costs it nothing.
    """

    n_state: int = field(default=3, init=False)
    param_names: tuple[str, ...] = field(default=("sigma", "rho"), init=False)

    def matrices(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        sigma = p[..., 0:1]
        lam = math.sqrt(5.0) / p[..., 1:2]
        return _matern52_matrices(sigma, lam, dt)

    def generator(self, p: mx.array):
        """``lambda`` times a constant matrix -- the triple root, whitened.

        In the scaled frame the companion generator of ``(d/dt + lambda)^3``
        is ``lambda [[0,1,0],[0,0,1],[-1,-3,-3]]``; conjugating by
        ``_M52_LW`` puts it in the same ``P_inf = I`` basis
        :func:`_matern52_matrices` uses.
        """
        lam = math.sqrt(5.0) / p[..., 1:2]
        const = mx.array(_M52_GEN, dtype=lam.dtype)
        return [lam[..., None] * const]

    def value_np(self, p, tau):
        sigma, rho = float(p[0]), float(p[1])
        x = math.sqrt(5.0) * np.abs(np.asarray(tau, dtype=np.float64)) / rho
        return sigma ** 2 * (1.0 + x + x * x / 3.0) * np.exp(-x)


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
#: The whitened companion generator, divided by lambda.  `Matern52Term.
#: generator` scales it; verified against `logm(Phi(dt))/dt` to 9e-15.
_M52_GEN = _M52_LW_INV @ np.array([[0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [-1.0, -3.0, -3.0]]) @ _M52_LW
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
    mom = [lower_gamma_reg(n, z) * (math.factorial(n) / 2.0 ** (n + 1))
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

    def _modes(self, p: mx.array):
        """``(w1, w2, q1, q2)`` for the two SHO modes.

        One definition, three callers (:meth:`matrices_blocks`,
        :meth:`celerite_coefficients`, :meth:`generator`).  It was written
        out three times while the last two were added; the omega_0 formula
        is fiddly enough -- the damped period, not the undamped one, matches
        ``period`` -- that three copies were going to drift.
        """
        period = p[..., 1:2]
        q1 = 0.5 + p[..., 2:3] + p[..., 3:4]
        q2 = 0.5 + p[..., 2:3]
        # omega_0 chosen so the *damped* period matches: w0 eta = 2 pi / P.
        w1 = 4.0 * math.pi * q1 / (period * mx.sqrt(mx.maximum(
            (2.0 * q1 - 1.0) * (2.0 * q1 + 1.0), mx.zeros_like(q1) + 1e-30)))
        w2 = 8.0 * math.pi * q2 / (period * mx.sqrt(mx.maximum(
            (2.0 * q2 - 1.0) * (2.0 * q2 + 1.0), mx.zeros_like(q2) + 1e-30)))
        return w1, w2, q1, q2

    def _amps(self, p: mx.array):
        """``(amp, f * amp)`` -- the two modes' stationary variances."""
        sigma, f = p[..., 0:1], p[..., 4:5]
        amp = sigma * sigma / (1.0 + f)
        return amp, f * amp

    def matrices_blocks(self, p: mx.array, dt: mx.array):
        if dt.ndim == 1:
            dt = dt[None, :]
        w1, w2, q1, q2 = self._modes(p)
        a1, a2 = self._amps(p)
        s1 = mx.sqrt(a1)
        s2 = mx.sqrt(a2)

        phi1, qq1, h1 = _sho_matrices(s1, w1, q1, dt)
        phi2, qq2, h2 = _sho_matrices(s2, w2, q2, dt)
        return [phi1, phi2], [qq1, qq2], mx.concatenate([h1, h2], axis=-1)

    def generator(self, p: mx.array):
        w1, w2, q1, q2 = self._modes(p)
        return [_sho_generator(w1, q1), _sho_generator(w2, q2)]

    def celerite_coefficients(self, p: mx.array):
        """Two complex pairs -- so a rotation kernel *can* be convolved.

        Both modes have ``Q >= 1/2 + Q0 > 1/2`` whenever ``Q0, dQ > 0``, the
        regime the class documents and the priors sample, so neither pair
        can reach the critical-damping division a bare :class:`SHOTerm` can.
        That makes this the *safest* term to wrap in
        :class:`TermConvolution`, and the one the motivating case needs:
        long-cadence photometry of a rotating star.
        """
        w1, w2, q1, q2 = self._modes(p)
        out = []
        for amp_i, w, qual in zip(self._amps(p), (w1, w2), (q1, q2)):
            c = w / (2.0 * qual)
            d = mx.sqrt(mx.maximum(w * w - c * c, mx.zeros_like(c)))
            out.append((amp_i, amp_i * c / d, c, d))
        empty = p[..., 0:1][..., :0]
        return (empty, empty,
                mx.concatenate([o[0] for o in out], axis=-1),
                mx.concatenate([o[1] for o in out], axis=-1),
                mx.concatenate([o[2] for o in out], axis=-1),
                mx.concatenate([o[3] for o in out], axis=-1))

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

    def celerite_coefficients(self, p: mx.array):
        parts, off = [], 0
        for t in self.terms:
            k = t.n_params
            parts.append(t.celerite_coefficients(p[..., off:off + k]))
            off += k
        return tuple(mx.concatenate([q[i] for q in parts], axis=-1)
                     for i in range(6))

    def generator(self, p: mx.array):
        out, off = [], 0
        for t in self.terms:
            k = t.n_params
            out.extend(t.generator(p[..., off:off + k]))
            off += k
        return out

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
