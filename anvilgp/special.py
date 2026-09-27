"""Cancellation-free scalar special functions for the state-space GP algebra.

Every function here exists because the naive expression it replaces loses
significant digits somewhere in the (timescale / cadence / damping) plane
that a 4096-chain warmup *will* visit.  Three families:

**Generalised trigonometric functions of ``u = d^2 dt^2``.**  The SHO
propagator is built from ``cos(d dt)`` and ``sin(d dt)/d``.  Both are entire
functions of ``u`` alone -- they continue smoothly through ``u = 0``
(critical damping, ``Q = 1/2``) into ``u < 0`` (overdamped, where they
become ``cosh`` and ``sinh(.)/.``).  The celerite coefficient
parametrisation instead carries a ``1/sqrt(4Q^2-1)`` that is singular
exactly there; a sampler crossing ``Q = 1/2`` in float32 cannot tolerate
that.  ``sinc_gen``/``cosm1_gen``/``sincm1`` take ``u`` of either sign and
never divide by ``d``.

**Regularised lower incomplete gammas.**  The SHO process-noise matrix
contains ``1 - e^{-z}(1 + z + z^2/2)`` (= ``gamma3``), which cancels to
``z^3/6`` for small ``z``.  This is the ``P_inf - Phi P_inf Phi^T``
catastrophic cancellation the brief forbids, in its irreducible form; the
series branch removes it.

**Accurate fp64 ``exp``.**  MLX 0.32's float64 ``exp``/``sin``/``cos`` are
only *float32*-accurate (``log``, ``expm1``, ``sqrt``, arithmetic are true
fp64).  ``exp_acc`` and ``sincos`` provide genuinely fp64 versions for the
verification path, via Cody--Waite argument reduction plus exact powers of
two.  Note ``E = exp(-2 c dt)`` cannot be recovered as ``1 + expm1(-2 c dt)``:
at ``c dt = 10`` that form leaves ~8 significant digits.

All functions are dtype-polymorphic MLX ops: float32 on the GPU takes the
native transcendentals (native accuracy is exactly right there), float64
takes the reduced paths.  Every branch is an ``mx.where`` with *both* sides
sanitised, so no NaN can leak through a masked lane's gradient.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

__all__ = [
    "sinc_gen", "cosm1_gen", "cos_gen", "sincm1", "sincm1_series",
    "damped_cos_sinc",
    "gamma3", "lower_gamma_reg", "exp_acc", "sincos",
]

# ---------------------------------------------------------------------------
# series/closed-form switch points.  Chosen so the series is converged to
# below fp64 eps at the switch AND the closed form has lost < ~1.2 digits
# there; the overlap makes the switch invisible in either precision.
# ---------------------------------------------------------------------------

_U_SERIES = 2.0        # |d^2 dt^2| below which the trig series is used
_Z_SERIES = 2.0        # |2 c dt| below which the gamma series is used
_N_TRIG = 12           # series terms (converged to 4e-18 at |u| = 2)
_N_GAMMA = 24          # series terms (converged to 4e-18 at z = 2)
#: above this the regularised lower gamma is 1 to within a float32 ulp for
#: every n we use, and the direct form's polynomial would overflow.
_Z_DIRECT_SAT = 100.0


def _horner(z: mx.array, coeffs) -> mx.array:
    """Evaluate sum_k coeffs[k] z^k with Horner's rule (fp64-exact-order)."""
    acc = coeffs[-1] + 0.0 * z
    for c in reversed(coeffs[:-1]):
        acc = acc * z + c
    return acc


def _safe(x: mx.array, mask: mx.array, fill: float = 1.0) -> mx.array:
    """Replace masked-out lanes by ``fill`` *before* a hazardous op.

    Both-branch sanitisation: a clamped-tiny denominator gives a finite
    forward value but an overflowing -x/y^2 in the VJP, and 0 * inf = NaN
    in the backward pass (MetalPlanet learned this the hard way).
    """
    return mx.where(mask, x, mx.array(fill, dtype=x.dtype))


# ---------------------------------------------------------------------------
# generalised trigonometric functions of u = d^2 dt^2
# ---------------------------------------------------------------------------

# sinc_gen(u) = sum_k (-u)^k / (2k+1)!
_SINC_C = [(-1.0) ** k / math.factorial(2 * k + 1) for k in range(_N_TRIG)]
# cosm1_gen(u) = cos(sqrt u) - 1 = sum_{k>=1} (-u)^k / (2k)!  -- factored as
# u * sum_{k>=0} (-1)^{k+1} u^k / (2k+2)!
_COSM1_C = [(-1.0) ** (k + 1) / math.factorial(2 * k + 2) for k in range(_N_TRIG)]
# sincm1(u) = (1 - sinc_gen(u)) / u = sum_k (-1)^k u^k / (2k+3)!
_SINCM1_C = [(-1.0) ** k / math.factorial(2 * k + 3) for k in range(_N_TRIG)]


def sincm1_series(u: mx.array) -> mx.array:
    """:func:`sincm1` restricted to ``|u| <= _U_SERIES``, series branch only.

    **Why this exists, and it is not about speed.** ``sincm1`` selects
    between a series and a ``sinc_gen`` call, and ``sinc_gen`` is itself a
    three-way ``mx.where`` with each branch separately sanitised.  Nested
    both-branch evaluation like that exhausts MLX's 24-array limit for a
    single fused Metal kernel, so ``mx.compile`` over a graph containing
    ``_sho_matrices`` failed with "Too many inputs/outputs fused in the
    Metal Compiled primitive".  Since anvil's ChEES kernel compiles its
    leapfrog step, that is why a gradient sampler needed a process-global
    ``mx.disable_compile()``.

    The budget is spent **collectively**, which is worth knowing before
    blaming this function: ``sinc_gen`` alone compiles, and so does
    ``sincm1`` alone; only ``sincm1`` inside ``_sho_matrices``' surrounding
    expression crosses the line.  This is not the fault, it is the part
    that can be *removed*.

    Both of ``_sho_matrices``' calls are guarded by ``|u| <= 0.5`` and pass
    ``4u`` or ``u``, so ``|argument| <= 2 == _U_SERIES`` and the other
    branch is *unreachable* there.  This entry point drops it.  The result
    is bit-identical to :func:`sincm1` on the whole closed interval --
    pinned in ``tests/test_special.py`` -- so nothing about the numerics
    changes; only the fused-argument count does.

    :func:`sincm1` itself is unchanged and stays the general-domain
    function, still pinned against mpmath.
    """
    return _horner(u, _SINCM1_C)


def sinc_gen(u: mx.array) -> mx.array:
    """``sin(sqrt(u))/sqrt(u)`` for u > 0, ``sinh(sqrt(-u))/sqrt(-u)`` for u < 0.

    Entire in ``u``; equals 1 at u = 0.  This is ``S(dt)/dt`` for the SHO
    propagator, and is what keeps the propagator smooth through Q = 1/2.
    """
    small = mx.abs(u) <= _U_SERIES
    ser = _horner(_safe(u, small, 0.0), _SINC_C)

    pos = u > _U_SERIES
    up = _safe(u, pos, 1.0)
    th = mx.sqrt(up)
    trig = mx.sin(th) / th if u.dtype != mx.float64 else sincos(th)[0] / th

    neg = u < -_U_SERIES
    un = _safe(u, neg, -1.0)
    ph = mx.sqrt(-un)
    # sinh(ph)/ph via the accurate exponential; ph >= sqrt(2) here so the
    # e^ph - e^-ph difference carries no cancellation.
    ep = exp_acc(ph)
    hyp = 0.5 * (ep - 1.0 / ep) / ph

    out = mx.where(small, ser, mx.where(pos, trig, hyp))
    return out


def cos_gen(u: mx.array) -> mx.array:
    """``cos(sqrt u)`` for u > 0, ``cosh(sqrt(-u))`` for u < 0.  Entire in u."""
    return 1.0 + cosm1_gen(u)


def cosm1_gen(u: mx.array) -> mx.array:
    """``cos_gen(u) - 1``, computed without the 1 - cos cancellation.

    For u > 0 this is ``-2 sin^2(sqrt(u)/2)``; the naive ``cos(x) - 1``
    loses all significant digits as x -> 0, and that is precisely the
    dense-cadence regime (omega_0 dt << 1) the sampler spends its life in.
    """
    small = mx.abs(u) <= _U_SERIES
    ser = _safe(u, small, 0.0) * _horner(_safe(u, small, 0.0), _COSM1_C)

    pos = u > _U_SERIES
    up = _safe(u, pos, 1.0)
    half = 0.5 * mx.sqrt(up)
    s = mx.sin(half) if u.dtype != mx.float64 else sincos(half)[0]
    trig = -2.0 * s * s

    neg = u < -_U_SERIES
    un = _safe(u, neg, -1.0)
    halfp = 0.5 * mx.sqrt(-un)
    eh = exp_acc(halfp)
    sh = 0.5 * (eh - 1.0 / eh)
    hyp = 2.0 * sh * sh

    return mx.where(small, ser, mx.where(pos, trig, hyp))


def damped_cos_sinc(a: mx.array, u: mx.array,
                    lo: mx.array, hi: mx.array):
    """``(e^-a cos_gen(u), e^-a sinc_gen(u))`` without overflowing.

    The damped propagator only ever needs these two *products*, never their
    factors, and forming the factors separately is a trap: in the overdamped
    regime at long lag ``cosh(phi)`` overflows to inf while ``e^-a``
    underflows to 0, and ``inf * 0`` is NaN -- a whole chain lost to a
    quantity that is in fact bounded by 1.

    ``lo`` and ``hi`` are the two decay rates times dt, ``a -/+ phi`` with
    ``phi = sqrt(-u)``.  The caller supplies them because only it can form
    ``a - phi`` without cancellation: for the SHO that difference is
    ``omega_0^2 dt / (c + |d|)``, which stays accurate as ``Q -> 0`` where
    the literal subtraction loses ``log10(1/(2Q^2))`` digits.  They are
    ignored unless ``u < -_U_SERIES``, so they need only be finite elsewhere.
    """
    e = exp_acc(-a)

    small = mx.abs(u) <= _U_SERIES
    us = _safe(u, small, 0.0)
    ec_s = e * (1.0 + us * _horner(us, _COSM1_C))
    es_s = e * _horner(us, _SINC_C)

    pos = u > _U_SERIES
    th = mx.sqrt(_safe(u, pos, 1.0))
    sn_t, cs_t = sincos(th) if u.dtype == mx.float64 else (mx.sin(th),
                                                           mx.cos(th))
    ec_p, es_p = e * cs_t, e * sn_t / th

    neg = u < -_U_SERIES
    phi = _safe(0.5 * (hi - lo), neg, 1.0)
    # lo = a - phi is non-negative for any real overdamped SHO (|d| < c
    # whenever omega_0 > 0).  Clamping enforces that against rounding rather
    # than trusting it: a negative lo would make e^-lo overflow, and the
    # whole point of this routine is that its outputs are bounded by 1.
    elo, ehi = exp_acc(-mx.maximum(lo, mx.zeros_like(lo))), exp_acc(-hi)
    ec_n = 0.5 * (elo + ehi)
    es_n = 0.5 * (elo - ehi) / phi

    ec = mx.where(small, ec_s, mx.where(pos, ec_p, ec_n))
    es = mx.where(small, es_s, mx.where(pos, es_p, es_n))
    return ec, es


def sincm1(u: mx.array) -> mx.array:
    """``(1 - sinc_gen(u)) / u``, cancellation-free.  Equals 1/6 at u = 0.

    Appears twice in the SHO process-noise closed form; the direct
    expression loses ~log10(6/u) digits as u -> 0.
    """
    small = mx.abs(u) <= _U_SERIES
    ser = _horner(_safe(u, small, 0.0), _SINCM1_C)
    big = mx.logical_not(small)
    ub = _safe(u, big, 1.0)
    direct = (1.0 - sinc_gen(ub)) / ub
    return mx.where(small, ser, direct)


# ---------------------------------------------------------------------------
# regularised lower incomplete gamma:  gamma3(z) = 1 - e^{-z}(1 + z + z^2/2)
# ---------------------------------------------------------------------------

# P(n+1, z) = 1 - e^{-z} sum_{j<=n} z^j/j!
#           = z^{n+1} sum_k (-z)^k / (k! (n+1+k) n!)
_GAMMA_C: dict[int, list[float]] = {}


def _gamma_coeffs(n: int) -> list[float]:
    if n not in _GAMMA_C:
        _GAMMA_C[n] = [(-1.0) ** k
                       / (math.factorial(k) * (n + 1 + k) * math.factorial(n))
                       for k in range(_N_GAMMA)]
    return _GAMMA_C[n]


def lower_gamma_reg(n: int, z: mx.array) -> mx.array:
    """Regularised lower incomplete gamma ``P(n+1, z)``, cancellation-free.

    ``= 1 - e^{-z} sum_{j<=n} z^j/j!`` for ``z >= 0``, tending to
    ``z^{n+1}/(n+1)!`` as ``z -> 0``.  The series form is used below
    ``_Z_SERIES`` and the direct form above it.

    This is the one function standing between the process noise and a total
    loss of significance at ``lambda dt << 1``: the direct form *is* the
    ``P_inf - Phi P_inf Phi^T`` cancellation, which at ``c dt = 1e-3``
    leaves nothing at all in float32.  It is also bounded in [0, 1), so it
    removes the ``e^{2 c dt}`` overflow the unscaled ``expm1`` form would
    carry at large lag.  SHO needs ``n = 2``; Matern-5/2 needs up to 4.
    """
    small = z <= _Z_SERIES
    zs = _safe(z, small, 0.0)
    ser = _horner(zs, _gamma_coeffs(n))
    for _ in range(n + 1):
        ser = ser * zs
    # The direct branch forms `sum_j z^j/j!`, which overflows float32 at
    # z ~ 1e8 for n = 5 while `exp(-z)` has long since underflowed -- inf
    # times zero, so a NaN where the answer is simply 1.  P(n+1, z) is
    # within one float32 ulp of 1 for z >= 100 at every n we use (n <= 5:
    # the neglected tail is e^-100 100^5/120 ~ 3e-36), so clamp the
    # argument there.  Exact below the clamp, and it removes the NaN.
    big = z > _Z_DIRECT_SAT
    zb = _safe(mx.minimum(z, _Z_DIRECT_SAT), mx.logical_not(small), 1.0)
    tot, term = mx.ones_like(zb), mx.ones_like(zb)
    for j in range(1, n + 1):
        term = term * zb / j
        tot = tot + term
    direct = 1.0 - exp_acc(-zb) * tot
    direct = mx.where(big, mx.ones_like(direct), direct)
    return mx.where(small, ser, direct)


def gamma3(z: mx.array) -> mx.array:
    """``1 - e^{-z}(1 + z + z^2/2)`` for z >= 0; ``-> z^3/6`` as z -> 0.

    The ``n = 2`` case of :func:`lower_gamma_reg`, named because it is what
    the SHO process noise asks for and what the float32 viability of that
    process noise rests on.
    """
    return lower_gamma_reg(2, z)



# ---------------------------------------------------------------------------
# accurate fp64 exp / sin / cos
# ---------------------------------------------------------------------------

# fdlibm three-part ln2 and pi/2: the HI parts have enough trailing zero bits
# that n * HI is exact for the |n| we ever see.
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.4426950408889634

_PIO2_HI = 1.57079632673412561417e+00
_PIO2_MID = 6.07710050650619224932e-11
_PIO2_LO = 2.02226624879595063154e-21
_TWO_OVER_PI = 0.6366197723675814

_EXP_C = [1.0 / math.factorial(k) for k in range(16)]
_SIN_C = [(-1.0) ** k / math.factorial(2 * k + 1) for k in range(10)]
_COS_C = [(-1.0) ** k / math.factorial(2 * k) for k in range(10)]

# exact powers of two, as a table: 2^k is exactly representable, so a gather
# is exact where mx.power / exp2 are not guaranteed to be.
_POW2_LO, _POW2_HI = -1080, 1023
_pow2_table: dict = {}


def _pow2(k: mx.array) -> mx.array:
    """Exact 2**k for integer-valued float k, by table gather.

    Range-clamped rather than range-checked: k below the table underflows to
    the subnormal 2^-1080 = 0.0, which is the right answer for an exponential
    that has already underflowed, and k above it saturates instead of
    producing an inf that would poison a whole chain's gradient.
    """
    key = k.dtype
    if key not in _pow2_table:
        exps = np.arange(_POW2_LO, _POW2_HI + 1, dtype=np.int64)
        with np.errstate(over="ignore", under="ignore"):
            vals = np.ldexp(np.ones(exps.size), exps)
        # dtype= is mandatory: mx.array of a float64 numpy array silently
        # yields float32, which would clip this table at 2^+/-127 and make
        # exp_acc return 0/inf outside that -- see tests/test_mlx_gotchas.py.
        _pow2_table[key] = mx.array(vals, dtype=key)
    tab = _pow2_table[key]
    idx = mx.clip(k, float(_POW2_LO), float(_POW2_HI)) - float(_POW2_LO)
    return mx.take(tab, mx.stop_gradient(idx).astype(mx.int32))


def exp_acc(x: mx.array) -> mx.array:
    """``exp(x)`` at full working precision in both dtypes.

    float32 takes the native op (already correctly rounded on Metal);
    float64 takes Cody--Waite reduction ``x = k ln2 + r``, a 16-term Taylor
    series on ``|r| <= ln2/2``, and an exact table power of two.  MLX's own
    fp64 ``exp`` is only float32-accurate, which would silently cap the
    verification path at the precision it is supposed to be verifying.

    The gradient is correct by construction: ``k`` carries ``stop_gradient``
    and ``round`` has zero derivative a.e., so d/dx [2^k e^r] = exp(x).
    """
    if not isinstance(x, mx.array):
        x = mx.array(x)
    if x.dtype != mx.float64:
        return mx.exp(x)
    k = mx.stop_gradient(mx.round(x * _INV_LN2))
    r = (x - k * _LN2_HI) - k * _LN2_LO
    return _horner(r, _EXP_C) * _pow2(k)



def sincos(x: mx.array):
    """``(sin x, cos x)`` at full working precision in both dtypes.

    Same Cody--Waite treatment as :func:`exp_acc`; MLX's fp64 sin/cos are
    float32-accurate (independently rediscovered by MetalPlanet's trig.py,
    whose reduction constants this shares).
    """
    if x.dtype != mx.float64:
        return mx.sin(x), mx.cos(x)
    n = mx.round(x * _TWO_OVER_PI)
    y = ((x - n * _PIO2_HI) - n * _PIO2_MID) - n * _PIO2_LO
    z2 = y * y
    s = y * _horner(z2, _SIN_C)
    c = _horner(z2, _COS_C)
    q = n - 4.0 * mx.floor(n * 0.25)
    sin = mx.where(q == 0.0, s,
                   mx.where(q == 1.0, c, mx.where(q == 2.0, -s, -c)))
    cos = mx.where(q == 0.0, c,
                   mx.where(q == 1.0, -s, mx.where(q == 2.0, -c, s)))
    return sin, cos
