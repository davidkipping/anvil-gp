"""MLX 0.32.2: mx.compile(mx.value_and_grad(f)) exceeds the fused-argument
limit for an f built from nested, both-branch-sanitised mx.where.

Self-contained.  Run with: python repro_hand.py
"""
import math
import mlx.core as mx

N = 12
C_SIN = [(-1.0) ** k / math.factorial(2 * k + 1) for k in range(N)]
C_SM1 = [(-1.0) ** k / math.factorial(2 * k + 3) for k in range(N)]
C_CM1 = [(-1.0) ** (k + 1) / math.factorial(2 * k + 2) for k in range(N)]
U = 2.0


def horner(u, c):
    out = mx.full(u.shape, c[-1], dtype=u.dtype)
    for a in reversed(c[:-1]):
        out = out * u + a
    return out


def safe(u, m, fill):
    """Sanitise the values the *other* branch would evaluate, so a masked
    inf/NaN never reaches a multiply-by-zero.  This is the pattern."""
    return mx.where(m, u, mx.full(u.shape, fill, dtype=u.dtype))


def sinc_gen(u):                                    # level 1: three-way
    sm = mx.abs(u) <= U
    ser = horner(safe(u, sm, 0.0), C_SIN)
    pos = u > U
    th = mx.sqrt(safe(u, pos, 1.0))
    trig = mx.sin(th) / th
    neg = u < -U
    ph = mx.sqrt(-safe(u, neg, -1.0))
    ep = mx.exp(ph)
    return mx.where(sm, ser, mx.where(pos, trig, 0.5 * (ep - 1.0 / ep) / ph))


def cosm1_gen(u):                                   # level 1: three-way
    sm = mx.abs(u) <= U
    ser = safe(u, sm, 0.0) * horner(safe(u, sm, 0.0), C_CM1)
    pos = u > U
    th = mx.sqrt(safe(u, pos, 1.0))
    neg = u < -U
    ph = mx.sqrt(-safe(u, neg, -1.0))
    ep = mx.exp(ph)
    return mx.where(sm, ser, mx.where(pos, mx.cos(th) - 1.0,
                                      0.5 * (ep + 1.0 / ep) - 1.0))


def sincm1(u):                                      # level 2: wraps level 1
    sm = mx.abs(u) <= U
    ser = horner(safe(u, sm, 0.0), C_SM1)
    big = safe(u, ~sm, 1.0)
    return mx.where(sm, ser, (1.0 - sinc_gen(big)) / big)


def f(v):
    """One SHO-propagator process-noise entry, which is where this bites in
    practice: a sum of terms each selecting between a series form (bounded
    arguments, no cancellation) and a difference-of-products form."""
    a = 0.4 * mx.exp(v[0:1])
    u = 1.3 * mx.exp(v[1:2]) - 1.0
    ee = mx.exp(-2.0 * a)
    a2 = a * a
    small = mx.abs(u) <= 0.5
    us = mx.where(small, u, mx.zeros_like(u))
    ub = safe(u, ~small, 1.0)
    es, ec = ee * sinc_gen(ub), ee * (1.0 + cosm1_gen(ub))
    ta = mx.where(small, ee * 8.0 * a * us * sincm1(4.0 * us),
                  2.0 * a * (ee - es * ec))
    tb = mx.where(small, ee * 2.0 * a2 * us * sincm1(us) * (1.0 + sinc_gen(us)),
                  2.0 * a2 * (ee - es * es))
    q11 = (-mx.expm1(-2.0 * a)) + ta + tb
    q22 = (-mx.expm1(-2.0 * a)) + 2.0 * a * es * ec - 2.0 * a2 * es * es
    return mx.sum(q11 + q22 + 2.0 * a * es * es)


vg = mx.value_and_grad(f)
v = mx.zeros((2,), dtype=mx.float32)
with mx.stream(mx.gpu):
    mx.eval(vg(v))
    print("eager value_and_grad:    ok")
    mx.eval(mx.compile(f)(v))
    print("compiled forward:        ok")
    mx.eval(mx.compile(vg)(v))
    print("compiled value_and_grad: ok   <-- did NOT reproduce")
