# MLX: nested both-branch `mx.where` exhausts the fused-argument limit under `mx.compile`

Status: written 2026-09-27, **not yet filed**. Companion to
[../mlx-issue-strided-slice-vjp.md](../mlx-issue-strided-slice-vjp.md),
also unfiled.

MLX 0.32.2, macOS 27.0 (Darwin), Apple M2 Max.

## Summary

`mx.compile(mx.value_and_grad(f))` fails with

```
[compile] Too many inputs/outputs fused in the Metal Compiled primitive
which exhausted the available argument buffers for the kernel.
```

for an `f` whose forward is a few dozen elementwise operations. The
*backward* of several nested branch selections — each branch sanitising its
own input so the unused arm cannot produce inf/NaN — fuses into one Metal
kernel with more than 24 array arguments.

Eager `value_and_grad` is fine. Compiling the **forward** alone is fine.
It is not size-dependent: the case below has two elements.

## Reproducer

Self-contained, no third-party imports. Reproduced on the machine above;
the reported kernel name is the `Broadcast/Negative/Multiply/Add/.../Sign`
chain quoted in the error.

```python
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
```

Output:

```
eager value_and_grad:    ok
compiled forward:        ok
Traceback (most recent call last):
  ...
RuntimeError: [compile] Too many inputs/outputs fused in the Metal Compiled
primitive which exhausted the available argument buffers for the kernel.
```

## What is and is not enough to trigger it

Worth stating precisely, because we got this wrong at first and it may
save a maintainer some bisection:

* `sinc_gen` alone (one three-way sanitised `where`): **compiles**.
* `sincm1` alone (that, wrapped in a second sanitised `where`):
  **compiles**.
* `sincm1` *inside the surrounding expression* in `f` above, whose other
  terms contribute their own branch selections: **fails**.

So no single function is the culprit. Each level is affordable and the
budget is consumed collectively; the last contributor to cross the line
looks like the offender but is not special. Removing any one of the nested
levels is enough to get back under the limit.

## What it cost us

anvil-gp evaluates Gaussian-process likelihoods for thousands of parameter
vectors on the GPU. The functions above are the cancellation-free special
functions that keep a state-space propagator smooth through critical
damping — the branch structure is there for numerical reasons, not
convenience. Because our sampler compiles its leapfrog step, this forced a
process-global `mx.disable_compile()` on every user who wanted a gradient
sampler, which also disabled compilation for everything else in their
process.

We worked around it: at the two call sites that matter the argument is
provably bounded by the series threshold, so one branch is unreachable and
we call a series-only entry point there. That depends on domain knowledge
about the arguments and is not generally available.

## What would help

In decreasing order of usefulness:

1. **Split a fused group when it would exceed the argument limit**, rather
   than failing. The graph is compilable; only the fusion is not.
2. **Name the offending subgraph.** The message says "please file an issue
   with the function that results in this error" and gives a mangled
   kernel name. Answering "which function" took a twenty-probe bisection,
   and the answer turned out to be "no single one" (above).
3. **Document the limit and the pattern that reaches it.** Nested
   sanitised `mx.where` is a natural way to write a branchy special
   function and nothing warns against it.

## One correction, in case you find our earlier notes

Our own documentation described this for months as an *uncatchable process
abort*. It is not: it is an ordinary Python `RuntimeError` and the process
continues. Separately, `mx.eval` *inside* a compiled function raises a
different and clearly-worded error, so eval placement was never involved.
