"""Regression tests for MLX behaviours this package works around.

Each of these cost real debugging time and each is silent -- the forward
computation is correct and only something downstream is wrong.  They are
pinned here so that (a) an MLX upgrade that fixes one shows up as a failing
test rather than as dead workaround code, and (b) nobody rediscovers them.

Recorded against MLX 0.32.2 on Apple Silicon.
"""

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import (GPLogLike, GPPolicy, ParamMap, a64)

# float64 arrays cannot be created on the GPU stream, so the module default is
# the CPU; the device-specific checks below opt into the GPU explicitly.  Set
# here rather than relying on another test module having done it -- these
# tests must hold when this file is run on its own.


def test_mx_array_silently_downcasts_float64_numpy():
    """``mx.array(np_float64_array)`` returns **float32** unless the dtype is
    named.  Downstream arithmetic with a float64 operand promotes the result
    back to float64, so a "float64 path" can be carrying float32 data with
    every dtype it reports looking correct.  :func:`anvilgp.likelihood.a64`
    exists solely for this."""
    x = np.array([1.0 + 2 ** -30], dtype=np.float64)
    assert mx.array(x).dtype == mx.float32
    assert a64(x).dtype == mx.float64
    # and the value really is lost, not merely the label
    assert float(np.array(mx.array(x), dtype=np.float64)[0]) == 1.0
    assert float(np.array(a64(x))[0]) != 1.0


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_length_one_strided_slice_has_a_wrong_vjp(device):
    """``x[1:2:2]`` scatters its cotangent to *both* elements of a length-2
    axis: the gradient comes back as if the pair were tied together.

    Values are unaffected.  A binary reduction tree's last level is always
    ``k = 2``, so this silently corrupted every gradient the element algebra
    produced -- see :func:`anvilgp.elements._pairs` for the workaround
    (reshape to ``(..., k//2, 2, ...)`` and index, never stride).

    Step-2 slices of length >= 2, and step-1 slices, are correct.
    """
    dtype = mx.float32 if device == mx.gpu else mx.float64
    with mx.stream(device):
        def grad_of(n, s1, s2):
            x = mx.array(np.arange(1.0, n + 1), dtype=dtype)
            _, g = mx.vjp(lambda z: mx.sum(z[s1]) + 10.0 * mx.sum(z[s2]),
                          [x], [mx.array(1.0, dtype=dtype)])
            return np.array(g[0])

        bad = grad_of(2, slice(0, 2, 2), slice(1, 2, 2))
        assert np.allclose(bad, [1.0, 11.0]), (
            "MLX may have fixed the length-1 strided-slice VJP; if so, "
            "anvilgp.elements._pairs can go back to plain striding")
        assert np.allclose(grad_of(2, slice(0, 1), slice(1, 2)), [1.0, 10.0])
        assert np.allclose(grad_of(4, slice(0, 4, 2), slice(1, 4, 2)),
                           [1.0, 10.0, 1.0, 10.0])

        # Sharper characterisation, and the form filed upstream (see
        # docs/mlx-issue-strided-slice-vjp.md): the backward pass fills
        # start:stop rather than start:stop:step, so *any* single-element
        # strided slice smears its cotangent over the whole span.
        def single(n, sl):
            x = mx.array([float(i + 1) for i in range(n)], dtype=dtype)
            _, g = mx.vjp(lambda z: mx.sum(z[sl]), [x],
                          [mx.array(1.0, dtype=dtype)])
            return np.array(g[0])

        assert np.allclose(single(2, slice(0, 2, 2)), [1, 1])       # want 1,0
        assert np.allclose(single(3, slice(0, 3, 3)), [1, 1, 1])    # want 1,0,0
        assert np.allclose(single(5, slice(1, 5, 4)), [0, 1, 1, 1, 1])
        # two or more elements selected: correct
        assert np.allclose(single(6, slice(0, 6, 3)), [1, 0, 0, 1, 0, 0])


def test_reshape_and_index_has_the_correct_vjp():
    """The workaround itself, pinned."""
    for n in (2, 4, 8):
        x = mx.array(np.arange(1.0, n + 1), dtype=mx.float64)

        def f(z):
            r = mx.reshape(z, (n // 2, 2))
            return mx.sum(r[:, 0]) + 10.0 * mx.sum(r[:, 1])

        _, g = mx.vjp(f, [x], [mx.array(1.0, dtype=mx.float64)])
        assert np.allclose(np.array(g[0]),
                           np.tile([1.0, 10.0], n // 2))


def test_fp64_transcendentals_are_only_fp32_accurate():
    """``exp``/``sin``/``cos`` in float64 run a single-precision path;
    ``log``, ``expm1``, ``sqrt`` and arithmetic are true float64.  Anything
    in the verification path that needs a real float64 exponential has to
    write one (:func:`anvilgp.special.exp_acc`)."""
    x = a64(np.linspace(-20.0, 20.0, 501))
    ref = np.exp(np.array(x, dtype=np.float64))
    assert np.max(np.abs(np.array(mx.exp(x)) - ref) / ref) > 1e-9
    assert np.max(np.abs(np.array(mx.sin(x)) - np.sin(np.array(x)))) > 1e-9
    # the ones that are fine
    ref_l = np.log(np.abs(np.array(x)) + 1.0)
    assert np.max(np.abs(np.array(mx.log(mx.abs(x) + 1.0)) - ref_l)) < 1e-15
    z = a64(np.logspace(-12, 1, 200))          # strictly positive
    ref_e = np.expm1(np.array(z, dtype=np.float64))
    assert np.max(np.abs(np.array(mx.expm1(z)) - ref_e) / ref_e) < 1e-14


def test_linalg_is_cpu_only_and_has_no_vjp():
    """Why :mod:`anvilgp.linalg` hand-rolls a Cholesky at all."""
    a = a64(np.eye(3) * 2.0)[None]
    with mx.stream(mx.cpu):
        mx.eval(mx.linalg.cholesky(a, stream=mx.cpu))     # fine
    with pytest.raises(Exception):
        with mx.stream(mx.gpu):
            g = mx.array(np.eye(3, dtype=np.float32) * 2.0)[None]
            mx.eval(mx.linalg.cholesky(g, stream=mx.gpu))


def test_compiled_value_and_grad_exhausts_metal_argument_buffers():
    """``mx.compile`` over the *backward* pass of the whole likelihood
    fails with "Too many inputs/outputs fused in the Metal Compiled
    primitive".

    Structural, not size-dependent: it fails identically at N = 128 and
    N = 1000, and disappears at m = 1.  MLX fuses a long dependent
    elementwise chain into one Metal kernel whose argument count exceeds
    its limit of 24 arrays per fused group.  Bisected 2026-09-19: the
    element algebra is NOT the culprit -- chol, combine, combine_leaf and
    the whole leaf/fold/tree chain each compile under mx.vjp on their own.
    The offending group is in the front end, inside terms._sho_matrices.

    It matters because anvil's ChEES kernel compiles its leapfrog step, so
    the gradient sampler cannot run on a GP target without either
    ``mx.disable_compile()`` (verified to work, and what
    ``examples/`` documents).  M4's backward kernel was expected to moot
    this and did not: the over-fused group is in ``terms._sho_matrices``,
    which both paths share, so the kernel path aborts identically.

    Forward-only compilation is unaffected at every size and every m.
    """
    import anvilgp as ag
    from anvilgp.likelihood import (GPLogLike, GPPolicy, KernelParam,
                                    ParamMap)

    rng = np.random.default_rng(5)
    n = 256
    t = np.sort(rng.uniform(0.0, 20.0, n))
    y = rng.standard_normal(n) * 0.5
    yerr = np.full(n, 0.25)

    def build(term, refs):
        pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=r)
                              for i, r in enumerate(refs)], n_dim=len(refs))
        return GPLogLike(None, t, None, y, yerr, term, pm,
                         GPPolicy(block=256))

    def grad_compiles(gp, k):
        u = mx.zeros((8, k), dtype=mx.float32)
        try:
            mx.eval(mx.compile(lambda z: mx.vjp(
                gp, [z], [mx.ones((8,), dtype=mx.float32)])[1][0])(u))
            return True
        except Exception:
            return False

    with mx.stream(mx.gpu):
        # m = 1: the Cholesky is one sqrt, and it compiles
        assert grad_compiles(build(ag.RealTerm(), [1.0, 0.5]), 2)
        # m = 2: it does not
        assert not grad_compiles(build(ag.SHOTerm(), [1.0, 4.0, 3.0]), 3), (
            "MLX may have raised the fused-argument limit; if so, the "
            "mx.disable_compile() note in the docs can go")
        # forward-only is fine either way
        gp = build(ag.SHOTerm(), [1.0, 4.0, 3.0])
        mx.eval(mx.compile(gp.__call__)(mx.zeros((8, 3), dtype=mx.float32)))


def test_the_lag_gather_makes_the_gradient_slightly_non_deterministic():
    """A known, bounded MLX behaviour, pinned so it cannot grow unnoticed.

    On a regular cadence the lag table is reused through ``mx.take``, whose
    VJP is a scatter-add.  Atomic accumulation order is not fixed, so two
    evaluations of the *same* state give gradients that differ, and so do
    two lanes holding identical parameters.  Measured at ~1e-7 relative,
    i.e. float32 epsilon: far below the leapfrog integrator's own rounding,
    but it does mean the gradient is not a pure function of the state.

    With an all-distinct cadence the gather is switched off and the
    difference is exactly zero, which is what identifies the cause.
    """
    n, n_chains = 4000, 8
    y = np.random.default_rng(2).standard_normal(n)
    yerr = np.full(n, 0.3)
    u = mx.array(np.tile(np.array([[0.2, -0.1, 0.3]], dtype=np.float32),
                         (n_chains, 1)))
    ones = mx.ones((n_chains,), dtype=mx.float32)

    def spreads(t):
        gp = GPLogLike(None, t, None, y, yerr, ag.SHOTerm(),
                       ParamMap.log([1.0, 4.0, 3.0]), GPPolicy(block=1024))
        with mx.stream(mx.gpu):
            runs = []
            for _ in range(3):
                _, g = mx.vjp(gp, [u], [ones])
                mx.eval(g)
                runs.append(np.array(g[0]))
        scale = np.abs(runs[0]).max()
        lane = np.max(np.abs(runs[0] - runs[0][0])) / scale
        run = max(np.max(np.abs(r - runs[0])) for r in runs[1:]) / scale
        return gp.n_distinct_lags, lane, run

    lags, lane, run = spreads(np.arange(n) * 0.02)
    assert lags < 10, "regular cadence should reuse the lag table"
    assert lane < 1e-5 and run < 1e-5, (lane, run)

    lags, lane, run = spreads(
        np.sort(np.random.default_rng(1).uniform(0, 80, n)))
    assert lags == n, "all-distinct cadence should not reuse"
    assert lane == 0.0 and run == 0.0, "no gather, no non-determinism"
