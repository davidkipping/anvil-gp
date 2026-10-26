"""Tiny fixed-size batched linear algebra for the scan-element combine.

The element algebra needs, per composition, a Cholesky factor of an m-by-m
positive-semidefinite matrix and one triangular solve, for m = 2..8 over a
batch of (chains x elements) that can reach 10^7.  MLX 0.32's
``mx.linalg.cholesky`` is **CPU-only and carries no VJP** (issues #1781,
#1238), so the differentiable float32 GPU path gets a hand-rolled,
statically unrolled factorisation instead: m is known at trace time, the
loops are Python loops over indices, and the result is a fixed-length graph
of elementwise ops -- no data-dependent control flow, ``mx.compile``-safe.

The float64 CPU verification path takes the built-in instead (one op rather
than ~90 graph nodes); it is value-only, so the missing VJP costs nothing
there, and both routes apply the *same* diagonal floor so they agree on
singular inputs.

Two invariants the callers rely on:

* ``chol`` floors each pivot at ``eps * trace(a)/m``.  ``C`` is genuinely
  singular in normal operation -- duplicate timestamps give ``dt = 0``,
  hence ``Phi = I`` and ``Q = 0`` exactly -- and the floor is what makes
  that a defined outcome rather than a NaN.  celerite2 accepts duplicate
  timestamps, so we must.  The floor is relative to the matrix's own scale,
  so it perturbs results at the rounding level, not at the problem level;
  this is only safe because :mod:`anvilgp.terms` hands over a
  *non-dimensionalised* state in which ``P_inf = I`` and no covariance
  carries an ``omega_0^2`` scale disparity.
* Products that are symmetric in exact arithmetic are *formed* symmetrically
  (``y @ y.T``, never ``L @ S_inv @ L.T``), so rounding cannot make a
  covariance asymmetric and seed a drift into indefiniteness.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["chol", "inv_logdet", "tri_solve_right", "sym_from_factor",
           "quad_form", "matvec", "matmul", "outer", "sym", "FORCE_UNROLLED"]


#: state dimension at or below which the unrolled elementwise product beats
#: ``mx.matmul``.  Measured on an M2 Max over batches of (256, 256) through
#: (4096, 256) tiny matrices, unrolled-vs-GEMM wall-time ratio (< 1 means
#: unrolled wins), both compiled:
#:
#:     m        2     4     6     8    10
#:     ratio  0.03  0.19  0.89  2.20  4.30      at batch (1024, 256)
#:
#: ``mx.matmul``'s cost is essentially *flat* in m -- it is all per-matrix
#: overhead at this size -- while the unrolled form is m^3 elementwise ops.
#: So the unrolled form is a 30x win at m = 2 and a 2x loss at m = 8.  The
#: crossover sat between 6 and 8 at every batch shape tried.
UNROLL_MAX_M = 6
#: The threshold is on the *work*, ``rows * inner * cols``, not on any single
#: dimension.  A block-diagonal propagator multiplies (2 x 2) blocks against
#: an 8-wide operand: 32 multiply-adds, obviously worth unrolling, even
#: though one dimension is 8.  Keying on ``max(dim)`` instead sent those to
#: ``mx.matmul`` and made the block-structured path 2.6x *slower* than the
#: dense one it was meant to beat.
UNROLL_MAX_WORK = UNROLL_MAX_M ** 3


def matmul(a: mx.array, b: mx.array) -> mx.array:
    """``a @ b`` for a large batch of tiny matrices, by whichever route wins.

    Below ``UNROLL_MAX_WORK`` multiply-adds the product is written out as
    elementwise operations over full-size planes, which is the shape MLX's
    fusion pass is good at; above it, ``mx.matmul``'s fixed per-matrix
    overhead finally becomes the cheaper of the two.

    Getting this right at m = 2 was worth more than every other optimisation
    in the reference path put together -- the reduction was 91% of the
    runtime and ~13 of these products per composition were nearly all of it.
    Getting the *fallback* right matters for multi-term kernels: four SHO
    terms is m = 8, where unrolling a dense product would cost 2.2x.
    """
    n_row, n_inner, n_col = a.shape[-2], a.shape[-1], b.shape[-1]
    if n_row * n_inner * n_col > UNROLL_MAX_WORK:
        return mx.matmul(a, b)
    rows = []
    for i in range(n_row):
        cols = []
        for j in range(n_col):
            acc = a[..., i, 0] * b[..., 0, j]
            for k in range(1, n_inner):
                acc = acc + a[..., i, k] * b[..., k, j]
            cols.append(acc)
        rows.append(mx.stack(cols, axis=-1))
    return mx.stack(rows, axis=-2)


def outer(u: mx.array, v: mx.array) -> mx.array:
    """``u v^T`` for batched vectors (..., m)."""
    return u[..., :, None] * v[..., None, :]


#: Set True to force the unrolled route even in float64.  Only the tests use
#: this, to check the two factorisation routes against each other in the same
#: precision -- otherwise a bug in the unrolled path would hide behind the
#: dtype switch and only surface on the GPU.
FORCE_UNROLLED = False


def prefer_builtin(x: mx.array) -> bool:
    """Whether to route through MLX's CPU LAPACK path for this array.

    True only for float64, which is by construction the CPU verification
    path: value-only, so the built-in's missing VJP is irrelevant.
    """
    return x.dtype == mx.float64 and not FORCE_UNROLLED


def diag_floor(a: mx.array) -> mx.array:
    """Scale-aware pivot floor ``eps * trace(a)/m``, shape ``a.shape[:-2]``.

    Additively offset by a few times the smallest normal so an all-zero
    matrix (dt = 0: duplicate timestamps) still factors.
    """
    m = a.shape[-1]
    tr = a[..., 0, 0]
    for i in range(1, m):
        tr = tr + a[..., i, i]
    fi = mx.finfo(a.dtype)
    return fi.eps * mx.maximum(tr / m, mx.zeros_like(tr)) + 4.0 * fi.smallest_normal


def chol(a: mx.array, floor: mx.array | None = None) -> mx.array:
    """Lower Cholesky factor of a batch of PSD matrices, shape (..., m, m)."""
    m = a.shape[-1]
    fl = diag_floor(a) if floor is None else floor

    if prefer_builtin(a):
        jit = fl[..., None, None] * mx.eye(m, dtype=a.dtype)
        return mx.linalg.cholesky(a + jit, upper=False, stream=mx.cpu)

    ent: list[list[mx.array]] = [[None] * m for _ in range(m)]  # type: ignore
    for j in range(m):
        d = a[..., j, j]
        for k in range(j):
            d = d - ent[j][k] * ent[j][k]
        ljj = mx.sqrt(mx.maximum(d, fl))
        ent[j][j] = ljj
        inv = 1.0 / ljj
        for i in range(j + 1, m):
            s = a[..., i, j]
            for k in range(j):
                s = s - ent[i][k] * ent[j][k]
            ent[i][j] = s * inv
    zero = mx.zeros_like(ent[0][0])
    rows = [mx.stack([ent[i][j] if j <= i else zero for j in range(m)], axis=-1)
            for i in range(m)]
    return mx.stack(rows, axis=-2)


def inv_logdet(x: mx.array) -> tuple[mx.array, mx.array]:
    """``(x^-1, log|det x|)`` for a batch of small *general* matrices.

    For the composition of a convolved term's elements, whose covariances
    are only formally Gaussian and can be indefinite, so :func:`chol` has
    nothing to factor (see :func:`anvilgp.elements.combine`).  Gauss-Jordan
    with partial pivoting, statically unrolled like :func:`chol`: the pivot
    row is chosen per batch element by comparison and moved by
    ``mx.where``, so there is no data-dependent control flow and the graph
    is ``mx.compile``-safe and differentiable.  The determinant's sign is
    not returned -- the caller needs only the magnitude (the product of
    every factor in the likelihood is positive, so the magnitudes carry the
    whole answer).
    """
    m = x.shape[-1]
    a = [[x[..., i, j] for j in range(m)] for i in range(m)]
    one, zero = mx.ones_like(a[0][0]), mx.zeros_like(a[0][0])
    g = [[one if i == j else zero for j in range(m)] for i in range(m)]
    logdet = zero
    for k in range(m):
        # choose the pivot row: largest |a[r][k]| over r >= k
        best = mx.zeros(a[0][0].shape, dtype=mx.int32) + k
        bval = mx.abs(a[k][k])
        for r in range(k + 1, m):
            better = mx.abs(a[r][k]) > bval
            best = mx.where(better, r, best)
            bval = mx.where(better, mx.abs(a[r][k]), bval)
        if k + 1 < m:
            rk_a, rk_g = list(a[k]), list(g[k])
            new_a, new_g = list(a[k]), list(g[k])
            for r in range(k + 1, m):
                sel = best == r
                new_a = [mx.where(sel, a[r][j], new_a[j]) for j in range(m)]
                new_g = [mx.where(sel, g[r][j], new_g[j]) for j in range(m)]
                a[r] = [mx.where(sel, rk_a[j], a[r][j]) for j in range(m)]
                g[r] = [mx.where(sel, rk_g[j], g[r][j]) for j in range(m)]
            a[k], g[k] = new_a, new_g
        piv = a[k][k]
        logdet = logdet + mx.log(mx.abs(piv))
        inv = 1.0 / piv
        a[k] = [v * inv for v in a[k]]
        g[k] = [v * inv for v in g[k]]
        for r in range(m):
            if r == k:
                continue
            f = a[r][k]
            a[r] = [a[r][j] - f * a[k][j] for j in range(m)]
            g[r] = [g[r][j] - f * g[k][j] for j in range(m)]
    inv_x = mx.stack([mx.stack(row, axis=-1) for row in g], axis=-2)
    return inv_x, logdet


def inv_logdet_fast(x: mx.array) -> tuple[mx.array, mx.array]:
    """:func:`inv_logdet` with an analytic VJP, and one Metal thread per
    matrix for float32 on the GPU (`metal.small_inv_logdet`).

    Differentiating the unrolled elimination is ``p^3`` more launches: 74
    ms for the gradient of 1024 10 x 10 inverses, measured 2026-10-26.  The
    VJP here is ``d log|x| = x^-T`` and ``d x^-1 = -x^-T (.) x^-T``, two
    batched products.  For the linear marginalisation's small Gram
    matrices (`GPLogLike._marginal`).
    """
    @mx.custom_function
    def f(x):
        from .metal import MAX_SMALL_INV, metal_available, small_inv_logdet
        if (x.dtype == mx.float32 and mx.default_device() == mx.gpu
                and x.shape[-1] <= MAX_SMALL_INV and metal_available()):
            return small_inv_logdet(x)
        return inv_logdet(x)

    @f.vjp
    def f_vjp(primals, cotangents, outputs):
        xit = mx.swapaxes(outputs[0], -1, -2)
        g_inv, g_ld = cotangents
        return (g_ld[..., None, None] * xit - xit @ g_inv @ xit,)

    inv, ld = f(x)
    return inv, ld


def tri_solve_right(lo: mx.array, b: mx.array) -> mx.array:
    """Solve ``x @ lo.T == b`` for x, with ``lo`` (..., m, m) lower triangular.

    Equivalently ``lo @ x.T == b.T``: forward substitution over the rows of
    ``b``.  Used to form ``Y = L1 @ chol(S)^{-T}`` so that ``M = Y @ Y.T``
    is symmetric by construction.
    """
    m = lo.shape[-1]
    if prefer_builtin(lo):
        bt = mx.swapaxes(b, -1, -2)
        y = mx.linalg.solve_triangular(lo, bt, upper=False, stream=mx.cpu)
        return mx.swapaxes(y, -1, -2)

    cols: list[mx.array] = []
    for i in range(m):
        acc = b[..., i]                       # (..., m), free index is b's row
        for j in range(i):
            acc = acc - lo[..., i, j][..., None] * cols[j]
        cols.append(acc / lo[..., i, i][..., None])
    return mx.stack(cols, axis=-1)


def sym_from_factor(y: mx.array) -> mx.array:
    """``y @ y.T`` -- a symmetric PSD matrix that is symmetric *exactly*."""
    m = y.shape[-1]
    if m > UNROLL_MAX_M:
        return mx.matmul(y, mx.swapaxes(y, -1, -2))
    rows = []
    for i in range(m):
        cols = []
        for j in range(m):
            acc = y[..., i, 0] * y[..., j, 0]
            for k in range(1, m):
                acc = acc + y[..., i, k] * y[..., j, k]
            cols.append(acc)
        rows.append(mx.stack(cols, axis=-1))
    return mx.stack(rows, axis=-2)


def matvec(a: mx.array, v: mx.array) -> mx.array:
    """``a @ v`` for batched (..., m, m) and (..., m), written out by index.

    Same reason as :func:`matmul`: the broadcast-and-reduce form materialises
    an (..., m, m) temporary per call, which at this batch size costs more
    than the m^2 multiply-adds it replaces.
    """
    m = a.shape[-1]
    cols = []
    for i in range(a.shape[-2]):
        acc = a[..., i, 0] * v[..., 0]
        for k in range(1, m):
            acc = acc + a[..., i, k] * v[..., k]
        cols.append(acc)
    return mx.stack(cols, axis=-1)


def quad_form(v: mx.array, a: mx.array) -> mx.array:
    """``v^T a v`` for batched vectors (..., m) and matrices (..., m, m)."""
    return mx.sum(v * matvec(a, v), axis=-1)


def sym(x: mx.array) -> mx.array:
    """Symmetrise the trailing two axes.

    Covariances and information matrices are symmetric in exact arithmetic;
    letting rounding break that lets a drift into indefiniteness accumulate
    over log2(N) composition levels.  It also matters in reverse: because
    the forward symmetrises, the adjoint of a symmetrised quantity is the
    *symmetrised* adjoint, which is the one thing that made dQ wrong when it
    was missed.
    """
    return 0.5 * (x + mx.swapaxes(x, -1, -2))
