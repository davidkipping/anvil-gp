"""The affine-Gaussian scan-element algebra, and its reduction to log L.

An *element* is what one contiguous stretch of the time series does to the
state that enters it:

    L(x_out, x_in) = N(x_out; A x_in + b, C)
                     * exp(-1/2 x_in^T J x_in + eta^T x_in + ell)

-- an affine-Gaussian transition, times the (unnormalised) likelihood of
the observations in the stretch, expressed as a function of the state that
entered it.  Composition of adjacent stretches is the Gaussian integral
``L(x_out, x_in) = int L2(x_out, z) L1(z, x_in) dz``, which is associative;
:func:`combine` is that integral in closed form.  This is the
Sarkka & Garcia-Fernandez (1905.13002) parallel-filter element with a sixth
component, the log-normaliser ``ell``, carried along.

**A reduction, not a scan.**  The marginal likelihood needs only the root
of a binary reduction tree, never the prefix at every leaf.  The very first
element has ``A = 0``, ``J = 0``, ``eta = 0`` (its "input state" is the
stationary prior, already absorbed), so those stay zero all the way up and
the composite of the whole series is ``N(x_N; b, C) exp(ell)``; integrating
out ``x_N`` leaves ``exp(ell)``.  **log L is the root element's ell.**  That
halves the working set relative to a prefix scan and, just as usefully,
makes the float32 accumulation of both the log-determinant and the
quadratic form a pairwise tree -- the friendliest summation order there is.

**One algebra, two executors.**  This module is the dtype-polymorphic MLX
reference: run it float64-on-CPU and it is the verification path, float32-
on-GPU and it is the parity baseline for the fused Metal kernel, which
executes the *same* composition on the same block boundaries.

**Blocked.**  Leaves are built and reduced one data block at a time, with
the per-block roots composed by a pairwise tree (a left fold made the block
size a float32 precision knob; see ``GPLogLike._evaluate``).  The block loop bounds peak
memory (an element is ``3m^2 + 2m + 1`` numbers, so 1024 chains x 4096
points x m=4 is already ~1 GB in fp32) and is exactly the structure the
Metal fast path uses -- pass 1 condenses a block, pass 2 composes the block
elements.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from .linalg import (chol, inv_logdet, matmul, matvec, outer, quad_form, sym,
                     sym_from_factor, tri_solve_right)

__all__ = ["Element", "LeafRank1", "LeafParts", "first_point",
           "leaf_elements", "leaf_parts",
           "expand_leaf", "combine", "combine_leaf", "combine_leaf_blocked",
           "tree_reduce", "chunked_reduce", "suffix_scan", "chunked_reduce_blocked",
           "prefix_scan",
           "identity_element", "use_blocked_leaves"]


class Element(NamedTuple):
    """One affine-Gaussian scan element with its log-normaliser.

    Batch dims lead: ``A``/``C``/``J`` are ``(..., m, m)``, ``b``/``eta``
    are ``(..., m)``, ``ell`` is ``(...)``.
    """

    A: mx.array
    b: mx.array
    C: mx.array
    eta: mx.array
    J: mx.array
    ell: mx.array

    @property
    def m(self) -> int:
        return self.A.shape[-1]


class LeafRank1(NamedTuple):
    """The rank-one factors of a leaf: ``J = z z^T``, ``eta = g z``.

    ``z = (h Phi)^T / sqrt(S)`` and ``g = y / sqrt(S)``.  Carrying these
    alongside the assembled element is what lets :func:`combine_leaf` skip
    both Cholesky factorisations.

    ``ld`` is the leaf's log-determinant term ``-1/2 log(S / r_ref)``, i.e.
    its ``ell`` *without* the ``-g^2/2``.  It is carried separately because
    ``combine_leaf`` needs exactly that: the assembled ``ell`` is
    ``ld - g^2/2`` and the composition immediately adds ``+g^2/2`` back, so
    forming either one costs a cancellation that ``g^2`` makes expensive.
    At sigma_GP/yerr = 100 on-model, ``g^2 ~ 1e4`` against an ``ld`` of
    order 1, so the round trip through the sum loses four digits at every
    observation.  The dense ``Element.ell`` keeps the full value, because
    the general :func:`combine` (and hence the pure-tree path) needs it.
    """

    z: mx.array
    g: mx.array
    ld: mx.array


class LeafParts(NamedTuple):
    """A leaf in compact form -- the pieces, not the assembled matrices.

    Every component of the dense :class:`Element` is recoverable::

        A = Phi - kappa z^T        b   = kappa g
        C = Q   - kappa kappa^T    eta = g z
        J = z z^T

    with ``kappa = Q h^T / sqrt(S)``, ``z = (h Phi)^T / sqrt(S)`` and
    ``g = y / sqrt(S)``.  Storing the pieces costs ``2 sum m_i^2 + 2m + 2``
    numbers against the dense element's ``3m^2 + 2m + 1``: 14 against 17 at
    m = 2, 2.2x smaller at m = 4, 4.2x smaller at m = 8.  The leaf array is
    the largest object in the whole evaluation (1024 chains x 4096 points at
    m = 8 is 3.5 GB dense), so this is the dominant memory term.
    """

    phi_blocks: list
    q_blocks: list
    kappa: mx.array
    z: mx.array
    g: mx.array
    ell: mx.array
    #: the log-determinant half of ``ell``; see :class:`LeafRank1`.
    ld: mx.array

    @property
    def sizes(self):
        return tuple(b.shape[-1] for b in self.phi_blocks)


def use_blocked_leaves(block_sizes) -> bool:
    """Whether the compact block-structured leaf beats the dense one.

    Measured end-to-end, compiled, A/B in one process on a rested machine at
    1024 chains x N = 1e4, dense versus blocked:

        m              2       4       6       8      10
        blocked is  1.01x   0.80x   1.16x   1.11x   1.44x  faster
        peak dense  1.16    2.48    4.57    7.67   11.93   GiB
        peak blkd   1.01    1.58    1.89    2.03    2.36   GiB

    Two things to read off this.  The **speed** effect is modest and
    thermally sensitive -- the same A/B on a heat-soaked machine put m = 8 at
    2.36x -- so it decides the default but nothing rests on it.  The
    **memory** effect is large, monotone and reproducible: 5x at m = 10,
    where the dense leaf array alone is 5 GiB (~12 GiB peak with the
    composition's intermediates).  At many-chain sizes that is
    the difference between running and not.

    Note also that an *eager* microbenchmark of a single composition ranks
    these the other way round.  The compact form's benefit is memory traffic
    on the (chains x points) leaf array, which only shows up in the compiled
    whole-pipeline measurement; per-composition timings see only the extra
    concatenations.  Decide this kind of thing on the real pipeline.

    Default: at least two blocks and m >= 6.  Set ``GPPolicy.blocked=True``
    explicitly at m = 4 if peak memory rather than wall time is what binds.
    """
    return len(block_sizes) >= 2 and sum(block_sizes) >= 6


def identity_element(batch: tuple[int, ...], m: int, dtype) -> Element:
    """The composition identity: pass the state through, observe nothing.

    Used to pad a block up to a power of two so the number of tree levels
    is a trace-time constant and ``mx.compile`` never retraces on a ragged
    final block.
    """
    z2 = mx.zeros(batch + (m, m), dtype=dtype)
    z1 = mx.zeros(batch + (m,), dtype=dtype)
    return Element(
        A=mx.broadcast_to(mx.eye(m, dtype=dtype), batch + (m, m)),
        b=z1, C=z2, eta=z1, J=z2,
        ell=mx.zeros(batch, dtype=dtype),
    )


def first_point(phi, q, dtype=None, p_inf=None):
    """Impose "the first observation has no predecessor" on a block.

    Sets ``Phi = 0`` and ``Q = P_inf`` at index 0 along the time axis, where
    ``P_inf`` is the identity unless ``p_inf`` says otherwise.  It says
    otherwise for exactly one term: a :class:`~anvilgp.terms.TermConvolution`
    is *not* non-dimensionalised to ``P_inf = I`` -- the convolution changes
    the stationary covariance and no change of basis restores the identity --
    so it supplies its own, one array per block (or one dense array when the
    leaves are not blocked).
    The generic leaf then produces ``A = 0``, ``J = 0``, ``eta = 0`` -- the
    "no incoming state" element -- so nothing downstream needs a special
    case for the start of the series.  Accepts either a dense ``(n, K, m,
    m)`` pair or the per-block lists of the compact representation.

    The Metal kernel expresses the same fact differently, as row 0 of its
    lag table; both are the same convention, written for their own data
    layout.
    """
    blocked = isinstance(phi, (list, tuple))
    ref = phi[0] if blocked else phi
    dtype = ref.dtype if dtype is None else dtype
    k = ref.shape[1]
    keep = mx.concatenate([
        mx.zeros((1, 1, 1, 1), dtype=dtype),
        mx.ones((1, k - 1, 1, 1), dtype=dtype)], axis=1)

    def fix(p_b, q_b, pin):
        pin = (mx.eye(q_b.shape[-1], dtype=dtype) if pin is None
               else pin[..., None, :, :].astype(dtype))
        return p_b * keep, q_b * keep + (1.0 - keep) * pin

    if blocked:
        pins = [None] * len(phi) if p_inf is None else list(p_inf)
        pairs = [fix(a, b, c) for a, b, c in zip(phi, q, pins)]
        return [a for a, _ in pairs], [b for _, b in pairs]
    return fix(phi, q, p_inf)


def leaf_elements(phi: mx.array, q: mx.array, h: mx.array, y: mx.array,
                  r_var: mx.array, r_ref: mx.array,
                  s_floor: mx.array | None = None, formal: bool = False
                  ) -> tuple[Element, "LeafRank1"]:
    """One element per observation, from the term matrices and the data.

    Returns ``(element, rank1)``: the dense element, and the rank-one pieces
    ``(kappa, z, g)`` that :func:`combine_leaf` needs to skip both
    Choleskys.  Callers that only want the element take ``[0]``.

    ``phi``, ``q``: (n, K, m, m).  ``h``: (n, m).  ``y``: (n, K) residuals
    (data minus mean model).  ``r_var``: (n, K) observation variance
    ``yerr^2 + jitter^2`` -- ``yerr`` alone is *not* the diagonal once a
    jitter is being sampled.  ``r_ref``: (K,) the float64 host reference
    variance the log-determinant is measured against.

    The construction is the Kalman update written as a map on the *incoming*
    state: predict through ``phi``, add ``q``, absorb ``y``.  For the first
    point of the series pass ``phi = 0`` and ``q = P_inf = I``; then
    ``A = 0`` and the element correctly carries no dependence on a state
    that does not exist.

    Reference-subtracted log-determinant
    ------------------------------------
    ``ell`` accumulates ``-1/2 log(S / r_ref)``, an O(0.01-5) quantity,
    *not* ``-1/2 log(2 pi S)``.  At N = 1e5 and 500 ppm errors
    ``log|K| ~ -1.5e6``; accumulating that in float32 loses ~20 log-
    likelihood units, twenty times the scale a Metropolis decision runs on.
    The parameter-independent remainder ``-1/2 sum log(2 pi r_ref)`` is a
    float64 host constant added outside the graph.  This is anvil's offset
    discipline applied to the log-determinant, and it is why ``r_ref`` is
    the *fixed* ``yerr^2`` rather than the jitter-dependent ``r_var``: the
    jitter's contribution stays inside the well-conditioned ratio.

    ``formal`` is for a :class:`~anvilgp.terms.TermConvolution`; see the
    comment at the clamp below.
    """
    # Written as per-plane multiply-adds rather than `mx.sum` over a
    # broadcast product.  The reduction kernels over a size-m axis cost
    # 10.5 ms per block against ~5 ms for the unrolled form; measured
    # end to end at 1024 chains x N = 1e4, compiled, in one window:
    # forward 112.3 -> 82.9 ms (min of 15), gradient unchanged at ~1039 ms,
    # peak memory 1.44 -> 1.34 GiB forward and 5.44 -> 5.27 GiB with the
    # gradient.  `m` is a trace-time constant, so the loop unrolls into the
    # graph and nothing here is dynamic.
    _m = h.shape[-1]
    hphi = sum(h[:, None, i, None] * phi[:, :, i, :] for i in range(_m))
    qh = sum(q[:, :, :, j] * h[:, None, None, j] for j in range(_m))
    # `h Q h^T` is a variance, hence non-negative in exact arithmetic, and
    # float32 can round it slightly negative in the dense-cadence corner.
    # Clamp *it*, not the sum: `r_var` beside it is exact and strictly
    # positive, so clamping the term that can go non-physical leaves
    # `s_q >= r_var > 0` and the division is always safe.  Clamping the
    # total instead (the old `max(s_q, 1e-6 r_ref)`) allowed a negative
    # residue to survive inside the sum, and -- because the sequential
    # filter's own total is a *different* quantity, `h (Phi P Phi^T + Q) h^T
    # + r_var` -- made the two forms clamp different things and disagree by
    # ~0.1 log-units wherever it bound.  See docs/m4-plan.md, stage 0.
    #
    # `formal`: NOT for a `TermConvolution`.  Its `Q` is only formally a
    # covariance -- the convolved exponential sum is not a positive-definite
    # function for Matern-3/2, Matern-5/2 or a low-Q SHO, so no PSD `Q`
    # exists -- and `h Q h^T` is genuinely negative there, not by rounding.
    # The algebra is still exact with the negative value; the clamp is what
    # breaks it.  Measured on a convolved Matern-5/2 at rho = 10-50
    # exposures: 0.004 to 0.46 log-units wrong at N = 300 with the clamp,
    # 1e-11 without (the "7e-7" this comment used to quote came from a grid
    # with large error bars, where the clamp barely binds).  So the formal
    # path takes the sum as it is.  If the *sum* is not positive the
    # representation has genuinely failed, and the log and rsqrt below turn
    # that into a NaN the likelihood refuses.
    hqh = sum(h[:, None, i] * qh[:, :, i] for i in range(_m))
    if formal:
        s_q = hqh + r_var
    else:
        s_q = mx.maximum(hqh, mx.zeros_like(hqh)) + r_var
        if s_floor is not None:
            # secondary guard, kept for tests; unreachable in production
            # since s_q >= r_var >= r_ref and s_floor is a small fraction
            # of r_ref.
            s_q = mx.maximum(s_q, s_floor)

    inv_s = 1.0 / s_q
    inv_rs = mx.rsqrt(s_q)
    k = qh * inv_s[..., None]                              # gain, (n, K, m)

    a_mat = phi - outer(k, hphi)
    c_mat = sym(q - outer(k, qh))
    # A leaf's information matrix is rank one -- J = z z^T with
    # z = (h Phi)^T / sqrt(S) -- and its information vector is g z with
    # g = y / sqrt(S).  Nothing downstream *needs* that structure, but
    # exploiting it removes both Choleskys from a composition; see
    # combine_leaf.  The factors are returned alongside so the caller can
    # choose, and the assembled J/eta stay available for the tree path.
    z = hphi * inv_rs[..., None]
    g = y * inv_rs
    j_mat = outer(z, z)
    eta = g[..., None] * z
    b = k * y[..., None]

    ld = -0.5 * mx.log(s_q / r_ref)
    ell = ld - 0.5 * g * g
    return Element(A=a_mat, b=b, C=c_mat, eta=eta, J=j_mat, ell=ell), \
        LeafRank1(z=z, g=g, ld=ld)


def combine(e1: Element, e2: Element, formal: bool = False) -> Element:
    """Compose two adjacent elements, ``e1`` earlier than ``e2``.

    The Gaussian integral ``int L2(x_out, z) L1(z, x_in) dz`` in closed
    form.  Every inverse is routed through two Cholesky factorisations
    rather than an explicit inverse, because MLX has no GPU ``inv`` and
    because the factored route keeps the symmetric objects exactly
    symmetric:

        L1 = chol(C1)
        S  = I + L1^T J2 L1        (SPD, eigenvalues >= 1 -- no pivoting)
        Y  = L1 chol(S)^{-T}
        M  = Y Y^T = (I + C1 J2)^{-1} C1
        G  = I - M J2 = (I + C1 J2)^{-1}
        Jt = J2 - (J2 Y)(J2 Y)^T = (I + J2 C1)^{-1} J2

    ``S`` is positive definite with every eigenvalue at least 1 (those of
    ``L1^T J2 L1`` are non-negative), so its Cholesky needs no pivoting and
    no floor beyond the rounding guard -- the only genuinely singular
    matrix in the recursion is ``C1``, which :func:`anvilgp.linalg.chol`
    floors on the way in.

    ``log|I + C1 J2| = log|S| = 2 sum log diag(chol(S))`` by Sylvester's
    identity: the log-determinant is read straight off a factor that has
    already been computed, never from a separate determinant.

    ``formal`` is for a :class:`~anvilgp.terms.TermConvolution`, whose
    state-space form is only formally Gaussian: ``C1`` and ``J2`` can be
    indefinite, so there may be no Cholesky factor to take, and flooring one
    substituted a different matrix (up to 500 log-units wrong).  The same
    three matrices come instead from ``(I + C1 J2)^-1`` by a pivoted direct
    inverse, with ``log|I + C1 J2|`` off its pivots.  Only magnitudes enter:
    individual determinants (and :func:`combine_leaf`'s ``alpha``) can be
    negative, but their product is the data's covariance determinant, which
    is positive, so the magnitudes carry the whole answer.  Exact wherever
    the inverse exists; in float32 it matched the Cholesky route to every
    digit printed on points where both apply.
    """
    m = e1.m
    dtype = e1.A.dtype
    ident = mx.eye(m, dtype=dtype)

    if formal:
        g, logdet_x = inv_logdet(ident + matmul(e1.C, e2.J))
        m_mat = sym(matmul(g, e1.C))                # (I + C1 J2)^-1 C1
        jt = sym(matmul(e2.J, g))                   # (I + J2 C1)^-1 J2
        log_det = 0.5 * logdet_x
    else:
        l1 = chol(e1.C)
        x = matmul(mx.swapaxes(l1, -1, -2), matmul(e2.J, l1))
        s = ident + sym(x)
        ls = chol(s)
        y = tri_solve_right(ls, l1)                 # Y Ls^T = L1
        m_mat = sym_from_factor(y)                  # (I + C1 J2)^-1 C1
        g = ident - matmul(m_mat, e2.J)             # (I + C1 J2)^-1
        v = matmul(e2.J, y)
        jt = sym(e2.J - sym_from_factor(v))        # (I + J2 C1)^-1 J2
        log_det = mx.log(ls[..., 0, 0])
        for i in range(1, m):
            log_det = log_det + mx.log(ls[..., i, i])

    a = matmul(e2.A, matmul(g, e1.A))
    g_b1 = matvec(g, e1.b)                         # G b1 -- the ell term
    # b carries the extra M eta2 that the log-normaliser must NOT: the
    # composite mean is A2 (G b1 + M eta2) + b2, while the ell cross term is
    # eta2^T G b1 alone.  Folding them together double-counts eta2^T M eta2.
    b = matvec(e2.A, g_b1 + matvec(m_mat, e2.eta)) + e2.b
    c = sym(matmul(e2.A, matmul(m_mat, mx.swapaxes(e2.A, -1, -2)))) + e2.C

    gt_eta = matvec(mx.swapaxes(g, -1, -2), e2.eta - matvec(e2.J, e1.b))
    eta = matvec(mx.swapaxes(e1.A, -1, -2), gt_eta) + e1.eta
    j = sym(matmul(mx.swapaxes(e1.A, -1, -2), matmul(jt, e1.A))) + e1.J

    ell = (e1.ell + e2.ell
           - log_det
           - 0.5 * quad_form(e1.b, jt)
           + mx.sum(e2.eta * g_b1, axis=-1)
           + 0.5 * quad_form(e2.eta, m_mat))
    return Element(A=a, b=b, C=c, eta=eta, J=j, ell=ell)


def combine_leaf(e1: Element, e2: Element, r1: LeafRank1,
                 formal: bool = False) -> Element:
    """``combine(e1, e2)`` where ``e2`` is a raw leaf -- no factorisation.

    Because a leaf's information matrix is ``J2 = z z^T``, every inverse in
    the general composition collapses under Sherman--Morrison.  With
    ``p = C1 z`` and ``alpha = 1 + z^T p``:

        (I + C1 J2)^-1 = I - p z^T / alpha          [G]
        (I + C1 J2)^-1 C1 = C1 - p p^T / alpha      [M]
        (I + J2 C1)^-1 J2 = z z^T / alpha           [Jt]
        log|I + C1 J2| = log(alpha)                 [determinant lemma]

    so the two Choleskys and the triangular solve disappear and ~13 tiny
    matrix products drop to three.  The log-normaliser collapses further: the
    three quadratic cross terms sum to ``g^2/2 - beta^2/(2 alpha)`` with
    ``beta = g - z^T b1``, which -- once the leaf's own ``-g^2/2`` cancels --
    is exactly the Kalman prediction-error form,
    ``-1/2 [log(alpha S / r_ref) + beta^2/alpha]``.  ``beta/sqrt(alpha)`` is
    the normalised innovation and ``alpha S`` the innovation variance; seeing
    the filter fall out of the element algebra is a good sign the algebra is
    right.

    This is the composition the fused Metal kernel's pass 1 runs
    sequentially in registers: O(m^2) per observation, nothing to factor.

    ``formal`` (a convolved term): ``C1`` can be indefinite, so ``alpha`` can
    be negative, and only ``log|alpha|`` enters -- see :func:`combine`.
    """
    z, g = r1.z, r1.g
    p = matvec(e1.C, z)                              # C1 z
    inv_a = 1.0 / (1.0 + mx.sum(z * p, axis=-1))     # 1/alpha
    za1 = matvec(mx.swapaxes(e1.A, -1, -2), z)       # A1^T z
    beta = g - mx.sum(z * e1.b, axis=-1)             # innovation

    g_a1 = e1.A - outer(p, za1) * inv_a[..., None, None]
    a = matmul(e2.A, g_a1)
    b = matvec(e2.A, e1.b + p * (beta * inv_a)[..., None]) + e2.b
    m_mat = e1.C - outer(p, p) * inv_a[..., None, None]
    c = sym(matmul(e2.A, matmul(m_mat, mx.swapaxes(e2.A, -1, -2)))) + e2.C
    eta = za1 * (beta * inv_a)[..., None] + e1.eta
    j = outer(za1, za1) * inv_a[..., None, None] + e1.J
    # `r1.ld`, not `e2.ell + g^2/2`: the leaf's ell is `ld - g^2/2` and this
    # sum puts the `g^2/2` straight back, so the two cancel exactly in real
    # arithmetic and lose four digits in float32 whenever g^2 >> ld.
    ell = (e1.ell + r1.ld
           + 0.5 * mx.log(mx.abs(inv_a) if formal else inv_a)
           - 0.5 * beta * beta * inv_a)
    return Element(A=a, b=b, C=c, eta=eta, J=j, ell=ell)


def _concat(a: Element, b: Element) -> Element:
    return Element(*(mx.concatenate([getattr(a, f), getattr(b, f)], axis=1)
                     for f in Element._fields))


def _pairs(e: Element):
    """Split axis 1 into (even, odd) halves by reshape, never by a step-2 slice.

    MLX 0.32.2 has a wrong VJP for a strided slice that yields exactly ONE
    element: ``x[1:2:2]`` scatters its cotangent to *both* positions, so the
    gradient comes back as if the two entries were tied together.  Values are
    unaffected, which is what makes it dangerous -- a tree reduction's last
    level is always ``k = 2``, so every gradient this module produced would
    have been silently wrong while every log-likelihood was exact.  Verified
    on CPU and GPU, float32 and float64; ``x[0:1]``/``x[1:2]`` (step 1) and
    step-2 slices of length >= 2 are all correct.

    Reshaping to ``(n, k//2, 2, ...)`` and taking integer index 0 and 1 avoids
    the strided gather entirely, and is the cheaper op anyway.

    Upstream: ml-explore/mlx issue #4444 (root cause: ``normalize_slice``
    collapses the stride of a one-element result without adjusting ``stop``),
    fix pending in PR #4446.  Keep the reshape after the fix; it is faster.
    """
    out_even, out_odd = [], []
    for f in Element._fields:
        x = getattr(e, f)
        n, k = x.shape[0], x.shape[1]
        r = mx.reshape(x, (n, k // 2, 2) + x.shape[2:])
        out_even.append(r[:, :, 0])
        out_odd.append(r[:, :, 1])
    return Element(*out_even), Element(*out_odd)


def tree_reduce(e: Element, formal: bool = False) -> Element:
    """Reduce elements along axis 1 (time) to a single element per chain.

    Balanced pairwise: ``ceil(log2 K)`` levels, each a batched
    :func:`combine` over half as many pairs as the last.  The number of
    levels is a trace-time constant, so the graph is fixed-shape and
    ``mx.compile``-safe; the count is padded up to a power of two with
    :func:`identity_element` (the composition identity, verified as such in
    the tests) so no level ever sees a ragged pair.

    Pairwise is also the summation order the float32 log-determinant and
    quadratic form want: error grows like ``eps sqrt(log K)`` instead of
    ``eps K``.
    """
    k = e.A.shape[1]
    if k == 1:
        return Element(*(getattr(e, f)[:, 0] for f in Element._fields))
    pad = 1
    while pad < k:
        pad *= 2
    if pad != k:
        e = _concat(e, identity_element((e.A.shape[0], pad - k), e.m,
                                        e.A.dtype))
        k = pad
    while k > 1:
        e = combine(*_pairs(e), formal=formal)
        k //= 2
    return Element(*(getattr(e, f)[:, 0] for f in Element._fields))


def _pad_to_pow2(e: Element, r1: LeafRank1):
    k = e.A.shape[1]
    pad = 1
    while pad < k:
        pad *= 2
    if pad == k:
        return e, r1, k
    extra = (e.A.shape[0], pad - k)
    e = _concat(e, identity_element(extra, e.m, e.A.dtype))
    r1 = LeafRank1(
        z=mx.concatenate([r1.z, mx.zeros(extra + (e.m,), dtype=r1.z.dtype)],
                         axis=1),
        g=mx.concatenate([r1.g, mx.zeros(extra, dtype=r1.g.dtype)], axis=1),
        ld=mx.concatenate([r1.ld, mx.zeros(extra, dtype=r1.ld.dtype)],
                          axis=1))
    return e, r1, pad


def _fold_ell_tree(acc, step, lo):
    """Sequential fold whose ``ell`` is summed by a tree, not accumulated.

    ``ell`` is a pure accumulator: no matrix recursion in :func:`combine` or
    :func:`combine_leaf` reads it, so its additions can be reordered freely.
    That matters because the running ``ell`` reaches the full ``|log L|``
    (order 1e4-1e5 at production sizes) while each increment is order 1, so
    a sequential float32 fold rounds at ``eps |log L|`` per step and those
    roundings do not cancel.  Measured on-model at sigma_GP/yerr = 100,
    N = 25000, sub_block 1024: +0.99 log-units sequential against -0.02 for
    the same walk with the increments summed pairwise.

    So: zero the accumulator's ``ell`` before each step, which makes the
    step return its own increment, and add the increments up at the end.
    ``step(acc, j)`` performs one composition.
    """
    zero = mx.zeros_like(acc.ell)
    deltas = [acc.ell]
    acc = acc._replace(ell=zero)
    for j in range(1, lo):
        acc = step(acc, j)
        deltas.append(acc.ell)
        acc = acc._replace(ell=zero)
    return acc._replace(ell=mx.sum(mx.stack(deltas, axis=-1), axis=-1))


def chunked_reduce(e: Element, r1: LeafRank1, sub_len: int = 16,
                   formal: bool = False) -> Element:
    """Reduce along axis 1 by a chunked scan: fold, then tree.

    Split the block into sub-blocks of ``sub_len`` observations, fold each
    sub-block *sequentially* with the factorisation-free
    :func:`combine_leaf`, then :func:`tree_reduce` the sub-block roots with
    the general composition.  This is the Mamba/S5 chunked-scan pattern, and
    the structure the fused Metal kernel of M3 executes: pass 1 condenses
    a sub-block in registers, pass 2 composes the condensed elements.

    Why it is faster than a plain binary tree, even though it does the same
    total number of compositions: a tree spends *half* its work at the leaf
    level and every one of those compositions pays for two Choleskys, while
    the fold pays for none.  Raising ``sub_len`` moves work from the
    expensive general composition into the cheap rank-one one -- the count of
    general compositions per chain drops from ``K`` to ``K/sub_len``.

    The cost is that the fold is ``sub_len`` *sequential* steps, each a
    smaller batch, so past some point the GPU starves.  ``sub_len = 16``
    keeps the batch at ``n_chains * K/16`` elements, which is still hundreds
    of thousands at production sizes.  Set it to 1 to recover the pure tree.
    """
    e, r1, k = _pad_to_pow2(e, r1)
    n = e.A.shape[0]
    lo = max(1, min(sub_len, k))
    while k % lo:                       # keep both axes exact powers of two
        lo //= 2
    if lo == 1:
        return tree_reduce(e, formal=formal)

    n_sub = k // lo
    fields = {f: mx.reshape(getattr(e, f),
                            (n, n_sub, lo) + getattr(e, f).shape[2:])
              for f in Element._fields}
    zr = mx.reshape(r1.z, (n, n_sub, lo, e.m))
    gr = mx.reshape(r1.g, (n, n_sub, lo))
    ldr = mx.reshape(r1.ld, (n, n_sub, lo))

    acc = Element(*(fields[f][:, :, 0] for f in Element._fields))

    def step(a, j):
        leaf = Element(*(fields[f][:, :, j] for f in Element._fields))
        return combine_leaf(a, leaf,
                            LeafRank1(z=zr[:, :, j], g=gr[:, :, j],
                                      ld=ldr[:, :, j]), formal=formal)

    return tree_reduce(_fold_ell_tree(acc, step, lo), formal=formal)

# ---------------------------------------------------------------------------
# block-structured leaves: same algebra, Phi's zeros never touched
# ---------------------------------------------------------------------------


def _bmm_left(blocks, x):
    """``block_diag(blocks) @ x`` for dense ``x``."""
    if len(blocks) == 1:
        return matmul(blocks[0], x)
    rows, off = [], 0
    for b in blocks:
        mi = b.shape[-1]
        rows.append(matmul(b, x[..., off:off + mi, :]))
        off += mi
    return mx.concatenate(rows, axis=-2)


def _bmm_right_t(x, blocks):
    """``x @ block_diag(blocks).T`` for dense ``x``."""
    if len(blocks) == 1:
        return matmul(x, mx.swapaxes(blocks[0], -1, -2))
    cols, off = [], 0
    for b in blocks:
        mi = b.shape[-1]
        cols.append(matmul(x[..., :, off:off + mi], mx.swapaxes(b, -1, -2)))
        off += mi
    return mx.concatenate(cols, axis=-1)


def _bmv(blocks, v):
    """``block_diag(blocks) @ v``."""
    if len(blocks) == 1:
        return matvec(blocks[0], v)
    parts, off = [], 0
    for b in blocks:
        mi = b.shape[-1]
        parts.append(matvec(b, v[..., off:off + mi]))
        off += mi
    return mx.concatenate(parts, axis=-1)


def _add_block_diag(dense, blocks):
    """``dense + block_diag(blocks)``, without materialising the zeros."""
    if len(blocks) == 1:
        return dense + blocks[0]
    m = sum(b.shape[-1] for b in blocks)
    rows, off = [], 0
    for b in blocks:
        mi = b.shape[-1]
        row = dense[..., off:off + mi, :]
        parts = []
        if off:
            parts.append(row[..., :, :off])
        parts.append(row[..., :, off:off + mi] + b)
        if m - off - mi:
            parts.append(row[..., :, off + mi:])
        rows.append(mx.concatenate(parts, axis=-1))
        off += mi
    return mx.concatenate(rows, axis=-2)


def leaf_parts(phi_blocks, q_blocks, h, y, r_var, r_ref, s_floor=None,
               formal=False):
    """Compact leaves: the pieces of :func:`leaf_elements`, unassembled.

    Identical arithmetic; the difference is that ``A``, ``C``, ``J``, ``b``
    and ``eta`` are never formed, so the (chains x points) leaf array holds
    ``2 sum m_i^2 + 2m + 2`` numbers instead of ``3m^2 + 2m + 1``.
    """
    # Per-plane multiply-adds rather than `mx.sum` over a broadcast product,
    # for the same reason as `leaf_elements`: the reduction kernels over a
    # size-m axis cost about twice the unrolled form.  The block sizes are
    # trace-time constants, so both loops unroll into the graph.
    hphi = mx.concatenate(
        [sum(h[:, None, o + i, None] * b[:, :, i, :]
             for i in range(b.shape[-1]))
         for b, o in zip(phi_blocks, _offsets(phi_blocks))], axis=-1)
    qh = mx.concatenate(
        [sum(b[:, :, :, j] * h[:, None, None, o + j]
             for j in range(b.shape[-1]))
         for b, o in zip(q_blocks, _offsets(q_blocks))], axis=-1)
    # clamp the variance term, not the sum, and not at all on the formal
    # path -- see `leaf_elements`
    hqh = sum(h[:, None, i] * qh[:, :, i] for i in range(h.shape[-1]))
    if formal:
        s_q = hqh + r_var
    else:
        s_q = mx.maximum(hqh, mx.zeros_like(hqh)) + r_var
        if s_floor is not None:
            s_q = mx.maximum(s_q, s_floor)
    inv_rs = mx.rsqrt(s_q)
    g_v = y * inv_rs
    ld = -0.5 * mx.log(s_q / r_ref)
    return LeafParts(phi_blocks=phi_blocks, q_blocks=q_blocks,
                     kappa=qh * inv_rs[..., None], z=hphi * inv_rs[..., None],
                     g=g_v, ell=ld - 0.5 * g_v * g_v, ld=ld)


def _offsets(blocks):
    out, off = [], 0
    for b in blocks:
        out.append(off)
        off += b.shape[-1]
    return out


def expand_leaf(parts: LeafParts):
    """Compact leaf -> ``(Element, LeafRank1)``, for the accumulator seed."""
    from .terms import _block_diag
    phi = _block_diag(*parts.phi_blocks)
    q = _block_diag(*parts.q_blocks)
    k, z, g = parts.kappa, parts.z, parts.g
    return (Element(A=phi - outer(k, z), b=k * g[..., None],
                    C=sym(q - outer(k, k)), eta=g[..., None] * z,
                    J=outer(z, z), ell=parts.ell),
            LeafRank1(z=z, g=g, ld=parts.ld))


def combine_leaf_blocked(e1: Element, parts: LeafParts,
                         formal: bool = False) -> Element:
    """:func:`combine_leaf` with the leaf kept in pieces.

    Substituting ``A2 = Phi - kappa z^T`` and ``C2 = Q - kappa kappa^T``
    turns every product with ``A2`` into a block-diagonal product plus a
    rank-one correction::

        A2 X       = Phi X - kappa (z^T X)
        A2 M A2^T  = Phi M Phi^T - y kappa^T - kappa y^T + (z^T M z) kappa kappa^T
                     with y = Phi M z

    so ``Phi``'s zeros are never multiplied and ``A2`` is never assembled.
    Exactly the same numbers as :func:`combine_leaf` -- asserted to 1e-12 in
    the tests, for one, two, three and four blocks.
    """
    phi_b, q_b = parts.phi_blocks, parts.q_blocks
    z, g, kappa = parts.z, parts.g, parts.kappa

    p = matvec(e1.C, z)
    inv_a = 1.0 / (1.0 + mx.sum(z * p, axis=-1))
    za1 = matvec(mx.swapaxes(e1.A, -1, -2), z)
    beta = g - mx.sum(z * e1.b, axis=-1)

    ga = e1.A - outer(p, za1) * inv_a[..., None, None]
    a = _bmm_left(phi_b, ga) - outer(kappa,
                                     mx.sum(z[..., :, None] * ga, axis=-2))

    v = e1.b + p * (beta * inv_a)[..., None]
    b = _bmv(phi_b, v) + kappa * (g - mx.sum(z * v, axis=-1))[..., None]

    m_mat = e1.C - outer(p, p) * inv_a[..., None, None]
    mz = matvec(m_mat, z)
    y = _bmv(phi_b, mz)
    c = _bmm_right_t(_bmm_left(phi_b, m_mat), phi_b)
    c = c - outer(y, kappa) - outer(kappa, y) \
        + outer(kappa, kappa) * (mx.sum(z * mz, axis=-1) - 1.0)[..., None, None]
    c = sym(_add_block_diag(c, q_b))

    eta = za1 * (beta * inv_a)[..., None] + e1.eta
    j = outer(za1, za1) * inv_a[..., None, None] + e1.J
    # see combine_leaf: `parts.ld`, never `parts.ell + g^2/2`
    ell = (e1.ell + parts.ld
           + 0.5 * mx.log(mx.abs(inv_a) if formal else inv_a)
           - 0.5 * beta * beta * inv_a)
    return Element(A=a, b=b, C=c, eta=eta, J=j, ell=ell)


def _pad_parts_to_pow2(parts: LeafParts):
    """Pad the sub-block axis with composition identities.

    The identity leaf is ``Phi = I``, ``Q = 0``, ``kappa = z = g = 0``:
    ``A = I``, ``C = 0``, ``b = eta = 0``, ``J = 0``, ``ell = 0``.
    """
    k = parts.g.shape[1]
    pad = 1
    while pad < k:
        pad *= 2
    if pad == k:
        return parts, k
    n, extra = parts.g.shape[0], pad - k
    dtype = parts.g.dtype

    def grow_blocks(blocks, eye):
        out = []
        for b in blocks:
            mi = b.shape[-1]
            fill = (mx.broadcast_to(mx.eye(mi, dtype=dtype), (n, extra, mi, mi))
                    if eye else mx.zeros((n, extra, mi, mi), dtype=dtype))
            out.append(mx.concatenate([b, fill], axis=1))
        return out

    def grow_vec(v):
        return mx.concatenate(
            [v, mx.zeros((n, extra) + v.shape[2:], dtype=dtype)], axis=1)

    return LeafParts(grow_blocks(parts.phi_blocks, True),
                     grow_blocks(parts.q_blocks, False),
                     grow_vec(parts.kappa), grow_vec(parts.z),
                     grow_vec(parts.g), grow_vec(parts.ell),
                     grow_vec(parts.ld)), pad


def chunked_reduce_blocked(parts: LeafParts, sub_len: int = 16,
                           formal: bool = False) -> Element:
    """:func:`chunked_reduce` over compact leaves."""
    parts, k = _pad_parts_to_pow2(parts)
    n = parts.g.shape[0]
    lo = max(1, min(sub_len, k))
    while k % lo:
        lo //= 2
    n_sub = k // lo

    def resh(x):
        return mx.reshape(x, (n, n_sub, lo) + x.shape[2:])

    phi_r = [resh(b) for b in parts.phi_blocks]
    q_r = [resh(b) for b in parts.q_blocks]
    ka, zr, gr, er, ldr = (resh(parts.kappa), resh(parts.z), resh(parts.g),
                           resh(parts.ell), resh(parts.ld))

    def slice_j(j):
        return LeafParts([b[:, :, j] for b in phi_r], [b[:, :, j] for b in q_r],
                         ka[:, :, j], zr[:, :, j], gr[:, :, j], er[:, :, j],
                         ldr[:, :, j])

    acc, _ = expand_leaf(slice_j(0))
    for j in range(1, lo):
        acc = combine_leaf_blocked(acc, slice_j(j), formal=formal)
    if n_sub == 1:
        return Element(*(getattr(acc, f)[:, 0] for f in Element._fields))
    return tree_reduce(acc, formal=formal)



def suffix_scan(e: Element) -> Element:
    """Running composition from the right, with an identity past the end.

    ``out[:, j]`` is ``e[:, j] * e[:, j+1] * ... * e[:, k-1]`` and
    ``out[:, k]`` is :func:`identity_element`, so the result has ``k + 1``
    entries along axis 1.

    The mirror of :func:`prefix_scan`, and needed for the same reason M4
    needs that one.  A prefix gives the filter state a block starts from;
    a suffix gives *the future as a Gaussian in that state*, which is what
    turns the backward pass's seed into a closed form instead of a chain.
    Only ``J`` and ``eta`` of the result are used there -- they carry the
    information form of everything after the block -- but the recursion
    that produces them needs the other fields, so the whole element is
    composed.
    """
    k = e.A.shape[1]
    n, m, dtype = e.A.shape[0], e.m, e.A.dtype
    acc = identity_element((n,), m, dtype)
    outs = [acc]
    for j in range(k - 1, -1, -1):
        acc = combine(Element(*(getattr(e, f)[:, j] for f in Element._fields)),
                      acc)
        outs.append(acc)
    outs.reverse()
    return Element(*(mx.stack([getattr(o, f) for o in outs], axis=1)
                     for f in Element._fields))


def prefix_scan(e: Element) -> Element:
    """Running composition along axis 1: ``out[:, j] = e[:,0] * ... * e[:,j]``.

    The reduction gives log L; the *prefix* gives what the backward pass
    needs, which is the state each block starts from.

    The connection that makes this useful: the first element of the series
    has ``A = 0``, ``J = 0``, ``eta = 0``, and composition preserves that, so
    every prefix element also has them zero.  What is left in a prefix is
    ``(b, C)`` -- and those *are* the Kalman filtered mean and covariance
    after that block, with ``ell`` the log-likelihood accumulated so far.  So
    one prefix scan over the ~1e2 block elements hands every block both its
    starting filter state and its starting log-likelihood, with no storage
    of anything per observation.

    Sequential in the block index on purpose: there are tens to hundreds of
    blocks, so a Blelloch scan would buy nothing over a Python loop whose
    body is one batched :func:`combine`.
    """
    k = e.A.shape[1]
    if k == 1:
        return e
    acc = Element(*(getattr(e, f)[:, 0] for f in Element._fields))
    outs = [acc]
    for j in range(1, k):
        acc = combine(acc, Element(*(getattr(e, f)[:, j]
                                     for f in Element._fields)))
        outs.append(acc)
    return Element(*(mx.stack([getattr(o, f) for o in outs], axis=1)
                     for f in Element._fields))
