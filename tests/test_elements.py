"""The scan-element algebra: identity, associativity, and the log-normaliser.

The log-normaliser is the component that a values-only test cannot see going
wrong in an interesting way -- it is off by a *constant* per composition, so
every relative check passes and the absolute likelihood is silently shifted.
:func:`test_combine_against_numerical_integration` is the guard: it evaluates
the composed element as a function of ``(x_out, x_in)`` against a brute-force
Gaussian integral, which pins ``ell`` and every other component at once.
"""

import mlx.core as mx
import numpy as np
import pytest

from anvilgp.elements import (Element, LeafRank1, chunked_reduce, combine,
                              first_point,
                              combine_leaf, identity_element, leaf_elements,
                              tree_reduce)

D = mx.float64
M = 2


@pytest.fixture(autouse=True)
def _unrolled(unrolled):
    """Exercise the hand-rolled factorisation even in float64, so a bug in
    the GPU route cannot hide behind the dtype switch.  (The fixture itself
    lives in conftest.py; this makes it autouse for this module.)"""


def _f(x):
    return mx.array(np.asarray(x, dtype=np.float64)[None], dtype=D)


def _rand_element(rng, m=M, scale=1.0):
    def psd(s):
        a = rng.standard_normal((m, m))
        return s * (a @ a.T + 0.4 * np.eye(m))
    return Element(A=_f(rng.standard_normal((m, m))),
                   b=_f(rng.standard_normal(m)),
                   C=_f(psd(scale)),
                   eta=_f(rng.standard_normal(m)),
                   J=_f(psd(0.5 * scale)),
                   ell=mx.array(np.array([rng.standard_normal()]), dtype=D))


def _log_element(e, x_out, x_in):
    """log L(x_out, x_in) for a single-batch element, in numpy."""
    A, b, C, eta, J, ell = (np.array(getattr(e, f))[0] for f in Element._fields)
    d = x_out - (A @ x_in + b)
    m = len(x_out)
    return (-0.5 * (m * np.log(2 * np.pi) + np.log(np.linalg.det(C))
                    + d @ np.linalg.solve(C, d))
            - 0.5 * x_in @ J @ x_in + eta @ x_in + ell)


def test_combine_against_numerical_integration():
    from scipy.integrate import dblquad   # declared test dep
    rng = np.random.default_rng(5)
    e1, e2 = _rand_element(rng), _rand_element(rng)
    out = combine(e1, e2)

    def brute(x_out, x_in):
        def f(z2, z1):
            z = np.array([z1, z2])
            return np.exp(_log_element(e2, x_out, z)
                          + _log_element(e1, z, x_in))
        val, _ = dblquad(f, -14, 14, lambda a: -14, lambda a: 14,
                         epsabs=1e-13, epsrel=1e-11)
        return np.log(val)

    for x_out, x_in in [(np.array([0.2, -0.4]), np.array([0.1, 0.3])),
                        (np.array([-1.0, 0.5]), np.array([0.7, -0.2]))]:
        assert abs(brute(x_out, x_in) - _log_element(out, x_out, x_in)) < 1e-7


def test_identity_element_is_an_identity():
    rng = np.random.default_rng(11)
    e = _rand_element(rng)
    ident = identity_element((1,), M, D)
    for combined in (combine(e, ident), combine(ident, e)):
        for f in Element._fields:
            assert np.max(np.abs(np.array(getattr(combined, f))
                                 - np.array(getattr(e, f)))) < 1e-12


def test_combine_is_associative():
    rng = np.random.default_rng(13)
    a, b, c = (_rand_element(rng) for _ in range(3))
    left = combine(combine(a, b), c)
    right = combine(a, combine(b, c))
    for f in Element._fields:
        lv, rv = np.array(getattr(left, f)), np.array(getattr(right, f))
        assert np.max(np.abs(lv - rv)) < 1e-10 * max(1.0, np.abs(rv).max())


def _leaves(rng, k, term, p):
    dt = np.concatenate([[0.0], rng.uniform(0.01, 0.5, k - 1)])
    y = rng.standard_normal((1, k))
    P = mx.array(np.array([p]), dtype=D)
    phi, q, h = term.matrices(P, mx.array(dt, dtype=D))
    phi, q = first_point(phi, q)
    return leaf_elements(phi, q, h, mx.array(y, dtype=D),
                         mx.full((1, k), 0.09, dtype=D),
                         mx.full((k,), 0.09, dtype=D))     # (Element, LeafRank1)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8, 13, 16, 33])
def test_tree_reduction_equals_the_left_fold(k):
    """Associativity where it matters: the balanced tree and the sequential
    fold must agree, including the padded non-power-of-two sizes."""
    import anvilgp as ag
    rng = np.random.default_rng(100 + k)
    lv, r1 = _leaves(rng, k, ag.SHOTerm(), [1.3, 4.0, 3.0])
    tree = float(np.array(tree_reduce(lv).ell)[0])
    fold = Element(*[getattr(lv, f)[:, 0] for f in Element._fields])
    for i in range(1, k):
        fold = combine(fold, Element(*[getattr(lv, f)[:, i]
                                       for f in Element._fields]))
    assert abs(tree - float(np.array(fold.ell)[0])) < 1e-10 * max(1.0, abs(tree))


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_tree_gradient_equals_the_left_fold_gradient(k):
    """Regression for the MLX strided-slice VJP bug (see elements._pairs).

    Values were exact while gradients were silently wrong, which is the worst
    possible failure mode for an HMC sampler: it converges, to the wrong
    thing.  The tree's last level is always k = 2, so this bit every size.
    """
    import anvilgp as ag
    rng = np.random.default_rng(7)
    dt = np.concatenate([[0.0], rng.uniform(0.01, 0.5, k - 1)])
    y = mx.array(rng.standard_normal((1, k)), dtype=D)
    term, p0 = ag.SHOTerm(), np.array([1.3, 4.0, 3.0])

    def build(P):
        phi, q, h = term.matrices(P, mx.array(dt, dtype=D))
        phi, q = first_point(phi, q)
        return leaf_elements(phi, q, h, y, mx.full((1, k), 0.09, dtype=D),
                             mx.full((k,), 0.09, dtype=D))

    def f(P):
        return tree_reduce(build(P)[0]).ell

    P0 = mx.array(p0[None, :], dtype=D)
    _, g = mx.vjp(f, [P0], [mx.ones((1,), dtype=D)])
    ad = np.array(g[0])[0]
    h_ = 1e-6
    fd = np.array([
        (float(np.array(f(mx.array((p0 + h_ * np.eye(3)[i])[None], dtype=D)))[0])
         - float(np.array(f(mx.array((p0 - h_ * np.eye(3)[i])[None], dtype=D)))[0]))
        / (2 * h_) for i in range(3)])
    assert np.max(np.abs(ad - fd)) < 1e-5 * max(1.0, np.abs(fd).max())


@pytest.mark.parametrize("sub_len", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("k", [8, 16, 32, 48])
def test_chunked_reduce_equals_the_tree(k, sub_len):
    """The chunked scan is a reassociation, not an approximation.

    Every sub-block length must give the tree's answer, including sub_len
    larger than k and sub_len = 1 (which is the tree).
    """
    import anvilgp as ag
    rng = np.random.default_rng(200 + k + sub_len)
    lv, r1 = _leaves(rng, k, ag.SHOTerm(), [1.3, 4.0, 3.0])
    ref = float(np.array(tree_reduce(lv).ell)[0])
    got = float(np.array(chunked_reduce(lv, r1, sub_len=sub_len).ell)[0])
    assert abs(got - ref) < 1e-10 * max(1.0, abs(ref))


def test_combine_leaf_equals_the_general_combine():
    """The rank-one shortcut is exact, not a fast approximation."""
    import anvilgp as ag
    rng = np.random.default_rng(31)
    k = 12
    lv, r1 = _leaves(rng, k, ag.SHOTerm(), [1.3, 4.0, 3.0])

    def at(i):
        return Element(*[getattr(lv, f)[:, i] for f in Element._fields])

    gen = fast = at(0)
    for i in range(1, k):
        gen = combine(gen, at(i))
        fast = combine_leaf(fast, at(i),
                            LeafRank1(z=r1.z[:, i], g=r1.g[:, i],
                                      ld=r1.ld[:, i]))
    for f in Element._fields:
        a, b = np.array(getattr(gen, f)), np.array(getattr(fast, f))
        assert np.max(np.abs(a - b)) < 1e-12 * max(1.0, np.abs(a).max()), f


def test_combine_leaf_accepts_the_identity_padding():
    """Padded slots carry z = 0, g = 0, and must pass the accumulator through."""
    import anvilgp as ag
    rng = np.random.default_rng(41)
    lv, r1 = _leaves(rng, 8, ag.SHOTerm(), [1.3, 4.0, 3.0])
    acc = Element(*[getattr(lv, f)[:, 0] for f in Element._fields])
    ident = identity_element((1,), M, D)
    zero = LeafRank1(z=mx.zeros((1, M), dtype=D), g=mx.zeros((1,), dtype=D),
                     ld=mx.zeros((1,), dtype=D))
    out = combine_leaf(acc, ident, zero)
    for f in Element._fields:
        assert np.max(np.abs(np.array(getattr(out, f))
                             - np.array(getattr(acc, f)))) < 1e-14


@pytest.mark.parametrize("sub_len", [1, 4, 16])
def test_chunked_reduce_gradient_matches_finite_differences(sub_len):
    import anvilgp as ag
    rng = np.random.default_rng(9)
    k = 16
    dt = np.concatenate([[0.0], rng.uniform(0.01, 0.5, k - 1)])
    y = mx.array(rng.standard_normal((1, k)), dtype=D)
    term, p0 = ag.SHOTerm(), np.array([1.3, 4.0, 3.0])

    def f(P):
        phi, q, h = term.matrices(P, mx.array(dt, dtype=D))
        phi, q = first_point(phi, q)
        lv, r1 = leaf_elements(phi, q, h, y, mx.full((1, k), 0.09, dtype=D),
                               mx.full((k,), 0.09, dtype=D))
        return chunked_reduce(lv, r1, sub_len=sub_len).ell

    P0 = mx.array(p0[None, :], dtype=D)
    _, g = mx.vjp(f, [P0], [mx.ones((1,), dtype=D)])
    ad = np.array(g[0])[0]
    h_ = 1e-6
    fd = np.array([
        (float(np.array(f(mx.array((p0 + h_ * np.eye(3)[i])[None], dtype=D)))[0])
         - float(np.array(f(mx.array((p0 - h_ * np.eye(3)[i])[None], dtype=D)))[0]))
        / (2 * h_) for i in range(3)])
    assert np.max(np.abs(ad - fd)) < 1e-5 * max(1.0, np.abs(fd).max())


# ---------------------------------------------------------------------------
# compact block-structured leaves
# ---------------------------------------------------------------------------

BLOCKED_TERMS = [
    ("sho", "SHOTerm", [1.3, 4.0, 3.0]),
    ("rotation", "RotationTerm", [1.5, 7.0, 3.0, 1.0, 0.4]),
    ("sho3", "sum3", [1.0, 4.0, 3.0, 0.5, 1.0, 2.0, 0.3, 0.5, 5.0]),
    ("sho4", "sum4", [1.0, 4.0, 3.0, 0.5, 1.0, 2.0, 0.3, 0.5, 5.0,
                      0.2, 0.25, 1.5]),
]


def _term(name):
    import anvilgp as ag
    if name == "sum3":
        return ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    if name == "sum4":
        return ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm() + ag.SHOTerm()
    return getattr(ag, name)()


def _both_leaf_forms(rng, k, term, p):
    """(dense Element + rank1, compact LeafParts) from identical inputs."""
    from anvilgp.elements import leaf_parts
    from anvilgp.terms import _block_diag
    dt = np.concatenate([[0.0], rng.uniform(0.01, 0.5, k - 1)])
    y = mx.array(rng.standard_normal((1, k)), dtype=D)
    P = mx.array(np.array([p]), dtype=D)
    phi_b, q_b, h = term.matrices_blocks(P, mx.array(dt, dtype=D))
    phi_b, q_b = first_point(phi_b, q_b)
    rv, rr = mx.full((1, k), 0.09, dtype=D), mx.full((k,), 0.09, dtype=D)
    dense = leaf_elements(_block_diag(*phi_b), _block_diag(*q_b), h, y, rv, rr)
    return dense, leaf_parts(phi_b, q_b, h, y, rv, rr)


@pytest.mark.parametrize("name,cls,p", BLOCKED_TERMS,
                         ids=[c[0] for c in BLOCKED_TERMS])
def test_expand_leaf_reproduces_the_dense_leaf(name, cls, p):
    """The compact form is a *representation*, not an approximation."""
    from anvilgp.elements import expand_leaf
    term = _term(cls)
    (dense, r1), parts = _both_leaf_forms(np.random.default_rng(5), 16, term, p)
    got, r1_got = expand_leaf(parts)
    for f in Element._fields:
        a, b = np.array(getattr(dense, f)), np.array(getattr(got, f))
        assert np.max(np.abs(a - b)) < 1e-13 * max(1.0, np.abs(a).max()), f
    assert np.max(np.abs(np.array(r1.z) - np.array(r1_got.z))) < 1e-14
    assert np.max(np.abs(np.array(r1.g) - np.array(r1_got.g))) < 1e-14


@pytest.mark.parametrize("name,cls,p", BLOCKED_TERMS,
                         ids=[c[0] for c in BLOCKED_TERMS])
def test_blocked_composition_equals_the_dense_one(name, cls, p):
    from anvilgp.elements import LeafParts, combine_leaf_blocked
    term = _term(cls)
    k = 12
    (lv, r1), parts = _both_leaf_forms(np.random.default_rng(6), k, term, p)

    def at(i):
        return Element(*[getattr(lv, f)[:, i] for f in Element._fields])

    def parts_at(i):
        return LeafParts([b[:, i] for b in parts.phi_blocks],
                         [b[:, i] for b in parts.q_blocks],
                         parts.kappa[:, i], parts.z[:, i], parts.g[:, i],
                         parts.ell[:, i], parts.ld[:, i])

    dense_acc = blocked_acc = at(0)
    for i in range(1, k):
        dense_acc = combine_leaf(dense_acc, at(i),
                                 LeafRank1(z=r1.z[:, i], g=r1.g[:, i],
                                      ld=r1.ld[:, i]))
        blocked_acc = combine_leaf_blocked(blocked_acc, parts_at(i))
    for f in Element._fields:
        a = np.array(getattr(dense_acc, f))
        b = np.array(getattr(blocked_acc, f))
        assert np.max(np.abs(a - b)) < 1e-11 * max(1.0, np.abs(a).max()), f


@pytest.mark.parametrize("sub_len", [1, 4, 16, 32])
@pytest.mark.parametrize("k", [8, 16, 48])
@pytest.mark.parametrize("name,cls,p", BLOCKED_TERMS,
                         ids=[c[0] for c in BLOCKED_TERMS])
def test_blocked_chunked_reduce_equals_the_dense_one(name, cls, p, k, sub_len):
    """Including the identity padding, which in compact form is
    Phi = I, Q = 0, kappa = z = g = 0."""
    from anvilgp.elements import chunked_reduce_blocked
    term = _term(cls)
    (lv, r1), parts = _both_leaf_forms(np.random.default_rng(70 + k), k,
                                       term, p)
    ref = float(np.array(chunked_reduce(lv, r1, sub_len=sub_len).ell)[0])
    got = float(np.array(chunked_reduce_blocked(parts, sub_len=sub_len).ell)[0])
    assert abs(got - ref) < 1e-10 * max(1.0, abs(ref))


def test_use_blocked_leaves_criterion():
    """Pinned to the measurement in the docstring: two or more blocks, m >= 6."""
    from anvilgp.elements import use_blocked_leaves
    assert not use_blocked_leaves((2,))            # m=2, one block
    assert not use_blocked_leaves((2, 2))          # m=4: dense is faster
    assert use_blocked_leaves((2, 2, 2))           # m=6
    assert use_blocked_leaves((2, 2, 2, 2))        # m=8
    assert use_blocked_leaves((3, 3))              # m=6, Matern-5/2 pair
    assert not use_blocked_leaves((8,))            # single block: nothing to do


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64])
def test_strict_chol_refuses_a_clearly_negative_pivot_and_nothing_else(dtype):
    """``strict`` turns a floored *indefinite* matrix into a NaN factor.

    The formal model of a convolved term hands the composition covariances
    that are genuinely indefinite; flooring them substitutes a different
    matrix and a wrong likelihood.  A PSD matrix whose last pivot rounds a
    hair below zero -- the case the floor exists for -- must factor exactly
    as before.  In float64 ``strict`` takes the unrolled route (LAPACK on an
    indefinite input is not checked by MLX), so it is compared against the
    built-in there.
    """
    from anvilgp.linalg import chol
    with mx.stream(mx.cpu if dtype == mx.float64 else mx.gpu):
        bad = mx.array([[[1.0, 0.2], [0.2, -0.1]]], dtype=dtype)
        assert np.all(np.isnan(np.array(chol(bad, strict=True))[0, 1, 1]))
        assert np.all(np.isfinite(np.array(chol(bad))))
        eps = 1e-7 if dtype == mx.float32 else 1e-15
        near = mx.array([[[1.0, 1.0], [1.0, 1.0 - eps]]], dtype=dtype)
        good = mx.array([[[2.0, 0.3], [0.3, 1.5]]], dtype=dtype)
        for a in (near, good):
            s = np.array(chol(a, strict=True), np.float64)
            p = np.array(chol(a), np.float64)
            assert np.all(np.isfinite(s))
            np.testing.assert_allclose(s, p, rtol=1e-6 if dtype == mx.float32
                                       else 1e-12, atol=0)
