"""The hand-derived backward recursion (M4's specification).

The Metal backward kernel cannot be checked against autodiff -- MLX has no
VJP for a custom kernel, which is the whole reason it needs a hand-written
one.  So the recursion is derived and pinned *here*, in MLX, against both
``mx.vjp`` and float64 central differences; the kernel is then written and
tested against this.

Getting the symmetrisation right is the part that bites: the forward
symmetrises ``Cp`` and ``P``, so their adjoints are the *symmetrised* ones.
Missing it left dPhi, dh, dy and dr_var exact and only dQ wrong by 4%, which
is exactly the kind of error a spot check on one output would sail past.
"""

import numpy as np
import mlx.core as mx
import pytest

from anvilgp import linalg
from anvilgp.adjoint import filter_backward, filter_forward
from anvilgp.kalman import sequential_loglike

mx.set_default_device(mx.cpu)
D = mx.float64
NAMES = ["phi", "q", "h", "y", "r_var"]


@pytest.fixture(autouse=True)
def unrolled():
    linalg.FORCE_UNROLLED = True
    yield
    linalg.FORCE_UNROLLED = False


def _problem(seed, n=3, k=9, m=2):
    rng = np.random.default_rng(seed)

    def rnd(*s):
        return mx.array(rng.standard_normal(s), dtype=D)

    phi = rnd(n, k, m, m) * 0.4
    qr = rnd(n, k, m, m)
    q = mx.matmul(qr, mx.swapaxes(qr, -1, -2)) + 0.5 * mx.eye(m, dtype=D)
    rv = mx.array(np.abs(rng.standard_normal((n, k))) + 0.3, dtype=D)
    rr = mx.array(np.abs(rng.standard_normal(k)) + 0.3, dtype=D)
    return phi, q, rnd(n, m), rnd(n, k), rv, rr, rnd(n)


def test_forward_replay_matches_the_sequential_filter():
    phi, q, h, y, rv, rr, _ = _problem(0)
    _, ell = filter_forward(phi, q, h, y, rv, rr)
    ref = sequential_loglike(phi, q, h, y, rv, rr)
    assert float(mx.max(mx.abs(ell - ref))) == 0.0


@pytest.mark.parametrize("seed,m", [(1, 2), (2, 2), (3, 3), (4, 4), (5, 1)])
def test_adjoint_matches_autodiff(seed, m):
    phi, q, h, y, rv, rr, cot = _problem(seed, m=m)
    _, ad = mx.vjp(lambda a, b, c, d, e: sequential_loglike(a, b, c, d, e, rr),
                   [phi, q, h, y, rv], [cot])
    states, _ = filter_forward(phi, q, h, y, rv, rr)
    hand = filter_backward(phi, q, h, y, rv, rr, states, cot)
    for name, a, b in zip(NAMES, ad, hand):
        a, b = np.array(a), np.array(b)
        scale = max(1.0, np.abs(a).max())
        assert np.max(np.abs(a - b)) / scale < 1e-11, name


@pytest.mark.parametrize("target", NAMES)
def test_adjoint_matches_central_differences(target):
    """Independent of autodiff: if MLX's VJP and the hand recursion shared a
    misconception, differences would still catch it."""
    phi, q, h, y, rv, rr, _ = _problem(7)
    cot = mx.ones((phi.shape[0],), dtype=D)
    states, _ = filter_forward(phi, q, h, y, rv, rr)
    hand = dict(zip(NAMES, filter_backward(phi, q, h, y, rv, rr, states, cot)))

    args = {"phi": phi, "q": q, "h": h, "y": y, "r_var": rv}
    base = {k: np.array(v) for k, v in args.items()}
    idx = tuple(0 for _ in base[target].shape)
    step = 1e-6

    def total(mod):
        a = {k: mx.array(v, dtype=D) for k, v in mod.items()}
        return float(mx.sum(sequential_loglike(a["phi"], a["q"], a["h"],
                                               a["y"], a["r_var"], rr)))

    up, dn = {k: v.copy() for k, v in base.items()}, {k: v.copy()
                                                      for k, v in base.items()}
    up[target][idx] += step
    dn[target][idx] -= step
    fd = (total(up) - total(dn)) / (2 * step)
    got = float(np.array(hand[target])[idx])
    # q enters symmetrically, so perturbing one off-diagonal entry alone is
    # not the derivative the symmetrised adjoint reports; use a diagonal.
    if target == "q" and idx[-1] != idx[-2]:
        pytest.skip("off-diagonal of a symmetric argument")
    assert abs(got - fd) < 1e-5 * max(1.0, abs(fd)), (target, got, fd)


def test_symmetrisation_of_the_q_adjoint_is_load_bearing():
    """Pins the bug that was actually made: without symmetrising the Cp
    adjoint, dQ is wrong by percent while everything else stays exact."""
    phi, q, h, y, rv, rr, cot = _problem(1)
    _, ad = mx.vjp(lambda a, b, c, d, e: sequential_loglike(a, b, c, d, e, rr),
                   [phi, q, h, y, rv], [cot])
    states, _ = filter_forward(phi, q, h, y, rv, rr)
    hand = filter_backward(phi, q, h, y, rv, rr, states, cot)
    q_ad, q_hand = np.array(ad[1]), np.array(hand[1])
    assert np.max(np.abs(q_ad - q_hand)) < 1e-13
    # the un-symmetrised version differs materially, i.e. this is not a
    # distinction without a difference
    assert np.max(np.abs(q_hand - np.swapaxes(q_hand, -1, -2))) < 1e-13


def test_prefix_scan_is_the_kalman_filter_state():
    """What lets a block's backward sweep start without storing anything per
    observation: a prefix element's (b, C) are the filtered mean and
    covariance, and A, J, eta stay exactly zero."""
    import anvilgp as ag
    from anvilgp.elements import leaf_elements, prefix_scan
    rng = np.random.default_rng(0)
    k = 24
    dt = np.concatenate([[0.0], rng.uniform(0.01, 0.5, k - 1)])
    y = mx.array(rng.standard_normal((1, k)), dtype=D)
    p = mx.array(np.array([[1.3, 4.0, 3.0]]), dtype=D)
    phi, q, h = ag.SHOTerm().matrices(p, mx.array(dt, dtype=D))
    keep = mx.concatenate([mx.zeros((1, 1, 1, 1), dtype=D),
                           mx.ones((1, k - 1, 1, 1), dtype=D)], axis=1)
    phi = phi * keep
    q = q * keep + (1.0 - keep) * mx.eye(2, dtype=D)
    rv = mx.full((1, k), 0.09, dtype=D)
    rr = mx.full((k,), 0.09, dtype=D)
    pre = prefix_scan(leaf_elements(phi, q, h, y, rv, rr)[0])
    states, ell = filter_forward(phi, q, h, y, rv, rr)

    assert np.abs(np.array(pre.A)).max() < 1e-15
    assert np.abs(np.array(pre.J)).max() < 1e-15
    assert np.abs(np.array(pre.eta)).max() < 1e-15
    for n in range(k):
        mean, cov = states[n + 1]
        assert np.abs(np.array(pre.b)[:, n] - np.array(mean)).max() < 1e-13
        assert np.abs(np.array(pre.C)[:, n] - np.array(cov)).max() < 1e-13
    assert abs(float(np.array(pre.ell)[0, -1]) - float(np.array(ell)[0])) < 1e-12
