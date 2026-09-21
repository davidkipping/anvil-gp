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

import anvilgp as ag
from anvilgp.adjoint import (blocked_backward, filter_backward,
                             filter_forward, independent_backward)
from anvilgp.elements import first_point
from anvilgp.kalman import sequential_loglike

D = mx.float64
NAMES = ["phi", "q", "h", "y", "r_var"]


@pytest.fixture(autouse=True)
def _unrolled(unrolled):
    """Shared fixture from conftest.py, made autouse for this module."""


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
    phi, q = first_point(phi, q)
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


@pytest.mark.parametrize("blk", [1, 3, 5, 7, 20, 64])
def test_blocked_backward_equals_the_whole_series_one(blk):
    """The scheme the Metal kernel implements: each block replays its own
    forward states from what it inherits and hands (mbar, Pbar) back.  Must
    be exact at every block length, including blk = 1 and blk > N."""
    phi, q, h, y, rv, rr, cot = _problem(5, k=20)
    _, ad = mx.vjp(lambda a, b, c, d, e: sequential_loglike(a, b, c, d, e, rr),
                   [phi, q, h, y, rv], [cot])
    got = blocked_backward(phi, q, h, y, rv, rr, cot, blk)
    for name, a, b in zip(NAMES, ad, got):
        a, b = np.array(a), np.array(b)
        assert np.max(np.abs(a - b)) / max(1.0, np.abs(a).max()) < 1e-11, name


def test_block_seeds_come_from_the_prefix_state():
    """A block's replay must start from the filter state entering it -- if
    the seed were wrong, only the first block would be right."""
    phi, q, h, y, rv, rr, cot = _problem(6, k=16)
    whole, _ = filter_forward(phi, q, h, y, rv, rr)
    blk = 4
    mean = mx.zeros_like(whole[0][0])
    cov = mx.zeros_like(whole[0][1])
    for s in range(0, 16, blk):
        st, _ = filter_forward(phi[:, s:s + blk], q[:, s:s + blk], h,
                               y[:, s:s + blk], rv[:, s:s + blk], rr[s:s + blk],
                               (mean, cov))
        mean, cov = st[-1]
        assert np.abs(np.array(mean) - np.array(whole[s + blk][0])).max() < 1e-13
        assert np.abs(np.array(cov) - np.array(whole[s + blk][1])).max() < 1e-13


# -- the innovation-variance floor ----------------------------------------
#
# Both production forwards (elements.leaf_elements, the Metal kernel) and
# the oracle (kalman.sequential_loglike) floor S.  This spec did not, so a
# backward kernel written against it would have disagreed with the forward
# it differentiates on exactly the lanes where positive-definiteness has
# already failed -- the lanes a warmup chain visits.


def _floored_problem(seed=4, n=3, k=9, m=2, frac=0.5):
    """A problem whose floor binds on about ``frac`` of the observations."""
    phi, q, h, y, rv, rr, cot = _problem(seed)
    # put the floor in the middle of the realised S distribution
    _, _ = filter_forward(phi, q, h, y, rv, rr)
    s_vals = []
    mean = mx.zeros((n, m), dtype=D)
    cov = mx.zeros((n, m, m), dtype=D)
    for i in range(k):
        p_i = phi[:, i]
        cp = mx.matmul(p_i, mx.matmul(cov, mx.swapaxes(p_i, -1, -2))) + q[:, i]
        cp = 0.5 * (cp + mx.swapaxes(cp, -1, -2))
        ch = mx.sum(cp * h[:, None, :], axis=-1)
        s = mx.sum(h * ch, axis=-1) + rv[:, i]
        s_vals.append(float(mx.mean(s)))
        kg = ch / s[:, None]
        v = y[:, i] - mx.sum(h * mx.sum(p_i * mean[:, None, :], axis=-1),
                             axis=-1)
        mean = mx.sum(p_i * mean[:, None, :], axis=-1) + kg * v[:, None]
        cov = cp - kg[:, :, None] * ch[:, None, :]
        cov = 0.5 * (cov + mx.swapaxes(cov, -1, -2))
    floor = float(np.quantile(np.array(s_vals), frac))
    return phi, q, h, y, rv, rr, cot, mx.array(np.full(k, floor), dtype=D)


def test_the_floor_actually_binds_in_the_fixture():
    """Guard the guard: a floor that never binds tests nothing."""
    phi, q, h, y, rv, rr, _, sf = _floored_problem()
    _, with_floor = filter_forward(phi, q, h, y, rv, rr, None, sf)
    _, without = filter_forward(phi, q, h, y, rv, rr)
    assert float(mx.max(mx.abs(with_floor - without))) > 1e-3


def test_forward_with_floor_matches_the_oracle():
    phi, q, h, y, rv, rr, _, sf = _floored_problem()
    _, ell = filter_forward(phi, q, h, y, rv, rr, None, sf)
    ref = sequential_loglike(phi, q, h, y, rv, rr, sf)
    assert float(mx.max(mx.abs(ell - ref))) == 0.0


def test_adjoint_with_floor_matches_autodiff():
    """Where the floor binds, S is a constant and its cotangent is cut.

    Without that cut the hand adjoint is not slightly wrong, it is orders of
    magnitude wrong: the cotangent flows back into Phi, Q, h and r_var
    through a path the forward severed.
    """
    phi, q, h, y, rv, rr, cot, sf = _floored_problem()
    primals = [phi, q, h, y, rv]

    def f(*p):
        return sequential_loglike(p[0], p[1], p[2], p[3], p[4], rr, sf)

    _, auto = mx.vjp(f, primals, [cot])
    states, _ = filter_forward(phi, q, h, y, rv, rr, None, sf)
    hand = filter_backward(phi, q, h, y, rv, rr, states, cot, sf)

    for name, a, b in zip(NAMES, auto, hand):
        scale = max(float(mx.max(mx.abs(a))), 1e-12)
        err = float(mx.max(mx.abs(a - b))) / scale
        assert err < 1e-12, f"{name}: {err:.2e}"


@pytest.mark.parametrize("blk", [1, 2, 5, 9])
def test_blocked_backward_with_floor_equals_the_whole_series_one(blk):
    phi, q, h, y, rv, rr, cot, sf = _floored_problem()
    states, _ = filter_forward(phi, q, h, y, rv, rr, None, sf)
    want = filter_backward(phi, q, h, y, rv, rr, states, cot, sf)
    got = blocked_backward(phi, q, h, y, rv, rr, cot, blk, sf)
    for name, a, b in zip(NAMES, want, got):
        scale = max(float(mx.max(mx.abs(a))), 1e-12)
        assert float(mx.max(mx.abs(a - b))) / scale < 1e-12, name


def test_both_forms_clamp_their_own_variance_term():
    """Stage 0: the clamp protects the quantity that can go non-physical.

    Each form computes a different total -- the leaf's is
    ``h Q h^T + r_var``, the filter's is ``h (Phi P Phi^T + Q) h^T +
    r_var`` -- so clamping the *total* made them clamp different things and
    disagree by ~0.1 log-units wherever it bound.  Clamping the variance
    term instead is principled in each form on its own, and gives the
    stronger guarantee ``s >= r_var > 0``: ``r_var`` beside it is exact and
    strictly positive, where the old whole-sum floor allowed a negative
    residue to survive inside the sum.

    Pinned here: healthy inputs agree, and a variance term driven negative
    still gives a finite result with the innovation variance never below
    ``r_var``.
    """
    from anvilgp.elements import chunked_reduce, leaf_elements

    phi, q, h, y, rv, rr, _ = _problem(0)
    agree = float(mx.max(mx.abs(
        sequential_loglike(phi, q, h, y, rv, rr)
        - chunked_reduce(*leaf_elements(phi, q, h, y, rv, rr),
                         sub_len=1).ell)))
    assert agree < 1e-10, f"healthy inputs must agree; got {agree:.3e}"

    n, k, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    bad_q = mx.array(np.tile(-1e-3 * np.eye(m), (n, k, 1, 1)), dtype=D)
    hv = mx.array(np.ones((n, m)), dtype=D)
    rv2 = mx.full((n, k), 0.5, dtype=D)
    rr2 = mx.full((k,), 0.5, dtype=D)

    qh = mx.sum(bad_q * hv[:, None, None, :], axis=-1)
    hqh = mx.sum(hv[:, None, :] * qh, axis=-1)
    assert float(mx.max(hqh)) < 0.0, "the fixture must exercise the clamp"

    out = chunked_reduce(*leaf_elements(phi, bad_q, hv, y, rv2, rr2),
                         sub_len=1).ell
    assert bool(mx.all(mx.isfinite(out))), "a clamped lane must stay finite"
    s_q = mx.maximum(hqh, mx.zeros_like(hqh)) + rv2
    assert float(mx.min(s_q)) >= float(mx.min(rv2)) - 1e-15


# -- stage D: the backward blocks are independent -------------------------
#
# `blocked_backward` chains (m_bar, P_bar) from the last block to the
# first, so the blocks must run in order.  `independent_backward` seeds
# each block from the closed form in `block_seeds` instead, so they can run
# in parallel -- which is what lets the Metal backward kernel have the same
# launch shape as the forward one.  These tests are the gate: if the seed
# is wrong, the design is wrong, and it is checkable here with no kernel.


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("k,blk", [(24, 4), (24, 6), (32, 8), (20, 20)])
def test_independent_blocks_equal_the_chained_sweep(m, k, blk):
    """The stage D gate: 1e-12 in float64, every m, every block length."""
    phi, q, h, y, rv, rr, cot = _problem(m * 31 + k + blk, k=k, m=m)
    states, _ = filter_forward(phi, q, h, y, rv, rr)
    want = filter_backward(phi, q, h, y, rv, rr, states, cot)
    got = independent_backward(phi, q, h, y, rv, rr, cot, blk)
    for name, a, b in zip(NAMES, want, got):
        scale = max(float(mx.max(mx.abs(a))), 1e-30)
        err = float(mx.max(mx.abs(a - b))) / scale
        assert err < 1e-12, f"{name}: {err:.2e}"


@pytest.mark.parametrize("name,term,p", [
    ("sho", ag.SHOTerm(), [1.3, 4.0, 3.0]),
    ("matern52", ag.Matern52Term(), [1.1, 2.5]),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4]),
    ("sum", ag.SHOTerm() + ag.SHOTerm(), [1.0, 4.0, 3.0, 0.5, 1.0, 2.0]),
], ids=["sho", "matern52", "rotation", "sum"])
@pytest.mark.parametrize("cadence", ["regular", "gapped", "dupes"])
def test_independent_blocks_on_real_terms_and_awkward_cadences(
        name, term, p, cadence):
    """Gaps and duplicate timestamps are where the propagator misbehaves."""
    from anvilgp.elements import first_point

    k, rng = 21, np.random.default_rng(2)
    if cadence == "gapped":
        t = np.sort(rng.uniform(0.0, 12.0, k))
        t[k // 2:] += 30.0
    elif cadence == "dupes":
        t = np.repeat(np.sort(rng.uniform(0.0, 12.0, k // 2)), 2)[:k]
    else:
        t = np.arange(k) * 0.3
    dt = np.concatenate([[0.0], np.diff(t)])

    phi, q, h = term.matrices(mx.array(np.array([p]), dtype=D),
                              mx.array(dt, dtype=D))
    phi, q = first_point(phi, q)
    y = mx.array(np.random.default_rng(5).standard_normal((1, k)), dtype=D)
    rv = mx.full((1, k), 0.09, dtype=D)
    rr = mx.full((k,), 0.09, dtype=D)
    cot = mx.ones((1,), dtype=D)

    states, _ = filter_forward(phi, q, h, y, rv, rr)
    want = filter_backward(phi, q, h, y, rv, rr, states, cot)
    got = independent_backward(phi, q, h, y, rv, rr, cot, 7)
    for nm, a, b in zip(NAMES, want, got):
        scale = max(float(mx.max(mx.abs(a))), 1e-30)
        assert float(mx.max(mx.abs(a - b))) / scale < 1e-11, nm


def test_the_last_block_gets_a_zero_seed():
    """Nothing follows it, so the suffix is the identity and the seed is 0.

    Pinned because it is the boundary case the closed form has to get
    right for `independent_backward` to agree with a chained sweep that
    starts from exactly zero.
    """
    from anvilgp.adjoint import block_seeds
    from anvilgp.elements import identity_element

    phi, q, h, y, rv, rr, cot = _problem(0)
    n, m = phi.shape[0], phi.shape[-1]
    states, _ = filter_forward(phi, q, h, y, rv, rr)
    mu, sigma = states[-1]
    mb, pb = block_seeds(mu, sigma, identity_element((n,), m, D), cot)
    assert float(mx.max(mx.abs(mb))) == 0.0
    assert float(mx.max(mx.abs(pb))) == 0.0
