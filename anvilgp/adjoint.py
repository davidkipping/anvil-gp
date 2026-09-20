"""Hand-derived backward recursion for the filter log-likelihood -- M4.

The forward kernel walks a block keeping the filter state in registers and
writing nothing per observation.  That is why it is fast, and it is also why
its gradient needs care: reverse-mode normally wants the intermediate states
the forward pass deliberately threw away.

The scheme, following the brief: **block boundaries are the checkpoints.**
:func:`anvilgp.elements.prefix_scan` hands each block the filter state it
starts from (a prefix element's ``(b, C)`` *are* the filtered mean and
covariance -- see its docstring), so a block's backward sweep can replay its
own ``L`` steps forward from that state, keep those ``L`` states locally, and
then sweep back.  Nothing per-observation crosses memory.

What must never happen is reconstructing the forward state by running the
filter *backwards* through the propagator.  That divides by ``Phi``, which
underflowed on the way forward at exactly the lags where it would now
overflow -- every gap in the data.  Replay forward, always.

This module is the recursion in MLX: slow (a Python loop over observations),
but differentiable-free, exactly matching :func:`anvilgp.kalman.sequential_loglike`,
and checkable against autodiff.  It is the specification the Metal backward
kernel is written against and tested against.

The recursion
-------------
Forward, per observation, with ``(m, P)`` the filtered state::

    mp = Phi m ;  Cp = Phi P Phi^T + Q
    ch = Cp h^T ;  S = h.ch + R ;  v = y - h.mp ;  Kg = ch / S
    m' = mp + Kg v ;  P' = Cp - Kg ch^T
    ell = -1/2 [ log(S/Rref) + v^2/S ]

Backward, carrying ``(mbar, Pbar)`` from later observations and emitting the
parameter adjoints.  The two seeds are the only places ``ell`` enters::

    vbar = -v/S ;  Sbar = (v^2 - S) / (2 S^2)
"""

from __future__ import annotations

import mlx.core as mx

from .linalg import matmul, matvec, outer, sym

__all__ = ["filter_forward", "filter_backward", "blocked_backward"]


def _predict_step(p_i, q_i, h, y_i, rv_i, mean, cov, floor_i=None):
    """One filter prediction + innovation, shared by the sweep and the replay.

    The checkpointing scheme *requires* that the backward sweep replay
    exactly the forward it is differentiating -- not an equivalent
    computation, the same one, down to the order of operations, or the
    adjoint is the adjoint of a slightly different function.  Writing it
    once makes that true by construction rather than by inspection.

    Returns ``(mp, cp, ch, s, v, kg, floored)``; ``floored`` is the mask of
    lanes where the innovation-variance floor bound, or None when no floor
    is in force.
    """
    mp = matvec(p_i, mean)
    cp = sym(matmul(p_i, matmul(cov, mx.swapaxes(p_i, -1, -2))) + q_i)
    ch = matvec(cp, h)
    s = mx.sum(h * ch, axis=-1) + rv_i
    floored = None
    if floor_i is not None:
        floored = s < floor_i
        s = mx.maximum(s, floor_i)
    v = y_i - mx.sum(h * mp, axis=-1)
    return mp, cp, ch, s, v, ch / s[:, None], floored


def filter_forward(phi, q, h, y, r_var, r_ref, init=None, s_floor=None):
    """Run the filter, keeping every state.  Returns (states, logL).

    ``states`` is the list of ``(m_n, P_n)`` *after* each update, plus the
    state it began from, which the backward sweep consumes in reverse.
    ``init`` seeds it -- for a block, with what the prefix scan says enters
    that block; for the whole series, with zeros (Phi = 0, Q = P_inf = I at
    the first observation makes that the stationary prior).

    ``s_floor`` is the per-observation floor on the innovation variance,
    applied exactly as :func:`anvilgp.kalman.sequential_loglike` applies it.
    It is not optional decoration: this function is the specification the
    Metal backward kernel is written against, and a kernel that omitted the
    floor would disagree with the forward it is differentiating precisely on
    the lanes where positive-definiteness has already failed.

    **But it is not the same quantity that production floors, and that has
    to be settled before the kernel is written.**  Here and in
    ``kalman.py`` the floor lands on the *filter* innovation variance
    ``S = h (Phi P Phi^T + Q) h^T + r_var``.  In
    :func:`anvilgp.elements.leaf_elements` -- and in the Metal forward
    kernel, which is what a gradient would actually differentiate -- it
    lands on the *leaf* quantity ``s_q = h Q h^T + r_var``.  The two differ
    by ``alpha = 1 + z^T C1 z >= 1``, so they bind on different lanes: with
    a floor high enough to bind, the two paths disagree by ~0.1 log-units
    in float64.  In production neither binds (``s_floor_rel = 1e-6`` against
    ``r_var >= r_ref``), so this is latent rather than live, and
    ``tests/test_adjoint.py`` cannot see it because both sides there use the
    filter convention.
    """
    n, k, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dtype = phi.dtype
    mean = mx.zeros((n, m), dtype=dtype) if init is None else init[0]
    cov = mx.zeros((n, m, m), dtype=dtype) if init is None else init[1]
    ell = mx.zeros((n,), dtype=dtype)
    states = [(mean, cov)]
    for i in range(k):
        p_i = phi[:, i]
        mp, cp, ch, s, v, kg, _ = _predict_step(
            p_i, q[:, i], h, y[:, i], r_var[:, i], mean, cov,
            None if s_floor is None else s_floor[i])
        mean = mp + kg * v[:, None]
        cov = cp - outer(kg, ch)
        cov = sym(cov)
        ell = ell - 0.5 * (mx.log(s / r_ref[i]) + v * v / s)
        states.append((mean, cov))
    return states, ell


def filter_backward(phi, q, h, y, r_var, r_ref, states, cot, s_floor=None):
    """Adjoints of ``sequential_loglike`` w.r.t. (phi, q, h, y, r_var).

    ``cot`` is the cotangent on the returned log-likelihood, shape (n,).
    Hand-derived; :mod:`tests.test_adjoint` checks every output against
    ``mx.vjp`` and against float64 central differences.
    """
    n, m = phi.shape[0], phi.shape[-1]
    dtype = phi.dtype
    out = _backward_span(phi, q, h, y, r_var, r_ref, states, cot,
                         mx.zeros((n, m), dtype=dtype),
                         mx.zeros((n, m, m), dtype=dtype), s_floor)
    return out[:5]


def _backward_span(phi, q, h, y, r_var, r_ref, states, cot, m_bar, p_bar,
                   s_floor=None):
    """One reverse sweep over a span, seeded by the adjoints leaving it.

    Returns the span's parameter adjoints plus the ``(mbar, Pbar)`` to hand
    to the span before it.  The whole-series case is this with zero seeds.
    """
    n, k, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dtype = phi.dtype
    c = cot[:, None]

    h_bar = mx.zeros((n, m), dtype=dtype)
    phi_rows, q_rows, y_cols, rv_cols = [], [], [], []

    for i in range(k - 1, -1, -1):
        mean_prev, cov_prev = states[i]
        p_i = phi[:, i]
        # replay the forward quantities this step needs -- the SAME
        # computation the forward ran, not an equivalent one
        mp, cp, ch, s, v, kg, floored = _predict_step(
            p_i, q[:, i], h, y[:, i], r_var[:, i], mean_prev, cov_prev,
            None if s_floor is None else s_floor[i])

        # -- seeds: the only place ell enters --
        v_bar = -c[:, 0] * v / s
        s_bar = c[:, 0] * (v * v - s) / (2.0 * s * s)

        # P' = Cp - Kg ch^T
        cp_bar = p_bar
        kg_bar = -matvec(p_bar, ch)
        ch_bar = -matvec(mx.swapaxes(p_bar, -1, -2), kg)
        # m' = mp + Kg v
        mp_bar = m_bar
        kg_bar = kg_bar + m_bar * v[:, None]
        v_bar = v_bar + mx.sum(kg * m_bar, axis=-1)
        # v = y - h.mp
        y_cols.append(v_bar)
        h_bar = h_bar - v_bar[:, None] * mp
        mp_bar = mp_bar - v_bar[:, None] * h
        # Kg = ch / S
        ch_bar = ch_bar + kg_bar / s[:, None]
        s_bar = s_bar - mx.sum(kg_bar * kg, axis=-1) / s
        # S = max(h.ch + R, floor).  Where the floor binds, S is a constant
        # -- it no longer depends on Phi, Q, h or r_var -- so its cotangent
        # must be cut here, after everything that *divides* by S (which is
        # just a number, floored or not) and before everything that feeds
        # back into what produced it.  Omitting this is not a small error:
        # on a problem where the floor binds it puts the hand adjoint orders
        # of magnitude away from the true VJP.
        if floored is not None:
            s_bar = mx.where(floored, mx.zeros_like(s_bar), s_bar)
        # S = h.ch + R
        h_bar = h_bar + s_bar[:, None] * ch
        ch_bar = ch_bar + s_bar[:, None] * h
        rv_cols.append(s_bar)
        # ch = Cp h
        cp_bar = cp_bar + outer(ch_bar, h)
        h_bar = h_bar + matvec(mx.swapaxes(cp, -1, -2), ch_bar)
        # Cp = sym(Phi P Phi^T + Q).  The forward symmetrises, so the
        # adjoint of Q -- and of P -- is the *symmetrised* Cp adjoint, not
        # Cp's raw one.  Phi's happens to be right either way because its
        # own formula already sums a term and its transpose, which is why
        # only dQ came out wrong when this was missed.
        cp_sym = sym(cp_bar)
        q_rows.append(cp_sym)
        pb = matmul(2.0 * cp_sym, matmul(p_i, cov_prev))
        p_bar = matmul(mx.swapaxes(p_i, -1, -2), matmul(cp_sym, p_i))
        p_bar = sym(p_bar)
        # mp = Phi mean
        phi_rows.append(pb + outer(mp_bar, mean_prev))
        m_bar = matvec(mx.swapaxes(p_i, -1, -2), mp_bar)

    return (mx.stack(phi_rows[::-1], axis=1), mx.stack(q_rows[::-1], axis=1),
            h_bar, mx.stack(y_cols[::-1], axis=1),
            mx.stack(rv_cols[::-1], axis=1), m_bar, p_bar)


def blocked_backward(phi, q, h, y, r_var, r_ref, cot, blk, s_floor=None):
    """The same adjoints, computed block by block -- the kernel's structure.

    Nothing per-observation is stored: each block replays its own ``blk``
    forward states from the state it inherits, sweeps back, and hands
    ``(mbar, Pbar)`` to the block before it.  This is what the Metal backward
    kernel does per thread, so verifying it here against
    :func:`filter_backward` validates the *scheme* before any kernel exists.

    It is sequential in the block index, which is the one structural problem
    left for the kernel: the forward blocks are independent, the backward
    ones are chained through ``(mbar, Pbar)``.  The way out is the same trick
    that made the forward parallel -- a block's backward sweep is an *affine*
    map on ``(mbar, Pbar)``, so the maps compose, and the composition is a
    cheap scan over the ~1e2 blocks. See the M4 notes in the README.
    """
    n, k, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dtype = phi.dtype
    starts = list(range(0, k, blk))

    # forward once, keeping only the state entering each block
    seeds, mean, cov = [], mx.zeros((n, m), dtype=dtype), mx.zeros((n, m, m), dtype=dtype)
    for s in starts:
        seeds.append((mean, cov))
        e = min(s + blk, k)
        st, _ = filter_forward(phi[:, s:e], q[:, s:e], h, y[:, s:e],
                              r_var[:, s:e], r_ref[s:e], (mean, cov),
                              None if s_floor is None else s_floor[s:e])
        mean, cov = st[-1]

    m_bar = mx.zeros((n, m), dtype=dtype)
    p_bar = mx.zeros((n, m, m), dtype=dtype)
    h_bar = mx.zeros((n, m), dtype=dtype)
    phi_p, q_p, y_p, rv_p = [], [], [], []
    for s, seed in zip(reversed(starts), reversed(seeds)):
        e = min(s + blk, k)
        sf = None if s_floor is None else s_floor[s:e]
        st, _ = filter_forward(phi[:, s:e], q[:, s:e], h, y[:, s:e],
                               r_var[:, s:e], r_ref[s:e], seed, sf)
        pb, qb, hb, yb, rb, m_bar, p_bar = _backward_span(
            phi[:, s:e], q[:, s:e], h, y[:, s:e], r_var[:, s:e], r_ref[s:e],
            st, cot, m_bar, p_bar, sf)
        phi_p.append(pb); q_p.append(qb); y_p.append(yb); rv_p.append(rb)
        h_bar = h_bar + hb
    return (mx.concatenate(phi_p[::-1], axis=1),
            mx.concatenate(q_p[::-1], axis=1), h_bar,
            mx.concatenate(y_p[::-1], axis=1),
            mx.concatenate(rv_p[::-1], axis=1))
