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

from .linalg import matmul, matvec, outer

__all__ = ["filter_forward", "filter_backward"]


def filter_forward(phi, q, h, y, r_var, r_ref):
    """Run the filter, keeping every state.  Returns (states, logL).

    ``states`` is the list of ``(m_n, P_n)`` *after* each update, plus the
    zero state it began from, which the backward sweep consumes in reverse.
    """
    n, k, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dt = phi.dtype
    mean = mx.zeros((n, m), dtype=dt)
    cov = mx.zeros((n, m, m), dtype=dt)
    ell = mx.zeros((n,), dtype=dt)
    states = [(mean, cov)]
    for i in range(k):
        p_i = phi[:, i]
        mp = matvec(p_i, mean)
        cp = matmul(p_i, matmul(cov, mx.swapaxes(p_i, -1, -2))) + q[:, i]
        cp = 0.5 * (cp + mx.swapaxes(cp, -1, -2))
        ch = matvec(cp, h)
        s = mx.sum(h * ch, axis=-1) + r_var[:, i]
        v = y[:, i] - mx.sum(h * mp, axis=-1)
        kg = ch / s[:, None]
        mean = mp + kg * v[:, None]
        cov = cp - outer(kg, ch)
        cov = 0.5 * (cov + mx.swapaxes(cov, -1, -2))
        ell = ell - 0.5 * (mx.log(s / r_ref[i]) + v * v / s)
        states.append((mean, cov))
    return states, ell


def filter_backward(phi, q, h, y, r_var, r_ref, states, cot):
    """Adjoints of ``sequential_loglike`` w.r.t. (phi, q, h, y, r_var).

    ``cot`` is the cotangent on the returned log-likelihood, shape (n,).
    Hand-derived; :mod:`tests.test_adjoint` checks every output against
    ``mx.vjp`` and against float64 central differences.
    """
    n, k, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dt = phi.dtype
    c = cot[:, None]

    phi_bar = mx.zeros_like(phi)
    q_bar = mx.zeros_like(q)
    h_bar = mx.zeros((n, m), dtype=dt)
    y_bar = mx.zeros((n, k), dtype=dt)
    rv_bar = mx.zeros((n, k), dtype=dt)

    m_bar = mx.zeros((n, m), dtype=dt)
    p_bar = mx.zeros((n, m, m), dtype=dt)
    phi_rows, q_rows, y_cols, rv_cols = [], [], [], []

    for i in range(k - 1, -1, -1):
        mean_prev, cov_prev = states[i]
        p_i = phi[:, i]
        # -- replay the forward quantities this step needs --
        mp = matvec(p_i, mean_prev)
        cp = matmul(p_i, matmul(cov_prev, mx.swapaxes(p_i, -1, -2))) + q[:, i]
        cp = 0.5 * (cp + mx.swapaxes(cp, -1, -2))
        ch = matvec(cp, h)
        s = mx.sum(h * ch, axis=-1) + r_var[:, i]
        v = y[:, i] - mx.sum(h * mp, axis=-1)
        kg = ch / s[:, None]

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
        cp_sym = 0.5 * (cp_bar + mx.swapaxes(cp_bar, -1, -2))
        q_rows.append(cp_sym)
        pb = matmul(2.0 * cp_sym, matmul(p_i, cov_prev))
        p_bar = matmul(mx.swapaxes(p_i, -1, -2), matmul(cp_sym, p_i))
        p_bar = 0.5 * (p_bar + mx.swapaxes(p_bar, -1, -2))
        # mp = Phi mean
        phi_rows.append(pb + outer(mp_bar, mean_prev))
        m_bar = matvec(mx.swapaxes(p_i, -1, -2), mp_bar)

    phi_bar = mx.stack(phi_rows[::-1], axis=1)
    q_bar = mx.stack(q_rows[::-1], axis=1)
    y_bar = mx.stack(y_cols[::-1], axis=1)
    rv_bar = mx.stack(rv_cols[::-1], axis=1)
    return phi_bar, q_bar, h_bar, y_bar, rv_bar
