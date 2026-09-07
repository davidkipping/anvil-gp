"""The sequential Kalman prediction-error filter -- an independent oracle.

Same marginal likelihood as :mod:`anvilgp.elements`, by a different
recursion: a left fold over observations carrying ``(mean, covariance)``
instead of a balanced reduction over affine-Gaussian elements.  In exact
arithmetic they agree identically, and the left fold of the element algebra
*is* this filter (the running composite keeps ``A = 0``, ``J = 0``,
``eta = 0``, so its ``(b, C, ell)`` are exactly this filter's mean,
covariance, and accumulated log-likelihood).  Implementing both and testing
that they agree is the cheapest strong check available on the element
algebra, and it is how the combine formula was validated here.

Its cost is a Python loop over data points, which makes it useless above
N ~ 1e4 (one MLX graph node per step): this is a *test* path, not a
production one.  The production reference is the blocked tree reduction.
"""

from __future__ import annotations

import mlx.core as mx

from .linalg import matvec

__all__ = ["sequential_loglike"]


def sequential_loglike(phi: mx.array, q: mx.array, h: mx.array, y: mx.array,
                       r_var: mx.array, r_ref: mx.array,
                       s_floor: mx.array | None = None) -> mx.array:
    """Reference-subtracted log-likelihood by the sequential filter.

    Arguments match :func:`anvilgp.elements.leaf_elements` exactly, so the
    same ``GPLogLike.leaf_inputs`` tuple drives both recursions and any
    disagreement is attributable to the recursion alone.  ``phi[:, 0]`` and
    ``q[:, 0]`` must already be ``0`` and ``P_inf = I`` for the first point.
    Returns ``(n_chains,)``.
    """
    n, k, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dtype = phi.dtype
    mean = mx.zeros((n, m), dtype=dtype)
    cov = mx.zeros((n, m, m), dtype=dtype)
    ell = mx.zeros((n,), dtype=dtype)

    for i in range(k):
        p_i = phi[:, i]
        m_pred = matvec(p_i, mean)
        c_pred = mx.matmul(p_i, mx.matmul(cov, mx.swapaxes(p_i, -1, -2))) + q[:, i]
        c_pred = 0.5 * (c_pred + mx.swapaxes(c_pred, -1, -2))

        ch = mx.sum(c_pred * h[:, None, :], axis=-1)          # (n, m)
        s = mx.sum(h * ch, axis=-1) + r_var[:, i]             # (n,)
        if s_floor is not None:
            s = mx.maximum(s, s_floor[i])
        v = y[:, i] - mx.sum(h * m_pred, axis=-1)
        gain = ch / s[:, None]
        mean = m_pred + gain * v[:, None]
        cov = c_pred - gain[:, :, None] * ch[:, None, :]
        cov = 0.5 * (cov + mx.swapaxes(cov, -1, -2))
        ell = ell - 0.5 * (mx.log(s / r_ref[i]) + v * v / s)
    return ell
