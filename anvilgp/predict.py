"""Prediction: the GP's conditional mean and variance at new inputs.

``predict(gp, v, t_pred)`` -- also :meth:`GPLogLike.predict` -- returns the
GP component's posterior mean and variance at ``t_pred`` given the data, per
chain, in float64: what a detrending figure, a residual plot or a
posterior-predictive band needs.  The mean model and the noise are not
included (celerite2's ``include_mean=False``): the caller evaluates its own
``mean_fn`` on whatever covariates go with ``t_pred``, and adds noise if it
wants an observation-level band.

The method is the state-space smoother, O(N + M) in the data and
prediction points, over the merged, sorted grid: prediction points are
steps with no observation.  A Kalman filter forward, then the modified
Bryson-Frazier smoother backward -- the adjoint form ``(lambda, Lambda)``,
which needs no inverse of a predicted covariance (the RTS form's ``P^-1``
is singular at duplicate timestamps), and whose quantities are the
likelihood gradient's own: ``lambda`` is ``d log L / d (predicted mean)``.
At a step, ``E[x | y] = m_pred + P_pred lambda`` and ``Cov[x | y] = P_pred -
P_pred Lambda P_pred``.

It is deliberately off the likelihood path: a host-side loop in float64,
vectorised over chains (and over the columns below), so a figure's worth
of predictions is exact rather than float32, and the sampler's cost is
untouched.  Measured cost is in docs/general-features-plan.md (item 4).

Everything the likelihood models comes from the likelihood itself
(`GPLogLike.leaf_inputs`): the residual after ``mean_fn`` and offsets, and
the per-datum noise variance from ``yerr``, jitter or ``noise_fn``.  With a
``linear_basis`` the basis columns are smoothed beside the residual (one
covariance, ``1 + p`` means), and the prediction integrates the linear
parameters over their posterior (`GPLogLike.linear_posterior`):

    E[f* | y]   = s_r - s_Phi beta_hat
    Var[f* | y] = var + s_Phi C_beta s_Phi^T

with ``s_c`` the smoothed GP component of column ``c``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

__all__ = ["predict", "apply_inverse"]

#: Rough cap on the per-chunk working set of stored filter steps, bytes.
_CHUNK_BYTES = 512 * 2 ** 20


def predict(gp, v, t_pred, return_var: bool = True):
    """``(mean, var)``, each float64 ``(n_chains, M)``: the GP component's
    posterior at ``t_pred`` given the data, for each row of ``v``.

    ``t_pred`` is in the data's own coordinate (``t``), in any order; the
    outputs follow it.  A prediction point may coincide with a data point.
    ``var`` is ``None`` when ``return_var`` is false.  A
    :class:`~anvilgp.terms.TermConvolution` predicts the exposure-averaged
    process, whose state-space form holds only at spacings of at least the
    exposure width: every prediction point must be that far from every
    data point and from every other prediction point.
    """
    v = _check(gp, v)
    t_pred = np.asarray(t_pred, dtype=np.float64).ravel()
    if not np.all(np.isfinite(t_pred)):
        raise ValueError("t_pred must be finite")
    grid = _grid(gp, t_pred)
    n_chains = v.shape[0]
    k = 1 if gp.linear_basis is None else 1 + _n_basis(gp, v[:1])
    mean = np.empty((n_chains, t_pred.size))
    var = np.empty((n_chains, t_pred.size)) if return_var else None
    for c, vc in _chunks(gp, v, grid, k):
        out = _smooth(gp, vc, grid, return_var=return_var)
        mu, va = out["mean"][:, 0], out.get("var")
        if k > 1:
            beta, c_beta = gp.linear_posterior(mx.array(vc, dtype=mx.float64))
            s_phi = out["mean"][:, 1:]                          # (n, p, M)
            mu = mu - np.einsum("np,npm->nm", beta, s_phi)
            if return_var:
                va = va + np.einsum("npm,npq,nqm->nm", s_phi, c_beta, s_phi)
        mean[c:c + vc.shape[0]] = mu
        if return_var:
            var[c:c + vc.shape[0]] = va
    return mean, var


def apply_inverse(gp, v, Y):
    """``K^-1 Y`` in float64 for each row of ``v``: ``K`` the data's full
    covariance (GP plus the per-datum noise the likelihood uses), ``Y``
    ``(N,)``, ``(N, k)`` or ``(n_chains, N, k)``; the result is
    ``(n_chains, N)`` for a 1-D ``Y``, else ``(n_chains, N, k)``.

    From the same filter and smoother as :func:`predict`: the log-density
    ``l(y)`` of columns ``y`` has ``-dl/dy = K^-1 y``, and at observation
    ``t`` that is ``v_t / S_t - K_t^T lambda_t`` -- the innovation over its
    variance, less the gain against the smoother's adjoint of the filtered
    state.  No ``y - E[f | y]`` subtraction, so nothing cancels.  O(N k),
    value only (no gradient: see `GPLogLike.gram` for a differentiable
    quadratic form).
    """
    v = _check(gp, v)
    n_chains, N = v.shape[0], gp.n_data
    Y = np.asarray(Y, dtype=np.float64)
    flat = Y.ndim == 1
    if flat:
        Y = Y[:, None]
    if Y.ndim == 2:
        Y = np.broadcast_to(Y, (n_chains,) + Y.shape)
    if Y.ndim != 3 or Y.shape[:2] != (n_chains, N):
        raise ValueError(f"Y must be ({N},), ({N}, k) or ({n_chains}, {N}, "
                         f"k); got {np.shape(Y)}")
    grid = _grid(gp, np.zeros(0))
    out = np.empty(Y.shape)
    for c, vc in _chunks(gp, v, grid, Y.shape[-1]):
        cols = np.swapaxes(Y[c:c + vc.shape[0]], 1, 2)      # (n, k, N)
        out[c:c + vc.shape[0]] = _smooth(gp, vc, grid, return_var=False,
                                         cols=cols, alpha=True)["alpha"]
    return out[..., 0] if flat else out


def _check(gp, v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != gp.param_map.n_dim:
        raise ValueError(f"v must be (n_chains, {gp.param_map.n_dim}), "
                         f"got {v.shape}")
    return v


def _grid(gp, t_pred):
    """The merged grid of data and prediction points: ``(is_obs, data_pos,
    pred_slot, lag_u, lag_idx)``."""
    t_data = gp._t64
    n_data, n_pred = t_data.size, t_pred.size
    # a stable sort puts a data point before a prediction point at the
    # same time, so the prediction sees it through dt = 0
    t_all = np.concatenate([t_data, t_pred])
    order = np.argsort(t_all, kind="stable")
    t_all = t_all[order]
    is_obs = order < n_data
    data_pos = order[is_obs]                       # == arange(n_data)
    pred_slot = np.empty(n_pred, dtype=np.int64)   # merged index of each pred
    pred_slot[order[~is_obs] - n_data] = np.nonzero(~is_obs)[0]
    dt = np.concatenate([[0.0], np.diff(t_all)])
    if gp._formal:
        width = gp.term.delta
        near = dt < width
        near[0] = False
        # data-data spacings were checked at construction
        both_obs = is_obs & np.concatenate([[False], is_obs[:-1]])
        if np.any(near & ~both_obs):
            raise ValueError(
                f"TermConvolution has an exposure width of {width:g}: every "
                f"prediction point must be at least that far from the data "
                f"and from the other prediction points, where the "
                f"state-space form of the convolved kernel holds")
    lag_u, lag_idx = np.unique(dt[1:], return_inverse=True)
    return is_obs, data_pos, pred_slot, lag_u, lag_idx


def _chunks(gp, v, grid, k):
    """``(start, rows)`` over chains, each chunk's stored steps within
    `_CHUNK_BYTES`."""
    m, T = gp.term.n_state, grid[0].size
    per_chain = 8 * T * (k * m + m * m + 2 * m + k + 2)
    step = max(1, min(v.shape[0], _CHUNK_BYTES // max(per_chain, 1)))
    for c in range(0, v.shape[0], step):
        yield c, v[c:c + step]


def _n_basis(gp, v1):
    with mx.stream(mx.cpu):
        cols, _ = _columns(gp, mx.array(v1, dtype=mx.float64))
    return cols.shape[1] - 1


def _columns(gp, v64):
    """``((n, k, N) columns, r_var (n, N))`` in float64: the residual, then
    the basis columns if any, and the per-datum noise variance."""
    pm = gp.param_map
    N = gp.n_data
    a64 = lambda x: mx.array(x, dtype=mx.float64)     # noqa: E731
    kp = pm.kernel_params(v64)
    get = gp._matrices_fn(kp, a64(gp._dt64), gp._uniq64)
    _, _, _, r, r_var, _, _ = gp.leaf_inputs(
        v64, a64(gp._dt64), a64(gp._y64), a64(gp._yerr64 ** 2),
        None if gp._x_mean64 is None else a64(gp._x_mean64),
        gp._inst32, 0, N, get)
    cols = [r[:, None, :]]
    if gp.linear_basis is not None:
        vm = v64[:, pm.mean] if pm.mean is not None else v64[:, :0]
        if gp.mean_fn is not None and pm.mean is not None:
            m0 = gp.mean_fn(vm, a64(gp._x_mean64))
        else:
            m0 = mx.zeros((v64.shape[0], N), dtype=mx.float64)
        phi = gp.linear_basis(vm, a64(gp._noise_x64), m0).astype(mx.float64)
        phi = mx.broadcast_to(phi, (v64.shape[0], N, phi.shape[-1]))
        cols.append(mx.swapaxes(phi, 1, 2))
    return mx.concatenate(cols, axis=1), r_var


def _smooth(gp, v, grid, return_var, cols=None, alpha=False):
    """The filter and smoother over ``grid`` for one chunk of chains.

    Returns ``{"mean": (n, k, M)}`` -- each column's smoothed GP component
    at the prediction points -- plus ``"var"`` ``(n, M)`` with
    ``return_var``, and ``"alpha"`` ``(n, N, k)``, ``K^-1`` of the columns,
    with ``alpha``.  ``cols`` ``(n, k, N)`` replaces the likelihood's own
    (the residual, then any basis).
    """
    is_obs, data_pos, pred_slot, lag_u, lag_idx = grid
    pm = gp.param_map
    n = v.shape[0]
    with mx.stream(mx.cpu):
        v64 = mx.array(v, dtype=mx.float64)
        own, r_var = _columns(gp, v64)
        if cols is None:
            cols = own
        kp = pm.kernel_params(v64)
        phi_u, q_u, h = gp.term.matrices(kp, mx.array(lag_u,
                                                      dtype=mx.float64))
        pin = gp._p_inf(kp, mx.float64, dense=True) if gp._has_p_inf else None
        corr = gp.term.diag_correction(kp)
        mx.eval(cols, r_var, phi_u, q_u, h)
        cols, r_var = np.asarray(cols, dtype=np.float64), np.array(r_var)
        phi_u, q_u, h = np.array(phi_u), np.array(q_u), np.array(h)
        p_inf = (np.broadcast_to(np.eye(h.shape[-1]), (n,) + (h.shape[-1],) * 2)
                 if pin is None else np.array(pin))
        corr = None if corr is None else np.array(corr)[:, 0]
    m, k = h.shape[-1], cols.shape[1]
    T = is_obs.size
    phi_ut = np.ascontiguousarray(np.swapaxes(phi_u, -1, -2))
    cols_t = np.ascontiguousarray(np.swapaxes(cols, 1, 2))     # (n, N, k)

    # Columns ride as the trailing axis of the means, (n, m, k), so every
    # product below is a batched `@`: `np.einsum` on arrays this small is
    # dominated by its own overhead, and this loop is overhead-bound.
    # ---- forward: Kalman filter over the merged grid ----------------------
    hc = h[:, :, None]                                  # (n, m, 1)
    hr = h[:, None, :]                                  # (n, 1, m)
    eye = np.eye(m)[None]
    mp_s = np.empty((T, n, m, k))
    cp_s = np.empty((T, n, m, m))
    gain_s = np.zeros((T, n, m, 1))
    inn_s = np.zeros((T, n, 1, k))
    sv_s = np.ones((T, n, 1, 1))
    mean = np.zeros((n, m, k))
    cov = np.zeros((n, m, m))
    obs_i = 0
    for i in range(T):
        if i == 0:
            mp = np.zeros((n, m, k))
            cp = p_inf.copy()
        else:
            li = lag_idx[i - 1]
            ph = phi_u[:, li]
            mp = ph @ mean
            cp = ph @ cov @ phi_ut[:, li] + q_u[:, li]
            cp = 0.5 * (cp + np.swapaxes(cp, 1, 2))
        mp_s[i], cp_s[i] = mp, cp
        if is_obs[i]:
            j = data_pos[obs_i]
            obs_i += 1
            ch = cp @ hc                                # (n, m, 1)
            sv = hr @ ch + r_var[:, j, None, None]      # (n, 1, 1)
            gain = ch / sv
            inn = cols_t[:, j:j + 1, :] - hr @ mp       # (n, 1, k)
            mean = mp + gain @ inn
            # Joseph form: (I - K h^T) cp (I - K h^T)^T + r K K^T
            a = eye - gain @ hr
            cov = (a @ cp @ np.swapaxes(a, 1, 2)
                   + r_var[:, j, None, None] * (gain @ np.swapaxes(gain, 1, 2)))
            gain_s[i], inn_s[i], sv_s[i] = gain, inn, sv
        else:
            mean, cov = mp, cp

    # ---- backward: modified Bryson-Frazier smoother ------------------------
    lam = np.zeros((n, m, k))
    Lam = np.zeros((n, m, m))
    want = np.full(T, -1, dtype=np.int64)
    want[pred_slot] = np.arange(pred_slot.size)
    f_mean = np.empty((n, k, pred_slot.size))
    f_var = np.empty((n, pred_slot.size)) if return_var else None
    hh = hc @ hr
    alpha_out = np.empty((n, cols.shape[2], k)) if alpha else None
    obs_i = cols.shape[2]
    for i in range(T - 1, -1, -1):
        if is_obs[i]:
            gain, sv = gain_s[i], sv_s[i]
            if alpha:
                # -d log L / d y_t: the innovation over S, less the gain
                # against lambda, which here is still the adjoint of the
                # filtered state at t
                obs_i -= 1
                alpha_out[:, data_pos[obs_i]] = (
                    inn_s[i] / sv - np.swapaxes(gain, 1, 2) @ lam)[:, 0, :]
            at = eye - hc @ np.swapaxes(gain, 1, 2)      # (I - K h^T)^T
            lam = hc @ (inn_s[i] / sv) + at @ lam
            Lam = hh / sv + at @ Lam @ np.swapaxes(at, 1, 2)
        p = want[i]
        if p >= 0:
            mp, cp = mp_s[i], cp_s[i]
            f_mean[:, :, p] = (hr @ (mp + cp @ lam))[:, 0, :]
            if return_var:
                f_var[:, p] = (hr @ (cp - cp @ Lam @ cp) @ hc)[:, 0, 0]
        if i > 0:
            li = lag_idx[i - 1]
            lam = phi_ut[:, li] @ lam
            Lam = phi_ut[:, li] @ Lam @ phi_u[:, li]

    out = {"mean": f_mean}
    if return_var:
        va = np.maximum(f_var, 0.0)
        if corr is not None:              # the convolution's zero-lag excess
            va = va + corr[:, None]
        out["var"] = va
    if alpha:
        out["alpha"] = alpha_out
    return out
