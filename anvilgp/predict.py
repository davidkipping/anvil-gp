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
parameters over their posterior, from the same filter (`linear_posterior`):

    E[f* | y]   = s_r - s_Phi beta_hat
    Var[f* | y] = var + s_Phi C_beta s_Phi^T

with ``s_c`` the smoothed GP component of column ``c``.

With points held out (`GPLogLike`'s ``held_out``) the GP is conditioned on
S alone: the smoother runs over S, a prediction is S's conditional, and
the linear parameters' posterior comes from the masked Gram, S's own plus
``(Z_M - P Z_S)^T W_M^-1 (Z_M - P Z_S)`` with ``P Z_S`` the smoothed
columns at M (`_posterior_gram`).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

__all__ = ["predict", "apply_inverse", "linear_posterior"]

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
    held = gp._held is not None
    # held out with a basis: the posterior needs a second run, with M as
    # its prediction points; chunked for whichever run is larger
    g_grid = _gram_grid(gp) if held and k > 1 else None
    size = (grid if g_grid is None or g_grid[2].size <= grid[2].size
            else g_grid)
    mean = np.empty((n_chains, t_pred.size))
    var = np.empty((n_chains, t_pred.size)) if return_var else None
    for c, vc in _chunks(gp, v, size, k, return_var):
        out = _smooth(gp, vc, grid, return_var=return_var,
                      gram=k > 1 and not held)
        mu, va = out["mean"][:, 0], out.get("var")
        if k > 1:
            # The posterior from this run's own innovations: polarising the
            # float64 likelihood for it instead was two-thirds of the cost
            # (1.16 of 1.71 s at N = 1e4, 64 chains).
            beta, c_beta = _beta_posterior(gp, _posterior_gram(
                gp, out if not held else _smooth(gp, vc, g_grid, False,
                                                 gram=True)))
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
    if gp._held is not None:
        raise ValueError("apply_inverse is not implemented for a likelihood "
                         "with held-out points (Sigma^-1 Y); gram gives "
                         "Y^T Sigma^-1 Y")
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
    for c, vc in _chunks(gp, v, grid, Y.shape[-1], False):
        cols = np.swapaxes(Y[c:c + vc.shape[0]], 1, 2)      # (n, k, N)
        out[c:c + vc.shape[0]] = _smooth(gp, vc, grid, return_var=False,
                                         cols=cols, alpha=True)["alpha"]
    return out[..., 0] if flat else out


def linear_posterior(gp, v):
    """The marginalised linear parameters' conditional posterior for each
    row of ``v``: ``(mean (n_chains, p), cov (n_chains, p, p))``, float64.

    From one forward pass of the float64 filter over the residual and the
    basis columns (`_smooth` with no prediction points, so no backward
    pass): the Gram matrix ``[r Phi]^T K^-1 [r Phi]`` is the sum over
    observations of the whitened innovations' outer products, and the
    posterior is the usual Gaussian algebra on it (`_beta_posterior`).
    Agrees with polarising the float64 likelihood, its previous route, to
    ~1e-11, and is 3-5x faster.  Under a flat prior a rank-deficient basis
    has no posterior and gets NaN; under a Gaussian one it has.
    """
    if gp.linear_basis is None:
        raise ValueError("no linear_basis: nothing is marginalised")
    v = _check(gp, v)
    grid = _gram_grid(gp)
    k = 1 + _n_basis(gp, v[:1])
    mean = np.empty((v.shape[0], k - 1))
    cov = np.empty((v.shape[0], k - 1, k - 1))
    for c, vc in _chunks(gp, v, grid, k, False):
        b, cv = _beta_posterior(gp, _posterior_gram(
            gp, _smooth(gp, vc, grid, return_var=False, gram=True)))
        if gp._basis_a is not None:       # gamma -> beta = A gamma
            a = gp._basis_a
            b, cv = b @ a.T, a @ cv @ a.T
        mean[c:c + vc.shape[0]], cov[c:c + vc.shape[0]] = b, cv
    return mean, cov


def _gram_grid(gp):
    """The grid for the Gram matrix: no prediction points, or with points
    held out, the held-out points (whose smoothed columns it needs)."""
    if gp._held is None:
        return _grid(gp, np.zeros(0))
    return _grid(gp, gp._t64[gp._held["m_pos_np"]])


def _posterior_gram(gp, out):
    """The Gram matrix of the columns under the likelihood's covariance,
    from a `_smooth` run with ``gram`` over `_gram_grid`: the filter's own,
    plus with points held out ``D^T W_M^-1 D``, ``D = Z_M - P Z_S``."""
    if gp._held is None:
        return out["gram"]
    z_m, w_m = out["held"]
    d = z_m - out["mean"]                                   # (n, k, M)
    return out["gram"] + np.einsum("nkm,nlm->nkl", d / w_m[:, None, :], d)


def _beta_posterior(gp, gram):
    """``(mean, cov)`` of the linear parameters from the Gram matrix of
    ``[r Phi]``: ``C = Phi^T K^-1 Phi``, ``d = Phi^T K^-1 r``; flat prior
    ``C^-1 d`` and ``C^-1``, Gaussian ``N(mu, S)`` ``(C + S^-1)^-1 (d +
    S^-1 mu)`` and ``(C + S^-1)^-1``.  Solved on the unit-diagonal form
    (exact; it keeps columns of very different scale from costing
    digits), and NaN for a chain whose system is singular."""
    d, C = gram[:, 1:, 0], gram[:, 1:, 1:]
    if gp._linear_prior is not None:      # in the columns' own coordinates
        C = C + gp._prior_prec_eff
        d = d + gp._prior_prec_eff @ gp._prior_mu_eff
    sc = np.sqrt(np.maximum(np.diagonal(C, axis1=1, axis2=2), 1e-300))
    Cn = C / (sc[:, :, None] * sc[:, None, :])
    p = Cn.shape[-1]
    cov = np.full(C.shape, np.nan)
    ok = np.ones(C.shape[0], dtype=bool)
    try:
        np.linalg.cholesky(Cn)
    except np.linalg.LinAlgError:          # find the singular chains
        for i in range(C.shape[0]):
            try:
                np.linalg.cholesky(Cn[i])
            except np.linalg.LinAlgError:
                ok[i] = False
    if np.any(ok):
        inv = np.linalg.solve(Cn[ok], np.broadcast_to(np.eye(p),
                                                      Cn[ok].shape))
        cov[ok] = inv / (sc[ok, :, None] * sc[ok, None, :])
    mean = np.einsum("cij,cj->ci", cov, d)
    return mean, cov


def _check(gp, v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != gp.param_map.n_dim:
        raise ValueError(f"v must be (n_chains, {gp.param_map.n_dim}), "
                         f"got {v.shape}")
    return v


def _grid(gp, t_pred):
    """The merged grid of data and prediction points: ``(is_obs, data_pos,
    pred_slot, lag_u, lag_idx)``.  The data are S when points are held
    out."""
    t_data = (gp._t64 if gp._held is None
              else gp._t64[gp._held["s_pos_np"]])
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


def _chunks(gp, v, grid, k, return_var=True):
    """``(start, rows)`` over chains, each chunk's stored steps within
    `_CHUNK_BYTES`: per chain the predicted states at the M prediction
    points, each observation's gain, innovations and variance, and the
    columns."""
    m, T, M = gp.term.n_state, grid[0].size, grid[2].size
    N = T - M
    per_chain = 8 * (M * (k * m + m * m + k + 1)
                     + T * (m + k + 1) + N * (2 * k + 1))
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
    if gp.linear_basis is not None and gp._basis_q64 is not None:
        # shared and orthonormalised (`GPLogLike._cache_shared_basis`): the
        # posterior below is in its coordinates, gamma = A^-1 beta
        q = mx.array(gp._basis_q64.T, dtype=mx.float64)[None]
        cols.append(mx.broadcast_to(q, (v64.shape[0],) + q.shape[1:]))
    elif gp.linear_basis is not None:
        vm = v64[:, pm.mean] if pm.mean is not None else v64[:, :0]
        if gp.mean_fn is not None and pm.mean is not None:
            m0 = gp.mean_fn(vm, a64(gp._x_mean64))
        else:
            m0 = mx.zeros((v64.shape[0], N), dtype=mx.float64)
        phi = gp.linear_basis(vm, a64(gp._noise_x64), m0).astype(mx.float64)
        phi = mx.broadcast_to(phi, (v64.shape[0], N, phi.shape[-1]))
        cols.append(mx.swapaxes(phi, 1, 2))
    return mx.concatenate(cols, axis=1), r_var


def _smooth(gp, v, grid, return_var, cols=None, alpha=False, gram=False):
    """The filter and smoother over ``grid`` for one chunk of chains.

    Returns ``{"mean": (n, k, M)}`` -- each column's smoothed GP component
    at the prediction points -- plus ``"var"`` ``(n, M)`` with
    ``return_var``, ``"alpha"`` ``(n, N, k)``, ``K^-1`` of the columns,
    with ``alpha``, and ``"gram"`` ``(n, k, k)``, the columns' ``Z^T K^-1
    Z`` from the filter's innovations, with ``gram``.  ``cols`` ``(n, k,
    N)`` replaces the likelihood's own (the residual, then any basis).
    With no prediction points and no ``alpha`` the backward pass is
    skipped.  With points held out the filter runs over S, and ``"held"``
    is M's own columns ``(n, k, M)`` and white noise ``(n, M)``.
    """
    is_obs, data_pos, pred_slot, lag_u, lag_idx = grid
    pm = gp.param_map
    n = v.shape[0]
    with mx.stream(mx.cpu):
        v64 = mx.array(v, dtype=mx.float64)
        own, r_var = _columns(gp, v64)
        kp = pm.kernel_params(v64)
        phi_u, q_u, h = gp.term.matrices(kp, mx.array(lag_u,
                                                      dtype=mx.float64))
        pin = gp._p_inf(kp, mx.float64, dense=True) if gp._has_p_inf else None
        corr = gp.term.diag_correction(kp)
        mx.eval(own, r_var, phi_u, q_u, h)
        own, r_var = np.asarray(own, dtype=np.float64), np.array(r_var)
        phi_u, q_u, h = np.array(phi_u), np.array(q_u), np.array(h)
        p_inf = (np.broadcast_to(np.eye(h.shape[-1]), (n,) + (h.shape[-1],) * 2)
                 if pin is None else np.array(pin))
        corr = None if corr is None else np.array(corr)[:, 0]
    held = None
    if gp._held is not None:
        # M's white noise is the noise model's, without a convolution's
        # zero-lag correction (part of the GP), as in the likelihood
        s_pos, m_pos = gp._held["s_pos_np"], gp._held["m_pos_np"]
        held = (own[:, :, m_pos], r_var[:, m_pos]
                - (0.0 if corr is None else corr[:, None]))
        own, r_var = own[:, :, s_pos], r_var[:, s_pos]
    cols = own if cols is None else np.asarray(cols, dtype=np.float64)
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
    # Predicted states are read back only at the prediction points, so
    # only those are kept: 1.04 -> 0.37 MB a chain at N = 1e4, M = 1e3,
    # m = 2 (more chains a chunk), ~6x at m = 6.
    want = np.full(T, -1, dtype=np.int64)
    want[pred_slot] = np.arange(pred_slot.size)
    mp_s = np.empty((pred_slot.size, n, m, k))
    cp_s = np.empty((pred_slot.size, n, m, m))
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
        if want[i] >= 0:
            mp_s[want[i]], cp_s[want[i]] = mp, cp
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

    out = {} if held is None else {"held": held}
    if gram:
        # sum_t v_t v_t^T / S_t over the observations (the prediction
        # steps' innovations are zero)
        w = inn_s[:, :, 0, :] / np.sqrt(sv_s[:, :, 0, :])      # (T, n, k)
        out["gram"] = np.einsum("tnk,tnl->nkl", w, w)
    if pred_slot.size == 0 and not alpha:
        return out

    # ---- backward: modified Bryson-Frazier smoother ------------------------
    lam = np.zeros((n, m, k))
    Lam = np.zeros((n, m, m))
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
            if return_var:          # Lambda feeds the variance alone
                Lam = hh / sv + at @ Lam @ np.swapaxes(at, 1, 2)
        p = want[i]
        if p >= 0:
            mp, cp = mp_s[p], cp_s[p]
            f_mean[:, :, p] = (hr @ (mp + cp @ lam))[:, 0, :]
            if return_var:
                f_var[:, p] = (hr @ (cp - cp @ Lam @ cp) @ hc)[:, 0, 0]
        if i > 0:
            li = lag_idx[i - 1]
            lam = phi_ut[:, li] @ lam
            if return_var:
                Lam = phi_ut[:, li] @ Lam @ phi_u[:, li]

    out["mean"] = f_mean
    if return_var:
        va = np.maximum(f_var, 0.0)
        if corr is not None:              # the convolution's zero-lag excess
            va = va + corr[:, None]
        out["var"] = va
    if alpha:
        out["alpha"] = alpha_out
    return out
