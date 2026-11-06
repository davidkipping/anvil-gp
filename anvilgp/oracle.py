"""Float64 host-side oracles for adjudicating the GP likelihood.

Two independent references, used at different scales:

* :func:`dense_loglike` -- form the full covariance matrix from the term's
  own ``value_np`` and factor it with LAPACK.  O(N^3), so N <= ~2000, but it
  assumes *nothing* about the recursion and is the court of last resort.
  It is the oracle the brief reserves for the corners where celerite2 itself
  degrades: near-critical Q (celerite2's SHOTerm carries an ``eps`` to dodge
  the ``1/sqrt(4Q^2-1)`` singularity) and Matern-3/2 (which celerite2 can
  only *approximate*, since it is not celerite-representable -- its
  ``Matern32Term`` takes an explicit ``eps`` approximation parameter).

* :func:`celerite2_loglike` -- the fast fp64 CPU oracle for everything else.

The dense route also gives :func:`dense_predict`, the conditional mean used
for detrending figures: deliberately out of the hot path, exactly once per
plot rather than once per leapfrog step.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dense_covariance", "dense_loglike", "dense_predict", "sample_ssm",
           "celerite2_loglike", "dense_masked_covariance",
           "dense_masked_loglike"]


def dense_covariance(term, p, t, yerr, jitter=0.0) -> np.ndarray:
    """``K + Sigma`` in float64, built from the term's covariance function."""
    t = np.asarray(t, dtype=np.float64).ravel()
    tau = np.abs(t[:, None] - t[None, :])
    k = term.value_np(np.asarray(p, dtype=np.float64), tau.ravel())
    k = np.asarray(k, dtype=np.float64).reshape(tau.shape)
    return k + np.diag(np.asarray(yerr, dtype=np.float64) ** 2 + jitter ** 2)


def dense_loglike(term, p, t, y, yerr, jitter=0.0) -> float:
    """Absolute float64 log-likelihood by dense Cholesky.  O(N^3)."""
    k = dense_covariance(term, p, t, yerr, jitter)
    n = k.shape[0]
    lo = np.linalg.cholesky(k)
    alpha = np.linalg.solve(lo.T, np.linalg.solve(lo, np.asarray(y, np.float64)))
    return float(-0.5 * (np.asarray(y, np.float64) @ alpha)
                 - np.sum(np.log(np.diag(lo)))
                 - 0.5 * n * np.log(2.0 * np.pi))


def dense_predict(term, p, t, y, yerr, t_pred=None, jitter=0.0):
    """Conditional mean (and variance) of the GP at ``t_pred``.

    For detrending figures and residual plots.  Deliberately *not* part of
    the sampling path: an O(N^3) float64 solve is the right tool once per
    figure and the wrong tool once per leapfrog step.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    t_pred = t if t_pred is None else np.asarray(t_pred, np.float64).ravel()
    k = dense_covariance(term, p, t, yerr, jitter)
    lo = np.linalg.cholesky(k)
    alpha = np.linalg.solve(lo.T, np.linalg.solve(lo, np.asarray(y, np.float64)))
    tau = np.abs(t_pred[:, None] - t[None, :])
    ks = np.asarray(term.value_np(np.asarray(p, np.float64), tau.ravel()),
                    dtype=np.float64).reshape(tau.shape)
    mean = ks @ alpha
    v = np.linalg.solve(lo, ks.T)
    var = np.asarray(term.value_np(np.asarray(p, np.float64),
                                   np.zeros(1)), dtype=np.float64)[0]
    return mean, var - np.sum(v * v, axis=0)


def dense_masked_covariance(term, p, t, noise_var, mask) -> np.ndarray:
    """The held-out likelihood's covariance ``Sigma`` (`GPLogLike`'s
    ``held_out``), built directly in float64: ``K + W`` on S, the GP's own
    ``K_MS`` across, and ``K_MS C^-1 K_SM + W_M`` on M, where ``C = K_SS +
    W_S`` and ``W`` is ``diag(noise_var)``.  ``mask`` is True on M."""
    t = np.asarray(t, dtype=np.float64).ravel()
    tau = np.abs(t[:, None] - t[None, :])
    k = np.asarray(term.value_np(np.asarray(p, np.float64), tau.ravel()),
                   dtype=np.float64).reshape(tau.shape)
    w = np.asarray(noise_var, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    s = ~m
    c = k[np.ix_(s, s)] + np.diag(w[s])
    k_ms = k[np.ix_(m, s)]
    out = k + np.diag(w)
    out[np.ix_(m, m)] = k_ms @ np.linalg.solve(c, k_ms.T) + np.diag(w[m])
    return out


def dense_masked_loglike(term, p, t, r, noise_var, mask) -> float:
    """Absolute float64 ``log N(r; 0, Sigma)`` for
    :func:`dense_masked_covariance`, by dense Cholesky.  O(N^3)."""
    sig = dense_masked_covariance(term, p, t, noise_var, mask)
    lo = np.linalg.cholesky(sig)
    a = np.linalg.solve(lo, np.asarray(r, np.float64))
    return float(-0.5 * a @ a - np.sum(np.log(np.diag(lo)))
                 - 0.5 * sig.shape[0] * np.log(2.0 * np.pi))


def celerite2_loglike(c2term, t, y, yerr, jitter=0.0) -> float:
    """Absolute float64 log-likelihood from celerite2 (test-only import)."""
    import celerite2

    gp = celerite2.GaussianProcess(c2term, mean=0.0)
    gp.compute(np.asarray(t, np.float64),
               yerr=np.sqrt(np.asarray(yerr, np.float64) ** 2 + jitter ** 2))
    return float(gp.log_likelihood(np.asarray(y, np.float64)))


def sample_ssm(term, p, dt, n, rng):
    """Draw ``n`` on-model observations from a term's own state-space model.

    **Not valid for a :class:`~anvilgp.terms.TermConvolution`.** It draws
    ``x_0 ~ N(0, I)``, which assumes ``P_inf = I``, and it clips negative
    eigenvalues when it factors ``Q`` -- and a convolved term's ``Q`` is
    only formally a covariance, with genuinely negative directions.  Both
    assumptions fail silently, giving on-model-looking data from the wrong
    process.  For a convolved term the on-model sampler is the dense one:
    :func:`dense_covariance` on its exact piecewise ``value_np``, then a
    Cholesky.

    O(n) and exact in float64, so it reaches the N where float32 error in
    the reduction becomes visible.  :func:`dense_covariance` plus a Cholesky
    is O(n^3) and stops a couple of thousand points short of that.

    On-model data is the honest input for a precision test.  Off-model
    residuals are large and the quadratic form dominates; on model the
    log-determinant and the quadratic form very nearly cancel, and what
    survives that cancellation is what a sampler actually sees.  Regular
    cadence, one lag, so ``dt`` is a scalar.
    """
    import mlx.core as mx
    # float64 does not exist on Metal, and the default stream is the GPU in
    # any normal session, so pin this to the CPU rather than asking every
    # caller to remember.
    with mx.stream(mx.cpu):
        pp = mx.array([list(p)], dtype=mx.float64)
        phi, q, h = term.matrices(pp, mx.array([dt], dtype=mx.float64))
        mx.eval(phi, q, h)
    Phi = np.array(phi, dtype=np.float64)[0, 0]
    Q = np.array(q, dtype=np.float64)[0, 0]
    H = np.array(h, dtype=np.float64)[0]
    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
    LQ = V * np.sqrt(np.maximum(w, 0.0))
    m = Phi.shape[0]
    x = rng.standard_normal(m)
    out = np.empty(n)
    for i in range(n):
        out[i] = H @ x
        x = Phi @ x + LQ @ rng.standard_normal(m)
    return out
