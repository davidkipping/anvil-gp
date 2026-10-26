"""Oracles for the kernel's block boundary quantities (docs/boundary-plan.md).

The kernel's backward starts every block from an *entry state* ``(m, C)``
and a *seed* ``(m_bar, P_bar)``, both composed across blocks.  At
float32 on bright, densely sampled data the dense information form gets
both wrong (docs/float32.md, section 7, step 2).  This module holds:

* :func:`exact_boundary` -- the float64 truth: a sequential filter's state
  at each block boundary, and the reverse-mode adjoint of that state.
* :func:`sqrt_boundary` -- the fix, in NumPy: block elements in the
  square-root information form, each block centred on its own start state
  (block-local, as a kernel thread can do it), folded per block, then a
  prefix and a suffix.  float32 by default; float64 reproduces
  :func:`exact_boundary`, which is the algebra check.
* :func:`boundary_from` -- runs the kernel's own gradient with the boundary
  quantities replaced, so a test can ask what the float32 backward gives
  from any of them.

Taken from ``benchmarks/boundary_prototype.py``, which measured it; this
copy is the reference the Metal implementation is validated against, stage
by stage.  Arrays follow the kernel's layout: ``(n_chains, n_blocks, m[,
m])``, residuals ``(n_chains, npts)``.
"""

from contextlib import contextmanager

import mlx.core as mx
import numpy as np

import anvilgp.metal as am


def T(x):
    return np.swapaxes(x, -1, -2)


# ------------------------------------------------------------------ truth
def exact_entries(pt, qt, lag, h, res, r, blk):
    """float64 filter: the (mean, cov) entering each block."""
    nc, m = pt.shape[0], pt.shape[-1]
    nb = -(-res.shape[1] // blk)
    em = np.zeros((nc, nb, m))
    eC = np.zeros((nc, nb, m, m))
    for c in range(nc):
        mean = np.zeros(m)
        cov = np.zeros((m, m))
        for t in range(res.shape[1]):
            if t % blk == 0:
                em[c, t // blk] = mean
                eC[c, t // blk] = cov
            P, Q = pt[c, lag[t]], qt[c, lag[t]]
            mp = P @ mean
            cp = P @ cov @ P.T + Q
            cp = 0.5 * (cp + cp.T)
            ch = cp @ h[c]
            s = max(ch @ h[c], 0.0) + r[c, t]
            k = ch / s
            mean = mp + k * (res[c, t] - mp @ h[c])
            cov = cp - np.outer(k, ch)
            cov = 0.5 * (cov + cov.T)
    return em, eC


def exact_seeds(pt, qt, lag, h, res, r, r_ref, cot, blk):
    """The seeds: reverse-mode adjoints of the float64 filter's state at
    each block boundary (MLX float64 on the CPU).  The last block's is 0."""
    nc, m = pt.shape[0], pt.shape[-1]
    n = res.shape[1]
    nb = -(-n // blk)
    with mx.stream(mx.cpu):
        f64 = lambda x: mx.array(np.asarray(x, np.float64), dtype=mx.float64)
        Pa, Qa, H, R, RV, RR, C0 = (f64(pt), f64(qt), f64(h), f64(res.T),
                                    f64(r.T), f64(r_ref), f64(cot))

        def F(dm, dC):
            mean = mx.zeros((nc, m), dtype=mx.float64)
            cov = mx.zeros((nc, m, m), dtype=mx.float64)
            ll = mx.zeros((nc,), dtype=mx.float64)
            for t in range(n):
                if t % blk == 0 and t > 0:
                    mean = mean + dm[:, t // blk - 1]
                    cov = cov + dC[:, t // blk - 1]
                P, Q = Pa[:, int(lag[t])], Qa[:, int(lag[t])]
                mp = (P @ mean[..., None])[..., 0]
                cp = P @ cov @ mx.swapaxes(P, -1, -2) + Q
                cp = 0.5 * (cp + mx.swapaxes(cp, -1, -2))
                ch = (cp @ H[..., None])[..., 0]
                s = mx.maximum((ch * H).sum(-1), 0) + RV[t]
                v = R[t] - (mp * H).sum(-1)
                k = ch / s[:, None]
                mean = mp + k * v[:, None]
                cov = cp - k[:, :, None] * ch[:, None, :]
                ll = ll - 0.5 * (mx.log(s / RR[t]) + v * v / s)
            return (C0 * ll).sum()

        gm, gC = mx.grad(F, argnums=(0, 1))(
            mx.zeros((nc, nb, m), dtype=mx.float64),
            mx.zeros((nc, nb, m, m), dtype=mx.float64))
        mx.eval(gm, gC)
    gm, gC = np.array(gm), np.array(gC)
    return gm, 0.5 * (gC + T(gC))


def exact_boundary(pt, qt, lag, h, res, r, r_ref, cot, blk):
    em, eC = exact_entries(pt, qt, lag, h, res, r, blk)
    sm, sC = exact_seeds(pt, qt, lag, h, res, r, r_ref, cot, blk)
    return em, eC, sm, sC


# ------------------------------------------------------- square-root form
def tria(M):
    """Lower-triangular L with L L^T = M M^T (QR of M^T)."""
    return T(np.linalg.qr(T(M), mode="r"))


def sqrt_q(qt, dt=np.float32):
    """A lower Cholesky factor of every Q table entry, in ``dt``, with each
    pivot clamped at zero and no floor -- what a kernel does.

    No floor, deliberately: at 20-s cadence Q's eigenvalues reach 1e-9 of
    its trace, below ``eps * trace``, so the usual relative floor rewrites
    exactly the directions that matter at high s/e.  It cost the seeds
    20-50x here (7.8e-4 -> 4.8e-5 relative), against a float64
    eigendecomposition that this matches.  Q is PSD for every term but a
    convolved one, which keeps the dense path (docs/boundary-plan.md).
    """
    q = (0.5 * (qt + T(qt))).astype(dt)
    out = np.zeros_like(q)
    m = q.shape[-1]
    for j in range(m):
        d = q[..., j, j] - (out[..., j, :j] ** 2).sum(-1)
        out[..., j, j] = np.sqrt(np.maximum(d, dt(0)))
        piv = out[..., j, j]
        safe = np.where(piv > 0, piv, dt(1))
        for i in range(j + 1, m):
            a = q[..., i, j] - (out[..., i, :j] * out[..., j, :j]).sum(-1)
            out[..., i, j] = np.where(piv > 0, a / safe, dt(0))
    return out


def leaves_sqrt(phi, lq, h, y, r):
    """Square-root information leaves ``(A, b, U, eta, Z)``; C = U U^T,
    J = Z Z^T.  phi, lq: (nc, n, m, m); h: (nc, m); y, r: (nc, n)."""
    nc, n, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dt = phi.dtype
    M = np.zeros((nc, n, 1 + m, 1 + m), dt)
    M[..., 0, 0] = np.sqrt(r)
    M[..., 0, 1:] = np.einsum("ci,cnij->cnj", h, lq)
    M[..., 1:, 1:] = lq
    L = tria(M)
    sS = L[..., 0, 0]
    K = L[..., 1:, 0] / sS[..., None]          # sign of sS cancels
    hphi = np.einsum("ci,cnij->cnj", h, phi)
    A = phi - K[..., :, None] * hphi[..., None, :]
    b = K * y[..., None]
    Z = np.zeros((nc, n, m, m), dt)
    Z[..., :, 0] = hphi / np.abs(sS)[..., None]
    eta = hphi * (y / (sS * sS))[..., None]
    return [A, b, L[..., 1:, 1:], eta, Z]


def _xi(U, Z):
    m = U.shape[-1]
    sh = U.shape[:-2]
    eye = np.broadcast_to(np.eye(m, dtype=U.dtype), sh + (m, m))
    M = np.concatenate([np.concatenate([T(U) @ Z, eye], -1),
                        np.concatenate([Z, np.zeros(sh + (m, m), U.dtype)], -1)],
                       -2)
    X = tria(M)
    return X[..., :m, :m], X[..., m:, :m], X[..., m:, m:]


def combine_sqrt(ei, ej):
    """Element i then element j, square-root information form."""
    Ai, bi, Ui, ni, Zi = ei
    Aj, bj, Uj, nj, Zj = ej
    X11, X21, X22 = _xi(Ui, Zj)
    ginv = lambda v: v - Ui @ np.linalg.solve(T(X11), T(X21) @ v)   # (I+CJ)^-1
    hinv = lambda v: v - X21 @ np.linalg.solve(X11, T(Ui) @ v)      # (I+JC)^-1
    Aij = Aj @ ginv(Ai)
    bij = (Aj @ ginv((bi + (Ui @ (T(Ui) @ nj[..., None]))[..., 0])[..., None])
           )[..., 0] + bj
    Uij = tria(np.concatenate([T(np.linalg.solve(X11, T(Ui) @ T(Aj))), Uj], -1))
    d = nj - (Zj @ (T(Zj) @ bi[..., None]))[..., 0]
    nij = (T(Ai) @ hinv(d[..., None]))[..., 0] + ni
    Zij = tria(np.concatenate([T(Ai) @ X22, Zi], -1))
    return [Aij, bij, Uij, nij, Zij]


def identity_sqrt(shape, m, dt):
    z = np.zeros(shape + (m, m), dt)
    return [np.broadcast_to(np.eye(m, dtype=dt), shape + (m, m)).copy(),
            np.zeros(shape + (m,), dt), z, np.zeros(shape + (m,), dt), z.copy()]


def seeds_sqrt(mu, U, suf, cot):
    """``(m_bar, P_bar)`` from a block's exit state ``(mu, U U^T)`` and the
    suffix after it, the closed form of `adjoint.block_seeds`."""
    Zs, ns = suf[4], suf[3]
    X11, X21, X22 = _xi(U, Zs)
    d = ns - (Zs @ (T(Zs) @ mu[..., None]))[..., 0]
    rr = (d[..., None] - X21 @ np.linalg.solve(X11, T(U) @ d[..., None]))[..., 0]
    jt = X22 @ T(X22)
    return (cot[..., None] * rr,
            0.5 * cot[..., None, None] * (rr[..., :, None] * rr[..., None, :] - jt))


def replay(pt, qt, lag, h, res, r, ent_m, ent_C, blk, dt=np.float32):
    """The filter through each block from its entry state, in ``dt`` (the
    kernel's `_FWD_STEP`, Joseph form): filtered means, (nc, npts, m)."""
    nc, m = pt.shape[0], pt.shape[-1]
    n = res.shape[1]
    nb = ent_m.shape[1]
    out = np.zeros((nc, n, m), dt)
    mean = ent_m.astype(dt).copy()
    cov = ent_C.astype(dt).copy()
    ci = np.arange(nc)[:, None]
    eye = np.eye(m, dtype=dt)
    for i in range(blk):
        idx = np.arange(nb) * blk + i
        ok = idx < n
        idx = np.minimum(idx, n - 1)
        P = pt[ci, lag[idx][None, :]].astype(dt)
        Q = qt[ci, lag[idx][None, :]].astype(dt)
        mp = np.einsum("cbij,cbj->cbi", P, mean)
        cp = P @ cov @ T(P) + Q
        cp = 0.5 * (cp + T(cp))
        ch = np.einsum("cbij,cj->cbi", cp, h.astype(dt))
        rv = r[:, idx].astype(dt)
        s = np.maximum(np.einsum("cbi,ci->cb", ch, h.astype(dt)), 0) + rv
        v = res[:, idx].astype(dt) - np.einsum("cbi,ci->cb", mp, h.astype(dt))
        k = ch / s[..., None]
        nm = mp + k * v[..., None]
        a = eye - k[..., :, None] * h.astype(dt)[:, None, None, :]
        nC = a @ cp @ T(a) + rv[..., None, None] * k[..., :, None] * k[..., None, :]
        nC = 0.5 * (nC + T(nC))
        mean = np.where(ok[None, :, None], nm, mean)
        cov = np.where(ok[None, :, None, None], nC, cov)
        out[:, idx[ok]] = nm[:, ok]
    return out


def sqrt_blocks(pt, qt, lag, h, res, r, start_m, start_C, blk,
                dt=np.float32, lq=None):
    """Block elements ``[A, b, U, eta, Z]``, each ``(nc, nb, ...)``, from
    block-local centred square-root leaves -- what stage 2's kernel emits.

    ``start_m``, ``start_C``: approximate entry states (the kernel's own
    prefix serves; zero does not).  Each block's leaves are centred on the
    filter replayed from its own start state: residual ``y_t - h Phi
    m_{t-1}`` and ``b' = A m_{t-1} + K y_t - m_t``, so the block element
    describes the correction to that trajectory.  A block's output is then
    relative to its own trajectory end, the next block's input to *its*
    start, and the difference is moved into the output ``b`` (``eta`` and
    ``J`` untouched).  Requires ``npts % blk == 0``.  ``lq``: a factor
    table of ``qt`` to use instead of :func:`sqrt_q` (the kernel's own).
    """
    nc, m = pt.shape[0], pt.shape[-1]
    n = res.shape[1]
    nb = n // blk
    assert nb * blk == n, "the oracle wants whole blocks"
    M = replay(pt, qt, lag, h, res, r, start_m, start_C, blk, dt)
    Mprev = np.concatenate([np.zeros((nc, 1, m), dt), M[:, :-1]], axis=1)
    Mprev[:, ::blk] = start_m.astype(dt)            # each block's own start
    phi = pt[:, lag].astype(dt)
    pm = np.einsum("cnij,cnj->cni", phi, Mprev)
    resid = res.astype(dt) - np.einsum("cni,ci->cn", pm, h.astype(dt))
    lq = sqrt_q(qt, dt) if lq is None else np.asarray(lq).astype(dt)
    e = leaves_sqrt(phi, lq[:, lag], h.astype(dt), resid, r.astype(dt))
    e[1] = e[1] + (pm - M)
    e = [x.reshape((nc, nb, blk) + x.shape[2:]) for x in e]
    acc = [x[:, :, 0] for x in e]
    for i in range(1, blk):
        acc = combine_sqrt(acc, [x[:, :, i] for x in e])
    # block k's output, re-expressed relative to block k+1's start
    acc[1] = acc[1].copy()
    acc[1][:, :-1] += M[:, blk - 1:-1:blk] - start_m[:, 1:].astype(dt)
    return acc


def sqrt_scan(blocks, start_m, start_C, cot, dt=np.float32):
    """Entry states and seeds from square-root block elements: a prefix and
    a suffix per chain -- what stage 3's kernel does."""
    acc = [np.asarray(x).astype(dt) for x in blocks]
    nc, nb, m = acc[1].shape
    blkel = lambda k: [x[:, k] for x in acc]
    pre = [blkel(0)]
    for k in range(1, nb):
        pre.append(combine_sqrt(pre[-1], blkel(k)))
    suf = [identity_sqrt((nc,), m, dt)]
    for k in range(nb - 1, -1, -1):
        suf.append(combine_sqrt(blkel(k), suf[-1]))
    suf = suf[::-1]                         # suf[k] = blocks k..; suf[nb] = I
    em = np.zeros((nc, nb, m))
    eC = np.zeros((nc, nb, m, m))
    sm = np.zeros((nc, nb, m))
    sC = np.zeros((nc, nb, m, m))
    em[:, 0], eC[:, 0] = start_m[:, 0], start_C[:, 0]
    for k in range(nb):
        if k >= 1:
            em[:, k] = start_m[:, k] + pre[k - 1][1]
            eC[:, k] = pre[k - 1][2] @ T(pre[k - 1][2])
        sm[:, k], sC[:, k] = seeds_sqrt(pre[k][1], pre[k][2], suf[k + 1],
                                        np.asarray(cot).astype(dt))
    return em, eC, sm, sC


def sqrt_boundary(pt, qt, lag, h, res, r, cot, start_m, start_C, blk,
                  dt=np.float32, lq=None):
    """Entry states and seeds: :func:`sqrt_blocks`, then :func:`sqrt_scan`."""
    blocks = sqrt_blocks(pt, qt, lag, h, res, r, start_m, start_C, blk, dt, lq)
    return sqrt_scan(blocks, start_m, start_C, cot, dt)


# ------------------------------------------------- through the kernel
@contextmanager
def boundary_from(source):
    """Run the kernel's gradient with its boundary quantities replaced.

    ``source(inputs, kernel)`` gets the backward's inputs as NumPy arrays
    (dict: ``pt, qt, lag, h, res, r, r_ref, cot, blk``, with ``r`` the
    per-observation variance including the jitter, and ``mx`` the kernel's
    own arguments ``(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2)``)
    and the kernel's ``(ent_m, ent_C, seed_m, seed_C)``, and returns the
    four to use; it may also just record them.
    """
    orig = am.backward_ckpt

    def patched(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot,
                ent_m, ent_C, seed_m, seed_C, *rest, **kw):
        blk = rest[-1]
        j2 = np.array(jit2)
        rr = np.array(r_ref)
        if j2.ndim == 3:            # per datum, (npts, n_chains, 1)
            jv = j2[:, :, 0].T
        elif j2.shape[1] == 0:
            jv = np.zeros((phi_tab.shape[0], 1), np.float32)
        elif j2.shape[1] == 1:
            jv = j2
        else:
            jv = j2[:, np.array(inst)]
        inputs = dict(pt=np.array(phi_tab), qt=np.array(q_tab),
                      lag=np.array(lag), h=np.array(hvec),
                      res=np.array(resid).T, r=rr[None, :] + jv,
                      r_ref=rr, cot=np.array(cot), blk=int(blk),
                      mx=(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2))
        kernel = tuple(np.array(x) for x in (ent_m, ent_C, seed_m, seed_C))
        new = source(inputs, kernel)
        new = [mx.array(np.asarray(x), dtype=mx.float32) for x in new]
        return orig(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot,
                    *new, *rest, **kw)

    am.backward_ckpt = patched
    try:
        yield
    finally:
        am.backward_ckpt = orig


def rel_err(got, want, last_seed=False):
    """max |got - want| / max |want|, the measure docs/float32.md quotes.
    Seeds skip the last block, whose exact seed is 0."""
    if last_seed:
        got, want = got[:, :-1], want[:, :-1]
    return float(np.abs(got - want).max() / np.abs(want).max())
