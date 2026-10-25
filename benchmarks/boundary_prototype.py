"""Prototype of the kernel's block boundary quantities in the square-root
information form, centred on a float32 replay (docs/float32.md s7, step 2).

The reference the Metal implementation is to be validated against.  Entry
states and backward seeds are built by NumPy (float32 = LAPACK sgeqrf /
sgesv) and handed to the kernel's own backward, whose gradient is compared
with float64 central differences.

usage: boundary_prototype.py BLK [local | start]
  BLK          block length on the 20-s, N = 4096, SHO + SHO, s/e = 100 problem
  (no flag)    dense vs square-root, plain vs centred, plus float64 checks
  local        block-local centring (implementable: reads only its own block)
  start        centring from different trajectory starts
  SQRTQ=f32chol  factor Q by float32 Cholesky (as a kernel would)
"""

import sys, warnings, numpy as np, mlx.core as mx, anvilgp as ag
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from floor_exposure import sample, hi, posterior_draws, TWO_PI, GQ
import anvilgp.metal as am
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
warnings.simplefilter("ignore")

def exact_entries(pt, qt, lag, h, res, rr, jv, blk, nb):
    """float64 filter: (mean, cov) entering each block, (nc, nb, m[, m])."""
    nc, m = pt.shape[0], pt.shape[-1]
    em = np.zeros((nc, nb, m)); eC = np.zeros((nc, nb, m, m))
    for c in range(nc):
        mean = np.zeros(m); cov = np.zeros((m, m))
        for t in range(len(lag)):
            if t % blk == 0:
                em[c, t // blk] = mean; eC[c, t // blk] = cov
            P = pt[c, lag[t]]; Q = qt[c, lag[t]]
            mp = P @ mean; cp = P @ cov @ P.T + Q; cp = 0.5 * (cp + cp.T)
            ch = cp @ h[c]; s = max(ch @ h[c], 0) + rr[t] + jv[c]
            v = res[c, t] - mp @ h[c]; k = ch / s
            mean = mp + k * v; cov = cp - np.outer(k, ch); cov = 0.5 * (cov + cov.T)
    return em, eC

def adj_seeds(pt, qt, lag, h, res, rr, jv, cot, blk, nb):
    """Seeds as reverse-mode adjoints of the float64 filter state at each block boundary."""
    nc, m = pt.shape[0], pt.shape[-1]
    with mx.stream(mx.cpu):
        P_all = mx.array(pt); Q_all = mx.array(qt); H = mx.array(h); R = mx.array(res.T); RR = mx.array(rr)
        JV = mx.array(jv); C0 = mx.array(cot)
        def F(dm, dC):
            mean = mx.zeros((nc, m), dtype=mx.float64); cov = mx.zeros((nc, m, m), dtype=mx.float64)
            ll = mx.zeros((nc,), dtype=mx.float64)
            for t in range(len(lag)):
                if t % blk == 0 and t > 0:
                    k = t // blk - 1
                    mean = mean + dm[:, k]; cov = cov + dC[:, k]
                P = P_all[:, int(lag[t])]; Q = Q_all[:, int(lag[t])]
                mp = (P @ mean[..., None])[..., 0]
                cp = P @ cov @ mx.swapaxes(P, -1, -2) + Q
                cp = 0.5 * (cp + mx.swapaxes(cp, -1, -2))
                ch = (cp @ H[..., None])[..., 0]
                s = mx.maximum((ch * H).sum(-1), 0) + RR[t] + JV
                v = R[t] - (mp * H).sum(-1)
                kk = ch / s[:, None]
                mean = mp + kk * v[:, None]
                cov = cp - kk[:, :, None] * ch[:, None, :]
                ll = ll - 0.5 * (mx.log(s / RR[t]) + v * v / s)
            return (C0 * ll).sum()
        dm = mx.zeros((nc, nb, m), dtype=mx.float64); dC = mx.zeros((nc, nb, m, m), dtype=mx.float64)
        gm, gC = mx.grad(F, argnums=(0, 1))(dm, dC)
        mx.eval(gm, gC)
    gm, gC = np.array(gm), np.array(gC)
    return gm, 0.5 * (gC + np.swapaxes(gC, -1, -2))

def replay_means(pt, qt, lag, h, res, rr, jv, ent_m, ent_C, blk):
    """float32 filter through each block from its entry state; filtered means (nc, n, m)."""
    nc, m = pt.shape[0], pt.shape[-1]; n = res.shape[1]; nb = ent_m.shape[1]
    M = np.zeros((nc, n, m), np.float32)
    mean, cov = ent_m.astype(np.float32).copy(), ent_C.astype(np.float32).copy()
    ci = np.arange(nc)[:, None]
    for i in range(blk):
        idx = np.arange(nb) * blk + i
        ok = idx < n; idx = np.minimum(idx, n - 1)
        P = pt[ci, lag[idx][None, :]]; Q = qt[ci, lag[idx][None, :]]
        mp = np.einsum("cbij,cbj->cbi", P, mean)
        cp = P @ cov @ np.swapaxes(P, -1, -2) + Q; cp = 0.5 * (cp + np.swapaxes(cp, -1, -2))
        ch = np.einsum("cbij,cj->cbi", cp, h)
        s = np.maximum(np.einsum("cbi,ci->cb", ch, h), 0) + rr[idx][None, :] + jv[:, None]
        v = res[:, idx] - np.einsum("cbi,ci->cb", mp, h)
        kg = ch / s[..., None]
        a_ = np.eye(m, dtype=np.float32) - kg[..., :, None] * h[:, None, None, :]
        cn = a_ @ cp @ np.swapaxes(a_, -1, -2) + (rr[idx][None, :] + jv[:, None])[..., None, None] * kg[..., :, None] * kg[..., None, :]
        mean = np.where(ok[None, :, None], mp + kg * v[..., None], mean)
        cov = np.where(ok[None, :, None, None], 0.5 * (cn + np.swapaxes(cn, -1, -2)), cov)
        M[:, idx[ok]] = mean[:, ok]
    return M


T = lambda x: np.swapaxes(x, -1, -2)
def tria(M):
    """Lower-triangular L with L L^T = M M^T, by QR of M^T (M has at least as many columns as rows)."""
    return T(np.linalg.qr(T(M), mode="r"))

# ---------------------------------------------------------------- dense form
def leaves_dense(phi, q, h, y, r):
    """phi, q: (nc, n, m, m); h: (nc, m); y, r: (nc, n)."""
    hphi = np.einsum("ci,cnij->cnj", h, phi)                  # h Phi
    qh = np.einsum("cnij,cj->cni", q, h)                      # Q h^T
    S = np.einsum("cni,ci->cn", qh, h) + r
    K = qh / S[..., None]
    A = phi - K[..., :, None] * hphi[..., None, :]
    b = K * y[..., None]
    C = q - qh[..., :, None] * qh[..., None, :] / S[..., None, None]
    J = hphi[..., :, None] * hphi[..., None, :] / S[..., None, None]
    eta = hphi * (y / S)[..., None]
    return [A, b, 0.5 * (C + T(C)), eta, J]

def combine_dense(ei, ej):
    Ai, bi, Ci, ni, Ji = ei; Aj, bj, Cj, nj, Jj = ej
    I = np.eye(Ai.shape[-1], dtype=Ai.dtype)
    Mx = I + Ci @ Jj; Ny = I + Jj @ Ci
    Aij = Aj @ np.linalg.solve(Mx, Ai)
    bij = (Aj @ np.linalg.solve(Mx, (bi + (Ci @ nj[..., None])[..., 0])[..., None]))[..., 0] + bj
    Cij = Aj @ np.linalg.solve(Mx, Ci) @ T(Aj) + Cj
    nij = (T(Ai) @ np.linalg.solve(Ny, (nj - (Jj @ bi[..., None])[..., 0])[..., None]))[..., 0] + ni
    Jij = T(Ai) @ np.linalg.solve(Ny, Jj) @ Ai + Ji
    return [Aij, bij, 0.5 * (Cij + T(Cij)), nij, 0.5 * (Jij + T(Jij))]

def ident_dense(shape, m, dt):
    z = np.zeros(shape + (m, m), dt)
    return [np.broadcast_to(np.eye(m, dtype=dt), shape + (m, m)).copy(), np.zeros(shape + (m,), dt), z, np.zeros(shape + (m,), dt), z.copy()]

def seeds_dense(mu, pre, suf, cot):
    Sig = pre[2]; Js, ns = suf[4], suf[3]
    I = np.eye(Sig.shape[-1], dtype=Sig.dtype)
    d = ns - (Js @ mu[..., None])[..., 0]
    r = np.linalg.solve(T(I + Sig @ Js), d[..., None])[..., 0]       # G^T d, G = (I + Sig J)^-1
    jt = np.linalg.solve(I + Js @ Sig, Js); jt = 0.5 * (jt + T(jt))
    return cot[..., None] * r, 0.5 * cot[..., None, None] * (r[..., :, None] * r[..., None, :] - jt)

def cov_dense(e): return e[2]

# ---------------------------------------------------------------- square-root form
def leaves_sqrt(phi, lq, h, y, r):
    """lq: a square root of Q, (nc, n, m, m)."""
    nc, n, m = phi.shape[0], phi.shape[1], phi.shape[-1]
    dt = phi.dtype
    hl = np.einsum("ci,cnij->cnj", h, lq)                     # h L_Q, (nc, n, m)
    M = np.zeros((nc, n, 1 + m, 1 + m), dt)
    M[..., 0, 0] = np.sqrt(r); M[..., 0, 1:] = hl; M[..., 1:, 1:] = lq
    L = tria(M)
    sS = L[..., 0, 0]                                           # sign-agnostic below
    kb = L[..., 1:, 0]                                          # K sqrt(S) (same sign as sS)
    K = kb / sS[..., None]
    U = L[..., 1:, 1:]
    hphi = np.einsum("ci,cnij->cnj", h, phi)
    S = sS * sS
    A = phi - K[..., :, None] * hphi[..., None, :]
    b = K * y[..., None]
    Z = np.zeros((nc, n, m, m), dt); Z[..., :, 0] = hphi / np.abs(sS)[..., None]
    eta = hphi * (y / S)[..., None]
    return [A, b, U, eta, Z]

def combine_sqrt(ei, ej):
    Ai, bi, Ui, ni, Zi = ei; Aj, bj, Uj, nj, Zj = ej
    m = Ai.shape[-1]; dt = Ai.dtype; sh = Ai.shape[:-2]
    I = np.broadcast_to(np.eye(m, dtype=dt), sh + (m, m))
    M = np.concatenate([np.concatenate([T(Ui) @ Zj, I], -1),
                        np.concatenate([Zj, np.zeros(sh + (m, m), dt)], -1)], -2)
    X = tria(M); X11, X21, X22 = X[..., :m, :m], X[..., m:, :m], X[..., m:, m:]
    ginv = lambda v: v - Ui @ np.linalg.solve(T(X11), T(X21) @ v)          # (I + C_i J_j)^-1 v
    hinv = lambda v: v - X21 @ np.linalg.solve(X11, T(Ui) @ v)             # (I + J_j C_i)^-1 v
    Aij = Aj @ ginv(Ai)
    bij = (Aj @ ginv((bi + (Ui @ (T(Ui) @ nj[..., None]))[..., 0])[..., None]))[..., 0] + bj
    W = T(np.linalg.solve(X11, T(Ui) @ T(Aj)))                              # A_j U_i X11^-T
    Uij = tria(np.concatenate([W, Uj], -1))
    d = nj - (Zj @ (T(Zj) @ bi[..., None]))[..., 0]
    nij = (T(Ai) @ hinv(d[..., None]))[..., 0] + ni
    Zij = tria(np.concatenate([T(Ai) @ X22, Zi], -1))
    return [Aij, bij, Uij, nij, Zij]

def ident_sqrt(shape, m, dt): return ident_dense(shape, m, dt)

def seeds_sqrt(mu, pre, suf, cot):
    U = pre[2]; Zs, ns = suf[4], suf[3]
    m = U.shape[-1]; dt = U.dtype; sh = U.shape[:-2]
    I = np.broadcast_to(np.eye(m, dtype=dt), sh + (m, m))
    M = np.concatenate([np.concatenate([T(U) @ Zs, I], -1),
                        np.concatenate([Zs, np.zeros(sh + (m, m), dt)], -1)], -2)
    X = tria(M); X11, X21, X22 = X[..., :m, :m], X[..., m:, :m], X[..., m:, m:]
    d = ns - (Zs @ (T(Zs) @ mu[..., None]))[..., 0]
    r = (d[..., None] - X21 @ np.linalg.solve(X11, T(U) @ d[..., None]))[..., 0]   # G^T d
    jt = X22 @ T(X22)
    return cot[..., None] * r, 0.5 * cot[..., None, None] * (r[..., :, None] * r[..., None, :] - jt)

def cov_sqrt(e): return e[2] @ T(e[2])

FORMS = {"dense": (leaves_dense, combine_dense, ident_dense, seeds_dense, cov_dense),
         "sqrt": (leaves_sqrt, combine_sqrt, ident_sqrt, seeds_sqrt, cov_sqrt)}

import os
def sqrt_q(qt):
    """A square root of each Q table entry.  SQRTQ=f32chol: float32 Cholesky with a relative
    diagonal floor (what a kernel would do); default: float64 eigendecomposition, cast."""
    if os.environ.get("SQRTQ") == "f32chol":
        q32 = (0.5 * (qt + T(qt))).astype(np.float32)
        out = np.zeros_like(q32)
        for idx in np.ndindex(q32.shape[:-2]):
            a = q32[idx]; fl = np.float32(1e-7) * max(np.trace(a), np.float32(1e-30))
            for _ in range(8):
                try:
                    out[idx] = np.linalg.cholesky(a + fl * np.eye(a.shape[-1], dtype=np.float32)); break
                except np.linalg.LinAlgError:
                    fl *= 10
        return out
    w, V = np.linalg.eigh(0.5 * (qt + T(qt)).astype(np.float64))
    return (V * np.sqrt(np.clip(w, 0, None))[..., None, :]).astype(qt.dtype)

def boundary(pt, qt, lag, h, res, rr, jv, cot, ent_m, ent_C, blk, form, dt, centre, local=False):
    """Entry states and seeds from one pass of the chosen element algebra."""
    leaves, comb, ident, seeds, cov = FORMS[form]
    nc, m = pt.shape[0], pt.shape[-1]; n = res.shape[1]; nb = n // blk
    if centre:
        M = replay_means(pt, qt, lag, h, res, rr, jv, ent_m, ent_C, blk).astype(np.float64)
    else:
        M = np.zeros((nc, n, m))
    Mprev = np.concatenate([np.zeros((nc, 1, m)), M[:, :-1]], axis=1)
    if local and centre:
        # block-local: each block centres on its own start; no other block's trajectory is read
        for k in range(1, nb):
            Mprev[:, k * blk] = ent_m[:, k]
    phi64 = pt[:, lag].astype(np.float64)
    pm = np.einsum("cnij,cnj->cni", phi64, Mprev)
    # the centred residual and b-shift are O(1) differences, formed in the working dtype
    resid = (res.astype(dt) - np.einsum("cni,ci->cn", pm.astype(dt), h.astype(dt)))
    phi = pt[:, lag].astype(dt)
    q = (sqrt_q(qt) if form == "sqrt" else qt)[:, lag].astype(dt)
    r = (rr[None, :] + jv[:, None]).astype(dt)
    e = leaves(phi, q, h.astype(dt), resid, r)
    e[1] = e[1] + (pm.astype(dt) - M.astype(dt))
    e = [x.reshape((nc, nb, blk) + x.shape[2:]) for x in e]
    acc = [x[:, :, 0] for x in e]
    for i in range(1, blk):
        acc = comb(acc, [x[:, :, i] for x in e])
    if local and centre:
        # block k-1's output is relative to its trajectory end; block k's input to block k's
        # start.  Re-express k-1's OUTPUT (a shift of b only, no change to eta / J).
        acc[1] = acc[1].copy()
        for k in range(1, nb):
            acc[1][:, k - 1] = acc[1][:, k - 1] + (M[:, k * blk - 1] - ent_m[:, k]).astype(dt)
    pre = [[x[:, 0] for x in acc]]
    for k in range(1, nb):
        pre.append(comb(pre[-1], [x[:, k] for x in acc]))
    suf = [ident((nc,), m, dt)]
    for k in range(nb - 1, -1, -1):
        suf.append(comb([x[:, k] for x in acc], suf[-1]))
    suf = suf[::-1]                                             # suf[k] = blocks k..end; suf[nb] = identity
    new_m = ent_m.astype(np.float64).copy(); new_C = ent_C.astype(np.float64).copy()
    sm = np.zeros((nc, nb, m)); sC = np.zeros((nc, nb, m, m))
    for k in range(nb):
        if k >= 1:
            new_m[:, k] = (ent_m[:, k] if (local and centre) else M[:, k * blk - 1]) + pre[k - 1][1]
            new_C[:, k] = cov(pre[k - 1])
        a, c_ = seeds(pre[k][1], pre[k], suf[k + 1], cot.astype(dt))
        sm[:, k], sC[:, k] = a, c_
    return new_m, new_C, sm, sC

CACHE, MODE, LOG = {}, {}, {}
_orig = am.backward_ckpt
def patched(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot, ent_m, ent_C, seed_m, seed_C, *rest, **kw):
    if MODE.get("passthrough"):
        return _orig(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot, ent_m, ent_C, seed_m, seed_C, *rest, **kw)
    blk = rest[-1]; nb = ent_m.shape[1]
    pt, qt = np.array(phi_tab), np.array(q_tab); lg = np.array(lag); h = np.array(hvec)
    res = np.array(resid).T; rr = np.array(r_ref); cv = np.array(cot)
    jv = np.array(jit2)[:, 0] if jit2.shape[1] else np.zeros(pt.shape[0], np.float32)
    km, kC = np.array(ent_m), np.array(ent_C)
    if MODE.get("noref"):
        em, eC, sm, sC = boundary(pt, qt, lg, h, res, rr, jv, cv, km, kC, blk,
                                  MODE["form"], MODE["dt"], MODE["centre"])
        return _orig(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot,
                     mx.array(em, dtype=mx.float32), mx.array(eC, dtype=mx.float32),
                     mx.array(sm, dtype=mx.float32), mx.array(sC, dtype=mx.float32), *rest, **kw)
    if "ex" not in CACHE:
        f64 = lambda x: x.astype(np.float64)
        CACHE["ex"] = exact_entries(f64(pt), f64(qt), lg, f64(h), f64(res), f64(rr), f64(jv), blk, nb)
        CACHE["adj"] = adj_seeds(f64(pt), f64(qt), lg, f64(h), f64(res), f64(rr), f64(jv), f64(cv), blk, nb)
    xm, xC = CACHE["ex"]; am_, aC = CACHE["adj"]
    if MODE["form"] == "kernel":
        em, eC, sm, sC = km, kC, np.array(seed_m), np.array(seed_C)
    elif MODE["form"] == "exact":
        em, eC, sm, sC = xm, xC, am_, aC
    else:
        start = MODE.get("start", "kernel")
        if start == "kernel":
            sm0, sC0 = km, kC
        elif start == "sqrtplain":
            sm0, sC0, _, _ = boundary(pt, qt, lg, h, res, rr, jv, cv, km, kC, blk, "sqrt", np.float32, False)
        elif start == "zero":                       # zero means, stationary (whitened: identity) covariance
            sm0 = np.zeros_like(km); sC0 = np.broadcast_to(np.eye(km.shape[-1], dtype=np.float32), kC.shape).copy()
            sC0[:, 0] = kC[:, 0]
        em, eC, sm, sC = boundary(pt, qt, lg, h, res, rr, jv, cv, sm0, sC0, blk,
                                  MODE["form"], MODE["dt"], MODE["centre"], MODE.get("local", False))
        for _ in range(MODE.get("extra", 0)):
            em, eC, sm, sC = boundary(pt, qt, lg, h, res, rr, jv, cv, em, eC, blk,
                                      MODE["form"], MODE["dt"], MODE["centre"], MODE.get("local", False))
    LOG["ent_m"] = np.abs(em - xm).max() / np.abs(xm).max()
    LOG["ent_C"] = np.abs(eC - xC).max() / np.abs(xC).max()
    LOG["seed"] = np.abs(sm[:, :-1] - am_[:, :-1]).max() / np.abs(am_).max()
    return _orig(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot,
                 mx.array(em, dtype=mx.float32), mx.array(eC, dtype=mx.float32),
                 mx.array(sm, dtype=mx.float32), mx.array(sC, dtype=mx.float32), *rest, **kw)
am.backward_ckpt = patched

if __name__ == "__main__":
    blk = int(sys.argv[1])
    t = np.arange(4096) / 4320; kp = [0.1, 3/TWO_PI, 5.0, 0.03, 0.2, GQ]
    term = ag.SHOTerm() + ag.SHOTerm(); yerr = np.full(len(t), 1e-3)
    y = sample(term, kp, t, yerr, np.random.default_rng(1)); d = len(kp)
    pm_ = ParamMap(kernel=[KernelParam(link="log", index=i, ref=float(x)) for i, x in enumerate(kp)], n_dim=d)
    ref = GPLogLike(None, t, None, y, yerr, term, pm_, GPPolicy(kappa_max=None))
    V, sd = posterior_draws(ref, d, 8); V = V[:3]
    G = []
    for v in V:
        P = np.repeat(v[None], 2 * d, 0)
        for i in range(d):
            P[2*i, i] += 1e-5; P[2*i+1, i] -= 1e-5
        ff = hi(ref, P); G.append((ff[0::2] - ff[1::2]) / 2e-5)
    G = np.array(G)
    gp = GPLogLike(None, t, None, y, yerr, term, pm_, GPPolicy(metal=True, kappa_max=None, metal_block=blk))
    RUNS = [("kernel as is", dict(form="kernel")), ("exact entries + seeds", dict(form="exact"))]
    if len(sys.argv) > 2 and sys.argv[2] == "local":
        RUNS += [(f"local sqrt centred, from {st}" + (f" +{x} round" if x else ""),
                  dict(form="sqrt", dt=np.float32, centre=True, start=st, extra=x, local=True))
                 for st in ("kernel", "sqrtplain", "zero") for x in (0, 1)]
    if len(sys.argv) > 2 and sys.argv[2] == "start":
        RUNS += [(f"sqrt centred, from {st}" + (f" +{x} round" if x else ""),
                  dict(form="sqrt", dt=np.float32, centre=True, start=st, extra=x))
                 for st in ("kernel", "sqrtplain", "zero") for x in (0, 1)]
    for form in (("dense", "sqrt") if len(sys.argv) <= 2 else ()):
        RUNS.append((f"{form} float64 (algebra check)", dict(form=form, dt=np.float64, centre=False)))
    for form in (("dense", "sqrt") if len(sys.argv) <= 2 else ()):
        for cen in (False, True):
            RUNS.append((f"{form} float32 {'centred' if cen else 'plain'}", dict(form=form, dt=np.float32, centre=cen)))
    for name, mode in RUNS:
        MODE.clear(); MODE.update(mode); LOG.clear()
        with mx.stream(mx.gpu):
            g = np.array(mx.grad(lambda u: gp(u).sum())(mx.array(V, dtype=mx.float32)))
        e = (np.abs(g - G) * sd).max(0)
        print(f"blk {blk} {name:30s} dg*sd {np.round(e, 3)}  ent_m {LOG['ent_m']:.1e} ent_C {LOG['ent_C']:.1e} seed {LOG['seed']:.1e}", flush=True)
