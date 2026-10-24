"""Prototype of the proposed fix for the residual float32 floor
(docs/float32.md, section 7): take log L from a float32 Kalman filter
replayed through each kernel block, started from that block's entry state.

The entry states are the ``(b, C)`` of the float32 prefix over the kernel's
own block elements (``elements.prefix_scan``, as the backward already uses).
A filtered mean and covariance are O(1) numbers, and every replay term is an
O(1) innovation ``-1/2 (log S + v^2 / S)``, so the dense-information ``ell``
-- the ~1e4 numbers whose cancellation is the floor -- is never used.

This is a NumPy float32 emulation of the replay, not the kernel, and the
value only.  Columns: the default MLX path, the kernel, the replay, and a
single float32 filter over the whole series (no blocks), all against the
float64 ``.hi`` at the posterior draws of ``floor_exposure.py``.

    python benchmarks/block_replay_prototype.py
"""

import os
import sys
import warnings

import mlx.core as mx
import numpy as np

import anvilgp as ag
import anvilgp.metal as amet
from anvilgp.elements import prefix_scan
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_exposure import GQ, TWO_PI, hi, posterior_draws, sample  # noqa: E402

_CAP = {}
_scan_blocks = amet.scan_blocks


def _spy(*a, **k):
    """Keep the kernel's inputs and block elements from the last call."""
    el = _scan_blocks(*a, **k)
    _CAP["args"], _CAP["el"] = a, el
    return el


amet.scan_blocks = _spy


def replay(blocks=True):
    """float32 log L of the last kernel call, by per-block filter replay."""
    phi_t, q_t, lag, _, hvec, resid, r_ref = _CAP["args"][:7]
    blk = _CAP["args"][9]
    phi_t, q_t = np.array(phi_t), np.array(q_t)
    lag, h = np.array(lag), np.array(hvec)
    res, rr = np.array(resid).T, np.array(r_ref)
    nc, m = phi_t.shape[0], phi_t.shape[-1]
    n = res.shape[1]
    if blocks:
        el = _CAP["el"]
        with mx.stream(mx.gpu):
            pre = prefix_scan(el)
            mx.eval(pre.b, pre.C)
        nb = el.b.shape[1]
    else:
        blk, nb = n, 1
    mean = np.zeros((nc, nb, m), np.float32)
    cov = np.zeros((nc, nb, m, m), np.float32)
    if blocks:                       # block j starts from prefix j - 1
        mean[:, 1:] = np.array(pre.b)[:, :-1]
        cov[:, 1:] = np.array(pre.C)[:, :-1]
    pad = nb * blk - n
    res = np.concatenate([res, np.zeros((nc, pad), np.float32)],
                         1).reshape(nc, nb, blk)
    lagp = np.concatenate([lag, np.zeros(pad, lag.dtype)]).reshape(nb, blk)
    rrp = np.concatenate([rr, np.ones(pad, np.float32)]).reshape(nb, blk)
    valid = (np.arange(nb * blk) < n).reshape(nb, blk)
    ell = np.zeros((nc, nb), np.float32)
    ci = np.arange(nc)[:, None]
    hh = h[:, None, :]
    for i in range(blk):
        P = phi_t[ci, lagp[None, :, i]]
        Q = q_t[ci, lagp[None, :, i]]
        mp = np.einsum("cbij,cbj->cbi", P, mean)
        cp = P @ cov @ np.swapaxes(P, -1, -2) + Q
        cp = 0.5 * (cp + np.swapaxes(cp, -1, -2))
        ch = np.einsum("cbij,cbj->cbi", cp, np.broadcast_to(hh, mp.shape))
        s = np.maximum(np.sum(hh * ch, -1), 0) + rrp[None, :, i]
        v = res[:, :, i] - np.sum(hh * mp, -1)
        kg = ch / s[..., None]
        ok = valid[None, :, i]
        mean = np.where(ok[..., None], mp + kg * v[..., None], mean)
        cn = cp - kg[..., :, None] * ch[..., None, :]
        cov = np.where(ok[..., None, None],
                       0.5 * (cn + np.swapaxes(cn, -1, -2)), cov)
        ell -= np.where(ok, 0.5 * (np.log(s / rrp[None, :, i]) + v * v / s),
                        0).astype(np.float32)
    return ell.sum(1, dtype=np.float32)


def case(label, t, term, kp, yerr, k=16):
    y = sample(term, kp, t, yerr, np.random.default_rng(1))
    d = len(kp)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=float(x))
                          for i, x in enumerate(kp)], n_dim=d)
    pol = dict(kappa_max=None)
    ref = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy(**pol))
    V, _ = posterior_draws(ref, d, k)
    fhi = hi(ref, V)
    gpd = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy(**pol))
    gpm = GPLogLike(None, t, None, y, yerr, term, pm,
                    GPPolicy(metal=True, **pol))
    out = {c: [] for c in ("default", "kernel", "replay", "one filter")}
    for v in V:
        vv = mx.array(v[None], dtype=mx.float32)
        with mx.stream(mx.gpu):
            out["default"].append(float(np.array(gpd(vv))[0]))
            out["kernel"].append(float(np.array(gpm(vv))[0]))
        out["replay"].append(float(replay(True)[0]))
        out["one filter"].append(float(replay(False)[0]))
    row = f"{label:34s} N={len(t):6d} |"
    for c, f in out.items():
        e = np.array(f) - fhi
        row += f" {c}: sd {e.std():.3f} max {np.abs(e).max():.3f} |"
    print(row, flush=True)


def main():
    warnings.simplefilter("ignore")
    sho2, sho = ag.SHOTerm() + ag.SHOTerm(), ag.SHOTerm()

    def tess(cad_min):
        n = int(13.2 * 1440 / cad_min)
        a = np.arange(n) * cad_min / 1440
        return np.concatenate([a, a + 14.2])

    for label, t, term, kp in [
        ("TESS 20-s, SHO+SHO, s/e=30", tess(1 / 3), sho2,
         [0.03, 3 / TWO_PI, 5.0, 0.009, 0.2, GQ]),
        ("TESS 2-min, SHO+SHO, s/e=100", tess(2), sho2,
         [0.1, 3 / TWO_PI, 5.0, 0.03, 0.2, GQ]),
        ("TESS 2-min, single SHO, s/e=300", tess(2), sho,
         [0.3, 3 / TWO_PI, 5.0]),
    ]:
        case(label, t, term, kp, np.full(len(t), 1e-3))


if __name__ == "__main__":
    main()
