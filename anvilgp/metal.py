"""Fused Metal kernel for the blocked scan -- Milestone 3's hot path.

The MLX reference in :mod:`anvilgp.elements` is bound by op count and by
*materialising* the scan-element array: at 1024 chains x 4096 points it
writes and re-reads ``3m^2 + 2m + 1`` numbers per observation at every step,
and measures ~40x below this machine's memory bandwidth -- i.e. it is not
moving data fast, it is moving it too many times.  A kernel that walks a
block of observations with the running element **in registers** writes none
of it: one element per *block* reaches memory instead of one per point.

This is Design C's pass 1.  Pass 2 -- composing the ~N/L block elements per
chain -- is a second kernel, `reduce_blocks`.  It was an MLX tree until
2026-09-20; those levels are launch-bound, costing a flat ~5.1 ms whatever
the block count, which was 79% of the whole path at N = 1e4.

What the kernel does *not* do, deliberately
-------------------------------------------
It does not evaluate the kernel-term algebra.  ``Phi`` and ``Q`` are handed
in as a **table indexed by lag**, built by the verified float64-parity code
in :mod:`anvilgp.terms`.  Two reasons:

* the cancellation-free special functions (``gamma3``, ``sincm1``, the
  damped products) are the most delicate numerics in the package and are
  pinned by tests against mpmath; re-deriving them in Metal Shading Language
  would put the one thing most likely to be silently wrong into the one
  place hardest to test;
* real photometry has one cadence plus one lag per gap, so the table has a
  handful of rows and lives in cache. Recomputing a propagator per point
  would be work the data does not require.

The first observation has no predecessor, which the table expresses rather
than the kernel special-casing: row 0 holds ``Phi = 0``, ``Q = P_inf = I``,
and ``lag_index[0] = 0``.  The generic leaf construction then produces
``A = 0``, ``J = 0``, ``eta = 0`` by itself.

Accumulation
------------
``ell`` is summed with Kahan compensation in-thread, per the float32
discipline: it carries both the reference-subtracted log-determinant and the
quadratic form, and a block of 1024 observations is 1024 sequential adds
with no tree to spread the error.
"""

from __future__ import annotations

import mlx.core as mx


from .elements import Element

__all__ = ["scan_blocks", "scan_loglike", "reduce_blocks",
           "backward_blocks", "metal_available", "MAX_STATE",
           "MAX_BWD_BLOCK"]

#: Largest state dimension the kernel accepts.  The running element is
#: ``3m^2 + 2m + 1`` floats held in registers -- 17 at m = 2, 57 at m = 4,
#: 209 at m = 8, which would spill.  m <= 4 covers SHOTerm, Matern-3/2,
#: Matern-5/2 and RotationTerm; larger sums stay on the MLX path until the
#: element is moved to threadgroup memory.
MAX_STATE = 4

_SRC = """
    // Flat grid, chains fastest.  Two reasons.  Adjacent lanes then hold
    // adjacent chains, so the residual read below is coalesced; and the
    // threadgroup is full whatever `n_blocks` is, where a
    // (n_blocks, n_chains) grid with a (min(n_blocks,32),1,1) group ran
    // 10-wide groups on a 10-block problem -- 31% of a SIMD width.
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    uint start = blk_i * (uint)blk_len;
    if (start >= (uint)npts) return;
    uint stop = min(start + (uint)blk_len, (uint)npts);

    // per-chain observation vector and jitter variance
    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    float jit2v = (n_jit > 0) ? jit2[chain * (uint)n_jit] : 0.0f;

    // running element, in registers
    float Am[MDIM][MDIM], Cm[MDIM][MDIM], Jm[MDIM][MDIM];
    float bv[MDIM], ev[MDIM];
    float ell = 0.0f, ell_c = 0.0f;          // Kahan accumulator
    bool first = true;

    for (uint t = start; t < stop; ++t) {
        uint li = (uint)lag[t];
        uint off = (chain * (uint)n_lags + li) * M * M;

        float Ph[MDIM][MDIM], Qm[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            for (uint j = 0; j < M; ++j) {
                Ph[i][j] = phitab[off + i * M + j];
                Qm[i][j] = qtab[off + i * M + j];
            }
        }

        // hphi_j = sum_i h_i Phi_ij ;  qh_i = sum_j Q_ij h_j
        float hphi[MDIM], qh[MDIM];
        for (uint j = 0; j < M; ++j) {
            float s = 0.0f;
            for (uint i = 0; i < M; ++i) s += hv[i] * Ph[i][j];
            hphi[j] = s;
        }
        float sq = 0.0f;
        for (uint i = 0; i < M; ++i) {
            float s = 0.0f;
            for (uint j = 0; j < M; ++j) s += Qm[i][j] * hv[j];
            qh[i] = s;
            sq += hv[i] * s;
        }
        // clamp the variance term, not the sum: h Q h^T is non-negative in
        // exact arithmetic and can round negative in float32, while rr and
        // the jitter beside it are exact and positive.  Leaves sq >= rr, so
        // the division below is always safe, and matches what
        // elements.leaf_elements clamps.  The sfloor line is a secondary
        // guard that cannot bind once this one is in place.
        float rr = r_ref[t];
        sq = max(sq, 0.0f);
        sq += rr + jit2v;
        sq = max(sq, sfloor * rr);              // PSD failure is defined

        float inv_s = 1.0f / sq;
        float inv_rs = metal::precise::rsqrt(sq);
        float yv = resid[t * (uint)n_chains + chain];

        // leaf, in the rank-one form: J2 = z z^T, eta2 = g z, b2 = kappa g
        float zz[MDIM], kk[MDIM];
        for (uint i = 0; i < M; ++i) {
            zz[i] = hphi[i] * inv_rs;
            kk[i] = qh[i] * inv_s;
        }
        float gg = yv * inv_rs;
        float A2[MDIM][MDIM], C2[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            for (uint j = 0; j < M; ++j) {
                A2[i][j] = Ph[i][j] - kk[i] * hphi[j];
                C2[i][j] = Qm[i][j] - kk[i] * qh[j];
            }
        }
        // Carry the log-determinant half separately.  The assembled leaf
        // ell is `ld2 - g^2/2` and the composition below immediately adds
        // `+g^2/2` back, so forming either one round-trips a quantity that
        // reaches ~1e4 on-model at sigma_GP/yerr = 100 against an ld2 of
        // order 1 -- four digits lost at every observation.  Same fix as
        // elements.leaf_elements; measured 11x better at the default block
        // in that corner.  `ell2` is still needed for the first point,
        // whose element is the leaf itself.
        float ld2 = -0.5f * metal::precise::log(sq / rr);
        float ell2 = ld2 - 0.5f * gg * gg;

        if (first) {
            for (uint i = 0; i < M; ++i) {
                bv[i] = kk[i] * yv;
                ev[i] = gg * zz[i];
                for (uint j = 0; j < M; ++j) {
                    Am[i][j] = A2[i][j];
                    Cm[i][j] = 0.5f * (C2[i][j] + C2[j][i]);
                    Jm[i][j] = zz[i] * zz[j];
                }
            }
            ell = ell2; ell_c = 0.0f;
            first = false;
            continue;
        }

        // ---- combine_leaf: rank-one J2 collapses every inverse ----
        float pv[MDIM], za1[MDIM];
        float alpha = 1.0f;
        for (uint i = 0; i < M; ++i) {
            float s = 0.0f;
            for (uint j = 0; j < M; ++j) s += Cm[i][j] * zz[j];
            pv[i] = s;
        }
        for (uint i = 0; i < M; ++i) alpha += zz[i] * pv[i];
        float inv_a = 1.0f / alpha;
        for (uint i = 0; i < M; ++i) {
            float s = 0.0f;
            for (uint j = 0; j < M; ++j) s += Am[j][i] * zz[j];   // A^T z
            za1[i] = s;
        }
        float beta = gg;
        for (uint i = 0; i < M; ++i) beta -= zz[i] * bv[i];
        float bia = beta * inv_a;

        // GA = A - p za1^T / alpha ;  Mm = C - p p^T / alpha
        float GA[MDIM][MDIM], Mm[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            for (uint j = 0; j < M; ++j) {
                GA[i][j] = Am[i][j] - pv[i] * za1[j] * inv_a;
                Mm[i][j] = Cm[i][j] - pv[i] * pv[j] * inv_a;
            }
        }
        float vv[MDIM];
        for (uint i = 0; i < M; ++i) vv[i] = bv[i] + pv[i] * bia;

        // A <- A2 GA ; b <- A2 v + b2 ; C <- sym(A2 Mm A2^T) + C2
        float An[MDIM][MDIM], Cn[MDIM][MDIM], tmp[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            float s = 0.0f;
            for (uint j = 0; j < M; ++j) s += A2[i][j] * vv[j];
            bv[i] = s + kk[i] * yv;
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f, c = 0.0f;
                for (uint k = 0; k < M; ++k) {
                    a += A2[i][k] * GA[k][j];
                    c += A2[i][k] * Mm[k][j];
                }
                An[i][j] = a;
                tmp[i][j] = c;
            }
        }
        for (uint i = 0; i < M; ++i) {
            for (uint j = 0; j < M; ++j) {
                float c = 0.0f;
                for (uint k = 0; k < M; ++k) c += tmp[i][k] * A2[j][k];
                Cn[i][j] = c;
            }
        }
        for (uint i = 0; i < M; ++i) {
            ev[i] += za1[i] * bia;
            for (uint j = 0; j < M; ++j) {
                Am[i][j] = An[i][j];
                Cm[i][j] = 0.5f * (Cn[i][j] + Cn[j][i]) + C2[i][j];
                Jm[i][j] += za1[i] * za1[j] * inv_a;
            }
        }
        // ell += ell2 + g^2/2 + log(1/alpha)/2 - beta^2/(2 alpha), Kahan
        float add = ld2
                  + 0.5f * metal::precise::log(inv_a) - 0.5f * beta * bia;
        float yk = add - ell_c;
        float tk = ell + yk;
        ell_c = (tk - ell) - yk;
        ell = tk;
    }

    uint omm = (chain * (uint)n_blocks + blk_i) * M * M;
    uint om = (chain * (uint)n_blocks + blk_i) * M;
    for (uint i = 0; i < M; ++i) {
        out_b[om + i] = bv[i];
        out_eta[om + i] = ev[i];
        for (uint j = 0; j < M; ++j) {
            out_A[omm + i * M + j] = Am[i][j];
            out_C[omm + i * M + j] = Cm[i][j];
            out_J[omm + i * M + j] = Jm[i][j];
        }
    }
    out_ell[chain * (uint)n_blocks + blk_i] = ell;
"""

_kernels: dict = {}
_metal_ok: bool | None = None


def _kernel(m: int):
    if m not in _kernels:
        _kernels[m] = mx.fast.metal_kernel(
            name=f"anvilgp_scan_m{m}",
            input_names=["phitab", "qtab", "lag", "hvec", "resid", "r_ref",
                         "jit2", "sfloor", "npts", "n_lags", "n_blocks",
                         "n_chains", "blk_len", "n_jit"],
            output_names=["out_A", "out_b", "out_C", "out_eta", "out_J",
                          "out_ell"],
            source=_SRC.replace("MDIM", str(m)),
        )
    return _kernels[m]



#: Pass 2: fold a chain's block elements into one log-likelihood.
#:
#: One thread per chain, sequential over the ~10-100 blocks.  The MLX tree
#: this replaces was launch-bound, costing a flat ~5.1 ms whatever the block
#: count -- 79% of the whole path at N = 1e4.
#:
#: The fold carries only ``(b, C, ell)``.  It can, because the first
#: observation has no predecessor, so the first block's element has
#: ``A = 0``, ``J = 0``, ``eta = 0``, and `combine` preserves all three:
#: ``A_out = A2 G A1``, ``eta_out = A1^T gt_eta + eta1`` and
#: ``J_out = A1^T jt A1 + J1`` are each zero when the accumulator's are.
#: So three of the six fields need never be computed here.
_SRC_REDUCE = """
    uint chain = thread_position_in_grid.x;
    if (chain >= (uint)n_chains) return;
    const uint M = MDIM;
    uint base = chain * (uint)n_blocks;

    // accumulator: the prefix element, whose A, J and eta are zero
    float bv[MDIM], Cm[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) {
        bv[i] = in_b[base * M + i];
        for (uint j = 0; j < M; ++j)
            Cm[i][j] = in_C[base * M * M + i * M + j];
    }
    float ell = in_ell[base], ell_c = 0.0f;      // Kahan, as pass 1

    for (uint k = 1; k < (uint)n_blocks; ++k) {
        uint ov = (base + k) * M, om = (base + k) * M * M;
        float A2[MDIM][MDIM], C2[MDIM][MDIM], J2[MDIM][MDIM];
        float b2[MDIM], et2[MDIM];
        for (uint i = 0; i < M; ++i) {
            b2[i] = in_b[ov + i];
            et2[i] = in_eta[ov + i];
            for (uint j = 0; j < M; ++j) {
                A2[i][j] = in_A[om + i * M + j];
                C2[i][j] = in_C[om + i * M + j];
                J2[i][j] = in_J[om + i * M + j];
            }
        }

        // L1 = chol(C1), with linalg.diag_floor's eps*trace/m pivot floor
        float tr = 0.0f;
        for (uint i = 0; i < M; ++i) tr += Cm[i][i];
        float fl = 1.1920929e-7f * tr / (float)M + 3.0e-38f;
        float L1[MDIM][MDIM];
        for (uint j = 0; j < M; ++j) {
            float d = Cm[j][j];
            for (uint t = 0; t < j; ++t) d -= L1[j][t] * L1[j][t];
            L1[j][j] = metal::precise::sqrt(max(d, fl));
            float inv = 1.0f / L1[j][j];
            for (uint i = j + 1; i < M; ++i) {
                float a = Cm[i][j];
                for (uint t = 0; t < j; ++t) a -= L1[i][t] * L1[j][t];
                L1[i][j] = a * inv;
            }
            for (uint i = 0; i < j; ++i) L1[i][j] = 0.0f;
        }

        // S = I + sym(L1^T J2 L1)
        float T[MDIM][MDIM], X[MDIM][MDIM], S[MDIM][MDIM];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += J2[i][t] * L1[t][j];
                T[i][j] = a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += L1[t][i] * T[t][j];
                X[i][j] = a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j)
                S[i][j] = (i == j ? 1.0f : 0.0f) + 0.5f * (X[i][j] + X[j][i]);

        // Ls = chol(S); eigenvalues are >= 1 so no pivoting is needed
        float tr2 = 0.0f;
        for (uint i = 0; i < M; ++i) tr2 += S[i][i];
        float fl2 = 1.1920929e-7f * tr2 / (float)M + 3.0e-38f;
        float Ls[MDIM][MDIM];
        for (uint j = 0; j < M; ++j) {
            float d = S[j][j];
            for (uint t = 0; t < j; ++t) d -= Ls[j][t] * Ls[j][t];
            Ls[j][j] = metal::precise::sqrt(max(d, fl2));
            float inv = 1.0f / Ls[j][j];
            for (uint i = j + 1; i < M; ++i) {
                float a = S[i][j];
                for (uint t = 0; t < j; ++t) a -= Ls[i][t] * Ls[j][t];
                Ls[i][j] = a * inv;
            }
            for (uint i = 0; i < j; ++i) Ls[i][j] = 0.0f;
        }

        // Y solves Y Ls^T = L1 -- forward substitution over columns, the
        // same order as linalg.tri_solve_right, so M = Y Y^T is symmetric
        // by construction rather than by symmetrising afterwards.
        float Y[MDIM][MDIM];
        for (uint c = 0; c < M; ++c)
            for (uint r = 0; r < M; ++r) {
                float a = L1[r][c];
                for (uint t = 0; t < c; ++t) a -= Ls[c][t] * Y[r][t];
                Y[r][c] = a / Ls[c][c];
            }

        float Mm[MDIM][MDIM], G[MDIM][MDIM], V[MDIM][MDIM], Jt[MDIM][MDIM];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += Y[i][t] * Y[j][t];
                Mm[i][j] = a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += Mm[i][t] * J2[t][j];
                G[i][j] = (i == j ? 1.0f : 0.0f) - a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += J2[i][t] * Y[t][j];
                V[i][j] = a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += V[i][t] * V[j][t];
                Jt[i][j] = J2[i][j] - a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < i; ++j) {
                float a = 0.5f * (Jt[i][j] + Jt[j][i]);
                Jt[i][j] = a; Jt[j][i] = a;
            }

        float Gb1[MDIM], Met[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f, b = 0.0f;
            for (uint j = 0; j < M; ++j) {
                a += G[i][j] * bv[j];
                b += Mm[i][j] * et2[j];
            }
            Gb1[i] = a; Met[i] = b;
        }

        // ell: log|S| off the factor, then the three quadratic cross terms
        float logdet = 0.0f, qb = 0.0f, ce = 0.0f, qe = 0.0f;
        for (uint i = 0; i < M; ++i) {
            logdet += metal::precise::log(Ls[i][i]);
            float t1 = 0.0f, t2 = 0.0f;
            for (uint j = 0; j < M; ++j) {
                t1 += Jt[i][j] * bv[j];
                t2 += Mm[i][j] * et2[j];
            }
            qb += bv[i] * t1;
            qe += et2[i] * t2;
            ce += et2[i] * Gb1[i];
        }
        float add = in_ell[base + k] - logdet - 0.5f * qb + ce + 0.5f * qe;
        float yk = add - ell_c;
        float tk = ell + yk;
        ell_c = (tk - ell) - yk;
        ell = tk;

        // b <- A2 (G b1 + M eta2) + b2 ;  C <- sym(A2 M A2^T) + C2
        float nb[MDIM], AM[MDIM][MDIM], nC[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = b2[i];
            for (uint j = 0; j < M; ++j) a += A2[i][j] * (Gb1[j] + Met[j]);
            nb[i] = a;
        }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += A2[i][t] * Mm[t][j];
                AM[i][j] = a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += AM[i][t] * A2[j][t];
                nC[i][j] = a;
            }
        for (uint i = 0; i < M; ++i) {
            bv[i] = nb[i];
            for (uint j = 0; j < M; ++j)
                Cm[i][j] = 0.5f * (nC[i][j] + nC[j][i]) + C2[i][j];
        }
    }
    out_ell[chain] = ell;
"""


_reduce_kernels: dict = {}


def _reduce_kernel(m: int):
    if m not in _reduce_kernels:
        _reduce_kernels[m] = mx.fast.metal_kernel(
            name=f"anvilgp_reduce_m{m}",
            input_names=["in_A", "in_b", "in_C", "in_eta", "in_J", "in_ell",
                         "n_blocks", "n_chains"],
            output_names=["out_ell"],
            source=_SRC_REDUCE.replace("MDIM", str(m)),
        )
    return _reduce_kernels[m]


def reduce_blocks(el: Element) -> mx.array:
    """Fold ``(n_chains, n_blocks)`` block elements to ``(n_chains,)``."""
    n_chains, n_blocks, m = el.b.shape[0], el.b.shape[1], el.b.shape[-1]
    if n_blocks == 1:
        return el.ell[:, 0]
    out = _reduce_kernel(m)(
        inputs=[el.A, el.b, el.C, el.eta, el.J, el.ell,
                int(n_blocks), int(n_chains)],
        output_shapes=[(n_chains,)], output_dtypes=[mx.float32],
        grid=(n_chains, 1, 1), threadgroup=(min(64, n_chains), 1, 1))
    return out[0]



#: Stage E: the reverse sweep, one thread per (chain, block).
#:
#: A line-by-line transcription of ``adjoint._backward_span``, which is
#: verified against ``mx.vjp`` to ~1e-16.  The thread replays its block
#: forward from the state the prefix scan says it starts in, keeping every
#: intermediate state, then sweeps back.  Keeping them all caps the block
#: length at what registers hold (``LCAP``); stage F replaces that with
#: checkpoints.
#:
#: Blocks are independent -- each seeds itself from ``adjoint.block_seeds``
#: -- so this has the same launch shape as the forward kernel and needs no
#: ordering between blocks.
#:
#: Never reconstruct a forward state by running the filter *backwards*
#: through the propagator.  That divides by Phi, which underflowed on the
#: way forward at exactly the lags where it would now overflow: every gap
#: in the data.  Replay forward, always.
_SRC_BWD = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    uint start = blk_i * (uint)blk_len;
    if (start >= (uint)npts) return;
    uint stop = min(start + (uint)blk_len, (uint)npts);
    uint L = stop - start;

    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    float jit2v = (n_jit > 0) ? jit2[chain * (uint)n_jit] : 0.0f;
    float cotv = cot[chain];

    // ---- replay forward, keeping every state ----
    float smean[LCAP][MDIM], scov[LCAP][MDIM][MDIM];
    float mean[MDIM], cov[MDIM][MDIM];
    uint ei = chain * (uint)n_blocks + blk_i;
    for (uint i = 0; i < M; ++i) {
        mean[i] = ent_m[ei * M + i];
        for (uint j = 0; j < M; ++j) cov[i][j] = ent_C[ei * M * M + i * M + j];
    }

    for (uint t = 0; t < L; ++t) {
        for (uint i = 0; i < M; ++i) {
            smean[t][i] = mean[i];
            for (uint j = 0; j < M; ++j) scov[t][i][j] = cov[i][j];
        }
        uint gi = start + t;
        uint off = (chain * (uint)n_lags + (uint)lag[gi]) * M * M;
        float mp[MDIM], cp[MDIM][MDIM], ch[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += phitab[off + i * M + j] * mean[j];
            mp[i] = a;
        }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint u = 0; u < M; ++u) {
                    float pu = phitab[off + i * M + u];
                    for (uint w = 0; w < M; ++w)
                        a += pu * cov[u][w] * phitab[off + j * M + w];
                }
                cp[i][j] = a + qtab[off + i * M + j];
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < i; ++j) {
                float a = 0.5f * (cp[i][j] + cp[j][i]);
                cp[i][j] = a; cp[j][i] = a;
            }
        float hch = 0.0f;
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += cp[i][j] * hv[j];
            ch[i] = a; hch += hv[i] * a;
        }
        float sv = max(hch, 0.0f) + r_ref[gi] + jit2v;
        float vv = resid[gi * (uint)n_chains + chain];
        for (uint i = 0; i < M; ++i) vv -= hv[i] * mp[i];
        for (uint i = 0; i < M; ++i) {
            float kg = ch[i] / sv;
            mean[i] = mp[i] + kg * vv;
        }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j)
                cov[i][j] = cp[i][j] - (ch[i] / sv) * ch[j];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < i; ++j) {
                float a = 0.5f * (cov[i][j] + cov[j][i]);
                cov[i][j] = a; cov[j][i] = a;
            }
    }

    // ---- reverse sweep ----
    float m_bar[MDIM], p_bar[MDIM][MDIM], h_bar[MDIM];
    for (uint i = 0; i < M; ++i) {
        m_bar[i] = seed_m[ei * M + i];
        h_bar[i] = 0.0f;
        for (uint j = 0; j < M; ++j)
            p_bar[i][j] = seed_C[ei * M * M + i * M + j];
    }

    for (int tt = (int)L - 1; tt >= 0; --tt) {
        uint t = (uint)tt, gi = start + t;
        uint off = (chain * (uint)n_lags + (uint)lag[gi]) * M * M;
        float Ph[MDIM][MDIM];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) Ph[i][j] = phitab[off + i * M + j];

        // replay this step's forward quantities from the stored state
        float mp[MDIM], cp[MDIM][MDIM], ch[MDIM], kg[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += Ph[i][j] * smean[t][j];
            mp[i] = a;
        }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint u = 0; u < M; ++u)
                    for (uint w = 0; w < M; ++w)
                        a += Ph[i][u] * scov[t][u][w] * Ph[j][w];
                cp[i][j] = a + qtab[off + i * M + j];
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < i; ++j) {
                float a = 0.5f * (cp[i][j] + cp[j][i]);
                cp[i][j] = a; cp[j][i] = a;
            }
        float hch = 0.0f;
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += cp[i][j] * hv[j];
            ch[i] = a; hch += hv[i] * a;
        }
        bool clamped = hch < 0.0f;
        float sv = max(hch, 0.0f) + r_ref[gi] + jit2v;
        float vv = resid[gi * (uint)n_chains + chain];
        for (uint i = 0; i < M; ++i) vv -= hv[i] * mp[i];
        for (uint i = 0; i < M; ++i) kg[i] = ch[i] / sv;

        // seeds -- the only place ell enters
        float v_bar = -cotv * vv / sv;
        float s_bar = cotv * (vv * vv - sv) / (2.0f * sv * sv);

        // P' = Cp - Kg ch^T ;  m' = mp + Kg v
        float kg_bar[MDIM], ch_bar[MDIM], mp_bar[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f, b = 0.0f;
            for (uint j = 0; j < M; ++j) {
                a += p_bar[i][j] * ch[j];
                b += p_bar[j][i] * kg[j];
            }
            kg_bar[i] = -a + m_bar[i] * vv;
            ch_bar[i] = -b;
            mp_bar[i] = m_bar[i];
        }
        for (uint i = 0; i < M; ++i) v_bar += kg[i] * m_bar[i];

        // v = y - h.mp
        out_dy[gi * (uint)n_chains + chain] = v_bar;
        for (uint i = 0; i < M; ++i) {
            h_bar[i] -= v_bar * mp[i];
            mp_bar[i] -= v_bar * hv[i];
        }
        // Kg = ch / S
        for (uint i = 0; i < M; ++i) {
            ch_bar[i] += kg_bar[i] / sv;
            s_bar -= kg_bar[i] * kg[i] / sv;
        }
        // S = max(h.ch, 0) + R : r_var takes s_bar whatever the clamp did
        out_drv[gi * (uint)n_chains + chain] = s_bar;
        float hch_bar = clamped ? 0.0f : s_bar;
        for (uint i = 0; i < M; ++i) {
            h_bar[i] += hch_bar * ch[i];
            ch_bar[i] += hch_bar * hv[i];
        }
        // ch = Cp h
        float cp_bar[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) {
                cp_bar[i][j] = p_bar[i][j] + ch_bar[i] * hv[j];
                a += cp[j][i] * ch_bar[j];
            }
            h_bar[i] += a;
        }
        // Cp = sym(Phi P Phi^T + Q): the forward symmetrises, so Q's
        // adjoint -- and P's -- is the SYMMETRISED Cp adjoint
        float cs[MDIM][MDIM];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j)
                cs[i][j] = 0.5f * (cp_bar[i][j] + cp_bar[j][i]);

        uint om = (chain * (uint)npts + gi) * M * M;
        float pc[MDIM][MDIM];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint u = 0; u < M; ++u) a += Ph[i][u] * scov[t][u][j];
                pc[i][j] = a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint u = 0; u < M; ++u) a += 2.0f * cs[i][u] * pc[u][j];
                out_dphi[om + i * M + j] = a + mp_bar[i] * smean[t][j];
                out_dq[om + i * M + j] = cs[i][j];
            }
        float nb[MDIM][MDIM];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint u = 0; u < M; ++u)
                    for (uint w = 0; w < M; ++w)
                        a += Ph[u][i] * cs[u][w] * Ph[w][j];
                nb[i][j] = a;
            }
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) {
                p_bar[i][j] = 0.5f * (nb[i][j] + nb[j][i]);
                a += Ph[j][i] * mp_bar[j];
            }
            m_bar[i] = a;
        }
    }

    for (uint i = 0; i < M; ++i) out_dh[ei * M + i] = h_bar[i];
"""


_bwd_kernels: dict = {}


def _bwd_kernel(m: int, lcap: int):
    key = (m, lcap)
    if key not in _bwd_kernels:
        _bwd_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_bwd_m{m}_L{lcap}",
            input_names=["phitab", "qtab", "lag", "hvec", "resid", "r_ref",
                         "jit2", "cot", "ent_m", "ent_C", "seed_m", "seed_C",
                         "npts", "n_lags", "n_blocks", "n_chains", "blk_len",
                         "n_jit"],
            output_names=["out_dphi", "out_dq", "out_dh", "out_dy",
                          "out_drv"],
            source=_SRC_BWD.replace("MDIM", str(m)).replace("LCAP",
                                                            str(lcap)),
        )
    return _bwd_kernels[key]



MAX_BWD_BLOCK = 32
"""Longest block the stage-E backward kernel will take.

It keeps every forward state of its block in thread-local storage --
``(m + m^2)`` floats per observation -- so the block length is bounded by
what a thread can hold.  Stage F replaces this with checkpoints and lifts
the bound.
"""


def backward_blocks(phi_tab, q_tab, lag, hvec, resid, r_ref, jit2, cot,
                    ent_m, ent_C, seed_m, seed_C, blk_len):
    """Reverse sweep for every (chain, block), independently.

    ``ent_m``/``ent_C`` are the filter state each block starts from and
    ``seed_m``/``seed_C`` the ``(m_bar, P_bar)`` it inherits, both
    ``(n_chains, n_blocks, ...)`` -- from
    :func:`anvilgp.elements.prefix_scan` and
    :func:`anvilgp.adjoint.block_seeds`.  Returns the five adjoints of
    :func:`anvilgp.adjoint.filter_backward`, with ``dphi``/``dq`` per
    observation (stage F accumulates them per lag instead, which is what
    makes them affordable at production N).
    """
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts = resid.shape[0]
    if m > MAX_STATE:
        raise ValueError(f"metal backward supports m <= {MAX_STATE}, got {m}")
    if blk_len > MAX_BWD_BLOCK:
        raise ValueError(
            f"blk_len {blk_len} exceeds MAX_BWD_BLOCK {MAX_BWD_BLOCK}: the "
            "stage-E kernel holds every forward state of its block")
    n_blocks = (npts + blk_len - 1) // blk_len
    n_jit = jit2.shape[1] if jit2.ndim == 2 else 0
    shapes = [(n_chains, npts, m, m), (n_chains, npts, m, m),
              (n_chains, n_blocks, m), (npts, n_chains), (npts, n_chains)]
    outs = _bwd_kernel(m, blk_len)(
        inputs=[phi_tab, q_tab, lag, hvec, resid, r_ref, jit2, cot,
                ent_m, ent_C, seed_m, seed_C,
                int(npts), int(n_lags), int(n_blocks), int(n_chains),
                int(blk_len), int(n_jit)],
        output_shapes=shapes, output_dtypes=[mx.float32] * 5,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return outs


def metal_available() -> bool:
    """Probe: can the fused kernel actually run here?"""
    global _metal_ok
    if _metal_ok is None:
        try:
            phi = mx.zeros((1, 1, 2, 2), dtype=mx.float32)
            q = mx.broadcast_to(mx.eye(2, dtype=mx.float32), (1, 1, 2, 2))
            out = _kernel(2)(
                inputs=[phi, q, mx.zeros((1,), dtype=mx.int32),
                        mx.ones((1, 2), dtype=mx.float32),
                        mx.zeros((1, 1), dtype=mx.float32),
                        mx.ones((1,), dtype=mx.float32),
                        mx.zeros((1, 1), dtype=mx.float32),
                        1e-6, 1, 1, 1, 1, 1, 0],
                output_shapes=[(1, 1, 2, 2), (1, 1, 2), (1, 1, 2, 2),
                               (1, 1, 2), (1, 1, 2, 2), (1, 1)],
                output_dtypes=[mx.float32] * 6,
                grid=(1, 1, 1), threadgroup=(1, 1, 1))
            mx.eval(out)
            _metal_ok = True
        except Exception:
            _metal_ok = False
    return _metal_ok


def scan_blocks(phi_tab, q_tab, lag, hvec, resid, r_ref, jit2, s_floor_rel,
                blk_len, n_blocks=None) -> Element:
    """Condense every block of ``blk_len`` observations into one element.

    ``phi_tab``/``q_tab``: (n_chains, n_lags, m, m) propagator and process
    noise indexed by lag, with row 0 reserved for the no-predecessor case.
    ``lag``: (npts,) int32.  ``resid``: **(npts, n_chains)** residuals --
    time-major, so adjacent threads (adjacent chains) read adjacent floats.
    ``r_ref``: (npts,) float32 reference variances.  ``jit2``: (n_chains, 1)
    jitter variance, or a (n_chains, 0) array for none.

    Returns an :class:`~anvilgp.elements.Element` with batch shape
    (n_chains, n_blocks), ready for :func:`anvilgp.elements.tree_reduce`.
    """
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts = resid.shape[0]          # resid is (npts, n_chains)
    if m > MAX_STATE:
        raise ValueError(f"metal scan supports m <= {MAX_STATE}, got {m}")
    if n_blocks is None:
        n_blocks = (npts + blk_len - 1) // blk_len
    n_jit = jit2.shape[1] if jit2.ndim == 2 else 0

    shapes = [(n_chains, n_blocks, m, m), (n_chains, n_blocks, m),
              (n_chains, n_blocks, m, m), (n_chains, n_blocks, m),
              (n_chains, n_blocks, m, m), (n_chains, n_blocks)]
    scalars = [float(s_floor_rel), int(npts), int(n_lags), int(n_blocks),
               int(n_chains), int(blk_len), int(n_jit)]

    outs = _kernel(m)(
        inputs=[phi_tab, q_tab, lag, hvec, resid, r_ref, jit2, *scalars],
        output_shapes=shapes, output_dtypes=[mx.float32] * 6,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return Element(A=outs[0], b=outs[1], C=outs[2], eta=outs[3], J=outs[4],
                   ell=outs[5])


def scan_loglike(phi_tab, q_tab, lag, hvec, resid, r_ref, jit2, s_floor_rel,
                 blk_len) -> mx.array:
    """``(n_chains,)`` log-likelihood: the kernel's blocks, reduced.

    **This, not :func:`scan_blocks`, is where the differentiation boundary
    sits**, and that is an M4 requirement rather than tidiness.  Wrapping
    the kernel alone would leave a ``custom_function`` whose outputs are six
    element fields per block, so its VJP would have to accept cotangents on
    all six -- a harder problem, and not the one the verified recursion in
    :mod:`anvilgp.adjoint` solves.  Wrapping the reduction too collapses the
    output to one scalar per chain, and a scalar cotangent is exactly what
    ``filter_backward`` takes.

    Both passes are kernels: :func:`scan_blocks` condenses the blocks and
    :func:`reduce_blocks` folds them.  The MLX tree this replaced was
    launch-bound, costing a flat ~5.1 ms whatever the block count, which
    was 79% of the whole path at N = 1e4.
    """
    @mx.custom_function
    def core(phi_t, q_t, lg, hv, rs, rr, j2):
        el = scan_blocks(phi_t, q_t, lg, hv, rs, rr, j2, s_floor_rel,
                         blk_len)
        return reduce_blocks(el)

    @core.vjp
    def _core_vjp(primals, cotangents, outputs):
        # M4 replaces this with the checkpointed backward kernel.  The
        # cotangent arriving here is on a scalar per chain, which is what
        # `adjoint.filter_backward` is derived for; the seed for each
        # block comes from the closed form in docs/m4-plan.md, so the
        # blocks are independent and need no chain between them.
        #
        # Never reconstruct forward state by running the filter backwards
        # through the propagator: that divides by Phi, which underflowed on
        # the way forward at exactly the lags where it would now overflow,
        # which is every gap in the data.  Replay forward, always.
        raise NotImplementedError(
            "the fused Metal scan is forward-only until M4. Use "
            "GPPolicy(metal=False) for a differentiable likelihood (the MLX "
            "path), or a gradient-free sampler such as anvil's stretch move.")

    return core(phi_tab, q_tab, lag, hvec, resid, r_ref, jit2)
