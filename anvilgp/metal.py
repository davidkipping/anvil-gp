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
           "backward_ckpt", "block_seeds_kernel",
           "metal_available", "MAX_STATE", "MAX_ADJOINT_BUFFER_BYTES"]

#: Largest adjoint buffer the kernel gradient will allocate.  The backward
#: accumulates ``dphi``/``dq`` into one slot per ``(block, lag)`` pair that
#: *occurs* -- two ``(n_chains, n_slots, m, m)`` float32 arrays -- which is
#: bounded by N however irregular the cadence.
#:
#: It was one slot per pair that could occur, ``n_blocks`` times larger,
#: and this fence was 2 GiB because that shape reached 40 GiB at 1024
#: chains x N = 1e5 and had to be refused.  With the slot layout the same
#: problem needs 3.05 GiB of buffer inside a 17.5 GiB gradient whose
#: forward is already 13.2 GiB, so the old value refused work that runs
#: comfortably.  8 GiB keeps this a sanity check: it still stops 4096
#: chains at m = 4, which would ask for 49 GiB.
MAX_ADJOINT_BUFFER_BYTES = 8 << 30

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

        // L1 = chol(C1), with linalg.diag_floor's eps*max(trace,0)/m pivot floor
        float tr = 0.0f;
        for (uint i = 0; i < M; ++i) tr += Cm[i][i];
        float fl = 1.1920929e-7f * max(tr, 0.0f) / (float)M + 3.0e-38f;
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
        float fl2 = 1.1920929e-7f * max(tr2, 0.0f) / (float)M + 3.0e-38f;
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



#: One Kalman step at global index ``gi``, updating ``mean``/``cov`` in
#: place.  Spliced into the stage-F kernel twice -- the checkpoint-writing
#: replay and the sub-block replay -- so both are byte-for-byte the same
#: arithmetic the sweep then differentiates.
_FWD_STEP = """
        {
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
            float kg[MDIM];
            for (uint i = 0; i < M; ++i) kg[i] = ch[i] / sv;
            for (uint i = 0; i < M; ++i) mean[i] = mp[i] + kg[i] * vv;
            // Joseph form, cov = (I - K h) cp (I - K h)^T + rk K K^T, with
            // rk = sv - h cp h^T formed without cancellation.  Equal to the
            // spec's cp - K ch^T in exact arithmetic for every sv (clamped
            // or not), but PSD by construction where that form is not: on
            // the first observation cp = I and h.h can exceed r_ref by 1e8,
            // so `1 - h.h/(h.h + r)` rounds to 0 or a negative ulp instead
            // of r/(h.h + r), and an undamped propagator then grows that
            // ulp until the replay overflows -- a correct value with an
            // exactly-zero gradient, once the guard zeroes the NaN.  The
            // Joseph form squares the cancelled factor and adds rk K K^T,
            // which carries the right 1e-8 on its own.  Stage R finding.
            float am[MDIM][MDIM], tc[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j)
                    am[i][j] = (i == j ? 1.0f : 0.0f) - kg[i] * hv[j];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f;
                    for (uint u = 0; u < M; ++u) a += am[i][u] * cp[u][j];
                    tc[i][j] = a;
                }
            float rk = r_ref[gi] + jit2v + max(-hch, 0.0f);
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = rk * kg[i] * kg[j];
                    for (uint u = 0; u < M; ++u) a += tc[i][u] * am[j][u];
                    cov[i][j] = a;
                }
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < i; ++j) {
                    float a = 0.5f * (cov[i][j] + cov[j][i]);
                    cov[i][j] = a; cov[j][i] = a;
                }
        }
"""

#: Stage F: the checkpointed reverse sweep, one thread per (chain, block).
#:
#: The stage-E prototype (removed after stage F superseded it; see git
#: history) held every forward state of its block and so capped the block
#: at 32 observations.  Here the thread replays its block once writing a
#: checkpoint every ``SLEN`` steps to device memory, then walks the
#: sub-blocks in reverse: reload the checkpoint, replay ``SLEN`` states into
#: registers, sweep them back.  Two forward replays plus one sweep, and the
#: block can be as long as the forward's.
#:
#: ``dphi``/``dq`` are accumulated **per lag**, not per observation: the
#: per-observation array is 1.6 GB at N = 1e5 x 1024 chains.  A run of one
#: lag is accumulated in registers and flushed to the thread's own
#: ``(chain, block, lag)`` slot when the lag changes -- real photometry has
#: one cadence plus one lag per gap, so ~14 flushes per 1e5 points.  Each
#: thread zeroes its own slots first and then does read-modify-write, and
#: no other thread touches them, so there is nothing to synchronise.
_SRC_BWD_CKPT = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    const uint SL = SLEN;
    uint start = blk_i * (uint)blk_len;
    uint stop = min(start + (uint)blk_len, (uint)npts);
    uint L = stop - start;
    uint ei = chain * (uint)n_blocks + blk_i;

    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    float jit2v = (n_jit > 0) ? jit2[chain * (uint)n_jit] : 0.0f;
    float cotv = cot[chain];

    // This thread's accumulator slots: the contiguous range block blk_i
    // owns, at this chain's offset.  One slot per (block, lag) pair that
    // *occurs*, where this was one per pair that could occur -- the same
    // on a regular cadence and 13x fewer at N = 1e5 when every timestamp
    // spacing is distinct, which is what makes the buffer affordable
    // there.  No other thread touches this range (the blocks partition
    // the slots), so there is still nothing to synchronise.
    uint s_lo = (uint)slot_start[blk_i];
    uint s_hi = (uint)slot_start[blk_i + 1];
    for (uint sl = s_lo; sl < s_hi; ++sl)
        for (uint i = 0; i < M * M; ++i) {
            out_dphi[(chain * (uint)n_slots + sl) * M * M + i] = 0.0f;
            out_dq[(chain * (uint)n_slots + sl) * M * M + i] = 0.0f;
        }

    // ---- pass 1: replay forward, checkpoint every SL ----
    float mean[MDIM], cov[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) {
        mean[i] = ent_m[ei * M + i];
        for (uint j = 0; j < M; ++j) cov[i][j] = ent_C[ei * M * M + i * M + j];
    }
    const uint CK = M + M * M;
    uint ckb = ei * (uint)n_ckpt;
    for (uint t = 0; t < L; ++t) {
        if (t % SL == 0) {
            uint cb = (ckb + t / SL) * CK;
            for (uint i = 0; i < M; ++i) {
                ckpt[cb + i] = mean[i];
                for (uint j = 0; j < M; ++j) ckpt[cb + M + i * M + j] = cov[i][j];
            }
        }
        uint gi = start + t;
        FWD_STEP
    }

    // ---- pass 2: sub-blocks in reverse ----
    float m_bar[MDIM], p_bar[MDIM][MDIM], h_bar[MDIM];
    for (uint i = 0; i < M; ++i) {
        m_bar[i] = seed_m[ei * M + i];
        h_bar[i] = 0.0f;
        for (uint j = 0; j < M; ++j) p_bar[i][j] = seed_C[ei * M * M + i * M + j];
    }
    float drv_acc = 0.0f;
    uint cur_slot = 0xFFFFFFFFu;
    float acc_p[MDIM][MDIM], acc_q[MDIM][MDIM];
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) { acc_p[i][j] = 0.0f; acc_q[i][j] = 0.0f; }

    uint n_sub = (L + SL - 1) / SL;
    for (int sbi = (int)n_sub - 1; sbi >= 0; --sbi) {
        uint sb = (uint)sbi;
        uint t0 = sb * SL, t1 = min(t0 + SL, L);

        uint cb = (ckb + sb) * CK;
        for (uint i = 0; i < M; ++i) {
            mean[i] = ckpt[cb + i];
            for (uint j = 0; j < M; ++j) cov[i][j] = ckpt[cb + M + i * M + j];
        }
        float smean[SLEN][MDIM], scov[SLEN][MDIM][MDIM];
        for (uint t = t0; t < t1; ++t) {
            uint li = t - t0;
            for (uint i = 0; i < M; ++i) {
                smean[li][i] = mean[i];
                for (uint j = 0; j < M; ++j) scov[li][i][j] = cov[i][j];
            }
            uint gi = start + t;
            FWD_STEP
        }

        for (int tt = (int)t1 - 1; tt >= (int)t0; --tt) {
            uint t = (uint)tt, li = t - t0, gi = start + t;
            uint lg = (uint)lag[gi];
            uint off = (chain * (uint)n_lags + lg) * M * M;
            float Ph[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) Ph[i][j] = phitab[off + i * M + j];

            float mp[MDIM], cp[MDIM][MDIM], ch[MDIM], kg[MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += Ph[i][j] * smean[li][j];
                mp[i] = a;
            }
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f;
                    for (uint u = 0; u < M; ++u)
                        for (uint w = 0; w < M; ++w)
                            a += Ph[i][u] * scov[li][u][w] * Ph[j][w];
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

            float v_bar = -cotv * vv / sv;
            float s_bar = cotv * (vv * vv - sv) / (2.0f * sv * sv);

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

            out_dy[gi * (uint)n_chains + chain] = v_bar;
            for (uint i = 0; i < M; ++i) {
                h_bar[i] -= v_bar * mp[i];
                mp_bar[i] -= v_bar * hv[i];
            }
            for (uint i = 0; i < M; ++i) {
                ch_bar[i] += kg_bar[i] / sv;
                s_bar -= kg_bar[i] * kg[i] / sv;
            }
            drv_acc += s_bar;
            float hch_bar = clamped ? 0.0f : s_bar;
            for (uint i = 0; i < M; ++i) {
                h_bar[i] += hch_bar * ch[i];
                ch_bar[i] += hch_bar * hv[i];
            }
            float cp_bar[MDIM][MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) {
                    cp_bar[i][j] = p_bar[i][j] + ch_bar[i] * hv[j];
                    a += cp[j][i] * ch_bar[j];
                }
                h_bar[i] += a;
            }
            float cs[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j)
                    cs[i][j] = 0.5f * (cp_bar[i][j] + cp_bar[j][i]);

            // Flush the accumulator when the slot changes.  Within a
            // block the slot changes exactly when the lag does -- a slot
            // *is* a (block, lag) pair -- so this is the same flush
            // schedule as before, with the test on the thing written.
            uint slt = (uint)slot_id[gi];
            if (slt != cur_slot) {
                if (cur_slot != 0xFFFFFFFFu) {
                    uint ob = (chain * (uint)n_slots + cur_slot) * M * M;
                    for (uint i = 0; i < M; ++i)
                        for (uint j = 0; j < M; ++j) {
                            out_dphi[ob + i * M + j] += acc_p[i][j];
                            out_dq[ob + i * M + j] += acc_q[i][j];
                        }
                }
                cur_slot = slt;
                for (uint i = 0; i < M; ++i)
                    for (uint j = 0; j < M; ++j) { acc_p[i][j] = 0.0f; acc_q[i][j] = 0.0f; }
            }
            float pc[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f;
                    for (uint u = 0; u < M; ++u) a += Ph[i][u] * scov[li][u][j];
                    pc[i][j] = a;
                }
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f;
                    for (uint u = 0; u < M; ++u) a += 2.0f * cs[i][u] * pc[u][j];
                    acc_p[i][j] += a + mp_bar[i] * smean[li][j];
                    acc_q[i][j] += cs[i][j];
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
    }
    if (cur_slot != 0xFFFFFFFFu) {
        uint ob = (chain * (uint)n_slots + cur_slot) * M * M;
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                out_dphi[ob + i * M + j] += acc_p[i][j];
                out_dq[ob + i * M + j] += acc_q[i][j];
            }
    }
    for (uint i = 0; i < M; ++i) out_dh[ei * M + i] = h_bar[i];
    out_drv[ei] = drv_acc;
"""


_bwd_ckpt_kernels: dict = {}

#: sub-block length by state dimension: what a thread can hold in
#: ``smean``/``scov`` alongside the sweep's own working set.
_SUB_LEN = {1: 32, 2: 32, 3: 16, 4: 8}


def _bwd_ckpt_kernel(m: int, slen: int):
    key = (m, slen)
    if key not in _bwd_ckpt_kernels:
        body = _SRC_BWD_CKPT.replace("FWD_STEP", _FWD_STEP)
        _bwd_ckpt_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_bwd_ckpt_m{m}_s{slen}",
            input_names=["phitab", "qtab", "lag", "hvec", "resid", "r_ref",
                         "jit2", "cot", "ent_m", "ent_C", "seed_m", "seed_C",
                         "slot_id", "slot_start",
                         "npts", "n_lags", "n_blocks", "n_chains", "blk_len",
                         "n_jit", "n_ckpt", "n_slots"],
            output_names=["out_dphi", "out_dq", "out_dh", "out_dy",
                          "out_drv", "ckpt"],
            source=body.replace("MDIM", str(m)).replace("SLEN", str(slen)),
        )
    return _bwd_ckpt_kernels[key]


def backward_ckpt(phi_tab, q_tab, lag, hvec, resid, r_ref, jit2, cot,
                  ent_m, ent_C, seed_m, seed_C, slot_id, slot_start, n_slots,
                  blk_len, sub_len=None):
    """Checkpointed reverse sweep for every (chain, block), independently.

    Inputs are the forward's table form plus the per-block entry state
    ``(ent_m, ent_C)`` and seed ``(seed_m, seed_C)`` from
    :func:`block_seeds_kernel`.  No cap on ``blk_len``.  ``dphi``/``dq``
    are returned **per slot**, ``(n_chains, n_slots, m, m)``, where a slot
    is one ``(block, lag)`` pair that occurs -- ``slot_id``/``slot_start``
    from :func:`anvilgp.likelihood._slot_map`.  The caller scatters them
    into a per-lag table with ``slot_lag``; the blocks are already summed,
    since distinct blocks own disjoint slots.  ``dh`` and ``drv`` come per
    ``(chain, block)``; ``dy`` per observation, ``(npts, n_chains)``, since
    it feeds the mean model.

    The forward replay uses the Joseph-form covariance update, which is
    PSD by construction; the plain ``cp - K ch^T`` is not, and a rounding
    ulp on the first observation grew until the replay overflowed at high
    ``sigma/yerr`` (stage R).  The two are equal in exact arithmetic, and
    the sweep differentiates the *states*, so :mod:`anvilgp.adjoint`'s
    specification is unchanged.
    """
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts = resid.shape[0]
    if m > MAX_STATE:
        raise ValueError(f"metal backward supports m <= {MAX_STATE}, got {m}")
    for nm, arr in (("lag", lag), ("slot_id", slot_id),
                    ("slot_start", slot_start)):
        # Metal reads these as 4-byte ints; a wider dtype indexes garbage
        # with no error, and the value would still be right.
        if arr.dtype != mx.int32:
            raise TypeError(f"{nm} must be int32 for the kernel, got "
                            f"{arr.dtype}")
    slen = _SUB_LEN.get(m, 8) if sub_len is None else int(sub_len)
    n_blocks = (npts + blk_len - 1) // blk_len
    n_ckpt = (blk_len + slen - 1) // slen
    n_jit = jit2.shape[1] if jit2.ndim == 2 else 0
    n_slots = int(n_slots)
    shapes = [(n_chains, n_slots, m, m),
              (n_chains, n_slots, m, m),
              (n_chains, n_blocks, m), (npts, n_chains), (n_chains, n_blocks),
              (n_chains, n_blocks, n_ckpt, m + m * m)]
    outs = _bwd_ckpt_kernel(m, slen)(
        inputs=[phi_tab, q_tab, lag, hvec, resid, r_ref, jit2, cot,
                ent_m, ent_C, seed_m, seed_C, slot_id, slot_start,
                int(npts), int(n_lags), int(n_blocks), int(n_chains),
                int(blk_len), int(n_jit), int(n_ckpt), int(n_slots)],
        output_shapes=shapes, output_dtypes=[mx.float32] * 6,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return outs[:5]



#: The `combine` core shared by the suffix step and the seed: from the
#: earlier element's ``(b1, C1)`` and the later one's ``(J2, et2)``, form
#: ``Mm = (I + C1 J2)^-1 C1``, ``G = (I + C1 J2)^-1``,
#: ``Jt = (I + J2 C1)^-1 J2`` and ``r = G^T (et2 - J2 b1)`` -- the last two
#: are what both consumers need.  Same Cholesky route as `reduce_blocks`.
_COMBINE_CORE = """
            float trc = 0.0f;
            for (uint i = 0; i < M; ++i) trc += C1[i][i];
            float flc = 1.1920929e-7f * max(trc, 0.0f) / (float)M + 3.0e-38f;
            float L1[MDIM][MDIM];
            for (uint j = 0; j < M; ++j) {
                float d = C1[j][j];
                for (uint t = 0; t < j; ++t) d -= L1[j][t] * L1[j][t];
                L1[j][j] = metal::precise::sqrt(max(d, flc));
                float inv = 1.0f / L1[j][j];
                for (uint i = j + 1; i < M; ++i) {
                    float a = C1[i][j];
                    for (uint t = 0; t < j; ++t) a -= L1[i][t] * L1[j][t];
                    L1[i][j] = a * inv;
                }
                for (uint i = 0; i < j; ++i) L1[i][j] = 0.0f;
            }
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
            float trs = 0.0f;
            for (uint i = 0; i < M; ++i) trs += S[i][i];
            float fls = 1.1920929e-7f * max(trs, 0.0f) / (float)M + 3.0e-38f;
            float Ls[MDIM][MDIM];
            for (uint j = 0; j < M; ++j) {
                float d = S[j][j];
                for (uint t = 0; t < j; ++t) d -= Ls[j][t] * Ls[j][t];
                Ls[j][j] = metal::precise::sqrt(max(d, fls));
                float inv = 1.0f / Ls[j][j];
                for (uint i = j + 1; i < M; ++i) {
                    float a = S[i][j];
                    for (uint t = 0; t < j; ++t) a -= Ls[i][t] * Ls[j][t];
                    Ls[i][j] = a * inv;
                }
                for (uint i = 0; i < j; ++i) Ls[i][j] = 0.0f;
            }
            float Y[MDIM][MDIM];
            for (uint c = 0; c < M; ++c)
                for (uint rr = 0; rr < M; ++rr) {
                    float a = L1[rr][c];
                    for (uint t = 0; t < c; ++t) a -= Ls[c][t] * Y[rr][t];
                    Y[rr][c] = a / Ls[c][c];
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
            float rr_[MDIM], jb[MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += J2[i][j] * b1[j];
                jb[i] = et2[i] - a;
            }
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += G[j][i] * jb[j];
                rr_[i] = a;
            }
"""

#: Seeds for the backward kernel, one thread per chain.
#:
#: Three sequential folds over a chain's ~10-100 block elements, all in one
#: launch: the prefix, giving the filter state each block *starts* from
#: (``ent``); the suffix, giving the future after each block as ``(J,
#: eta)``; and the closed form of `adjoint.block_seeds` from the two.
#: This replaces `prefix_scan`/`suffix_scan`/`block_seeds` in MLX, which
#: build ~2 x n_blocks sequential compositions in the graph: 42 ms at
#: m = 2 and 1.5 s at m = 4 for N = 1e5, against a backward kernel of 33
#: and 112 ms.  Here the whole thing is ~0.1 ms.
_SRC_SEEDS = """
    uint chain = thread_position_in_grid.x;
    if (chain >= (uint)n_chains) return;
    const uint M = MDIM;
    uint base = chain * (uint)n_blocks;

    // ---- prefix: fold (b, C) forward, emitting the ENTRY state of each
    // block, i.e. the state after the previous one; block 0 enters at 0.
    float bv[MDIM], Cm[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) {
        ent_m[base * M + i] = 0.0f;
        for (uint j = 0; j < M; ++j) ent_C[base * M * M + i * M + j] = 0.0f;
        bv[i] = in_b[base * M + i];
        for (uint j = 0; j < M; ++j) Cm[i][j] = in_C[base * M * M + i * M + j];
    }
    for (uint k = 1; k < (uint)n_blocks; ++k) {
        // entry of block k = exit of block k-1 = current accumulator
        for (uint i = 0; i < M; ++i) {
            ent_m[(base + k) * M + i] = bv[i];
            for (uint j = 0; j < M; ++j)
                ent_C[(base + k) * M * M + i * M + j] = Cm[i][j];
        }
        uint ov = (base + k) * M, om = (base + k) * M * M;
        float C1[MDIM][MDIM], J2[MDIM][MDIM], A2[MDIM][MDIM], C2[MDIM][MDIM];
        float b1[MDIM], et2[MDIM], b2[MDIM];
        for (uint i = 0; i < M; ++i) {
            b1[i] = bv[i]; b2[i] = in_b[ov + i]; et2[i] = in_eta[ov + i];
            for (uint j = 0; j < M; ++j) {
                C1[i][j] = Cm[i][j];
                J2[i][j] = in_J[om + i * M + j];
                A2[i][j] = in_A[om + i * M + j];
                C2[i][j] = in_C[om + i * M + j];
            }
        }
        COMBINE_CORE
        // b <- A2 (G b1 + Mm eta2) + b2 ;  C <- sym(A2 Mm A2^T) + C2
        float Gb1[MDIM], Met[MDIM], nb[MDIM], AM[MDIM][MDIM], nC[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f, b = 0.0f;
            for (uint j = 0; j < M; ++j) { a += G[i][j] * b1[j]; b += Mm[i][j] * et2[j]; }
            Gb1[i] = a; Met[i] = b;
        }
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
            for (uint j = 0; j < M; ++j) Cm[i][j] = 0.5f * (nC[i][j] + nC[j][i]) + C2[i][j];
        }
    }
    // exit state of the last block, needed for its (zero) seed below
    float xb[MDIM], xC[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) { xb[i] = bv[i]; for (uint j = 0; j < M; ++j) xC[i][j] = Cm[i][j]; }

    // ---- suffix: fold (J, eta) backward.  suf[k] = blocks k..B-1;
    // suf[B] = identity.  Stored per block in scratch.
    float Jm[MDIM][MDIM], ev[MDIM];
    for (uint i = 0; i < M; ++i) { ev[i] = 0.0f; for (uint j = 0; j < M; ++j) Jm[i][j] = 0.0f; }
    for (int kk = (int)n_blocks - 1; kk >= 0; --kk) {
        uint k = (uint)kk;
        // store suf[k+1] (the accumulator BEFORE absorbing block k)
        uint sv_ = (base + k) * M, sm_ = (base + k) * M * M;
        for (uint i = 0; i < M; ++i) {
            suf_eta[sv_ + i] = ev[i];
            for (uint j = 0; j < M; ++j) suf_J[sm_ + i * M + j] = Jm[i][j];
        }
        uint ov = (base + k) * M, om = (base + k) * M * M;
        float C1[MDIM][MDIM], J2[MDIM][MDIM], A1[MDIM][MDIM], J1[MDIM][MDIM];
        float b1[MDIM], et2[MDIM], e1[MDIM];
        for (uint i = 0; i < M; ++i) {
            b1[i] = in_b[ov + i]; e1[i] = in_eta[ov + i]; et2[i] = ev[i];
            for (uint j = 0; j < M; ++j) {
                C1[i][j] = in_C[om + i * M + j];
                A1[i][j] = in_A[om + i * M + j];
                J1[i][j] = in_J[om + i * M + j];
                J2[i][j] = Jm[i][j];
            }
        }
        COMBINE_CORE
        // eta <- A1^T r + eta1 ;  J <- sym(A1^T Jt A1) + J1
        float ne[MDIM], JA[MDIM][MDIM], nJ[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = e1[i];
            for (uint j = 0; j < M; ++j) a += A1[j][i] * rr_[j];
            ne[i] = a;
        }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += Jt[i][t] * A1[t][j];
                JA[i][j] = a;
            }
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += A1[t][i] * JA[t][j];
                nJ[i][j] = a;
            }
        for (uint i = 0; i < M; ++i) {
            ev[i] = ne[i];
            for (uint j = 0; j < M; ++j) Jm[i][j] = 0.5f * (nJ[i][j] + nJ[j][i]) + J1[i][j];
        }
    }

    // ---- seeds: block k from its EXIT state (entry of k+1; the stored
    // exit for the last) and suf[k+1]
    float cotv = cot[chain];
    for (uint k = 0; k < (uint)n_blocks; ++k) {
        float C1[MDIM][MDIM], J2[MDIM][MDIM];
        float b1[MDIM], et2[MDIM];
        uint sv_ = (base + k) * M, sm_ = (base + k) * M * M;
        if (k + 1 < (uint)n_blocks) {
            for (uint i = 0; i < M; ++i) {
                b1[i] = ent_m[(base + k + 1) * M + i];
                for (uint j = 0; j < M; ++j) C1[i][j] = ent_C[(base + k + 1) * M * M + i * M + j];
            }
        } else {
            for (uint i = 0; i < M; ++i) { b1[i] = xb[i]; for (uint j = 0; j < M; ++j) C1[i][j] = xC[i][j]; }
        }
        for (uint i = 0; i < M; ++i) {
            et2[i] = suf_eta[sv_ + i];
            for (uint j = 0; j < M; ++j) J2[i][j] = suf_J[sm_ + i * M + j];
        }
        COMBINE_CORE
        for (uint i = 0; i < M; ++i) {
            seed_m[sv_ + i] = cotv * rr_[i];
            for (uint j = 0; j < M; ++j)
                seed_C[sm_ + i * M + j] = 0.5f * cotv * (rr_[i] * rr_[j] - Jt[i][j]);
        }
    }
"""


_seed_kernels: dict = {}


def _seed_kernel(m: int):
    if m not in _seed_kernels:
        body = _SRC_SEEDS.replace("COMBINE_CORE", _COMBINE_CORE)
        _seed_kernels[m] = mx.fast.metal_kernel(
            name=f"anvilgp_seeds_m{m}",
            input_names=["in_A", "in_b", "in_C", "in_eta", "in_J", "cot",
                         "n_blocks", "n_chains"],
            output_names=["ent_m", "ent_C", "seed_m", "seed_C", "suf_J",
                          "suf_eta"],
            source=body.replace("MDIM", str(m)),
        )
    return _seed_kernels[m]


def block_seeds_kernel(el: Element, cot: mx.array):
    """``(ent_m, ent_C, seed_m, seed_C)`` for every (chain, block), in-kernel.

    The entry state each block starts from and the ``(m_bar, P_bar)`` it
    inherits, from the block elements the forward kernel emits.  One thread
    per chain; see `_SRC_SEEDS`.
    """
    n_chains, n_blocks, m = el.b.shape[0], el.b.shape[1], el.b.shape[-1]
    shapes = [(n_chains, n_blocks, m), (n_chains, n_blocks, m, m),
              (n_chains, n_blocks, m), (n_chains, n_blocks, m, m),
              (n_chains, n_blocks, m, m), (n_chains, n_blocks, m)]
    outs = _seed_kernel(m)(
        inputs=[el.A, el.b, el.C, el.eta, el.J, cot,
                int(n_blocks), int(n_chains)],
        output_shapes=shapes, output_dtypes=[mx.float32] * 6,
        grid=(n_chains, 1, 1), threadgroup=(min(64, n_chains), 1, 1))
    return outs[0], outs[1], outs[2], outs[3]


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
                 blk_len, slots=None) -> mx.array:
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
        # M4.  The cotangent arriving here is on a scalar per chain, which
        # is what `adjoint.filter_backward` is derived for.  Each block's
        # seed comes from the closed form (docs/m4-plan.md), so the blocks
        # are independent and need no chain between them -- same launch
        # shape as the forward.
        #
        # Never reconstruct forward state by running the filter backwards
        # through the propagator: that divides by Phi, which underflowed on
        # the way forward at exactly the lags where it would now overflow,
        # which is every gap in the data.  Replay forward, always -- which
        # is why the block elements are recomputed here.
        p_t, q_t, lg, hv, rs, rr, j2 = primals
        cot = cotangents[0] if isinstance(cotangents, (list, tuple)) \
            else cotangents

        n_chains, n_lags, m = p_t.shape[0], p_t.shape[1], p_t.shape[-1]
        npts = rs.shape[0]
        if slots is None:
            raise ValueError(
                "the kernel gradient needs the (block, lag) slot map; "
                "`scan_loglike` was called without `slots`")
        # n_slots arrives as a plain int: reading it off `slot_start`
        # would force a device sync on every gradient call.
        slot_id, slot_start, n_slots, fold_plan = slots

        # The adjoint buffer is one slot per (block, lag) pair that occurs.
        # It is bounded by N however irregular the cadence, so this fence
        # is a sanity check rather than the wall it was when the buffer had
        # a slot per pair that *could* occur (40 GiB at 1024 chains x
        # N = 1e5, where the slots are 3 GiB).
        slab = 2 * n_chains * n_slots * m * m * 4
        if slab > MAX_ADJOINT_BUFFER_BYTES:
            raise ValueError(
                f"the kernel gradient's adjoint buffer would be "
                f"{slab / 2**30:.1f} GiB (2 x {n_chains} chains x {n_slots} "
                f"(block, lag) slots x m^2 = {m * m} x 4 bytes), over "
                f"MAX_ADJOINT_BUFFER_BYTES = {MAX_ADJOINT_BUFFER_BYTES / 2**30:.0f} GiB. "
                f"Every distinct float32 spacing in the timestamps is a "
                f"lag, and this cadence has {n_lags - 1} of them over "
                f"{npts} points.  Bin the timestamps, use fewer chains, or "
                f"use GPPolicy(metal=False) for the gradient.")

        el = scan_blocks(p_t, q_t, lg, hv, rs, rr, j2, s_floor_rel, blk_len)
        ent_m, ent_C, seed_m, seed_C = block_seeds_kernel(el, cot)
        d_phi, d_q, d_h, d_y, d_rv = backward_ckpt(
            p_t, q_t, lg, hv, rs, rr, j2, cot, ent_m, ent_C, seed_m, seed_C,
            slot_id, slot_start, n_slots, blk_len)

        # `lag` is an integer index and `r_ref` is the data's own yerr^2 --
        # neither is ever a function of the sampled parameters, so neither
        # needs a cotangent.  `jit2` does: S = max(h Cp h^T, 0) + r_ref +
        # jit2, so d/d(jit2) is the sum of the r_var cotangents, which is
        # exactly what the kernel accumulates per block.
        # `metal_eligible` admits at most one jitter, and the kernel reads
        # column 0, so the whole per-block sum belongs to it.
        n_jit = j2.shape[1] if j2.ndim == 2 else 0
        d_j2 = (mx.zeros_like(j2) if n_jit == 0
                else mx.sum(d_rv, axis=1)[:, None])
        # Slots -> the per-lag table the term front end differentiates.
        # Distinct blocks own disjoint slots, so the blocks are already
        # summed and this only regroups.  Applied one fold group at a
        # time: each group's lags are distinct, which is what keeps the
        # scatter reproducible -- see `likelihood._lag_fold_plan`.  The
        # all-distinct cadence this work is for has exactly one group.
        kind, plan = fold_plan
        if kind == "dense":
            # One gather through a (n_lags, max_rank) table of slot indices
            # padded with a zero row, then one `mx.sum` -- which is
            # reproducible where a repeated-index scatter is not.
            pad = mx.zeros((n_chains, 1, m, m), dtype=d_phi.dtype)
            d_phi_tab = mx.sum(
                mx.take(mx.concatenate([d_phi, pad], axis=1), plan, axis=1),
                axis=2)
            d_q_tab = mx.sum(
                mx.take(mx.concatenate([d_q, pad], axis=1), plan, axis=1),
                axis=2)
        else:
            d_phi_tab = mx.zeros((n_chains, n_lags, m, m), dtype=d_phi.dtype)
            d_q_tab = d_phi_tab
            for sel, lags_of in plan:
                d_phi_tab = d_phi_tab.at[:, lags_of].add(
                    mx.take(d_phi, sel, axis=1))
                d_q_tab = d_q_tab.at[:, lags_of].add(
                    mx.take(d_q, sel, axis=1))
        return (d_phi_tab, d_q_tab,
                mx.zeros_like(lg), mx.sum(d_h, axis=1), d_y,
                mx.zeros_like(rr), d_j2)

    return core(phi_tab, q_tab, lag, hvec, resid, r_ref, jit2)
