"""Fused Metal kernel for the blocked scan -- Milestone 3's hot path.

The MLX reference in :mod:`anvilgp.elements` is bound by op count and by
*materialising* the scan-element array: at 1024 chains x 4096 points it
writes and re-reads ``3m^2 + 2m + 1`` numbers per observation at every step,
and measures ~40x below this machine's memory bandwidth -- i.e. it is not
moving data fast, it is moving it too many times.  A kernel that walks a
block of observations with the running element **in registers** writes none
of it: one element per *block* reaches memory instead of one per point.

This is Design C's pass 1.  The value then takes two more kernels: a
prefix over the ~N/L block elements per chain gives each block its entry
state (`entry_states`), and the filter replayed through each block from it
gives the log-likelihood (`replay_blocks`).  Until 2026-10-24 the second
pass was `reduce_blocks`, a fold of the block elements' own log-normalisers,
which carry the dense information form's floor (`_SRC_REPLAY`); and until
2026-09-20 that fold was an MLX tree, launch-bound at a flat ~5.1 ms.

Since 2026-10-25 the entry states, and the backward's seeds, are by default
not the prefix over pass 1's dense elements but a second condensation of
every block in the square-root information form, centred on the filter
replayed from those first entry states (`q_factor`, `sqrt_blocks`,
`sqrt_scan`; `scan_loglike`'s ``boundary``).  The dense ones are float32-
wrong on bright, densely sampled data; see docs/boundary-plan.md.

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
import numpy as np


from .elements import Element

__all__ = ["scan_blocks", "scan_loglike", "reduce_blocks",
           "entry_states", "replay_blocks", "q_factor", "sqrt_blocks", "sqrt_scan",
           "backward_ckpt", "block_seeds_kernel",
           "metal_available", "MAX_STATE",
           "MAX_ADJOINT_BUFFER_BYTES"]

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
def _default_adjoint_budget() -> int:
    """8 GiB, or a quarter of the device's memory if that is smaller.

    The fixed 8 GiB was calibrated on a 64 GB M2 Max, where the README
    records a 3.05 GiB buffer sitting inside a 17.5 GiB gradient -- a
    ~5.7x multiplier, so 8 GiB of buffer implies a ~45 GiB run.  On a
    16 GB machine the allocator kills the process long before the fence
    trips, and the user gets a kill instead of the fence's message naming
    the cadence and the remedies.  A quarter of device memory keeps the
    implied total inside what the device has.
    """
    try:
        info = mx.device_info()
    except Exception:                      # no Metal device, or older MLX
        return 8 << 30
    mem = int(info.get("memory_size", 0))
    return min(8 << 30, mem // 4) if mem else 8 << 30


MAX_ADJOINT_BUFFER_BYTES = _default_adjoint_budget()

#: Largest state dimension the kernel accepts.  The running element is
#: ``3m^2 + 2m + 1`` floats held per thread -- 17 at m = 2, 57 at m = 4,
#: 121 at m = 6, 209 at m = 8.  m <= 6 covers every single kernel and the
#: sums people use most, rotation + granulation (``RotationTerm() +
#: SHOTerm()``) and SHO + Matern-5/2 included.  m = 5 and 6 spill
#: registers: value + gradient is ~8x m = 4 rather than the (6/4)^3 = 3.4x
#: of the arithmetic (the replay 22x, the square-root block pass 9x), but
#: still ~10x faster than the MLX path, as accurate or more, and the only
#: gradient at N >= 16384, where the MLX path runs out of Metal buffers
#: (docs/boundary-plan.md, follow-on step 2).  Above 6 the MLX path, until
#: the element is moved to threadgroup memory.
MAX_STATE = 6

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
    JIT_HOIST

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
        //
        // NOT in the FORMAL variant (a TermConvolution), whose Q is only
        // formally a covariance: h Q h^T is genuinely negative there, the
        // algebra is exact with the negative value, and clamping it is what
        // made the answer wrong.  See elements.leaf_elements.  A sum that
        // is not positive is a real failure, and log/rsqrt make it a NaN.
        float rr = r_ref[t];
        JIT_PER_OBS_T
        LEAF_VARIANCE

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
        //ELL{
        float ld2 = -0.5f * metal::precise::log(sq / rr);
        float ell2 = ld2 - 0.5f * gg * gg;
        //ELL}

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
            //ELL{
            ell = ell2; ell_c = 0.0f;
            //ELL}
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
        //ELL{
        float add = ld2
                  + 0.5f * metal::precise::log(LOG_INV_A) - 0.5f * beta * bia;
        float yk = add - ell_c;
        float tk = ell + yk;
        ell_c = (tk - ell) - yk;
        ell = tk;
        //ELL}
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


def _jitter_src(src: str, n_jit: int) -> str:
    """Specialise a kernel source for ``n_jit`` jitter parameters.

    Two variants, on purpose.  With at most one jitter the variance is the
    same for every observation, so it is read **once per thread** -- which
    is what the kernel has always done, and the single-jitter path must
    not start paying a global read per observation for a feature it does
    not use.  ``n_jit < 0`` is a **per-datum** extra variance,
    ``(npts, n_chains)`` in the residual's layout -- a general noise model
    (`GPLogLike`'s ``noise_fn``), or several per-instrument jitters
    expanded by label: read per observation, and the backward writes each
    observation's variance adjoint.  Until 2026-10-25 several jitters had
    their own variant, indexed by instrument with one adjoint accumulator
    per jitter in registers, which capped them at eight (``MAX_JITTER``);
    the per-datum one measured within 0-10% of it and has no cap.
    """
    if n_jit > 1:
        raise ValueError(
            f"{n_jit} compact jitters: pass several per-instrument jitters "
            f"per datum, as jit2 of shape (npts, n_chains, 1)")
    drv_acc = "drv_acc[JIT_IDX] += s_bar;"
    drv_out = ("for (uint j = 0; j < NJIT; ++j)\n"
               "        out_drv[ei * NJIT + j] = drv_acc[j];")
    if n_jit < 0:
        hoist = ""
        per_obs = "float jit2v = jit2[GI * (uint)n_chains + chain];"
        drv_acc = "out_drv[gi * (uint)n_chains + chain] = s_bar;"
        drv_out = ""
    else:
        hoist = ("float jit2v = (n_jit > 0) ? jit2[chain * (uint)n_jit] "
                 ": 0.0f;")
        per_obs = ""
    jit_idx = "0"
    return (src.replace("DRV_ACC", drv_acc)
               .replace("DRV_OUT", drv_out)
               .replace("JIT_HOIST", hoist)
               .replace("JIT_PER_OBS_T", per_obs.replace("GI", "t"))
               .replace("JIT_PER_OBS", per_obs.replace("GI", "gi"))
               .replace("JIT_IDX", jit_idx)
               .replace("NJIT", str(max(1, n_jit))))


def _n_jit_of(jit2: mx.array) -> int:
    """The jitter variant ``jit2``'s shape asks for: ``(n_chains, n_jit)``
    per instrument, ``(npts, n_chains, 1)`` per datum (-1), else none."""
    if jit2.ndim == 3:
        return -1
    return jit2.shape[1] if jit2.ndim == 2 else 0


def _ell_src(src: str, with_ell: bool) -> str:
    """Keep or drop pass 1's own log-likelihood (the ``//ELL{`` ... ``//ELL}``
    regions).  Only :func:`reduce_blocks` reads it now; the value comes from
    the replay and the backward never used it, so both run without -- one
    precise ``log`` and a Kahan step per observation saved.  ``out_ell`` is
    then written as zero."""
    out, keep = [], True
    for line in src.split("\n"):
        tag = line.strip()
        if tag == "//ELL{":
            keep = with_ell
        elif tag == "//ELL}":
            keep = True
        elif keep:
            out.append(line)
    return "\n".join(out)


def _jit_key(n_jit: int) -> int:
    """Cache key for the jitter variant: 0 for none or one jitter (both use
    the hoisted read, whose ternary handles the no-jitter case), ``"d"``
    per datum.  (When several compact jitters had a variant, each count
    needed its own key: ``NJIT`` sized the backward's accumulator, and
    sharing one kernel between 2 and 3 wrote out of bounds, for a 0.44
    relative error in the jitter gradients with the value still right.)
    """
    return "d" if n_jit < 0 else 0


def _formal_src(src: str, formal: bool) -> str:
    """Specialise a kernel source for a formal (convolved) model, or not.

    A :class:`~anvilgp.terms.TermConvolution`'s state-space form is only
    formally Gaussian: its variances can be genuinely negative and its
    covariances indefinite, while the algebra stays exact.  So the formal
    variant drops the positivity clamps that repair float32 rounding for
    every other term (they would *change* this answer), takes ``log|alpha|``
    in the leaf fold, and composes blocks through a pivoted direct inverse
    instead of a Cholesky factor -- exactly what
    :func:`anvilgp.elements.combine` does with ``formal=True``.  The
    non-formal variant was, when this was introduced, identical to the
    previous source once comments and whitespace are stripped;
    ``tests/test_metal.py`` pins that every clamp survives in it.  The two
    also differ in how a composition forms ``M d`` and ``G^T d`` (through
    the Cholesky factor, or through the inverse) -- see `_COMBINE_RR_CHOL`.
    """
    if formal:
        leaf = "sq += rr + jit2v;"
        hch, rk, clamped = "hch", "", "false"
        red = _FORMAL_FACTOR.replace("C1N", "Cm")
        red_md, comb_rr = _REDUCE_MD_FORMAL, _COMBINE_RR_FORMAL
        logdet = "if (i == 0) logdet = 0.5f * ldx;"
        comb = _FORMAL_FACTOR.replace("C1N", "C1")
        log_a = "fabs(inv_a)"
    else:
        leaf = ("sq = max(sq, 0.0f);\n"
                "        sq += rr + jit2v;\n"
                "        sq = max(sq, sfloor * rr);              "
                "// PSD failure is defined")
        hch, rk, clamped = "max(hch, 0.0f)", " + max(-hch, 0.0f)", "hch < 0.0f"
        red, comb = _REDUCE_CHOL, _COMBINE_CHOL
        red_md, comb_rr = _REDUCE_MD_CHOL, _COMBINE_RR_CHOL
        logdet = "logdet += metal::precise::log(Ls[i][i]);"
        log_a = "inv_a"
    return (src.replace("LEAF_VARIANCE", leaf)
               .replace("REDUCE_CORE", red)
               .replace("REDUCE_MD", red_md)
               .replace("LOGDET_ACC", logdet)
               .replace("COMBINE_FACTOR", comb)
               .replace("COMBINE_RR", comb_rr)
               .replace("LOG_INV_A", log_a)
               .replace("HCH_CLAMPED", hch)
               .replace("RK_CLAMP_TERM", rk)
               .replace("CLAMPED_TEST", clamped))


def _kernel(m: int, n_jit: int = 0, formal: bool = False,
            with_ell: bool = True):
    key = (m, _jit_key(n_jit), bool(formal), bool(with_ell))
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_scan_m{m}_j{key[1]}" + ("_formal" if formal else "")
                 + ("" if with_ell else "_noell"),
            input_names=["phitab", "qtab", "lag", "inst", "hvec", "resid",
                         "r_ref",
                         "jit2", "sfloor", "npts", "n_lags", "n_blocks",
                         "n_chains", "blk_len", "n_jit"],
            output_names=["out_A", "out_b", "out_C", "out_eta", "out_J",
                          "out_ell"],
            source=_formal_src(_jitter_src(_ell_src(_SRC, with_ell), n_jit),
                               formal).replace("MDIM", str(m)),
        )
    return _kernels[key]



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

REDUCE_CORE
        // In the innovation d = eta2 - J2 b1, as `elements.combine` (whose
        // comment has the algebra and the float32 measurements): the
        // composite mean is A2 (b1 + M d) + b2 and the cross terms are
        // b1^T (eta2 - J2 b1 / 2) + d^T M d / 2.
        float jb[MDIM], dv[MDIM], Md[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += J2[i][j] * bv[j];
            jb[i] = a;
            dv[i] = et2[i] - a;
        }
REDUCE_MD
        float logdet = 0.0f, brk = 0.0f;
        for (uint i = 0; i < M; ++i) {
            LOGDET_ACC
            brk += bv[i] * (et2[i] - 0.5f * jb[i]);
        }
        float add = in_ell[base + k] - logdet + brk + 0.5f * quad;
        float yk = add - ell_c;
        float tk = ell + yk;
        ell_c = (tk - ell) - yk;
        ell = tk;

        // b <- A2 (b1 + M d) + b2 ;  C <- sym(A2 M A2^T) + C2
        float nb[MDIM], AM[MDIM][MDIM], nC[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = b2[i];
            for (uint j = 0; j < M; ++j) a += A2[i][j] * (bv[j] + Md[j]);
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


#: The default factorisation in `_SRC_REDUCE`: Cholesky of C1 and of S.
_REDUCE_CHOL = """        // L1 = chol(C1), with linalg.diag_floor's eps*max(trace,0)/m pivot floor
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

        float Mm[MDIM][MDIM];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += Y[i][t] * Y[j][t];
                Mm[i][j] = a;
            }

"""

#: ``M d`` and ``d^T M d`` for `_SRC_REDUCE`.  On the Cholesky route both go
#: through the factor, ``w = Y^T d``, never through ``M`` as a matrix; see
#: `elements.combine`.  The formal route has no factor and uses ``Mm``.
_REDUCE_MD_CHOL = """
        float wv[MDIM], quad = 0.0f;
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += Y[j][i] * dv[j];
            wv[i] = a;
            quad += a * a;
        }
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += Y[i][j] * wv[j];
            Md[i] = a;
        }
"""

_REDUCE_MD_FORMAL = """
        float quad = 0.0f;
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint j = 0; j < M; ++j) a += Mm[i][j] * dv[j];
            Md[i] = a;
            quad += dv[i] * a;
        }
"""

#: The formal variant's replacement for a composition's factorisation, for
#: a convolved term whose ``C1`` and ``J2`` can be indefinite: ``G = (I + C1
#: J2)^-1`` by Gauss-Jordan with partial pivoting, then ``Mm = sym(G C1)``,
#: ``Jt = sym(J2 G)`` and ``ldx = log|det(I + C1 J2)|`` -- the same quantities
#: `elements.combine(formal=True)` forms.  ``C1N`` names the caller's C1.
_FORMAL_FACTOR = """
            float Xf[MDIM][MDIM], G[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = (i == j ? 1.0f : 0.0f);
                    for (uint t = 0; t < M; ++t) a += C1N[i][t] * J2[t][j];
                    Xf[i][j] = a;
                    G[i][j] = (i == j ? 1.0f : 0.0f);
                }
            float ldx = 0.0f;
            for (uint k = 0; k < M; ++k) {
                uint pr = k;
                float pv = fabs(Xf[k][k]);
                for (uint r = k + 1; r < M; ++r) {
                    float av = fabs(Xf[r][k]);
                    if (av > pv) { pv = av; pr = r; }
                }
                if (pr != k) {
                    for (uint j = 0; j < M; ++j) {
                        float t1 = Xf[k][j]; Xf[k][j] = Xf[pr][j]; Xf[pr][j] = t1;
                        float t2 = G[k][j]; G[k][j] = G[pr][j]; G[pr][j] = t2;
                    }
                }
                float dk = Xf[k][k];
                ldx += metal::precise::log(fabs(dk));
                float ik = 1.0f / dk;
                for (uint j = 0; j < M; ++j) { Xf[k][j] *= ik; G[k][j] *= ik; }
                for (uint r = 0; r < M; ++r) {
                    if (r == k) continue;
                    float f = Xf[r][k];
                    for (uint j = 0; j < M; ++j) {
                        Xf[r][j] -= f * Xf[k][j];
                        G[r][j] -= f * G[k][j];
                    }
                }
            }
            float Mm[MDIM][MDIM], Jt[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f, c = 0.0f;
                    for (uint t = 0; t < M; ++t) {
                        a += G[i][t] * C1N[t][j];
                        c += J2[i][t] * G[t][j];
                    }
                    Mm[i][j] = a; Jt[i][j] = c;
                }
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < i; ++j) {
                    float a = 0.5f * (Mm[i][j] + Mm[j][i]);
                    Mm[i][j] = a; Mm[j][i] = a;
                    float c = 0.5f * (Jt[i][j] + Jt[j][i]);
                    Jt[i][j] = c; Jt[j][i] = c;
                }
"""


_reduce_kernels: dict = {}


def _reduce_kernel(m: int, formal: bool = False):
    key = (m, bool(formal))
    if key not in _reduce_kernels:
        _reduce_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_reduce_m{m}" + ("_formal" if formal else ""),
            input_names=["in_A", "in_b", "in_C", "in_eta", "in_J", "in_ell",
                         "n_blocks", "n_chains"],
            output_names=["out_ell"],
            source=_formal_src(_SRC_REDUCE, formal).replace("MDIM", str(m)),
        )
    return _reduce_kernels[key]


def reduce_blocks(el: Element, formal: bool = False) -> mx.array:
    """Fold ``(n_chains, n_blocks)`` block elements to ``(n_chains,)``."""
    n_chains, n_blocks, m = el.b.shape[0], el.b.shape[1], el.b.shape[-1]
    if n_blocks == 1:
        return el.ell[:, 0]
    out = _reduce_kernel(m, formal)(
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
MEAN_PREDICT
            float cp[MDIM][MDIM], ch[MDIM];
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
            JIT_PER_OBS
            float sv = HCH_CLAMPED + r_ref[gi] + jit2v;
INNOVATION
FWD_ELL
            float kg[MDIM];
            for (uint i = 0; i < M; ++i) kg[i] = ch[i] / sv;
MEAN_UPDATE
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
            float rk = r_ref[gi] + jit2v RK_CLAMP_TERM;
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

#: The replay's log-likelihood term, ``-1/2 (log(S / r_ref) + v^2 / S)``,
#: Kahan-summed into ``ell``/``ell_c`` as pass 1 sums its own.
_FWD_ELL = """            {
                float add = -0.5f * (metal::precise::log(sv / r_ref[gi])
                                     + vv * vv / sv);
                float yk = add - ell_c;
                float tk = ell + yk;
                ell_c = (tk - ell) - yk;
                ell = tk;
            }"""


#: The data-dependent lines of `_FWD_STEP`, one column: the predicted
#: mean, the innovation ``vv`` and the mean update.  Everything else in the
#: step -- the covariance recursion, ``sv``, the gain -- is the same for
#: every residual the filter is run on, which is what the multi-column
#: replay (`_FWD_MC`) uses.
_FWD_1 = {
    "MEAN_PREDICT": """            float mp[MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += phitab[off + i * M + j] * mean[j];
                mp[i] = a;
            }""",
    "INNOVATION": """            float vv = resid[gi * (uint)n_chains + chain];
            for (uint i = 0; i < M; ++i) vv -= hv[i] * mp[i];""",
    "MEAN_UPDATE": """            for (uint i = 0; i < M; ++i) mean[i] = mp[i] + kg[i] * vv;""",
}

#: The same lines for ``KCOL`` residual columns sharing one covariance:
#: ``mean[c]``, ``vv[c]``, the residual read from ``(npts, n_chains,
#: KCOL)``.  The log-likelihood is replaced by the log-determinant half
#: ``-1/2 sum log(S / r_ref)`` and the packed lower triangle of the
#: innovation Gram matrix ``sum vv_a vv_b / S``, each Kahan-summed as
#: `_FWD_ELL` is: for columns ``z_a`` that is ``z_a^T K^-1 z_b`` exactly,
#: since the innovations whitened by ``sqrt(S)`` are ``L^-1 z``.
_FWD_MC = {
    "MEAN_PREDICT": """            float mp[KCOL][MDIM];
            for (uint c = 0; c < KCOL; ++c)
                for (uint i = 0; i < M; ++i) {
                    float a = 0.0f;
                    for (uint j = 0; j < M; ++j)
                        a += phitab[off + i * M + j] * mean[c][j];
                    mp[c][i] = a;
                }""",
    "INNOVATION": """            float vv[KCOL];
            for (uint c = 0; c < KCOL; ++c) {
                float a = resid[(gi * (uint)n_chains + chain) * KCOL + c];
                for (uint i = 0; i < M; ++i) a -= hv[i] * mp[c][i];
                vv[c] = a;
            }""",
    "MEAN_UPDATE": """            for (uint c = 0; c < KCOL; ++c)
                for (uint i = 0; i < M; ++i) mean[c][i] = mp[c][i] + kg[i] * vv[c];""",
    "FWD_ELL": """            {
                float add = -0.5f * metal::precise::log(sv / r_ref[gi]);
                float yk = add - ell_c;
                float tk = ell + yk;
                ell_c = (tk - ell) - yk;
                ell = tk;
            }
            // The loop shape is measured, not tidy: the outer loop rolled,
            // the inner unrolled to a constant bound, 1/S hoisted.  Full
            // unrolling had a cliff at KCOL = 7-8 (10-15x slower than either
            // side of it, N = 4096 x 128 chains, m = 2) and a running index
            // was several x slower everywhere; this is the fastest of the
            // four tried at every KCOL from 3 to 16.
            float isv = 1.0f / sv;
            for (uint a = 0; a < KCOL; ++a) {
                float va = vv[a] * isv;
                #pragma unroll
                for (uint b = 0; b < KCOL; ++b)
                    if (b <= a) {
                        uint t = a * (a + 1) / 2 + b;
                        float add = va * vv[b];
                        float yk = add - gram_c[t];
                        float tk = gram[t] + yk;
                        gram_c[t] = (tk - gram[t]) - yk;
                        gram[t] = tk;
                    }
            }""",
}


#: The affine pass's addition to the multi-column mean update: the block's
#: mean map ``T <- (I - K h) Phi T``, so that the exit mean from an entry
#: mean ``m`` is ``T m + d`` with ``d`` the exit mean from zero.
_FWD_AFFINE_T = """
            {
                float pt[MDIM][MDIM], hpt[MDIM];
                for (uint i = 0; i < M; ++i)
                    for (uint j = 0; j < M; ++j) {
                        float a = 0.0f;
                        for (uint u = 0; u < M; ++u)
                            a += phitab[off + i * M + u] * tm[u][j];
                        pt[i][j] = a;
                    }
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f;
                    for (uint i = 0; i < M; ++i) a += hv[i] * pt[i][j];
                    hpt[j] = a;
                }
                for (uint i = 0; i < M; ++i)
                    for (uint j = 0; j < M; ++j)
                        tm[i][j] = pt[i][j] - kg[i] * hpt[j];
            }"""


#: `_FWD_AFFINE_T` plus the r-sensitivity: ``g[c] += (h^T Phi T)^T v_c /
#: S``, with ``T`` (before its update) the map from the block's entry mean
#: to the previous filtered mean, so ``-h^T Phi T`` is ``d v / d m_entry``
#: and ``g[c]`` is ``-1/2 d(sum v_c^2 / S) / d m_entry`` from the actual
#: innovations.
_FWD_AFFINE_TG = _FWD_AFFINE_T.replace(
    """                for (uint i = 0; i < M; ++i)
                    for (uint j = 0; j < M; ++j)
                        tm[i][j] = pt[i][j] - kg[i] * hpt[j];""",
    """                float isv = 1.0f / sv;
                for (uint c = 0; c < KCOL; ++c) {
                    float w = vv[c] * isv;
                    for (uint j = 0; j < M; ++j) gacc[c][j] += hpt[j] * w;
                }
                for (uint i = 0; i < M; ++i)
                    for (uint j = 0; j < M; ++j)
                        tm[i][j] = pt[i][j] - kg[i] * hpt[j];""")
assert _FWD_AFFINE_TG != _FWD_AFFINE_T


def _fwd_step(with_ell: bool = False, multi: bool = False,
              affine: bool = False, sens: bool = False) -> str:
    """`_FWD_STEP`, accumulating the log-likelihood or not.  The backward's
    replays take it without, so their source is what it always was.
    ``multi`` is the ``KCOL``-column step of `_FWD_MC` (``with_ell``: its
    log-determinant and Gram matrix accumulated); ``affine`` the
    ``KCOL``-column step of the affine pass, which accumulates nothing and
    carries the mean map ``tm`` (`_FWD_AFFINE_T`)."""
    if affine or sens:
        parts = dict(_FWD_MC, FWD_ELL="")
        parts["MEAN_UPDATE"] = (_FWD_MC["MEAN_UPDATE"]
                                + (_FWD_AFFINE_TG if sens else _FWD_AFFINE_T))
    elif multi:
        parts = dict(_FWD_MC, FWD_ELL=_FWD_MC["FWD_ELL"] if with_ell else "")
    else:
        parts = dict(_FWD_1, FWD_ELL=_FWD_ELL if with_ell else "")
    src = _FWD_STEP
    for tok in ("MEAN_PREDICT", "INNOVATION", "MEAN_UPDATE"):
        src = src.replace(tok + "\n", parts[tok] + "\n")
    return src.replace("FWD_ELL\n", parts["FWD_ELL"] + "\n"
                       if parts["FWD_ELL"] else "")


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
    JIT_HOIST
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
    // One r_var adjoint per jitter parameter.  Every slot is zeroed, not
    // only the ones this block's instruments touch: `mx.fast.metal_kernel`
    // does not zero its outputs, so an untouched slot would carry whatever
    // was in the buffer.
    float drv_acc[NJIT];
    for (uint j = 0; j < NJIT; ++j) drv_acc[j] = 0.0f;
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
            bool clamped = CLAMPED_TEST;
            JIT_PER_OBS
            float sv = HCH_CLAMPED + r_ref[gi] + jit2v;
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
            DRV_ACC
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
    DRV_OUT
"""


_bwd_ckpt_kernels: dict = {}

#: sub-block length by state dimension: what a thread can hold in
#: ``smean``/``scov`` alongside the sweep's own working set.
_SUB_LEN = {1: 32, 2: 32, 3: 16, 4: 8}


def _bwd_ckpt_kernel(m: int, slen: int, n_jit: int = 0, formal: bool = False):
    key = (m, slen, _jit_key(n_jit), bool(formal))
    if key not in _bwd_ckpt_kernels:
        body = _formal_src(_jitter_src(
            _SRC_BWD_CKPT.replace("FWD_STEP", _fwd_step()), n_jit), formal)
        _bwd_ckpt_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_bwd_ckpt_m{m}_s{slen}_j{key[2]}"
                 + ("_formal" if formal else ""),
            input_names=["phitab", "qtab", "lag", "inst", "hvec", "resid",
                         "r_ref",
                         "jit2", "cot", "ent_m", "ent_C", "seed_m", "seed_C",
                         "slot_id", "slot_start",
                         "npts", "n_lags", "n_blocks", "n_chains", "blk_len",
                         "n_jit", "n_ckpt", "n_slots"],
            output_names=["out_dphi", "out_dq", "out_dh", "out_dy",
                          "out_drv", "ckpt"],
            source=body.replace("MDIM", str(m)).replace("SLEN", str(slen)),
        )
    return _bwd_ckpt_kernels[key]


def backward_ckpt(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot,
                  ent_m, ent_C, seed_m, seed_C, slot_id, slot_start, n_slots,
                  blk_len, sub_len=None, formal=False):
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
    it feeds the mean model.  ``drv`` is ``(n_chains, n_blocks, n_jit)`` --
    one r_var adjoint per jitter parameter, so a per-instrument jitter gets
    its own gradient; with no jitter the last axis is 1 and unused.

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
    n_jit = _n_jit_of(jit2)
    n_slots = int(n_slots)
    shapes = [(n_chains, n_slots, m, m),
              (n_chains, n_slots, m, m),
              (n_chains, n_blocks, m), (npts, n_chains),
              ((npts, n_chains, 1) if n_jit < 0
               else (n_chains, n_blocks, max(1, n_jit))),
              (n_chains, n_blocks, n_ckpt, m + m * m)]
    outs = _bwd_ckpt_kernel(m, slen, n_jit, formal)(
        inputs=[phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, cot,
                ent_m, ent_C, seed_m, seed_C, slot_id, slot_start,
                int(npts), int(n_lags), int(n_blocks), int(n_chains),
                int(blk_len), int(n_jit), int(n_ckpt), int(n_slots)],
        output_shapes=shapes, output_dtypes=[mx.float32] * 6,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return outs[:5]




#: The multi-column backward: `_SRC_BWD_CKPT` for ``KCOL`` columns sharing
#: one covariance, differentiating ``c_ld * ld + sum_ab gs_ab G_ab`` (the
#: log-determinant half and the Gram matrix of `scan_gram`; ``gs`` the
#: symmetrised cotangent on ``G``).  Derived from the single-column source
#: by the replacements below, each asserted to apply exactly once, so the
#: covariance adjoint -- most of the kernel -- stays one source.  What
#: changes is per column: the means, their checkpoints and adjoints
#: ``m_bar[c]``, the innovations, ``dy``.  The step's own terms are
#:
#:     v_bar[a] = 2 (gs v)_a / S,   s_bar = -c_ld / (2 S) - v^T gs v / S^2
#:
#: -- with ``c_ld = cot`` and ``gs = -cot / 2`` (one column) exactly the
#: single-column ``-cot v / S`` and ``cot (v^2 - S) / (2 S^2)``.
_BWD_MC_EDITS = [
    ("""    float cotv = cot[chain];
""", """    float cld = cot[chain];
    float gs[KCOL][KCOL];
    for (uint a = 0; a < KCOL; ++a)
        for (uint b = 0; b < KCOL; ++b)
            gs[a][b] = gbar[(chain * KCOL + a) * KCOL + b];
"""),
    ("""    float mean[MDIM], cov[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) {
        mean[i] = ent_m[ei * M + i];
        for (uint j = 0; j < M; ++j) cov[i][j] = ent_C[ei * M * M + i * M + j];
    }
    const uint CK = M + M * M;
""", """    float mean[KCOL][MDIM], cov[MDIM][MDIM];
    for (uint c = 0; c < KCOL; ++c)
        for (uint i = 0; i < M; ++i) mean[c][i] = ent_m[(ei * KCOL + c) * M + i];
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) cov[i][j] = ent_C[ei * M * M + i * M + j];
    const uint CK = KCOL * M + M * M;
"""),
    ("""            for (uint i = 0; i < M; ++i) {
                ckpt[cb + i] = mean[i];
                for (uint j = 0; j < M; ++j) ckpt[cb + M + i * M + j] = cov[i][j];
            }
""", """            for (uint c = 0; c < KCOL; ++c)
                for (uint i = 0; i < M; ++i) ckpt[cb + c * M + i] = mean[c][i];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j)
                    ckpt[cb + KCOL * M + i * M + j] = cov[i][j];
"""),
    ("""    float m_bar[MDIM], p_bar[MDIM][MDIM], h_bar[MDIM];
    for (uint i = 0; i < M; ++i) {
        m_bar[i] = seed_m[ei * M + i];
        h_bar[i] = 0.0f;
""", """    float m_bar[KCOL][MDIM], p_bar[MDIM][MDIM], h_bar[MDIM];
    for (uint c = 0; c < KCOL; ++c)
        for (uint i = 0; i < M; ++i)
            m_bar[c][i] = seed_m[(ei * KCOL + c) * M + i];
    for (uint i = 0; i < M; ++i) {
        h_bar[i] = 0.0f;
"""),
    ("""        for (uint i = 0; i < M; ++i) {
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
""", """        for (uint c = 0; c < KCOL; ++c)
            for (uint i = 0; i < M; ++i) mean[c][i] = ckpt[cb + c * M + i];
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j)
                cov[i][j] = ckpt[cb + KCOL * M + i * M + j];
        float smean[SLEN][KCOL][MDIM], scov[SLEN][MDIM][MDIM];
        for (uint t = t0; t < t1; ++t) {
            uint li = t - t0;
            for (uint c = 0; c < KCOL; ++c)
                for (uint i = 0; i < M; ++i) smean[li][c][i] = mean[c][i];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) scov[li][i][j] = cov[i][j];
"""),
    ("""            float mp[MDIM], cp[MDIM][MDIM], ch[MDIM], kg[MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += Ph[i][j] * smean[li][j];
                mp[i] = a;
            }
""", """            float mp[KCOL][MDIM], cp[MDIM][MDIM], ch[MDIM], kg[MDIM];
            for (uint c = 0; c < KCOL; ++c)
                for (uint i = 0; i < M; ++i) {
                    float a = 0.0f;
                    for (uint j = 0; j < M; ++j)
                        a += Ph[i][j] * smean[li][c][j];
                    mp[c][i] = a;
                }
"""),
    ("""            float vv = resid[gi * (uint)n_chains + chain];
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
""", """            float vv[KCOL];
            for (uint c = 0; c < KCOL; ++c) {
                float a = resid[(gi * (uint)n_chains + chain) * KCOL + c];
                for (uint i = 0; i < M; ++i) a -= hv[i] * mp[c][i];
                vv[c] = a;
            }
            for (uint i = 0; i < M; ++i) kg[i] = ch[i] / sv;

            float isv = 1.0f / sv;
            float v_bar[KCOL], vgv = 0.0f;
            for (uint a = 0; a < KCOL; ++a) {
                float g = 0.0f;
                for (uint b = 0; b < KCOL; ++b) g += gs[a][b] * vv[b];
                vgv += vv[a] * g;
                v_bar[a] = 2.0f * g * isv;
            }
            float s_bar = -0.5f * cld * isv - vgv * isv * isv;

            float kg_bar[MDIM], ch_bar[MDIM], mp_bar[KCOL][MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f, b = 0.0f;
                for (uint j = 0; j < M; ++j) {
                    a += p_bar[i][j] * ch[j];
                    b += p_bar[j][i] * kg[j];
                }
                float mv = 0.0f;
                for (uint c = 0; c < KCOL; ++c) mv += m_bar[c][i] * vv[c];
                kg_bar[i] = -a + mv;
                ch_bar[i] = -b;
            }
            for (uint c = 0; c < KCOL; ++c) {
                float a = 0.0f;
                for (uint i = 0; i < M; ++i) {
                    a += kg[i] * m_bar[c][i];
                    mp_bar[c][i] = m_bar[c][i];
                }
                v_bar[c] += a;
                out_dy[(gi * (uint)n_chains + chain) * KCOL + c] = v_bar[c];
                for (uint i = 0; i < M; ++i) {
                    h_bar[i] -= v_bar[c] * mp[c][i];
                    mp_bar[c][i] -= v_bar[c] * hv[i];
                }
            }
"""),
    ("""                    acc_p[i][j] += a + mp_bar[i] * smean[li][j];
""", """                    float mm = 0.0f;
                    for (uint c = 0; c < KCOL; ++c)
                        mm += mp_bar[c][i] * smean[li][c][j];
                    acc_p[i][j] += a + mm;
"""),
    ("""            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) {
                    p_bar[i][j] = 0.5f * (nb[i][j] + nb[j][i]);
                    a += Ph[j][i] * mp_bar[j];
                }
                m_bar[i] = a;
            }
""", """            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j)
                    p_bar[i][j] = 0.5f * (nb[i][j] + nb[j][i]);
            for (uint c = 0; c < KCOL; ++c)
                for (uint i = 0; i < M; ++i) {
                    float a = 0.0f;
                    for (uint j = 0; j < M; ++j) a += Ph[j][i] * mp_bar[c][j];
                    m_bar[c][i] = a;
                }
"""),
]


def _bwd_mc_src() -> str:
    src = _SRC_BWD_CKPT
    for old, new in _BWD_MC_EDITS:
        if src.count(old) != 1:
            raise RuntimeError(
                "the multi-column backward no longer derives from "
                "_SRC_BWD_CKPT: an edit's anchor matched "
                f"{src.count(old)} times:\n{old}")
        src = src.replace(old, new)
    return src.replace("FWD_STEP", _fwd_step(multi=True))


_bwd_mc_kernels: dict = {}


def _sub_len_mc(m: int, k: int) -> int:
    """The multi-column sweep's sub-block.  Measured, not derived: 16 was
    the fastest of 1-32, or within 2%, at every (m, k) tried -- m = 2 with
    k = 3, 6, 11, m = 4 and 6 with k = 1 and 3 (N = 1e4 x 1024 chains,
    2026-10-26) -- where sizing it to the single-column sweep's register
    budget gave 4 at k = 11 and ran 12% slower."""
    return 16


def _bwd_mc_kernel(m: int, k: int, slen: int, n_jit: int = 0,
                   formal: bool = False):
    key = (m, k, slen, _jit_key(n_jit), bool(formal))
    if key not in _bwd_mc_kernels:
        body = _formal_src(_jitter_src(_bwd_mc_src(), n_jit), formal)
        _bwd_mc_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_bwd_mc_m{m}_k{k}_s{slen}_j{key[3]}"
                 + ("_formal" if formal else ""),
            input_names=["phitab", "qtab", "lag", "inst", "hvec", "resid",
                         "r_ref", "jit2", "cot", "gbar", "ent_m", "ent_C",
                         "seed_m", "seed_C", "slot_id", "slot_start",
                         "npts", "n_lags", "n_blocks", "n_chains", "blk_len",
                         "n_jit", "n_ckpt", "n_slots"],
            output_names=["out_dphi", "out_dq", "out_dh", "out_dy",
                          "out_drv", "ckpt"],
            source=body.replace("KCOL", str(k)).replace("MDIM", str(m))
                       .replace("SLEN", str(slen)),
        )
    return _bwd_mc_kernels[key]


def backward_ckpt_mc(phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2,
                     c_ld, gs, ent_m, ent_C, seed_m, seed_C, slot_id,
                     slot_start, n_slots, blk_len, formal=False):
    """:func:`backward_ckpt` for :func:`scan_gram`: the cotangent is
    ``c_ld`` ``(n_chains,)`` on the log-determinant half and the symmetric
    ``gs`` ``(n_chains, k, k)`` on the Gram matrix; ``cols``, ``ent_m``,
    ``seed_m`` and the returned ``dy`` carry a column axis
    (``(npts, n_chains, k)``, ``(n_chains, n_blocks, k, m)``)."""
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts, k = cols.shape[0], cols.shape[-1]
    slen = _sub_len_mc(m, k)
    n_blocks = (npts + blk_len - 1) // blk_len
    n_ckpt = (blk_len + slen - 1) // slen
    n_jit = _n_jit_of(jit2)
    n_slots = int(n_slots)
    shapes = [(n_chains, n_slots, m, m), (n_chains, n_slots, m, m),
              (n_chains, n_blocks, m), (npts, n_chains, k),
              ((npts, n_chains, 1) if n_jit < 0
               else (n_chains, n_blocks, max(1, n_jit))),
              (n_chains, n_blocks, n_ckpt, k * m + m * m)]
    outs = _bwd_mc_kernel(m, k, slen, n_jit, formal)(
        inputs=[phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2,
                c_ld.astype(mx.float32), mx.contiguous(gs.astype(mx.float32)),
                ent_m, ent_C, mx.contiguous(seed_m), mx.contiguous(seed_C),
                slot_id, slot_start,
                int(npts), int(n_lags), int(n_blocks), int(n_chains),
                int(blk_len), int(n_jit), int(n_ckpt), int(n_slots)],
        output_shapes=shapes, output_dtypes=[mx.float32] * 6,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return outs[:5]


#: The `combine` core shared by the prefix, the suffix step and the seed:
#: from the earlier element's ``(b1, C1)`` and the later one's ``(J2,
#: et2)``, form ``Mm = (I + C1 J2)^-1 C1``, ``Jt = (I + J2 C1)^-1 J2``, and,
#: in the innovation ``d = et2 - J2 b1``, ``Md = Mm d`` and
#: ``r = G^T d = (I + J2 C1)^-1 d``.  Same Cholesky route as
#: `reduce_blocks`; ``Md`` and ``r`` go through the factor (`COMBINE_RR`).
_COMBINE_CHOL = """
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
            float Mm[MDIM][MDIM], V[MDIM][MDIM], Jt[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f;
                    for (uint t = 0; t < M; ++t) a += Y[i][t] * Y[j][t];
                    Mm[i][j] = a;
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
"""

_COMBINE_CORE = """
COMBINE_FACTOR
            float rr_[MDIM], dv[MDIM], Md[MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += J2[i][j] * b1[j];
                dv[i] = et2[i] - a;
            }
COMBINE_RR
"""

#: ``Md`` and ``r`` through the factor: with ``w = Y^T d``, ``Md = Y w`` and
#: ``r = d - V w`` (``V = J2 Y``).  Forming ``G = I - Mm J2`` as a matrix
#: instead left an absolute error of eps |Mm| |J2| in it, ~1e-3 when a block
#: starts right after an observation, and ``r`` multiplies it by ``|d|``.
_COMBINE_RR_CHOL = """
            float wv[MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += Y[j][i] * dv[j];
                wv[i] = a;
            }
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f, c = 0.0f;
                for (uint j = 0; j < M; ++j) { a += Y[i][j] * wv[j]; c += V[i][j] * wv[j]; }
                Md[i] = a;
                rr_[i] = dv[i] - c;
            }
"""

_COMBINE_RR_FORMAL = """
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f, c = 0.0f;
                for (uint j = 0; j < M; ++j) { a += Mm[i][j] * dv[j]; c += G[j][i] * dv[j]; }
                Md[i] = a;
                rr_[i] = c;
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
_SEEDS_HEAD = """
    uint chain = thread_position_in_grid.x;
    if (chain >= (uint)n_chains) return;
    const uint M = MDIM;
    uint base = chain * (uint)n_blocks;
"""

#: The prefix over a chain's block elements, emitting each block's entry
#: state.  Shared by the seeds kernel and the forward's entry-state kernel,
#: so the replay that gives the value starts from exactly the states the
#: backward replays from.
_SRC_PREFIX = """
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
        // b <- A2 (b1 + Mm d) + b2 ;  C <- sym(A2 Mm A2^T) + C2
        float nb[MDIM], AM[MDIM][MDIM], nC[MDIM][MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = b2[i];
            for (uint j = 0; j < M; ++j) a += A2[i][j] * (b1[j] + Md[j]);
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
"""

_SRC_SEEDS = _SEEDS_HEAD + _SRC_PREFIX + """    // exit state of the last block, needed for its (zero) seed below
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


def _seed_kernel(m: int, formal: bool = False):
    key = (m, bool(formal))
    if key not in _seed_kernels:
        body = _formal_src(_SRC_SEEDS.replace("COMBINE_CORE", _COMBINE_CORE),
                           formal)
        _seed_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_seeds_m{m}" + ("_formal" if formal else ""),
            input_names=["in_A", "in_b", "in_C", "in_eta", "in_J", "cot",
                         "n_blocks", "n_chains"],
            output_names=["ent_m", "ent_C", "seed_m", "seed_C", "suf_J",
                          "suf_eta"],
            source=body.replace("MDIM", str(m)),
        )
    return _seed_kernels[key]


def block_seeds_kernel(el: Element, cot: mx.array, formal: bool = False):
    """``(ent_m, ent_C, seed_m, seed_C)`` for every (chain, block), in-kernel.

    The entry state each block starts from and the ``(m_bar, P_bar)`` it
    inherits, from the block elements the forward kernel emits.  One thread
    per chain; see `_SRC_SEEDS`.
    """
    n_chains, n_blocks, m = el.b.shape[0], el.b.shape[1], el.b.shape[-1]
    shapes = [(n_chains, n_blocks, m), (n_chains, n_blocks, m, m),
              (n_chains, n_blocks, m), (n_chains, n_blocks, m, m),
              (n_chains, n_blocks, m, m), (n_chains, n_blocks, m)]
    outs = _seed_kernel(m, formal)(
        inputs=[el.A, el.b, el.C, el.eta, el.J, cot,
                int(n_blocks), int(n_chains)],
        output_shapes=shapes, output_dtypes=[mx.float32] * 6,
        grid=(n_chains, 1, 1), threadgroup=(min(64, n_chains), 1, 1))
    return outs[0], outs[1], outs[2], outs[3]


_entry_kernels: dict = {}


def _entry_kernel(m: int, formal: bool = False):
    key = (m, bool(formal))
    if key not in _entry_kernels:
        body = _formal_src((_SEEDS_HEAD + _SRC_PREFIX)
                           .replace("COMBINE_CORE", _COMBINE_CORE), formal)
        _entry_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_entry_m{m}" + ("_formal" if formal else ""),
            input_names=["in_A", "in_b", "in_C", "in_eta", "in_J",
                         "n_blocks", "n_chains"],
            output_names=["ent_m", "ent_C"],
            source=body.replace("MDIM", str(m)),
        )
    return _entry_kernels[key]


def entry_states(el: Element, formal: bool = False):
    """``(ent_m, ent_C)``: the filtered state each block starts from.

    The prefix half of :func:`block_seeds_kernel`, the same source, so the
    forward's replay starts from exactly the states the backward replays
    from.  One thread per chain.
    """
    n_chains, n_blocks, m = el.b.shape[0], el.b.shape[1], el.b.shape[-1]
    outs = _entry_kernel(m, formal)(
        inputs=[el.A, el.b, el.C, el.eta, el.J, int(n_blocks), int(n_chains)],
        output_shapes=[(n_chains, n_blocks, m), (n_chains, n_blocks, m, m)],
        output_dtypes=[mx.float32] * 2,
        grid=(n_chains, 1, 1), threadgroup=(min(64, n_chains), 1, 1))
    return outs[0], outs[1]


#: Pass 3: the value, by replaying the filter through each block from its
#: entry state.  One thread per (chain, block), as pass 1.
#:
#: Why the value is not pass 2's fold of the block ``ell``: an element's
#: log-normaliser is for a *zero* incoming state, ~-1/2 sum g^2 over its
#: block -- 1e4-1e6 on bright, densely sampled data -- and every
#: composition cancels it against a cross term of the same size, so the
#: float32 storage of those numbers alone cost up to 1.7 log-units of
#: spread across a posterior on a TESS 20-s sector (docs/float32.md,
#: section 7).  Here every term is one O(1) innovation, and the ~1e4
#: numbers never exist.  The step is `_FWD_STEP`, the backward's own, so
#: the value is the function the backward differentiates.
_SRC_REPLAY = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    uint start = blk_i * (uint)blk_len;
    uint stop = min(start + (uint)blk_len, (uint)npts);
    uint ei = chain * (uint)n_blocks + blk_i;

    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    JIT_HOIST

    float mean[MDIM], cov[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) {
        mean[i] = ent_m[ei * M + i];
        for (uint j = 0; j < M; ++j) cov[i][j] = ent_C[ei * M * M + i * M + j];
    }
    float ell = 0.0f, ell_c = 0.0f;          // Kahan accumulator
    for (uint gi = start; gi < stop; ++gi) {
        FWD_STEP
    }
    out_ell[ei] = ell;
"""

_replay_kernels: dict = {}


def _replay_kernel(m: int, n_jit: int = 0, formal: bool = False):
    key = (m, _jit_key(n_jit), bool(formal))
    if key not in _replay_kernels:
        body = _formal_src(_jitter_src(
            _SRC_REPLAY.replace("FWD_STEP", _fwd_step(True)), n_jit), formal)
        _replay_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_replay_m{m}_j{key[1]}"
                 + ("_formal" if formal else ""),
            input_names=["phitab", "qtab", "lag", "inst", "hvec", "resid",
                         "r_ref", "jit2", "ent_m", "ent_C",
                         "npts", "n_lags", "n_blocks", "n_chains", "blk_len",
                         "n_jit"],
            output_names=["out_ell"],
            source=body.replace("MDIM", str(m)),
        )
    return _replay_kernels[key]


def replay_blocks(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2,
                  ent_m, ent_C, blk_len, formal=False) -> mx.array:
    """``(n_chains, n_blocks)`` log-likelihood of each block, by the filter
    replayed from its entry state ``(ent_m, ent_C)`` (:func:`entry_states`).
    Inputs otherwise as :func:`scan_blocks`."""
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts = resid.shape[0]
    n_blocks = ent_m.shape[1]
    n_jit = _n_jit_of(jit2)
    outs = _replay_kernel(m, n_jit, formal)(
        inputs=[phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2,
                ent_m, ent_C, int(npts), int(n_lags), int(n_blocks),
                int(n_chains), int(blk_len), int(n_jit)],
        output_shapes=[(n_chains, n_blocks)], output_dtypes=[mx.float32],
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return outs[0]



#: The multi-column replay: `_SRC_REPLAY` for ``KCOL`` residual columns at
#: once, one thread per (chain, block).  The covariance recursion is the
#: same for every column, so it runs once and only the means are per
#: column; the thread emits the block's log-determinant half and the packed
#: lower triangle of its innovation Gram matrix (`_FWD_MC`).  Summed over
#: blocks these are ``-1/2 log|K|`` (relative to ``r_ref``) and ``Z^T K^-1
#: Z`` for the columns ``Z`` -- what a linear marginalisation needs, for
#: the cost of one filter plus ``KCOL`` mean recursions instead of the
#: ``1 + k(k+1)/2`` filters of polarisation.
_SRC_REPLAY_MC = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    uint start = blk_i * (uint)blk_len;
    uint stop = min(start + (uint)blk_len, (uint)npts);
    uint ei = chain * (uint)n_blocks + blk_i;

    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    JIT_HOIST

    float mean[KCOL][MDIM], cov[MDIM][MDIM];
    for (uint c = 0; c < KCOL; ++c)
        for (uint i = 0; i < M; ++i)
            mean[c][i] = ent_m[(ei * KCOL + c) * M + i];
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) cov[i][j] = ent_C[ei * M * M + i * M + j];
    float ell = 0.0f, ell_c = 0.0f;          // Kahan accumulators
    float gram[KTRI], gram_c[KTRI];
    for (uint t = 0; t < KTRI; ++t) { gram[t] = 0.0f; gram_c[t] = 0.0f; }
    for (uint gi = start; gi < stop; ++gi) {
        FWD_STEP
    }
    out_ld[ei] = ell;
    for (uint t = 0; t < KTRI; ++t) out_g[ei * KTRI + t] = gram[t];
"""

_replay_mc_kernels: dict = {}


def _replay_mc_kernel(m: int, k: int, n_jit: int = 0, formal: bool = False):
    key = (m, k, _jit_key(n_jit), bool(formal))
    if key not in _replay_mc_kernels:
        body = _formal_src(_jitter_src(
            _SRC_REPLAY_MC.replace("FWD_STEP", _fwd_step(True, multi=True)),
            n_jit),
            formal)
        _replay_mc_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_replay_mc_m{m}_k{k}_j{key[2]}"
                 + ("_formal" if formal else ""),
            input_names=["phitab", "qtab", "lag", "inst", "hvec", "resid",
                         "r_ref", "jit2", "ent_m", "ent_C",
                         "npts", "n_lags", "n_blocks", "n_chains", "blk_len",
                         "n_jit"],
            output_names=["out_ld", "out_g"],
            source=body.replace("KTRI", str(k * (k + 1) // 2))
                       .replace("KCOL", str(k)).replace("MDIM", str(m)),
        )
    return _replay_mc_kernels[key]


def replay_blocks_mc(phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2,
                     ent_m, ent_C, blk_len, formal=False):
    """``(ld, gram)``, each per (chain, block): the log-determinant half and
    the packed lower-triangular innovation Gram matrix of ``cols``
    ``(npts, n_chains, k)``, replayed from per-column entry means ``ent_m``
    ``(n_chains, n_blocks, k, m)`` and the shared entry covariance
    ``ent_C`` ``(n_chains, n_blocks, m, m)``.  Inputs otherwise as
    :func:`replay_blocks`."""
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts, k = cols.shape[0], cols.shape[-1]
    n_blocks = ent_m.shape[1]
    n_jit = _n_jit_of(jit2)
    ktri = k * (k + 1) // 2
    outs = _replay_mc_kernel(m, k, n_jit, formal)(
        inputs=[phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2,
                ent_m, ent_C, int(npts), int(n_lags), int(n_blocks),
                int(n_chains), int(blk_len), int(n_jit)],
        output_shapes=[(n_chains, n_blocks), (n_chains, n_blocks, ktri)],
        output_dtypes=[mx.float32] * 2,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return outs[0], outs[1]


#: The multi-column boundary's middle pass: each block's mean map from its
#: shared entry covariance, one thread per (chain, block).  The block's
#: exit mean for column ``c`` entering at ``m`` is ``T m + d[c]``, and both
#: come from the replay's own step (`_FWD_STEP`, gains and all) started at
#: zero mean, so `_SRC_MEAN_PREFIX` composes them into every column's entry
#: mean with no inverse anywhere -- the means are linear in the data, and
#: their covariance is not a function of it.
_SRC_AFFINE = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    uint start = blk_i * (uint)blk_len;
    uint stop = min(start + (uint)blk_len, (uint)npts);
    uint ei = chain * (uint)n_blocks + blk_i;

    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    JIT_HOIST

    float mean[KCOL][MDIM], cov[MDIM][MDIM], tm[MDIM][MDIM];
    for (uint c = 0; c < KCOL; ++c)
        for (uint i = 0; i < M; ++i) mean[c][i] = 0.0f;
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) {
            cov[i][j] = ent_C[ei * M * M + i * M + j];
            tm[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    for (uint gi = start; gi < stop; ++gi) {
        FWD_STEP
    }
    for (uint c = 0; c < KCOL; ++c)
        for (uint i = 0; i < M; ++i)
            out_d[(ei * KCOL + c) * M + i] = mean[c][i];
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) out_T[ei * M * M + i * M + j] = tm[i][j];
"""

#: Every column's entry mean from the blocks' mean maps: block 0 enters at
#: zero, block b + 1 at ``T_b m_b + d_b``.  One thread per (chain, column),
#: sequential over the blocks.
_SRC_MEAN_PREFIX = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_chains * KCOL) return;
    uint chain = gid / KCOL;
    uint c = gid % KCOL;
    const uint M = MDIM;
    float m[MDIM];
    for (uint i = 0; i < M; ++i) m[i] = 0.0f;
    for (uint b = 0; b < (uint)n_blocks; ++b) {
        uint ei = chain * (uint)n_blocks + b;
        for (uint i = 0; i < M; ++i) ent_m[(ei * KCOL + c) * M + i] = m[i];
        float nm[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = in_d[(ei * KCOL + c) * M + i];
            for (uint j = 0; j < M; ++j) a += in_T[ei * M * M + i * M + j] * m[j];
            nm[i] = a;
        }
        for (uint i = 0; i < M; ++i) m[i] = nm[i];
    }
"""

_affine_kernels: dict = {}
_mean_prefix_kernels: dict = {}


def _affine_kernel(m: int, k: int, n_jit: int = 0, formal: bool = False):
    key = (m, k, _jit_key(n_jit), bool(formal))
    if key not in _affine_kernels:
        body = _formal_src(_jitter_src(
            _SRC_AFFINE.replace("FWD_STEP", _fwd_step(affine=True)), n_jit),
            formal)
        _affine_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_affine_m{m}_k{k}_j{key[2]}"
                 + ("_formal" if formal else ""),
            input_names=["phitab", "qtab", "lag", "inst", "hvec", "resid",
                         "r_ref", "jit2", "ent_C", "npts", "n_lags",
                         "n_blocks", "n_chains", "blk_len", "n_jit"],
            output_names=["out_d", "out_T"],
            source=body.replace("KCOL", str(k)).replace("MDIM", str(m)),
        )
    return _affine_kernels[key]


def column_entry_means(phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2,
                       ent_C, blk_len, formal=False):
    """``(n_chains, n_blocks, k, m)``: every column's entry mean, given the
    blocks' shared entry covariances ``ent_C`` -- the affine pass
    (`_SRC_AFFINE`) and the prefix over its maps (`_SRC_MEAN_PREFIX`)."""
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts, k = cols.shape[0], cols.shape[-1]
    n_blocks = ent_C.shape[1]
    n_jit = _n_jit_of(jit2)
    d, T = _affine_kernel(m, k, n_jit, formal)(
        inputs=[phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2, ent_C,
                int(npts), int(n_lags), int(n_blocks), int(n_chains),
                int(blk_len), int(n_jit)],
        output_shapes=[(n_chains, n_blocks, k, m), (n_chains, n_blocks, m, m)],
        output_dtypes=[mx.float32] * 2,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    key = (m, k)
    if key not in _mean_prefix_kernels:
        _mean_prefix_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_mean_prefix_m{m}_k{k}",
            input_names=["in_d", "in_T", "n_blocks", "n_chains"],
            output_names=["ent_m"],
            source=_SRC_MEAN_PREFIX.replace("KCOL", str(k))
                                   .replace("MDIM", str(m)),
        )
    return _mean_prefix_kernels[key](
        inputs=[d, T, int(n_blocks), int(n_chains)],
        output_shapes=[(n_chains, n_blocks, k, m)],
        output_dtypes=[mx.float32],
        grid=(n_chains * k, 1, 1),
        threadgroup=(min(64, n_chains * k), 1, 1))[0]


#: The gradient's r-sensitivity pass: `_SRC_AFFINE` from each column's
#: actual entry mean, accumulating ``g[c]`` (`_FWD_AFFINE_TG`) beside the
#: block's mean map ``T``.  `_SRC_R_SUFFIX` turns them into ``r_c`` at every
#: block exit -- what `scan_gram`'s VJP builds the block seeds from.
_SRC_RSENS = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    uint start = blk_i * (uint)blk_len;
    uint stop = min(start + (uint)blk_len, (uint)npts);
    uint ei = chain * (uint)n_blocks + blk_i;

    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    JIT_HOIST

    float mean[KCOL][MDIM], cov[MDIM][MDIM], tm[MDIM][MDIM], gacc[KCOL][MDIM];
    for (uint c = 0; c < KCOL; ++c)
        for (uint i = 0; i < M; ++i) {
            mean[c][i] = ent_m[(ei * KCOL + c) * M + i];
            gacc[c][i] = 0.0f;
        }
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) {
            cov[i][j] = ent_C[ei * M * M + i * M + j];
            tm[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    for (uint gi = start; gi < stop; ++gi) {
        FWD_STEP
    }
    for (uint c = 0; c < KCOL; ++c)
        for (uint i = 0; i < M; ++i)
            out_g[(ei * KCOL + c) * M + i] = gacc[c][i];
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) out_T[ei * M * M + i * M + j] = tm[i][j];
"""

#: ``r_c`` at each block's exit from the later blocks' ``(g, T)``: zero at
#: the last block, ``r(b) = g(b + 1) + T(b + 1)^T r(b + 1)``.  A linear
#: suffix of the actual gradient contributions -- no ``eta_s - J_s mu``
#: cancellation, which `block_seeds_kernel`'s closed form has.  One thread
#: per (chain, column).
_SRC_R_SUFFIX = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_chains * KCOL) return;
    uint chain = gid / KCOL;
    uint c = gid % KCOL;
    const uint M = MDIM;
    float r[MDIM];
    for (uint i = 0; i < M; ++i) r[i] = 0.0f;
    for (int bi = (int)n_blocks - 1; bi >= 0; --bi) {
        uint ei = chain * (uint)n_blocks + (uint)bi;
        for (uint i = 0; i < M; ++i) out_r[(ei * KCOL + c) * M + i] = r[i];
        float nr[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = in_g[(ei * KCOL + c) * M + i];
            for (uint j = 0; j < M; ++j) a += in_T[ei * M * M + j * M + i] * r[j];
            nr[i] = a;
        }
        for (uint i = 0; i < M; ++i) r[i] = nr[i];
    }
"""

_rsens_kernels: dict = {}
_r_suffix_kernels: dict = {}


def column_exit_sensitivities(phi_tab, q_tab, lag, inst, hvec, cols, r_ref,
                              jit2, ent_m, ent_C, blk_len, formal=False):
    """``(n_chains, n_blocks, k, m)``: ``r_c`` at every block's exit, ``-1/2``
    the derivative of the later blocks' ``sum v_c^2 / S`` with respect to
    the exit mean (`_SRC_RSENS`, `_SRC_R_SUFFIX`).  Zero at the last
    block."""
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts, k = cols.shape[0], cols.shape[-1]
    n_blocks = ent_C.shape[1]
    n_jit = _n_jit_of(jit2)
    key = (m, k, _jit_key(n_jit), bool(formal))
    if key not in _rsens_kernels:
        body = _formal_src(_jitter_src(
            _SRC_RSENS.replace("FWD_STEP", _fwd_step(sens=True)), n_jit),
            formal)
        _rsens_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_rsens_m{m}_k{k}_j{key[2]}"
                 + ("_formal" if formal else ""),
            input_names=["phitab", "qtab", "lag", "inst", "hvec", "resid",
                         "r_ref", "jit2", "ent_m", "ent_C", "npts", "n_lags",
                         "n_blocks", "n_chains", "blk_len", "n_jit"],
            output_names=["out_g", "out_T"],
            source=body.replace("KCOL", str(k)).replace("MDIM", str(m)),
        )
    g, T = _rsens_kernels[key](
        inputs=[phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2, ent_m,
                ent_C, int(npts), int(n_lags), int(n_blocks), int(n_chains),
                int(blk_len), int(n_jit)],
        output_shapes=[(n_chains, n_blocks, k, m), (n_chains, n_blocks, m, m)],
        output_dtypes=[mx.float32] * 2,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    if (m, k) not in _r_suffix_kernels:
        _r_suffix_kernels[(m, k)] = mx.fast.metal_kernel(
            name=f"anvilgp_r_suffix_m{m}_k{k}",
            input_names=["in_g", "in_T", "n_blocks", "n_chains"],
            output_names=["out_r"],
            source=_SRC_R_SUFFIX.replace("KCOL", str(k))
                                .replace("MDIM", str(m)))
    return _r_suffix_kernels[(m, k)](
        inputs=[g, T, int(n_blocks), int(n_chains)],
        output_shapes=[(n_chains, n_blocks, k, m)],
        output_dtypes=[mx.float32],
        grid=(n_chains * k, 1, 1),
        threadgroup=(min(64, n_chains * k), 1, 1))[0]


def _boundary_seeds(aux, cot, formal, use_sqrt):
    """``(ent_m, ent_C, seed_m, seed_C)`` from `_boundary_states`' ``aux``.
    The entry states are the forward's again: the same kernel source on the
    same inputs."""
    if not use_sqrt:
        A, b, C, eta, J = aux
        return block_seeds_kernel(Element(A=A, b=b, C=C, eta=eta, J=J,
                                          ell=None), cot, formal)
    *blocks, st_m, st_C = aux
    return sqrt_scan(blocks, st_m, st_C, cot)


def _check_adjoint_budget(n_chains, n_slots, n_lags, npts, m, fold_plan):
    """The kernel gradient's buffer fence.

    The adjoint buffer is one slot per (block, lag) pair that occurs.  It
    is bounded by N however irregular the cadence, so this fence is a
    sanity check rather than the wall it was when the buffer had a slot per
    pair that *could* occur (40 GiB at 1024 chains x N = 1e5, where the
    slots are 3 GiB).  The dense fold gathers (n_chains, n_lags, max_rank,
    m, m) on top of the buffer, and `_lag_fold_plan` only bounds that table
    at 4x the slots -- so a configuration comfortably inside the fence
    could still OOM in the fold.  The gather is charged to the same
    budget.  (The groups branch has no such intermediate.)
    """
    slab = 2 * n_chains * n_slots * m * m * 4
    kind, plan = fold_plan
    if kind == "dense":
        slab += 2 * n_chains * int(plan.shape[0]) * int(plan.shape[1]) \
            * m * m * 4
    if slab > MAX_ADJOINT_BUFFER_BYTES:
        raise ValueError(
            f"the kernel gradient's adjoint buffer would be "
            f"{slab / 2**30:.1f} GiB ({n_chains} chains x {n_slots} "
            f"(block, lag) slots x m^2 = {m * m}, plus the fold's "
            f"gather), over MAX_ADJOINT_BUFFER_BYTES = "
            f"{MAX_ADJOINT_BUFFER_BYTES / 2**30:.1f} GiB. "
            f"Every distinct float32 spacing in the timestamps is a "
            f"lag, and this cadence has {n_lags - 1} of them over "
            f"{npts} points.  Bin the timestamps, use fewer chains, or "
            f"use GPPolicy(metal=False) for the gradient.")


def _fold_slots(d_phi, d_q, fold_plan, n_chains, n_lags, m):
    """Slot adjoints -> the per-lag tables the term front end
    differentiates.

    Distinct blocks own disjoint slots, so the blocks are already summed
    and this only regroups.  Applied one fold group at a time: each group's
    lags are distinct, which is what keeps the scatter reproducible -- see
    `likelihood._lag_fold_plan`.  The all-distinct cadence this work is for
    has exactly one group.  The dense plan is one gather through a
    (n_lags, max_rank) table of slot indices padded with a zero row, then
    one `mx.sum` -- which is reproducible where a repeated-index scatter is
    not.
    """
    kind, plan = fold_plan
    if kind == "dense":
        pad = mx.zeros((n_chains, 1, m, m), dtype=d_phi.dtype)
        return (mx.sum(mx.take(mx.concatenate([d_phi, pad], axis=1), plan,
                               axis=1), axis=2),
                mx.sum(mx.take(mx.concatenate([d_q, pad], axis=1), plan,
                               axis=1), axis=2))
    d_phi_tab = mx.zeros((n_chains, n_lags, m, m), dtype=d_phi.dtype)
    # a separate zero, not an alias: `.at[].add` is functional so sharing
    # one would be correct today, but an empty plan would then return the
    # same array as two independent cotangents.
    d_q_tab = mx.zeros((n_chains, n_lags, m, m), dtype=d_q.dtype)
    for sel, lags_of in plan:
        d_phi_tab = d_phi_tab.at[:, lags_of].add(mx.take(d_phi, sel, axis=1))
        d_q_tab = d_q_tab.at[:, lags_of].add(mx.take(d_q, sel, axis=1))
    return d_phi_tab, d_q_tab


def _d_jit2(d_rv, j2):
    """``jit2``'s cotangent.  ``S = max(h Cp h^T, 0) + r_ref + jit2``, so
    d/d(jit2) is the sum of the r_var cotangents, which the kernel
    accumulates per block: ``d_rv`` is (n_chains, n_blocks, n_jit), split
    by the observation's instrument, so summing over blocks gives each
    jitter parameter its own gradient.  Per datum (`noise_fn`) the kernel
    wrote each observation's own adjoint, already in ``j2``'s (npts,
    n_chains, 1) shape."""
    n_jit = _n_jit_of(j2)
    return (d_rv if n_jit < 0 else mx.zeros_like(j2) if n_jit == 0
            else mx.sum(d_rv, axis=1)[:, :n_jit])


def scan_gram(phi_tab, q_tab, lag, inst, hvec, cols, r_ref, jit2,
              s_floor_rel, blk_len, slots=None, formal=False,
              boundary="dense"):
    """``(ld (n_chains,), gram (n_chains, k, k))``: the log-determinant half
    ``-1/2 sum log(S / r_ref)`` and ``Z^T K^-1 Z`` for the columns ``Z`` =
    ``cols`` ``(npts, n_chains, k)``, by one filter carrying ``k`` means.

    ``ld + -1/2 gram[c, c]`` is :func:`scan_loglike` of column ``c``.  The
    entry covariances come from `scan_loglike`'s own boundary, run once, on
    a zero residual (they are not a function of the data), and every
    column's entry means from :func:`column_entry_means`.  Running the
    boundary with the columns as ``k`` virtual chains instead measured
    k-fold: 8.7 and 16.1 ms at k = 3 and 6 against 3.0 for one (N = 1e4 x
    1024 chains, SHO), and the boundary was then most of the cost.

    Differentiable, like `scan_loglike` (``slots`` as there).  The VJP is
    `backward_ckpt_mc` from per-block seeds that generalise the single
    column's closed form (docs/m4-plan.md): for ``c_ld * ld + sum_ab
    gs_ab G_ab``,

        seed_m[a] = -2 (gs R)_a,   seed_C = -1/2 c_ld Jt - R gs R^T,

    with ``r_c`` (the columns of ``R``) from
    :func:`column_exit_sensitivities` and ``-1/2 c_ld Jt`` the existing
    seed of the zero residual at cotangent ``c_ld``.  One column, ``c_ld =
    cot`` and ``gs = -cot / 2``, is the single-column seed exactly.
    """
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts, k = cols.shape[0], cols.shape[-1]
    if m > MAX_STATE:
        raise ValueError(f"metal scan supports m <= {MAX_STATE}, got {m}")
    n_blocks = -(-npts // blk_len)
    use_sqrt = boundary == "sqrt" and n_blocks > 1
    rows, cs = np.tril_indices(k)
    lower, upper = mx.array(rows * k + cs), mx.array(cs * k + rows)
    off_diag = mx.array(rows != cs)

    @mx.custom_function
    def core(p_t, q_t, hv, cz, j2):
        cz = mx.contiguous(cz)
        if n_blocks == 1:
            # Block 0 enters at zero (its first lag is row 0, Phi = 0), as
            # in `_SRC_PREFIX`; no boundary to run.
            ent_m = mx.zeros((n_chains, 1, k, m), dtype=mx.float32)
            ent_C = mx.zeros((n_chains, 1, m, m), dtype=mx.float32)
            aux = ()
        else:
            zero = mx.zeros((npts, n_chains), dtype=mx.float32)
            _, ent_C, aux = _boundary_states(
                p_t, q_t, lag, inst, hv, zero, r_ref, j2, s_floor_rel,
                blk_len, formal, use_sqrt)
            ent_m = column_entry_means(p_t, q_t, lag, inst, hv, cz, r_ref,
                                       j2, ent_C, blk_len, formal)
        ent_m, ent_C = mx.contiguous(ent_m), mx.contiguous(ent_C)
        ld, g = replay_blocks_mc(p_t, q_t, lag, inst, hv, cz, r_ref, j2,
                                 ent_m, ent_C, blk_len, formal)
        ld, g = mx.sum(ld, axis=1), mx.sum(g, axis=1)
        full = mx.zeros((n_chains, k * k), dtype=g.dtype)
        full = full.at[:, lower].add(g)
        full = full.at[:, upper].add(mx.where(off_diag, g, 0.0))
        # entry states and the boundary's aux go out only for the VJP
        return (ld, full.reshape(n_chains, k, k), ent_m, ent_C, *aux)

    @core.vjp
    def _core_vjp(primals, cotangents, outputs):
        p_t, q_t, hv, cz, j2 = primals
        c_ld = cotangents[0]
        gs = 0.5 * (cotangents[1] + mx.swapaxes(cotangents[1], 1, 2))
        ent_m, ent_C, aux = outputs[2], outputs[3], tuple(outputs[4:])
        if slots is None:
            raise ValueError(
                "the kernel gradient needs the (block, lag) slot map; "
                "`scan_gram` was called without `slots`")
        slot_id, slot_start, n_slots, fold_plan = slots
        _check_adjoint_budget(n_chains, n_slots, n_lags, npts, m, fold_plan)
        if n_blocks == 1:
            seed_m = mx.zeros((n_chains, 1, k, m), dtype=mx.float32)
            seed_C = mx.zeros((n_chains, 1, m, m), dtype=mx.float32)
        else:
            *_, seed_ld = _boundary_seeds(aux, c_ld, formal, use_sqrt)
            R = column_exit_sensitivities(p_t, q_t, lag, inst, hv,
                                          mx.contiguous(cz), r_ref, j2,
                                          ent_m, ent_C, blk_len, formal)
            seed_m = -2.0 * mx.einsum("cab,cnbi->cnai", gs, R)
            seed_C = seed_ld - mx.einsum("cnai,cab,cnbj->cnij", R, gs, R)
        d_phi, d_q, d_h, d_y, d_rv = backward_ckpt_mc(
            p_t, q_t, lag, inst, hv, mx.contiguous(cz), r_ref, j2, c_ld, gs,
            ent_m, ent_C, seed_m, seed_C, slot_id, slot_start, n_slots,
            blk_len, formal)
        d_phi_tab, d_q_tab = _fold_slots(d_phi, d_q, fold_plan, n_chains,
                                         n_lags, m)
        return (d_phi_tab, d_q_tab, mx.sum(d_h, axis=1), d_y,
                _d_jit2(d_rv, j2))

    out = core(phi_tab, q_tab, hvec, cols, jit2)
    return out[0], out[1]




#: Weighted normal equations ``A = Phi^T W Phi`` (packed lower triangle)
#: and ``b = Phi^T W r`` per chain, one threadgroup per chain striding over
#: the data, reduced across SIMD groups in threadgroup memory.  For the
#: marginalisation's white-noise centring fit: MLX's batched matmul for
#: this shape (a p x p result over a long N) ran at ~5 GB/s, 3.9 ms at
#: N = 1e4 x 256 chains and 38 ms at 1e5 -- more than the likelihood.
#: The basis is read as ``(chain, column, datum)``, so adjacent threads read
#: adjacent floats; ``(chain, datum, column)`` measured 6x slower at p = 10.
#: ``phi_stride`` is 0 for a basis shared by every chain.
_SRC_NORMAL_EQ = """
    uint chain = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    const uint P = PCOL;
    float acc[NACC];
    for (uint i = 0; i < NACC; ++i) acc[i] = 0.0f;
    uint pb = chain * (uint)phi_stride;
    uint rb = chain * (uint)npts;
    for (uint t = tid; t < (uint)npts; t += NTHR) {
        float wt = w[t];
        float ph[PCOL];
        for (uint a = 0; a < P; ++a) ph[a] = phi[pb + a * (uint)npts + t];
        float rt = r[rb + t] * wt;
        #pragma unroll
        for (uint a = 0; a < P; ++a) {
            float pa = ph[a] * wt;
            #pragma unroll
            for (uint b = 0; b < P; ++b)
                if (b <= a) acc[a * (a + 1) / 2 + b] += pa * ph[b];
            acc[PTRI + a] += ph[a] * rt;
        }
    }
    threadgroup float shm[(NTHR / 32) * NACC];
    for (uint i = 0; i < NACC; ++i) {
        float v = simd_sum(acc[i]);
        if (lane == 0) shm[sg * NACC + i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < NACC; i += NTHR) {
        float v = 0.0f;
        for (uint g = 0; g < NTHR / 32; ++g) v += shm[g * NACC + i];
        out[chain * NACC + i] = v;
    }
"""

_normal_eq_kernels: dict = {}
_NORMAL_EQ_THREADS = 128
#: threadgroup memory: (threads / 32) * (p(p+1)/2 + p) floats, under 32 KiB
MAX_NORMAL_EQ_COLS = 40


def normal_equations(phi_t, r, w):
    """``(A (n, p, p), b (n, p))`` with ``A = Phi^T diag(w) Phi`` and
    ``b = Phi^T diag(w) r``, float32, for the basis **transposed**,
    ``phi_t`` ``(n | 1, p, N)``, ``r`` ``(n, N)`` and ``w`` ``(N,)``.  No
    VJP: for quantities under ``stop_gradient``."""
    n, N = r.shape
    p = phi_t.shape[1]
    if p > MAX_NORMAL_EQ_COLS:
        raise ValueError(f"normal_equations supports p <= "
                         f"{MAX_NORMAL_EQ_COLS}, got {p}")
    ptri = p * (p + 1) // 2
    nacc = ptri + p
    key = p
    if key not in _normal_eq_kernels:
        _normal_eq_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_normal_eq_p{p}",
            input_names=["phi", "r", "w", "npts", "phi_stride"],
            output_names=["out"],
            source=_SRC_NORMAL_EQ.replace("NACC", str(nacc))
                                 .replace("PTRI", str(ptri))
                                 .replace("PCOL", str(p))
                                 .replace("NTHR", str(_NORMAL_EQ_THREADS)),
        )
    shared = phi_t.shape[0] == 1 and n > 1
    phi_t = phi_t[0] if shared else phi_t
    out = _normal_eq_kernels[key](
        inputs=[phi_t.astype(mx.float32), r.astype(mx.float32),
                w.astype(mx.float32), int(N), 0 if shared else int(N * p)],
        output_shapes=[(n, nacc)], output_dtypes=[mx.float32],
        grid=(n * _NORMAL_EQ_THREADS, 1, 1),
        threadgroup=(_NORMAL_EQ_THREADS, 1, 1))[0]
    rows, cs = np.tril_indices(p)
    A = mx.zeros((n, p * p), dtype=mx.float32)
    A = A.at[:, mx.array(rows * p + cs)].add(out[:, :ptri])
    A = A.at[:, mx.array(cs * p + rows)].add(
        mx.where(mx.array(rows != cs), out[:, :ptri], 0.0))
    return A.reshape(n, p, p), out[:, ptri:]



#: ``(x^-1, log|det x|)`` for a batch of small general matrices, one thread
#: per matrix: `linalg.inv_logdet`'s Gauss-Jordan with partial pivoting, in
#: registers.  The MLX form is ``p^3`` separate elementwise launches -- 12
#: ms for 1024 10 x 10 matrices, 4 ms compiled -- where this is one.  A
#: singular matrix gives an infinite pivot inverse and non-finite output,
#: as the MLX form does.
_SRC_SMALL_INV = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_mat) return;
    const uint P = PDIM;
    float a[PDIM][PDIM], g[PDIM][PDIM];
    for (uint i = 0; i < P; ++i)
        for (uint j = 0; j < P; ++j) {
            a[i][j] = x[(gid * P + i) * P + j];
            g[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    float logdet = 0.0f;
    for (uint k = 0; k < P; ++k) {
        uint best = k;
        float bval = fabs(a[k][k]);
        for (uint r = k + 1; r < P; ++r)
            if (fabs(a[r][k]) > bval) { best = r; bval = fabs(a[r][k]); }
        if (best != k)
            for (uint j = 0; j < P; ++j) {
                float t = a[k][j]; a[k][j] = a[best][j]; a[best][j] = t;
                t = g[k][j]; g[k][j] = g[best][j]; g[best][j] = t;
            }
        float piv = a[k][k];
        logdet += metal::precise::log(fabs(piv));
        float inv = 1.0f / piv;
        for (uint j = 0; j < P; ++j) { a[k][j] *= inv; g[k][j] *= inv; }
        for (uint r = 0; r < P; ++r) {
            if (r == k) continue;
            float f = a[r][k];
            for (uint j = 0; j < P; ++j) {
                a[r][j] -= f * a[k][j];
                g[r][j] -= f * g[k][j];
            }
        }
    }
    for (uint i = 0; i < P; ++i)
        for (uint j = 0; j < P; ++j) out_inv[(gid * P + i) * P + j] = g[i][j];
    out_ld[gid] = logdet;
"""

_small_inv_kernels: dict = {}
#: Largest matrix `small_inv_logdet` takes (two p x p arrays per thread).
MAX_SMALL_INV = 32


def small_inv_logdet(x: mx.array):
    """``(x^-1, log|det x|)`` for float32 ``x`` ``(..., p, p)``, ``p <=
    MAX_SMALL_INV``, one thread per matrix.  No VJP: see
    `linalg.inv_logdet_fast`."""
    p = x.shape[-1]
    batch = x.shape[:-2]
    n = 1
    for d in batch:
        n *= int(d)
    if p not in _small_inv_kernels:
        _small_inv_kernels[p] = mx.fast.metal_kernel(
            name=f"anvilgp_small_inv_p{p}",
            input_names=["x", "n_mat"], output_names=["out_inv", "out_ld"],
            source=_SRC_SMALL_INV.replace("PDIM", str(p)))
    inv, ld = _small_inv_kernels[p](
        inputs=[x.reshape(n, p, p).astype(mx.float32), int(n)],
        output_shapes=[(n, p, p), (n,)], output_dtypes=[mx.float32] * 2,
        grid=(max(n, 1), 1, 1), threadgroup=(min(64, max(n, 1)), 1, 1))
    return inv.reshape(*batch, p, p), ld.reshape(batch)


#: The square-root boundary's process-noise factor: ``L`` with ``L L^T =
#: Q`` for every entry of the lag table, one thread per (chain, lag).
#: docs/boundary-plan.md, stage 1.
#:
#: Each pivot is clamped at zero and there is **no floor**, unlike the
#: composition kernels' ``eps * trace / m`` (`_COMBINE_CHOL`).  At 20-s
#: cadence ``Q``'s eigenvalues run from 2.5e-2 down to 1.7e-9, under that
#: floor, so the floor rewrote exactly the directions a high-s/e filter
#: depends on and cost the boundary seeds 20-50x; the clamp matches a
#: float64 eigendecomposition at 2-s, 20-s and 2-min cadences.  A pivot
#: that rounds to zero or below zeroes its column, which is the right
#: factor of a rank-deficient ``Q`` (row 0's ``P_inf`` and a zero lag
#: included).  Only for a positive semi-definite ``Q``: a convolved
#: term's is indefinite and keeps the dense boundary.
_SRC_QFACTOR = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_entries) return;
    const uint M = MDIM;
    uint off = gid * M * M;
    float L[MDIM][MDIM];
    for (uint j = 0; j < M; ++j) {
        float d = qtab[off + j * M + j];
        for (uint t = 0; t < j; ++t) d -= L[j][t] * L[j][t];
        float piv = metal::precise::sqrt(max(d, 0.0f));
        L[j][j] = piv;
        float inv = piv > 0.0f ? 1.0f / piv : 0.0f;
        for (uint i = j + 1; i < M; ++i) {
            float a = 0.5f * (qtab[off + i * M + j] + qtab[off + j * M + i]);
            for (uint t = 0; t < j; ++t) a -= L[i][t] * L[j][t];
            L[i][j] = a * inv;
        }
        for (uint i = 0; i < j; ++i) L[i][j] = 0.0f;
    }
    for (uint i = 0; i < M; ++i)
        for (uint j = 0; j < M; ++j) out_L[off + i * M + j] = L[i][j];
"""

_qfactor_kernels: dict = {}


def q_factor(q_tab: mx.array) -> mx.array:
    """Lower-triangular ``L`` with ``L L^T = Q`` for every lag-table entry.

    ``q_tab``: ``(n_chains, n_lags, m, m)``, as :func:`scan_blocks` takes
    it.  Not differentiated: the boundary quantities it feeds only fix
    where each block's replay and backward start (see `_SRC_QFACTOR`).
    """
    m = q_tab.shape[-1]
    if m > MAX_STATE:
        raise ValueError(f"metal scan supports m <= {MAX_STATE}, got {m}")
    if m not in _qfactor_kernels:
        _qfactor_kernels[m] = mx.fast.metal_kernel(
            name=f"anvilgp_qfactor_m{m}", input_names=["qtab", "n_entries"],
            output_names=["out_L"], source=_SRC_QFACTOR.replace("MDIM", str(m)))
    n_entries = q_tab.shape[0] * q_tab.shape[1]
    return _qfactor_kernels[m](
        inputs=[q_tab, int(n_entries)], output_shapes=[q_tab.shape],
        output_dtypes=[mx.float32], grid=(n_entries, 1, 1),
        threadgroup=(min(64, n_entries), 1, 1))[0]


#: The square-root boundary's block pass: one thread per (chain, block),
#: emitting each block's element in the square-root information form,
#: ``(A, b, U, eta, Z)`` with ``C = U U^T`` and ``J = Z Z^T``, centred on
#: the filter replayed from the block's own start state.
#: docs/boundary-plan.md, stage 2; NumPy reference
#: ``tests/boundary_oracle.py::sqrt_blocks``.
#:
#: Why: the dense form's block elements carry ``C`` and ``J`` themselves,
#: whose conditioning float32 cannot hold on bright, densely sampled data,
#: and ``b``, ``eta`` relative to a zero state, which ``eta - J b``
#: cancels.  Factors carry the square root of the condition number;
#: centring makes ``b``, ``eta`` corrections to a trajectory that is
#: already nearly right.  Each alone fails (docs/float32.md, section 7).
#:
#: Per observation the thread advances the replay (`_FWD_STEP`, giving
#: the trajectory ``m_t``) and builds the centred leaf from ``L_Q``
#: (`q_factor`): with ``u = L_Q^T h``, ``S = |u|^2 + r``,
#:
#:     residual  y' = y - h Phi m_{t-1}
#:     A2 = Phi - K h Phi,   K = L_Q u / S
#:     b2 = K y' + Phi m_{t-1} - m_t
#:     U2 = L_Q - c2 (L_Q u) u^T,   c2 = 1 / (sqrt S (sqrt S + sqrt r))
#:     J2 = z z^T, eta2 = g z,   z = (h Phi)^T / sqrt S,  g = y' / sqrt S
#:
#: (``U2 U2^T = Q - Q h h^T Q / S``; ``c2`` is the symmetric square root
#: of ``I - u u^T / S`` written without cancellation.)  The leaf's
#: information is rank one, so composing it onto the running element
#: needs no general inverse: with ``p = U^T z``, ``alpha = 1 + |p|^2``,
#: ``beta = g - z^T b`` and ``a = A^T z``,
#:
#:     A <- A2 (A - (U p) a^T / alpha)
#:     b <- A2 (b + (U p) beta / alpha) + b2
#:     U <- tria([A2 (U - c (U p) p^T), U2]),   c = 1 / (sqrt alpha (sqrt alpha + 1))
#:     eta <- eta + a beta / alpha
#:     Z <- tria([Z, a / sqrt alpha])            (a Givens rank-one update)
#:
#: the same scalars as pass 1's dense fold.  ``tria(X)`` is the lower
#: triangle ``L`` with ``L L^T = X X^T``, by Householder reflections from
#: the right, diagonal made non-negative.  Finally a block's output is
#: moved from its own trajectory end to the next block's start state:
#: ``b += m_end - start_m[k+1]``.
#:
#: Positive semi-definite ``Q`` only; a convolved term keeps the dense
#: boundary (docs/boundary-plan.md, decision 1).
_SRC_SQRT_BLOCKS = """
    uint gid = thread_position_in_grid.x;
    if (gid >= (uint)n_blocks * (uint)n_chains) return;
    uint chain = gid % (uint)n_chains;
    uint blk_i = gid / (uint)n_chains;

    const uint M = MDIM;
    uint start = blk_i * (uint)blk_len;
    if (start >= (uint)npts) return;
    uint stop = min(start + (uint)blk_len, (uint)npts);
    uint ei = chain * (uint)n_blocks + blk_i;

    float hv[MDIM];
    for (uint i = 0; i < M; ++i) hv[i] = hvec[chain * M + i];
    JIT_HOIST

    // the replay, from this block's own start state
    float mean[MDIM], cov[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) {
        mean[i] = start_m[ei * M + i];
        for (uint j = 0; j < M; ++j) cov[i][j] = start_C[ei * M * M + i * M + j];
    }
    // the running element
    float Am[MDIM][MDIM], Um[MDIM][MDIM], Zm[MDIM][MDIM];
    float bv[MDIM], ev[MDIM];
    bool first = true;

    for (uint gi = start; gi < stop; ++gi) {
        float mprev[MDIM];
        for (uint i = 0; i < M; ++i) mprev[i] = mean[i];
        FWD_STEP
        {
            JIT_PER_OBS
            // The leaf's white noise.  A convolved term's exposure
            // correction rides in jit2v and can be negative (Matern-5/2:
            // >= -4.6e-8 k(0) inside the guard); c2 below needs it positive.
            // Floored at 1e-6 r_ref, the innovation floor's default
            // (GPPolicy.s_floor_rel): a `noise_fn` may legitimately go
            // below r_ref (an error scale s < 1), so the floor that was
            // r_ref / 2 until 2026-10-25 would have bound there.  It never
            // binds for a variance >= 1e-6 r_ref, so those are unchanged.
            float rv = max(r_ref[gi] + jit2v, 1.0e-6f * r_ref[gi]);
            uint off = (chain * (uint)n_lags + (uint)lag[gi]) * M * M;
            float Ph[MDIM][MDIM], LQ[MDIM][MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    Ph[i][j] = phitab[off + i * M + j];
                    LQ[i][j] = lqtab[off + i * M + j];
                }
            float pm[MDIM], hphi[MDIM], uu[MDIM], lu[MDIM];
            float yv = resid[gi * (uint)n_chains + chain];
            float su = 0.0f;
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f, c = 0.0f, d = 0.0f;
                for (uint j = 0; j < M; ++j) {
                    a += Ph[i][j] * mprev[j];
                    c += hv[j] * Ph[j][i];
                    d += LQ[j][i] * hv[j];
                }
                pm[i] = a; hphi[i] = c; uu[i] = d;
                su += d * d;
            }
            for (uint i = 0; i < M; ++i) {
                yv -= hv[i] * pm[i];
                float a = 0.0f;
                for (uint j = 0; j < M; ++j) a += LQ[i][j] * uu[j];
                lu[i] = a;
            }
            float sv = su + rv;
            float rs = metal::precise::sqrt(sv);
            float inv_rs = 1.0f / rs;
            float c2 = 1.0f / (rs * (rs + metal::precise::sqrt(rv)));
            float zz[MDIM], kk[MDIM];
            for (uint i = 0; i < M; ++i) {
                zz[i] = hphi[i] * inv_rs;
                kk[i] = lu[i] / sv;
            }
            float gg = yv * inv_rs;
            float A2[MDIM][MDIM], U2[MDIM][MDIM], b2[MDIM];
            for (uint i = 0; i < M; ++i) {
                b2[i] = kk[i] * yv + pm[i] - mean[i];
                for (uint j = 0; j < M; ++j) {
                    A2[i][j] = Ph[i][j] - kk[i] * hphi[j];
                    U2[i][j] = LQ[i][j] - c2 * lu[i] * uu[j];
                }
            }

            if (first) {
                for (uint i = 0; i < M; ++i) {
                    bv[i] = b2[i];
                    ev[i] = gg * zz[i];
                    for (uint j = 0; j < M; ++j) {
                        Am[i][j] = A2[i][j];
                        Um[i][j] = U2[i][j];
                        Zm[i][j] = (j == 0) ? zz[i] : 0.0f;
                    }
                }
                first = false;
            } else {
                // p = U^T z ; Up = U p ; a = A^T z
                float pp[MDIM], up[MDIM], za[MDIM];
                float sp = 0.0f, zb = 0.0f;
                for (uint i = 0; i < M; ++i) {
                    float a = 0.0f, c = 0.0f;
                    for (uint j = 0; j < M; ++j) {
                        a += Um[j][i] * zz[j];
                        c += Am[j][i] * zz[j];
                    }
                    pp[i] = a; za[i] = c;
                    sp += a * a;
                    zb += zz[i] * bv[i];
                }
                for (uint i = 0; i < M; ++i) {
                    float a = 0.0f;
                    for (uint j = 0; j < M; ++j) a += Um[i][j] * pp[j];
                    up[i] = a;
                }
                float alpha = 1.0f + sp;
                float inv_a = 1.0f / alpha;
                float ra = metal::precise::sqrt(alpha);
                float cc = 1.0f / (ra * (ra + 1.0f));
                float bia = (gg - zb) * inv_a;

                // GA = A - Up a^T / alpha ; UR = U - c Up p^T ; v = b + Up beta / alpha
                float GA[MDIM][MDIM], UR[MDIM][MDIM], vv2[MDIM];
                for (uint i = 0; i < M; ++i) {
                    vv2[i] = bv[i] + up[i] * bia;
                    for (uint j = 0; j < M; ++j) {
                        GA[i][j] = Am[i][j] - up[i] * za[j] * inv_a;
                        UR[i][j] = Um[i][j] - cc * up[i] * pp[j];
                    }
                }
                // A <- A2 GA ; b <- A2 v + b2 ; X = [A2 UR, U2]
                float X[MDIM][2 * MDIM];
                for (uint i = 0; i < M; ++i) {
                    float bb = b2[i];
                    for (uint j = 0; j < M; ++j) bb += A2[i][j] * vv2[j];
                    bv[i] = bb;
                    for (uint j = 0; j < M; ++j) {
                        float a = 0.0f, w = 0.0f;
                        for (uint k = 0; k < M; ++k) {
                            a += A2[i][k] * GA[k][j];
                            w += A2[i][k] * UR[k][j];
                        }
                        Am[i][j] = a;
                        X[i][j] = w;
                        X[i][M + j] = U2[i][j];
                    }
                }
                TRIA_X
                for (uint i = 0; i < M; ++i)
                    for (uint j = 0; j < M; ++j) Um[i][j] = X[i][j];

                // eta <- eta + a beta / alpha ; Z <- tria([Z, a / sqrt alpha])
                float vz[MDIM];
                float ira = 1.0f / ra;
                for (uint i = 0; i < M; ++i) {
                    ev[i] += za[i] * bia;
                    vz[i] = za[i] * ira;
                }
                for (uint k = 0; k < M; ++k) {
                    // branch-free: a zero pair rotates by the identity
                    float a = Zm[k][k], bb = vz[k];
                    float rr = metal::precise::sqrt(a * a + bb * bb);
                    bool nz = rr > 0.0f;
                    float ir = nz ? 1.0f / rr : 0.0f;
                    float cg = nz ? a * ir : 1.0f, sg = bb * ir;
                    Zm[k][k] = nz ? rr : a;
                    for (uint i = 0; i < M; ++i) {
                        if (i <= k) continue;
                        float zi = Zm[i][k], vi = vz[i];
                        Zm[i][k] = cg * zi + sg * vi;
                        vz[i] = cg * vi - sg * zi;
                    }
                }
            }
        }
    }

    // output relative to the next block's start state
    if (blk_i + 1 < (uint)n_blocks) {
        uint en = (chain * (uint)n_blocks + blk_i + 1) * M;
        for (uint i = 0; i < M; ++i) bv[i] += mean[i] - start_m[en + i];
    }
    for (uint i = 0; i < M; ++i) {
        out_b[ei * M + i] = bv[i];
        out_eta[ei * M + i] = ev[i];
        for (uint j = 0; j < M; ++j) {
            out_A[ei * M * M + i * M + j] = Am[i][j];
            out_U[ei * M * M + i * M + j] = Um[i][j];
            out_Z[ei * M * M + i * M + j] = Zm[i][j];
        }
    }
"""

def _tria_src(x: str, rows: str, cols: str) -> str:
    """``tria`` in place on the array ``x`` (``rows`` x ``cols``, both MSL
    expressions, ``rows <= cols``): Householder reflections from the right
    zero row k beyond column k, leaving the first ``rows`` columns lower
    triangular, ``L``, with ``L L^T`` the original ``x x^T``; then each
    column is signed so the diagonal is non-negative.  A row already zero
    beyond its diagonal is left alone.  Wrapped in its own scope.

    Every loop has constant bounds and masks the triangle instead, and a
    zero tail selects rather than branches: written with triangular bounds
    and an early ``continue`` this cost 22 of `sqrt_blocks`' 36 ms at
    m = 4 (N = 1e4, 256 chains), and 9 of 22 written this way -- the
    compiler fully unrolls only constant bounds."""
    return f"""
                {{
                for (uint k = 0; k < {rows}; ++k) {{
                    float tail = 0.0f;
                    for (uint j = 0; j < {cols}; ++j)
                        tail += (j > k) ? {x}[k][j] * {x}[k][j] : 0.0f;
                    float x0 = {x}[k][k];
                    float nrm = metal::precise::sqrt(x0 * x0 + tail);
                    float al = (x0 >= 0.0f) ? -nrm : nrm;
                    float vk = x0 - al;
                    float tau = (tail > 0.0f)
                        ? 1.0f / (nrm * (nrm + fabs(x0))) : 0.0f;   // 2 / |v|^2
                    for (uint i = 0; i < {rows}; ++i) {{
                        if (i <= k) continue;
                        float dt = {x}[i][k] * vk;
                        for (uint j = 0; j < {cols}; ++j)
                            dt += (j > k) ? {x}[i][j] * {x}[k][j] : 0.0f;
                        float f = tau * dt;
                        {x}[i][k] -= f * vk;
                        for (uint j = 0; j < {cols}; ++j)
                            if (j > k) {x}[i][j] -= f * {x}[k][j];
                    }}
                    {x}[k][k] = (tail > 0.0f) ? al : x0;
                    for (uint j = 0; j < {cols}; ++j)
                        if (j > k && tail > 0.0f) {x}[k][j] = 0.0f;
                }}
                for (uint k = 0; k < {rows}; ++k) {{
                    float sg = ({x}[k][k] < 0.0f) ? -1.0f : 1.0f;
                    for (uint i = 0; i < {rows}; ++i)
                        if (i >= k) {x}[i][k] *= sg;
                }}
                }}
"""


_TRIA_X = _tria_src("X", "M", "2 * M")

_sqrt_block_kernels: dict = {}


def _sqrt_block_kernel(m: int, n_jit: int = 0):
    key = (m, _jit_key(n_jit))
    if key not in _sqrt_block_kernels:
        body = _SRC_SQRT_BLOCKS.replace("FWD_STEP", _fwd_step(False)) \
                               .replace("TRIA_X", _TRIA_X)
        body = _formal_src(_jitter_src(body, n_jit), False)
        _sqrt_block_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_sqrt_blocks_m{m}_j{key[1]}",
            input_names=["phitab", "qtab", "lqtab", "lag", "inst", "hvec",
                         "resid", "r_ref", "jit2", "start_m", "start_C",
                         "npts", "n_lags", "n_blocks", "n_chains", "blk_len",
                         "n_jit"],
            output_names=["out_A", "out_b", "out_U", "out_eta", "out_Z"],
            source=body.replace("MDIM", str(m)),
        )
    return _sqrt_block_kernels[key]


def sqrt_blocks(phi_tab, q_tab, lq_tab, lag, inst, hvec, resid, r_ref, jit2,
                start_m, start_C, blk_len):
    """``(A, b, U, eta, Z)``, each ``(n_chains, n_blocks, m[, m])``: every
    block's element in the square-root information form, centred on the
    filter replayed from ``(start_m, start_C)`` -- approximate entry states,
    as :func:`entry_states` gives.  ``lq_tab`` is :func:`q_factor` of
    ``q_tab``; the rest as :func:`scan_blocks`.  See `_SRC_SQRT_BLOCKS`.
    Not for a convolved term, whose ``Q`` has no real factor.
    """
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    if m > MAX_STATE:
        raise ValueError(f"metal scan supports m <= {MAX_STATE}, got {m}")
    npts = resid.shape[0]
    n_blocks = start_m.shape[1]
    n_jit = _n_jit_of(jit2)
    shapes = [(n_chains, n_blocks, m, m), (n_chains, n_blocks, m),
              (n_chains, n_blocks, m, m), (n_chains, n_blocks, m),
              (n_chains, n_blocks, m, m)]
    return tuple(_sqrt_block_kernel(m, n_jit)(
        inputs=[phi_tab, q_tab, lq_tab, lag, inst, hvec, resid, r_ref, jit2,
                start_m, start_C, int(npts), int(n_lags), int(n_blocks),
                int(n_chains), int(blk_len), int(n_jit)],
        output_shapes=shapes, output_dtypes=[mx.float32] * 5,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1)))


#: The square-root boundary's scan: one thread per chain, over its block
#: elements from `sqrt_blocks`.  docs/boundary-plan.md, stage 3; NumPy
#: reference ``tests/boundary_oracle.py::sqrt_scan``.
#:
#: A prefix gives each block's entry state, ``start_m[k] + b`` and
#: ``U U^T`` of the composition of blocks 0..k-1 (block k's input is
#: relative to its own start state, which is where `sqrt_blocks` put the
#: previous block's output).  It carries only ``(b, U)``: block 0 has
#: ``A``, ``eta``, ``Z`` exactly zero, and composing onto it keeps them so,
#: as in the dense prefix.  Then a suffix, backward, carries ``(eta, Z)``,
#: and block k's seed is formed when the suffix holds blocks k+1.. --
#: from it and the prefix through block k, the closed form of
#: `adjoint.block_seeds` -- so no suffix is stored.
#:
#: Each composition of ``i`` then ``j`` (Yaghoobi et al. 2022):
#: ``Xi = tria([[U_i^T Z_j, I], [Z_j, 0]])``, 2m x 2m, splits into
#: ``X11``, ``X21``, ``X22``, and
#:
#:     (I + C_i J_j)^-1 v = v - U_i X11^-T X21^T v
#:     (I + J_j C_i)^-1 v = v - X21 X11^-1 U_i^T v
#:     b  = A_j (I + C_i J_j)^-1 (b_i + U_i U_i^T eta_j) + b_j
#:     U  = tria([A_j U_i X11^-T, U_j])
#:     eta = A_i^T (I + J_j C_i)^-1 (eta_j - Z_j Z_j^T b_i) + eta_i
#:     Z  = tria([A_i^T X22, Z_i])
#:
#: ``X11 X11^T = I + U_i^T J_j U_i``, so it has a diagonal >= 1 and its
#: triangular solves are safe.  The seed of a block exiting at ``(mu, U)``
#: into a suffix ``(eta_s, Z_s)``: with ``d = eta_s - Z_s Z_s^T mu`` and
#: ``r = (I + J_s C)^-1 d``, ``m_bar = cot r`` and ``P_bar = cot (r r^T -
#: X22 X22^T) / 2``; ``X22 X22^T = (I + J_s C)^-1 J_s``.
_SQRT_XI = """
            // Xi = tria([[UI^T ZJ, I], [ZJ, 0]]) -> X11, X21, X22
            float XI[2 * MDIM][2 * MDIM];
            for (uint i = 0; i < M; ++i)
                for (uint j = 0; j < M; ++j) {
                    float a = 0.0f;
                    for (uint t = 0; t < M; ++t) a += UI[t][i] * ZJ[t][j];
                    XI[i][j] = a;
                    XI[i][M + j] = (i == j) ? 1.0f : 0.0f;
                    XI[M + i][j] = ZJ[i][j];
                    XI[M + i][M + j] = 0.0f;
                }
            TRIA_XI
"""

#: ``w = X11^-1 v`` (forward) and ``w = X11^-T v`` (backward), in place on
#: ``w`` holding ``v``.  ``X11 = XI[:M, :M]``, lower triangular.
_X11_SOLVE = """
            for (uint i = 0; i < M; ++i) {
                float a = WV[i];
                for (uint t = 0; t < i; ++t) a -= XI[i][t] * WV[t];
                WV[i] = a / XI[i][i];
            }"""
_X11T_SOLVE = """
            for (int ii = (int)M - 1; ii >= 0; --ii) {
                uint i = (uint)ii;
                float a = WV[i];
                for (uint t = i + 1; t < M; ++t) a -= XI[t][i] * WV[t];
                WV[i] = a / XI[i][i];
            }"""

_SRC_SQRT_SCAN = """
    uint chain = thread_position_in_grid.x;
    if (chain >= (uint)n_chains) return;
    const uint M = MDIM;
    uint base = chain * (uint)n_blocks;

    // ---- prefix: (b, U) through blocks 0..k; entry of block k from 0..k-1
    float pb[MDIM], PU[MDIM][MDIM];
    for (uint i = 0; i < M; ++i) {
        ent_m[base * M + i] = start_m[base * M + i];
        pb[i] = in_b[base * M + i];
        pre_b[base * M + i] = pb[i];
        for (uint j = 0; j < M; ++j) {
            ent_C[(base * M + i) * M + j] = start_C[(base * M + i) * M + j];
            PU[i][j] = in_U[(base * M + i) * M + j];
            pre_U[(base * M + i) * M + j] = PU[i][j];
        }
    }
    for (uint k = 1; k < (uint)n_blocks; ++k) {
        uint ov = (base + k) * M, om = (base + k) * M * M;
        for (uint i = 0; i < M; ++i) {
            ent_m[ov + i] = start_m[ov + i] + pb[i];
            for (uint j = 0; j < M; ++j) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += PU[i][t] * PU[j][t];
                ent_C[om + i * M + j] = a;
            }
        }
        float UI[MDIM][MDIM], ZJ[MDIM][MDIM], AJ[MDIM][MDIM], UJ[MDIM][MDIM];
        float bj[MDIM], ej[MDIM];
        for (uint i = 0; i < M; ++i) {
            bj[i] = in_b[ov + i]; ej[i] = in_eta[ov + i];
            for (uint j = 0; j < M; ++j) {
                UI[i][j] = PU[i][j];
                ZJ[i][j] = in_Z[om + i * M + j];
                AJ[i][j] = in_A[om + i * M + j];
                UJ[i][j] = in_U[om + i * M + j];
            }
        }
        SQRT_XI
        // v = b_i + U_i U_i^T eta_j ; g = (I + C_i J_j)^-1 v
        float ue[MDIM], vv[MDIM], WV[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint t = 0; t < M; ++t) a += UI[t][i] * ej[t];
            ue[i] = a;
        }
        for (uint i = 0; i < M; ++i) {
            float a = pb[i];
            for (uint t = 0; t < M; ++t) a += UI[i][t] * ue[t];
            vv[i] = a;
        }
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint t = 0; t < M; ++t) a += XI[M + t][i] * vv[t];   // X21^T v
            WV[i] = a;
        }
        X11T_SOLVE
        for (uint i = 0; i < M; ++i) {
            float a = vv[i];
            for (uint t = 0; t < M; ++t) a -= UI[i][t] * WV[t];
            ue[i] = a;                                                // g
        }
        // b <- A_j g + b_j ;  W = A_j U_i X11^-T, row by row: X11 w^T = (A_j U_i)^T
        float X[MDIM][2 * MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = bj[i];
            for (uint t = 0; t < M; ++t) a += AJ[i][t] * ue[t];
            pb[i] = a;
            for (uint j = 0; j < M; ++j) {
                float c = 0.0f;
                for (uint t = 0; t < M; ++t) c += AJ[i][t] * UI[t][j];
                WV[j] = c;
            }
            X11_SOLVE
            for (uint j = 0; j < M; ++j) { X[i][j] = WV[j]; X[i][M + j] = UJ[i][j]; }
        }
        TRIA_X
        for (uint i = 0; i < M; ++i) {
            pre_b[ov + i] = pb[i];
            for (uint j = 0; j < M; ++j) {
                PU[i][j] = X[i][j];
                pre_U[om + i * M + j] = X[i][j];
            }
        }
    }
//SEEDS{
    // ---- suffix, backward; the seed of block k when it holds blocks k+1..
    float cotv = cot[chain];
    float sZ[MDIM][MDIM], se[MDIM];
    for (uint i = 0; i < M; ++i) { se[i] = 0.0f; for (uint j = 0; j < M; ++j) sZ[i][j] = 0.0f; }
    for (int kk = (int)n_blocks - 1; kk >= 0; --kk) {
        uint k = (uint)kk;
        uint ov = (base + k) * M, om = (base + k) * M * M;
        float UI[MDIM][MDIM], ZJ[MDIM][MDIM], mu[MDIM];
        {
            for (uint i = 0; i < M; ++i) {
                mu[i] = pre_b[ov + i];
                for (uint j = 0; j < M; ++j) { UI[i][j] = pre_U[om + i * M + j]; ZJ[i][j] = sZ[i][j]; }
            }
            SQRT_XI
            // d = eta_s - Z_s Z_s^T mu ; r = d - X21 X11^-1 U^T d
            float zm[MDIM], dv[MDIM], WV[MDIM], rv[MDIM];
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += ZJ[t][i] * mu[t];
                zm[i] = a;
            }
            for (uint i = 0; i < M; ++i) {
                float a = se[i];
                for (uint t = 0; t < M; ++t) a -= ZJ[i][t] * zm[t];
                dv[i] = a;
            }
            for (uint i = 0; i < M; ++i) {
                float a = 0.0f;
                for (uint t = 0; t < M; ++t) a += UI[t][i] * dv[t];
                WV[i] = a;
            }
            X11_SOLVE
            for (uint i = 0; i < M; ++i) {
                float a = dv[i];
                for (uint t = 0; t < M; ++t) a -= XI[M + i][t] * WV[t];
                rv[i] = a;
            }
            for (uint i = 0; i < M; ++i) {
                seed_m[ov + i] = cotv * rv[i];
                for (uint j = 0; j < M; ++j) {
                    float jt = 0.0f;
                    for (uint t = 0; t < M; ++t) jt += XI[M + i][M + t] * XI[M + j][M + t];
                    seed_C[om + i * M + j] = 0.5f * cotv * (rv[i] * rv[j] - jt);
                }
            }
        }
        if (k == 0) break;          // nothing composes onto block 0's suffix
        // absorb block k: i = block k, j = the suffix
        float AI[MDIM][MDIM], ZI[MDIM][MDIM], bi[MDIM], ei[MDIM];
        for (uint i = 0; i < M; ++i) {
            bi[i] = in_b[ov + i]; ei[i] = in_eta[ov + i];
            for (uint j = 0; j < M; ++j) {
                UI[i][j] = in_U[om + i * M + j];
                AI[i][j] = in_A[om + i * M + j];
                ZI[i][j] = in_Z[om + i * M + j];
                ZJ[i][j] = sZ[i][j];
            }
        }
        SQRT_XI
        // d = eta_s - Z_s Z_s^T b_i ; h = d - X21 X11^-1 U_i^T d
        float zm[MDIM], dv[MDIM], WV[MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint t = 0; t < M; ++t) a += ZJ[t][i] * bi[t];
            zm[i] = a;
        }
        for (uint i = 0; i < M; ++i) {
            float a = se[i];
            for (uint t = 0; t < M; ++t) a -= ZJ[i][t] * zm[t];
            dv[i] = a;
        }
        for (uint i = 0; i < M; ++i) {
            float a = 0.0f;
            for (uint t = 0; t < M; ++t) a += UI[t][i] * dv[t];
            WV[i] = a;
        }
        X11_SOLVE
        for (uint i = 0; i < M; ++i) {
            float a = dv[i];
            for (uint t = 0; t < M; ++t) a -= XI[M + i][t] * WV[t];
            zm[i] = a;                                               // h
        }
        // eta <- A_i^T h + eta_i ;  Z <- tria([A_i^T X22, Z_i])
        float X[MDIM][2 * MDIM];
        for (uint i = 0; i < M; ++i) {
            float a = ei[i];
            for (uint t = 0; t < M; ++t) a += AI[t][i] * zm[t];
            se[i] = a;
            for (uint j = 0; j < M; ++j) {
                float c = 0.0f;
                for (uint t = 0; t < M; ++t) c += AI[t][i] * XI[M + t][M + j];
                X[i][j] = c;
                X[i][M + j] = ZI[i][j];
            }
        }
        TRIA_X
        for (uint i = 0; i < M; ++i)
            for (uint j = 0; j < M; ++j) sZ[i][j] = X[i][j];
    }
//SEEDS}
"""

_sqrt_scan_kernels: dict = {}


def _region_src(src: str, tag: str, keep: bool) -> str:
    """Keep or drop the ``//TAG{`` ... ``//TAG}`` regions of a source."""
    out, on = [], True
    for line in src.split("\n"):
        t = line.strip()
        if t == f"//{tag}{{":
            on = keep
        elif t == f"//{tag}}}":
            on = True
        elif on:
            out.append(line)
    return "\n".join(out)


def _sqrt_scan_kernel(m: int, seeds: bool):
    key = (m, bool(seeds))
    if key not in _sqrt_scan_kernels:
        body = (_region_src(_SRC_SQRT_SCAN, "SEEDS", seeds)
                .replace("SQRT_XI", _SQRT_XI)
                .replace("TRIA_XI", _tria_src("XI", "2 * M", "2 * M"))
                .replace("TRIA_X", _TRIA_X)
                .replace("X11T_SOLVE", _X11T_SOLVE)
                .replace("X11_SOLVE", _X11_SOLVE))
        outs = ["ent_m", "ent_C", "pre_b", "pre_U"]
        if seeds:
            outs += ["seed_m", "seed_C"]
        _sqrt_scan_kernels[key] = mx.fast.metal_kernel(
            name=f"anvilgp_sqrt_scan_m{m}" + ("" if seeds else "_entry"),
            input_names=["in_A", "in_b", "in_U", "in_eta", "in_Z",
                         "start_m", "start_C", "cot", "n_blocks", "n_chains"],
            output_names=outs, source=body.replace("MDIM", str(m)))
    return _sqrt_scan_kernels[key]


def sqrt_scan(blocks, start_m, start_C, cot=None):
    """Entry states, and seeds if ``cot`` is given, from `sqrt_blocks`'
    elements: ``(ent_m, ent_C)`` or ``(ent_m, ent_C, seed_m, seed_C)``,
    each ``(n_chains, n_blocks, m[, m])``, as :func:`block_seeds_kernel`
    returns them.  ``start_m``, ``start_C``: the start states the blocks
    were centred on.  One thread per chain; see `_SRC_SQRT_SCAN`.
    """
    A, b, U, eta, Z = blocks
    n_chains, n_blocks, m = b.shape
    seeds = cot is not None
    vec, mat = (n_chains, n_blocks, m), (n_chains, n_blocks, m, m)
    shapes = [vec, mat, vec, mat] + ([vec, mat] if seeds else [])
    if cot is None:
        cot = mx.zeros((n_chains,), dtype=mx.float32)
    outs = _sqrt_scan_kernel(m, seeds)(
        inputs=[A, b, U, eta, Z, start_m, start_C, cot,
                int(n_blocks), int(n_chains)],
        output_shapes=shapes, output_dtypes=[mx.float32] * len(shapes),
        grid=(n_chains, 1, 1), threadgroup=(min(64, n_chains), 1, 1))
    return (outs[0], outs[1], outs[4], outs[5]) if seeds else (outs[0], outs[1])


def metal_available() -> bool:
    """Probe: can the fused kernel actually run here?"""
    global _metal_ok
    if _metal_ok is None:
        try:
            phi = mx.zeros((1, 1, 2, 2), dtype=mx.float32)
            q = mx.broadcast_to(mx.eye(2, dtype=mx.float32), (1, 1, 2, 2))
            out = _kernel(2)(
                inputs=[phi, q, mx.zeros((1,), dtype=mx.int32),
                        mx.zeros((1,), dtype=mx.int32),      # inst
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


def scan_blocks(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2,
                s_floor_rel, blk_len, n_blocks=None,
                formal=False, with_ell=True) -> Element:
    """Condense every block of ``blk_len`` observations into one element.

    ``phi_tab``/``q_tab``: (n_chains, n_lags, m, m) propagator and process
    noise indexed by lag, with row 0 reserved for the no-predecessor case.
    ``lag``: (npts,) int32.  ``resid``: **(npts, n_chains)** residuals --
    time-major, so adjacent threads (adjacent chains) read adjacent floats.
    ``r_ref``: (npts,) float32 reference variances.  ``jit2``: (n_chains, 1)
    jitter variance, or a (n_chains, 0) array for none.  ``formal`` selects
    the convolved-term variant; see :func:`_formal_src`.  ``with_ell=False``
    skips the blocks' own log-normalisers (``ell`` comes back zero), which
    only :func:`reduce_blocks` reads; see :func:`_ell_src`.

    Returns an :class:`~anvilgp.elements.Element` with batch shape
    (n_chains, n_blocks), ready for :func:`anvilgp.elements.tree_reduce`.
    """
    n_chains, n_lags, m = phi_tab.shape[0], phi_tab.shape[1], phi_tab.shape[-1]
    npts = resid.shape[0]          # resid is (npts, n_chains)
    if m > MAX_STATE:
        raise ValueError(f"metal scan supports m <= {MAX_STATE}, got {m}")
    if n_blocks is None:
        n_blocks = (npts + blk_len - 1) // blk_len
    n_jit = _n_jit_of(jit2)

    shapes = [(n_chains, n_blocks, m, m), (n_chains, n_blocks, m),
              (n_chains, n_blocks, m, m), (n_chains, n_blocks, m),
              (n_chains, n_blocks, m, m), (n_chains, n_blocks)]
    scalars = [float(s_floor_rel), int(npts), int(n_lags), int(n_blocks),
               int(n_chains), int(blk_len), int(n_jit)]

    outs = _kernel(m, n_jit, formal, with_ell)(
        inputs=[phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2, *scalars],
        output_shapes=shapes, output_dtypes=[mx.float32] * 6,
        grid=(n_blocks * n_chains, 1, 1),
        threadgroup=(min(64, n_blocks * n_chains), 1, 1))
    return Element(A=outs[0], b=outs[1], C=outs[2], eta=outs[3], J=outs[4],
                   ell=outs[5])


def _boundary_states(p_t, q_t, lg, inst, hv, rs, rr, j2, s_floor_rel,
                     blk_len, formal, use_sqrt):
    """Entry states for `scan_loglike`'s ``boundary``, and what the seeds
    need: pass 1's elements (dense), or the square-root blocks and the
    start states they were centred on."""
    el = scan_blocks(p_t, q_t, lg, inst, hv, rs, rr, j2, s_floor_rel,
                     blk_len, formal=formal, with_ell=False)
    if not use_sqrt:
        ent_m, ent_C = entry_states(el, formal)
        return ent_m, ent_C, (el.A, el.b, el.C, el.eta, el.J)
    st_m, st_C = entry_states(el, formal)
    blocks = sqrt_blocks(p_t, q_t, q_factor(q_t), lg, inst, hv, rs, rr,
                         j2, st_m, st_C, blk_len)
    ent_m, ent_C = sqrt_scan(blocks, st_m, st_C)
    return ent_m, ent_C, (*blocks, st_m, st_C)


def scan_loglike(phi_tab, q_tab, lag, inst, hvec, resid, r_ref, jit2,
                 s_floor_rel, blk_len, slots=None, formal=False,
                 boundary="dense") -> mx.array:
    """``(n_chains,)`` log-likelihood: the kernel's blocks, reduced.

    **This, not :func:`scan_blocks`, is where the differentiation boundary
    sits**, and that is an M4 requirement rather than tidiness.  Wrapping
    the kernel alone would leave a ``custom_function`` whose outputs are six
    element fields per block, so its VJP would have to accept cotangents on
    all six -- a harder problem, and not the one the verified recursion in
    :mod:`anvilgp.adjoint` solves.  Wrapping the reduction too collapses the
    output to one scalar per chain, and a scalar cotangent is exactly what
    ``filter_backward`` takes.

    Three kernels: :func:`scan_blocks` condenses the blocks,
    :func:`entry_states` runs the prefix over them, and
    :func:`replay_blocks` replays the filter through each block from its
    entry state, giving one partial log-likelihood per block.  Until
    2026-10-24 the value was :func:`reduce_blocks`' fold of the block
    elements' own ``ell`` instead -- one pass cheaper, and carrying the
    dense information form's floor: up to 5.9 log-units on a TESS 20-s
    sector (see `_SRC_REPLAY`).  The fold stays as the element-level
    reduction the tests pin; the value no longer uses it.

    ``boundary`` picks how the entry states and the backward's seeds are
    formed.  ``"dense"``: the prefix (and suffix) over pass 1's dense
    information-form elements, as above.  ``"sqrt"``: those entry states
    are only the start -- :func:`sqrt_blocks` re-condenses every block in
    the square-root information form, centred on the filter replayed from
    them, and :func:`sqrt_scan` gives the entry states and seeds the
    replay and the backward use.  On bright, densely sampled data the
    dense ones are float32-wrong (entry means 5%, the gradient 4-5
    posterior sigma off at the worst posterior draws) and the square-root
    ones are not (docs/boundary-plan.md).  For a ``formal`` model `q_factor`
    clamps ``Q``'s negative part, which the caller must have checked is
    harmless; skipped for a single block, where both give the start state
    and a zero seed.
    """
    if boundary not in ("dense", "sqrt"):
        raise ValueError(f"boundary = {boundary!r} must be 'dense' or 'sqrt'")
    # A formal (convolved) model's Q is indefinite; `q_factor` clamps its
    # negative part, which the caller has to know is harmless -- only for
    # SHO and rotation members, measured (likelihood._convolution_takes_sqrt).
    # The replay and the backward stay formal and exact either way.
    use_sqrt = boundary == "sqrt" and -(-resid.shape[0] // blk_len) > 1

    def boundary(p_t, q_t, lg, hv, rs, rr, j2):
        return _boundary_states(p_t, q_t, lg, inst, hv, rs, rr, j2,
                                s_floor_rel, blk_len, formal, use_sqrt)

    def seeds(aux, cot):
        return _boundary_seeds(aux, cot, formal, use_sqrt)

    @mx.custom_function
    def core(phi_t, q_t, lg, hv, rs, rr, j2):
        # Every kernel below reads the residuals, and MLX handed each
        # further `metal_kernel` use its own copy of them: one (npts x
        # n_chains) buffer per extra use, 0.38 GiB each at N = 1e5 x 1024
        # chains.  Made contiguous once here, no use copies; measured
        # 2026-10-25, the cause inside MLX not pinned down.
        rs = mx.contiguous(rs)
        ent_m, ent_C, aux = boundary(phi_t, q_t, lg, hv, rs, rr, j2)
        # Partial sums per block, then a sum over a dozen or so: a single
        # float32 accumulator over 1e5 O(1) terms costs ~0.4 on its own.
        ll = mx.sum(replay_blocks(phi_t, q_t, lg, inst, hv, rs, rr, j2,
                                  ent_m, ent_C, blk_len, formal), axis=1)
        # ``aux`` goes out too, only so the VJP receives it in ``outputs``
        # instead of recomputing pass 1 and the square-root pass -- that
        # was 1.4-1.8x on value + gradient.  `scan_loglike` returns ``ll``
        # alone, so ``aux`` is never used downstream and its cotangents are
        # zero; the VJP ignores them.
        return (ll, *aux)

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
        # which is every gap in the data.  Replay forward, always.  The
        # block elements the seeds come from are the forward's own, passed
        # out of `core` as extra outputs (since 2026-10-25; recomputed
        # here before).
        p_t, q_t, lg, hv, rs, rr, j2 = primals
        # No `mx.contiguous(rs)` here, unlike `core`: the backward kernel is
        # now the only reader of the residuals in the VJP, so nothing is
        # copied without it, and with it value + gradient measured 78.5
        # against 56.9 ms (dense, N = 1e5 x 1024; the copy's cost here not
        # pinned down).
        cot = cotangents[0]             # on ``ll``; the rest are on ``aux``
        aux = tuple(outputs[1:])

        n_chains, n_lags, m = p_t.shape[0], p_t.shape[1], p_t.shape[-1]
        npts = rs.shape[0]
        if slots is None:
            raise ValueError(
                "the kernel gradient needs the (block, lag) slot map; "
                "`scan_loglike` was called without `slots`")
        # n_slots arrives as a plain int: reading it off `slot_start`
        # would force a device sync on every gradient call.
        slot_id, slot_start, n_slots, fold_plan = slots

        _check_adjoint_budget(n_chains, n_slots, n_lags, npts, m, fold_plan)

        ent_m, ent_C, seed_m, seed_C = seeds(aux, cot)
        d_phi, d_q, d_h, d_y, d_rv = backward_ckpt(
            p_t, q_t, lg, inst, hv, rs, rr, j2, cot, ent_m, ent_C, seed_m,
            seed_C, slot_id, slot_start, n_slots, blk_len, formal=formal)

        # `lag` is an integer index and `r_ref` is the data's own yerr^2 --
        # neither is ever a function of the sampled parameters, so neither
        # needs a cotangent.  `jit2` does: see `_d_jit2`.
        d_phi_tab, d_q_tab = _fold_slots(d_phi, d_q, fold_plan, n_chains,
                                         n_lags, m)
        d_j2 = _d_jit2(d_rv, j2)
        return (d_phi_tab, d_q_tab,
                mx.zeros_like(lg), mx.sum(d_h, axis=1), d_y,
                mx.zeros_like(rr), d_j2)

    return core(phi_tab, q_tab, lag, hvec, resid, r_ref, jit2)[0]
