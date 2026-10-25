"""The kernel's block boundary quantities on bright, densely sampled data.

A TESS 20-s cadence, rotation plus granulation (SHO + SHO), s/e = 100,
N = 4096, at 16 draws from the float64 Laplace posterior.  The kernel's
backward starts each block from an entry state and a seed composed across
blocks in the dense information form, and at float32 both are wrong here:
entry means 5%, seeds up to 1%, and the gradient 4-5 posterior sigma off
at the worst draw -- enough that ChEES does not converge at the default
blocks (docs/float32.md, section 7, step 2).

The fix is the square-root information form centred block-locally
(docs/boundary-plan.md).  Its NumPy reference, `boundary_oracle`, is
pinned here twice: in float64 it reproduces the sequential filter, and in
float32 it meets the bounds the kernel must meet, through the kernel's own
float32 backward.  The kernel's tests are expected failures until stage 4
wires the new boundary in; `strict` makes that stage remove the markers.
"""

import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
import boundary_oracle as bo
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap
import anvilgp.metal as am
from anvilgp.metal import q_factor
from anvilgp.oracle import sample_ssm

N, DT, N_DRAWS = 4096, 1.0 / 4320, 16          # 20 s, in days
KP = [0.1, 3 / (2 * np.pi), 5.0, 0.03, 0.2, 1 / np.sqrt(2)]

#: Relative to the float64 filter (max |error| / max |truth|).  The float32
#: oracle's worst over both block lengths and two data draws was 7.1e-4,
#: 1.5e-5 and 8.3e-4 (2026-10-25); the kernel is at 3-5e-2, 5e-3-2e-2 and
#: 5e-4-9e-3.
ENTRY_MEAN, ENTRY_COV, SEED = 1e-3, 1e-4, 2.5e-3
#: Gradient error per posterior sigma, worst over draws and parameters.
#: Oracle 0.33 / 0.24, kernel 4.0 / 5.0 (16 x 256 / 4 x 1024).
GRAD_SIGMA = 0.5

KERNEL_RED = pytest.mark.xfail(
    strict=True, reason="dense-form boundary; fixed by docs/boundary-plan.md")


def _hi(gp, V):
    with mx.stream(mx.cpu):
        return np.array(gp.hi(mx.array(V, dtype=mx.float64), refuse=False))


@pytest.fixture(scope="module", params=[256, 1024], ids=["16x256", "4x1024"])
def case(request):
    """Data, posterior draws, float64 reference gradients, and the kernel's
    backward inputs with its own and the exact boundary quantities."""
    blk = request.param
    term = ag.SHOTerm() + ag.SHOTerm()
    rng = np.random.default_rng(2)
    t = np.arange(N) * DT
    yerr = np.full(N, 1e-3)
    y = sample_ssm(term, KP, DT, N, rng) + yerr * rng.standard_normal(N)
    d = len(KP)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=x)
                          for i, x in enumerate(KP)], n_dim=d)
    ref = GPLogLike(None, t, None, y, yerr, term, pm, GPPolicy(kappa_max=None))

    # Laplace posterior from a float64 FD Hessian; widths capped at 0.5 in log
    hs = 1e-3
    pts = [np.zeros(d)]
    for i in range(d):
        for j in range(i, d):
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = np.zeros(d)
                p[i] += si * hs
                p[j] += sj * hs
                pts.append(p)
    f = _hi(ref, np.array(pts))
    H = np.zeros((d, d))
    c = 1
    for i in range(d):
        for j in range(i, d):
            fpp, fpm, fmp, fmm = f[c:c + 4]
            c += 4
            H[i, j] = H[j, i] = -(fpp - fpm - fmp + fmm) / (4 * hs ** 2)
    w, U = np.linalg.eigh(H)
    w = np.maximum(w, 4.0)
    sd = np.sqrt(np.diag((U / w) @ U.T))
    V = (U / np.sqrt(w) @ np.random.default_rng(99)
         .standard_normal((d, N_DRAWS))).T

    gh = 1e-5
    P = np.repeat(V, 2 * d, axis=0)
    for k in range(N_DRAWS):
        for i in range(d):
            P[k * 2 * d + 2 * i, i] += gh
            P[k * 2 * d + 2 * i + 1, i] -= gh
    ff = _hi(ref, P).reshape(N_DRAWS, d, 2)
    G = (ff[..., 0] - ff[..., 1]) / (2 * gh)

    gp = GPLogLike(None, t, None, y, yerr, term, pm,
                   GPPolicy(metal=True, kappa_max=None, metal_block=blk))
    got = {}

    def record(inputs, kernel):
        got["inputs"], got["kernel"] = inputs, kernel
        return kernel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with bo.boundary_from(record), mx.stream(mx.gpu):
            g = np.array(mx.grad(lambda u: gp(u).sum())(
                mx.array(V, dtype=mx.float32)))
    a = got["inputs"]
    exact = bo.exact_boundary(a["pt"], a["qt"], a["lag"], a["h"], a["res"],
                              a["r"], a["r_ref"], a["cot"], a["blk"])
    return dict(blk=blk, gp=gp, V=V, sd=sd, G=G, g_kernel=g,
                inputs=a, kernel=got["kernel"], exact=exact)


def _oracle(case, dt=np.float32):
    a, k = case["inputs"], case["kernel"]
    return bo.sqrt_boundary(a["pt"], a["qt"], a["lag"], a["h"], a["res"],
                            a["r"], a["cot"], k[0], k[1], a["blk"], dt=dt)


def _errors(got, exact):
    return [bo.rel_err(got[i], exact[i], last_seed=i >= 2) for i in range(4)]


def _check_bounds(got, exact):
    em, eC, sm, sC = _errors(got, exact)
    assert em < ENTRY_MEAN and eC < ENTRY_COV and max(sm, sC) < SEED, (
        f"entry means {em:.1e} (< {ENTRY_MEAN:g}), covariances {eC:.1e} "
        f"(< {ENTRY_COV:g}), seeds {sm:.1e} / {sC:.1e} (< {SEED:g})")


def _grad_sigma(case, g):
    return (np.abs(g - case["G"]) * case["sd"]).max(axis=0)


def test_the_oracle_reproduces_the_float64_filter(case):
    """The algebra check: in float64 the square-root elements, centred on
    a trajectory from the kernel's (float32-wrong) entry states, give the
    sequential filter's boundary quantities.  Measured 1e-12."""
    err = _errors(_oracle(case, np.float64), case["exact"])
    assert max(err) < 1e-9, err


def test_the_oracle_meets_the_boundary_bounds_in_float32(case):
    _check_bounds(_oracle(case), case["exact"])


def test_the_oracle_boundary_gives_the_kernel_backward_a_good_gradient(case):
    """What the kernel must reach is reachable: the oracle's float32
    boundary quantities, through the kernel's own float32 backward."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with bo.boundary_from(lambda inputs, kernel: _oracle(case)), \
                mx.stream(mx.gpu):
            g = np.array(mx.grad(lambda u: case["gp"](u).sum())(
                mx.array(case["V"], dtype=mx.float32)))
    e = _grad_sigma(case, g)
    assert e.max() < GRAD_SIGMA, f"per posterior sigma {np.round(e, 3)}"


@KERNEL_RED
def test_the_kernel_boundary_quantities_meet_the_bounds(case):
    _check_bounds(case["kernel"], case["exact"])


@KERNEL_RED
def test_the_kernel_gradient_is_within_half_a_sigma_across_the_posterior(case):
    e = _grad_sigma(case, case["g_kernel"])
    assert e.max() < GRAD_SIGMA, f"per posterior sigma {np.round(e, 3)}"


# ------------------------------------------------- stage 1: the Q factor
FACTOR_TERMS = [
    ("real", ag.RealTerm(), [1.0, 2.0]),
    ("sho", ag.SHOTerm(), [0.1, 0.48, 5.0]),
    ("sho-overdamped", ag.SHOTerm(), [1.0, 0.3, 0.2]),
    ("matern32", ag.Matern32Term(), [1.0, 0.5]),
    ("matern52", ag.Matern52Term(), [1.1, 2.5]),
    ("sho+sho", ag.SHOTerm() + ag.SHOTerm(), KP),
    ("rotation", ag.RotationTerm(), [1.5, 7.0, 3.0, 1.0, 0.4]),
    ("sho+matern32", ag.SHOTerm() + ag.Matern32Term(), [1.0, 5.0, 4.0, 0.5, 2.0]),
]
#: a duplicate timestamp, 2 s, 20 s, 2 min, 30 min, and a 10-day gap
FACTOR_LAGS = np.array([0.0, 2.0, 20.0, 120.0, 1800.0, 864000.0]) / 86400


def _factor(q):
    with mx.stream(mx.gpu):
        L = q_factor(mx.array(np.asarray(q, np.float32)))
        mx.eval(L)
    return np.array(L)


@pytest.mark.parametrize("name,term,p", FACTOR_TERMS,
                         ids=[t[0] for t in FACTOR_TERMS])
def test_the_q_factor_keeps_every_direction_of_q(name, term, p):
    """``L L^T = Q`` direction by direction, not just entry by entry.

    The eigenvalues of these ``Q`` span up to 22 decades (Matern-5/2 at
    2 s), and the small ones are what a high-s/e filter depends on.  The
    composition kernels' ``eps * trace / m`` floor reproduces every entry
    of ``Q`` to 1e-7 and still inflated the smallest eigenvalue 1.7x at
    20 s and 70x at 2 s, which cost the boundary seeds 20-50x.  Measured
    with the clamp: every direction to <= 3e-5, entries to 1e-7.
    """
    with mx.stream(mx.gpu):
        _, q, _ = term.matrices(mx.array([p], dtype=mx.float32),
                                mx.array(FACTOR_LAGS, dtype=mx.float32))
        mx.eval(q)
    Q = np.array(q)
    L = _factor(Q)
    Q, L = Q[0], L[0]
    assert np.all(np.isfinite(L))
    assert np.all(np.triu(L, 1) == 0) and np.all(np.diagonal(L, 0, -2, -1) >= 0)
    np.testing.assert_allclose(L, bo.sqrt_q(Q), rtol=0,
                               atol=1e-5 * np.abs(L).max())
    for k, lag in enumerate(FACTOR_LAGS):
        q64, l64 = Q[k].astype(np.float64), L[k].astype(np.float64)
        if not np.any(q64):
            assert not np.any(l64), f"lag {lag}: Q = 0 but L != 0"
            continue
        llt = l64 @ l64.T
        assert np.abs(llt - q64).max() < 1e-6 * np.abs(q64).max(), lag
        w, U = np.linalg.eigh(q64)
        for i in np.flatnonzero(w > 0):
            ratio = U[:, i] @ llt @ U[:, i] / w[i]
            assert abs(np.log(ratio)) < 1e-3, (
                f"lag {lag * 86400:g} s: eigenvalue {w[i]:.2e} "
                f"(of {w[-1]:.2e}) reproduced x{ratio:.4f}")


def test_the_q_factor_of_a_rank_deficient_q_is_finite_and_exact():
    """Pivots at or below zero zero their column instead of dividing by
    them: a rank-one ``Q``, one whose last pivot rounds negative, and zero
    -- every case a lag table can hand it (row 0's P_inf aside, which is
    the identity)."""
    v = np.array([0.3, -1.2, 0.7, 2.0], np.float32)
    rank1 = np.outer(v, v)
    near = np.array([[1.0, 1.0], [1.0, 1.0 - 3e-8]], np.float32)  # pivot < 0
    for Q in (rank1, near, np.zeros((3, 3), np.float32), np.eye(4, dtype=np.float32)):
        L = _factor(Q[None, None])[0, 0].astype(np.float64)
        assert np.all(np.isfinite(L)), Q
        assert np.all(np.triu(L, 1) == 0)
        assert np.abs(L @ L.T - Q).max() <= 1e-6 * max(np.abs(Q).max(), 1e-30), Q


def test_the_q_factor_batches_chains_and_lags_independently():
    """One thread per (chain, lag) entry: a batch gives what each entry
    gives alone."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 5, 4, 4)).astype(np.float32)
    Q = X @ np.swapaxes(X, -1, -2)
    L = _factor(Q)
    for c in range(3):
        for k in range(5):
            np.testing.assert_array_equal(L[c, k], _factor(Q[c:c + 1, k:k + 1])[0, 0])
    with pytest.raises(ValueError, match="m <= "):
        q_factor(mx.zeros((1, 1, 5, 5), dtype=mx.float32))


def test_the_oracle_on_the_kernels_q_factor_meets_the_bounds(case):
    """The stage 1 factor in the stage 0 oracle, on the 20-s problem: the
    boundary quantities stay inside the bounds the kernel must meet."""
    a, k = case["inputs"], case["kernel"]
    got = bo.sqrt_boundary(a["pt"], a["qt"], a["lag"], a["h"], a["res"],
                           a["r"], a["cot"], k[0], k[1], a["blk"],
                           lq=_factor(a["qt"]))
    _check_bounds(got, case["exact"])


# ------------------------------------------- stage 2: the block pass
def _kernel_blocks(a, start_m, start_C):
    """Stage 2's kernel on captured backward inputs, from given starts."""
    pt, qt, lag, inst, hv, rs, rr, j2 = a["mx"]
    with mx.stream(mx.gpu):
        lq = q_factor(qt)
        out = am.sqrt_blocks(pt, qt, lq, lag, inst, hv, rs, rr, j2,
                             mx.array(start_m, dtype=mx.float32),
                             mx.array(start_C, dtype=mx.float32), a["blk"])
        mx.eval(lq, *out)
    return [np.array(x) for x in out], np.array(lq)


def _canonical(U):
    """The lower-triangular factor of ``U U^T`` with non-negative diagonal,
    in float64: two square roots of one matrix compare through it."""
    L = bo.tria(np.asarray(U, np.float64))
    d = np.sign(np.diagonal(L, 0, -2, -1))
    d[d == 0] = 1
    return L * d[..., None, :]


def _block_errors(got, want):
    """Per field, max |error| / max |truth|; ``U``, ``Z`` through
    :func:`_canonical`."""
    out = []
    for i, (g, w) in enumerate(zip(got, want)):
        if i in (2, 4):
            g, w = _canonical(g), _canonical(w)
        out.append(float(np.abs(g - w).max() / max(np.abs(w).max(), 1e-30)))
    return out


def _as_good_as_numpy(a, start_m, start_C, blk=None):
    """Kernel blocks against the float64 oracle, field by field, each no
    worse than 5x the float32 oracle's own error (floor 2e-5).  Measured on
    the 20-s problem: within 1-3.5x, the worst being ``U``, whose error is
    Metal's float32 rounding rather than the algebra (the oracle on the
    kernel's exact algebra and Householder is within 1.6x of LAPACK)."""
    kb, lq = _kernel_blocks(a, start_m, start_C)
    args = (a["pt"], a["qt"], a["lag"], a["h"], a["res"], a["r"],
            start_m, start_C, a["blk"] if blk is None else blk)
    f64 = lambda x: np.asarray(x, np.float64)
    want = bo.sqrt_blocks(f64(args[0]), f64(args[1]), args[2],
                          *map(f64, args[3:6]), *args[6:], dt=np.float64,
                          lq=f64(lq))
    ref32 = bo.sqrt_blocks(*args, lq=lq)
    ek, en = _block_errors(kb, want), _block_errors(ref32, want)
    for name, k, n in zip("A b U eta Z".split(), ek, en):
        assert k <= max(5 * n, 2e-5), (
            f"{name}: kernel {k:.1e} against the float32 oracle's {n:.1e}")
    return kb


def test_the_block_pass_matches_the_oracle(case):
    a, k = case["inputs"], case["kernel"]
    kb = _as_good_as_numpy(a, k[0], k[1])
    # the first block has no predecessor (Phi = 0 at lag row 0), so its A,
    # eta and J are exactly zero, as in pass 1 -- stage 3 may rely on it
    for i in (0, 3, 4):
        assert not np.any(kb[i][:, 0]), "A, eta, Z of block 0 must be 0"


def test_the_block_pass_meets_the_boundary_bounds(case):
    """Kernel blocks, NumPy scan (stage 3's job): inside the bounds.
    Measured 3.2e-4 / 1.8e-5 / 2.5e-4 at 16 x 256, 1.6e-4 / 2.7e-5 /
    1.0e-3 at 4 x 1024."""
    a, k = case["inputs"], case["kernel"]
    kb, _ = _kernel_blocks(a, k[0], k[1])
    _check_bounds(bo.sqrt_scan(kb, k[0], k[1], a["cot"]), case["exact"])


def test_the_block_pass_gives_the_kernel_backward_a_good_gradient(case):
    a0, k0 = case["inputs"], case["kernel"]
    kb, _ = _kernel_blocks(a0, k0[0], k0[1])
    scan = bo.sqrt_scan(kb, k0[0], k0[1], a0["cot"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with bo.boundary_from(lambda inputs, kernel: scan), mx.stream(mx.gpu):
            g = np.array(mx.grad(lambda u: case["gp"](u).sum())(
                mx.array(case["V"], dtype=mx.float32)))
    e = _grad_sigma(case, g)
    assert e.max() < GRAD_SIGMA, f"per posterior sigma {np.round(e, 3)}"


def _capture(gp, n_par, n_chains=2, seed=5):
    v = np.random.default_rng(seed).normal(0, 0.05, (n_chains, n_par))
    got = {}

    def record(inputs, kernel):
        got["a"], got["k"] = inputs, kernel
        return kernel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with bo.boundary_from(record), mx.stream(mx.gpu):
            mx.eval(mx.grad(lambda u: gp(u).sum())(
                mx.array(v, dtype=mx.float32)))
    return got["a"], got["k"]


@pytest.mark.parametrize("n_jit", [0, 1, 3])
def test_the_block_pass_carries_each_instruments_jitter(n_jit):
    """The jitter enters both the replay and the leaf variance, per
    observation when there are several (the instrument-indexed read)."""
    n, blk = 768, 64
    rng = np.random.default_rng(4)
    t = np.arange(n) * 0.02
    inst = None if n_jit <= 1 else (np.arange(n) % n_jit).astype(np.int32)
    pm = ParamMap(
        kernel=[KernelParam(link="log", index=i, ref=r)
                for i, r in enumerate((1.3, 4.0, 3.0))],
        jitter=[KernelParam(link="log", index=3 + j, ref=(0.03, 0.3, 0.1)[j])
                for j in range(n_jit)],
        n_dim=3 + n_jit, instrument=inst)
    gp = GPLogLike(None, t, None, rng.standard_normal(n), np.full(n, 0.05),
                   ag.SHOTerm(), pm,
                   GPPolicy(metal=True, kappa_max=None, metal_block=blk))
    a, k = _capture(gp, 3 + n_jit)
    if n_jit == 3:
        assert np.ptp(a["r"][0, :3]) > 0, "jitters must differ by instrument"
    _as_good_as_numpy(a, k[0], k[1])


def test_the_block_pass_handles_a_short_last_block():
    """N not a multiple of the block: the last thread stops early.  Its
    element is the oracle's single block over the same points."""
    n, blk = 1000, 128
    term = ag.SHOTerm() + ag.SHOTerm()
    rng = np.random.default_rng(2)
    t = np.arange(n) * DT
    yerr = np.full(n, 1e-3)
    y = sample_ssm(term, KP, DT, n, rng) + yerr * rng.standard_normal(n)
    pm = ParamMap(kernel=[KernelParam(link="log", index=i, ref=x)
                          for i, x in enumerate(KP)], n_dim=len(KP))
    gp = GPLogLike(None, t, None, y, yerr, term, pm,
                   GPPolicy(metal=True, kappa_max=None, metal_block=blk))
    a, k = _capture(gp, len(KP))
    kb, lq = _kernel_blocks(a, k[0], k[1])
    tail = slice((n // blk) * blk, n)
    one = bo.sqrt_blocks(a["pt"].astype(np.float64), a["qt"].astype(np.float64),
                         a["lag"][tail], a["h"].astype(np.float64),
                         a["res"][:, tail].astype(np.float64),
                         a["r"][:, tail].astype(np.float64),
                         k[0][:, -1:], k[1][:, -1:], n - tail.start,
                         dt=np.float64, lq=lq.astype(np.float64))
    err = _block_errors([x[:, -1:] for x in kb], one)
    assert max(err) < 1e-3, err
