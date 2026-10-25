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
