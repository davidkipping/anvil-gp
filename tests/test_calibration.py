"""Does the likelihood point at the truth on average?

One injection-recovery at one truth has almost no power against a subtle
likelihood bias -- a wrong normalisation, a mis-scaled process noise, an
off-by-one in the recursion can all leave a single fit looking fine, because
a single realisation's own scatter is larger than the bias.  These tests
average over realisations instead.

This is the cheap, fast version of the M5 calibration work (SBC rank checks
at N = 1e3 through the float64 reference path).  It is not a substitute for
it: unbiasedness of a point estimate is a weaker claim than correct posterior
coverage.  It is, however, enough to catch the class of bug that a single
recovery would miss, and it runs in seconds.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap, a64
from anvilgp.oracle import dense_covariance

mx.set_default_device(mx.cpu)

N = 800
BASELINE = 40.0
SIGMA, RHO, QUAL, JIT, YERR = 1.5, 7.3, 8.0, 0.12, 0.20


def _profile_max(gp_factory, grid, n_real=12, seed0=1000):
    """MAP of one parameter over ``n_real`` fresh realisations of the data."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, BASELINE, N))
    yerr = np.full(N, YERR)
    cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                           yerr, jitter=JIT)
    chol = np.linalg.cholesky(cov)
    best = []
    for k in range(n_real):
        y = chol @ np.random.default_rng(seed0 + k).standard_normal(N)
        ll = np.array([float(np.array(
            gp_factory(t, y, yerr, g).hi_absolute(a64(np.zeros((1, 1)))))[0])
            for g in grid])
        best.append(grid[int(np.argmax(ll))])
    return np.array(best)


def _fixed(t, y, yerr, kp, jitter=JIT):
    pm = ParamMap(kernel=[KernelParam(link="const", value=float(v))
                          for v in kp], n_dim=1,
                  jitter=[KernelParam(link="const", value=float(jitter))])
    return GPLogLike(None, t, None, y, yerr, ag.SHOTerm(), pm,
                     GPPolicy(block=1024, hi_block=1024))


@pytest.mark.slow
def test_amplitude_is_recovered_without_bias():
    grid = np.linspace(0.8, 2.4, 17)
    best = _profile_max(lambda t, y, e, s: _fixed(t, y, e, [s, RHO, QUAL]),
                        grid)
    # the grid step is 0.1, so the mean cannot be resolved below ~0.03
    assert abs(best.mean() - SIGMA) < 0.15, best
    assert best.std() < 0.5, best


@pytest.mark.slow
def test_timescale_is_recovered_without_bias():
    grid = np.exp(np.linspace(math.log(3.0), math.log(18.0), 17))
    best = _profile_max(lambda t, y, e, r: _fixed(t, y, e, [SIGMA, r, QUAL]),
                        grid)
    assert abs(np.log(best.mean() / RHO)) < 0.25, best


@pytest.mark.slow
def test_jitter_is_recovered_without_bias():
    grid = np.exp(np.linspace(math.log(0.03), math.log(0.5), 17))
    best = _profile_max(
        lambda t, y, e, j: _fixed(t, y, e, [SIGMA, RHO, QUAL], jitter=j), grid)
    assert abs(np.log(best.mean() / JIT)) < 0.35, best


def test_a_single_realisation_is_not_evidence():
    """The point of the tests above, made explicit.

    The realised RMS of one draw scatters widely around the process sigma
    when the baseline holds only a handful of correlation times, so a single
    recovery landing 2-3 sigma from the truth is expected behaviour and says
    nothing about the likelihood.
    """
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, BASELINE, N))
    yerr = np.full(N, YERR)
    cov = dense_covariance(ag.SHOTerm(), np.array([SIGMA, RHO, QUAL]), t,
                           yerr, jitter=JIT)
    chol = np.linalg.cholesky(cov)
    rms = np.array([(chol @ np.random.default_rng(2000 + k)
                     .standard_normal(N)).std() for k in range(24)])
    expected = math.sqrt(SIGMA ** 2 + JIT ** 2 + YERR ** 2)
    assert 0.15 < rms.std() / expected < 0.8, rms.std() / expected
    assert abs(rms.mean() / expected - 1.0) < 0.25
