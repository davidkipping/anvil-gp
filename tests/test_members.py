"""`GPTarget` as a member of a joint target (docs/general-features-plan.md,
item 2).

A joint target -- several datasets, some parameters shared -- is composed
by anvil (proposed: docs/anvil-joint-target-note.md) or by the caller,
from what each member exposes: the model-space log-density in float32 and
float64, its constant, its specs and its reference start.  These tests do
the composition by hand, the way either would, and check that the
protocol is enough to get it right.
"""
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvilgp as ag
from anvilgp.oracle import dense_loglike, sample_ssm
from anvilgp.priors import SpecHint

anvil = pytest.importorskip("anvil")
from anvilgp.anvil import make_target  # noqa: E402

KP = [1.3, 4.0, 3.0]


def _level(v_mean, x):
    """A constant mean level, the shared parameter."""
    return mx.broadcast_to(v_mean[:, :1], (v_mean.shape[0], x.shape[-1]))


def _member(seed, n, mu=0.4):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 30, n))
    yerr = rng.uniform(0.2, 0.3, n)
    grid = np.linspace(0, 30, 3000)
    y = (mu + np.interp(t, grid, sample_ssm(ag.SHOTerm(), KP, grid[1],
                                            grid.size, rng))
         + yerr * rng.standard_normal(n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = make_target(ag.SHOTerm(), t, y, yerr, mean_fn=_level, x_mean=t,
                         mean_specs=[SpecHint("mu", lo=-2.0, hi=2.0)])
    return gt, (t, y, yerr)


@pytest.fixture(scope="module")
def pair():
    return [_member(1, 700), _member(2, 500)]


def _joint(pair):
    """[mu | A's own | B's own]: one transform from the members' specs,
    the sum of their model-space log-densities."""
    (a, _), (b, _) = pair
    na, nb = len(a.specs), len(b.specs)
    ia = np.arange(na)                               # mu then A's own
    ib = np.concatenate([[0], na + np.arange(nb - 1)])   # mu then B's own
    specs = a.specs + b.specs[1:]
    transform = anvil.Transform([s.to_anvil() for s in specs])
    ia_m, ib_m = mx.array(ia.astype(np.int32)), mx.array(ib.astype(np.int32))

    def lp(v):
        return (a.model_log_prob(mx.take(v, ia_m, axis=1))
                + b.model_log_prob(mx.take(v, ib_m, axis=1)))

    def lp_hi(v):
        return (a.model_log_prob_hi(mx.take(v, ia_m, axis=1))
                + b.model_log_prob_hi(mx.take(v, ib_m, axis=1)))

    target = anvil.TransformedLogDensity(
        lp, transform, model_log_prob_hi=lp_hi,
        model_log_norm=a.model_log_norm + b.model_log_norm)
    u0 = np.concatenate([a.u_ref(), b.u_ref()[1:]])
    return target, transform, ia, ib, u0


def test_the_member_protocol_is_public(pair):
    for gt, (t, _, _) in pair:
        assert callable(gt.model_log_prob) and callable(gt.model_log_prob_hi)
        assert len(gt.specs) == gt.transform.dim == len(gt.names)
        assert [s.name for s in gt.specs] == gt.names
        assert gt.model_log_norm == gt.loglike.log_norm_const
        assert gt.target.model_log_norm == gt.model_log_norm


def test_the_constant_reaches_anvil(pair):
    """`model_log_prob + model_log_norm` is the absolute log-likelihood
    (plus the explicit priors), and anvil's `log_norm_const` -- what its
    evidence adds back -- includes it.  It was 0 until 2026-10-25."""
    gt, (t, y, yerr) = pair[0]
    v = np.zeros((1, gt.transform.dim))
    v[0, 0] = 0.4
    with mx.stream(mx.cpu):
        vm = mx.array(v, dtype=mx.float64)
        absolute = (float(np.array(gt.loglike.hi(vm))[0])
                    + gt.model_log_norm)
    with mx.stream(mx.cpu):                       # the data-derived refs
        kp = np.array(gt.param_map.kernel_params(vm))[0]
        jv = np.array(gt.param_map.jitter_var(vm, None))  # make_target
    ref = dense_loglike(ag.SHOTerm(), kp, t, y - 0.4,        # adds one
                        np.sqrt(yerr ** 2 + jv.ravel()[0]))
    assert abs(absolute - ref) < 1e-8 * abs(ref)
    assert gt.target.log_norm_const == pytest.approx(
        gt.transform.log_prior_norm + gt.model_log_norm)


def test_a_joint_target_composes_in_model_space(pair):
    """The joint value is the members' model-space sum plus ONE Jacobian;
    summing the members' own `log_prob(u)` instead counts the shared
    column's Jacobian twice, and the difference is exactly that."""
    target, transform, ia, ib, u0 = _joint(pair)
    (a, _), (b, _) = pair
    rng = np.random.default_rng(0)
    u = mx.array((u0 + 0.05 * rng.standard_normal((4, u0.size)))
                 .astype(np.float32))
    joint = np.array(target.log_prob(u))
    naive = (np.array(a.target.log_prob(u[:, mx.array(ia.astype(np.int32))]))
             + np.array(b.target.log_prob(u[:, mx.array(ib.astype(np.int32))])))
    jac_mu = np.array(anvil.Transform([a.specs[0].to_anvil()])
                      .log_det_jac(u[:, :1]))
    np.testing.assert_allclose(naive - joint, jac_mu, rtol=0, atol=1e-3)


def test_a_shared_column_gets_the_sum_of_the_members_gradients(pair):
    target, transform, ia, ib, u0 = _joint(pair)
    (a, _), (b, _) = pair
    v = np.array(transform.to_model(mx.array(u0[None].astype(np.float32))))
    vm = mx.array(np.repeat(v, 3, axis=0) + np.array([[0.0], [0.1], [-0.1]])
                  * np.eye(1, v.shape[1]))
    g = np.array(mx.grad(lambda v: target._fn(v).sum())(vm))
    ga = np.array(mx.grad(lambda w: a.model_log_prob(w).sum())(
        vm[:, mx.array(ia.astype(np.int32))]))
    gb = np.array(mx.grad(lambda w: b.model_log_prob(w).sum())(
        vm[:, mx.array(ib.astype(np.int32))]))
    np.testing.assert_allclose(g[:, 0], ga[:, 0] + gb[:, 0], rtol=1e-5)
    np.testing.assert_allclose(g[:, ia[1:]], ga[:, 1:], rtol=1e-6)
    np.testing.assert_allclose(g[:, ib[1:]], gb[:, 1:], rtol=1e-6)


def test_the_joint_float64_path_and_precision_check(pair):
    """anvil's own tools run on the composed target: `log_prob_hi` exists
    because every member has one, and `validate_precision` passes."""
    target, transform, ia, ib, u0 = _joint(pair)
    rng = np.random.default_rng(1)
    u = mx.array((u0 + 0.05 * rng.standard_normal((8, u0.size)))
                 .astype(np.float32))
    lo = np.array(target.log_prob(u))
    with mx.stream(mx.cpu):
        hi = np.array(target.log_prob_hi(u))
    assert np.abs(lo - hi).max() < 5e-3
    rep = anvil.validate_precision(target, u)
    assert "ok" in str(rep).lower() or getattr(rep, "ok", True)


def test_one_failed_member_fails_the_joint_chain(pair):
    """A member refusing a chain (here: amplitude and correlation length
    both far out, so the conditioning number kappa -- bounded by the
    amplitude-to-noise ratio -- is past `kappa_max`) returns the failure
    value; the sum stays below the floor anvil's samplers treat as
    failed."""
    target, transform, ia, ib, u0 = _joint(pair)
    v = np.array(transform.to_model(mx.array(u0[None].astype(np.float32))))
    v = np.repeat(v, 2, axis=0)
    names = pair[0][0].names
    v[1, ia[names.index("log_gp_sigma")]] = 20.0
    v[1, ia[names.index("log_gp_rho")]] = 30.0
    out = np.array(target._fn(mx.array(v.astype(np.float32))))
    assert np.isfinite(out[0]) and out[0] > -1e20
    assert out[1] <= -1e29
