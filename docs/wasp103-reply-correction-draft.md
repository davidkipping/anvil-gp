# DRAFT, not sent: second update to the WASP103b_CHEOPS reply

To append to `WASP103b_CHEOPS/docs/upstream/anvil-gp_reply.md` (our copy:
`docs/wasp103-reply.md`) after sign-off.

---

## Update, 2026-10-26: what shipped, and one correction

Everything in section 3 is now built and tested, with the same acceptance
tests we promised. One thing differs from what the 2026-10-25 update said,
and we want to be exact about it.

**The correction.** We wrote that we would first build a batched,
differentiable `apply_inverse(Y)` (K⁻¹Y) and `gram(Y)` (YᵀK⁻¹Y), with
marginalisation and `predict` as thin layers on top. What shipped:

| | as promised | as shipped |
|---|---|---|
| `gram(v, Y)` = YᵀK⁻¹Y | batched, differentiable | **as promised**, on every path; marginalisation is built on it |
| `apply_inverse(v, Y)` = K⁻¹Y | batched, differentiable | batched and exact (float64), **but value only: no gradient** in the GP parameters |
| `predict` | a layer on `apply_inverse` | built on a Kalman smoother, which `apply_inverse` now shares |

A gradient of K⁻¹Y needs second derivatives through the filter, which
neither of our gradient routes provides. Nothing you asked for needs it:
the marginalised likelihood's gradient comes from `gram`. If your analysis
does need it, tell us and we will build it.

**Marginalised linear parameters.** Pass the basis to `make_target` (or
`GPLogLike`); it is rebuilt at every evaluation, so it can depend on the
mean model through `m0` (your (1 + m0)(...) baseline form) and on any
covariate:

```python
def baseline(v_mean, x, m0):          # -> (n_chains, N, p), linear in beta
    return mx.stack([1 + m0, (1 + m0) * (x - tc)], axis=-1)

gt = make_target(term, t, y, yerr, mean_fn=transit, x_mean=x, mean_specs=specs,
                 noise=ag.noise.error_scale(), linear_basis=baseline)
beta, cov = gt.linear_posterior(v)    # per posterior draw, float64
```

The restricted-likelihood form keeps −½ ln|ΦᵀK⁻¹Φ|; `linear_prior=(mu,
S)` gives a Gaussian prior instead of a flat one. Checked against brute-force
quadrature over β of the plain likelihood to 1e-8 relative; float32
gradients against float64 finite differences to 1e-4 relative; and a
calibration test in which marginalised and β-sampled fits agree to
0.02-0.05 posterior sd. Cost against the plain likelihood, value +
gradient, N = 1e4 × 1024 chains on the fused kernel: 1.7× (p = 1), 2.0×
(p = 2), 3.1× (p = 5), 7.5× (p = 10).

**Joint targets.** anvil 0.12.0 composes them: `anvil.joint(members,
shared)`, with `anvilgp.anvil.joint_u0` for the start. For the five 1-D
visits sharing the transit parameters:

```python
jt = anvil.joint({f"v{i}": gt_i.target for i, gt_i in enumerate(visits)},
                 shared={"rp": [f"v{i}" for i in range(5)], ...})
u0 = joint_u0(jt, {f"v{i}": gt_i for i, gt_i in enumerate(visits)}, 256)
```

A member can be any anvil log-density, so a dense likelihood of your own
for the seven multi-input visits plugs in beside them. Sharing a GP
hyperparameter across members needs equal specs on each.

**Prediction.** `gt.predict(v, t_pred)` gives the GP component's
posterior mean and variance at any points, per draw, in float64,
integrating β when it is marginalised; your mean model and the noise are
not included. It matches the dense conditional to 1e-14. It is a host-side
loop: about 2 s at N = 1e5 for one draw. A parallel version (~10× faster)
is on our open list.

**Still to come:** the documentation pass (README, tutorial, kernels.md)
covering all of the above.
