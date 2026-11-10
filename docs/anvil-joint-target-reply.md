# Reply: a joint target over several log-densities

From anvil, answering
[`anvil-joint-target-note.md`](anvil-joint-target-note.md).
Shipped in anvil **0.12.0** (`v0.12.0`, 2026-11-07) and corrected in
**0.12.1** (`v0.12.1`, 2026-11-10). **Use 0.12.1 or later.** 0.12.0 has
two defects in `joint`, described under "What changed in 0.12.1". Nothing
further is requested of you.

**Short version.** We took it, close to your sketch. `anvil.joint(members,
shared)` returns a `JointLogDensity`, a `TransformedLogDensity` subclass,
so every tool takes it unchanged. `model_log_prob`, `model_log_prob_hi`
and `names` are public on `TransformedLogDensity`, as you asked.

```python
jt = anvil.joint(
    {"v1": gt1.target, "v2": gt2.target, "v3": gt3.target},
    shared={"ror": ["v1", "v2", "v3"],                 # same name everywhere
            "t0":  {"v1": "t0", "v2": "tmid", "v3": "t0"}},   # or per member
)
jt.names                    # ['ror', 't0', 'v1.rho', 'v1.sigma', ...]
jt.member_columns("v2")     # joint columns v2 reads, in v2's own order
res = anvil.run(anvil.ChEESHMC(jt), jt, u0)
```

Documentation: https://anvil-mcmc.readthedocs.io/en/latest/joint.html

## Your five questions

1. **Welcome, and where.** Yes. It is `anvil.joint` in
   `anvil/transforms.py`, returning `anvil.JointLogDensity`. We made it a
   factory rather than a constructor, because the work is building the
   layout and the joint `Transform`, after which the result is an ordinary
   `TransformedLogDensity`.

2. **Public `model_log_prob` / `model_log_prob_hi`.** Done, as read-only
   properties, together with `names`. `log_prob_hi` sets the CPU stream as
   before. Since 0.12.1, a joint's `model_log_prob_hi` also sets it when
   called directly. A member's own `model_log_prob_hi` still runs on
   whatever stream the caller sets.

3. **Shared-parameter mismatch.** Equal `ParamSpec`s are required, with no
   override. A member may call the shared parameter by a different name,
   through the dict form of `shared`. Two models disagreeing about one
   quantity's bounds or prior is a modelling decision, so `joint` refuses
   it and prints both specs. Since 0.12.1 the priors are compared as
   resolved, so `prior=None` on a finite box agrees with an explicit
   `Uniform()` on that box.

4. **Naming.** `"<member>.<name>"` for unshared parameters, as you
   proposed. The joint columns are the shared parameters in `shared`'s
   order, then each member's unshared parameters in member order.
   `Transform` now rejects duplicate names in general. If a generated name
   collides with another joint name, the error says which two sources
   produced it.

5. **Interactions.**
   - **`optimize`:** `objective="mle"` maximises the pooled likelihood and
     `"map"` adds the joint priors once.
   - **Evidence:** `smc` tempers the summed likelihood, not the prior.
     `model_log_norm` is the members' sum, so ln Z is absolute when every
     member declares its constants. A review checked this against an
     analytic ln Z (within 1.4 standard errors), and 0.12.1 has a test for
     it. `bridge_evidence` and `laplace_evidence` need nothing special.
   - **`emcee_api`:** no special path. `HMCSampler(..., target=jt)` works.
     For the ensemble kernel, call `anvil.run(EnsembleKernel(jt), jt, u0)`.

## Your failure-value question, tested

You asked for one test of the inferred claim that a failed member's slope
reaches the shared columns. It holds, and 0.12.0 has the test.

- A member at the sloped sentinel puts the joint value below −1e29, and
  the shared column's gradient carries that member's inward slope.
- Two members failing at once sum to about −2e30, with both slopes
  intact and no NaN gradients.
- A failed member plus a NaN member gives NaN, which the samplers reject
  as they would for a single target.

## What changed in 0.12.1

A three-part review of 0.12.0 found no case where a well-formed joint
target computes the wrong density. It did find these:

- **A member parameter listed under two `shared` entries** silently
  dropped the first sharing. The later entry took the column while
  `jt.shared` still reported both. It is now refused.
- **The members' values were combined by a chained sum.** MLX fuses that
  across members into one Metal kernel and refused the graph from about 8
  `ChunkedGaussianLogLike(reduction="fixed_point")` members of several
  chunks each, though each member compiles alone. The run fell back to
  eager, about four times slower. 0.12.1 combines them through
  `mx.sum(mx.stack(...))`, a fusion barrier that costs nothing where the
  chain compiled. Joint values with three or more members change at
  float32 round-off.
- **`joint` now copies the members dict,** so later changes to the
  caller's dict no longer reach the target.
- **A member whose class overrides `log_prob` or `log_prob_hi` is
  refused,** because the sum is over `model_log_prob` and the override
  would be dropped.

On the second point: we did not run your GP targets. A GP member whose
model function ends in elementwise arithmetic on several per-chain
scalars (quadratic form, log-determinant, constants) was exposed to the
same refusal in 0.12.0 by the same mechanism. 0.12.1 removes it for any
member.

## Two things to plan around

- **Memory.** The joint target is one graph, so every member's
  intermediates are live together. Its peak memory is the members' sum,
  not the largest member's. Measured with 12 Gaussian-likelihood members,
  the gradient's peak was 6 to 8 times that of evaluating the members one
  at a time. Size a 12-visit fit by its total number of data points and
  chains. A member-at-a-time evaluation mode would lower the peak, but it
  would cost about 12 to 14% on gradients and could not run inside the
  compiled ChEES step. We have not built it. If the CHEOPS fit runs out of
  memory, tell us and we will reconsider.
- **Starting points.** `GPTarget` holds its target at `.target`, which is
  what `joint` takes. For a joint start, place each member's `u_ref()`
  into `jt.member_columns(name)`. For a shared column, take one member's
  value or their mean.

## Feature detection

`hasattr(anvil, "joint")` is true from 0.12.0. To avoid the two 0.12.0
defects, also require `anvil.__version__ >= "0.12.1"` (compare as a
version, not a string).
