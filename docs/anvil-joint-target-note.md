# Note for anvil: a joint target over several log-densities with shared parameters

**2026-10-25.** From anvil-gp. Read against anvil
0.11.0 (`231d41f`): `transforms.py`, `logdensity.py`, `precision.py`,
`engine.py`, `diagnostics.py`, `priors.py`. Where something below is
inferred rather than read, it says so.

**Short version.** We would like anvil to own one small primitive: a
target that sums several independent model-space log-densities, each
reading its own columns of one joint parameter vector, with some columns
shared between them. It is generic over the likelihood and serves turin,
Elome and anvil-gp alike, so we think it belongs in anvil rather than in
any one of them. **One API request** (below), and a few design questions.
If you would rather not take it, we will build it on our side, and nothing
here blocks us.

## Why we are asking

The request came from a downstream analysis (12 CHEOPS visits of one
planet): one transit shape shared by all visits, and a mid-time,
normalisation and GP per visit. Nothing in that is GP-specific: a joint
fit of several datasets, instruments or epochs with shared physics is the
same composition whatever the likelihood. Built in anvil-gp, it would only
help GP targets; built in anvil, it composes anvil-gp, turin and Elome
targets in one fit, and the sampler-side machinery (precision checks,
reanchoring, diagnostics) knows about it.

## What we found in anvil 0.11.0 that makes this small

- **`Transform` is elementwise per column** (`to_model`, `log_det_jac`,
  `model_np`, `from_model_np`), and it is built from a list of
  `ParamSpec`s. So a joint `Transform` is just
  `Transform(shared_specs + member_specs...)`, and a member's u column *is*
  the joint u column for the same spec: initial states, `from_model_np`
  and `to_physical` compose by concatenation.
- **Priors re-bind cleanly.** `ParamSpec.__post_init__` and
  `Transform._resolve_prior` call `bind`, which returns a *new* bound prior,
  so rebuilding a `Transform` from member specs recomputes `priors`,
  `is_proper`, `log_prior_norm` and `has_prior_terms` correctly.
- **Specs compare by value.** `ParamSpec` is a dataclass and every `Prior`
  is a frozen dataclass, so "the two members declare the shared parameter
  identically" is a plain `==`.
- **The tools need three things, all sums.** `validate_precision` calls
  `target.log_prob` and `target.log_prob_hi`; `run(reanchor_every=...)`
  checks `hasattr(target, "log_prob_hi")` and calls it;
  `log_norm_const = transform.log_prior_norm + model_log_norm`.
  `warmup_report` reads only the results' trace, not the target.

## The one trap, and the API request

The composition has to happen in **model space**, not by summing members'
`log_prob(u)`. Each `TransformedLogDensity.log_prob` adds its own
Jacobian and declared priors; summing members would count a shared
parameter's Jacobian and prior once per member. The joint target must sum
the members' *likelihood-and-explicit-prior* functions and apply one
Jacobian and one set of declared priors from the joint `Transform`.

Those functions are `TransformedLogDensity._fn` and `._fn_hi` today:
private. **Request:** expose them read-only, e.g. as
`model_log_prob` / `model_log_prob_hi` properties (and `model_log_norm` is
already public). `log_likelihood(u)` is not a substitute: it takes u and
applies the member's own transform.

## API sketch

```python
from anvil import JointLogDensity          # name is yours to choose

jt = JointLogDensity(
    members={"v1": t1, "v2": t2, ...},      # TransformedLogDensity each
    shared={"ror": {"v1": "ror", "v2": "ror", ...},   # joint name -> member's name
            "b":   {...}},
)
jt.log_prob(u); jt.log_prob_hi(u); jt.names; jt.transform; jt.log_norm_const
jt.member_columns("v1")                     # joint column indices, for reporting
```

How it would work (our sketch; your design wins):

- **Layout.** Joint columns are the shared parameters, then each member's
  unshared parameters, named `"<member>.<name>"`. Each member gets a fixed
  index array `idx_k` (its columns, in its own order).
- **Evaluation.** `model_log_prob(v) = sum_k member_k.model_log_prob(
  mx.take(v, idx_k, axis=1))`: one gather per member with static indices,
  so it compiles; `take`'s VJP scatter-adds, so a shared column receives
  the sum of every member's gradient with no extra code. `log_prob` then
  adds the joint `Transform`'s `log_det_jac` and `log_prior` exactly as
  `TransformedLogDensity` does (the joint target can *be* a
  `TransformedLogDensity` over the summed function).
- **Float64.** `model_log_prob_hi` is the same sum over members'
  `model_log_prob_hi` on float64 columns; it exists only if every member
  has one, otherwise `log_prob_hi` raises as today.
  `model_log_norm = sum_k member_k.model_log_norm`.
- **Checks at construction.** Every shared parameter's `ParamSpec` is equal
  across the members that declare it (or an explicit override spec is
  given -- your call); joint names are unique (`Transform` does not check
  this today); `supports_grad` is the AND of the members'.

## Failure values

We read the guidance in `LogDensity.log_prob`: a failed evaluation returns
a relatively sloped value under `stop_gradient`, not a constant, and the
samplers treat anything at or below -1e29 as "the model failed"
(`_FAILURE_FLOOR` in `kernels/chees.py`, `_LOST_FLOOR` in
`diagnostics.py`). Summing keeps that meaning: one failed member puts the
joint value below -1e29, so the joint chain is failed, which is the right
semantics (a joint model with one infeasible dataset is infeasible).
Inferred, not tested: since the sum's gradient is the sum of the members'
gradients, a failed member's inward slope still reaches the shared columns
undisturbed by the other members' finite values, even though those values
are rounded away in the float32 sum. Worth one test on your side.

## What anvil-gp provides either way

- `GPTarget` as a well-behaved member: its target already carries
  `model_log_prob_hi` and the failure semantics above; we will make sure
  its specs are self-contained, its names are stable, and that `u_ref()`
  (the reference starting point) is available per member so a joint
  starting ball can be assembled by concatenation.
- If the primitive lands, anvil-gp feature-detects it, as with every anvil
  addition (VERSIONS.md: an API addition bumps the version).
- If you decline, we build the same composition in anvil-gp, against our
  own closures rather than your private attributes, and say so in our
  docs.

## Questions

1. Is this welcome in anvil, and under what name and module?
2. Can `model_log_prob` / `model_log_prob_hi` become public on
   `TransformedLogDensity`?
3. Shared-parameter mismatch: require equal `ParamSpec`s, or allow an
   override?
4. Naming: `"<member>.<name>"` for unshared parameters, or another scheme?
   And should `Transform` check name uniqueness in general?
5. Any interaction with `anvil.optimize`, evidence or `emcee_api` we
   should know about? We have not read those paths for this.
