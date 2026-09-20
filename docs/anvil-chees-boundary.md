# ChEES-HMC cannot leave the prior boundary

**Status:** a finding about `anvil`, not about `anvil-gp`, written up here
because anvil-gp's tests are what found it. Measured 2026-09-20 on an M2
Max, MLX 0.32.2, anvil at `../anvil`.

## What happens

A chain started deep in a bounded parameter's prior tail never moves again
under `ChEESHMC`, at any leapfrog cap. The rest of the ensemble is
unaffected and samples normally.

Reproduced with a plain SHO GP target from `anvilgp.anvil.make_target`,
16 chains, half started at `u = 15` in the unconstrained space:

| start | log-probability | moved after 80 iterations |
|---|---:|---|
| `u ~ N(0, 0.3)` | -840 | 8 of 8 |
| `u = 15` | -9355 | 0 of 8 |

The stuck chains end at exactly their starting log-probability.

## Why

Two separate mechanisms, both in `anvil/transforms.py`.

**1. The Jacobian gradient blows up before the transform saturates.**
`log_det_jac` for a bounded parameter is `log(w) + log(sig) + log1p(-sig)`.
Its derivative grows without bound as `sig -> 1`. Measured on the same
target, gradient of the total log-probability:

| `u` | max abs gradient |
|---:|---:|
| 0 | 5.1e2 |
| 5 | 5.5e1 |
| 10 | 5.6e2 |
| 15 | **6.0e8** |
| 17 | 2.4e7 |

ChEES adapts **one** step size for the whole ensemble. A step that suits a
gradient of 5e2 throws a chain with a gradient of 6e8 to infinity, so every
proposal from the boundary diverges; a step small enough for 6e8 freezes
the healthy chains. There is no single step size that serves both.

**2. Past `u ~ 18` the log-probability is exactly `-inf`.**
`mx.sigmoid(18.0)` is exactly `1.0` in float32, so `log1p(-sig)` is
`log(0)`. `chees.py::_finish` then computes `dH = -inf`, sets
`diverged = ~isfinite(dH)`, and `accept = ... & ~diverged`. A proposal that
would move the chain from `-inf` to a finite value is therefore **rejected
as a divergence**, permanently.

## Suggested fixes, in order of value

1. **Accept any finite proposal from a non-finite current state.** In
   `_finish`, `diverged` should not veto a move whose *current* log
   probability is `-inf`. One line, and it makes `-inf` starts recoverable
   rather than absorbing.
2. **Compute `log(sig) + log1p(-sig)` without saturating.** It equals
   `-|u| - 2*log1p(exp(-|u|))`, which is finite and accurate for every `u`
   in float32 and removes the `-inf` region entirely.
3. **Consider a per-chain step size**, or clamping the initial state into
   the region where the Jacobian gradient is representable. The first two
   fixes make the boundary survivable; neither makes ChEES able to *sample*
   a chain whose gradient is 1e6 times its neighbours'.

## What anvil-gp does about it

Nothing, because nothing here is anvil-gp's. Its own contract holds
throughout: the GP log-likelihood and its gradient stay finite at every
point tested, including where the transform has already gone to `-inf`, and
a stuck chain does not affect its neighbours. That is pinned by
`tests/test_calibration.py::test_a_stuck_chain_does_not_poison_the_ensemble_under_chees`.

The gradient-free stretch move has neither problem and recovers from the
same starts; see the recovery test beside it.
