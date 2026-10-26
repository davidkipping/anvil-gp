*Draft, not sent: to be appended to `WASP103b_CHEOPS/docs/upstream/anvil-gp_reply.md`.*

## Update, 2026-10-25

A second review of our plan changed how three of the promised items are
built. Nothing is dropped, and section 1 (the multi-input kernel) is
unchanged.

**The error scale is now a general noise model, and it is done.**
Instead of a `jitter="scale"` option, `GPLogLike` takes a `noise_fn` (the
total per-point noise variance, batched over chains, with gradients on
every path, the fused kernel included), and `make_target` takes a ready
`noise=` model. Your B22 scale, with or without an added jitter:

```python
gt = make_target(term, t, y, yerr, mean_fn=..., x_mean=..., mean_specs=...,
                 noise=ag.noise.error_scale())          # (s yerr)^2
# or noise=ag.noise.scale_and_jitter(yerr)             # (s yerr)^2 + j^2
```

`s` is log-uniform on [0.1, 10] by default (`error_scale(lo, hi)` to
change it), and both take `n_instruments=` for one per instrument. For
anything else -- flux- or time-dependent noise -- write your own
`noise_fn(v_noise, d)`, with `d.yerr`, `d.x` and `d.inst` the per-point
arrays; `docs/kernels.md` ("Noise models") has the two rules it must
follow. The default stays σ² + jitter², so nothing changes unless you
pass it. Per-instrument jitters are also no longer capped at eight.

**The joint target's home may move to anvil.** Summing several datasets
under one target with shared parameters is not GP-specific, so we are
proposing it to anvil as a general feature. anvil-gp's side is making its
targets well-behaved members. If anvil does not take it, we build it here
as planned. Either way the promise in section 1 holds: a member can be
any anvil log-density, so dense likelihoods of your own for the seven
multi-input visits would plug in.

**Marginalisation sits on a general primitive.** We will first build a
batched, differentiable `apply_inverse(Y)` (K⁻¹Y for any columns Y, as in
celerite2) and `gram(Y)` (YᵀK⁻¹Y). `linear_basis` marginalisation and
`predict` are thin layers on top. What you asked for is unchanged: the
same restricted-likelihood formula with the −½ ln|ΦᵀK⁻¹Φ| term,
`linear_posterior(v)`, and the same acceptance tests.

**Order and timing.** `noise_fn` (done) → joint-target members (and the
anvil proposal) → `apply_inverse` and marginalisation → `predict` → docs.
About 8-12 working days in all, up from 6-9; the extra is doing each piece
in its general form.

**Interim, unchanged in substance.** The five 1-D visits work on anvil-gp
now, with β sampled, and with the multiplicative scale as above;
marginalisation follows. The joint 12-visit
target arrives with the joint-target item, from anvil or from us.
