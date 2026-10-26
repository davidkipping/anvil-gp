# Reply to WASP103b_CHEOPS: dense multi-input GP, noise scale, marginalisation

**Draft, 2026-10-25 -- not sent.** Goes to
`WASP103b_CHEOPS/docs/upstream/anvil-gp_reply.md` on sign-off. Answers
their brief of 2026-10-24 (`anvil-gp_prompt.md`).

## Summary

- **The multi-input Matérn-3/2 is out of scope for anvil-gp**, and will
  stay so: the library is an O(N) GP library, and no kernel of a distance
  over several inputs has an O(N) form. Below: why, and what your options
  are.
- **The 1-D visits work today**, verified on your visit 8.
- **The rest of the brief is general and we will build it:** a
  multiplicative error scale, marginalised linear mean parameters, joint
  multi-dataset targets with shared parameters, and batched O(N)
  prediction. Roughly 6-9 working days; order and interim options below.

## 1. Why anvil-gp cannot take a multi-input kernel

anvil-gp is fast because every kernel it supports is *semiseparable* in
one ordering of the data: once the points are sorted, the covariance
between two of them factorises into (a function of the later point) x (a
function of the earlier point), with a fixed small number of terms. For
the exponential that is exp(-c|t_i - t_j|) = exp(-c t_i) exp(c t_j) for
t_i >= t_j, and SHO, Matérn, rotation and their sums and products are the
same thing with m terms. That factorisation is what lets a Kalman filter
walk the sorted data once with an m-dimensional state instead of
factorising an N x N matrix.

It needs a total order on the input, so that "which point is earlier" is
always defined. A distance over two or more inputs,
r² = (Δroll/ℓ₁)² + (Δbg/ℓ₂)², mixes the coordinates under a square root;
exp(-√3 r) does not factorise into a product of one-point functions for
any finite number of terms, and no ordering of points in a plane gives
the matrix low-rank off-diagonal blocks (however you cut the set in two,
points on opposite sides can be arbitrarily close). In state-space terms:
a celerite kernel is a Markov process in one variable; in two variables
the analogue is a stochastic *partial* differential equation (Lindgren et
al. 2011), which has sparse approximations but no exact O(N) filter.

Two things that look like escape routes are not:

- **A sum of 1-D kernels on different inputs**, k₁(roll) + k₂(bg): each
  term is semiseparable in its own ordering, the sum in neither.
- **Embedding roll angle as (cos φ, sin φ)** for wrap-around: a
  two-input kernel, so out for the same reason. A GP on raw roll angle
  cannot see 359° as adjacent to 1°; anvil-gp has no circular domain.

A dense O(N³) path would be a different library with a different speed
model; we are not adding one. `docs/kernels.md` will say this with the
reasoning above.

**What is O(N):** any number of kernels, summed or (celerite-style)
multiplied, on *one* sorted input -- which can be any sorted quantity,
not only time.

**Your options for the seven multi-input visits**, your call, not ours:

1. A dense tool for those visits: george as B22 used, or tinygp (JAX)
   for gradients. Their N ≤ 540 makes dense cheap in float64 on the CPU.
2. An O(N) model: a GP on one sorted input (roll angle, where B22 always
   has it) plus *linear* decorrelation against the others (background,
   centroids, contamination) as extra columns of the linear basis that
   item 3 below marginalises. That is a different model from B22's, not
   a reproduction of it, so it is a modelling decision for you.
3. A mixed fit, if you build the joint target on your side: anvil-gp for
   the 1-D visits, a dense likelihood for the others, summed. Our joint
   target (item 4) takes any anvil log-density as a member, so a dense
   one of yours would plug in.

## 2. Your questions

**Q1. Non-time sorted 1-D input.** Valid today. Checked on your visit 8
(N = 526, sorted by roll angle, 454 distinct spacings, 0.5-2.5°,
Matérn-3/2 with A = 3 rms and ℓ = 60° = your lower prior bound): the
fused kernel and the MLX path are within 2.5e-4 and 1.2e-3 of a dense
float64 Cholesky in float32, 9e-10 in float64 (`.hi`), and the float32
gradients equal float64 central differences to three decimals. With the
roll rounded to whole degrees to force 204 exact ties, the same. The
fused kernel applies (m = 2); many distinct spacings cost nothing at this
N (the lag table is one row per distinct spacing; the memory concern in
the docs is for N ~ 1e5). Pass `t` = the sorted input, `x_mean` = the
times in that order, exactly as you describe. A test of this
configuration (irregular non-time input with ties, both paths against the
oracle) joins our suite.

**Q2. Multiplicative jitter.** Not today: `jitter` is an added variance,
σ² + j². We will add an error scale, noise variance s²σ² + j², per
instrument like the existing jitter, on every path (fused kernel, MLX,
`.hi`). Item 1 below.

**Q3. Route and timeline.** None of routes 1-3: see section 1. For the
rest, the order and sizes are below; the 1-D path is yours to use for the
five 1-D visits now, with jitter additive and β sampled until items 1 and
3 land.

## 3. What we will build

All of it general; nothing here is specific to CHEOPS or B22.

1. **Multiplicative error scale** (1-1.5 days). `jitter="scale"` or a
   combined form; per instrument; on all paths with gradients.
2. **Marginalised linear mean parameters** (3-5 days). `linear_basis`
   giving Φ (chains, N, p), rebuilt from the mean function each
   evaluation; flat or Gaussian prior on β; the restricted-likelihood
   form with the -½ ln|ΦᵀK⁻¹Φ| term kept, as you say;
   `linear_posterior(v)` returning β̂ and its covariance. K⁻¹Φ is p extra
   right-hand sides through the same O(N) filter, on the fused kernel,
   the MLX path and `.hi`, with gradients. Acceptance as you state:
   brute-force quadrature over β to 1e-8, and gradients against finite
   differences.
3. **Joint multi-dataset targets** (1.5 days). `make_joint_target([...],
   shared=[...])`: several datasets, shared and per-dataset parameters,
   one anvil target summing the log-likelihoods, with `u0`, names,
   `natural()`, refusals and `.hi` combined. Members can be anvil-gp
   targets or any anvil log-density.
4. **Batched O(N) prediction** (1-2 days): conditional mean and variance
   at new inputs from a Kalman smoother, batched over chains. Separate
   from the likelihood path; it does not change inference cost.
5. **Docs**: the reasoning of section 1 in `kernels.md`, and the new
   features in the tutorial and README.

About 6-9 working days in that order. Item 2 is the one with design risk
(two ways to carry the extra right-hand sides through the fused kernel;
we measure before choosing), so its size is the least certain.

## 4. Interim

- Five 1-D visits on anvil-gp now, β sampled, jitter additive.
- After item 1: multiplicative scale.
- After item 2: β marginalised.
- After item 3: one target for all twelve, if the seven dense visits are
  provided as anvil log-densities from your side.

We will update `anvil-gp_reply.md` as items land.
