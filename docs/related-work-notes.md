# Related work: reading notes

Short assessments of papers checked against this project's problem, so the
same ones are not re-read from scratch. The question each is judged on: does
it help evaluate an **exact** celerite-class marginal likelihood for
**thousands of parameter vectors at once**, with gradients, in float32 on
Apple Silicon?

## LightGP (arXiv 2605.17898v1)

*"LightGP: Lightweight Gaussian Process Inference in C++ on Metal and CUDA"*,
Yu-Hsueh Fang (NTU / Georgia Tech). Read 2026-09-07.

A dependency-free C++17 GP regression library with Python bindings and
hand-written Metal and CUDA backends, pitched against GPyTorch/GPflow's
framework overhead. Four inference paths: exact Cholesky, conjugate
gradients, sparse variational free energy, and structured kernel
interpolation with an FFT-accelerated matvec. Reports 2.6–8.7x over GPyTorch
on a Mac M4 CPU and 2.3–6.7x on CUDA for exact GP, with exact inference
topping out near N ≈ 42,000 on a 16 GB Mac.

**Verdict: not applicable, for three structural reasons.**

1. **Unbatched.** LightGP solves one GP at a time. Our entire premise is
   thousands of *different hyperparameter vectors* per call — that is what
   makes the GPU worth using, and it is the axis LightGP does not have.
2. **No gradients.** Its limitations section: "Kernel hyperparameter
   optimization uses finite differences; analytical gradients are available
   only for the legacy single-kernel API." Gradient-based MCMC needs a real
   VJP.
3. **Wrong algorithm family.** No celerite kernels, no state-space models, no
   Kalman filtering, no parallel/associative scans. Its scalable paths are CG
   and SKI, both of which this project rejects on the exactness rule (see
   [statistical-foundation.md](statistical-foundation.md)): stochastic
   log-determinant estimators make the likelihood noisy, which breaks
   Metropolis and is not a valid pseudo-marginal scheme. Its O(N^3) exact
   path is what our O(N) recursion exists to avoid.

It is also float32-only with no float64 verification path, so it has nothing
to say about the precision discipline that dominates our design.

**Two things worth keeping.**

*The AMX result, with a caveat.* Table 2 reports that Accelerate's `spotrf`
on Apple's AMX coprocessor is **9–27x faster than their Metal Cholesky**
(N=512: 0.12 ms vs 3.2 ms; N=4096: 41.5 ms vs 364 ms), and they attribute the
widest gap at small N to "Metal dispatch overhead". That is directly relevant
to the small-N dense backend this project scopes for RV datasets (N <~ 1e3) —
but the caveat matters: those are **single** factorisations, where a GPU has
nothing to fill itself with. Our small-N case is ~1024 *independent*
factorisations at once, which is precisely the regime where the GPU should
win and the AMX comparison does not transfer. Worth measuring both ways
before writing that backend; worth not assuming Metal loses.

*Independent confirmation of the diagnosis.* Their motivation — that
Python/deep-learning-framework dispatch overhead dominates real GP workloads —
is the same conclusion we reached by profiling, where MLX op count rather than
bandwidth or arithmetic was the binding constraint (we measured ~8 GB/s
effective against ~400 GB/s of hardware). Their answer is to drop to
hand-written C++ and Metal. That is the same answer as this project's
Milestone 3, arrived at independently on a different algorithm.
