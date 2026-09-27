# Plan: per-instrument jitter, exposure-time integration, and the compile split

Status: proposed 2026-09-27, not locked. Three items, deliberately in this
order: one certain, one substantial but contained, one whose obvious
approach is already measured not to work.

Everything below rests on experiments run before writing, recorded here so
the estimates are not guesses.

## What was measured first

**Exposure-time integration does not fit the current state-space
construction.** celerite2's `TermConvolution` does exactly two things: it
rescales the celerite amplitudes and adds a constant to the diagonal. The
*rates* are untouched — verified identical to machine precision — and for
lags at or beyond the exposure width the convolved kernel is exactly a sum
of exponentials with the modified amplitudes (agreement 4.4e-16). So it is
celerite-class and therefore a state-space model.

But it is not *this* library's state-space model. anvil-gp builds every
term with `P_inf = I` and the amplitude carried in `h`, which forces the
ratio `b/a = c/d`. The convolution breaks that ratio: at `delta = 1.5`
against `rho = 4`, `b/a` goes from 0.169 to 0.348 while `c/d` stays
0.169. Fitting a free two-vector `h` over the existing SHO basis to the
convolved kernel leaves a residual of 6.7e-2 against a kernel of
amplitude 1.04 — **6%, not roundoff**. A general complex amplitude pair
needs a general `P_inf`.

**The obvious term-matrix split does not work.** The stated fix has been
"split the VJP at the term-matrix table", and `_metal_matrices` already
wraps `Term.matrices` in a `custom_function` when the front end is
chunked. Forcing that wrapper on unconditionally and compiling
`value_and_grad` fails *identically to the baseline*, with the same fused
kernel name. The over-fused group is a long elementwise chain inside the
term algebra itself, and a `custom_function` boundary around the whole
thing does not break it. Whatever the fix is, it is not this.

**One correction to an earlier claim.** `docs/float32.md` and the M4 stage
R notes say the compile failure is an uncatchable process abort. In this
probe it surfaced as an ordinary Python `RuntimeError` and the process
continued. Either MLX changed or the characterisation was wrong; the docs
should be re-checked when this work starts.

## Stage 1 — per-instrument jitter in the kernel

Supported on the MLX path today and refused by `metal_eligible()`. The
host already computes per-datum jitter (`ParamMap.jitter_var(v, idx)`) and
already carries an instrument index (`_inst32`).

What changes:

* The forward kernel reads `jit2[chain * n_jit]` once per thread, outside
  the observation loop (`metal.py:114`). It becomes a per-observation read
  indexed by the instrument, `jit2[chain * n_jit + inst[gi]]`, in the
  forward and in both replays of the backward.
* The backward accumulates one `drv` scalar per `(chain, block)`. It needs
  one per `(chain, block, instrument)`. `n_jit` is a handful, so a
  per-thread `float drv_acc[MAX_JIT]` with a compile-time cap is enough —
  no flush-on-change machinery required.
* `metal_eligible()` drops the one-jitter refusal and gains a
  `n_jit <= MAX_JIT` one.

| gate | |
|---|---|
| correctness | kernel gradient matches `GPPolicy(metal=False)` to 1e-5 and float64 central differences to 1e-6, with 1, 2 and 4 instruments, on every cadence in `_fwd_setup` |
| the jitter gradient specifically | each instrument's jitter adjoint matches; a test that permutes the instrument labels must fail |
| no regression | single-jitter timings unchanged in an interleaved A/B; the one-jitter path must not pay the per-observation read |
| cost | one extra indexed global read per observation in three kernels |

Half a day. The risk is low and the failure mode is visible: a wrong
instrument index gives a wrong value, not just a wrong gradient, because
the jitter is inside the innovation variance.

## Stage 2 — a general complex term, which is what exposure time needs

This is the substantial one, and it buys three things at once because all
three are the same underlying gap:

* `ComplexTerm(a, b, c, d)` — the general celerite term.
* `OriginalCeleriteTerm` — a reparametrisation of it.
* `TermConvolution` — exposure-time integration, which is `ComplexTerm`
  with rescaled amplitudes plus a diagonal constant.

### 2a. Generalise `P_inf`

Today every term returns `(Phi, Q, h)` with `P_inf = I` implied, and the
first observation's absent predecessor is table row 0 holding
`Phi = 0, Q = I`. Generalising means a term may declare its own `P_inf`,
and row 0 holds `Q = P_inf`.

**The kernel needs no change.** Its only `P_inf = I` assumption is that
row 0, which is built on the host (`likelihood.py:750`). The kernel reads
whatever the table contains. That is the single most important finding
for this estimate: a general `P_inf` is a front-end change and the fused
kernel and its hand-derived gradient come along unchanged.

Gate: every existing term, re-expressed through the general path,
reproduces its current `Phi`, `Q`, `h` bit-identically in float64 and its
`value_np` to 1e-12. The whole suite passes with no tolerance change.

### 2b. `ComplexTerm`

`k(tau) = e^{-c tau} (a cos d tau + b sin d tau)`, two states, rates `c`
and `d`, amplitudes `a` and `b` carried in `P_inf`.

Gate: float64 parity with `celerite2.terms.ComplexTerm` to ~1e-11 over a
grid including the degenerate corners (`d -> 0`, which is a real term;
`b -> 0`; `c -> 0`, undamped). The existing `SHOTerm` at `Q > 1/2` must be
reproducible as a `ComplexTerm` to 1e-12, which is the cross-check that
the parametrisation is right.

### 2c. `TermConvolution`

Wraps any term: take its celerite coefficients, apply the closed-form
amplitude rescaling, and add the diagonal constant to `r_var`.

Two things to get right, both already visible in celerite2's source:

* **The diagonal correction is not optional.** For lags below the exposure
  width the kernel is *not* the sum of exponentials, and celerite2 handles
  this by adding a scalar to the diagonal. That scalar goes exactly where
  anvil-gp already puts `r_ref` and the jitter, so it costs nothing new.
* **The exposure width must not exceed the smallest non-zero spacing.**
  Otherwise a real lag falls inside the window where the sum-of-
  exponentials form is wrong. Physically that is automatic for one
  instrument, since you cannot sample faster than you expose. Two
  instruments observing simultaneously with different exposure times is
  the genuine edge case and should be **refused**, not approximated.

Gate: float64 parity with `celerite2.terms.TermConvolution` to ~1e-11
across `delta/rho` from 1e-3 to 1 and across the kernels it wraps;
`delta -> 0` must recover the unconvolved term to 1e-12; a cadence with a
spacing below `delta` must raise with a message naming the offending
spacing. Plus the float32 kernel path against the float64 oracle at the
usual tolerances, and a gradient check, since `delta` is data and not a
sampled parameter but the rescaled amplitudes are functions of the
sampled ones.

Two to three days. The risk is concentrated in 2a: `terms.py` holds the
cancellation-free special functions that are the most delicate numerics in
the package and are pinned against mpmath, so the gate for 2a is
bit-identical reproduction of what exists rather than "close enough".

## Stage 3 — the compile split, as an investigation with a go/no-go

The prize is removing the process-global `mx.disable_compile()` that
ChEES-HMC currently needs. The measured speed prize is small: the term
front end's backward is **32% of a gradient call at N = 1e4 and 5% at
N = 1e5**, and the one time a split was tried it bought 1.15-1.25x.

Since the `custom_function` boundary is now known not to work, stage 3
starts as a bounded investigation:

1. Bisect the fused group. The failing kernel name is a chain of
   `Broadcast/Negative/Multiply/Add/Divide/Sign` — elementwise, and
   `Sign` narrows it. Find which expression in `terms.py` produces it.
2. Try, in order: restructuring that expression to fewer live inputs;
   `mx.stop_gradient` on a subexpression that does not need a gradient;
   splitting the term algebra into two functions with an explicit
   materialisation between them.
3. **Go/no-go.** If none of those makes `mx.compile(value_and_grad)`
   succeed within a day, stop and write it up as an MLX limitation with a
   minimal reproducer, which is worth more than a partial workaround. The
   `RuntimeError` observed above makes a clean upstream report easy.

Half a day to one day, capped. I would not start this before stages 1 and
2, and I would not be surprised if it ends in a bug report rather than a
fix.

## Order, and why

Stage 1 first: certain, contained, and it removes a limitation a real
multi-instrument radial-velocity user hits immediately. Stage 2 second: it
is the largest capability gain in the list, closes three documented gaps
at once, and the kernel comes along for free. Stage 3 last, capped, and
explicitly allowed to fail.

Each stage lands as its own commit with its gates met, the docs
(`kernels.md`'s supported and unsupported tables, `benchmarks.md` if any
number moves) updated in the same commit, and the README's "What is left"
entry retired as each item is done.

## What this plan does not do

* It does not close the >=100x forward gate. Nothing here touches forward
  speed.
* It does not add `TermProduct`. Products of celerite terms are not
  state-space representable at fixed `m`.
* It does not raise the `m <= 4` kernel ceiling, which is separate work
  and is what a rotation-plus-granulation model actually needs.
