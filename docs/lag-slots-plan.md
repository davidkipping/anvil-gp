# Plan: the kernel gradient's per-lag buffer

Status: stage 0 done 2026-09-22; stages A-D revised by what it measured, not locked.

## The problem

The backward kernel accumulates the propagator adjoints `dPhi`, `dQ` per
distinct lag, flushing its register accumulator whenever the lag changes
into a buffer with one slot per `(chain, block, lag)`. That buffer is
`2 x chains x blocks x lags x m^2` floats. Every distinct float32 spacing
in the timestamps is a lag, so on a regular cadence the buffer is
megabytes and on an irregular one it is N-sized: 2.2 GiB at 64 chains x
N = 2e4, 40 GiB at 1024 chains x N = 1e5. Stage R of M4 fenced it at
`metal.MAX_LAG_SLAB_BYTES` (2 GiB). This plan removes the need for the
fence.

## What this plan is, and is not

It is a **capability** change, not a performance one. On the cadences the
README's numbers are measured on (one cadence plus a few gaps) the buffer
is ~2 MB today, so the plan changes neither speed nor memory there. It
matters only on cadences with many distinct spacings, where the kernel
gradient today refuses or allocates gigabytes. After it, the gradient's
buffer is never larger than the forward's own propagator tables.

## The correction to the earlier proposal

The README's "What is left" entry (and the stage R notes) proposed handing
the kernel per-parameter Jacobian tables of the term matrices, so that the
thread accumulates `n_params` scalars instead of an `m x m` matrix per
lag. That is the wrong fix: the Jacobian tables are the propagator table
times `n_params`, which on the cadences that matter is larger than the
buffer they replace, and they require derivatives of the special functions
in `terms.py`, the most delicate numerics in the package. Retract it.

## The design: one slot per (block, lag) pair that occurs

The flush-on-change scheme is kept. What changes is the slot it flushes
to: one per **distinct `(block, lag)` pair that actually occurs**, which is
at most `min(blocks x lags, N)`. Measured at N = 1e5, 13 blocks, 1024
chains, m = 2 (script in the stage R scratch, reproduced in stage A's
test):

| cadence | lags | slots | today's slab | slot buffer | forward's own tables |
|---|---:|---:|---:|---:|---:|
| regular, one gap | 4 | 15 | 2 MB | 0.5 MB | 0.5 MB |
| regular, 20 gaps | 4 | 26 | 2 MB | 0.8 MB | 0.5 MB |
| alternating duplicates | 3 | 27 | 1 MB | 0.8 MB | 0.4 MB |
| two interleaved cadences | 17 | 53 | 7 MB | 1.7 MB | 1.1 MB |
| every spacing distinct | 99,890 | 99,992 | 39.6 GiB | 3.05 GiB | 3.05 GiB |

The last row is the floor: on an all-distinct cadence the forward already
pays 3 GiB (12 GiB at m = 4) for its per-chain tables, because the table
design gives each chain its own row per spacing. The gradient will cost
what the forward costs. Binning the spacings is what makes such a cadence
cheap, and that changes the likelihood, so it stays the user's decision
and is not in this plan.

A per-run layout (one slot per contiguous run of one lag, no zeroing, no
read-modify-write) was the other candidate. It is 3 GiB on alternating
duplicates, which two-instrument simultaneous photometry produces.
Rejected.

Three host-side int32 arrays, computed once in `GPLogLike.__init__` next
to `_metal_lag`, from the lag array and the resolved `metal_block`, in
numpy:

* `slot_id[t]`, N entries: the slot observation `t` flushes to.
* `slot_start[b]`, `n_blocks + 1` entries: block `b` owns slots
  `slot_start[b] .. slot_start[b+1]`.
* `slot_lag[s]`, `n_slots` entries: the lag each slot belongs to, sorted
  by `(lag, block)` so the per-lag sum is a contiguous segment.

Kernel (`_SRC_BWD_CKPT`): zero the block's own slot range instead of all
`n_lags` slots; flush to `(chain * n_slots + slot_id[gi]) * m^2` instead
of `(ei * n_lags + lag) * m^2`. Two address changes; the accumulator, the
flush count and the arithmetic per step are unchanged.

VJP (`scan_loglike`): `d_phi_tab = zeros(n_chains, n_lags, m, m).at[:,
slot_lag].add(d_slots)`, then MLX's existing chain rule through
`term.matrices` carries it to the parameters exactly as today. Checked
that MLX 0.32.2's `at[].add` runs on the GPU with repeated indices.

## Stage 0, measured 2026-09-22 -- and what it overturned

Stage 0's job was to pin the baseline before changing anything. It did,
and the baseline says **this plan targets the smaller half of the
problem**. Recorded here rather than quietly rewritten, because the
earlier version of this document (and what I told the user) claimed the
slot layout would make the gradient's buffer "never larger than the
forward's tables".

Done in stage 0: two cadences added to `tests/test_metal.py::_fwd_setup`
and to the two per-lag gradient tests -- `distinct` (every spacing its own
float32 value, so one table row per observation) and `alternating` (two
lags, run length one, so the flush-on-change accumulator flushes at every
observation). 80 cases pass on today's code with no tolerance change.

**A fixture lesson.** The first `distinct` drew times uniformly over a
short span, which made the spacings 37x tighter *relative to the
correlation time* than `regular`'s as well as all-distinct. Two changed
properties, one fixture. At `dt/rho ~ 1e-3` the `dh` adjoint degrades to
7.4e-5 at m = 4 on the **regular** cadence too, so the failure I first saw
(1.25e-4, over the 1e-4 tolerance) was mostly the tight spacing, not the
degenerate table. `distinct` now has `regular`'s mean spacing and differs
in exactly one property; it passes at the existing tolerance. The tight
spacing corner is a separate, untested ill-conditioning worth its own
work; it is not this plan's.

### The baseline, measured (M2 Max, SHO m = 2 unless noted)

| cadence | N | chains | lags | today's slab | forward peak | gradient peak |
|---|---:|---:|---:|---:|---:|---:|
| regular | 2e4 | 64 | 3 | 0.000 | 0.005 | 0.021 |
| distinct | 2e4 | 64 | 20,001 | 0.381 | 0.516 | **2.165** |
| distinct | 2e4 | 256 | 20,001 | 1.526 | 1.138 | 6.322 |
| regular | 1e5 | 1024 | 3 | 0.001 | 0.387 | 1.613 |
| distinct | 1e5 | 256 | 100,001 | 9.918 | 3.407 | refused |
| distinct | 1e5 | 1024 | 100,001 | 39.673 | 13.236 | refused |

All GiB. The 2.165 reproduces the stage R reviewer's 2.17 independently.

### The finding: the per-lag slab is not the binding cost

On the distinct cadence the gradient's peak exceeds the forward's by far
more than the slab accounts for -- 1.65 GiB over at N = 2e4, 64 chains,
of which the slab is 0.38. The rest is **MLX's backward through the term
front end**, `term.matrices(params, uniq_dt)`, measured on its own:

| N | chains | m | Phi+Q tables | front end, forward | front end, backward |
|---:|---:|---:|---:|---:|---:|
| 2e4 | 64 | 2 | 0.038 | 0.472 | **1.775** |
| 1e5 | 256 | 2 | 0.763 | 2.838 | **19.54** |
| 1e5 | 1024 | 2 | 3.052 | 10.21 | **74.75** |
| 1e5 | 512 | 4 | 6.104 | 20.42 | **55.76** |

24x the table it produces, because the term algebra's ~20 intermediates
are each of table size and MLX retains them all. At production size it is
**74.7 GiB against the slab's 39.7**. Removing the slab entirely would
take an all-distinct gradient at 1e5 x 1024 from ~115 GiB to ~75 GiB:
still far past the machine, still refused. The slab fix alone does not
make irregular cadences work.

On a regular cadence the front end's backward is 0.009 GiB, so none of
this touches the workloads the README measures. The problem is specific to
an N-sized lag table, where the table design's premise (few rows) is gone.

### Two candidate fixes, both tested

**`mx.checkpoint` on the front end: no effect.** Identical peak to three
decimals at every size (1.775 vs 1.775, 55.76 vs 55.76), gradient
bit-identical. It recomputes but does not reduce retention here.

**Chunking the front-end backward over the lag axis: 7-15x, and exact.**
Split `uniq_dt` into chunks, run the VJP per chunk against that chunk's
slice of the table cotangent, sum the parameter gradients:

| N | chains | m | chunks | whole | chunked | gradient rel. diff |
|---:|---:|---:|---:|---:|---:|---:|
| 2e4 | 64 | 2 | 16 | 1.818 | **0.126** | 2.0e-07 |
| 1e5 | 256 | 2 | 32 | 20.39 | **2.325** | 2.3e-07 |
| 1e5 | 512 | 4 | 32 | 61.24 | **8.681** | 2.7e-07 |

One subtlety the first attempt got wrong, worth writing down: `h` does not
depend on the lag, so its cotangent must be charged to one chunk only.
Charging it to every chunk multiplies that part of the gradient by the
chunk count -- a 9e-2 relative error, large enough to see and small enough
to mistake for conditioning.

### Revised plan

Both changes are needed and they are independent. Front-end chunking is
the larger term, is pure host-side Python, and touches no kernel; the slot
buffer is the kernel half. Do them in that order, and stop after the first
if the second does not pay.

| # | Stage | Days | Model | Gate |
|---|---|---|---|---|
| 0 | Pin the baseline | 0.25 | Opus | **DONE** -- above. Two cadences in the suite, 80 cases green, baseline and the front-end split measured. |
| A | Chunk the front-end backward | 0.75 | Opus | **DONE** -- see below. 2.3x less peak and 5-6x faster at N = 1e5; regular cadences bit-identical; gradient within 1.7e-7. |
| B | Host slot map | 0.25 | Opus | Every observation's `(block, lag)` pair maps to one slot; counts on the five cadences match the table above. |
| C | Kernel and VJP on slots | 0.5 | Opus | Per-lag adjoint sums equal the pre-change kernel's to 1e-6 at m = 1..4 on all five cadences. The fence, re-based on slots, stops tripping. |
| D | Measure and document | 0.25 | Opus | All-distinct gradient runs at 1e5 x 1024 within ~2x the forward's peak. `benchmarks/verify_readme.py` green. README, plan and memory corrected. |

Opus throughout. The scatter-add's determinism question from the previous
draft still applies to stage C; the kernel itself is bit-reproducible
across runs (verified, 4 runs identical), so only the new scatter is in
question.

### Expected outcome, revised

An all-distinct cadence at N = 1e5 and 1024 chains needs ~115 GiB of
gradient peak today and is refused. Stage A takes it to ~55 GiB, stages
B-C to roughly 20-25 GiB against a forward that is itself 13 GiB. That is
the difference between refused and runnable. On the regular cadences every
README number is measured on, nothing changes: no speed difference, no
memory difference, both halves of the cost are already negligible there.

Still out of scope, and still the real answer for very irregular data:
binning the spacings, which shrinks the table at the cost of changing the
likelihood, and so stays the user's decision.

## Stage A, measured 2026-09-22

`GPLogLike._metal_matrices` wraps the term front end in a
`custom_function` whose VJP splits the backward over the lag axis, one
chunk at a time, when `_front_end_chunks` says the table is large enough
to need it. Chunk count is budget-driven: an estimate of the backward's
peak (48x the table, the conservative end of a measured 25-47x) divided by
a 512 MB target, so a regular cadence always gets exactly one chunk and
takes the original code path.

### Measured, whole gradient path, one window

| cadence | N | chains | lags | chunks | peak before | peak after | time before | time after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regular | 1e4 | 1024 | 2 | 1 | 0.167 | 0.166 | 8.7 | 8.5 |
| regular | 1e5 | 1024 | 2 | 1 | 1.612 | 1.612 | 78.9 | 75.9 |
| regular (m=4) | 1e5 | 512 | 2 | 1 | 1.301 | 1.301 | 404 | 418 |
| distinct | 2e4 | 64 | 19,964 | 4 | 2.174 | **0.970** | 74.8 | 76.7 |
| distinct | 2e4 | 256 | 19,964 | 15 | 6.327 | **2.781** | 298 | 300 |
| distinct | 1e5 | 256 | 99,083 | 73 | 31.30 | **13.67** | 11,386 | **1,813** |
| distinct | 1e5 | 512 | 99,083 | 146 | 61.66 | **27.30** | 27,547 | **4,868** |

GiB and ms. The two largest rows needed `MAX_LAG_SLAB_BYTES` lifted to
measure at all, since the kernel's own buffer refuses there first -- which
is stages B-C. Regular-cadence gradients are **bit-identical** (relative
difference exactly 0) and within timing noise; the chunked branch is not
entered. On the distinct cadence the gradient agrees with the unchunked
one to 1.7e-7 over four terms.

At N = 1e5 the change is 2.3x smaller *and* 5-6x faster, because the
unchunked version at 31-62 GiB is thrashing. At N = 2e4 it is 2.2x smaller
at no time cost.

### Two things that went wrong, both worth keeping

**`mx.eval` inside the chunk loop makes it 3.6x worse.** Forcing each
chunk to completion looks like the way to free it before building the
next. It does the opposite: it materialises the upstream graph -- the
backward kernel and its per-lag slab -- and pins it for the whole loop.
Measured 7.76 GiB against 2.17 unchunked, where leaving it lazy gives
0.97. MLX frees each chunk's intermediates as the sum is built.
`tests/test_metal.py::test_chunking_the_front_end_lowers_the_gradients_peak_memory`
fails if the eval comes back.

**The first correctness test was vacuous.** At test size the shipped
budget picks one chunk, so the comparison passed with the whole branch
unexercised -- and hid a crash: `mx.custom_function` hands a
single-argument VJP the primal array itself rather than a one-tuple, so
`primals[0]` silently took its first row. Any irregular cadence large
enough to chunk would have raised. The test now lowers the budget to force
the split and asserts the chunk count is above one before comparing.

### What is left after stage A

At 256 chains x N = 1e5 the gradient's peak is 13.67 GiB, of which the
kernel's per-lag slab is 9.83 -- 72%. Stages B-C address that and would
bring the same problem to roughly 4 GiB. The gate for stage A said "peak
down >= 5x at N = 1e5"; the achieved figure is 2.3x, because that gate was
set from the front end measured *in isolation* (7-15x) and the whole path
has the slab as its floor. Recorded rather than restated.
