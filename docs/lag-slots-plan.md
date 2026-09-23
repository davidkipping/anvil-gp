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
`metal.MAX_ADJOINT_BUFFER_BYTES` (2 GiB). This plan removes the need for the
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
* `slot_lag[s]`, `n_slots` entries: the lag each slot belongs to. Slots
  are ordered by `(block, lag)`, **not** `(lag, block)` as an earlier
  draft of this document said: the kernel's thread has to zero its own
  block's slots, which requires that block's range to be contiguous. The
  per-lag sum is then a scatter rather than a segment sum. If the scatter
  proves non-deterministic, the fallback is a host-side permutation to
  lag order and a segment sum over it -- the ordering the earlier draft
  wanted, recovered without costing the kernel its contiguous range.

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
| B | Host slot map | 0.25 | Opus | **DONE** -- `likelihood._slot_map`, built from the resolved `metal_block`. Counts match the table exactly (15/26/27/53/99,992); invariants pinned; 13x fewer slots than `blocks x lags` on an all-distinct cadence, 27.6 ms to build at N = 1e5. |
| C | Kernel and VJP on slots | 0.5 | Opus | **DONE** -- gradients match the pre-change kernel to 1.5e-7 over 32 cases; the headline case runs at 1.3x the forward's peak; reproducibility kept, which took two attempts. |
| D | Measure and document | 0.25 | Opus | All-distinct gradient runs at 1e5 x 1024 within ~2x the forward's peak. `benchmarks/verify_readme.py` green. README, plan and memory corrected. |
| R2 | Adversarial review of the slot map | 0.5 | **Fable** | **DONE** -- see "Stage R2, measured": five defects fixed, one hardening, one coverage gap closed, and the suite's sensitivity measured under nine mutations. Original brief: assume the slot map is wrong and prove it. A wrong slot index gives a correct *value* and a silently wrong parameter gradient, because the forward is untouched -- the one failure class this suite has repeatedly missed on its own. Specifically: a slot map stale against `metal_block`; the scatter-add's determinism across runs; blocks whose lag set changes between value and gradient calls; and whether any test would fail if `slot_id` were permuted. |

Opus for the work, **Fable for R2**. That split is this project's own
evidence rather than a guess: stage F of M4, the hardest kernel work in
the milestone, was written by Fable and still shipped a value-correct,
gradient-dead bug that a full green suite did not catch. Stage R, also
Fable, is what found it -- along with five other defects and three tests
that passed with their subjects deleted. On this codebase the capability
has paid in review, not in authoring. If stage C turns out to need more
than the two address changes assumed here -- a deterministic fallback
designed for the scatter, or a restructure forced by register pressure --
it stops being mechanical and should move to Fable too.

The scatter-add's determinism question from the previous draft still
applies to stage C; the kernel itself is bit-reproducible across runs
(verified, 4 runs identical), so only the new scatter is in question.

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

GiB and ms. The two largest rows needed `MAX_ADJOINT_BUFFER_BYTES` lifted to
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

## Stage B, measured 2026-09-22

`likelihood._slot_map(lag, blk_len)` returns the three int32 arrays stage
C's kernel needs, built in `GPLogLike.__init__` from the **resolved**
`self.metal_block` rather than the policy field, which may be `None`. A
map built against a different block length would misattribute every
adjoint while leaving the value correct, since the forward never reads it.

Slot counts at N = 1e5, 13 blocks -- the table this plan rests on,
reproduced by the shipped code rather than a scratch script:

| cadence | lags | slots | buffer, per-lag | buffer, slots |
|---|---:|---:|---:|---:|
| regular + 1 gap | 4 | 15 | 2 MB | 0.5 MB |
| regular, 20 gaps | 4 | 26 | 2 MB | 0.8 MB |
| alternating duplicates | 3 | 27 | 1 MB | 0.8 MB |
| two interleaved cadences | 17 | 53 | 7 MB | 1.7 MB |
| every spacing distinct | 99,890 | 99,992 | 39.6 GiB | 3.05 GiB |

At 1024 chains, m = 2. The all-distinct row is 13x fewer slots than
`blocks x lags`, which is the block count: each observation's block-lag
pair is distinct, so the slot count is N rather than N x blocks.

Ordering is by `(block, lag)`, so each block's slots are one contiguous
range and a thread can zero exactly its own. An earlier draft said
`(lag, block)`; that would have made the per-lag fold a segment sum but
left the kernel's zeroing scattered, which is the wrong trade.

Construction costs 27.6 ms at N = 1e5 on the worst cadence and is a single
`np.unique`; on a regular cadence it is microseconds. Pinned by two tests:
one checks the counts and the invariants (every observation in exactly one
slot, each block's range contiguous and exclusively its own,
`slot_lag[slot_id] == lag`), the other that the map tracks the resolved
block length across `metal_block` of `None`, 256 and 1024.

Nothing reads the map yet -- that is stage C. 562 tests pass.

## Stage C, measured 2026-09-22

The backward kernel flushes into one slot per `(block, lag)` pair that
occurs. Two address changes, as the plan assumed: the thread zeroes
`slot_start[b] .. slot_start[b+1]` instead of every lag, and flushes to
`slot_id[gi]` instead of the lag. The VJP folds slots back into the
per-lag table the term front end differentiates.

### Correctness

Against the **pre-change kernel**, 32 cases (4 terms covering m = 1, 2, 3,
4 x 4 cadences x 2 block lengths): values bit-identical, gradients within
**1.5e-7**. The float64 specification tests pass unchanged at their own
tolerances.

### Measured, whole gradient path

| cadence | N | chains | slots | buffer before | buffer after | peak | time |
|---|---:|---:|---:|---:|---:|---:|---:|
| regular | 1e5 | 1024 | 14 | 0.001 | 0.000 | 1.61 | 68 ms |
| distinct | 2e4 | 64 | 19,995 | 0.38 | 0.04 | 0.73 | 114 ms |
| distinct | 2e4 | 512 | 19,995 | 3.05 | 0.31 | 2.24 | 675 ms |
| distinct | 1e5 | 256 | 99,907 | 9.92 | 0.76 | 4.40 | 1.55 s |
| distinct | 1e5 | 512 | 99,907 | 19.84 | 1.52 | 8.78 | 5.59 s |
| distinct | 1e5 | 1024 | 99,907 | 39.67 | 3.05 | 17.5 | 10.2 s |

GiB. The last row is the case the whole plan was for. It needed ~115 GiB
and was refused; it now runs at **1.3x the forward's own peak** (13.2
GiB), which meets stage D's gate a stage early. End to end against the
stage 0 baseline, at 256 chains x N = 1e5: **31.3 -> 4.4 GiB and 11.4 ->
1.5 s**, 7.1x smaller and 7.4x faster.

`MAX_LAG_SLAB_BYTES` is renamed `MAX_ADJOINT_BUFFER_BYTES` and raised from
2 to 8 GiB: it was sized when the buffer was the dominant cost, and at the
old value it refused the 1024-chain case above, whose buffer is 17% of a
run the machine handles.

### Reproducibility nearly went, twice

**MLX's scatter-add is not reproducible when indices repeat.** Folding
slots to lags is a scatter, and a lag repeats across blocks on every
regular cadence, so the obvious fold quietly cost the gradient its
bit-reproducibility -- on the *common* path, not the irregular one this
work is about. Measured directly: 14 slots over 3 lags is enough, while
`mx.sum` over an axis is reproducible. The per-lag adjoints used to come
from exactly such a sum.

A slot is a `(block, lag)` pair, so within a block every lag is distinct
and a lag repeats only across blocks. Ranking each slot within its lag
gives groups that each have distinct lags. Two ways to use that, and
**which is cheaper depends on the shape**:

* few ranks -- an irregular cadence, where a lag occurs in one or two
  blocks -- one scatter per rank, no intermediate;
* many ranks over few lags -- a regular cadence, 3 lags over 98 blocks --
  a padded `(n_lags, max_rank)` table and one `mx.sum`.

The first attempt used the padded table everywhere, which materialises
`n_chains x n_lags x max_rank x m^2`: 4.9 GiB of gather against a 0.8 GiB
buffer at 256 chains x N = 1e5, and the measured peak went 4.40 -> 8.77
GiB. `_MAX_FOLD_RANKS` now picks by shape. Both paths are reproducible;
`test_the_per_lag_fold_is_reproducible` fails if the naive scatter returns.

### A watchdog scare that was not ours

While testing, `m = 4` at N = 1e5 and 512 chains started dying with
Metal's "Impacting Interactivity" kill. Checking the **pre-stage-C**
commit on the same machine: it dies too, and at 128 chains where stage C
survives to 256. Pre-existing fragility, made marginally better here, and
sensitive to what else is on the GPU -- the same case ran at 512 chains in
404 ms during stage A on a quieter machine. Worth its own work; not a
stage C regression, and not chased further.

### Test hygiene

`test_the_slot_buffer_admits_a_cadence_the_per_lag_buffer_refused` first
asserted the production shape, needed ~20 GiB free, and failed once then
passed on retry. It now uses 512 chains x N = 2e4 against the old 2 GiB
fence, with the production figures recorded here instead. A flaky test
teaches people to re-run rather than to look.

570 tests pass.

## Stage R2, measured 2026-09-23

Three inputs. One independent Opus attacker with the brief above (~400
configurations over 5 terms, m = 1-4, 4+ cadences, block lengths 1-8192,
N = 1-20000; 1200 fuzzed slot maps; 56 reproducibility configurations).
Nine mutations of the shipped code, each run against the kernel suite to
ask the plan's literal question -- would any test notice? And a direct
demonstration of the one route to a stale map. Everything below is fixed
on `main` and pinned by a test that fails with the fix reverted.

**Verdict first.** The code as shipped could not be broken: the kernel
gradient agreed with the MLX path to <= 8.4e-6 and with float64 central
differences to <= 1.8e-6 everywhere, every call was bit-reproducible, and
no fuzzed map violated an invariant. The failure class the brief is about
-- correct value, silently wrong gradient -- did not occur in the code. It
occurred the moment `gp.metal_block` was assigned after construction,
which nothing defended against.

### Defects, all fixed

**1. `metal_block` was a plain attribute; the slot map built from it was
not rebuilt.** The map is indexed by block, so it is valid only for the
block length it was built with. Map built at 256 and launched at:

| launch | value | gradient |
|---:|---|---|
| 512 | exact | exact -- by scheduling luck (a race that never bit) |
| 1024 | exact | **wrong by 2.6x**, finite, silent |
| 128 | -- | read past the end of the block table; **hung the process at 0% CPU for 26 minutes** until killed, holding a GPU queue |

The hang is the worst of the three: it presents identically to the
pre-existing m = 4 watchdog fragility and would be misdiagnosed as it.
Fixed: `metal_block` is a property whose setter validates and rebuilds
the map and fold plan, plus a host-side shape check at launch for anyone
who bypasses the setter. `GPPolicy` refuses `metal_block < 1`.

**2. The fold chose its dense path on `max_rank` alone.** The padded
gather is `n_chains x n_lags x max_rank x m^2`; the docstring reasoned
about the two extremes and the condition read neither. A mostly-regular
cadence with thousands of odd spacings -- dominant lag in every block,
every other lag in one -- has both `n_lags` and `max_rank` large and took
the dense path: measured **1.61x peak** at N = 2e4 (1113 MB against 692),
a gather 10x the buffer it folds, and 15.9 GiB extrapolated at the plan's
headline shape. This is the same intermediate stage C removed, back by a
different door. Fixed: dense only when `n_lags x max_rank <= 4 x n_slots`
as well.

**3. The fold's row count was `lag.max() + 1`, not the table's.** Equal
for every N >= 2; at N = 1 the table has two rows and the value one, and
the dense branch raises. Unreachable as shipped only because N = 1 has one
rank -- a memory heuristic accidentally guarding a shape bug. Fixed.

**4. The chunk count's 64-lags floor overrode its own budget.** Meant to
keep a chunk "worth a launch", it returned 1 chunk with the estimate 3x
over budget above ~1365 chains at m = 4, because a chunk's work is
`n_chains x lags x m^2` and few lags is plenty at many chains. Removed;
the count is bounded by one lag per chunk and nothing coarser.

**5. Both dedicated gradient tests used all-distinct cadences.** There,
any one lag carries ~1/1500 of the gradient, so a per-lag mis-routing
worth **6.1e-2** on a gapped cadence read as 2.9e-4 and passed. The
attacker's two-slot swap in the fold survived **563 of 570 tests**. Both
tests are now parametrised over the gapped cadence too -- the README's
benchmark cadence and every space light curve -- and the same mutation
fails all five gapped cases while still passing all five random ones,
which is the mechanism confirmed rather than assumed.

**Hardening, no defect found:** the backward refuses index arrays that are
not int32. Metal reads them as 4-byte ints; a wider dtype indexes garbage
with the value still right.

### What the suite noticed, mutation by mutation

| mutation | what it corrupts | caught by |
|---|---|---|
| global permutation of `slot_id` | lag invariant and block ownership | 7 tests: the gradient tests and the structural map test |
| relabelling within each block's range, fold updated | nothing -- must be harmless | only the structural equality test; every gradient test passes, so the kernel does not depend on slot order |
| `metal_block` halved after construction | map stale, shrunk direction | **hung**; see defect 1 |
| `slot_start` shifted by one | each block zeroes its neighbour's first slot | only the structural test -- the cross-thread race it introduces never bit in 155 runs, so a zeroing race is invisible to numerical tests and the structural invariant test is load-bearing |
| buffer written slot-major, read chain-major | layout | 68 tests |
| kernel never zeroes | stale data across calls | 73 tests, including both reproducibility tests |
| fold routes every slot to lag 0 | everything | 6 tests |
| two slots swapped between lags (the attacker's) | one lag pair | **7 of 570 before the fix**; 5 of 5 gapped cases after |

### Attacked and held

Host/launch consistency by every route but the mutable attribute (`npts`
pinned to `n_data` on all three paths; `n_blocks` computed identically in
three places; `_auto_metal_block` called once; a block's lag set is a
function of the data alone). Every host invariant over 1200 fuzzed maps.
Degenerate shapes: N = 1, 2, all timestamps equal, `metal_block` > N,
`metal_block` = 1, gaps and duplicates at and around block boundaries, 96
ragged-block combinations (worst 7.9e-5 at blk = 1, m = 3). Both fold
branches forced against each other: 12 of 16 bit-identical, worst 1.9e-7;
sentinel row cannot alias a slot; an absent lag gets an exact zero. The
chunked front end at 31 forced chunks with jitter, one and two instrument
offsets, a mean model, all four terms and three `TermSum`s: <= 7.6e-7
against unchunked; the `h`-once rule confirmed sensitive (charging it per
chunk is a 17x error). `mx.compile` of the forward with chunking active:
0 ulp. Failed lanes: five infeasible points on four cadences, chunked and
not, every rejected lane exactly zero, every healthy lane bit-identical
alone and in the mixed batch. Reproducibility: 56 configurations, `max|d|`
exactly 0. Kernel index arithmetic peaks 4x below uint32 overflow under
the 8 GiB fence.

### Not settled

Defect 2's 15.9 GiB at production shape is the formula evaluated, not
measured (the two measured points matched it within 3%). Whether the
watchdog kill in the shrunk-block case was purely the out-of-bounds read
-- a two-day PyTorch job and a second reviewer shared the GPU throughout;
the N = 512 repro shows chain-dependent garbage cleanly and is the
evidence to rely on. And `MAX_ADJOINT_BUFFER_BYTES` still refuses an
all-distinct RotationTerm above ~6.7e7 chain-observations, which is where
the forward's own tables also reach 8 GiB, so the policy docstring's
"sanity check rather than a limit" is defensible but was not run.

### The lesson for the next review

Both of R2's real findings were **interfaces, not arithmetic**: a public
attribute that a derived structure silently depended on, and a
two-variable decision written against one variable. The attacker's ~400
numerical configurations found nothing because the arithmetic was right.
The mutations and the attribute assignment found the defects in minutes.
Review the contracts between components before the numerics inside them.
