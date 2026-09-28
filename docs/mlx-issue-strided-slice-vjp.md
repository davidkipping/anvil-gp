# MLX issue: wrong VJP for a single-element strided slice — **already filed upstream**

**Do not file this.** It is [ml-explore/mlx #4444](https://github.com/ml-explore/mlx/issues/4444),
opened 2026-09-01 by a third party — five days *before* this write-up was
dated "ready to file" on 2026-09-06 and three weeks before anyone here
searched the tracker (Fable review, 2026-09-28). A fix is pending as
[PR #4446](https://github.com/ml-explore/mlx/pull/4446) ("fix: keep
single-element strided slice region round-tripping"), open, labelled
"low priority" / "await verification" by a maintainer on 2026-09-02, with no
substantive review as of 2026-09-28.

**Root cause, from the issue:** `normalize_slice` in `mlx/ops.cpp` collapses
the stride of a one-element result to 1 but does not adjust `stop`, so the
stored `(start, stop, strides)` triple describes an oversized contiguous
region. Two consumers re-derive the region from that triple and both go
wrong: `Slice::vjp` (scatters the cotangent over the whole span) and
`Slice::vmap` (wrong value *and* wrong shape), in `mlx/primitives.cpp`.
That also explains the one case that is right by accident: `z[0:1:2]`, whose
`start:stop` span is already a single element.

Present in **MLX 0.32.1, 0.32.2** (the latest PyPI release as of
2026-09-28) and on `main`. Re-verified here on 0.32.2, CPU and GPU, float32
and float64.

**Three symptoms**, the first of which is all this write-up originally
described:

1. `mx.grad` of a one-element strided slice fills `start:stop` instead of
   the selected index.
2. A **negative** stride does not misplace the gradient — it **zeroes** it.
   `z[3::-4]` on four elements returns `[0, 0, 0, 0]`.
3. `mx.vmap` of the slice returns the wrong shape: `vmap(z[0:2:2])` over a
   `(3, 2)` batch gives `(3, 2)`, not `(3, 1)`.

**Why it bites, in nine lines** — a pairwise sum, the standard
tree-reduction split, on four numbers:

```python
def pairsum(x):
    while x.shape[0] > 1:
        x = x[0::2] + x[1::2]
    return x[0]

(v,), (g,) = mx.vjp(pairsum, [mx.array([1., 2., 3., 4.])], [mx.array(1.)])
# v = 10 exactly;  g = [1, 1, 2, 2]  -- expected [1, 1, 1, 1]
```

The last level always has two elements, both slices there are length one,
and the second one's cotangent lands on both positions. Value exact,
gradient wrong by 2x on half the inputs, no error. Reshaping to
`(k//2, 2)` and indexing gives `[1, 1, 1, 1]` and is the cheaper op.

**What to do when #4446 merges:** nothing to the code — `elements._pairs`
keeps its reshape, which is faster anyway. Delete
`tests/test_mlx_gotchas.py::test_length_one_strided_slice_has_a_wrong_vjp`,
which asserts the *buggy* behaviour and will start failing on the fixed
release; that failure is the signal.

## The report as originally drafted (kept for the record; superseded by #4444)

---

**Title:** Wrong VJP for a strided slice that yields a single element

**Body:**

When a strided slice selects exactly **one** element, its vector-Jacobian
product scatters the cotangent across the entire `start:stop` span instead of
to the single selected index — the stride is ignored on the backward pass.
Slices yielding two or more elements are correct, and the forward value is
correct in every case.

```python
import mlx.core as mx

x = mx.array([1.0, 2.0])
_, g = mx.vjp(lambda z: mx.sum(z[0:2:2]), [x], [mx.array(1.0)])
print(g[0])       # array([1, 1], dtype=float32)   expected [1, 0]
```

`z[0:2:2]` selects only index 0, so the gradient should be `[1, 0]`.

The pattern is that the backward pass fills `start:stop` rather than
`start:stop:step`:

| slice | elements selected | gradient | expected |
|---|---|---|---|
| `z[0:2:2]`, n=2 | 1 | `[1, 1]` | `[1, 0]` |
| `z[0:3:3]`, n=3 | 1 | `[1, 1, 1]` | `[1, 0, 0]` |
| `z[1:5:4]`, n=5 | 1 | `[0, 1, 1, 1, 1]` | `[0, 1, 0, 0, 0]` |
| `z[2:8:6]`, n=8 | 1 | `[0, 0, 1, 1, 1, 1, 1, 1]` | `[0, 0, 1, 0, 0, 0, 0, 0]` |
| `z[0:4:2]`, n=4 | 2 | `[1, 0, 1, 0]` | `[1, 0, 1, 0]` ✔ |
| `z[0:6:3]`, n=6 | 2 | `[1, 0, 0, 1, 0, 0]` | `[1, 0, 0, 1, 0, 0]` ✔ |
| `z[0:6:2]`, n=6 | 3 | `[1, 0, 1, 0, 1, 0]` | `[1, 0, 1, 0, 1, 0]` ✔ |

Reproducer that prints the whole table:

```python
import mlx.core as mx, numpy as np

def grad_of(n, sl):
    x = mx.array([float(i + 1) for i in range(n)])
    _, g = mx.vjp(lambda z: mx.sum(z[sl]), [x], [mx.array(1.0)])
    return np.array(g[0]).astype(int).tolist()

for n, sl in [(2, slice(0, 2, 2)), (3, slice(0, 3, 3)), (5, slice(1, 5, 4)),
              (8, slice(2, 8, 6)), (4, slice(0, 4, 2)), (6, slice(0, 6, 3))]:
    expected = [0] * n
    for i in range(*sl.indices(n)):
        expected[i] = 1
    got = grad_of(n, sl)
    print(f"n={n} [{sl.start}:{sl.stop}:{sl.step}] -> {got}, expected {expected}",
          "OK" if got == expected else "WRONG")
```

**Environment (as of the 2026-09-06 draft; the machine now reports macOS 27.0 /
Darwin 27.0.0):** MLX 0.32.2 / mlx-metal 0.32.2, Apple M2 Max, Python 3.11. Reproduces with `mx.set_default_device(mx.cpu)`
and with the GPU default, and in float64 as well as float32.

**Why it bites in practice:** any binary reduction tree splits its input into
even and odd halves with `x[0:k:2]` / `x[1:k:2]`, and the final level always
has `k = 2` — so every gradient through a tree reduction is silently wrong
while every forward value is exact. In our case (a Gaussian-process marginal
likelihood built as a scan-element reduction) the log-likelihood matched an
independent oracle to 1e-12 while the gradient was ~50% off, which an HMC
sampler would have converged on without complaint.

**Workaround** (for anyone who finds this before it is fixed): reshape and
index instead of striding — `mx.reshape(x, (k // 2, 2))[:, 0]` and `[:, 1]`
have correct VJPs and are cheaper anyway.

---

## Local status

* Worked around in `anvilgp/elements.py::_pairs` (reshape, never a step-2
  slice). Kept regardless of the upstream fix: it is the cheaper op.
* Pinned by `tests/test_mlx_gotchas.py::test_length_one_strided_slice_has_a_wrong_vjp`,
  which asserts the buggy behaviour and fails loudly when MLX fixes it.
* Upstream: issue #4444 (open), fix PR #4446 (open, low priority, awaiting
  verification). Nothing further to file.
