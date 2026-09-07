# Ready-to-file MLX issue: wrong VJP for a single-element strided slice

Present in **MLX 0.32.2** (the current release as of 2026-09-06), on both the
CPU and GPU backends and in both float32 and float64. Forward values are
correct; only gradients are wrong, which is what makes it dangerous.

Paste the below into <https://github.com/ml-explore/mlx/issues>.

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

**Environment:** MLX 0.32.2 / mlx-metal 0.32.2, Apple M2 Max, macOS 15.6
(Darwin 25.6.0), Python 3.11. Reproduces with `mx.set_default_device(mx.cpu)`
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

* Worked around in `anvilgp/elements.py::_pairs`.
* Pinned by
  `tests/test_mlx_gotchas.py::test_length_one_strided_slice_has_a_wrong_vjp`,
  which asserts the *buggy* behaviour and tells you to delete the workaround
  when it starts failing.
