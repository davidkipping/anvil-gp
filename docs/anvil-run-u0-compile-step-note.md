# Note for anvil: `run()` with a NumPy `u0`, and a `compile_step` set on the kernel

**2026-11-06. DRAFT, not sent.** From anvil-gp, passing on two small
things a downstream user (a JWST/NIRSpec analysis using anvil-gp) hit with
`anvil.run`. Both reproduced against anvil 0.12.0 as installed here; the
code cited is `anvil/engine.py` at `231d41f`, which the uncommitted 0.12.0
changes do not touch. Neither blocks anyone: each has a one-line
workaround, and anvil-gp's tutorial already uses it.

## 1. A NumPy `u0` fails with a dtype error that names neither

```python
u0 = np.random.default_rng(0).normal(size=(64, 3))
anvil.run(EnsembleKernel(target, seed=1), target, u0, n_samples=5)
```

```
  File ".../anvil/engine.py", line 477, in run
    u0 = u0.astype(mx.float32) if u0.dtype != mx.float32 else u0
TypeError: Cannot interpret 'mlx.core.float32' as a data type
```

`run` assumes an `mx.array`, and NumPy's `astype` cannot take an MLX
dtype. A float32 NumPy array fails the same way, because a NumPy dtype
never compares equal to `mx.float32`. The signature does say
`u0: mx.array | None`. But `EnsembleSampler.run_mcmc` accepts NumPy
(`emcee_api.py:80`), and the README's first example passes it a NumPy
`p0`. So a user moving from that API to `run` reasonably expects NumPy to
work, and the error points at neither NumPy nor `u0`.

**Suggested fix:** accept anything array-like, as `run_mcmc` already does:

```python
if u0 is not None and not isinstance(u0, mx.array):
    u0 = mx.array(np.asarray(u0, dtype=np.float32))
```

Or, if `run` should stay MLX-only, raise a `TypeError` that says
"`u0` must be an `mx.array`; got numpy.ndarray".

## 2. `kernel.compile_step = False` is silently overwritten

```python
k = EnsembleKernel(target, seed=1)
k.compile_step = False             # looks like it should be honoured
anvil.run(k, target, u0, n_samples=5)
```

`run` assigns `kernel.compile_step = compile_step` from its own default of
`True` before `retrace` (engine.py:492; and :519 on resume). So the
attribute the user set is discarded without a word. `kernels/base.py`
documents the attribute as engine-owned, so this is by design. Nothing
the user can see says so, however, and the attribute is public.

It matters for a log density that leaves MLX (the user's evaluated part
of the model in NumPy on the CPU). Under `mx.compile` that fails with
MLX's own message, which does not mention anvil or `compile_step`:

```
  File ".../anvil/compile.py", line 85, in __call__
    out = self._compiled_fn(*args, **kwargs)
ValueError: [eval] Attempting to eval an array during function
transformations like compile or vmap is not allowed.
```

`CompiledOrEager` does not fall back here. The error is a `ValueError`
raised by the compiled call itself, not the `RuntimeError` refusal it
catches when it evaluates the output, and that is correct: this is not a
fusion refusal. `run(..., compile_step=False)` works (checked: same target,
runs eagerly to completion).

**Suggested fixes, any one of which would have saved the user the
detour:**

- Respect an explicit setting: make `run`'s default `compile_step=None`,
  meaning "leave the kernel's attribute as it is".
- Or warn when `run` overrides an attribute that differs from its own
  argument.
- And/or catch this specific MLX error around the compiled step and
  re-raise it with a pointer to `compile_step=False`. For example: "the log
  density evaluated an array during mx.compile, which usually means it
  leaves MLX (NumPy, .item(), host control flow); pass
  run(compile_step=False) to run it uncompiled". The README already says
  `log_prob` must be built from `mlx.core` ops, so this is the message for
  the user who missed that line.

## What anvil-gp does meanwhile

Nothing in anvil-gp depends on either change. Our tutorial's warm-start
example passes `u0` as an `mx.array`.
