"""Noise models: the per-datum observation variance, celerite2's ``diag``.

`GPLogLike` takes ``noise_fn(v_noise, d) -> (n_chains, m)``, the **total**
noise variance of each datum, from the noise columns of the parameter
vector (raw model-space values, ``param_map.noise``) and ``d``, a
:class:`NoiseData` of per-datum arrays sliced alike.  Any expression
works, and its gradient comes from autodiff.  It must act pointwise along
the data axis, because the reference path evaluates it one block of data
at a time.

The default, with no ``noise_fn``, is ``yerr^2 + jitter^2`` from
``param_map.jitter``.  This module has the other common forms as
:class:`NoiseModel` objects -- a function plus the bounds of its
parameters -- for `anvilgp.anvil.make_target`'s ``noise=``:

* :func:`error_scale` -- ``(s yerr)^2``: error bars that are right up to
  a factor;
* :func:`scale_and_jitter` -- ``(s yerr)^2 + j^2``;

each optionally per instrument.

**float64 caution** for hand-written noise functions: MLX's float64
``exp`` and ``sin`` are only float32-accurate, which puts a float32 floor
under the ``.hi`` reference path.  Use :func:`anvilgp.special.exp_acc` and
:func:`anvilgp.special.sincos`; ``log`` and ``sqrt`` are fine.  The
models here parametrise scales by their logarithm and use ``exp_acc``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import mlx.core as mx
import numpy as np

from .priors import SpecHint
from .special import exp_acc

__all__ = ["NoiseData", "NoiseModel", "error_scale", "scale_and_jitter"]


class NoiseData(NamedTuple):
    """Per-datum inputs of a noise function, sliced alike.

    ``yerr`` the error bars ``(m,)``; ``x`` the abscissa, the mean model's
    ``x_mean`` if there is one, else ``t`` (``(m,)`` or ``(c, m)``);
    ``inst`` the instrument label of each datum ``(m,)`` int32, or None.
    """

    yerr: mx.array
    x: mx.array
    inst: mx.array | None


@dataclass
class NoiseModel:
    """A noise function and the bounds of its parameters.

    ``fn(v_noise, d)`` is `GPLogLike`'s ``noise_fn``; ``specs`` give one
    :class:`~anvilgp.priors.SpecHint` per column of ``v_noise``, in order;
    ``natural(v_noise)`` maps model-space draws to named physical values
    for reporting (`GPTarget.natural`).
    """

    fn: Callable
    specs: list[SpecHint]
    natural: Callable | None = field(default=None)


def _per_inst(col: mx.array, d: NoiseData, n_inst: int) -> mx.array:
    """``(n_chains, n_inst)`` per-instrument values -> one per datum."""
    if n_inst == 1:
        return col
    if d.inst is None:
        raise ValueError(
            f"a noise model with {n_inst} instruments needs "
            f"param_map.instrument to say which datum is which")
    return mx.take(col, d.inst, axis=1)


def _names(stem: str, n_inst: int) -> list[str]:
    return [stem] if n_inst == 1 else [f"{stem}{i}" for i in range(n_inst)]


def error_scale(lo: float = 0.1, hi: float = 10.0,
                n_instruments: int = 1) -> NoiseModel:
    """Noise variance ``(s yerr)^2``: the error bars, right up to a factor
    ``s`` (one per instrument with ``n_instruments > 1``).

    ``s`` is log-uniform on ``[lo, hi]``.  Keep ``lo`` well above 1e-3: the
    likelihood floors the innovation variance at ``1e-6 yerr^2``
    (`GPPolicy.s_floor_rel`).
    """
    if not 0.0 < lo < hi:
        raise ValueError(f"need 0 < lo < hi, got lo={lo}, hi={hi}")
    k = int(n_instruments)
    names = _names("log_noise_scale", k)

    def fn(vn, d):
        s = _per_inst(exp_acc(vn[:, :k]), d, k)
        return (s * d.yerr) ** 2

    def natural(vn):
        vn = np.asarray(vn, dtype=np.float64)
        return {n[len("log_"):]: np.exp(vn[..., i])
                for i, n in enumerate(names)}

    specs = [SpecHint(n, lo=math.log(lo), hi=math.log(hi)) for n in names]
    return NoiseModel(fn, specs, natural)


def scale_and_jitter(yerr, scale: tuple[float, float] = (0.1, 10.0),
                     jitter: tuple[float, float] | None = None,
                     n_instruments: int = 1) -> NoiseModel:
    """Noise variance ``(s yerr)^2 + j^2``, per instrument if asked.

    ``s`` log-uniform on ``scale``; ``j`` log-uniform on ``jitter``, by
    default ``1e-3`` to ``10`` times the median error bar.  The lower
    bound is the default additive jitter's (`priors.jitter_hyper`, which
    runs to ``1e3`` times): it keeps ``j`` from wandering into the region
    where it no longer changes the likelihood.  The upper is tighter
    because ``s`` already scales the error bars.
    """
    lo, hi = scale
    if not 0.0 < lo < hi:
        raise ValueError(f"need 0 < scale lo < hi, got {scale}")
    med = float(np.median(np.asarray(yerr, dtype=np.float64)))
    jlo, jhi = jitter if jitter is not None else (1e-3 * med, 10.0 * med)
    if not 0.0 < jlo < jhi:
        raise ValueError(f"need 0 < jitter lo < hi, got {(jlo, jhi)}")
    k = int(n_instruments)
    s_names = _names("log_noise_scale", k)
    j_names = _names("log_noise_jitter", k)

    def fn(vn, d):
        s = _per_inst(exp_acc(vn[:, :k]), d, k)
        j2 = _per_inst(exp_acc(2.0 * vn[:, k:2 * k]), d, k)
        return (s * d.yerr) ** 2 + j2

    def natural(vn):
        vn = np.asarray(vn, dtype=np.float64)
        return {n[len("log_"):]: np.exp(vn[..., i])
                for i, n in enumerate(s_names + j_names)}

    specs = ([SpecHint(n, lo=math.log(lo), hi=math.log(hi)) for n in s_names]
             + [SpecHint(n, lo=math.log(jlo), hi=math.log(jhi))
                for n in j_names])
    return NoiseModel(fn, specs, natural)
