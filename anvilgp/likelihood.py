"""``GPLogLike`` -- the anvil engine contract for a GP marginal likelihood.

Drop-in replacement for ``anvil.precision.ChunkedGaussianLogLike``: same
``__call__(v) -> (n_chains,)`` and ``.hi(v)`` surface, same float32
discipline, but the observation covariance is ``K + Sigma`` rather than a
diagonal.  The mean model, the GP hyperparameters, the per-instrument
jitters and the per-instrument offsets all come out of the *same* parameter
vector, which is the point: joint sampling of hyperparameters with the mean
model is the requirement, and any design that needs them frozen or pre-fit
is out of scope.

Where the tiling goes
---------------------
``ChunkedGaussianLogLike.hi`` tiles over the *data* axis, because a white-
noise likelihood is a sum over independent points.  A GP factorisation is
global-sequential over data, so ``.hi`` here tiles over **chains** (<= 16
per fp64 tile) and streams the data axis in recursion order.  Peak working
set is one block of elements: ``n_tile x block x (3m^2 + 2m + 1)`` numbers.
The default 16-chain x 1024-point fp64 tile at m = 4 holds 7.5 MB of
elements and peaks around 60 MB measured, the rest being the intermediates
of the composition; an untiled fp64 pass over 1024 chains x 1e5 points
would transiently ask for ~50 GB.

The gradient split
------------------
``d logL / d(mean model) = alpha = (K + Sigma)^-1 r`` -- the mean model's
gradient is the GP-whitened residual, chain-ruled into whatever VJP the
mean model already has (MetalPlanet's is analytic).  So the hard gradient
engineering is confined to the hyperparameters; the transit path is
already solved and rides ordinary autodiff through ``mean_fn``.

Known v1 limitations, stated rather than discovered
---------------------------------------------------
* No exposure-time integration of the kernel (long-cadence smearing).
  S+LEAF (2201.02440) is the v2 path.
* No 2D/multiband GPs.
* ``predict`` / conditional mean is deliberately out of the hot path; see
  :func:`anvilgp.oracle.dense_predict` for the offline float64 utility.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .special import exp_acc
from .elements import (Element, chunked_reduce, chunked_reduce_blocked,
                       first_point,
                       leaf_elements, leaf_parts, tree_reduce,
                       use_blocked_leaves)
from .terms import Term

__all__ = ["GPLogLike", "GPPolicy", "ParamMap", "KernelParam", "a64"]

_LARGE_NEGATIVE = -1e30
#: anything at or below this is a sanitised failure, not a real value.
_FAILED = 0.5 * _LARGE_NEGATIVE


def a64(x: np.ndarray) -> mx.array:
    """``mx.array`` that actually keeps float64.

    ``mx.array(np_float64_array)`` silently produces a **float32** array --
    the dtype must be named explicitly.  Every float64 path in this package
    goes through here, because the failure mode is invisible: the graph runs,
    reports float64 dtypes downstream (arithmetic promotes it back), and
    quietly carries data that was rounded to float32 on the way in.
    """
    return mx.array(np.ascontiguousarray(np.asarray(x, dtype=np.float64)),
                    dtype=mx.float64)


@dataclass
class GPPolicy:
    """Blocking and precision policy.

    ``block`` is the number of data points condensed into one element before
    the per-block roots are composed.  It is padded to a power of two so the
    tree depth is a trace-time constant.  It trades memory against graph
    size: peak element memory is ``n_chains * block * (3m^2+2m+1) * bytes``.

    ``eval_blocks`` forces evaluation after each block, bounding peak memory
    at the cost of making the call ineligible for ``mx.compile``.  The float32
    production path leaves it off (the fused Metal kernel of M3 is what makes
    that affordable); the float64 ``.hi`` path turns it on.
    """

    block: int = 1024
    #: observations folded sequentially per chunked-scan sub-block, or None
    #: to pick from the state dimension.  Trades cheap rank-one compositions
    #: against expensive general ones: raising it cuts general compositions by
    #: the same factor, at the cost of more sequential steps over smaller
    #: batches.  The general composition costs ~13 m^3 against the fold's
    #: ~3 m^3, so the optimum climbs with m -- measured at 1024 chains x
    #: N = 1e4: 16 at m = 2 and 4, 32 at m = 6, 64 at m = 8.
    #: See :func:`anvilgp.elements.chunked_reduce`; 1 recovers the pure tree.
    sub_block: int | None = None
    hi_chain_block: int = 16
    hi_block: int = 1024
    eval_blocks: bool = False
    #: keep leaves in compact block form (None = decide from the term's block
    #: structure).  See :func:`anvilgp.elements.use_blocked_leaves`.
    blocked: bool | None = None
    #: use the fused Metal scan kernel (:mod:`anvilgp.metal`).
    #: Differentiable since M4: the kernel carries its own VJP, so a
    #: gradient sampler can use it too.  Measured 45-74x over the MLX path
    #: on the forward and ~100x on the gradient, with 20-30x less peak
    #: memory; see the README.  Limits: m <= 4 and one jitter.  Any
    #: cadence is fine, including one where every timestamp spacing is
    #: distinct: the gradient's adjoint buffer is one slot per
    #: ``(block, lag)`` pair that *occurs*, so it is bounded by N, and
    #: :data:`anvilgp.metal.MAX_ADJOINT_BUFFER_BYTES` is a sanity check
    #: rather than a limit to design around.
    metal: bool = False
    #: observations condensed per kernel thread, or None to pick from N.
    #: Larger means fewer, longer sequential walks and less work left for
    #: the MLX tree above, which is launch-bound and is most of the
    #: remaining wall time at small N.  See :func:`_auto_metal_block`.
    metal_block: int | None = None
    #: floor on the leaf innovation variance, in units of the reference
    #: variance.  Positive-definiteness failure is a *defined* outcome:
    #: with thousands of chains, warmup will visit float32-infeasible
    #: corners, and the likelihood must come back finite there.  Since
    #: stage 0 of M4 the non-physical term ``h Q h^T`` is clamped at zero
    #: on its own, so the innovation variance is at least the reference
    #: variance and this floor **cannot bind** for any value below 1.
    #: Values >= 1 are refused: they would change the value, and the
    #: kernel's backward does not differentiate the floor (stage R).
    s_floor_rel: float = 1e-6

    def __post_init__(self):
        if self.metal_block is not None and int(self.metal_block) < 1:
            raise ValueError(
                f"metal_block = {self.metal_block!r} must be None or >= 1")
        if not 0.0 <= self.s_floor_rel < 1.0:
            raise ValueError(
                f"s_floor_rel = {self.s_floor_rel!r} must be in [0, 1): the "
                f"innovation variance is already at least r_ref, so a floor "
                f"below 1 never binds, and one at or above 1 would alter the "
                f"likelihood without the kernel gradient following it")


@dataclass
class KernelParam:
    """How one natural-unit kernel parameter is produced from ``v``.

    ``link``:

    * ``"log"``    -- ``ref * exp(v[index])``.  The default for every scale
      parameter.  Sampling the *offset* from a float64 host reference is
      anvil's rule 2 applied to hyperparameters: it conditions the graph and
      centres the shared diagonal preconditioner, so a well-chosen ``ref``
      puts the posterior near ``v = 0`` where float32 has the most room.
    * ``"logit"``  -- ``sigmoid(v[index])``, for mixture fractions.
    * ``"linear"`` -- ``ref + scale * v[index]``.
    * ``"const"``  -- fixed ``value``; e.g. Q = 1/sqrt(2) for granulation.
    """

    link: str = "log"
    index: int | None = None
    ref: float = 1.0
    scale: float = 1.0
    value: float = 0.0

    def __post_init__(self):
        # Plain Python floats, always.  A np.float64 here propagates into
        # mx.full/mx.concatenate and fails with an unreadable pybind
        # signature error far from the line that caused it.
        self.ref = float(self.ref)
        self.scale = float(self.scale)
        self.value = float(self.value)

    def apply(self, v: mx.array) -> mx.array:
        if self.link != "const" and self.index is None:
            raise ValueError(
                f"KernelParam(link={self.link!r}) needs an `index`: which "
                "column of v it reads. Only link='const' may omit it.")
        # exp_acc, not mx.exp: MLX's float64 exp is only float32-accurate
        # (~7e-7 relative), which puts a float32-sized floor under the
        # ".hi" verification path.  Measured against a dense fp64 Cholesky
        # through a log link: 2.0e-6 log-units at N = 2048 and 4.8e-6 at
        # N = 8192, i.e. growing about as sqrt(N), so ~1e-5 at N = 1e5.
        # With exp_acc the same comparison is 1.7e-10, four orders better.
        # Small in absolute terms, but .hi is the *reference* -- it should
        # not be the thing limiting what it can certify.  In float32
        # exp_acc *is* the native op, so the production path is unchanged.
        n = v.shape[0]
        if self.link == "const":
            return mx.full((n, 1), self.value, dtype=v.dtype)
        col = v[:, self.index:self.index + 1]
        if self.link == "log":
            return self.ref * exp_acc(col)
        if self.link == "logit":
            return 1.0 / (1.0 + exp_acc(-col))
        if self.link == "linear":
            return self.ref + self.scale * col
        raise ValueError(f"unknown link {self.link!r}")


@dataclass
class ParamMap:
    """Which slices of ``v`` play which role.

    ``kernel`` has one :class:`KernelParam` per entry of
    ``term.param_names``, in that order.  ``jitter`` and ``offset`` have one
    per instrument, and ``instrument`` assigns each datum to one of them --
    per-instrument jitters and mean offsets are v1 features, not extras:
    they are the first thing a real dataset needs, and ``yerr`` alone is not
    the observation diagonal once a jitter is being sampled.
    """

    kernel: list[KernelParam]
    n_dim: int
    mean: slice | None = None
    jitter: list[KernelParam] = field(default_factory=list)
    offset: list[KernelParam] = field(default_factory=list)
    instrument: np.ndarray | None = None
    #: groups of kernel-parameter positions constrained to be strictly
    #: increasing.  See :meth:`kernel_params`.
    order_groups: list[list[int]] | None = None

    @classmethod
    def const(cls, values, jitter=None, **kw) -> "ParamMap":
        """A map whose kernel parameters are fixed host constants.

        The parametrisation under test is often not the point -- parity and
        oracle comparisons want a known kernel and an ignored ``v``.  This
        is that, in one line instead of a list comprehension.
        """
        jit = ([] if jitter is None
               else [KernelParam(link="const", value=float(jitter))])
        return cls(kernel=[KernelParam(link="const", value=float(x))
                           for x in values],
                   n_dim=kw.pop("n_dim", 1), jitter=jit, **kw)

    @classmethod
    def log(cls, refs, **kw) -> "ParamMap":
        """A map with one log-link parameter per reference, in order.

        ``v[:, i]`` is then the log-offset of the i-th kernel parameter from
        ``refs[i]``, which is the shipped parametrisation.
        """
        refs = list(refs)
        return cls(kernel=[KernelParam(link="log", index=i, ref=float(r))
                           for i, r in enumerate(refs)],
                   n_dim=kw.pop("n_dim", len(refs)), **kw)

    def kernel_params(self, v: mx.array) -> mx.array:
        kp = mx.concatenate([k.apply(v) for k in self.kernel], axis=-1)
        if not self.order_groups:
            return kp
        # Ordering constraint for sums of same-type terms.  A k-term sum has
        # k! identical posterior modes under permutation of the terms, which
        # makes every convergence diagnostic meaningless and lets chains
        # label-switch forever.  Reinterpreting each subsequent member of a
        # group as a *ratio* -- x_j <- x_i (1 + r_j) with r_j > 0 -- makes
        # the ordering hold by construction, with no rejection and no
        # discontinuity for the sampler to trip over.  (anvil's Transform
        # layer has no ordered-vector transform; this keeps the fix local
        # instead of requiring an engine change.)
        #
        # ``r_j`` must be DIMENSIONLESS.  It used to be ``x_j`` itself, a
        # timescale in the user's time unit, so ``1 + x_j`` added a pure
        # number to a quantity with units and the whole prior moved when the
        # data were expressed in hours instead of days: the ratio at v = 0
        # was 1.4 in days and 3.9e4 in seconds, and the documented
        # ``[2 dt, T/2]`` bound on every ordered member after the first was
        # simply false.  Dividing by the column's own host reference fixes
        # the units: for a log link ``x_j / ref = exp(v_j)``, so the
        # sampled bounds are already ratio bounds and need no rescaling.
        cols = [kp[:, i:i + 1] for i in range(kp.shape[1])]
        for group in self.order_groups:
            for lo, hi in zip(group[:-1], group[1:]):
                k = self.kernel[hi]
                if k.link != "log":
                    # Only the log link gives a dimensionless ratio
                    # (x/ref = exp(v)).  With any other link the `1 +` below
                    # adds a pure number to a quantity carrying the user's
                    # time unit, which silently changes the kernel and makes
                    # the prior depend on whether t is in days or seconds.
                    raise ValueError(
                        f"ordered kernel parameter {hi} has link "
                        f"{k.link!r}; ordering groups require 'log'. "
                        "Drop it from order_groups, or give it a log link.")
                cols[hi] = cols[lo] * (1.0 + cols[hi] / float(k.ref))
        return mx.concatenate(cols, axis=-1)

    @staticmethod
    def _per_datum(v: mx.array, specs, idx: mx.array | None) -> mx.array | None:
        """(n_chains, 1) for a single instrument, (n_chains, K) for several."""
        if not specs:
            return None
        cols = mx.concatenate([s.apply(v) for s in specs], axis=-1)
        if len(specs) == 1:
            return cols
        if idx is None:
            raise ValueError("param_map.instrument is required when more "
                             "than one jitter/offset is declared")
        return mx.take(cols, idx, axis=1)

    def jitter_var(self, v: mx.array, idx: mx.array | None) -> mx.array | None:
        j = self._per_datum(v, self.jitter, idx)
        return None if j is None else j * j

    def offsets(self, v: mx.array, idx: mx.array | None) -> mx.array | None:
        return self._per_datum(v, self.offset, idx)


class GPLogLike:
    """GP marginal log-likelihood over a batch of parameter vectors.

    ``mean_fn(v_mean, x)`` is anvil's batched model contract -- parameters
    ``(n_chains, k)`` and abscissa ``(m,)`` or ``(c, m)`` -> predictions
    ``(n_chains, m)`` -- or ``None`` for a pure-GP fit.

    ``t`` is the physical time vector (float64, host).  ``x_mean`` is the
    mean model's *own* abscissa: for MetalPlanet that is the ``(2, m)`` stack
    of epoch-centred residual times and orbit numbers, which is a different
    array from ``t`` and must not be conflated with it -- ``t`` drives the
    GP's lag structure, ``x_mean`` drives the transit.

    Sorted input is required (unsorted raises).  Duplicate timestamps are
    legal: celerite2 accepts them, and ``dt = 0`` gives ``Phi = I``,
    ``Q = 0`` exactly here, with no NaN.
    """

    def __init__(self, mean_fn, t, x_mean, y, yerr, kernel_spec: Term,
                 param_map: ParamMap, policy: GPPolicy | None = None):
        self.mean_fn = mean_fn
        self.term = kernel_spec
        self.param_map = param_map
        self.policy = policy or GPPolicy()
        self.sub_block = (self.policy.sub_block
                          if self.policy.sub_block is not None
                          else _auto_sub_block(kernel_spec.n_state))
        #: whether to keep leaves in compact block form.  See
        #: :func:`anvilgp.elements.use_blocked_leaves` for the measurement.
        self.blocked = (use_blocked_leaves(kernel_spec.block_sizes)
                        if self.policy.blocked is None else self.policy.blocked)

        self._t64 = np.asarray(t, dtype=np.float64).ravel()
        self._y64 = np.asarray(y, dtype=np.float64).ravel()
        self._yerr64 = np.asarray(yerr, dtype=np.float64).ravel()
        self.n_data = self._t64.size
        if self._y64.size != self.n_data or self._yerr64.size != self.n_data:
            raise ValueError("t, y and yerr must have the same length")
        if self.n_data == 0:
            raise ValueError("t, y and yerr must be non-empty")
        # Non-finite inputs pass every comparison below (NaN < 0 is False,
        # NaN <= 0 is False), reach the graph, and come back as a sanitised
        # -1e30 for EVERY chain -- a likelihood that is flat in every
        # direction, which a sampler happily explores forever.  Name the
        # array and the count so the fix is obvious.
        for name, arr in (("t", self._t64), ("y", self._y64),
                          ("yerr", self._yerr64)):
            bad = ~np.isfinite(arr)
            if bad.any():
                raise ValueError(
                    f"{name} contains {int(bad.sum())} non-finite "
                    f"value(s) (first at index {int(np.argmax(bad))}). "
                    "Mask them out before building the likelihood.")
        d = np.diff(self._t64)
        if np.any(d < 0.0):
            raise ValueError(
                "t must be sorted ascending (duplicates allowed). "
                "Sort your data and reorder y/yerr to match.")
        if np.any(self._yerr64 <= 0.0):
            raise ValueError("yerr must be strictly positive")
        # The graph carries yerr^2, not yerr.  Below ~1e-19 that squares to
        # a float32 denormal-or-zero, every innovation variance collapses,
        # and the likelihood comes back -1e30 for every chain -- a flat
        # posterior with no error.  Refuse it and say what to do.
        if float(np.min(self._yerr64)) ** 2 < 1e-36:
            raise ValueError(
                f"yerr as small as {float(np.min(self._yerr64)):.3g} squares "
                "to a float32 zero. Rescale y and yerr (the likelihood is "
                "invariant up to a constant) or work in relative flux.")

        # A mean model needs all three of mean_fn, param_map.mean and
        # x_mean.  With any one missing the GP silently fits y itself: the
        # transit is never subtracted, the parameters that should describe
        # it are sampled but inert, and the result looks like a converged
        # GP-only fit rather than an error.
        has_fn, has_slice = mean_fn is not None, param_map.mean is not None
        if has_fn != has_slice:
            raise ValueError(
                "a mean model needs both `mean_fn` and `param_map.mean`; "
                f"got mean_fn={'set' if has_fn else 'None'} and "
                f"param_map.mean={'set' if has_slice else 'None'}. "
                "Without both, the mean model is silently ignored.")
        if has_fn and x_mean is None:
            raise ValueError(
                "a mean model needs `x_mean` (the covariates it is "
                "evaluated on); got None.")

        # dt is precomputed in float64 on the host and enters the graph as a
        # well-conditioned O(cadence) quantity; the absolute epoch never does.
        self._dt64 = np.concatenate([[0.0], d])
        self._x_mean64 = (None if x_mean is None
                          else np.asarray(x_mean, dtype=np.float64))

        self._dt32 = mx.array(self._dt64.astype(np.float32))
        # Distinct lags.  Real photometry has one cadence plus one lag per
        # gap; only a synthetic irregular grid has N of them.
        #
        # Grouped *per dtype*, because that is the only way the reuse is
        # exact.  A nominally regular cadence is not regular in float64 --
        # `arange(n) * 0.02` has 15 distinct diffs at n = 1e4, pure
        # representation noise -- but it has exactly **one** in float32.
        # So the production path groups the float32 lags it will actually
        # evaluate, and the verification path groups the float64 ones; each
        # is exact for itself, and neither needs a tolerance.
        self._uniq32 = self._distinct_lags(np.float32)
        self._uniq64 = self._distinct_lags(np.float64)
        # Lag table for the Metal kernel.  Row 0 is reserved for the
        # no-predecessor case (Phi = 0, Q = P_inf = I), so the kernel needs
        # no special case for the first observation -- the table says it.
        uniq32, inv32 = np.unique(self._dt64.astype(np.float32),
                                  return_inverse=True)
        lag = inv32.astype(np.int32) + 1
        lag[0] = 0
        self._metal_uniq_dt = mx.array(uniq32, dtype=mx.float32)
        self._metal_lag = mx.array(lag)
        self.n_distinct_lags = (self.n_data if self._uniq32 is None
                                else int(self._uniq32[0].size))
        self._y32 = mx.array(self._y64.astype(np.float32))
        self._r_ref32 = mx.array((self._yerr64 ** 2).astype(np.float32))
        self._x_mean32 = (None if self._x_mean64 is None
                          else mx.array(self._x_mean64.astype(np.float32)))
        inst = param_map.instrument
        self._inst32 = (None if inst is None
                        else mx.array(np.asarray(inst, dtype=np.int32)))

        #: float64 host constant ``-1/2 sum log(2 pi yerr^2)``.
        #: Deliberately *outside* the graph: at N = 1e5 and 500 ppm it is
        #: ~7e5, and adding that to an O(1) float32 number would quantise the
        #: answer at 0.06 -- worse than the Metropolis decision scale it is
        #: supposed to preserve.  The hyperparameter-*dependent* part of
        #: log|K| is inside, as ``log(S / yerr^2)``.  Use
        #: :meth:`log_prob_absolute` for parity tests, which must compare
        #: absolute values: a wrong constant is invisible to MCMC.
        self.log_norm_const = float(
            -0.5 * np.sum(np.log(2.0 * np.pi * self._yerr64 ** 2)))

        self._lag_np = lag
        # Assigning `metal_block` (a property) builds the kernel's slot map
        # from it, so the map can never disagree with the launch.  Stage R2
        # showed what a stale map does: a correct value with a gradient
        # wrong by 2.6x, or a read past the end of the block table that
        # hangs the process at 0% CPU.
        self.metal_block = (self.policy.metal_block
                            if self.policy.metal_block is not None
                            else _auto_metal_block(self.n_data,
                                                   kernel_spec.n_state))

        # Decided once, not per call: whether the fused kernel can serve this
        # problem, and the gradient guard that wraps whichever path does.
        self._metal_ok, self._metal_why = self.metal_eligible()
        if self.policy.metal and not self._metal_ok:
            # Asking for the kernel and silently getting the MLX path is a
            # 40x performance surprise with no signal.  Say so once, here,
            # rather than leaving the user to infer it from the wall clock.
            warnings.warn(
                f"GPPolicy(metal=True) cannot be honoured: {self._metal_why}. "
                "Falling back to the MLX path, which is correct but ~40x "
                "slower. Call metal_eligible() to check in advance.",
                RuntimeWarning, stacklevel=2)
        self._guard = self._build_guard()

    def _distinct_lags(self, dtype):
        """(values, inverse index) for the lags as seen in ``dtype``, or None.

        Reuse only pays once the saving is at least 4x; below that the gather
        costs more than the transcendentals it avoids.
        """
        lags = self._dt64.astype(dtype)
        uniq, inv = np.unique(lags, return_inverse=True)
        if uniq.size * 4 > lags.size:
            return None
        mxd = mx.float32 if dtype is np.float32 else mx.float64
        return (mx.array(uniq, dtype=mxd), mx.array(inv.astype(np.int32)))

    # -- shared machinery -------------------------------------------------

    def _matrices_fn(self, kp, dt, uniq):
        """Return ``get(s, e) -> (phi, q, h)`` for a block, reusing cadences.

        ``Phi`` and ``Q`` depend on the parameters and on ``dt`` -- and ``dt``
        is *data*, which for real photometry takes a handful of distinct
        values (one cadence, plus one per gap).  Evaluating the propagator's
        transcendentals and series once per distinct lag instead of once per
        point turns the term matrices from a third of the runtime into a
        gather.  Measured: at 1024 chains x N = 1e4 the term matrices cost
        96 ms of a 285 ms eager evaluation on *randomly* sampled times (9999
        distinct lags, the worst case); on a regular cadence there are two.

        Falls back to the direct route when the lags really are all distinct,
        because then the gather is pure overhead.
        """
        get = (self.term.matrices_blocks if self.blocked
               else self.term.matrices)
        if uniq is None:
            return lambda s, e: get(kp, dt[s:e])
        dt_u, inv = uniq
        phi_u, q_u, h = get(kp, dt_u)
        if self.blocked:
            return lambda s, e: ([mx.take(b, inv[s:e], axis=1) for b in phi_u],
                                 [mx.take(b, inv[s:e], axis=1) for b in q_u], h)
        return lambda s, e: (mx.take(phi_u, inv[s:e], axis=1),
                             mx.take(q_u, inv[s:e], axis=1), h)

    def leaf_inputs(self, v, dt, y, r_ref, x_mean, idx, s, e, get_mat=None):
        """``(phi, q, h, residual, r_var, r_ref, s_floor)`` for one data block.

        Exactly the argument list of :func:`~anvilgp.elements.leaf_elements`
        and :func:`~anvilgp.kalman.sequential_loglike`, so the tree reduction
        and the sequential-filter oracle consume byte-identical inputs and
        any disagreement between them is attributable to the recursion
        alone.  The Metal path does *not* come through here: it builds its
        own lag table, residual and jitter in :meth:`_evaluate_metal`, and
        is tied to this one by the parity tests rather than by sharing the
        tuple.
        """
        pm = self.param_map
        dtype = v.dtype
        n = v.shape[0]
        sub = None if idx is None else idx[s:e]

        if get_mat is None:
            get_mat = self._matrices_fn(pm.kernel_params(v), dt, None)
        phi, q, h = get_mat(s, e)
        if s == 0:
            phi, q = first_point(phi, q, dtype)

        r = mx.broadcast_to(y[s:e], (n, e - s))
        if self.mean_fn is not None and pm.mean is not None:
            r = r - self.mean_fn(v[:, pm.mean], x_mean[..., s:e])
        off = pm.offsets(v, sub)
        if off is not None:
            r = r - off

        r_ref = r_ref[s:e]
        jit2 = pm.jitter_var(v, sub)
        r_var = r_ref if jit2 is None else r_ref + jit2
        r_var = mx.broadcast_to(r_var, r.shape)
        s_floor = self.policy.s_floor_rel * r_ref
        return phi, q, h, r, r_var, r_ref, s_floor

    @property
    def metal_block(self) -> int:
        """Observations condensed per kernel thread.

        Assigning it rebuilds the backward kernel's slot map, which is
        indexed by block and so is only valid for the block length it was
        built with.  Stage R2 of the lag-buffer work found that mutating
        this attribute on a built instance left the map stale: the value
        stayed correct (the forward never reads the map) while the gradient
        came back wrong by 2.6x at one setting, exactly right by scheduling
        luck at another, and at a third the kernel read past the end of
        the block table and parked the process at 0% CPU indefinitely.
        """
        return self._metal_block

    @metal_block.setter
    def metal_block(self, blk) -> None:
        blk = int(blk)
        if blk < 1:
            raise ValueError(f"metal_block must be >= 1, got {blk}")
        self._metal_block = blk
        self._build_slot_map()

    def _build_slot_map(self) -> None:
        """Per-(block, lag) slot map and fold plan for ``self.metal_block``."""
        sid, sstart, slag = _slot_map(self._lag_np, self._metal_block)
        self._metal_slot_id = mx.array(sid)
        self._metal_slot_start = mx.array(sstart)
        self._metal_slot_lag = mx.array(slag)
        # Row count from the table itself, not `lag.max() + 1`: the two
        # differ at N = 1, where the table still has its constant row 0
        # and one lag row but every observation is lag 0 (stage R2).
        n_lags = int(self._metal_uniq_dt.shape[0]) + 1
        kind, plan = _lag_fold_plan(slag, n_lags)
        self._metal_fold_plan = (
            (kind, mx.array(plan)) if kind == "dense"
            else (kind, [(mx.array(a), mx.array(b)) for a, b in plan]))
        #: slots the backward's adjoint buffer actually needs, as against
        #: ``n_blocks x n_distinct_lags`` if every block used every lag.
        #: Equal on a regular cadence; ~1/13 of it when every timestamp
        #: spacing is distinct at 13 blocks.
        self.n_metal_slots = int(slag.size)

    def metal_eligible(self) -> tuple[bool, str]:
        """Whether the fused kernel can serve this problem, and why not."""
        from .metal import MAX_STATE, metal_available
        if not self.policy.metal:
            return False, "GPPolicy.metal is off"
        if self.term.n_state > MAX_STATE:
            return False, (f"m = {self.term.n_state} exceeds the kernel's "
                           f"register budget (max {MAX_STATE})")
        if len(self.param_map.jitter) > 1:
            return False, "per-instrument jitter is not in the kernel yet"
        if not metal_available():
            return False, "mx.fast.metal_kernel is unavailable here"
        return True, ""

    def _metal_matrices(self, v: mx.array):
        """``(Phi, Q, h)`` for the lag table, with a chunked backward.

        The value is exactly :meth:`Term.matrices`; only the *backward* is
        different, and only when the lag table is large.  MLX retains every
        intermediate of the term algebra at full table size, so its
        backward costs ~25-47x the table -- 0.009 GiB on a regular cadence
        and **75 GiB** at 1024 chains x N = 1e5 when every timestamp
        spacing is distinct, which is what made the kernel gradient
        unusable there rather than the per-lag adjoint buffer it was first
        blamed on.  Splitting the backward over the lag axis bounds it:
        each chunk builds, differentiates and frees its own slice.

        ``mx.checkpoint`` was tried first and does nothing here (identical
        peak to three decimals at every size).

        **``h`` does not depend on the lag.** Its cotangent is therefore
        charged to the first chunk only; charging it to every chunk
        multiplies that part of the gradient by the chunk count, an error
        large enough to see (9e-2 relative) and small enough to mistake for
        conditioning.  Pinned by
        ``tests/test_metal.py::test_the_chunked_front_end_charges_h_once``.
        """
        term, uniq = self.term, self._metal_uniq_dt
        kp = self.param_map.kernel_params
        n_chunk = _front_end_chunks(v.shape[0], uniq.shape[0], term.n_state)
        if n_chunk <= 1:
            return term.matrices(kp(v), uniq)

        n_lags = uniq.shape[0]
        step = -(-n_lags // n_chunk)

        @mx.custom_function
        def core(vv):
            return term.matrices(kp(vv), uniq)

        @core.vjp
        def _core_vjp(primals, cotangents, outputs):
            # one input, so MLX hands the array itself rather than a
            # one-tuple -- `primals[0]` would silently take its first row.
            vv = primals[0] if isinstance(primals, (list, tuple)) else primals
            c_phi, c_q, c_h = cotangents
            z_h = mx.zeros_like(c_h)
            acc = None
            for i, lo in enumerate(range(0, n_lags, step)):
                hi = min(lo + step, n_lags)
                du = uniq[lo:hi]

                def part(w, du=du):
                    return term.matrices(kp(w), du)

                g = mx.vjp(part, [vv],
                           [c_phi[:, lo:hi], c_q[:, lo:hi],
                            c_h if i == 0 else z_h])[1][0]
                # Deliberately no `mx.eval(g)` here.  Forcing each chunk to
                # complete looks like the way to free it and does the
                # opposite: it materialises the upstream graph -- the
                # backward kernel and its per-lag slab -- and pins it for
                # the whole loop, measured 3.6x *worse* than not chunking
                # at all (7.76 GiB against 2.17).  Left lazy, MLX frees
                # each chunk's intermediates as the sum is built: 0.97 GiB.
                acc = g if acc is None else acc + g
            return acc

        return core(v)

    def _evaluate_metal(self, v: mx.array) -> mx.array:
        """Design C: one kernel condenses the blocks, a second folds them.

        The whole thing sits behind :func:`anvilgp.metal.scan_loglike`'s
        ``custom_function``, whose VJP runs the backward kernels, so this
        path is differentiable (M4).
        """
        from .metal import scan_loglike
        pm = self.param_map
        n, m, dtype = v.shape[0], self.term.n_state, v.dtype
        # The property setter makes this unreachable; it stays because the
        # failure it guards is silent (see `metal_block`) and the check is
        # a host-side shape comparison.
        if self._metal_slot_start.shape[0] - 1 != -(-self.n_data // self.metal_block):
            raise RuntimeError(
                "the kernel's slot map was built for a different block "
                "length than the launch; assign `metal_block` rather than "
                "`_metal_block` so the map is rebuilt")
        phi, q, h = self._metal_matrices(v)
        pt = mx.concatenate(
            [mx.zeros((n, 1, m, m), dtype=dtype), phi], axis=1)
        qt = mx.concatenate(
            [mx.broadcast_to(mx.eye(m, dtype=dtype), (n, 1, m, m)), q], axis=1)

        # Built time-major, (npts, n_chains), to match the kernel's flat
        # chains-fastest grid: adjacent threads then read adjacent floats.
        # Constructed transposed rather than transposed afterwards, which
        # would be a full copy of the residual every call.
        r = mx.broadcast_to(self._y32[:, None], (self.n_data, n))
        if self.mean_fn is not None and pm.mean is not None:
            r = r - mx.swapaxes(self.mean_fn(v[:, pm.mean], self._x_mean32),
                                0, 1)
        off = pm.offsets(v, self._inst32)
        if off is not None:
            r = r - mx.swapaxes(off, 0, 1)
        jit = pm.jitter_var(v, self._inst32)
        jit2 = (mx.zeros((n, 0), dtype=dtype) if jit is None
                else mx.broadcast_to(jit, (n, 1)))

        # scan_loglike, not scan_blocks + tree_reduce here: the
        # differentiation boundary has to enclose the reduction so the
        # custom function's output is one scalar per chain.  See its
        # docstring.
        return scan_loglike(pt, qt, self._metal_lag, h, r, self._r_ref32,
                            jit2, self.policy.s_floor_rel, self.metal_block,
                            slots=(self._metal_slot_id,
                                   self._metal_slot_start,
                                   self.n_metal_slots,
                                   self._metal_fold_plan))

    def _evaluate(self, v, dt, y, r_ref, x_mean, idx, block, do_eval, uniq=None):
        """Blocked chunked-scan reduction over the whole series -> (n_chains,).

        The per-block roots are composed by a *pairwise tree*, not folded
        left to right.  ``ell`` grows to the full ``|log L|`` along the way,
        so a sequential fold accumulates it in float32 at magnitude ~1e5 and
        the block size silently becomes a precision knob: measured at
        N = 1e5, a left fold costs 0.10 log-units at ``block = 1024`` and
        0.40 at ``block = 256``, against 0.012 for the tree.  (The existing
        block-independence test runs in float64, where neither shows.)  The
        Metal path has always composed its blocks this way; this makes the
        MLX path agree.
        """
        get_mat = self._matrices_fn(self.param_map.kernel_params(v), dt, uniq)
        roots: list[Element] = []
        for s in range(0, self.n_data, block):
            e = min(s + block, self.n_data)
            args = self.leaf_inputs(v, dt, y, r_ref, x_mean, idx, s, e, get_mat)
            if self.blocked:
                root = chunked_reduce_blocked(leaf_parts(*args),
                                              sub_len=self.sub_block)
            else:
                root = chunked_reduce(*leaf_elements(*args),
                                      sub_len=self.sub_block)
            if do_eval:
                # bound peak memory: the block's leaf arrays die here, and
                # the root that survives is one element per chain.
                mx.eval(root)
            roots.append(root)
        if len(roots) == 1:
            return roots[0].ell
        stacked = Element(*(mx.stack([getattr(r, f) for r in roots], axis=1)
                            for f in Element._fields))
        return tree_reduce(stacked).ell

    # -- the anvil LogDensity contract ------------------------------------

    def _build_guard(self):
        """Identity on the way in; zeroes non-finite gradients on the way out.

        The guard sits on the *input* rather than wrapping the likelihood,
        which makes it free.  An earlier version wrapped the whole call, and
        its VJP had to re-run the graph it was guarding: one extra forward
        pass on every gradient, measured at +21% of a value-and-gradient
        call at 1024 chains x N = 1e4 and +82% at 256 x 2000, where the
        forward is most of the call.

        It is equivalent because of what ``_sanitize`` already does.  On a
        failed lane ``mx.where`` routes an exactly-zero cotangent backwards,
        so by the time the cotangent reaches ``v`` that lane is either
        exactly zero or NaN -- the NaN arising upstream as ``0 * inf``.
        Selecting on ``isfinite`` here discards exactly those NaNs, and
        chains never interact (anvil's contract), so a failed lane cannot
        contaminate its neighbours.  Verified bit-identical to the
        wrap-everything version on every point in
        ``tests/test_likelihood.py::FAILURE_POINTS`` plus healthy lanes.
        """
        @mx.custom_function
        def guard_in(v):
            return v

        @guard_in.vjp
        def _guard_vjp(primals, cotangents, outputs):
            g = (cotangents[0] if isinstance(cotangents, (list, tuple))
                 else cotangents)
            # `mx.where` SELECTS; it never multiplies a cotangent by zero, so
            # an infinite derivative on a failed lane is discarded rather
            # than turned into NaN.
            return mx.where(mx.isfinite(g), g, mx.zeros_like(g))

        raw = self._raw_call
        return lambda v: raw(guard_in(v))

    def _check_v(self, v: mx.array) -> None:
        """``v`` is the whole model vector; a wrong width is silent otherwise.

        Too wide and the extra columns are ignored, so a mis-built parameter
        map fits the wrong thing; too narrow and MLX raises somewhere deep in
        a broadcast with no mention of the parameter map.
        """
        if v.ndim != 2:
            raise ValueError(
                f"v must be (n_chains, n_dim); got shape {tuple(v.shape)}")
        want = self.param_map.n_dim
        if v.shape[1] != want:
            raise ValueError(
                f"v has {v.shape[1]} columns but the parameter map declares "
                f"n_dim = {want}")

    def _raw_call(self, v: mx.array) -> mx.array:
        """The float32 likelihood itself, sanitised but not gradient-guarded."""
        if self._metal_ok:
            return _sanitize(self._evaluate_metal(v))
        out = self._evaluate(
            v, self._dt32, self._y32, self._r_ref32, self._x_mean32,
            self._inst32, _pow2(self.policy.block), self.policy.eval_blocks,
            uniq=self._uniq32)
        return _sanitize(out)

    def __call__(self, v: mx.array) -> mx.array:
        """float32 GP log-likelihood, offset by :attr:`log_norm_const`.

        A chain in a float32-infeasible corner gets a large-negative finite
        value **and a zero gradient**, which is what the brief asks for and
        what an adapting sampler needs.

        Sanitising the value is not enough to get the gradient.  ``_sanitize``
        routes a zero cotangent to the failed lanes, but the graph upstream
        of it has already produced an infinite or NaN derivative there, and
        ``0 * inf`` is NaN however the zero arrives.  The only way to stop it
        is to cut the graph, so the guard is a ``custom_function`` whose VJP
        *selects* zero on failed lanes rather than multiplying by it.

        The guard is an identity on the *input* ``v`` rather than a wrapper
        on the output, so it re-runs nothing: an earlier version wrapped the
        whole evaluation and cost an extra forward pass (~20%) on every
        gradient.  Neither path pays anything measurable now.

        The contract is one-way.  A rejected lane has a zero gradient; a
        zero gradient does not prove the lane was rejected, since the guard
        also zeroes a NaN that the backward produced on a healthy value.
        Stage R found one such point on the kernel path and fixed its cause
        (the Joseph-form update in the replay); the converse is now pinned
        by ``tests/test_metal.py::test_a_healthy_value_never_comes_with_a_dead_gradient``.
        """
        self._check_v(v)
        return self._guard(v)

    def log_prob_absolute(self, v: mx.array) -> mx.array:
        """Absolute log-likelihood, with the host constant added in float64.

        For parity tests and reporting -- never inside a sampler's hot loop,
        where the constant cancels and only costs precision.
        """
        out = self(v)                      # on whatever stream the caller set
        return mx.add(out.astype(mx.float64, stream=mx.cpu),
                      mx.array(self.log_norm_const, dtype=mx.float64),
                      stream=mx.cpu)

    def hi(self, v: mx.array) -> mx.array:
        """Full-float64 CPU evaluation, tiled over CHAINS.

        A GP factorisation is global-sequential over data, so unlike
        ``ChunkedGaussianLogLike.hi`` this cannot tile the data axis -- it
        streams it in recursion order and tiles chains instead.
        """
        self._check_v(v)
        pol = self.policy
        n_chains = v.shape[0]
        blk = _pow2(pol.hi_block)
        out = np.zeros(n_chains, dtype=np.float64)
        dt = a64(self._dt64)
        y = a64(self._y64)
        r_ref = a64(self._yerr64 ** 2)
        xm = None if self._x_mean64 is None else a64(self._x_mean64)
        with mx.stream(mx.cpu):
            v64 = v.astype(mx.float64)
            for c in range(0, n_chains, pol.hi_chain_block):
                vb = v64[c:c + pol.hi_chain_block]
                acc = self._evaluate(vb, dt, y, r_ref, xm, self._inst32,
                                     blk, True, uniq=self._uniq64)
                mx.eval(acc)
                out[c:c + vb.shape[0]] = np.array(acc, dtype=np.float64)
        return a64(out)

    def hi_absolute(self, v: mx.array) -> mx.array:
        """float64 absolute log-likelihood.  The addition is pinned to the CPU
        stream: float64 does not exist on Metal, and the default stream is
        the GPU in any normal session."""
        return mx.add(self.hi(v),
                      mx.array(self.log_norm_const, dtype=mx.float64),
                      stream=mx.cpu)


def _auto_sub_block(m: int) -> int:
    """Default chunked-scan sub-block length.

    Chosen for *accuracy* first, speed second, because the two agree here.
    The fold inside a sub-block is a cheap rank-one composition; the tree
    above it is the expensive general one, and the general one is also
    where float32 loses digits -- it cancels three terms of order ``g^2``
    against each other, and ``g = y/sqrt(S)`` reaches ~100 on-model when
    the GP amplitude is 100x the error bars.  So the residual error scales
    with the *number* of general compositions, ``N / sub_block``.

    Measured on-model at N = 1e5, sigma_GP/yerr = 30, and at
    1024 chains x N = 1e4 for the wall times (interleaved, same window):

        sub_block     error        forward     value+grad
            16        -0.63         1.00x         1.00x
            32        -0.17         0.92x         1.06x
            64        -0.09         0.99x         1.38x
           128        -0.043        1.49x         2.20x

    32 is the knee: essentially free on both paths and 3.7x more accurate
    than 16.  Past it accuracy is still bought, but with real time.  Raise
    it by hand for a large, high-amplitude problem -- ``validate_precision``
    on your own data is the thing to trust, and it will say so.
    """
    if m <= 6:
        return 32
    return 64


#: longest block we will hand one thread, by state dimension.  A thread
#: walks its whole block sequentially, so block length sets how long a
#: single command buffer runs, and macOS kills one that runs too long
#: ("Impacting Interactivity").  Per-observation work grows like m^3, so
#: the safe length falls as m rises.  Measured at N = 1e5: m = 2 is fine at
#: 8192 (13 ms); m = 4 at 4096 and 8192 is *killed* at 512 chains, and 2048
#: sits close enough to the edge that it failed in one run and not another.
_MAX_METAL_BLOCK = {1: 8192, 2: 8192, 3: 2048}
_MAX_METAL_BLOCK_HI_M = 1024


def _slot_map(lag: np.ndarray, blk_len: int):
    """Index map from observation to per-lag adjoint slot, and back to lag.

    The backward kernel accumulates the propagator adjoints ``dPhi``,
    ``dQ`` in registers and flushes when the lag changes.  It flushed into
    a buffer with one slot per ``(chain, block, lag)``, which is fine when
    the timestamps have one cadence plus a lag per gap and N-sized when
    every spacing is distinct -- 40 GiB at 1024 chains x N = 1e5, which is
    why :data:`anvilgp.metal.MAX_ADJOINT_BUFFER_BYTES` has to refuse.

    Most of that buffer is never written: a block of 8192 observations can
    touch at most 8192 lags, and usually touches one or two.  This maps
    each observation to a slot indexed by the ``(block, lag)`` pair it
    actually uses, so the buffer is one slot per pair that occurs, at most
    ``min(n_blocks x n_lags, N)``.

    Returns ``(slot_id, slot_start, slot_lag)``:

    * ``slot_id[t]`` -- the slot observation ``t`` flushes to.
    * ``slot_start[b] .. slot_start[b+1]`` -- the slots block ``b`` owns.
      A thread zeroes exactly this range, so it must be contiguous, which
      is why the slots are ordered by ``(block, lag)``.
    * ``slot_lag[s]`` -- the lag slot ``s`` belongs to, for folding the
      slots back into a per-lag table.

    ``blk_len`` must be the block length the kernel is launched with; a
    map built for a different one silently misattributes every adjoint,
    with the *value* still correct because the forward never reads it.
    """
    n = int(lag.size)
    n_blocks = -(-n // int(blk_len))
    blk = np.arange(n, dtype=np.int64) // int(blk_len)
    stride = int(lag.max()) + 1
    key = blk * stride + lag.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)      # sorted: (block, lag)
    slot_lag = (uniq % stride).astype(np.int32)
    slot_blk = (uniq // stride).astype(np.int32)
    slot_start = np.searchsorted(
        slot_blk, np.arange(n_blocks + 1, dtype=np.int32)).astype(np.int32)
    return inv.astype(np.int32), slot_start, slot_lag


#: Above this many blocks sharing one lag, fold through a padded table and
#: one ``mx.sum`` rather than one scatter per rank.  Which way round is
#: cheaper depends on the shape, and the two costs pull opposite ways: the
#: padded gather materialises ``n_chains x n_lags x max_rank x m^2``
#: floats, so it is right when ``n_lags`` is small and ``max_rank`` large
#: (a regular cadence: 3 lags over 98 blocks, a few kilobytes), and wrong
#: when ``n_lags`` is N and ``max_rank`` is 3 -- measured 4.9 GiB of
#: gather against 0.8 GiB of buffer at 256 chains x N = 1e5.  Grouping by
#: rank has no such intermediate but costs a launch per rank, so it wins
#: exactly when the ranks are few.  Both are reproducible.
_MAX_FOLD_RANKS = 8


def _lag_fold_plan(slot_lag: np.ndarray, n_lags: int):
    """How to fold per-slot adjoints back into a per-lag table, *reproducibly*.

    The obvious fold is one scatter-add on ``slot_lag``, and **MLX's
    scatter-add is not reproducible when indices repeat**: the accumulation
    order across threads is not fixed, so two identical calls differ in the
    last float32 digit.  Measured directly -- 14 slots over 3 lags is
    already enough -- while ``mx.sum`` over an axis is reproducible.  The
    per-lag adjoints used to come from such a sum, so scattering naively
    would have quietly cost the gradient its bit-reproducibility, and on
    the regular cadences that are the common case rather than on the
    irregular ones this work is about.

    A slot is a ``(block, lag)`` pair, so within one block every lag is
    distinct and a lag repeats only across blocks.  Ranking each slot
    within its lag gives a dense ``(n_lags, max_rank)`` table of slot
    indices, padded with a sentinel row of zeros, which folds in one
    ``mx.sum`` -- reproducible, and one launch rather than ``max_rank``.

    Where every timestamp spacing is distinct, each lag occurs in one or
    two blocks, so there are one or two groups.  A regular cadence stacks
    every block on the same handful of lags, and the padded table is then
    a few kilobytes.

    Returns ``("groups", [(slots, lags), ...])`` when the slots stack no
    more than :data:`_MAX_FOLD_RANKS` deep -- each group having distinct
    lags, so each scatter is reproducible on its own -- and
    ``("dense", table)`` of shape ``(n_lags, max_rank)`` otherwise.
    """
    n_slots = int(slot_lag.size)
    if n_slots == 0:
        return ("dense", np.zeros((n_lags, 1), dtype=np.int32))
    order = np.argsort(slot_lag, kind="stable")
    ordered = slot_lag[order]
    rank = np.arange(n_slots) - np.searchsorted(ordered, ordered, side="left")
    max_rank = int(rank.max()) + 1

    # Dense only when the ranks are many *and* the padded table is not
    # much larger than the slots it holds.  Deciding on `max_rank` alone
    # (as this first did) sent a mostly-regular cadence with thousands of
    # odd spacings -- dominant lag in every block, every other lag in one
    # -- to the dense path, whose gather is `n_lags x max_rank` against
    # `n_slots` of buffer: measured 1.61x peak at N = 2e4, and the same
    # 10x-the-buffer intermediate stage C removed, at production shape
    # (stage R2).  Ten grouped launches are the right trade there.
    if max_rank > _MAX_FOLD_RANKS and n_lags * max_rank <= 4 * n_slots:
        table = np.full((n_lags, max_rank), n_slots, dtype=np.int32)
        table[ordered, rank] = order            # sentinel stays where absent
        return ("dense", table)

    return ("groups", [(order[rank == r].astype(np.int32),
                        ordered[rank == r].astype(np.int32))
                       for r in range(max_rank)])


#: Peak bytes MLX's backward through :meth:`Term.matrices` costs per byte of
#: the ``(n_chains, n_lags, m, m)`` table it produces.  The term algebra's
#: ~20 intermediates are each of table size and all are retained, and the
#: measured ratio is 25-47x (higher at small tables, where a fixed overhead
#: shows).  48 is the conservative end: over-estimating splits the backward
#: into more chunks than strictly needed, which is cheap, while
#: under-estimating lets the peak through, which is what this exists to stop.
_FRONT_END_RETENTION = 48

#: Target peak for one chunk of that backward.  Chosen from a measured
#: sweep at N = 2e4, 64 chains, an all-distinct cadence: the unchunked
#: backward is 1.82 GiB and 41 ms; this budget picks 4 chunks, which is
#: 0.57 GiB and 28 ms -- *both* smaller and faster, because the unchunked
#: allocation costs more in bandwidth than the extra launches cost in
#: overhead.  Chunking harder keeps shrinking the peak and starts costing
#: real time (16 chunks: 0.18 GiB and 127 ms), so this is deliberately not
#: the smallest achievable peak.
_FRONT_END_BUDGET_BYTES = 512 << 20

def _front_end_chunks(n_chains: int, n_lags: int, n_state: int) -> int:
    """How many lag-chunks to split the term front end's backward into.

    Returns 1 -- meaning "do not chunk" -- for every table small enough to
    backpropagate in one piece, which is every regular cadence: one cadence
    plus a lag per gap is a handful of rows whatever N is.  Chunking those
    would cost 1.8x on the front end to save memory that is already
    negligible (measured 0.009 GiB at 1024 chains x N = 1e5).

    It only bites on a cadence whose every spacing is distinct, where the
    lag table degenerates to one row per observation and its backward
    reaches 75 GiB at production size.
    """
    table = 2 * n_chains * n_lags * n_state * n_state * 4
    want = _FRONT_END_RETENTION * table
    if want <= _FRONT_END_BUDGET_BYTES:
        return 1
    # Bounded by one lag per chunk and nothing coarser.  A floor of 64
    # lags per chunk was here first, "so a chunk stays worth a launch";
    # it silently overrode the budget above ~1365 chains at m = 4 (a chunk
    # is n_chains x lags x m^2, so few lags is plenty of work at many
    # chains), returning 1 with the estimate 3x over budget (stage R2).
    n = -(-want // _FRONT_END_BUDGET_BYTES)
    return max(1, min(n, n_lags))


def _auto_metal_block(n_data: int, n_state: int = 2) -> int:
    """Default observations per kernel thread.

    Two costs pull against each other: a longer block means fewer, longer
    sequential walks in registers (good, the kernel's whole point), but also
    fewer threads to fill the GPU with (bad below a few thousand threads).
    Aim for a few dozen blocks per chain -- N/12 rounded to a power of two --
    then cap by state dimension so a single thread cannot run long enough
    for the GPU watchdog to kill the command buffer.

    Measured, whole compiled path at 1024 chains, interleaved in one window:

        N        512    1024    2048    4096    8192   16384    rule
        1e4      3.5     2.9     3.1                            1024
        3e4      6.7     6.2     4.9     5.7                    2048
        1e5     17.5    17.2    17.2    16.6    13.3    22.2     8192

    and at m = 4 (rotation), where the cap binds:

        N = 1e4, 1024 chains:  256:110  512:102  1024:65  2048:63  4096:83
        N = 1e5,  512 chains:  256:173  512:156  1024:187  2048:191
                               4096: KILLED     8192: KILLED

    The uncapped rule picked 8192 at N = 1e5 for every m, which is a hard
    failure at m = 4 rather than a slow path.
    """
    target = max(1.0, n_data / 12.0)
    p = 1
    while p * 2 <= target:            # largest power of two <= target
        p *= 2
    if target - p > 2 * p - target:   # round to the *nearest*, not up
        p *= 2
    cap = _MAX_METAL_BLOCK.get(n_state, _MAX_METAL_BLOCK_HI_M)
    return max(256, min(cap, p))


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _sanitize(x: mx.array) -> mx.array:
    """Positive-definiteness failure is a defined, finite outcome.

    With 4096 chains, warmup *will* visit float32-infeasible corners --
    sigma_GP/yerr of 10-300 with a timescale far above the cadence makes the
    innovation-variance update a catastrophic cancellation, and ``phi``
    rounds to exactly 1.0 for ``c dt <~ 6e-8``.  A chain that lands there
    must get a large-negative *finite* log-likelihood and be rejected, not
    a NaN that poisons the whole ensemble's adaptation.

    This makes the **value** finite.  It does not on its own make the
    gradient finite: ``mx.where``'s VJP does route an exactly-zero
    cotangent to the failed lane, but the graph above has already produced
    an infinite derivative there, and ``0 * inf`` is NaN however the zero
    arrives.  Cutting that is the input-side guard's job; see
    :meth:`GPLogLike.__call__`.
    """
    bad = mx.logical_not(mx.isfinite(x))
    safe = mx.where(bad, mx.zeros_like(x), x)
    return mx.where(bad, mx.full(x.shape, _LARGE_NEGATIVE, dtype=x.dtype), safe)
