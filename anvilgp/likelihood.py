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
The default 16-chain x 1024-point fp64 tile at m = 4 is ~60 MB; an untiled
fp64 pass over 1024 chains x 1e5 points would transiently ask for ~50 GB.

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

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .special import exp_acc
from .elements import (Element, chunked_reduce, chunked_reduce_blocked,
                       combine, leaf_elements, leaf_parts, tree_reduce,
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
    #: use the fused Metal scan kernel (:mod:`anvilgp.metal`).  **Forward
    #: only** until M4 adds its VJP -- differentiating through it raises
    #: rather than silently falling back, so a gradient sampler must leave
    #: this off.  Measured 37-71x over the MLX path with 15-23x less peak
    #: memory; see the README.
    metal: bool = False
    #: observations condensed per kernel thread.  Larger means fewer, longer
    #: sequential walks and less work left for the MLX tree above; 1024 was
    #: best at N = 1e4 and 4096 at N = 1e5.
    metal_block: int = 1024
    #: floor on the leaf innovation variance, in units of the reference
    #: variance.  Positive-definiteness failure is a *defined* outcome:
    #: with thousands of chains, warmup will visit float32-infeasible
    #: corners, and the likelihood must come back finite there.
    s_floor_rel: float = 1e-6


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
    index: int = -1
    ref: float = 1.0
    scale: float = 1.0
    value: float = 0.0

    def apply(self, v: mx.array) -> mx.array:
        # exp_acc, not mx.exp: MLX's float64 exp is only float32-accurate
        # (~7e-7 relative), which would put a float32 floor under the whole
        # ".hi" verification path -- measured at 2.1e-4 log-likelihood units
        # at N = 2048 with a log link, i.e. 1e-2 at N = 1e5, which is the
        # size of the float32 error .hi exists to measure.  In float32
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
                unit = float(k.ref) if k.link == "log" else 1.0
                cols[hi] = cols[lo] * (1.0 + cols[hi] / unit)
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
        d = np.diff(self._t64)
        if np.any(d < 0.0):
            raise ValueError(
                "t must be sorted ascending (duplicates allowed). "
                "Sort your data and reorder y/yerr to match.")
        if np.any(self._yerr64 <= 0.0):
            raise ValueError("yerr must be strictly positive")

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
        self._rref32 = mx.array((self._yerr64 ** 2).astype(np.float32))
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

        # Decided once, not per call: whether the fused kernel can serve this
        # problem, and the gradient guard that wraps whichever path does.
        self._metal_ok, self._metal_why = self.metal_eligible()
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

    def leaf_inputs(self, v, dt, y, rref, x_mean, idx, s, e, get_mat=None):
        """``(phi, q, h, residual, r_var, r_ref, s_floor)`` for one data block.

        Exactly the argument list of :func:`~anvilgp.elements.leaf_elements`
        and :func:`~anvilgp.kalman.sequential_loglike`, so the tree reduction,
        the sequential-filter oracle and (in M3) the Metal kernel all consume
        byte-identical inputs and any disagreement between them is
        attributable to the recursion alone.
        """
        pm = self.param_map
        dtype = v.dtype
        n = v.shape[0]
        sub = None if idx is None else idx[s:e]

        if get_mat is None:
            get_mat = self._matrices_fn(pm.kernel_params(v), dt, None)
        phi, q, h = get_mat(s, e)
        if s == 0:
            # First point: no predecessor.  Phi = 0 with Q = P_inf = I makes
            # the generic leaf produce A = 0, J = 0, eta = 0 -- exactly the
            # "no incoming state" element, with no special-cased code path.
            keep = mx.concatenate([
                mx.zeros((1, 1, 1, 1), dtype=dtype),
                mx.ones((1, e - s - 1, 1, 1), dtype=dtype)], axis=1)
            if self.blocked:
                phi = [b * keep for b in phi]
                q = [b * keep + (1.0 - keep) * mx.eye(b.shape[-1], dtype=dtype)
                     for b in q]
            else:
                phi = phi * keep
                q = (q * keep
                     + (1.0 - keep) * mx.eye(self.term.n_state, dtype=dtype))

        r = mx.broadcast_to(y[s:e], (n, e - s))
        if self.mean_fn is not None and pm.mean is not None:
            r = r - self.mean_fn(v[:, pm.mean], x_mean[..., s:e])
        off = pm.offsets(v, sub)
        if off is not None:
            r = r - off

        r_ref = rref[s:e]
        jit2 = pm.jitter_var(v, sub)
        r_var = r_ref if jit2 is None else r_ref + jit2
        r_var = mx.broadcast_to(r_var, r.shape)
        s_floor = self.policy.s_floor_rel * r_ref
        return phi, q, h, r, r_var, r_ref, s_floor

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

    def _evaluate_metal(self, v: mx.array) -> mx.array:
        """Design C: the kernel condenses blocks, MLX composes the roots."""
        from .metal import scan_blocks
        pm = self.param_map
        n, m, dt = v.shape[0], self.term.n_state, v.dtype
        phi, q, h = self.term.matrices(pm.kernel_params(v),
                                       self._metal_uniq_dt)
        pt = mx.concatenate(
            [mx.zeros((n, 1, m, m), dtype=dt), phi], axis=1)
        qt = mx.concatenate(
            [mx.broadcast_to(mx.eye(m, dtype=dt), (n, 1, m, m)), q], axis=1)

        r = mx.broadcast_to(self._y32, (n, self.n_data))
        if self.mean_fn is not None and pm.mean is not None:
            r = r - self.mean_fn(v[:, pm.mean], self._x_mean32)
        off = pm.offsets(v, self._inst32)
        if off is not None:
            r = r - off
        jit = pm.jitter_var(v, self._inst32)
        jit2 = (mx.zeros((n, 0), dtype=dt) if jit is None
                else mx.broadcast_to(jit, (n, 1)))

        el = scan_blocks(pt, qt, self._metal_lag, h, r, self._rref32, jit2,
                         self.policy.s_floor_rel, self.policy.metal_block)
        return tree_reduce(el).ell

    def _evaluate(self, v, dt, y, rref, x_mean, idx, block, do_eval, uniq=None):
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
            args = self.leaf_inputs(v, dt, y, rref, x_mean, idx, s, e, get_mat)
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
        """``custom_function`` wrapper that zeroes gradients on failed lanes."""
        raw = self._raw_call

        @mx.custom_function
        def guard(v):
            return raw(v)

        @guard.vjp
        def _guard_vjp(primals, cotangents, outputs):
            v = primals[0] if isinstance(primals, (list, tuple)) else primals
            cot = (cotangents[0] if isinstance(cotangents, (list, tuple))
                   else cotangents)
            out = (outputs[0] if isinstance(outputs, (list, tuple))
                   else outputs)
            _, grads = mx.vjp(raw, [v], [cot])
            g = grads[0]
            z = mx.zeros_like(g)
            # `mx.where` SELECTS; it never multiplies a cotangent by zero, so
            # an infinite derivative on a failed lane is discarded rather
            # than turned into NaN.
            failed = out <= _FAILED
            g = mx.where(failed[:, None], z, g)
            return mx.where(mx.isfinite(g), g, z)

        return guard

    def _raw_call(self, v: mx.array) -> mx.array:
        """The float32 likelihood itself, sanitised but not gradient-guarded."""
        if self._metal_ok:
            return _sanitize(self._evaluate_metal(v))
        out = self._evaluate(
            v, self._dt32, self._y32, self._rref32, self._x_mean32,
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

        That costs one extra forward pass on the gradient path, because the
        VJP has to re-run the graph it is guarding.  Measured at ~20% of a
        value-and-gradient call.  The forward-only path -- the stretch move,
        and anything on the Metal kernel -- pays nothing.
        """
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
        pol = self.policy
        n_chains = v.shape[0]
        blk = _pow2(pol.hi_block)
        out = np.zeros(n_chains, dtype=np.float64)
        dt = a64(self._dt64)
        y = a64(self._y64)
        rref = a64(self._yerr64 ** 2)
        xm = None if self._x_mean64 is None else a64(self._x_mean64)
        with mx.stream(mx.cpu):
            v64 = v.astype(mx.float64)
            for c in range(0, n_chains, pol.hi_chain_block):
                vb = v64[c:c + pol.hi_chain_block]
                acc = self._evaluate(vb, dt, y, rref, xm, self._inst32,
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
    """Default chunked-scan sub-block length for a state dimension."""
    if m <= 4:
        return 16
    if m <= 6:
        return 32
    return 64


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

    Both branches are sanitised: the good branch sees a finite value even on
    failed lanes, so ``mx.where``'s VJP contributes zero rather than
    ``0 * inf``.
    """
    bad = mx.logical_not(mx.isfinite(x))
    safe = mx.where(bad, mx.zeros_like(x), x)
    return mx.where(bad, mx.full(x.shape, _LARGE_NEGATIVE, dtype=x.dtype), safe)
