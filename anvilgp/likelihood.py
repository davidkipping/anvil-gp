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
mean model already has (Elome's is analytic).  So the hard gradient
engineering is confined to the hyperparameters; the transit path is
already solved and rides ordinary autodiff through ``mean_fn``.

Known v1 limitations, stated rather than discovered
---------------------------------------------------
* Exposure-time integration (:class:`~anvilgp.terms.TermConvolution`)
  rests on a formal state-space form (indefinite ``P_inf``, process noise by
  subtraction).  Its lag table is computed in float64 on the host since
  2026-10-25 (`_Float64FrontEnd`), which removed a ~10x float32 floor on
  bright data, and SHO, rotation and Matern members use the kernel's
  square-root block boundary with ``Q``'s negative part clamped while the
  exposure is short against the timescale (``||F delta|| <= 0.5`` at the
  reference point): a TESS 2-min convolved rotation at s/e = 100 went from
  10 posterior sigma and value sd 0.32 to 0.63 and 0.016.  ``RealTerm``
  members, and longer exposures, keep the dense boundary.  The exposure must stay
  below the validated ``||F delta|| <= 8``.  See docs/boundary-plan.md,
  VI.
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
from . import terms
from .terms import (Matern32Term, Matern52Term, RotationTerm, SHOTerm, Term,
                    TermSum, _block_diag)

__all__ = ["GPLogLike", "GPPolicy", "ParamMap", "KernelParam", "a64"]

_LARGE_NEGATIVE = -1e30
#: anything at or below this is a sanitised failure, not a real value.
_FAILED = 0.5 * _LARGE_NEGATIVE
#: |v| beyond which the infeasible-region pull stops growing: keeps v**2
#: (<= 1e30) and so the sentinel (>= -7e31) finite in float32.
_PULL_CLIP = 1e15


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
    #: ~3 m^3, so the speed optimum climbs with m (measured at 1024 chains
    #: x N = 1e4: 16 at m = 2 and 4, 32 at m = 6, 64 at m = 8).  The default
    #: is 32 up to m = 6 and 64 above, for accuracy: 32 is 3.7x more
    #: accurate than 16 at no measurable cost (`_auto_sub_block`).
    #: See :func:`anvilgp.elements.chunked_reduce`; 1 recovers the pure tree.
    sub_block: int | None = None
    #: chains per tile of the float64 ``.hi`` pass.  Bit-identical at any
    #: value; 64 was 3.1-3.3x faster than 16 (SHO N = 2000, SHO+SHO
    #: N = 2e4, Rotation N = 5000 x 256 chains) at a peak of <= 0.06 GiB.
    hi_chain_block: int = 64
    hi_block: int = 1024
    eval_blocks: bool = False
    #: keep leaves in compact block form (None = decide from the term's block
    #: structure).  See :func:`anvilgp.elements.use_blocked_leaves`.
    blocked: bool | None = None
    #: use the fused Metal scan kernel (:mod:`anvilgp.metal`): ``None``
    #: (the default since 2026-10-25) uses it wherever it is eligible and
    #: the MLX path otherwise, saying once per process which it chose;
    #: ``True`` asks for it and warns if it cannot be honoured; ``False``
    #: is the MLX path.  The kernel became the default because it is now
    #: the more accurate path on bright, densely sampled data (value and
    #: gradient), the only one whose gradient runs at N ~ 1e5, and 17-200x
    #: faster (docs/boundary-plan.md, I).  Differentiable since M4: the
    #: kernel carries its own VJP.  Limits: m <= 6 (any number of
    #: per-instrument jitters, and any `noise_fn`, since 2026-10-25).  The
    #: gradient's adjoint buffer is one slot per ``(block, lag)`` pair
    #: that *occurs*, so it is bounded by N rather than by N x blocks,
    #: and an irregular cadence is affordable at m = 2 (3.05 GiB at 1024
    #: chains x N = 1e5).  It is **not** unconditional: the buffer scales
    #: with ``m^2``, so the same cadence at m = 4 needs 12.2 GiB and
    #: :data:`anvilgp.metal.MAX_ADJOINT_BUFFER_BYTES` (8 GiB) refuses it.
    metal: bool | None = None
    #: observations condensed per kernel thread, or None to pick from N.
    #: Larger means fewer, longer sequential walks and less work left for
    #: the MLX tree above, which is launch-bound and is most of the
    #: remaining wall time at small N.  See :func:`_auto_metal_block`.
    metal_block: int | None = None
    #: how the fused kernel forms the state each block starts from and the
    #: adjoint each block's backward is seeded with: ``"sqrt"``, the
    #: square-root information form centred on the replayed filter, or
    #: ``"dense"``, the dense information form it replaced.  ``None`` picks
    #: ``"sqrt"``, except for a convolved term with a `RealTerm` member or
    #: an exposure long against the timescale (``||F delta|| > 0.5``), whose
    #: indefinite ``Q`` the square-root form cannot clamp harmlessly, which
    #: keeps ``"dense"``.  On bright, densely sampled data the dense form's
    #: float32 boundaries left the gradient 4-5 posterior sigma off at the
    #: default blocks and ChEES did not converge; see docs/boundary-plan.md
    #: and docs/float32.md, section 7.  ``"dense"`` stays, as a speed option
    #: for samplers that use the value alone, at m <= 3 only: a 1.6-1.7x
    #: faster forward, its value within sd 0.012 of float64 across the
    #: posterior at every m <= 3 row measured; on bright data 2-4x worse
    #: than the default's at m = 4 and up to 15x at m = 6.  Its gradient is
    #: 1.6-1.8 sigma off even at m = 2 (docs/boundary-plan.md, follow-on
    #: step 4).
    metal_boundary: str | None = None
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
    #: refuse a chain whose conditioning number :meth:`GPLogLike.kappa`
    #: exceeds this, returning the failure value instead of a float32
    #: evaluation; ``None`` evaluates everywhere.  Above ~1e6 float32 is
    #: off by about a log-unit or more where a posterior actually lives
    #: (data drawn at those parameters), and above ~3e7 the evaluation
    #: fails outright.  Chains that wander there are what collapse ChEES's
    #: shared step size: the error makes the density rough enough that only
    #: steps below float32's resolution of ``u`` are "accepted".  See
    #: docs/float32.md.
    kappa_max: float | None = 1e6

    def __post_init__(self):
        if self.kappa_max is not None and not self.kappa_max > 0:
            raise ValueError(
                f"kappa_max = {self.kappa_max!r} must be positive or None")
        if self.metal_block is not None and int(self.metal_block) < 1:
            raise ValueError(
                f"metal_block = {self.metal_block!r} must be None or >= 1")
        if (self.metal_block is not None
                and int(self.metal_block) > max(_MAX_METAL_BLOCK.values())):
            # `_auto_metal_block` caps this per state dimension because the
            # GPU watchdog *kills the command buffer* above the cap at
            # m = 4 -- a hard failure, not a slow path.  An explicit
            # `metal_block` used to bypass that cap entirely.  The exact
            # per-m cap needs the term, which a policy does not have, so
            # refuse only what no m allows and let the instance check the
            # rest.
            raise ValueError(
                f"metal_block = {self.metal_block} exceeds "
                f"{max(_MAX_METAL_BLOCK.values())}, which the GPU watchdog "
                f"cannot survive at any state dimension; see "
                f"_auto_metal_block for the per-m caps")
        if self.metal_boundary not in (None, "sqrt", "dense"):
            raise ValueError(
                f"metal_boundary = {self.metal_boundary!r} must be None, "
                f"'sqrt' or 'dense'")
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
            # sigmoid with no exp that can overflow: z = e^{-|v|} <= 1.  The
            # textbook 1 / (1 + e^{-v}) has a float32 gradient of
            # inf / inf = NaN past v ~ -89 (and loses it past ~ -44), which
            # the guard zeroed -- a healthy value with a dead gradient.  |v|
            # comes from a `where`, not `abs`, so the gradient at v = 0 is
            # right (abs's is 0 there), and both branches stay finite, so
            # the unselected one cannot leak a NaN through `where`'s VJP.
            # exp_acc keeps the float64 path exact; mx.sigmoid is not.
            neg_abs = mx.where(col >= 0, -col, col)
            z = exp_acc(neg_abs)
            return mx.where(col >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
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
    #: the columns `GPLogLike`'s ``noise_fn`` reads, raw model-space values
    #: like the mean block's (their ``0`` is not a reference)
    noise: slice | None = None
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

    def jitter_var_per_instrument(self, v: mx.array) -> mx.array | None:
        """``(n_chains, n_jitter)`` variances, **one column per instrument**.

        Not the same thing as :meth:`jitter_var`, which expands to one
        column per *datum* through ``mx.take`` and needs the instrument
        index.  The fused kernel wants the compact form and does its own
        indexing per observation, so it needs this; the MLX path wants the
        expanded form because it multiplies elementwise.
        """
        if not self.jitter:
            return None
        j = mx.concatenate([s.apply(v) for s in self.jitter], axis=-1)
        return j * j

    def offsets(self, v: mx.array, idx: mx.array | None) -> mx.array | None:
        return self._per_datum(v, self.offset, idx)


class GPLogLike:
    """GP marginal log-likelihood over a batch of parameter vectors.

    ``mean_fn(v_mean, x)`` is anvil's batched model contract -- parameters
    ``(n_chains, k)`` and abscissa ``(m,)`` or ``(c, m)`` -> predictions
    ``(n_chains, m)`` -- or ``None`` for a pure-GP fit.

    ``t`` is the physical time vector (float64, host).  ``x_mean`` is the
    mean model's *own* abscissa: for Elome that is the ``(2, m)`` stack
    of epoch-centred residual times and orbit numbers, which is a different
    array from ``t`` and must not be conflated with it -- ``t`` drives the
    GP's lag structure, ``x_mean`` drives the transit.

    Sorted input is required (unsorted raises).  Duplicate timestamps are
    legal: celerite2 accepts them, and ``dt = 0`` gives ``Phi = I``,
    ``Q = 0`` exactly here, with no NaN.

    ``noise_fn(v_noise, d)`` is the general noise model, celerite2's
    ``diag``: the **total** per-datum noise variance ``(n_chains, m)``,
    from the noise columns of ``v`` (``param_map.noise``, raw model-space
    values) and ``d``, an `anvilgp.noise.NoiseData` of per-datum arrays
    sliced alike (``yerr``, the abscissa ``x`` -- ``x_mean`` if given,
    else ``t`` -- and the instrument labels).  It must act pointwise along
    the data axis: the reference path evaluates it per block.  ``None``
    keeps ``yerr^2 + jitter^2`` from ``param_map.jitter``; the two cannot
    be combined (write the jitter into ``noise_fn``).  `anvilgp.noise` has
    the common forms, and says why a hand-written one should use
    `special.exp_acc` rather than ``mx.exp``.  The variance must stay above
    ``s_floor_rel * yerr^2`` (1e-6 by default), where the innovation floor
    starts to bind.
    """

    def __init__(self, mean_fn, t, x_mean, y, yerr, kernel_spec: Term,
                 param_map: ParamMap, policy: GPPolicy | None = None,
                 noise_fn=None):
        self.mean_fn = mean_fn
        self.noise_fn = noise_fn
        if param_map.noise is not None and noise_fn is None:
            raise ValueError(
                "`param_map.noise` is set but `noise_fn` is None: the noise "
                "columns would be sampled but inert.")
        if noise_fn is not None and param_map.jitter:
            raise ValueError(
                "`noise_fn` gives the whole noise variance, so it cannot be "
                "combined with `param_map.jitter`; write the jitter into "
                "`noise_fn` (see `anvilgp.noise.scale_and_jitter`).")
        self.term = kernel_spec
        self.param_map = param_map
        #: The columns of ``v`` whose reference point is ``v = 0`` by
        #: construction -- the GP's own kernel, jitter and offset parameters.
        #: The infeasible-region sentinel slopes towards that point; see
        #: `_sanitize`.  Mean-model columns are excluded deliberately: their
        #: model-space ``0`` is not a reference (a transit's ``a/R* = 0`` is
        #: not even inside its bounds), and pulling them there would be wrong.
        cols = sorted({k.index for k in (list(param_map.kernel)
                                         + list(param_map.jitter)
                                         + list(param_map.offset))
                       if k.link != "const" and k.index is not None})
        self._pull_cols = (mx.array(np.array(cols, dtype=np.int32))
                           if cols else None)
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
        # Instrument label per observation, for the kernel's per-instrument
        # jitter.  Always present and always int32: the kernel indexes it
        # unconditionally in the multi-jitter variant, and a zero vector is
        # the right answer when there is one instrument or none.
        self._metal_inst = (mx.zeros((self.n_data,), dtype=mx.int32)
                            if param_map.instrument is None
                            else mx.array(np.asarray(param_map.instrument,
                                                     dtype=np.int32)))
        self.n_distinct_lags = (self.n_data if self._uniq32 is None
                                else int(self._uniq32[0].size))
        self._y32 = mx.array(self._y64.astype(np.float32))
        self._r_ref32 = mx.array((self._yerr64 ** 2).astype(np.float32))
        self._x_mean32 = (None if self._x_mean64 is None
                          else mx.array(self._x_mean64.astype(np.float32)))
        # `noise_fn`'s abscissa: the mean model's own if there is one.
        self._noise_x64 = (self._t64 if self._x_mean64 is None
                           else self._x_mean64)
        self._noise_x32 = mx.array(self._noise_x64.astype(np.float32))
        inst = param_map.instrument
        if inst is not None:
            # `mx.take` is silent on an out-of-range label -- it clamps or
            # wraps -- so a typo'd instrument would quietly score against
            # the wrong jitter.  In the kernel it would index the *next
            # chain's* row, or past the buffer on the last chain.
            n_lab = max(len(param_map.jitter), len(param_map.offset))
            lab = np.asarray(inst)
            if lab.shape != (len(self._y64),):
                raise ValueError(
                    f"param_map.instrument has shape {lab.shape}, expected "
                    f"({len(self._y64)},) -- one label per observation")
            if n_lab and (lab.min() < 0 or lab.max() >= n_lab):
                raise ValueError(
                    f"param_map.instrument labels run {lab.min()}..{lab.max()}"
                    f" but only {n_lab} instrument(s) are declared "
                    f"({len(param_map.jitter)} jitter, "
                    f"{len(param_map.offset)} offset)")
        self._inst32 = (None if inst is None
                        else mx.array(np.asarray(inst, dtype=np.int32)))
        self._init_kappa(inst)

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

        # A convolved term's kernel is the exponential sum only at lags at
        # or beyond its exposure width.  Inside it the sum is the wrong
        # function, and for a term with a real component the observed
        # matrix goes indefinite (measured min eigenvalue -3.0e-2 at
        # delta = 0.1), which makes the innovation variance negative and
        # the likelihood NaN.  Zero spacing is the case that bites, and
        # this library supports duplicate timestamps -- so the check
        # includes it.  Two instruments observing simultaneously is this
        # same case, not a separate one.
        delta = getattr(kernel_spec, "delta", None)
        if delta is not None and self.n_data > 1:
            gaps = np.diff(self._t64)
            worst = float(gaps.min())
            if worst < delta * (1.0 - 1e-9):
                where = int(np.argmin(gaps))
                raise ValueError(
                    f"{type(kernel_spec).__name__} has an exposure width of "
                    f"{delta:g}, but observations {where} and {where + 1} are "
                    f"{worst:g} apart. Every spacing must be at least the "
                    f"exposure width: inside it the convolved kernel is not "
                    f"the sum of exponentials the state-space form "
                    f"evaluates, and for a real component the covariance "
                    f"goes indefinite. Duplicate timestamps (spacing 0) are "
                    f"the common case -- bin them, or drop the convolution.")

        # A convolved term's matrix functions are series in `(F delta)^2`.
        # The coefficients are positive, `F delta` is not, so far enough out
        # the alternating powers cost float32 digits -- and unlike the
        # amplitude cancellation this replaced, nothing analytic removes it.
        # Refuse beyond the validated range rather than degrade quietly.
        # `||F delta|| > 8` means the exposure is longer than the fastest
        # correlation time, where the process is nearly white over a frame.
        #: spectral ||F delta|| at the reference point, for a convolved term
        #: (`_convolution_takes_sqrt`)
        self._ref_fdelta = None
        if delta is not None:
            ref = np.zeros((1, param_map.n_dim), dtype=np.float64)
            with mx.stream(mx.cpu):
                kp = param_map.kernel_params(mx.array(ref, dtype=mx.float64))
                gens = kernel_spec.term.generator(kp)
                norms = [float(np.max(np.abs(np.asarray(g)))) * delta
                         for g in gens]
                self._ref_fdelta = max(
                    float(np.linalg.norm(np.asarray(g)[0], 2)) * delta
                    for g in gens)
                probe = [np.asarray(x) for x in kernel_spec.p_inf_blocks(kp)]
                corr = kernel_spec.diag_correction(kp)
                if corr is not None:
                    probe.append(np.asarray(corr))
            worst_fd = max(norms)
            if worst_fd > terms.MAX_FDELTA:
                raise ValueError(
                    f"{type(kernel_spec).__name__} has ||F delta|| = "
                    f"{worst_fd:.3g} at the parameter map's reference point, "
                    f"above the validated {terms.MAX_FDELTA:g}. The "
                    f"convolution's matrix functions are series in "
                    f"(F delta)^2 and lose float32 digits beyond it. The "
                    f"exposure width is then longer than the kernel's "
                    f"fastest correlation time, so the process is nearly "
                    f"white across one frame -- model it as white noise, or "
                    f"bin.")
            if not all(np.all(np.isfinite(x)) for x in probe):
                raise ValueError(
                    f"{type(kernel_spec).__name__} is not finite at the "
                    f"parameter map's reference point "
                    f"({np.asarray(kp)[0].tolist()}).")

        #: Whether the term wants a stationary covariance other than `I`.
        #: Structural, so it is settled once here: every term but a
        #: convolved one returns `None`, and asking per evaluation would
        #: cost a second `kernel_params` on the hot path for nothing.
        with mx.stream(mx.cpu):
            self._has_p_inf = kernel_spec.p_inf_blocks(
                param_map.kernel_params(
                    mx.zeros((1, param_map.n_dim), dtype=mx.float64))
            ) is not None
        #: Whether the state-space form is only *formally* Gaussian, which is
        #: exactly when ``P_inf`` is not the identity: a convolved term.  Its
        #: ``Q`` can be indefinite for ordinary parameters and the algebra
        #: is still exact, so long as nothing assumes positivity: on this
        #: path the variance clamps are dropped and the block composition
        #: uses a direct inverse instead of a Cholesky factor.  See
        #: `elements.combine`.  Every other term is untouched.
        self._formal = self._has_p_inf
        if self._formal:
            # its lag table in float64: see _Float64FrontEnd
            self.term = _Float64FrontEnd(self.term)
        sqrt_ok = (not self._formal
                   or _convolution_takes_sqrt(self.term._term.term,
                                              self._ref_fdelta))
        if self.policy.metal_boundary == "sqrt" and not sqrt_ok:
            raise ValueError(
                "GPPolicy(metal_boundary='sqrt') cannot serve this convolved "
                "term: the square-root boundary clamps its (formally "
                "indefinite) Q, which is measured safe only for SHOTerm, "
                "RotationTerm, Matern32Term and Matern52Term members -- a "
                "RealTerm's exposure correction drives the leaf's white "
                "noise below zero -- and only up to ||F delta|| = "
                f"{_SQRT_FDELTA:g} at the reference point (here "
                f"{self._ref_fdelta:.3g}).  Use None (which picks 'dense' "
                "here) or 'dense'")
        #: the kernel's block boundary; see GPPolicy.metal_boundary
        self._metal_boundary = (self.policy.metal_boundary
                                or ("sqrt" if sqrt_ok else "dense"))

        self._lag_np = lag
        # Assigning `metal_block` (a property) builds the kernel's slot map
        # from it, so the map can never disagree with the launch.  Stage R2
        # showed what a stale map does: a correct value with a gradient
        # wrong by 2.6x, or a read past the end of the block table that
        # hangs the process at 0% CPU.
        if self._may_use_metal() and self.policy.metal_block is not None:
            cap = _MAX_METAL_BLOCK.get(kernel_spec.n_state,
                                       _MAX_METAL_BLOCK_HI_M)
            if int(self.policy.metal_block) > cap:
                # Allowed -- one long block is a real option, e.g. on a
                # quiet machine -- but it is the GPU watchdog's trigger:
                # say so once, here, rather than let a long run die.
                warnings.warn(
                    f"metal_block = {self.policy.metal_block} exceeds the "
                    f"automatic cap of {cap} at m = {kernel_spec.n_state}: "
                    f"threads that long can be killed by the GPU watchdog "
                    f"('Impacting Interactivity') while the display is in "
                    f"use.  See likelihood._MAX_METAL_BLOCK.")
        self.metal_block = (self.policy.metal_block
                            if self.policy.metal_block is not None
                            else _auto_metal_block(self.n_data,
                                                   kernel_spec.n_state))

        # Decided once, not per call: whether the fused kernel can serve this
        # problem, and the gradient guard that wraps whichever path does.
        self._metal_ok, self._metal_why = self.metal_eligible()
        if self.policy.metal is None:
            _notice_metal_choice(self._metal_ok, self._metal_why)
        if self.policy.metal is True and not self._metal_ok:
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
            pin = (self._p_inf(pm.kernel_params(v), dtype)
                   if self._has_p_inf else None)
            phi, q = first_point(phi, q, dtype, pin)

        r = mx.broadcast_to(y[s:e], (n, e - s))
        if self.mean_fn is not None and pm.mean is not None:
            r = r - self.mean_fn(v[:, pm.mean], x_mean[..., s:e])
        off = pm.offsets(v, sub)
        if off is not None:
            r = r - off

        r_ref = r_ref[s:e]
        jit2 = (self._noise_extra(v, r_ref, s, e) if self.noise_fn is not None
                else pm.jitter_var(v, sub))
        # A convolved term is not a sum of exponentials below its exposure
        # width, and the difference at zero lag is a constant on the
        # diagonal.  It is a function of the sampled parameters, so it goes
        # here rather than into `r_ref`, and MLX differentiates it.
        corr = self.term.diag_correction(pm.kernel_params(v))
        if corr is not None:
            jit2 = corr if jit2 is None else jit2 + corr
        r_var = r_ref if jit2 is None else r_ref + jit2
        r_var = mx.broadcast_to(r_var, r.shape)
        s_floor = self.policy.s_floor_rel * r_ref
        return phi, q, h, r, r_var, r_ref, s_floor

    def _noise_extra(self, v: mx.array, r_ref: mx.array, s: int,
                     e: int) -> mx.array:
        """``noise_fn``'s variance minus ``r_ref`` over data ``[s, e)``,
        ``(n_chains, e - s)``: the part the likelihood adds to the
        reference variance, as it adds a jitter."""
        from .noise import NoiseData
        pm = self.param_map
        vn = v[:, pm.noise] if pm.noise is not None else v[:, :0]
        x = (self._noise_x32 if v.dtype == mx.float32
             else mx.array(self._noise_x64, dtype=v.dtype))
        inst = None if self._inst32 is None else self._inst32[s:e]
        var = self.noise_fn(vn, NoiseData(mx.sqrt(r_ref), x[..., s:e], inst))
        return mx.broadcast_to(var, (v.shape[0], e - s)) - r_ref

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
        """Per-(block, lag) slot map and fold plan for ``self.metal_block``.

        Only the Metal backward reads this, so it is skipped when the
        policy does not ask for the kernel: at N = 1e5 it is an
        ``np.unique`` over 1e5 keys plus an argsort and three device
        arrays, paid by every float64-only or MLX-path instance, and
        `error_scaling.py`-style sweeps build many.  The setter still
        rebuilds on every assignment, so the never-stale property holds.
        """
        if not self._may_use_metal():
            # Deferred, not skipped: `_evaluate_metal` rebuilds on first
            # use, so flipping `policy.metal` after construction still
            # works.  `_slot_map_blk` stays unset so the launch guard
            # cannot mistake a deferred map for a stale one.
            self._metal_slot_id = self._metal_slot_start = None
            self._metal_slot_lag = self._metal_fold_plan = None
            self._slot_map_blk = None
            self.n_metal_slots = 0
            return

        sid, sstart, slag = _slot_map(self._lag_np, self._metal_block)
        #: the block length this map was built for.  Comparing block
        #: *counts* is not enough: 512 and 600 both give 2 blocks at
        #: N = 1000, so a bypassed change between them passed the guard
        #: and misattributed adjoints with the value still exact.
        self._slot_map_blk = int(self._metal_block)
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

    def _may_use_metal(self) -> bool:
        """Whether this instance can end up on the kernel: asked for, or
        left to the default with a state dimension the kernel holds."""
        from .metal import MAX_STATE
        if self.policy.metal is None:
            return self.term.n_state <= MAX_STATE
        return bool(self.policy.metal)

    def metal_eligible(self) -> tuple[bool, str]:
        """Whether the fused kernel can serve this problem, and why not."""
        from .metal import MAX_STATE, metal_available
        if self.policy.metal is False:
            return False, "GPPolicy.metal is off"
        if self.term.n_state > MAX_STATE:
            return False, (f"m = {self.term.n_state} exceeds the kernel's "
                           f"register budget (max {MAX_STATE})")
        if len(self.param_map.jitter) > 1 and self.param_map.instrument is None:
            return False, ("per-instrument jitter needs "
                           "param_map.instrument to say which datum is which")
        if not metal_available():
            return False, "mx.fast.metal_kernel is unavailable here"
        return True, ""

    def _p_inf(self, kp: mx.array, dtype, dense: bool = False):
        """The term's stationary covariance, or ``None`` for the identity.

        ``None`` is the fast path and the common one: every term but
        :class:`~anvilgp.terms.TermConvolution` keeps ``P_inf = I``, and then
        row 0 of the lag table is a constant that needs no per-chain array.

        ``dense`` assembles the blocks into one ``m x m`` matrix, which is
        what the Metal kernel's table row and the unblocked leaves want; the
        blocked leaves take the list.
        """
        blocks = self.term.p_inf_blocks(kp)
        if blocks is None:
            return None
        blocks = [b.astype(dtype) for b in blocks]
        if dense or not self.blocked:
            return _block_diag(*blocks) if len(blocks) > 1 else blocks[0]
        return blocks

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

        # The chunk geometry depends only on (n_chains, n_lags, m), all
        # fixed for an instance after the first call, so build the
        # custom_function once: it is on the hot path (one or two per MCMC
        # step) and a fresh primitive each time defeats any graph-level
        # caching keyed on identity.  Same pattern as
        # `metal._bwd_ckpt_kernels`.
        cached = getattr(self, "_metal_matrices_fn", None)
        if cached is not None and cached[0] == (v.shape[0], n_chunk):
            return cached[1](v)

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

        self._metal_matrices_fn = ((v.shape[0], n_chunk), core)
        return core(v)

    def _evaluate_metal(self, v: mx.array):
        """Design C: one kernel condenses the blocks, a second folds them.

        The whole thing sits behind :func:`anvilgp.metal.scan_loglike`'s
        ``custom_function``, whose VJP runs the backward kernels, so this
        path is differentiable (M4).  Returns ``(log L, (Q, h))``: the lag
        table goes on to :meth:`kappa`.
        """
        from .metal import scan_loglike
        pm = self.param_map
        n, m, dtype = v.shape[0], self.term.n_state, v.dtype
        if self._metal_slot_id is None:        # deferred at construction
            self._build_slot_map()
        # The property setter makes this unreachable; it stays because the
        # failure it guards is silent (see `metal_block`) and the check is
        # a host-side comparison of the block length the map was built for.
        if self._slot_map_blk != self.metal_block:
            raise RuntimeError(
                f"the kernel's slot map was built for metal_block="
                f"{self._slot_map_blk} but the launch uses "
                f"{self.metal_block}; assign `metal_block` rather than "
                f"`_metal_block` so the map is rebuilt")
        phi, q, h = self._metal_matrices(v)
        pt = mx.concatenate(
            [mx.zeros((n, 1, m, m), dtype=dtype), phi], axis=1)
        # Row 0 is the first point's absent predecessor, so its `Q` is the
        # stationary covariance.  Identity for every term but a convolved
        # one, which supplies its own -- see `Term.p_inf_blocks`.
        pin = (self._p_inf(pm.kernel_params(v), dtype, dense=True)
               if self._has_p_inf else None)
        q0 = (mx.broadcast_to(mx.eye(m, dtype=dtype), (n, 1, m, m))
              if pin is None else pin[:, None, :, :])
        qt = mx.concatenate([q0, q], axis=1)

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
        # Same correction as the reference path.  It is per chain and
        # applies to every observation, so it is added to the jitter
        # wherever it is; with no jitter declared it becomes the one
        # column, which is why stage 2's slot had to be additive rather
        # than "the jitter".
        corr = self.term.diag_correction(pm.kernel_params(v))
        if self.noise_fn is not None or len(pm.jitter) > 1:
            # Per datum, (npts, n_chains, 1) in the residual's layout: the
            # kernel reads it per observation and writes its adjoint per
            # observation (`metal._jitter_src`).  Several per-instrument
            # jitters go this way too since 2026-10-25 -- measured within
            # 0-10% of the old compact per-instrument variant, which kept
            # one adjoint accumulator per jitter in registers and capped
            # them at eight.
            if self.noise_fn is not None:
                extra = self._noise_extra(v, self._r_ref32, 0, self.n_data)
            else:
                extra = mx.take(pm.jitter_var_per_instrument(v),
                                self._metal_inst, axis=1)
            if corr is not None:
                extra = extra + corr
            jit2 = mx.swapaxes(extra, 0, 1)[:, :, None]
        else:
            # One jitter or none: one value per chain, read once per thread.
            jit = pm.jitter_var_per_instrument(v)
            if corr is not None:
                jit = corr if jit is None else jit + corr
            jit2 = (mx.zeros((n, 0), dtype=dtype) if jit is None
                    else mx.broadcast_to(jit, (n, 1)))

        # scan_loglike, not scan_blocks + tree_reduce here: the
        # differentiation boundary has to enclose the reduction so the
        # custom function's output is one scalar per chain.  See its
        # docstring.
        out = scan_loglike(pt, qt, self._metal_lag, self._metal_inst, h, r,
                           self._r_ref32,
                           jit2, self.policy.s_floor_rel, self.metal_block,
                           slots=(self._metal_slot_id,
                                  self._metal_slot_start,
                                  self.n_metal_slots,
                                  self._metal_fold_plan),
                           formal=self._formal,
                           boundary=self._metal_boundary)
        return out, (q, h)

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
            fm = self._formal
            if self.blocked:
                root = chunked_reduce_blocked(leaf_parts(*args, formal=fm),
                                              sub_len=self.sub_block,
                                              formal=fm)
            else:
                root = chunked_reduce(*leaf_elements(*args, formal=fm),
                                      sub_len=self.sub_block, formal=fm)
            if do_eval:
                # bound peak memory: the block's leaf arrays die here, and
                # the root that survives is one element per chain.
                mx.eval(root)
            roots.append(root)
        if len(roots) == 1:
            return roots[0].ell
        stacked = Element(*(mx.stack([getattr(r, f) for r in roots], axis=1)
                            for f in Element._fields))
        return tree_reduce(stacked, formal=self._formal).ell

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

    def _infeasible_pull(self, v: mx.array):
        """``log1p(mean v_ref**2)`` per chain, gradient-free, or ``None``.

        The distance of each chain from the reference point, over the columns
        in ``_pull_cols``.  Only ever *used* on lanes whose likelihood failed;
        see `_sanitize` for why it exists and why it carries no gradient.
        """
        if self._pull_cols is None:
            return None
        g = mx.take(v, self._pull_cols, axis=1)
        g = mx.where(mx.isfinite(g), mx.clip(g, -_PULL_CLIP, _PULL_CLIP),
                     mx.full(g.shape, _PULL_CLIP, dtype=g.dtype))
        return mx.stop_gradient(mx.log1p(mx.mean(g * g, axis=1)))

    def _init_kappa(self, inst) -> None:
        """Host constants for :meth:`kappa`, and a warning if the data alone
        say a posterior could sit past ``policy.kappa_max``."""
        lags = self._dt64[1:]
        lags = lags[lags > 0]
        self._kappa_dt = float(np.median(lags)) if lags.size else None
        # The median lag's row in the kernel's lag table, when it has one
        # (any regular or gapped-regular cadence): `kappa` can then read
        # h Q h^T from the forward's own table instead of rebuilding the
        # term's matrices, which cost ~1.1 ms a call eager.
        self._kappa_row = None
        if self._kappa_dt is not None:
            hit = np.flatnonzero(np.unique(self._dt64.astype(np.float32))
                                 == np.float32(self._kappa_dt))
            if hit.size:
                self._kappa_row = int(hit[0])
        n_jit = len(self.param_map.jitter)
        if n_jit > 1:
            lab = np.asarray(inst)
            r = [float(np.median(self._yerr64[lab == i] ** 2))
                 if np.any(lab == i) else np.inf for i in range(n_jit)]
        else:
            r = [float(np.median(self._yerr64 ** 2))]
        self._kappa_r = np.array(r)
        rms2 = float(np.mean((self._y64 - self._y64.mean()) ** 2))
        kmax = self.policy.kappa_max
        if kmax is None or self._kappa_dt is None:
            return
        # kappa <= sigma^2 / r for any timescale, and a fitted sigma is about
        # the data's rms, so this ratio bounds the kappa of a posterior.
        if rms2 / float(np.min(self._kappa_r)) > kmax:
            warnings.warn(
                f"the data's rms is {np.sqrt(rms2 / np.min(self._kappa_r)):.3g}"
                f" times the median error bar, so a GP posterior with a long "
                f"timescale could reach kappa > kappa_max = {kmax:.3g}, where "
                f"float32 is not accurate and chains are refused. Check "
                f"GPLogLike.kappa at your posterior draws; see docs/float32.md.",
                stacklevel=3)

    def kappa(self, v: mx.array, tables=None) -> mx.array:
        """Float32 conditioning number per chain, ``(n_chains,)``.

        ``k(0) / (h Q h^T + r)``: the process variance over the variance a
        single step adds, the GP's own process noise ``h Q h^T`` at the
        median cadence plus the white noise ``r`` (the median ``yerr^2`` plus
        the jitter, the smallest over instruments).  It measures how nearly
        deterministic the state is between samples, which is what float32
        cannot resolve: with ``eps ~ 1.2e-7`` the trouble starts where
        ``kappa * eps`` approaches 1.

        Measured on SHO kernels (docs/float32.md), the float32 error is a
        function of ``kappa`` alone to within the scatter, does not move with
        N from 2e3 to 1e5, and crosses about a log-unit near ``kappa = 1e6``
        when the data are drawn at those parameters.  Evaluated at the
        median cadence, so a dataset with much shorter gaps can be worse
        locally than ``kappa`` says.

        ``tables`` is the kernel path's ``(Q, h)`` lag table for these
        chains; with it, ``h Q h^T`` is read from the median lag's row
        rather than rebuilt (same value to float32 rounding).  Not for a
        convolved term, whose ``k(0)`` needs its own ``P_inf``.
        """
        n = v.shape[0]
        if self._kappa_dt is None:
            return mx.zeros((n,), dtype=v.dtype)
        if self.noise_fn is not None:
            # A per-datum variance: its geometric mean over the data, the
            # analogue of the median error bar the jitter form uses.
            rr = (self._r_ref32 if v.dtype == mx.float32
                  else mx.array(self._yerr64 ** 2, dtype=v.dtype))
            var = self._noise_extra(v, rr, 0, self.n_data) + rr
            w = mx.exp(mx.mean(mx.log(mx.maximum(var, 1e-38)), axis=-1,
                               keepdims=True))
        else:
            r = mx.array(self._kappa_r, dtype=v.dtype)
            jit = self.param_map.jitter_var_per_instrument(v)
            w = mx.broadcast_to(r if jit is None else r + jit,
                                (n, r.shape[0]))
        if (tables is not None and self._kappa_row is not None
                and not self._has_p_inf):
            q, h = (mx.stop_gradient(x) for x in tables)
            qm = q[:, self._kappa_row]
            hqh = mx.sum(h * (qm @ h[..., None])[..., 0], -1)
            return mx.sum(h * h, -1) / (mx.maximum(hqh, 0.0)
                                        + mx.min(w, axis=-1))
        kp = self.param_map.kernel_params(v)
        dt = mx.full((n, 1), self._kappa_dt, dtype=kp.dtype)
        _, qs, h = self.term.matrices_blocks(kp, dt)
        p_inf = self.term.p_inf_blocks(kp)
        k0 = hqh = 0.0
        off = 0
        for b, qb in enumerate(qs):
            mb = qb.shape[-1]
            hb = h[:, off:off + mb]
            off += mb
            hqh = hqh + mx.sum(hb * (qb[:, 0] @ hb[..., None])[..., 0], -1)
            pb = hb if p_inf is None else (p_inf[b] @ hb[..., None])[..., 0]
            k0 = k0 + mx.sum(hb * pb, -1)
        # A convolved term adds its zero-lag correction c to every
        # observation (`Term.diag_correction`; the likelihood passes it in
        # with the jitter).  It is white beyond the exposure, so it belongs
        # in both halves: the zero-lag variance is k0 + c -- the exact
        # convolved k(0), positive where the formal k0 = h P_inf h^T can be
        # negative (P_inf is indefinite; kappa came out as low as -1e8 for
        # an overdamped SHO and slipped past the refusal, which only refuses
        # *above* kappa_max) -- and the one-step part is h Q h^T + c.  c runs
        # from -2e5 r to +1e8 r over ordinary parameters, so omitting it was
        # no rounding matter.
        corr = self.term.diag_correction(kp)
        if corr is not None:
            k0 = k0 + corr[:, 0]
            hqh = hqh + corr[:, 0]
        # h Q h^T can come out slightly negative in float32 (see the clamp
        # in elements.py), and the formal one can be genuinely negative; the
        # one-step variance it stands for is not, so floor it at zero.
        return k0 / (mx.maximum(hqh, 0.0) + mx.min(w, axis=-1))

    def _refuse_ill_conditioned(self, v: mx.array, out: mx.array,
                                tables=None) -> mx.array:
        """NaN on chains past ``policy.kappa_max``, so ``_sanitize`` gives
        them the failure value.  ``mx.where`` selects, so a feasible lane's
        value passes through bit for bit.  Its gradient does too on the MLX
        path; on the kernel path, where ``kappa`` reads the forward's own
        lag table, it moves by up to 4e-6 relative.  Reading the table
        in-graph changes MLX's scheduling of the backward even with the
        gradient stopped (a host copy of the same numbers does not), and the
        two gradients are equally accurate: 5.1e-7 and 5.7e-7 median error
        against a float64 finite difference, the same maximum."""
        kmax = self.policy.kappa_max
        if kmax is None:
            return out
        k = mx.stop_gradient(self.kappa(v, tables))
        # `not <=` so that a NaN kappa (inf / inf) is refused too.
        bad = mx.logical_not(k <= kmax)
        return mx.where(bad, mx.array(float("nan"), dtype=out.dtype), out)

    def _refuse_outside_series(self, v: mx.array, out: mx.array) -> mx.array:
        """NaN on chains whose ``||F delta||`` leaves the convolution's
        validated range.

        The construction-time check only looks at the reference point, and a
        chain can walk anywhere: at ``rho = 2 dt``, ``Q = 0.05`` (inside the
        default prior box) ``||F delta|| = 63``, the 20-term series has not
        converged, and the value came back 432 log-units wrong in float64.
        Same norm as the construction check, per chain.  A no-op for every
        term but a convolved one.
        """
        if not self._formal:
            return out
        kp = mx.stop_gradient(self.param_map.kernel_params(v))
        worst = None
        for g in self.term.term.generator(kp):
            nrm = mx.max(mx.abs(g), axis=(-2, -1)) * self.term.delta
            worst = nrm if worst is None else mx.maximum(worst, nrm)
        bad = mx.logical_not(worst <= terms.MAX_FDELTA)
        return mx.where(bad, mx.array(float("nan"), dtype=out.dtype), out)

    def _raw_call(self, v: mx.array) -> mx.array:
        """The float32 likelihood itself, sanitised but not gradient-guarded."""
        pull = self._infeasible_pull(v)
        tables = None
        if self._metal_ok:
            out, tables = self._evaluate_metal(v)
        else:
            out = self._evaluate(
                v, self._dt32, self._y32, self._r_ref32, self._x_mean32,
                self._inst32, _pow2(self.policy.block),
                self.policy.eval_blocks, uniq=self._uniq32)
        out = self._refuse_outside_series(v, out)
        return _sanitize(self._refuse_ill_conditioned(v, out, tables), pull)

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

    def hi(self, v: mx.array, refuse: bool = True) -> mx.array:
        """Full-float64 CPU evaluation, tiled over CHAINS.

        A GP factorisation is global-sequential over data, so unlike
        ``ChunkedGaussianLogLike.hi`` this cannot tile the data axis -- it
        streams it in recursion order and tiles chains instead.

        By default this is the float64 value of the *same target* as
        :meth:`__call__`: chains that the float32 path refuses (past
        ``kappa_max``, or outside a convolution's series range) and lanes
        whose value is not finite get the same finite failure value.  That
        matters because anvil's ``reanchor_every`` writes this value into
        each chain's stored ``log_prob``: unguarded, it handed a refused
        chain a real likelihood (undoing the inward ratchet ``_sanitize``
        exists for) and a float64 NaN went straight in.

        ``refuse=False`` returns the raw float64 likelihood everywhere, for
        measuring float32 error where the float32 path refuses.
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
                if refuse:
                    acc = _sanitize(self._refuse_ill_conditioned(
                        vb, self._refuse_outside_series(vb, acc)),
                        self._infeasible_pull(vb))
                mx.eval(acc)
                out[c:c + vb.shape[0]] = np.array(acc, dtype=np.float64)
        return a64(out)

    def hi_absolute(self, v: mx.array, refuse: bool = True) -> mx.array:
        """float64 absolute log-likelihood.  The addition is pinned to the CPU
        stream: float64 does not exist on Metal, and the default stream is
        the GPU in any normal session."""
        return mx.add(self.hi(v, refuse=refuse),
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

    That table is a single SHO (m = 2, rho = 100 cadences, Q = 3).  Sums of
    terms also suffered a separate cancellation in the general composition
    until 2026-10-24 (docs/float32.md, section 7), so the default was
    re-checked after that fix, same N and amplitude, worst of two draws:

        sub_block       8       16      32      64      128
        m = 2         -0.63   -0.64   -0.18   -0.08   -0.04   (as above)
        m = 4  before -0.33   +0.57   -0.18   -0.44   -0.21
               after  +0.18   +0.18   +0.05   +0.04   +0.06
        m = 6  before -1.81   +1.62   -0.95   +0.10   -0.25
               after  -0.10   +0.02   +0.04   -0.01   +0.02

    32 still stands; what remains at m = 2 is the composites' stored
    log-normaliser, not the composition's arithmetic.  m >= 7 (the 64) was
    not re-measured.
    """
    if m <= 6:
        return 32
    return 64


#: longest block we will hand one thread, by state dimension.  A thread
#: walks its whole block sequentially, and macOS kills a command buffer
#: whose work it cannot preempt for too long ("Impacting Interactivity").
#: What it responds to is per-thread length, stretched by how many threads
#: share the GPU -- not total GPU time (459 ms of backward kernels in one
#: evaluation survive, a whole second at short blocks survives) and not
#: command-buffer packing.  The limit is stricter while the display is in
#: use, so the caps are set with the display busy (docs/boundary-plan.md,
#: II and III).  Per-observation work grows like m^3, so the safe length
#: falls as m rises.  Measured 2026-10-25 under display load, value +
#: gradient: m = 2 at 8192 and m = 3 at 2048 survive up to 4096 chains at
#: N = 1e5; m = 4 at 1024 is killed from ~224 chains at N = 1e5 and at 4096
#: chains even at N = 1e4, at 512 from 4096 chains, and at 256 survives
#: every case measured (up to 4096 chains at N = 1e5, 3.6 s).  (The earlier
#: cap of 1024 was set on a quiet machine.)  m = 6 (rotation + SHO) at 256
#: and at 128 is killed from 512 chains at N = 1e5, forward included, and
#: at 64 survives up to 4096 chains, with 8 per-instrument jitters too;
#: m = 5 (SHO + Matern-5/2) at 256 is killed from 512 chains and at 128
#: survives 4096 chains.  Each cap is half the shortest block killed.
#: Shorter blocks cost accuracy where the boundary carries it: m = 5 at
#: 20 s, s/e 100 is 0.36 posterior sigma at 256, 0.67 at 64.
_MAX_METAL_BLOCK = {1: 8192, 2: 8192, 3: 2048, 4: 256, 5: 128}
_MAX_METAL_BLOCK_HI_M = 64


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

    Returns ``("dense", table)`` of shape ``(n_lags, max_rank)`` only when
    the slots stack deeper than :data:`_MAX_FOLD_RANKS` **and** the padded
    table stays within 4x the slots it holds; ``("groups", [(slots, lags),
    ...])`` otherwise, each group having distinct lags so each scatter is
    reproducible on its own.  Both conditions matter: a mostly-regular
    cadence has a deep stack *and* a huge table, and takes the groups
    branch.
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
    # Known limitation: `_core_vjp` holds c_phi and c_q -- one full table
    # each -- live across the whole chunk loop and slices them per chunk,
    # so ~`table` bytes are a floor on peak that no chunk count reduces.
    # The budget below covers only the *retained intermediates*, so the
    # achieved peak is roughly the budget plus that floor: small at
    # N = 2e4, where this was validated, and ~6 GiB at 1e5 x 1024 chains.
    # Subtracting the floor here would raise the chunk count, and the
    # docstring's own measurements show chunking harder costs real time
    # (16 chunks: 127 ms against 28 ms at 4), so it is documented rather
    # than "fixed" without a measurement to justify the trade.
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


def _flatten(out):
    """Arrays out of nested lists / tuples, and how to put them back."""
    if isinstance(out, (list, tuple)):
        parts = [_flatten(o) for o in out]
        flat = [a for f, _ in parts for a in f]
        kind = type(out)
        sizes = [len(f) for f, _ in parts]
        builders = [b for _, b in parts]

        def build(arrs):
            res, k = [], 0
            for n, b in zip(sizes, builders):
                res.append(b(arrs[k:k + n]))
                k += n
            return kind(res)
        return flat, build
    return [out], lambda arrs: arrs[0]


def _float64_bridge(fn):
    """``fn(p, *rest)`` evaluated in float64 on the host when ``p`` is
    float32, forward and VJP, then rounded back to float32; outputs may be
    nested lists of arrays.  ``rest`` (the lags) carries no gradient."""
    shape = {}

    @mx.custom_function
    def bridged(p, *rest):
        with mx.stream(mx.cpu):
            out = fn(p.astype(mx.float64),
                     *[x.astype(mx.float64) for x in rest])
            if out is None:             # e.g. no correction: nothing to carry
                shape["build"], shape["none"] = (lambda arrs: None), True
                return (mx.zeros((1,), dtype=mx.float32),)
            flat, build = _flatten(out)
            shape["build"], shape["none"] = build, False
            return tuple(o.astype(mx.float32) for o in flat)

    @bridged.vjp
    def bridged_vjp(primals, cotangents, outputs):
        # a single-argument custom_function gets its primal bare, not in a
        # tuple; normalise both
        prim = list(primals) if isinstance(primals, (list, tuple)) \
            else [primals]
        cots = list(cotangents) if isinstance(cotangents, (list, tuple)) \
            else [cotangents]
        p, rest = prim[0], prim[1:]
        with mx.stream(mx.cpu):
            r64 = [x.astype(mx.float64) for x in rest]

            if shape["none"]:
                dp = mx.zeros(p.shape, dtype=mx.float64)
            else:
                def f(pp):
                    return _flatten(fn(pp, *r64))[0]
                _, (dp,) = mx.vjp(f, [p.astype(mx.float64)],
                                  [c.astype(mx.float64) for c in cots])
            return (dp.astype(mx.float32),
                    *[mx.zeros_like(x) for x in rest])

    def call(p, *rest):
        flat = bridged(p, *rest)
        flat = list(flat) if isinstance(flat, (list, tuple)) else [flat]
        return shape["build"](flat)
    return call


class _Float64FrontEnd:
    """A convolved term whose lag-table functions run in float64.

    A :class:`~anvilgp.terms.TermConvolution`'s formal state-space form
    forms ``Q = P_inf - Phi P_inf Phi^T`` by subtraction (float32.md, rule
    1's documented exception), and in float32 that costs ~10x the
    unconvolved term's error on bright data: value sd 0.185 against 0.021
    across a posterior with the same tables computed in float64, and a
    gradient 1.3 posterior sigma off at one block against 0.07
    (docs/boundary-plan.md, VI).  So ``matrices`` and ``matrices_blocks``
    evaluate in float64 on the host when called with float32 parameters --
    one small table per chain and distinct lag -- and everything else is the
    wrapped term's.  Only those two: bridging ``p_inf_blocks`` and
    ``diag_correction`` as well changed nothing measurable (one block,
    0.08 -> 0.07 sigma) and costs as much again.  Float64 calls (the
    ``.hi`` path) go straight through.
    """

    _BRIDGED = ("matrices", "matrices_blocks")

    def __init__(self, term):
        self._term = term
        # looked up per call, not bound here, so a method replaced on the
        # term or its class afterwards is the one that runs
        self._bridges = {
            name: _float64_bridge(
                lambda p, *rest, _n=name: getattr(self._term, _n)(p, *rest))
            for name in self._BRIDGED}

    def __getattr__(self, name):
        attr = getattr(self._term, name)
        if name not in self._BRIDGED:
            return attr

        def call(p, *rest):
            if p.dtype != mx.float32:
                return attr(p, *rest)
            return self._bridges[name](p, *rest)
        return call


#: largest reference-point spectral ``||F delta||`` at which a convolved
#: term takes the square-root boundary.  Clamping Q's negative part skews
#: the entry states the value is replayed from as that part grows: value sd
#: across a posterior, s/e = 100, against one block's -- 0.002 vs 0.000 at
#: 0.25, 0.006 vs 0.000 at 0.41, 0.027 vs 0.000 at 0.99, 0.074 vs 0.001 at
#: 1.98 -- while the dense boundary measured 0.00 sigma / 0.001 at 0.41,
#: 0.99 and 1.98 (docs/boundary-plan.md, VI).  The dense boundary's own
#: failure is at small ||F delta|| and high s/e (10 sigma at 0.007).
_SQRT_FDELTA = 0.5


#: convolved member types the square-root boundary is measured on.  Inside
#: the ``_SQRT_FDELTA`` guard their leaf's white-noise part ``r + c`` stays
#: positive -- ``c >= 0`` for SHO, rotation and Matern-3/2, and Matern-5/2's
#: negative ``c`` is >= -4.6e-8 k(0), so it would take s/e ~ 5000 -- and the
#: boundary beat the dense one (SHO + Matern-3/2, s/e 100: 0.05 against 2.06
#: posterior sigma; Matern-5/2 at rho = 1 d: 0.04 against 1.38).
#: `RealTerm`'s ``c`` reaches -0.17 k(0), ``r + c`` goes negative and the
#: leaf has no real square root, so it keeps the dense boundary, which
#: measured fine on it.
_SQRT_CONVOLVED = (SHOTerm, RotationTerm, Matern32Term, Matern52Term)


def _convolution_takes_sqrt(inner: Term, ref_fdelta: float | None) -> bool:
    """Whether a convolved ``inner`` can use the kernel's square-root block
    boundary.  The boundary needs a factor of ``Q``; a convolved term's is
    formally indefinite, and `metal.q_factor` clamps the negative part.
    With the lag table in float64 (`_Float64FrontEnd`) that is measured
    within 1.2-2x of exact boundaries for SHO and rotation terms -- 0.47
    against 0.33 posterior sigma on a TESS 2-min convolved rotation at
    s/e = 100, where the dense boundary is 10 -- and their exposure
    correction is non-negative everywhere in the validated range.  A
    real-type member's correction is negative and can push the leaf
    variance below zero (`RealTerm`: NaN), and Matern-5/2 measured worse
    than dense, so anything else keeps the dense boundary
    (docs/boundary-plan.md, VI; Matern-3/2 and -5/2 added after
    measuring them inside the guard).  And only while ``Q``'s negative part is
    small, which it is below ``_SQRT_FDELTA`` at the reference point.  A
    chain that walks well past it keeps the square-root boundary (the
    choice is fixed at construction) and its value degrades as above."""
    members = inner.terms if isinstance(inner, TermSum) else (inner,)
    return (all(isinstance(t, _SQRT_CONVOLVED) for t in members)
            and ref_fdelta is not None and ref_fdelta <= _SQRT_FDELTA)


class MetalDefaultNotice(UserWarning):
    """Which likelihood path ``GPPolicy(metal=None)`` chose.  Issued once
    per process and per outcome; filter it to silence it."""


_METAL_NOTICES: set = set()


def _notice_metal_choice(ok: bool, why: str) -> None:
    """Say once per process what the default chose.  The default moved
    from the MLX path to the kernel on 2026-10-25, so float32 numbers and
    timings changed under scripts that never asked for either; this is
    the signal that they did."""
    key = "kernel" if ok else f"mlx: {why}"
    if key in _METAL_NOTICES:
        return
    _METAL_NOTICES.add(key)
    if ok:
        msg = ("anvilgp: using the fused Metal kernel, the default since "
               "2026-10-25 (GPPolicy(metal=None)); GPPolicy(metal=False) "
               "selects the MLX path.")
    else:
        msg = (f"anvilgp: using the MLX path because the fused kernel "
               f"cannot serve this problem ({why}); on bright, densely "
               f"sampled data its float32 value varies more across the "
               f"posterior (docs/float32.md, section 7).")
    warnings.warn(msg, MetalDefaultNotice, stacklevel=4)


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

    and at m = 4 (SHO + SHO, value + gradient, ms, 2026-10-25, display in
    use; the cap binds at every N):

        N = 1e4,  256 chains:  256:40.5   512:48.2   1024:74.6
        N = 1e4, 1024 chains:  256:113    512:115    1024:101
        N = 1e4, 4096 chains:  256:369    512:374    1024: KILLED
        N = 1e5, 1024 chains:  256:978    512:929    1024: KILLED
        N = 1e5, 4096 chains:  256:3557   512: KILLED

    and at m = 6 (rotation + SHO, 512 chains, N = 1e5, display in use):
    256 and 128 KILLED, value + gradient 5373 at 64 and 7285 at 32; m = 5
    at 256 KILLED.  The rule's floor is the cap itself, not 256, above
    m = 4.

    The uncapped rule picked 8192 at N = 1e5 for every m, which is a hard
    failure at m = 4 rather than a slow path; the cap was 1024 until
    2026-10-25, set on a quiet machine.
    """
    target = max(1.0, n_data / 12.0)
    p = 1
    while p * 2 <= target:            # largest power of two <= target
        p *= 2
    if target - p > 2 * p - target:   # round to the *nearest*, not up
        p *= 2
    cap = _MAX_METAL_BLOCK.get(n_state, _MAX_METAL_BLOCK_HI_M)
    return max(min(256, cap), min(cap, p))


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _sanitize(x: mx.array, pull: mx.array | None = None) -> mx.array:
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

    **The failure value slopes back towards the reference point** when
    ``pull`` is given: ``-1e30 * (1 + pull)``, with ``pull`` a chain's
    distance from ``v = 0`` over the GP's own columns.  A constant ``-1e30``
    made the infeasible region perfectly flat, and a flat region is not a
    rejection: a sampler accepts moves *within* it and drifts.  That was
    masked until anvil made its bounded log-Jacobian finite (it had been
    exactly ``-inf`` past ``|u| ~ 18``, an accidental wall); after that,
    stretch walkers started in the infeasible region diffused to
    ``|u| ~ 1e11`` and never came back -- 11 of 32 in
    `test_chains_started_in_the_infeasible_region_recover`, proven causal by
    restoring the old Jacobian in-process.

    Three constraints fix the form, and each rules out the obvious
    alternative:

    * **It must never beat a real likelihood**, so the magnitude stays at
      ``1e30`` and every ``<= -1e29`` test still means "failed".
    * **The slope must survive float32 at that magnitude**, which an additive
      one cannot -- ``-1e30 + (-120) == -1e30`` exactly, which is the bug.
      Hence relative: a change of ``1e-7`` in ``pull`` is resolvable.
    * **It must carry no gradient.**  The derivative of ``-1e30 * pull`` is
      of order ``1e30`` and would wreck an HMC trajectory, so ``pull`` is
      ``stop_gradient``-ed and a failed lane keeps the exactly-zero gradient
      it always had.  The slope acts through *values* alone.

    That last choice buys the property the change is verified against: a
    chain that never enters the infeasible region sees **bit-identical**
    values and gradients, so no posterior can move.  Only recovery changes --
    both samplers now ratchet inwards, accepting moves towards the reference
    (a gain of order ``1e30``) and rejecting moves away.
    """
    bad = mx.logical_not(mx.isfinite(x))
    safe = mx.where(bad, mx.zeros_like(x), x)
    if pull is None:
        sentinel = mx.full(x.shape, _LARGE_NEGATIVE, dtype=x.dtype)
    else:
        sentinel = _LARGE_NEGATIVE * (1.0 + pull.astype(x.dtype))
    return mx.where(bad, sentinel, safe)
