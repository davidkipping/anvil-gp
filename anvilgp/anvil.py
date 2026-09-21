"""anvil engine integration (lazy import; the core package stays standalone).

``make_target`` assembles the whole sampling problem -- transform, priors,
GP likelihood, float64 verification path -- from data and a kernel, so the
common case is one call rather than a page of bookkeeping.  anvil is imported
inside the functions that need it, mirroring ``metalplanet/anvil.py``, so
``import anvilgp`` still works with only mlx and numpy installed.

The mean model, if any, follows anvil's usual model contract:
``mean_fn(v_mean, x) -> (n_chains, m)`` with ``v_mean`` the leading block of
the parameter vector and ``x`` the mean model's own abscissa.  For MetalPlanet
that is ``make_quad_transit_flux(period_ref)`` and the ``(2, m)`` stack of
epoch-centred times and orbit numbers -- a *different* array from the GP's
``t``, which carries the physical times that set the lag structure.

Gradient samplers need one extra line today::

    import mlx.core as mx
    mx.disable_compile()

anvil's ChEES kernel compiles its leapfrog step, and ``mx.compile`` over the
*backward* pass of the whole likelihood exceeds MLX's fused-group limit
(the front end, not the element algebra; see docs/float32.md).  Forward-only compilation is fine, so the
gradient-free stretch move needs nothing.  M4's backward kernel will
remove the constraint; M3's forward kernel does not, being forward-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .likelihood import GPLogLike, GPPolicy
from .priors import (SpecHint, auto_order_groups, build_param_map,
                     default_hypers, jitter_hyper, make_log_prior,
                     offset_hyper, periodogram_period)
from .terms import Term

__all__ = ["import_engine", "make_target", "GPTarget"]


def import_engine():
    """Import the sampling engine under either of its names."""
    try:
        import anvil as engine
    except ImportError:  # pragma: no cover - legacy name
        import applemcmc as engine
    return engine


@dataclass
class GPTarget:
    """Everything a sampler needs, plus the pieces needed to report results."""

    target: object            # anvil.TransformedLogDensity
    transform: object         # anvil.Transform
    loglike: GPLogLike
    names: list[str]
    hypers: list
    param_map: object

    def u0(self, n_chains: int, seed: int = 0, spread: float = 0.05):
        """Initial states: near the reference point, which the parametrisation
        has arranged to be a sensible guess (``v = 0`` means every scale
        parameter sits at its data-derived host reference)."""
        rng = np.random.default_rng(seed)
        return mx.array((spread * rng.standard_normal(
            (n_chains, self.transform.dim))).astype(np.float32))

    def natural(self, v_model: np.ndarray) -> dict:
        """Model-space draws -> natural units, for reporting.

        Covers the kernel block, the jitters and the offsets; the mean
        model's own parameters stay the caller's business, since only it
        knows what they mean.
        """
        pm = self.param_map
        v = mx.array(np.asarray(v_model, dtype=np.float32))
        cols = [np.array(pm.kernel_params(v), dtype=np.float64)]
        for group in (pm.jitter, pm.offset):
            if group:
                cols.append(np.array(
                    mx.concatenate([k.apply(v) for k in group], axis=-1),
                    dtype=np.float64))
        allc = np.concatenate(cols, axis=-1)
        out = {}
        for i, h in enumerate(self.hypers):
            key = h.name
            for p in ("log_", "logit_"):
                if key.startswith(p):
                    key = key[len(p):]
            out[key] = allc[..., i]
        return out


def make_target(term: Term, t, y, yerr, mean_fn=None, x_mean=None,
                mean_specs: list[SpecHint] | None = None,
                n_jitter: int = 1, n_offset: int = 0,
                instrument: np.ndarray | None = None,
                order_groups: list[list[int]] | None = None,
                policy: GPPolicy | None = None,
                seed_period: bool = False) -> GPTarget:
    """Build a ready-to-sample GP target with shipped defaults.

    Parameters
    ----------
    term
        The kernel, e.g. ``SHOTerm()`` or ``SHOTerm() + Matern32Term()``.
    t, y, yerr
        The data.  ``t`` must be sorted ascending (duplicates are legal);
        ``yerr`` strictly positive; none of the three may contain a
        non-finite value.
    mean_fn
        ``(params, x_mean) -> (n_chains, n_data)``, the mean model
        evaluated batched over chains.  ``params`` is the slice of the
        model vector named by ``mean_specs``.  Requires ``x_mean`` and
        ``mean_specs``; supplying only some of the three is an error, not a
        silent GP-only fit.
    x_mean
        Covariates the mean model is evaluated on.  Often just ``t``.
    mean_specs
        One :class:`~anvilgp.priors.SpecHint` per mean-model parameter,
        giving its name, bounds and scale.  These occupy the leading
        columns of the model vector; the GP's own parameters follow.
    n_jitter, n_offset
        How many per-instrument jitters and mean offsets to sample.  With
        more than one of either, pass ``instrument``.
    instrument
        ``(n_data,)`` integers assigning each datum to an instrument.
    order_groups
        Defaults to :func:`~anvilgp.priors.auto_order_groups`, which
        constrains the timescales of each same-type family in a sum to be
        increasing.  Without it a k-term sum has k! permutation modes and
        R-hat means nothing; pass ``[]`` to opt out deliberately.  Note the
        ordered members sample a dimensionless *ratio*, so their bounds are
        ratio bounds -- see :func:`~anvilgp.priors.auto_order_groups`.
    policy
        :class:`~anvilgp.likelihood.GPPolicy`.  ``GPPolicy(metal=True)``
        turns on the fused kernel, which is forward-only until M4: the
        returned target then declares ``supports_grad=False`` and a
        gradient sampler refuses it up front.
    seed_period
        Run a host-side Lomb-Scargle first and centre a rotation kernel's
        period reference on the peak, so chains start inside one harmonic
        instead of spread across P, P/2 and 2P.

    Returns
    -------
    GPTarget
        With ``.target`` (hand to ``anvil.run``), ``.transform``,
        ``.loglike``, ``.names``, ``.hypers`` and ``.param_map``.
    """
    engine = import_engine()

    period_ref = periodogram_period(t, y) if seed_period else None
    hypers = default_hypers(term, t, y, yerr, period_ref=period_ref)
    jitters = [jitter_hyper(yerr, f"log_jitter{i}" if n_jitter > 1
                            else "log_jitter") for i in range(n_jitter)]
    offsets = [offset_hyper(y, f"offset{i}" if n_offset > 1 else "offset")
               for i in range(n_offset)]
    mean_specs = list(mean_specs or [])

    if order_groups is None:
        order_groups = auto_order_groups(term)
    pm, specs, names = build_param_map(
        hypers, n_mean=len(mean_specs), jitters=jitters, offsets=offsets,
        instrument=instrument, order_groups=order_groups or None)

    loglike = GPLogLike(mean_fn, t, x_mean, y, yerr, term, pm,
                        policy or GPPolicy())
    log_prior = make_log_prior(hypers, jitters, offsets,
                               n_mean=len(mean_specs))

    transform = engine.Transform([s.to_anvil() for s in mean_specs + specs])

    def model_log_prob(v):
        return loglike(v) + log_prior(v)

    def model_log_prob_hi(v):
        # The prior is cheap and dtype-polymorphic; the expensive part is the
        # likelihood's chain-tiled float64 pass.  Evaluate it AT float64:
        # `log_prior(v).astype(float64)` would compute in the caller's
        # float32 and then widen the rounding error, putting a float32 floor
        # under the reference that `validate_precision` measures against.
        # make_log_prior caches per dtype for exactly this.
        with mx.stream(mx.cpu):
            return loglike.hi(v) + log_prior(v.astype(mx.float64))

    # Both paths are differentiable since M4: the fused kernel has its own
    # VJP (the checkpointed backward sweep), so a gradient sampler can take
    # either.
    target = engine.TransformedLogDensity(
        model_log_prob, transform, model_log_prob_hi=model_log_prob_hi)
    return GPTarget(target=target, transform=transform, loglike=loglike,
                    names=[s.name for s in mean_specs] + names,
                    hypers=hypers + jitters + offsets, param_map=pm)
