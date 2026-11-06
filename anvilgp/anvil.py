"""anvil engine integration (lazy import; the core package stays standalone).

``make_target`` assembles the whole sampling problem -- transform, priors,
GP likelihood, float64 verification path -- from data and a kernel, so the
common case is one call rather than a page of bookkeeping.  anvil is imported
inside the functions that need it, mirroring ``elome/anvil.py``, so
``import anvilgp`` still works with only mlx and numpy installed.

The mean model, if any, follows anvil's usual model contract:
``mean_fn(v_mean, x) -> (n_chains, m)`` with ``v_mean`` the leading block of
the parameter vector and ``x`` the mean model's own abscissa.  For Elome
that is ``make_quad_transit_flux(period_ref)`` and the ``(2, m)`` stack of
epoch-centred times and orbit numbers -- a *different* array from the GP's
``t``, which carries the physical times that set the lag structure.

Gradient samplers need nothing extra since 2026-09-27.  ``mx.compile``
over the *backward* pass used to exceed MLX's fused-group limit, so
anvil's ChEES kernel -- which compiles its leapfrog step -- required a
process-global ``mx.disable_compile()``.  The cause was one function,
``special.sincm1``, whose branch selection nests a three-way ``mx.where``
with each arm separately sanitised; ``terms._sho_matrices`` calls it only
where that branch is unreachable and now uses ``sincm1_series`` there.
See docs/float32.md.
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
    """Everything a sampler needs, plus the pieces needed to report results.

    **As a member of a joint target** (several datasets with some
    parameters shared, composed by anvil or by the caller), the pieces a
    composition needs are public here, all in the layout of ``names``:

    * ``model_log_prob(v)`` / ``model_log_prob_hi(v)`` -- the model-space
      log-density (likelihood plus the explicit Gaussian priors), float32
      and float64, **without** the transform's Jacobian or its bounded
      priors.  A composition must sum these and apply one joint transform:
      summing members' ``target.log_prob(u)`` would count a shared
      parameter's Jacobian and prior once per member.
    * ``model_log_norm`` -- the float64 constant kept out of the graph, so
      ``model_log_prob(v) + model_log_norm`` is the absolute value; a
      composition sums them.
    * ``specs`` -- one `anvilgp.priors.SpecHint` per column, the bounds a
      joint transform is built from; ``u_ref()`` the reference start.
    * failures come back as anvil's sloped failure value (<= -1e29), which
      a sum keeps: one failed member fails the joint chain.
    """

    target: object            # anvil.TransformedLogDensity
    transform: object         # anvil.Transform
    loglike: GPLogLike
    names: list[str]
    hypers: list
    param_map: object
    #: the `anvilgp.noise.NoiseModel` whose columns end ``v``, or None
    noise: object = None
    #: the member protocol: see the class docstring
    model_log_prob: object = None
    model_log_prob_hi: object = None
    model_log_norm: float = 0.0
    specs: list = None

    def u0(self, n_chains: int, seed: int = 0, spread: float = 0.05,
           units: str = "u"):
        """Initial states: a Gaussian ball around :meth:`u_ref`, the point
        where every hyperparameter sits at its data-derived host reference
        (and, with ``seed_period``, a rotation period at the periodogram
        peak).

        That point is *not* ``u = 0`` for a bounded parameter whose bounds
        are asymmetric about its reference: anvil maps ``u = 0`` to the
        middle of the box.  Before 2026-10-23 the ball was centred on
        ``u = 0``, so a rotation period started at ``sqrt(2 cadence T)``
        whatever the periodogram said, and every SHO/rotation quality factor
        at 5 rather than its reference 3.

        ``units`` says what ``spread`` is measured in:

        * ``"u"`` (default): the sampler's unconstrained coordinates.  One
          unit means very different things for different parameters: for a
          bounded parameter it is ``(hi - lo) / 4`` in model space at the
          middle of the box, and less off it, which for the default jitter (bounds 1e-3 to 1e3 times the
          error bar, prior sd 1) is **3.4 prior standard deviations**, about
          a factor of 30 in jitter.
        * ``"prior"``: prior standard deviations, per parameter, so a spread
          of 1 is one prior sigma for every column.  The width comes from the
          parameter's Gaussian prior, or for a bounded parameter without one
          from its uniform prior, ``(hi - lo) / sqrt(12)``, and is converted
          to ``u`` with the transform's slope at the centre.  Columns with
          neither (the mean model's, an unbounded flat parameter) keep
          ``spread`` in ``u``.  This is the one to use for an overdispersed
          start, e.g. for R-hat.

        The same ``seed`` draws the same standard normals in both units.

        **For ChEES-HMC, a wide start needs a long warmup, and a start far
        outside the prior can trap the run.**  ChEES adapts one step size for
        all chains from the harmonic mean of their acceptance, so the
        slowest chain sets it for everyone.  Measured on the tutorial problem
        at 1024 chains, 300 warmup, with the old ``u = 0`` centre (Q starting
        at 5, not 3): the final step size is 0.29 at the
        default ``spread=0.05``; 0.03 at ``spread=0.3``; 3e-3 at
        ``spread=1, units="prior"``, with R-hat 2 to 5.  That run is slow,
        not stuck: with 3000 warmup it reaches step 0.09 to 0.26 and R-hat
        1.003.  ``spread=1`` in ``u`` can be stuck: in one of two seeds a
        single chain wandered to an amplitude 1e13 times the error bar,
        where float32 is off by 2 log-units and only steps of ~1e-7 are
        accepted, and the step size stayed at 3e-7 through 3000 warmup with
        78 other chains frozen behind it.  ``GPPolicy.kappa_max`` now refuses
        such chains (docs/float32.md, section 6): the same seed ends warmup
        at 4.7e-4, and at 1.6e-3 after 3000 steps with R-hat 1.5, a wide
        start still travelling in.  None of this can bias a posterior
        -- adaptation only sets the step size -- and ``warmup_report`` names
        the stranded chains.  See
        ``docs/anvil-chees-stuck-chains-measurement.md``.  The stretch move
        is not affected this way.
        """
        if units not in ("u", "prior"):
            raise ValueError(f"units must be 'u' or 'prior', got {units!r}")
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_chains, self.transform.dim))
        scale = spread if units == "u" else spread * self._prior_sd_in_u()
        return mx.array((self.u_ref() + scale * z).astype(np.float32))

    #: how far inside its box a reference on or past a bound is started:
    #: ``|u| <= U_REF_MAX``, i.e. within 1.8% of the box width of the edge.
    U_REF_MAX = 4.0

    def u_ref(self) -> np.ndarray:
        """The unconstrained point of the hyperparameters' host references.

        Every hyperparameter is sampled as an offset from its reference, so
        the reference is model-space ``v = 0``; this maps it through the
        transform's inverse.  The mean model's columns stay at ``u = 0``:
        their reference is the caller's business.  A reference on a bound
        (a periodogram peak at the edge of its grid) would map to infinity,
        so the result is clipped to ``|u| <= U_REF_MAX``.
        """
        n_mean, n_noise = self._n_mean(), self._n_noise()
        u = self.transform.from_model_np(np.zeros(self.transform.dim))
        u[:n_mean] = 0.0
        # The noise model's columns are raw values like the mean model's:
        # its box's centre (u = 0) is the start, e.g. s = 1 for a scale
        # log-uniform on [0.1, 10].
        u[self.transform.dim - n_noise:] = 0.0
        return np.clip(u, -self.U_REF_MAX, self.U_REF_MAX)

    def linear_posterior(self, v_model: np.ndarray):
        """The marginalised linear parameters' conditional posterior at
        model-space draws: ``(mean, cov)``, see
        `GPLogLike.linear_posterior`."""
        # dtype explicit: `mx.array` of a float64 NumPy array is float32
        return self.loglike.linear_posterior(
            mx.array(np.asarray(v_model, dtype=np.float64), dtype=mx.float64))

    def apply_inverse(self, v_model: np.ndarray, Y):
        """``K^-1 Y`` at model-space draws: see `GPLogLike.apply_inverse`."""
        return self.loglike.apply_inverse(v_model, Y)

    def predict(self, v_model: np.ndarray, t_pred, return_var: bool = True):
        """The GP component's posterior mean and variance at ``t_pred`` for
        model-space draws (``var`` None with ``return_var=False``): see
        `GPLogLike.predict`."""
        return self.loglike.predict(v_model, t_pred, return_var=return_var)

    def _n_noise(self) -> int:
        return 0 if self.noise is None else len(self.noise.specs)

    def _n_mean(self) -> int:
        return self.transform.dim - len(self.hypers) - self._n_noise()

    def _prior_sd_in_u(self) -> np.ndarray:
        """Each column's prior standard deviation, mapped to ``u`` at
        :meth:`u_ref`; 1.0 where there is no prior width to use."""
        specs = self.transform.specs
        n_mean = self._n_mean()
        jac = self.transform.jacobian_np(self.u_ref())
        out = np.ones(len(specs))
        for j, h in enumerate(self.hypers):
            s = specs[n_mean + j]
            bounded = np.isfinite(s.lo) and np.isfinite(s.hi)
            if h.prior is not None:
                sd_v = float(h.prior[2])
            elif bounded:
                sd_v = (s.hi - s.lo) / np.sqrt(12.0)
            else:
                continue
            # dv/du at the centre: (hi - lo) p (1 - p) for the sigmoid map,
            # scale for the affine map.
            out[n_mean + j] = sd_v / jac[n_mean + j]
        return out

    def natural(self, v_model: np.ndarray) -> dict:
        """Model-space draws -> natural units, for reporting.

        Covers the kernel block, the jitters, the offsets and a noise
        model's parameters (through its ``natural``); the mean model's own
        parameters stay the caller's business, since only it knows what
        they mean.
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
        if self.noise is not None:
            vn = np.asarray(v_model, dtype=np.float64)[..., pm.noise]
            if self.noise.natural is not None:
                out.update(self.noise.natural(vn))
            else:
                out.update({s.name: vn[..., i]
                            for i, s in enumerate(self.noise.specs)})
        return out


def make_target(term: Term, t, y, yerr, mean_fn=None, x_mean=None,
                mean_specs: list[SpecHint] | None = None,
                n_jitter: int | None = None, n_offset: int = 0,
                instrument: np.ndarray | None = None,
                order_groups: list[list[int]] | None = None,
                policy: GPPolicy | None = None,
                seed_period: bool = False, noise=None, linear_basis=None,
                linear_prior=None, held_out=None) -> GPTarget:
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
        more than one of either, pass ``instrument``.  ``n_jitter``
        defaults to 1, or 0 with a ``noise`` model.
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
        :class:`~anvilgp.likelihood.GPPolicy`.  The default uses the fused
        kernel wherever it is eligible (since 2026-10-25) and the MLX path
        otherwise; ``GPPolicy(metal=False)`` forces the MLX path.  The
        returned target declares ``supports_grad=True`` on both paths.
    noise
        An `anvilgp.noise.NoiseModel` (e.g. ``noise.error_scale()``), the
        whole per-datum noise variance in place of ``yerr^2 + jitter^2``.
        Its parameters end the model vector, raw values within its specs'
        bounds; ``u0`` starts them at the centre of each box.
    linear_basis, linear_prior
        Linear mean parameters to integrate out rather than sample:
        ``linear_basis(v_mean, x, m0) -> (n_chains, n_data, p)``, the
        columns the mean is linear in (``m0`` is ``mean_fn``'s output, zeros
        without one; ``x`` is ``x_mean``, else ``t``), or ``(n_data, p)``
        for a basis shared by every chain -- a constant, and on the kernel
        2.4x less memory than the per-chain form.  ``linear_prior``
        None for a flat prior on them (the restricted likelihood), or
        ``(mu, cov)``.  :meth:`GPTarget.linear_posterior` reports them.
        On the fused kernel all ``p + 1`` columns ride one filter: value +
        gradient 1.5x the plain likelihood at p = 1, 1.7x at 2, 5.5x at 10
        for a shared basis, 1.7x, 1.9x and 6.3x per chain (N = 1e4 x 1024
        chains).  The MLX path, and the kernel above 31
        basis columns, polarise instead, at ``1 + (p+1)(p+2)/2`` likelihood
        evaluations (7 at p = 2).  A shared ``(n_data, p)`` basis is
        evaluated once in float64 and orthonormalised, so raw times and
        collinear columns are handled for it.  A per-chain basis is
        rebuilt in float32: centre and scale its columns (``x -
        x.mean()``, not raw ``x``); exactly collinear per-chain columns (a
        global level beside one offset per instrument) are refused in
        float32 -- drop the redundant one.
    held_out
        Points held out of the GP's conditioning: a boolean ``(n_data,)``
        mask or a list of ``(t_lo, t_hi)`` windows.  The GP is fitted to
        the rest and applied to these through its conditional mean; they
        are scored with their white noise alone, so a short feature whose
        shape is the measurement cannot be absorbed by the GP.  Does not
        model correlated noise inside the windows; see
        `anvilgp.likelihood.GPLogLike` for the construction and its
        caveat.  The parameter vector is the same as without it.  Runs on
        the kernel (a convolved term on the MLX path); see
        `anvilgp.likelihood.GPLogLike`.
    seed_period
        Run a host-side Lomb-Scargle first and centre a rotation kernel's
        period reference on the peak, so chains start inside one harmonic
        instead of spread across P, P/2 and 2P.

    Returns
    -------
    GPTarget
        With ``.target`` (hand to ``anvil.run``), ``.transform``,
        ``.loglike``, ``.names``, ``.hypers``, ``.param_map``, ``.noise``
        (the noise model or None), and the joint-member protocol
        (``.model_log_prob``, ``.model_log_prob_hi``, ``.model_log_norm``,
        ``.specs``, ``.u_ref()``; see `GPTarget`).  Methods: ``u0``,
        ``natural``, ``predict``, ``apply_inverse``, ``linear_posterior``.
    """
    engine = import_engine()
    if n_jitter is None:
        n_jitter = 0 if noise is not None else 1
    if noise is not None and n_jitter:
        raise ValueError(
            "`noise` gives the whole noise variance, so it cannot be "
            "combined with `n_jitter` > 0; use a noise model that includes "
            "a jitter (`anvilgp.noise.scale_and_jitter`).")
    noise_specs = [] if noise is None else list(noise.specs)

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
        instrument=instrument, order_groups=order_groups or None,
        n_noise=len(noise_specs))

    loglike = GPLogLike(mean_fn, t, x_mean, y, yerr, term, pm,
                        policy or GPPolicy(),
                        noise_fn=None if noise is None else noise.fn,
                        linear_basis=linear_basis, linear_prior=linear_prior,
                        held_out=held_out)
    log_prior = make_log_prior(hypers, jitters, offsets,
                               n_mean=len(mean_specs))

    transform = engine.Transform([s.to_anvil()
                                  for s in mean_specs + specs + noise_specs])

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
    # The likelihood keeps -1/2 sum log(2 pi yerr^2) out of its float32
    # graph; anvil adds `model_log_norm` back wherever an absolute value
    # matters (evidence, `log_norm_const`).  Until 2026-10-25 it was not
    # passed, so `anvil.evidence` on a GP target was off by that constant.
    target = engine.TransformedLogDensity(
        model_log_prob, transform, model_log_prob_hi=model_log_prob_hi,
        model_log_norm=loglike.log_norm_const)
    return GPTarget(target=target, transform=transform, loglike=loglike,
                    names=([s.name for s in mean_specs] + names
                           + [s.name for s in noise_specs]),
                    hypers=hypers + jitters + offsets, param_map=pm,
                    noise=noise, model_log_prob=model_log_prob,
                    model_log_prob_hi=model_log_prob_hi,
                    model_log_norm=loglike.log_norm_const,
                    specs=mean_specs + specs + noise_specs)


def joint_u0(jt, members: dict, n_chains: int, seed: int = 0,
             spread: float = 0.05) -> mx.array:
    """Initial states for an ``anvil.joint`` target built from GPTargets.

    Each member's :meth:`GPTarget.u_ref` goes to the joint columns it reads
    (``jt.member_columns(name)``); a shared column takes the mean of its
    members' values (they agree for a shared mean-model parameter, which
    starts at ``u = 0``), then a Gaussian ball of ``spread`` in ``u``.
    ``members`` maps the names given to ``anvil.joint`` to the GPTargets
    whose ``.target`` went in.

    Sharing a GP *hyperparameter* across members needs equal specs, which
    ``make_target``'s data-derived bounds will not give: build those
    members' specs to agree, or share only mean-model parameters.
    """
    dim = jt.transform.dim
    acc, cnt = np.zeros(dim), np.zeros(dim)
    for name, gt in members.items():
        cols = jt.member_columns(name)
        acc[cols] += gt.u_ref()
        cnt[cols] += 1.0
    if np.any(cnt == 0):
        raise ValueError("members do not cover every joint column; pass "
                         "every member given to anvil.joint")
    u = acc / cnt
    rng = np.random.default_rng(seed)
    return mx.array((u + spread * rng.standard_normal((n_chains, dim)))
                    .astype(np.float32))
