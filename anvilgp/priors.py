"""Default parametrisation, priors and initialisation -- shipped, not homework.

Three things a GP-plus-mean-model fit gets wrong by default, all of which
cost throughput rather than announcing themselves as errors:

**The parametrisation.**  Sample ``(log sigma, log rho, log Q)`` per SHO
term, never ``(S0, omega_0, Q)``.  The stationary variance and the period are
what the data constrain; ``S0 = sigma^2/(omega_0 Q)`` ties three of them
together, so a high-Q posterior that is an axis-aligned box in the first
parametrisation is a curved ridge in the second, and a diagonal
preconditioner cannot help with a curved ridge.  Granulation fixes
``Q = 1/sqrt(2)`` and samples ``(sigma, rho)``.

**Offsets from float64 host references.**  Every log-parameter is sampled as
``log(x / x_ref)`` with ``x_ref`` a host-side float64 constant taken from the
data (the RMS of y, the median error, the median cadence).  This is anvil's
rule 2 applied to hyperparameters: it puts the posterior near ``v = 0``, which
is where float32 has the most room, and it centres the *shared* diagonal
preconditioner that all chains have to agree on.

**Proper priors are a throughput feature.**  With thousands of chains, an
improper flat prior on a log-amplitude guarantees that some chains sit on the
``sigma -> 0`` ridge, where the likelihood is flat and the step size that
suits them is enormous.  ChEES adapts one step size from the *harmonic mean*
acceptance across chains, so those few chains drag the step size down for
everyone.  The defaults below are weak enough to be uncontroversial and
proper enough to keep the ensemble together.

Identifiability ridges to expect (each is real, none is a bug):

* ``sigma_GP`` vs jitter as ``rho -> cadence``: an unresolved GP is
  indistinguishable from white noise.
* GP vs baseline as ``rho -> T``: a GP longer than the baseline is a constant.
* GP vs transit depth when ``rho ~ transit duration``: the GP can eat the
  signal.  This is the one that biases parameters you care about.
* Period harmonics (P, P/2, 2P) in rotation kernels: genuinely multimodal.
  Seed chains inside one periodogram-selected mode --
  :func:`periodogram_period` is the legitimate use of Whittle machinery
  (the Whittle *likelihood* is a quasi-likelihood and fails the exactness
  rule, but a periodogram peak is just an initialisation).

Geometry note, to calibrate fear: the marginalised GP has no Neal's-funnel
latent hierarchy.  What is left is mild ridges, which ChEES's shared diagonal
preconditioner plus trajectory adaptation handle -- conditional on the
parametrisation above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .likelihood import KernelParam, ParamMap
from .terms import (Matern32Term, Matern52Term, RealTerm, RotationTerm,
                    SHOTerm, Term, TermSum)

__all__ = ["Hyper", "SpecHint", "default_hypers", "build_param_map",
           "make_log_prior", "periodogram_period", "data_references",
           "auto_order_groups", "jitter_hyper", "offset_hyper"]


@dataclass
class SpecHint:
    """Bounds/scaling for one model-space dimension.

    Deliberately a plain dataclass rather than ``anvil.ParamSpec``: anvilgp
    depends on mlx and numpy only, and anvil is imported lazily where it is
    needed (mirroring ``metalplanet/anvil.py``).  :meth:`to_anvil` converts.
    """

    name: str
    lo: float = -math.inf
    hi: float = math.inf
    loc: float = 0.0
    scale: float = 1.0

    def to_anvil(self):
        import anvil
        return anvil.ParamSpec(name=self.name, lo=self.lo, hi=self.hi,
                               loc=self.loc, scale=self.scale)


@dataclass
class Hyper:
    """One sampled hyperparameter: link, host reference, bounds, prior."""

    name: str
    link: str = "log"
    ref: float = 1.0
    lo: float = -math.inf
    hi: float = math.inf
    scale: float = 1.0
    #: ("normal", mu, sigma) in *model* space (the log-offset), or None for
    #: the improper-flat/bounded-uniform default implied by the bounds.
    prior: tuple | None = None

    def spec(self) -> SpecHint:
        return SpecHint(self.name, lo=self.lo, hi=self.hi, scale=self.scale)


def data_references(t, y, yerr) -> dict:
    """The float64 host constants every default below is expressed against."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yerr = np.asarray(yerr, dtype=np.float64)
    d = np.diff(t)
    d = d[d > 0]
    return dict(
        rms=float(np.sqrt(np.mean((y - y.mean()) ** 2))),
        median_yerr=float(np.median(yerr)),
        cadence=float(np.median(d)) if d.size else 1.0,
        baseline=float(t[-1] - t[0]),
    )


def _log_rho_bounds(ref: dict):
    """log rho uniform on [2 dt_med, T/2], as an offset from the geometric
    mean of those two -- the widest range the data can say anything about.
    Below 2 dt_med the kernel is unresolved (it degenerates into the jitter);
    above T/2 it is a baseline offset."""
    lo, hi = 2.0 * ref["cadence"], 0.5 * ref["baseline"]
    if not hi > lo:                     # pathological, e.g. 3 data points
        lo, hi = 0.5 * ref["cadence"], 4.0 * ref["baseline"]
    if not (lo > 0.0 and hi > 0.0):
        # zero baseline: one epoch, or every timestamp identical.  There is
        # no timescale to put a prior on.
        raise ValueError(
            "cannot choose a timescale prior: the data span zero time "
            f"(baseline = {ref['baseline']:.3g}, cadence = "
            f"{ref['cadence']:.3g}). Supply hyperparameters explicitly.")
    rho_ref = math.sqrt(lo * hi)
    return rho_ref, math.log(lo / rho_ref), math.log(hi / rho_ref)


def default_hypers(term: Term, t, y, yerr, prefix: str = "gp",
                   period_ref: float | None = None) -> list[Hyper]:
    """Default :class:`Hyper` for every parameter of ``term``, in term order.

    ``period_ref`` seeds a :class:`~anvilgp.terms.RotationTerm`; pass the
    peak from :func:`periodogram_period` so the chains start inside one
    harmonic rather than spread over P, P/2 and 2P.
    """
    ref = data_references(t, y, yerr)
    if isinstance(term, TermSum):
        out: list[Hyper] = []
        for i, sub in enumerate(term.terms):
            out.extend(default_hypers(sub, t, y, yerr, f"{prefix}{i}",
                                      period_ref))
        return out

    rho_ref, rho_lo, rho_hi = _log_rho_bounds(ref)
    # Floor the amplitude reference at the median error bar.  `rms` is 0 for
    # constant y (a flat-lined detector, a fully-detrended column), and a
    # zero reference pins sigma at exactly 0 for the whole run through the
    # log link -- a GP that is white noise, silently, from the first step.
    sigma_ref = max(float(ref["rms"]), float(ref["median_yerr"]))
    sigma = Hyper(f"log_{prefix}_sigma", ref=sigma_ref, scale=1.5,
                  prior=("normal", 0.0, 1.5))

    if isinstance(term, SHOTerm):
        return [sigma,
                Hyper(f"log_{prefix}_rho", ref=rho_ref, lo=rho_lo, hi=rho_hi),
                Hyper(f"log_{prefix}_Q", ref=3.0, lo=math.log(0.05 / 3.0),
                      hi=math.log(500.0 / 3.0), prior=("normal", 0.0, 1.0))]
    if isinstance(term, (Matern32Term, Matern52Term)):
        return [sigma,
                Hyper(f"log_{prefix}_rho", ref=rho_ref, lo=rho_lo, hi=rho_hi)]
    if isinstance(term, RealTerm):
        return [sigma,
                Hyper(f"log_{prefix}_c", ref=1.0 / rho_ref,
                      lo=-rho_hi, hi=-rho_lo)]
    if isinstance(term, RotationTerm):
        p_ref = period_ref or rho_ref
        return [sigma,
                Hyper(f"log_{prefix}_P", ref=p_ref,
                      lo=math.log(2.0 * ref["cadence"] / p_ref),
                      hi=math.log(ref["baseline"] / p_ref)),
                Hyper(f"log_{prefix}_Q0", ref=3.0, lo=math.log(0.05 / 3.0),
                      hi=math.log(500.0 / 3.0), prior=("normal", 0.0, 1.0)),
                Hyper(f"log_{prefix}_dQ", ref=1.0, lo=math.log(1e-3),
                      hi=math.log(1e3), prior=("normal", 0.0, 2.0)),
                Hyper(f"logit_{prefix}_f", link="logit",
                      prior=("normal", 0.0, 2.0))]
    raise TypeError(f"no default hyperparameters for {type(term).__name__}")


def jitter_hyper(yerr, name: str = "log_jitter") -> Hyper:
    """log jitter ~ N(log median(yerr), 1^2), floored well below the errors.

    The lower bound matters: as jitter -> 0 the parameter is unidentifiable
    (it stops changing the likelihood), and an unbounded log-jitter sends
    chains to -inf along a perfectly flat direction.
    """
    med = float(np.median(np.asarray(yerr, dtype=np.float64)))
    return Hyper(name, ref=med, lo=math.log(1e-3), hi=math.log(1e3),
                 prior=("normal", 0.0, 1.0))


def offset_hyper(y, name: str = "offset") -> Hyper:
    """An additive per-instrument mean offset, linear in model space."""
    y = np.asarray(y, dtype=np.float64)
    s = float(np.sqrt(np.mean((y - y.mean()) ** 2))) or 1.0
    return Hyper(name, link="linear", ref=0.0, scale=s,
                 prior=("normal", 0.0, 3.0))


#: position of the timescale-like parameter within each term's parameter
#: list.  Every v1 term happens to put it at index 1 (rho, period, or the
#: rate c for RealTerm, which orders equivalently); the table is explicit so
#: a new term cannot silently inherit the wrong one.
_ORDER_INDEX = {SHOTerm: 1, Matern32Term: 1, Matern52Term: 1,
                RealTerm: 1, RotationTerm: 1}


def auto_order_groups(term: Term) -> list[list[int]]:
    """Ordering groups for the same-type terms of a sum.

    A sum of ``k`` identical-type terms has ``k!`` posterior modes related by
    permuting them.  Every convergence diagnostic is meaningless across those
    modes, and chains label-switch forever.  Constraining the timescales of
    each same-type family to be increasing picks one mode; the
    reparametrisation in :meth:`ParamMap.kernel_params` does it by
    construction rather than by rejection.

    Returns positions in the *kernel parameter vector*, ready to hand to
    ``ParamMap(order_groups=...)``.  Empty unless the term is a sum with at
    least two members of the same type.

    **What the bounds mean for an ordered member.**  For the first member of
    a group the sampled quantity is the timescale itself, so
    :func:`_log_rho_bounds` applies as written: log-uniform on
    ``[2 dt_med, T/2]``.  For every member after it the sampled quantity is
    the dimensionless *ratio* to the one before, and the same bounds become
    bounds on that ratio: ``exp(v)`` runs over ``[2 dt_med, T/2] / rho_ref``
    and the timescale is ``rho_prev * (1 + exp(v))``.  At ``v = 0`` the
    ratio is exactly 2 whatever unit the epochs are in.
    """
    if not isinstance(term, TermSum):
        return []
    by_type: dict = {}
    offset = 0
    for sub in term.terms:
        idx = _ORDER_INDEX.get(type(sub))
        if idx is not None:
            by_type.setdefault(type(sub), []).append(offset + idx)
        offset += sub.n_params
    return [g for g in by_type.values() if len(g) > 1]


def build_param_map(hypers: list[Hyper], n_mean: int = 0,
                    n_kernel: int | None = None,
                    jitters: list[Hyper] | None = None,
                    offsets: list[Hyper] | None = None,
                    instrument: np.ndarray | None = None,
                    order_groups: list[list[int]] | None = None):
    """Assemble a :class:`~anvilgp.likelihood.ParamMap` and its spec hints.

    Layout of ``v``: ``[mean block | kernel | jitters | offsets]``.  Returns
    ``(param_map, spec_hints, names)``; the mean block's own specs are the
    caller's business (they belong to the mean model, not to the GP).
    """
    jitters = jitters or []
    offsets = offsets or []
    idx = n_mean
    kernel, specs, names = [], [], []

    def emit(hs, sink):
        nonlocal idx
        for h in hs:
            sink.append(KernelParam(link=h.link, index=idx, ref=h.ref,
                                    scale=h.scale))
            specs.append(h.spec())
            names.append(h.name)
            idx += 1

    jit_params: list[KernelParam] = []
    off_params: list[KernelParam] = []
    emit(hypers, kernel)
    emit(jitters, jit_params)
    emit(offsets, off_params)

    pm = ParamMap(kernel=kernel, n_dim=idx, jitter=jit_params,
                  offset=off_params, instrument=instrument,
                  mean=slice(0, n_mean) if n_mean else None,
                  order_groups=order_groups)
    return pm, specs, names


def make_log_prior(hypers, jitters=None, offsets=None, n_mean: int = 0):
    """Batched MLX log-prior over the model-space vector ``v``.

    Only the explicitly-Gaussian priors appear here; bounded parameters get
    their uniform prior (and its Jacobian) from anvil's ``Transform``, which
    is exact.  Returns ``(n_chains, dim) -> (n_chains,)``.
    """
    terms = []
    idx = n_mean
    for h in list(hypers) + list(jitters or []) + list(offsets or []):
        if h.prior is not None:
            kind, mu, sd = h.prior
            if kind != "normal":
                raise ValueError(f"unsupported prior {kind!r}")
            terms.append((idx, float(mu), float(sd)))
        idx += 1

    if not terms:
        return lambda v: mx.zeros(v.shape[:1], dtype=v.dtype)

    cols = mx.array(np.array([t[0] for t in terms], dtype=np.int32))
    mus = np.array([t[1] for t in terms], dtype=np.float64)
    sds = np.array([t[2] for t in terms], dtype=np.float64)
    norm = float(-np.sum(np.log(sds) + 0.5 * math.log(2.0 * math.pi)))

    cache: dict = {}

    def log_prior(v: mx.array) -> mx.array:
        # Built once per dtype, from the float64 host values -- going via a
        # float32 numpy intermediate would quietly round the prior's own
        # location and scale before the float64 path ever saw them.
        if v.dtype not in cache:
            cache[v.dtype] = (mx.array(mus, dtype=v.dtype),
                              mx.array(sds, dtype=v.dtype))
        mu, sd = cache[v.dtype]
        z = (mx.take(v, cols, axis=1) - mu) / sd
        return -0.5 * mx.sum(z * z, axis=-1) + norm

    return log_prior


def periodogram_period(t, y, min_period=None, max_period=None, n_grid=20000):
    """Peak of a Lomb-Scargle periodogram, for seeding a rotation kernel.

    Legitimate use of spectral machinery: an *initialisation*, not a
    likelihood.  The Whittle likelihood itself is a quasi-likelihood and
    fails the exactness rule this package is built on -- but the chains have
    to start somewhere, and starting them scattered across P, P/2 and 2P
    turns a multimodal posterior into a convergence-diagnostic failure.

    Pure numpy: an O(n_grid * N) direct evaluation, once, on the host.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    d = np.diff(t)
    d = d[d > 0]
    cadence = float(np.median(d)) if d.size else 1.0
    baseline = float(t[-1] - t[0])
    p_lo = min_period or 2.0 * cadence
    p_hi = max_period or baseline
    periods = np.exp(np.linspace(math.log(p_lo), math.log(p_hi), n_grid))
    w = 2.0 * np.pi / periods
    power = np.empty(n_grid)
    for i in range(0, n_grid, 512):                  # blocked to bound memory
        wi = w[i:i + 512][:, None]
        wt = wi * t[None, :]
        s2 = np.sum(np.sin(2 * wt), axis=1)
        c2 = np.sum(np.cos(2 * wt), axis=1)
        tau = 0.5 * np.arctan2(s2, c2) / wi[:, 0]
        ph = wi * (t[None, :] - tau[:, None])
        c, s = np.cos(ph), np.sin(ph)
        yc, ys = y @ c.T, y @ s.T
        cc, ss = np.sum(c * c, axis=1), np.sum(s * s, axis=1)
        power[i:i + 512] = 0.5 * (yc ** 2 / np.maximum(cc, 1e-300)
                                  + ys ** 2 / np.maximum(ss, 1e-300))
    return float(periods[int(np.argmax(power))])
