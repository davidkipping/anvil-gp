"""anvilgp -- batched Gaussian-process likelihoods for the anvil MCMC engine.

Time-series GP likelihoods of the celerite class, evaluated for thousands of
parameter vectors at once on the Apple GPU, with gradients, so that anvil can
sample GP hyperparameters *jointly* with mean-model parameters.

Every kernel is represented as a linear Gaussian state-space model.  The
Kalman prediction-error decomposition and the celerite semiseparable
Cholesky compute the same marginal likelihood in exact arithmetic; the SSM
route is chosen for numerics and GPU scheduling, and it buys exact
Matern-3/2 and Matern-5/2 terms plus a propagator that is smooth through
Q = 1/2 (the celerite coefficient parametrisation is singular there).

Layers:

* :mod:`anvilgp.terms`      -- kernel-term algebra (propagator, stationary
  covariance, process noise) in closed forms that never cancel.
* :mod:`anvilgp.elements`   -- the affine-Gaussian scan-element algebra and
  its blocked tree *reduction* to the marginal likelihood.  Dtype-polymorphic
  MLX: fp64-on-CPU is the verification path, fp32-on-GPU the parity baseline,
  and the same algebra is what the Metal kernel executes.
* :mod:`anvilgp.kalman`     -- the sequential prediction-error filter, an
  independent implementation used to adjudicate the tree at small N.
* :mod:`anvilgp.likelihood` -- :class:`GPLogLike`, the anvil engine contract.
"""

from __future__ import annotations

from . import oracle, priors
from .likelihood import GPLogLike, GPPolicy, KernelParam, ParamMap, a64
from .terms import (
    ComplexTerm,
    Matern32Term,
    Matern52Term,
    RealTerm,
    RotationTerm,
    SHOTerm,
    Term,
    TermConvolution,
    TermSum,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "GPLogLike", "GPPolicy", "ParamMap", "KernelParam", "a64",
    "priors", "oracle",
    "Term", "TermSum", "RealTerm", "SHOTerm", "ComplexTerm", "TermConvolution",
    "Matern32Term", "Matern52Term", "RotationTerm",
]
