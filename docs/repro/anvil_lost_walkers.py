"""anvil-only reproducer: warmup_report says OK while ensemble walkers are lost.

Needs only anvil and MLX -- no anvil-gp. A 4-D standard normal whose model
"fails" beyond a radius, returning a constant large-negative value there:
the pattern any target with a float32-infeasible corner produces when it
reports failure as a finite number (anvil-gp and Elome both do).

Half the ensemble starts inside the failure region. Run it twice:

    python anvil_lost_walkers.py            # constant failure value
    python anvil_lost_walkers.py sloped     # failure value that slopes home

Prints how many walkers were lost, where they ended up, anvil's verdicts, and
the lost-chain check the brief proposes.
"""
import sys

import mlx.core as mx
import numpy as np

import anvil
from anvil.kernels.ensemble import EnsembleKernel

DIM, RADIUS, FAIL = 4, 25.0, -1e30
SLOPED = len(sys.argv) > 1 and sys.argv[1] == "sloped"


def model_log_prob(v):
    """Standard normal inside RADIUS; a 'failed' evaluation outside it."""
    r2 = mx.sum(v * v, axis=-1)
    good = -0.5 * r2
    if SLOPED:   # relative, so float32 can resolve it at 1e30
        failed = FAIL * (1.0 + mx.stop_gradient(mx.log1p(r2 / DIM)))
    else:        # a constant: a flat region
        failed = mx.full(r2.shape, FAIL, dtype=v.dtype)
    return mx.where(mx.sqrt(r2) < RADIUS, good, failed)


transform = anvil.Transform([anvil.ParamSpec(f"x{i}") for i in range(DIM)])
target = anvil.TransformedLogDensity(model_log_prob, transform)

rng = np.random.default_rng(5)
u0 = (0.3 * rng.standard_normal((64, DIM))).astype(np.float32)
u0[:32] = 15.0                    # |u| = 30, past the failure radius of 25

with mx.stream(mx.gpu):
    res = anvil.run(EnsembleKernel(target, seed=2), target, mx.array(u0),
                    n_warmup=400, n_samples=2000, seed=3)
    last = np.array(res.final_state["u"])
    lp = np.array(target.log_prob(mx.array(last)))

lost = lp <= 0.5 * FAIL
print(f"anvil {anvil.__version__} | failure value: "
      f"{'sloped' if SLOPED else 'constant'}")
print(f"lost walkers: {lost.sum()}/64"
      + (f", at |u| up to {np.linalg.norm(last[lost], axis=1).max():.2g}"
         if lost.any() else ""))
wr = anvil.warmup_report(res)
print(f"anvil.warmup_report: {wr.verdict} -- {wr.suggestion}")
print("anvil.diagnose:\n" + str(anvil.diagnose(
    np.asarray(res.get_chain(), dtype=np.float64))))

# The proposed lost-chain check: final log-prob far below the ensemble,
# measured robustly so the lost walkers cannot inflate the scale.
med = np.median(lp)
mad = 1.4826 * np.median(np.abs(lp - med))
flagged = np.where(lp < med - 10.0 * max(mad, 1e-12))[0]
print(f"proposed check flags {flagged.size} chain(s): {flagged.tolist()}")
