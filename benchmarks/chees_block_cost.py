"""What the kernel's short-block gradient error costs ChEES (docs/float32.md s7, step 2).

ChEES on SHO + SHO at s/e = 100 with `make_target` defaults, at the
automatic `metal_block` (BLK 0) or an explicit one, e.g. `metal_block = N`.
Prints one JSON line: adapted step size, mean leapfrog count, acceptance,
max split R-hat, bulk ESS and ESS per 1000 gradient evaluations.

usage: chees_block_cost.py CASE BLK SEED NCH NWARM NSAMP MAXL [OUT.npy]
  docs table: t20s100 {0,4096} {1,2} 1024 300 200 128, and 0 at 2000 / 500
"""
import sys, time, json, warnings, numpy as np, mlx.core as mx
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import anvil, anvilgp as ag
from anvil.diagnostics import split_rhat, ess_bulk
from anvil.halton import halton_jitter
from anvilgp import GPPolicy
from anvilgp.anvil import make_target
from floor_exposure import sample, TWO_PI, GQ
warnings.simplefilter("ignore")
case, blk, seed, nch, nw, ns, maxl = sys.argv[1], *map(int, sys.argv[2:8])
assert case == "t20s100", case   # the one case measured so far
t, kp = np.arange(4096) / 4320, [0.1, 3/TWO_PI, 5.0, 0.03, 0.2, GQ]  # 20-s cadence
term = ag.SHOTerm() + ag.SHOTerm()
yerr = np.full(len(t), 1e-3)
y = sample(term, kp, t, yerr, np.random.default_rng(1))
pol = GPPolicy(metal=True, metal_block=blk or None)
gt = make_target(term, t, y, yerr, policy=pol)
print(case, "metal_block", gt.loglike.metal_block, gt.names, flush=True)
ker = anvil.ChEESHMC(gt.target, max_leapfrog=maxl)
t0 = time.time()
with mx.stream(mx.gpu):
    res = anvil.run(ker, gt.target, gt.u0(nch, seed=1, spread=0.3),
                    n_warmup=nw, n_samples=ns, seed=seed, progress=False)
wall = time.time() - t0
eps = float(np.array(res.final_params["step_size"])); T = float(np.array(res.final_params["traj_length"]))
Ls = [max(1, min(maxl, int(np.ceil(halton_jitter(i) * T / eps)))) for i in range(nw, nw + ns)]
ch = np.asarray(res.get_chain(), dtype=np.float64)
rh = split_rhat(ch); ess = ess_bulk(ch)
ngrad = nch * sum(Ls)
out = dict(case=case, blk=blk, seed=seed, eps=eps, T=T, meanL=float(np.mean(Ls)),
           accept=float(np.mean(res.accept_fraction)), ndiv=res.extras["n_divergent"],
           rhat_max=float(rh.max()), ess=ess.tolist(), ess_min=float(ess.min()),
           ess_per_kgrad=float(ess.min() / ngrad * 1e3), mean=ch.reshape(-1, ch.shape[-1]).mean(0).tolist(),
           sd=ch.reshape(-1, ch.shape[-1]).std(0).tolist(), wall=wall, lost=list(map(int, res.extras["lost_chains"])))
print(json.dumps(out), flush=True)
if len(sys.argv) > 8:
    np.save(sys.argv[8], ch.astype(np.float32))
