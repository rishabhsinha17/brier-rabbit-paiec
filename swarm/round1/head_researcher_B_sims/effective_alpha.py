"""Effective likelihood power of HierModel's Laplace+probit: alpha such that the exact alpha-tempered posterior
predictive best matches the Laplace predictions (prior-predictive weighted), m0=-1.5, single pair."""
import sys, math
import numpy as np
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + '/swarm/round1/head_researcher_B_sims'))
from check_laplace import model_pred, pbar
m0, v, s = -1.5, 2.94, 0.6
L = np.linspace(m0-9*math.sqrt(v), m0+9*math.sqrt(v), 4001)
prior = np.exp(-0.5*(L-m0)**2/v); prior /= prior.sum(); pb = pbar(L, s)
for n in (1, 3, 7, 15):
    pk = np.array([float(prior @ (math.comb(n, k)*pb**k*(1-pb)**(n-k))) for k in range(n+1)])
    lap = np.array([model_pred(m0, k, n) for k in range(n+1)])
    best = None
    for a in np.linspace(0.4, 1.2, 81):
        pr = []
        for k in range(n+1):
            lp = np.log(prior) + a*(k*np.log(pb) + (n-k)*np.log1p(-pb))
            w = np.exp(lp-lp.max()); w /= w.sum(); pr.append(float(w @ pb))
        err = float(pk @ (np.array(pr)-lap)**2)
        if best is None or err < best[0]: best = (err, a)
    print(f"n={n:2d}: effective alpha of Laplace+probit = {best[1]:.2f} (residual rms {math.sqrt(best[0]):.4f})")
