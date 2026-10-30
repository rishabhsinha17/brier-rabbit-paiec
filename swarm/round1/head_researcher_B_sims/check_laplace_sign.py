"""Direction of the Laplace+probit error (HierModel, pair alone on its benchmark) vs exact quadrature."""
import sys, math
import numpy as np
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + '/swarm/round1/head_researcher_B_sims'))
from check_laplace import exact_pred, model_pred, P, pbar
m0 = -1.5
v = P['tau_mu']**2 + P['tau_phi']**2 + P['tau_delta']**2 + P['lam']**2*P['tau_theta']**2
s = P['sigma_i']
L = np.linspace(m0 - 9*math.sqrt(v), m0 + 9*math.sqrt(v), 4001)
wL = np.exp(-0.5*(L-m0)**2/v); wL /= wL.sum(); pb = pbar(L, s)
for n in (1, 3, 7, 15):
    print(f"n={n}")
    for k in range(n+1):
        pk = float(wL @ (math.comb(n, k)*pb**k*(1-pb)**(n-k)))
        pe, _, _ = exact_pred(k, n, m0, v, s)
        pm = model_pred(m0, k, n)
        if pk > 0.004:
            print(f"   k={k:2d} P(k)={pk:.3f} exact={pe:.4f} laplace={pm:.4f} diff={pm-pe:+.4f}")
