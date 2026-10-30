"""Check 0: refit posterior on the current evidence; reproduce the design grid's REF/W20 expected Brier with
hr_common.evaluate (must match lc.Grid to float32 precision); print posterior marginals vs the 10-06 design cache."""
import time

import numpy as np

import hr_common as hc

t0 = time.time()
s = hc.setup()
print(f"setup {time.time() - t0:.1f}s; pairs {len(s['th'])}; pools {len(s['pools'])}")
P = hc.posterior()
m = P.marginals()
print(f"posterior fit {time.time() - t0:.1f}s; obs {P.nobs}")
print("MAP", m["map"])
print("d:", {k: round(v, 3) for k, v in m["d"].items() if v > 0.005}, "E[d]", round(m["E[d]"], 2),
      "P(H_pub)", round(m["P(d<=0.85)"], 3))
for key in ("shift", "scale", "pi_low", "lam"):
    print(key, {k: round(v, 3) for k, v in m[key].items()})
print("pool mean r %.3f rr %.3f p<5%% %.3f" % (m["pool_mean_r"], m["pool_mean_rr"], m["pool_p_lt05"]))

# reproduce the grid
G = P.G
for name, prm in (("REF", hc.REFP), ("W20", hc.W20P)):
    T = hc.family_tables(prm)                         # (nD, 2, P, 63)
    E = hc.evaluate(T)                                # (nPool, nD, 2, 6)
    fam = {"m0": prm["m0"], "tau_mu": prm["tau_mu"], "tau_delta": prm["tau_delta"], "tau_phi": prm["tau_phi"]}
    worst = 0.0
    for j, d in enumerate(hc.DS):
        g = G.budgets(None, d, fam=fam)               # (nPool, 2, 6)
        worst = max(worst, float(np.abs(g - E[:, j]).max()))
    print(f"{name}: max |E_mine - E_grid| = {worst:.2e}")
print(f"total {time.time() - t0:.1f}s")
