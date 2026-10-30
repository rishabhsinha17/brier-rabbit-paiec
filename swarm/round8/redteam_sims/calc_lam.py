"""Theta weight (lam) and attribute/name usage in the alone-pair calculus: W20 at lam in {0.5..1.5} vs W20 (lam 1),
public world and the round-4 live posterior (pool model keeps the public theta-rate relation, slope 1, offset d)."""
import math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calc_state as cs
BUD = cs.BUD
if __name__ == "__main__":
    el, th, tv = cs.setup()
    w = cs.fr.bench_weights(el)
    worlds = {"public": [({"shift": 0.0, "scale": 1.0, "pi_low": 0.0}, 0.0, 1.0)], "live": cs.posterior_cells()}
    lams = (0.5, 0.75, 1.25, 1.5)
    for wname, cells in worlds.items():
        acc = {}
        for pool, d, wt in cells:
            pool = dict(pool); pool.setdefault("mu_low", math.log(0.06 / 0.94)); pool.setdefault("sd_low", 0.5)
            r = cs.fr.transform(el, pool)
            for br in (0, 1):
                e0 = cs.expected(cs.table_cached(cs.W20, th, tv, d, br), r, w)
                for lam in lams:
                    # hidden subjects: theta_eff = theta + d enters the prior mean as lam * theta + d (offset not scaled)
                    lvl = dict(cs.W20, m0=cs.W20["m0"] + d - lam * d)
                    q = cs.table_cached(lvl, th, tv, d, br, tag=f"lam{lam}", lam=lam)
                    e = cs.expected(q, r, w)
                    a = acc.setdefault(lam, {n: 0.0 for n in (*BUD, "alc")})
                    for n in (*BUD, "alc"):
                        a[n] += 0.5 * wt * 1e3 * (e[n] - e0[n])
        print(f"\n=== {wname}: W20 at lam - W20 at lam 1, x1e-3")
        for lam, v in acc.items():
            print(f"  lam {lam:4.2f}  ALC {v['alc']:+.3f}  " + " ".join(f"B{n} {v[n]:+.2f}" for n in BUD))
        sys.stdout.flush()
