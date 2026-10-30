"""Cross-fitted version of calc_oracle: the bin-conditional discrete prior (pool rates of pairs in the same theta bin)
is built from the OTHER half of the public pairs (alternating within benchmark), so the oracle cannot use a pair's own
rate. Honest(er) ceiling for a pool-shaped, theta-binned prior. x1e-3, W20 - oracle."""
import math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calc_state as cs
BUD = cs.BUD

def oracle_cf(th_eff, r, w, bench, nbins):
    pmf = cs.fr.pmf_table(r)
    half = np.zeros(len(r), int)
    for b in np.unique(bench):
        idx = np.where(bench == b)[0]
        half[idx[1::2]] = 1
    q = np.zeros_like(pmf)
    for h in (0, 1):
        tr, te = half != h, half == h
        edges = np.quantile(th_eff[tr], np.linspace(0, 1, nbins + 1))
        btr = np.clip(np.searchsorted(edges, th_eff[tr], side="right") - 1, 0, nbins - 1)
        bte = np.clip(np.searchsorted(edges, th_eff[te], side="right") - 1, 0, nbins - 1)
        rtr, wtr, ptr = r[tr], w[tr], pmf[tr]
        for j in range(nbins):
            m = btr == j
            A = wtr[m, None] * ptr[m]
            qj = (A * rtr[m, None]).sum(0) / np.maximum(A.sum(0), 1e-300)
            q[np.where(te)[0][bte == j]] = qj[None, :]
    return q

if __name__ == "__main__":
    el, th, tv = cs.setup()
    w = cs.fr.bench_weights(el)
    bench = el.bench.values
    worlds = {"public": [({"shift": 0.0, "scale": 1.0, "pi_low": 0.0}, 0.0, 1.0)], "live": cs.posterior_cells()}
    for wname, cells in worlds.items():
        acc = {}
        for pool, d, wt in cells:
            pool = dict(pool); pool.setdefault("mu_low", math.log(0.06 / 0.94)); pool.setdefault("sd_low", 0.5)
            r = cs.fr.transform(el, pool)
            for nb in (1, 3, 5):
                eo = cs.expected(oracle_cf(th + d, r, w, bench, nb), r, w)
                for br in (0, 1):
                    e = cs.expected(cs.table_cached(cs.W20, th, tv, d, br), r, w)
                    a = acc.setdefault(nb, {n: 0.0 for n in (*BUD, "alc")})
                    for n in (*BUD, "alc"):
                        a[n] += 0.5 * wt * 1e3 * (e[n] - eo[n])
        print(f"\n=== {wname}: W20 - cross-fitted oracle, x1e-3")
        for nb, v in acc.items():
            print(f"  theta bins {nb}: ALC {v['alc']:+.3f}  " + " ".join(f"B{n} {v[n]:+.2f}" for n in BUD))
        sys.stdout.flush()
