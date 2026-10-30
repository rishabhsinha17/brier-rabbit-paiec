"""Ceiling for ANY pair-level predictor (level block, prior shape, state corrections) in the alone-pair calculus.

Oracle: the Bayes posterior mean of the pair rate given its own (k, n) AND its subject-prior bin, under the TRUE pool:
  q*(k, n | bin) = E[r | k, n, bin],  bins = quintiles of the pool's theta + d (pairs are benchmark-weighted).
It knows the pool exactly (an upper bound on what a level prior tuned to it could do). Reported: W20 - oracle and
REF - oracle per budget and ALC (x1e-3), public world and posterior mean over the round-4 live posterior cells.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calc_state as cs  # noqa: E402

BUD = cs.BUD


def oracle_table(th_eff, r, w, nbins=5):
    pmf = cs.fr.pmf_table(r)
    q = np.zeros_like(pmf)
    edges = np.quantile(th_eff, np.linspace(0, 1, nbins + 1))
    b = np.clip(np.searchsorted(edges, th_eff, side="right") - 1, 0, nbins - 1)
    for j in range(nbins):
        m = b == j
        A = w[m, None] * pmf[m]
        q[m] = ((A * r[m, None]).sum(0) / np.maximum(A.sum(0), 1e-300))[None, :]
    return q


if __name__ == "__main__":
    el, th, tv = cs.setup()
    w = cs.fr.bench_weights(el)
    worlds = {"public": [({"shift": 0.0, "scale": 1.0, "pi_low": 0.0}, 0.0, 1.0)], "live": cs.posterior_cells()}
    for wname, cells in worlds.items():
        acc = {}
        for pool, d, wt in cells:
            pool = dict(pool)
            pool.setdefault("mu_low", math.log(0.06 / 0.94))
            pool.setdefault("sd_low", 0.5)
            r = cs.fr.transform(el, pool)
            for nb in (1, 5):
                qo = oracle_table(th + d, r, w, nbins=nb)
                eo = cs.expected(qo, r, w)
                for br in (0, 1):
                    for arm, lvl in (("W20", cs.W20), ("REF", cs.REF)):
                        e = cs.expected(cs.table_cached(lvl, th, tv, d, br), r, w)
                        key = f"{arm} - oracle(theta bins {nb})"
                        acc.setdefault(key, {n: 0.0 for n in (*BUD, 'alc')})
                        for n in (*BUD, "alc"):
                            acc[key][n] += 0.5 * wt * 1e3 * (e[n] - eo[n])
        print(f"\n=== {wname}: regret vs the pool-aware Bayes oracle, x1e-3")
        for key, v in acc.items():
            print(f"  {key:28s} ALC {v['alc']:+.3f}  " + " ".join(f"B{n} {v[n]:+.2f}" for n in BUD))
        sys.stdout.flush()
