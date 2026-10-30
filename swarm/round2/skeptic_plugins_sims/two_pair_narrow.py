"""Round-1 two-pair exact calculus (swarm/round1/postdoc_adaptation_sims/two_pair_exact.py, same formulas and tables)
re-scored on the pools that round 1's priors postdoc found consistent with the live global summaries: the public
regular component with its logit spread scaled around the pooled mean ('scale', as fast_eval.transform_pool) and
shifted ('shift'), plus an optional pair-level near-zero share pi. Tables from ./tables (two_pair_rebuild.py).
usage: python two_pair_narrow.py [ref]
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta, binom  # noqa: E402  (analysis only)

from paiec.data import DATA_ROOT  # noqa: E402

TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
BUD = (0, 1, 3, 7, 15, 31)
W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
rates = {}
for b in ("matharena", "multi_swebench", "real_webagents", "researchcodebench"):
    r = pd.read_parquet(DATA_ROOT / b / "response.parquet", columns=["subject_id", "item_id", "response"])
    g = r[r.response.isin([0., 1.])].groupby("subject_id").agg(rate=("response", "mean"), n=("item_id", "nunique"))
    rates[b] = np.clip(g[g.n >= 80].rate.values, 1e-3, 1 - 1e-3)
ALL = np.concatenate(list(rates.values()))
MU = float(np.mean(np.log(ALL / (1 - ALL))))
hq = beta.ppf((np.arange(50) + 0.5) / 50, 0.5, 30)


def transform(r, scale, shift):
    lg = np.log(r / (1 - r))
    lg = MU + scale * (lg - MU) + shift
    return 1 / (1 + np.exp(-lg))


def combos(scale, shift, pi):
    out = []
    pHH, pHN, pNN = pi * pi, pi * (1 - pi), (1 - pi) ** 2
    for b, r in rates.items():
        r = transform(r, scale, shift)
        n = len(r)
        i, j = np.where(~np.eye(n, dtype=bool))
        out.append((r[i], r[j], np.full(len(i), 0.25 / (n * (n - 1)) * pNN)))
        hT = np.repeat(hq, n)
        rPr = np.tile(r, len(hq))
        out.append((hT, rPr, np.full(len(hT), 0.25 / (n * len(hq)) * pHN)))
        out.append((rPr, hT, np.full(len(hT), 0.25 / (n * len(hq)) * pHN)))
    a, c = np.meshgrid(hq, hq, indexing="ij")
    out.append((a.ravel(), c.ravel(), np.full(a.size, pHH / a.size)))
    rT = np.concatenate([o[0] for o in out])
    rP = np.concatenate([o[1] for o in out])
    w = np.concatenate([o[2] for o in out])
    keep = w > 0
    return rT[keep], rP[keep], w[keep] / w[keep].sum()


def pmfs(rT, rP):
    return {n: (binom.pmf(np.arange(n + 1)[None, :], n, rT[:, None]), binom.pmf(np.arange(n + 1)[None, :], n, rP[:, None]))
            for n in BUD}


def alc(tab, rT, rP, w, pm):
    tot = 0.0
    for n in BUD:
        q = tab[f"n{n}"]
        pT, pP = pm[n]
        e2 = np.sum((pT @ (q ** 2)) * pP, axis=1)
        e1 = np.sum((pT @ q) * pP, axis=1)
        tot += W[n] * np.sum(w * (e2 - 2 * rT * e1 + rT ** 2 + rT * (1 - rT)))
    return tot


def alc_alone(tab, rT, w, pm):
    tot = 0.0
    for n in BUD:
        q = tab[f"a{n}"]
        pT = pm[n][0]
        tot += W[n] * np.sum(w * ((pT @ (q ** 2)) - 2 * rT * (pT @ q) + rT ** 2 + rT * (1 - rT)))
    return tot


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "cur_m20"
    tabs = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(TAB + "/*.npz"))}
    scen = [(1.0, 0.0, 0.0), (0.8, -0.5, 0.0), (0.7, -0.5, 0.0), (0.8, -0.5, 0.15), (1.0, -1.0, 0.0), (0.8, -1.0, 0.0)]
    cols = [f"s{s}/sh{sh}/pi{pi}" for s, sh, pi in scen]
    for title, fn in (("shared pair", alc), ("alone pair", None)):
        print(f"\n{title}: expected ALC, {ref} absolute, others x1e-3 vs {ref} (negative = better)")
        print(f"{'model':>16} " + " ".join(f"{c:>15}" for c in cols))
        res = {}
        for sc in scen:
            rT, rP, w = combos(*sc)
            pm = pmfs(rT, rP)
            res[sc] = {m: (alc(t, rT, rP, w, pm) if fn else alc_alone(t, rT, w, pm)) for m, t in tabs.items()}
        for m in tabs:
            vals = [f"{res[sc][m]:15.4f}" if m == ref else f"{1e3 * (res[sc][m] - res[sc][ref]):+15.2f}" for sc in scen]
            print(f"{m:>16} " + " ".join(vals))


if __name__ == "__main__":
    main()
