"""Exact expected Brier ALC of a pair that shares its benchmark with one partner, for prediction tables built by
two_pair_tables.py, under hypothetical hidden pools (public-data regular component + near-zero 'hard' component).
Regular component: two distinct public pairs of the same benchmark (benchmark-equal weights; empirical rates),
optionally shifted on the logit scale. Hard component: rate ~ Beta(0.5, 30) (50 quantile nodes).
Hardness structure: 'pair' (independent, prob pi), 'bench' (both or neither, prob pi), 'mix' (half the hard mass
benchmark-wide, half pair-level, same marginal pi). Items exchangeable: E[Brier | q, r] = (q - r)^2 + r (1 - r)."""
import sys, glob, os
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np, pandas as pd
from scipy.stats import binom, beta
from paiec.data import DATA_ROOT

TAB = __import__("os").environ.get("TPE_TAB", (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/swarm/round1/postdoc_adaptation_sims/tables"))
BUD = (0, 1, 3, 7, 15, 31); W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
rates = {}
for b in ("matharena", "multi_swebench", "real_webagents", "researchcodebench"):
    r = pd.read_parquet(DATA_ROOT / b / "response.parquet", columns=["subject_id", "item_id", "response"])
    g = r[r.response.isin([0., 1.])].groupby("subject_id").agg(rate=("response", "mean"), n=("item_id", "nunique"))
    rates[b] = np.clip(g[g.n >= 80].rate.values, 1e-3, 1 - 1e-3)
hq = beta.ppf((np.arange(50) + 0.5) / 50, 0.5, 30)

def combos(shift, pi, struct):
    """Return arrays rT, rP, weight for one hidden-pool hypothesis."""
    out = []
    for b, r in rates.items():
        if shift:
            r = 1 / (1 + np.exp(-(np.log(r / (1 - r)) + shift)))
        n = len(r)
        i, j = np.where(~np.eye(n, dtype=bool))
        wb = 0.25 / (n * (n - 1))
        if struct == "pair":
            pHH, pHN, pNN = pi * pi, pi * (1 - pi), (1 - pi) ** 2
        elif struct == "bench":
            pHH, pHN, pNN = pi, 0.0, 1 - pi
        else:  # half the hard mass benchmark-wide, half pair-level with the same marginal
            pb = pi / 2; pp = (pi - pb) / (1 - pb)
            pHH = pb + (1 - pb) * pp * pp; pHN = (1 - pb) * pp * (1 - pp); pNN = (1 - pb) * (1 - pp) ** 2
        out.append((r[i], r[j], np.full(len(i), wb * pNN)))              # both regular
        hT = np.repeat(hq, n); rPr = np.tile(r, len(hq))
        out.append((hT, rPr, np.full(len(hT), 0.25 / (n * len(hq)) * pHN)))   # T hard, P regular
        out.append((rPr, hT, np.full(len(hT), 0.25 / (n * len(hq)) * pHN)))   # T regular, P hard
    a, c = np.meshgrid(hq, hq, indexing="ij")
    out.append((a.ravel(), c.ravel(), np.full(a.size, pHH / a.size)))      # both hard
    rT = np.concatenate([o[0] for o in out]); rP = np.concatenate([o[1] for o in out]); w = np.concatenate([o[2] for o in out])
    return rT, rP, w / w.sum()

def pmfs(rT, rP):
    return {n: (binom.pmf(np.arange(n + 1)[None, :], n, rT[:, None]), binom.pmf(np.arange(n + 1)[None, :], n, rP[:, None]))
            for n in BUD}

def alc_alone(tab, rT, w, pm):
    tot = 0.0
    for n in BUD:
        q = tab[f"a{n}"]
        pT = pm[n][0]
        tot += W[n] * np.sum(w * ((pT @ (q ** 2)) - 2 * rT * (pT @ q) + rT ** 2 + rT * (1 - rT)))
    return tot

def alc(tab, rT, rP, w, pm):
    tot = 0.0
    for n in BUD:
        q = tab[f"n{n}"]
        pT, pP = pm[n]
        e2 = np.sum((pT @ (q ** 2)) * pP, axis=1); e1 = np.sum((pT @ q) * pP, axis=1)
        tot += W[n] * np.sum(w * (e2 - 2 * rT * e1 + rT ** 2 + rT * (1 - rT)))
    return tot

tabs = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(TAB + "/*.npz"))}
ref = sys.argv[1] if len(sys.argv) > 1 else "mix35_m10"
import os as _os
if _os.environ.get("TPE_SCEN") == "compact":
    scen = [(0.0, 0.0, "pair")] + [(0.0, pi, s) for s in ("pair", "bench") for pi in (0.15, 0.35, 0.5)] + [(-0.5, 0.35, "pair")]
else:
    scen = [(0.0, pi, s) for s in ("pair", "mix", "bench") for pi in (0.15, 0.35, 0.5)] + [(-0.5, 0.35, s) for s in ("pair", "bench")]
res, res_al = {}, {}
for sh, pi, st in scen:
    rT, rP, w = combos(sh, pi, st)
    keep = w > 0
    rT, rP, w = rT[keep], rP[keep], w[keep]
    pm = pmfs(rT, rP)
    res[(sh, pi, st)] = {m: alc(t, rT, rP, w, pm) for m, t in tabs.items()}
    res_al[(sh, pi, st)] = {m: alc_alone(t, rT, w, pm) for m, t in tabs.items()}
cols = [f"{st}{'' if not sh else '/sh' + str(sh)} pi{pi}" for sh, pi, st in scen]
print(f"Shared-pair expected ALC: {ref} absolute, others as difference x1e-3 (negative = better than {ref})")
print(f"{'model':>16} " + " ".join(f"{c:>17}" for c in cols))
for m in tabs:
    vals = []
    for k in scen:
        v = res[k][m] - res[k][ref]
        vals.append(f"{res[k][m]:17.4f}" if m == ref else f"{1e3 * v:+17.2f}")
    print(f"{m:>16} " + " ".join(vals))
print(f"\nAlone-pair expected ALC (same marginal pools): {ref} absolute, others x1e-3 difference")
print(f"{'model':>16} " + " ".join(f"{c:>17}" for c in cols))
for m in tabs:
    vals = [f"{res_al[k][m]:17.4f}" if m == ref else f"{1e3 * (res_al[k][m] - res_al[k][ref]):+17.2f}" for k in scen]
    print(f"{m:>16} " + " ".join(vals))
