"""Headroom per information channel on the paired.py LOBO draws (runs 0-19, k=1, platform-like sizes).
Floors for a pair's evaluation rows (lower = more information):
  pair      : (y - r)^2 with r = the pair's acquisition-pool rate (all ~31-113 acquisition items; a pair-level limit)
  group     : r_g = Beta-smoothed rate of the pair's acquisition items in the eval item's first-token group (k=2 toward r)
  rasch     : crossed Rasch fit on ALL public responses of the benchmark (in-sample; an optimistic item-level limit)
Compared with the base model's B15/B31 Brier on the same draws (results/paired_mix.csv, model 'base')."""
import sys
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np, pandas as pd
from paiec.data import load_responses, BINARY_BENCHMARKS
from paiec.evaluate import FoldData

df = load_responses()
folds = FoldData(df, BINARY_BENCHMARKS)
res = pd.read_csv("results/paired_mix.csv")
base = res[(res.model == "base") & (res.k == 1)].set_index(["bench", "run", "pair"])

def rasch(g, iters=30, ridge=0.05):
    s_idx = {s: i for i, s in enumerate(g.subject_id.unique())}
    i_idx = {s: i for i, s in enumerate(g.item_id.unique())}
    S = g.subject_id.map(s_idx).values; I = g.item_id.map(i_idx).values; y = g.label.values.astype(float)
    a = np.zeros(len(s_idx)); d = np.zeros(len(i_idx))
    for _ in range(iters):
        for which in (0, 1):
            eta = a[S] + d[I]; p = 1 / (1 + np.exp(-eta)); w = p * (1 - p)
            idx, par = (S, a) if which == 0 else (I, d)
            gr = np.bincount(idx, y - p, len(par)) - ridge * par
            h = np.bincount(idx, w, len(par)) + ridge
            par += np.clip(gr / h, -2, 2)
    return s_idx, i_idx, a, d

rows = []
for bench in ("matharena", "multi_swebench", "real_webagents", "researchcodebench"):
    g = df[df.bench == bench]
    s_idx, i_idx, a, d = rasch(g)
    sid_of_pair = g.drop_duplicates("pair").set_index("pair").subject_id
    for r, pairs in folds.draws(bench, 1, 20, 0):
        for p in pairs:
            acq = [y for i in p.acq for y in p.items[i]["labels"]]
            rp = np.mean(acq)
            grp = {}
            for i in p.acq:
                f = p.items[i]["item"]["item_features"]; key = f.split(";")[0] if f else ""
                k, n = grp.get(key, (0, 0)); grp[key] = (k + sum(p.items[i]["labels"]), n + len(p.items[i]["labels"]))
            sid = sid_of_pair[p.key]; ai = a[s_idx[sid]]
            e_pair = e_grp = e_rasch = 0.0; m = 0
            for i in p.eval:
                f = p.items[i]["item"]["item_features"]; key = f.split(";")[0] if f else ""
                k, n = grp.get(key, (0, 0)); rg = (k + 2 * rp) / (n + 2)
                pr = 1 / (1 + np.exp(-(ai + d[i_idx[i]])))
                for y in p.items[i]["labels"]:
                    e_pair += (y - rp) ** 2; e_grp += (y - rg) ** 2; e_rasch += (y - pr) ** 2; m += 1
            b = base.loc[(bench, r, p.key)]
            rows.append({"bench": bench, "pair_floor": e_pair / m, "group_floor": e_grp / m, "rasch_floor": e_rasch / m,
                         "base_b7": b.b7, "base_b15": b.b15, "base_b31": b.b31})
out = pd.DataFrame(rows).groupby("bench").mean()
out.loc["macro"] = out.mean()
print(out.round(4).to_string())
