"""Exploratory, no replay: does ability measured on hard benchmarks predict the original (live-like) folds better than ability from all
benchmarks? For each fold: centred pair logit on the held-out benchmark regressed on theta. Global numbers only."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "swarm/round12")
import full_prior
rows = pd.read_pickle("swarm/round12/pair_rates.pkl")
level = rows.groupby("bench").apply(lambda g: float((g["s"] / g["n"]).mean()), include_groups=False)
print("benchmarks in fits by mean success: <0.3:", int((level < .3).sum()), "| 0.3-0.5:", int(((level >= .3) & (level < .5)).sum()), "| >=0.5:", int((level >= .5).sum()))
print("original folds' mean success:", {b: round(level[b], 3) for b in ("matharena", "multi_swebench", "real_webagents", "researchcodebench")})
def score(y, x):
    ok = np.isfinite(x)
    if ok.sum() < 5 or np.std(x[ok]) == 0: return np.nan, np.nan, 0
    b = np.polyfit(x[ok], y[ok], 1)[0]
    return np.corrcoef(x[ok], y[ok])[0, 1], b, int(ok.sum())
def variants(bench):
    ref = json.load(open(f"swarm/round12/cache/ref_prior_{bench}.json"))
    r = rows[rows["bench"] != bench]
    out = {"shipped-style": {k: v[0] for k, v in ref["names"].items()}}
    out["F2 all"] = {k: v[0] for k, v in full_prior.fit(r, "F2")["names"].items()}
    for thr in (0.5, 0.35):
        keep = level[level < thr].index
        keep = set(keep) | {"swe_rebench"}
        out[f"F2 mean success <{thr}"] = {k: v[0] for k, v in full_prior.fit(r[r["bench"].isin(keep)], "F2")["names"].items()}
    return out
res = []
for bench in ("real_webagents", "researchcodebench", "multi_swebench", "matharena"):
    g = rows[(rows["bench"] == bench) & (rows["key"] != "")].copy()
    g["y"] = np.log((g["s"] + .5) / (g["n"] - g["s"] + .5))
    y = (g.groupby("key")["y"].mean() - g["y"].mean())
    v = variants(bench)
    common = [k for k in y.index if all(k in d for d in v.values())]
    for name, d in v.items():
        x = np.array([d.get(k, np.nan) for k in y.index])
        c_all, b_all, n_all = score(y.to_numpy(), x)
        x2 = np.array([d[k] for k in common]); c2, b2, n2 = score(y.loc[common].to_numpy(), x2)
        res.append({"fold": bench, "variant": name, "names covered": n_all, "corr": round(c_all, 3), "slope": round(b_all, 2), "corr on common names": round(c2, 3), "common": n2})
t = pd.DataFrame(res)
print(t.to_string(index=False))
print(t.groupby("variant")[["names covered", "corr", "slope", "corr on common names"]].mean().round(3).to_string())
