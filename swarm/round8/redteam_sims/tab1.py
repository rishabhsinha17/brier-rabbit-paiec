"""Tables from cells.parquet: regret vs the pair-rate oracle by budget x pair type x k (W20+SCP), with bias sign."""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
c = pd.read_parquet(OUT / "cells.parquet")
c["bias"] = c.pbar - c.r
c["regret"] = c.brier - c.unc
w = c[c.model == "w20_SCP"]
pd.set_option("display.width", 250)
# pair-draw counts per type (budget 0 rows)
b0 = w[w.budget == 0]
print("pair-draws by type x bench (k pooled):")
print(pd.crosstab(b0.ptype, b0.bench))
print("\npair-draws by type x k:")
print(pd.crosstab(b0.ptype, b0.k))
t = w.groupby(["ptype", "budget"]).agg(n=("brier", "size"), regret=("regret", "mean"), lvl2=("lvl2", "mean"),
                                         item=("item", "mean"), bias=("bias", "mean")).reset_index()
t[["regret", "lvl2", "item", "bias"]] *= 1000
print("\nW20+SCP regret vs pair-rate oracle (x1e-3), by type x budget:")
print(t.round(2).to_string(index=False))
# ALC-weighted contribution to the macro-like mean (pair-draw-equal here)
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
w = w.assign(wt=w.budget.map(W))
tot = (w.regret * w.wt).groupby([w.bench, w.k, w.run, w.pair]).sum()
print("\nmean ALC regret per pair-draw (x1e-3):", round(tot.mean() * 1000, 2))
contrib = (w.regret * w.wt).groupby([w.ptype, w.budget]).sum() / tot.size * 1000
print("\ncontribution to mean ALC regret (x1e-3), type x budget:")
print(contrib.unstack().round(2).to_string())
print("\nby k:")
contrib_k = (w.regret * w.wt).groupby([w.k, w.budget]).sum() / w.groupby('k').apply(lambda g: g[['bench','run','pair']].drop_duplicates().shape[0]) .reindex(w.k).values[0] if False else None
for k in (1, 2):
    wk = w[w.k == k]
    n = wk[["bench", "run", "pair"]].drop_duplicates().shape[0]
    print(k, n, ((wk.regret * wk.wt).groupby(wk.budget).sum() / n * 1000).round(2).to_dict())
