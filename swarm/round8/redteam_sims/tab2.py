"""Level error of every model at late budgets (incl. the empirical-mean and smoothed baselines), by pair type."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose import load, cell_table, pair_type
OUT = Path(__file__).resolve().parent
df = load(models=("w20_SCP", "SCP_combo", "base", "smoothed_k2", "empirical_mean"))
c = cell_table(df)
c["ptype"] = c.r.map(pair_type)
c["bias"] = c.pbar - c.r
c.to_parquet(OUT / "cells_all.parquet", index=False)
t = c[c.budget.isin([7, 15, 31])].groupby(["budget", "ptype", "model"]).agg(lvl2=("lvl2", "mean"), bias=("bias", "mean"),
                                                                           item=("item", "mean")).reset_index()
t[["lvl2", "bias", "item"]] *= 1000
pd.set_option("display.width", 250)
print(t.pivot_table(index=["budget", "ptype"], columns="model", values="bias").round(1).to_string())
print()
print(t.pivot_table(index=["budget", "ptype"], columns="model", values="lvl2").round(2).to_string())
print()
print((c.groupby(["budget", "model"]).lvl2.mean().unstack() * 1000).round(2).to_string())
