"""Pool every matched-mode quick check (out/q*matched*.csv; runs deduplicated) into one table: ALC difference vs base
per benchmark and arm (mean +- SE over pairs, n pairs), plus coverage at B31."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(HERE / "out" / "q*matched*.csv")))], ignore_index=True)
df = df[(df["mode"] == "matched") & (df.k == 2)].copy()
df["model"] = df["model"].str.replace(r"_x(\d)$", r"_x\1.0", regex=True)
df = df.drop_duplicates(["bench", "run", "pair", "model"])
base = df[df.model == "base"].set_index(["bench", "run", "pair"])
rows = []
for (bench, model), g in df[df.model != "base"].groupby(["bench", "model"]):
    g = g.set_index(["bench", "run", "pair"])
    idx = g.index.intersection(base.index)
    d = g.loc[idx, "alc"] - base.loc[idx, "alc"]
    rows.append({"bench": bench, "model": model, "n": len(d), "dALC": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)),
                 "d_b31": (g.loc[idx, "b31"] - base.loc[idx, "b31"]).mean(), "cov31": base.loc[idx, "cov31"].mean(),
                 "base_alc": base.loc[idx, "alc"].mean()})
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.round(5).to_string(index=False))
