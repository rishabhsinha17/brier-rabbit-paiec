"""Exploratory: which fold set resembles the live pool? Global Brier-by-budget shape of live runs vs the replay sets (arm-level only)."""
import csv, glob, json
from pathlib import Path
import numpy as np, pandas as pd
fb = Path("submissions/feedback")
rows = list(csv.DictReader(open("submissions/ledger.csv")))
names = ["alc", "b0", "b1", "b3", "b7", "b15", "b31"]
def live(arms):
    out = []
    for r in rows:
        if r["arm"] in arms:
            j = fb / f"{r['id']}.json"
            if j.exists():
                d = json.loads(j.read_text())
                for p in d.get("p") or []:
                    out.append(p[2:9])
    return pd.DataFrame(out, columns=names).astype(float)
t = {"live W20 (all pair appearances)": live({"v5_w20_m20"}).mean(), "live composite W20 (4 runs)": live({"v6_combo_w20", "v6_combo_w20_c1"}).mean()}
for which in ("L1E", "L2b"):
    fs = [f for f in glob.glob(f"swarm/round12/results/{which}/*.csv") if Path(f).stat().st_size > 5]
    d = pd.concat(map(pd.read_csv, fs), ignore_index=True)
    d = d[d.model == "w20_SCP"]
    t[f"replay {which} reference, pair mean"] = d[names].mean()
    t[f"replay {which} reference, benchmark-equal"] = d.groupby("bench")[names].mean().mean()
print((pd.DataFrame(t).T * 1e3).round(1).to_string())
