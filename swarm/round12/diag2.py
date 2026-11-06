"""Exploratory: where E and E2 lose on the original folds. Pairs by whether the fold's reference prior knows the name and by how many
benchmarks the full data have for it."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "swarm/round12")
import full_prior
rows = pd.read_pickle("swarm/round12/pair_rates.pkl")
rows["pair"] = rows["bench"] + "|" + rows["subject_id"].astype(str)
keymap = rows.set_index("pair")["key"].to_dict()
d = pd.concat(map(pd.read_csv, glob.glob("swarm/round12/results/L1E/*.csv")), ignore_index=True)
d["key"] = d["pair"].map(keymap).fillna("")
info = []
for bench, g in d.groupby("bench"):
    ref = json.load(open(f"swarm/round12/cache/ref_prior_{bench}.json"))
    full = full_prior.fit(rows, "F2", exclude=bench)
    for key in g["key"].unique():
        v = full["names"].get(key)
        info.append({"bench": bench, "key": key, "in_ref": key in ref["names"], "k_full": round(1 / v[1] - 1) if v else 0, "theta_full": v[0] if v else np.nan,
                     "theta_ref": ref["names"].get(key, [np.nan])[0]})
info = pd.DataFrame(info)
d = d.merge(info, on=["bench", "key"])
cols = ["alc", "b0", "b1", "b3", "b7"]
w = d.pivot_table(index=["bench", "k", "run", "pair", "in_ref", "k_full"], columns="model", values=cols)
w = w.reset_index()
w["kbin"] = pd.cut(w["k_full"], [-1, 0, 1, 2, 4, 9, 99], labels=["0", "1", "2", "3-4", "5-9", "10+"])
for arm in ("E", "E2"):
    out = []
    for (in_ref, kb), g in w.groupby(["in_ref", "kbin"], observed=True):
        line = {"in fold ref prior": in_ref, "k in full": kb, "n": len(g)}
        for c in cols:
            line[c] = round(float((g[(c, arm)] - g[(c, "w20_SCP")]).mean() * 1e3), 2)
        out.append(line)
    print(arm, "- w20_SCP on L1, x1e-3\n" + pd.DataFrame(out).to_string(index=False))
