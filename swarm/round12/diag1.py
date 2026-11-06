"""Exploratory: why F2 helps on the wide folds and hurts on the original folds. Splits each pair by how the prior sees its name."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "swarm/round12")
import full_prior
from paiec.data import SUBJECT_FIELDS
rows = pd.read_pickle("swarm/round12/pair_rates.pkl")
rows["pair"] = rows["bench"] + "|" + rows["subject_id"].astype(str)
keymap = rows.set_index("pair")["key"].to_dict()
shipped = set(json.load(open("submission_src/v6_combo_w20_c1/subject_prior.json"))["names"])
def load(which):
    fs = [f for f in glob.glob(f"swarm/round12/results/{which}/*.csv") if Path(f).stat().st_size > 5]
    return pd.concat(map(pd.read_csv, fs), ignore_index=True)
for which in ("L1", "L2"):
    d = load(which)
    d["key"] = d["pair"].map(keymap).fillna("")
    full_names = set(full_prior.fit(rows, "F2")["names"])
    d["cls"] = np.where(d["key"] == "", "unnamed", np.where(d["key"].isin(shipped), "in shipped prior", np.where(d["key"].isin(full_names), "new name", "name not in any prior")))
    cols = ["alc", "b0", "b1", "b3", "b7"]
    w = d.pivot_table(index=["bench", "k", "run", "pair", "cls"], columns="model", values=cols)
    print(f"\n{which}: arm - w20_SCP by class, x1e-3 (pair-score mean; n = pair-scores; benches = distinct benchmarks)")
    for arm in ("F0", "F2"):
        rowsout = []
        for cls, g in w.groupby(level="cls"):
            line = {"class": cls, "n": len(g), "benches": g.index.get_level_values("bench").nunique()}
            for c in cols:
                line[c] = round(float((g[(c, arm)] - g[(c, "w20_SCP")]).mean() * 1e3), 2)
            line["ref_b0"] = round(float(g[("b0", "w20_SCP")].mean()), 3)
            rowsout.append(line)
        print(arm); print(pd.DataFrame(rowsout).to_string(index=False))
    if which == "L2":
        per = w.groupby(level=["bench"]).mean()
        t = pd.DataFrame({"F2-ref alc": (per[("alc", "F2")] - per[("alc", "w20_SCP")]) * 1e3, "F2-ref b0": (per[("b0", "F2")] - per[("b0", "w20_SCP")]) * 1e3, "ref alc": per[("alc", "w20_SCP")], "ref b0": per[("b0", "w20_SCP")]})
        cls = d.groupby("bench")["cls"].agg(lambda s: s.value_counts().index[0])
        t["main class"] = cls
        print(t.sort_values("F2-ref alc").round(3).to_string())
