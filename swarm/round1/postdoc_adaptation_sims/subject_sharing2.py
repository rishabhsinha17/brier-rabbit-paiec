"""Cross-benchmark same-name residual correlation, split by whether the name's ability came from the name table
(seen in the training benchmarks) or from the attribute ridge only (unseen name). Leave-two-out, public data."""
import sys, itertools
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np, pandas as pd
from paiec.data import load_responses, SUBJECT_FIELDS
from paiec.hier import fit_subject_prior
from paiec.subjects import name_key, predict_attr

df = load_responses()
B = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
pairs = df.groupby("pair").agg(bench=("bench", "first"), k=("label", "sum"), n=("label", "size"), nit=("item_id", "nunique"))
first = df.drop_duplicates("pair").set_index("pair")
pairs = pairs[pairs.nit >= 80]
pairs["logit"] = np.log((pairs.k + 0.5) / (pairs.n - pairs.k + 0.5))
pairs["key"] = [name_key({f: first.at[p, f] for f in SUBJECT_FIELDS}) for p in pairs.index]
res = {"known": [], "unknown": []}
for b1, b2 in itertools.combinations(B, 2):
    prior = fit_subject_prior(df[~df.bench.isin([b1, b2])])
    sub = pairs[pairs.bench.isin([b1, b2]) & (pairs.key != "")].copy()
    th, known = [], []
    for p in sub.index:
        subj = {f: first.at[p, f] for f in SUBJECT_FIELDS}
        k = name_key(subj)
        if k in prior["names"]:
            th.append(prior["names"][k][0]); known.append(True)
        else:
            th.append(predict_attr(prior["attr"]["coef"], subj)); known.append(False)
    sub["theta"], sub["known"] = th, known
    sub["res"] = sub.logit - sub.groupby("bench").logit.transform("mean") - sub.theta
    sub["res"] -= sub.groupby("bench").res.transform("mean")
    g = sub.groupby(["key", "bench"]).agg(res=("res", "mean"), known=("known", "first")).reset_index()
    w = g.pivot(index="key", columns="bench", values="res").dropna()
    kn = g.groupby("key").known.first()
    for nm, row in w.iterrows():
        res["known" if kn[nm] else "unknown"].append((row[b1], row[b2]))
for k, v in res.items():
    a = np.array(v)
    if len(a) > 3:
        r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
        print(f"{k:>8} names: n={len(a):3d}  corr {r:+.3f}  (z-SE {1 / np.sqrt(len(a) - 3):.2f}); residual sd {a.std():.2f}")
