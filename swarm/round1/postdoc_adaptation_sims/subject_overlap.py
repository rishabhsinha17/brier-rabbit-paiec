"""How do subjects overlap across public benchmarks? (subject_id per benchmark, full runtime dict, canonical name)"""
import sys, json, itertools, functools
print = functools.partial(print, flush=True)
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import pandas as pd, numpy as np
from paiec.data import load_responses, SUBJECT_FIELDS, ALL_BENCHMARKS
from paiec.subjects import name_key

df = load_responses(ALL_BENCHMARKS)
pairs = df.drop_duplicates("pair")[["bench", "pair", "subject_id", *SUBJECT_FIELDS]].copy()
cnt = df.groupby("pair").item_id.nunique()
pairs = pairs[pairs.pair.map(cnt) >= 80]
pairs["dict"] = [json.dumps({f: r[f] for f in SUBJECT_FIELDS}, sort_keys=True) for _, r in pairs.iterrows()]
pairs["key"] = [name_key({f: r[f] for f in SUBJECT_FIELDS}) for _, r in pairs.iterrows()]
print("eligible pairs:", len(pairs), "benchmarks:", pairs.bench.nunique())
print("subject_id values appearing in >1 benchmark:", (pairs.groupby("subject_id").bench.nunique() > 1).sum())
print("identical runtime subject dicts in >1 benchmark:", (pairs.groupby("dict").bench.nunique() > 1).sum())
kb = pairs[pairs.key != ""].groupby("key").bench.nunique()
print("canonical names:", len(kb), "| in >=2 benchmarks:", (kb >= 2).sum(), "| >=3:", (kb >= 3).sum())
print("share of eligible pairs whose canonical name appears on another benchmark:",
      round(pairs.key.map(kb).fillna(0).ge(2).mean(), 3))
# within a hypothetical draw: one random pair per benchmark (5 benchmarks with >=10 pairs), how often is a name repeated?
rng = np.random.default_rng(0)
bs = [b for b, g in pairs.groupby("bench") if len(g) >= 10]
lists = [list(pairs[pairs.bench == b].key) for b in bs]
hits = []
for _ in range(5000):
    keys = [l[rng.integers(len(l))] for l in lists]
    keys = [k for k in keys if k]
    hits.append(len(keys) - len(set(keys)) > 0)
print(f"P(some canonical name repeats) in a random one-pair-per-benchmark draw over {len(bs)} public benchmarks: {np.mean(hits):.3f}")
# example of a subject that spans benchmarks, show its dicts
ex = kb[kb >= 3].index[:2]
for k in ex:
    print("\nname", k)
    for _, r in pairs[pairs.key == k].head(4).iterrows():
        print("  ", r.bench, r.subject_id, {f: r[f] for f in SUBJECT_FIELDS if r[f]})
