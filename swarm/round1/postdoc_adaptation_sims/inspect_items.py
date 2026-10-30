"""What does the runtime item dict contain in the public data? (postdoc_adaptation, swarm round 1)"""
import sys, collections, re
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np, pandas as pd
from paiec.data import load_responses, ALL_BENCHMARKS

df = load_responses(ALL_BENCHMARKS, binary_only=True)
print("rows", len(df))
for b, g in df.groupby("bench"):
    items = g.drop_duplicates("item_id")
    feats = items["item_features"]
    print(f"\n=== {b}: rows {len(g)}, items {len(items)}, pairs {g.pair.nunique()}, subjects {g.subject_id.nunique()}")
    print("  share items with item_features:", round((feats != "").mean(), 3))
    toks = collections.Counter()
    keys = collections.defaultdict(set)
    for f in feats:
        for t in f.split(";"):
            t = t.strip()
            if t:
                k = t.split("=", 1)[0]
                keys[k].add(t)
    for k, vals in keys.items():
        ex = sorted(vals)[:4]
        print(f"  key {k!r}: {len(vals)} distinct values; e.g. {ex}")
    print("  first-token groups:", feats.map(lambda f: f.split(';')[0].strip() if f else '').nunique())
    print("  example features:", list(feats.head(3)))
    inter = g["interactors"]
    print("  interactors distinct:", inter.nunique(), "e.g.", list(inter.drop_duplicates().head(3)))
    L = items["item_content"].str.len()
    print("  item_content length: median", int(L.median()), "p10", int(L.quantile(.1)), "p90", int(L.quantile(.9)))
    print("  example content head:", repr(items["item_content"].iloc[0][:160]))
    sub = g.drop_duplicates("pair")
    for f in ["harness", "reasoning_effort", "provider", "normalized_name"]:
        print(f"  subject field {f}: {sub[f].nunique()} distinct; e.g. {list(sub[f].drop_duplicates().head(4))}")
