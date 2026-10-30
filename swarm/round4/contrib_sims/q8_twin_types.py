"""Q8: break down MathArena within-pair twins by type; check _item_key collisions (RCB, Multi-SWE) vs true duplicates."""
import hashlib, sys
from collections import Counter
import numpy as np
import pandas as pd
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
from paiec.data import load_responses
from paiec.protocol import index_pairs
from paiec.hier import _item_key

df = load_responses()
raw = index_pairs(df)
def typ(item):
    t = str(item["item_content"])
    f = str(item["item_features"])
    if t.startswith("You are ChatGPT, a large language model") or t.startswith("Date: 20"): return "system_prompt_or_header"
    if t.startswith("You are given a Math Kangaroo problem"): return "kangaroo_generic_text"
    if any(f"competition={c}" in f for c in ("imc_2025", "imo_2025", "miklos_2025", "putnam_2025", "usamo_2025")): return "proof_criteria"
    return "other_duplicate_text (cross-competition etc.)"
tot = Counter(); n_items = 0
for rp in raw:
    if rp["bench"] != "matharena": continue
    keys = {i: _item_key(v["item"]) for i, v in rp["items"].items()}
    cnt = Counter(keys.values())
    n_items += len(keys)
    for i, v in rp["items"].items():
        if cnt[keys[i]] > 1:
            tot[typ(v["item"])] += 1
print("MathArena eligible-pair items:", n_items, "| items with a same-text twin in the same pair, by type:")
for k, v in tot.most_common():
    print(f"   {k:48s} {v:6d} ({v / n_items:.1%})")
# _item_key collisions vs full-text equality per benchmark
for bench in ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]:
    d = df[df.bench == bench].drop_duplicates("item_id")
    k = d.apply(lambda r: _item_key({"item_content": r.item_content}), axis=1)
    full = d.item_content.map(lambda t: hashlib.sha1(t.encode("utf-8", "replace")).hexdigest())
    print(f"{bench:18s} item IDs {len(d):5d} | distinct full texts {full.nunique():5d} | distinct _item_key {k.nunique():5d}"
          f" | _item_key collisions merging different texts: {full.nunique() - k.nunique()}")
    if bench == "researchcodebench" and full.nunique() != k.nunique():
        g = pd.DataFrame({"k": k, "full": full, "feat": d.item_features, "len": d.item_content.str.len()})
        bad = g.groupby("k").full.nunique(); bad = bad[bad > 1]
        print("   colliding _item_key groups:", len(bad), "| example features:",
              g[g.k.isin(bad.index)].sort_values("k")[["feat", "len"]].head(6).values.tolist())
# Multi-SWE-bench: why do 2,126 IDs map to 2,078 texts?
d = df[df.bench == "multi_swebench"].drop_duplicates("item_id")
g = d.groupby("item_content").item_features.agg(lambda s: tuple(sorted(set(s))))
dup = g[g.map(len) > 1]
print("\nMulti-SWE-bench texts with >1 item ID:", int((d.groupby('item_content').item_id.nunique() > 1).sum()),
      "| of which differ by item_features:", len(dup), "| feature sets:", Counter(dup).most_common(6))
