"""Q7 (our pipeline): within-pair same-text twins. HierModel keys eps on (benchmark, item text), so a pair's own revealed
label on one item informs its eval item with identical text. Measures exposure in the local replay (pair-keyed
subsample to 88-226 items, persistent 50/50 split, n revealed labels = uniform subset of the acquisition pool)."""
import sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
from paiec.data import load_responses
from paiec.protocol import index_pairs, materialize
from paiec.hier import _item_key

df = load_responses()
raw = [rp for rp in index_pairs(df) if rp["bench"] != "swe_rebench"]
print("eligible pairs (binary rows, >=80 item IDs):", len(raw))
SYS = ("You are ChatGPT, a large language model trained by OpenAI.", "Date: 20")
for bench in ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]:
    rps = [rp for rp in raw if rp["bench"] == bench]
    full_share, sys_pairs = [], 0
    for rp in rps:
        keys = [_item_key(v["item"]) for v in rp["items"].values()]
        u, c = np.unique(keys, return_counts=True)
        cnt = dict(zip(u, c))
        full_share.append(np.mean([cnt[k] > 1 for k in keys]))
        if any(str(v["item"]["item_content"]).startswith(SYS) for v in rp["items"].values()):
            sys_pairs += 1
    exp = {n: [] for n in (1, 3, 7, 15, 31)}
    for seed in range(5):
        for p in materialize(rps, size_range=(88, 226), seed=seed):
            ak = [_item_key(p.items[i]["item"]) for i in p.acq]
            A = len(ak)
            from collections import Counter
            ac = Counter(ak)
            for n in exp:
                vals = []
                for e in p.eval:
                    t = ac.get(_item_key(p.items[e]["item"]), 0)
                    vals.append(1 - comb(A - t, n) / comb(A, n) if t else 0.0)
                exp[n].append(np.mean(vals))
    print(f"{bench:18s} pairs {len(rps):3d} | full pair: mean share of items with a same-text twin {np.mean(full_share):.3f} "
          f"(pairs with any twin {int(np.sum(np.array(full_share) > 0))}; with system-prompt items {sys_pairs})")
    print("    replay 88-226, 5 seeds: mean share of EVAL items whose same-text twin is among the pair's own revealed labels: "
          + ", ".join(f"B{n} {np.mean(v):.3f}" for n, v in exp.items())
          + f" | max pair at B31 {np.max(exp[31]):.3f}")
