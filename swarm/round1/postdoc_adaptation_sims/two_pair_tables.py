"""Prediction tables q[n][kT, kP] for a target pair T that shares its benchmark with one partner pair P.
Both pairs have n labels (budget n), on distinct items, unknown subjects, no item_features. Saved as .npz per model.
Usage: python two_pair_tables.py <name> '<params json>' [class]"""
import sys, json, time
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np
from paiec.hier import HierModel

OUT = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/swarm/round1/postdoc_adaptation_sims/tables")
BUD = (0, 1, 3, 7, 15, 31)
def subj(name):
    return {"normalized_name": name, "provider": "", "release_date": "", "access_date": "", "harness": "",
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
def item(i):
    return {"item_content": f"task {i}", "item_features": "", "interactors": "", "benchmark_id": "benchmark_1"}
T, P = subj("unknown-target-x"), subj("unknown-partner-y")
def lab(s, k, n, off):
    return [[[s, item(off + i)], int(i < k)] for i in range(n)]

name, params = sys.argv[1], json.loads(sys.argv[2])
cls = HierModel
if len(sys.argv) > 3:
    import importlib
    mod, c = sys.argv[3].split(":")
    cls = getattr(importlib.import_module(mod), c)
m = cls({}, params)
t0 = time.time()
tabs = {}
for n in BUD:
    q = np.zeros((n + 1, n + 1))
    for kT in range(n + 1):
        for kP in range(n + 1):
            q[kT, kP] = m.predict([T, item(10 ** 6)], lab(T, kT, n, 0) + lab(P, kP, n, 1000))
    tabs[f"n{n}"] = q
# alone tables (no partner) for reference
for n in BUD:
    tabs[f"a{n}"] = np.array([m.predict([T, item(10 ** 6)], lab(T, k, n, 0)) for k in range(n + 1)])
import os
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/{name}.npz", **tabs)
print(f"{name}: {time.time() - t0:.1f}s; q[1]={np.round(tabs['n1'], 3).tolist()}")
