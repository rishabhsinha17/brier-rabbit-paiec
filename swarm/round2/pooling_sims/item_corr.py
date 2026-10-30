"""Public-data ceiling for cross-subject item pooling: for every two eligible pairs of one benchmark, on their common
items, the Brier gain of an oracle that knows both pair rates and the 2x2 joint table and predicts A's outcome from B's
outcome: Var_yB(E[y_A | y_B]) = Cov(y_A, y_B)^2 / Var(y_B) (first recorded run per item). Also the phi coefficient.
Gain per covered evaluation item is the ceiling for what a partner label can buy; multiply by coverage."""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
c = pickle.load(open(HERE / "cache" / "replay_cache.pkl", "rb"))
for b in ("real_webagents", "researchcodebench", "multi_swebench", "matharena"):
    raw = c["raw"][b]
    lab = {rp["key"]: {i: np.mean(rec["labels"]) for i, rec in rp["items"].items()} for rp in raw}
    harness = {rp["key"]: rp["subject"].get("harness", "") for rp in raw}
    gains, phis, same_h, diff_h = [], [], [], []
    for a, bb in itertools.permutations(lab, 2):
        common = sorted(set(lab[a]) & set(lab[bb]))
        if len(common) < 40:
            continue
        ya = np.array([lab[a][i] for i in common])
        yb = np.array([round(lab[bb][i]) for i in common])
        vb = yb.var()
        if vb < 1e-9 or ya.var() < 1e-9:
            continue
        cov = np.mean((ya - ya.mean()) * (yb - yb.mean()))
        g = cov ** 2 / vb
        gains.append(g)
        phis.append(cov / np.sqrt(vb * ya.var()))
        (same_h if harness[a] and harness[a] == harness[bb] else diff_h).append(g)
    gains, phis = np.array(gains), np.array(phis)
    print(f"{b:18s} ordered pairs {len(gains):5d}  oracle gain/covered item mean {gains.mean():.4f} "
          f"median {np.median(gains):.4f}  phi mean {phis.mean():.3f}  "
          f"same-harness gain {np.mean(same_h) if same_h else float('nan'):.4f} (n={len(same_h)}) "
          f"diff-harness {np.mean(diff_h) if diff_h else float('nan'):.4f} (n={len(diff_h)})")
