"""Shape of the public item-difficulty distribution (eligible pairs only): histogram of per-item solve rates across
the subjects that attempted the item (first-run labels), for items attempted by >= 8 subjects; plus the crossed
Rasch-style item effects u_i from a joint MAP fit logit P = a_s + u_i (a_s free, u_i ~ N(0, 3^2)), whose quantiles
show whether the hard side has a heavy tail (many never-solved items) relative to a normal."""
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
c = pickle.load(open(HERE / "cache" / "replay_cache.pkl", "rb"))
for b in ("real_webagents", "researchcodebench", "multi_swebench", "matharena"):
    raw = c["raw"][b]
    subj = {rp["key"]: k for k, rp in enumerate(raw)}
    items = {}
    for rp in raw:
        for i, rec in rp["items"].items():
            items.setdefault(rec["item"]["item_content"], []).append((subj[rp["key"]], rec["labels"][0]))
    keep = {t: v for t, v in items.items() if len(v) >= 8}
    rates = np.array([np.mean([y for _, y in v]) for v in keep.values()])
    hist = [np.mean(rates == 0), np.mean((rates > 0) & (rates <= 0.1)), np.mean((rates > 0.1) & (rates < 0.9)),
            np.mean((rates >= 0.9) & (rates < 1)), np.mean(rates == 1)]
    # joint MAP fit (alternating 1-D Newton), items with >= 8 subjects
    S = len(raw)
    tlist = list(keep)
    a = np.zeros(S)
    u = np.zeros(len(tlist))
    obs = [(si, ti, y) for ti, t in enumerate(tlist) for si, y in keep[t]]
    si_arr = np.array([o[0] for o in obs]); ti_arr = np.array([o[1] for o in obs]); y_arr = np.array([o[2] for o in obs], float)
    for _ in range(60):
        p = 1 / (1 + np.exp(-(a[si_arr] + u[ti_arr])))
        g_a = np.bincount(si_arr, y_arr - p, S) - a / 25.0
        h_a = np.bincount(si_arr, p * (1 - p), S) + 1 / 25.0
        a += g_a / h_a
        p = 1 / (1 + np.exp(-(a[si_arr] + u[ti_arr])))
        g_u = np.bincount(ti_arr, y_arr - p, len(tlist)) - u / 9.0
        h_u = np.bincount(ti_arr, p * (1 - p), len(tlist)) + 1 / 9.0
        u += g_u / h_u
    u = u - np.median(u)
    q = np.percentile(u, [5, 10, 25, 50, 75, 90, 95])
    print(f"{b:18s} items(>=8 subj) {len(rates):4d} | rate=0 {hist[0]:.2f} (0,.1] {hist[1]:.2f} (.1,.9) {hist[2]:.2f} "
          f"[.9,1) {hist[3]:.2f} =1 {hist[4]:.2f} | u quantiles (median-centred) 5/10/25/50/75/90/95%: "
          + " ".join(f"{x:+.1f}" for x in q) + f" | sd {u.std():.2f} skew {((u - u.mean()) ** 3).mean() / u.std() ** 3:+.2f}")
