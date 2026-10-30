"""Is the item_features group channel under-used by the hierarchical model? A deliberately simple own-label
group smoother (no subject prior, no partners) on the same LOBO draws as results/paired_mix.csv (k=1, runs 0-19):
  p_pair = (k + c0 p0) / (n + c0);  p = (k_g + c p_pair) / (n_g + c)   (k_g, n_g: the pair's labels in the item's group)
Compared per budget with the 'base' (cur_final) rows of results/paired_mix.csv (identical labels: same draws, same seed)."""
import sys
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import json
import numpy as np, pandas as pd
from paiec.data import load_responses, BINARY_BENCHMARKS
from paiec.evaluate import FoldData
from paiec.protocol import run_protocol, BUDGETS, pair_alc

class GroupSmoothed:
    def __init__(self, p0=0.4, c0=2.0, c=2.0):
        self.p0, self.c0, self.c = p0, c0, c
        self._ck, self._counts = None, None
    def _tally(self, labeled):
        ck = (id(labeled), len(labeled or []))
        if ck != self._ck:
            pair, grp = {}, {}
            for (s, it), y in labeled or []:
                key = (s["normalized_name"], s["subject_features_extra"], s["harness"], s["reasoning_effort"], it["benchmark_id"])
                k, n = pair.get(key, (0, 0)); pair[key] = (k + y, n + 1)
                fi = it.get("item_features", ""); g = fi.split(";")[0] if fi else ""
                k, n = grp.get((key, g), (0, 0)); grp[(key, g)] = (k + y, n + 1)
            self._ck, self._counts = ck, (pair, grp)
        return self._counts
    def predict(self, input, labeled=None):
        subject, item = input
        pair, grp = self._tally(labeled)
        key = (subject["normalized_name"], subject["subject_features_extra"], subject["harness"], subject["reasoning_effort"], item["benchmark_id"])
        f = item.get("item_features", ""); g = f.split(";")[0] if f else ""
        k, n = pair.get(key, (0, 0)); kg, ng = grp.get((key, g), (0, 0))
        pp = (k + self.c0 * self.p0) / (n + self.c0)
        return (kg + self.c * pp) / (ng + self.c) if g else pp

df = load_responses()
folds = FoldData(df, BINARY_BENCHMARKS)
ref = pd.read_csv("results/paired_mix.csv")
ref = ref[(ref.k == 1) & ref.model.isin(["base", "smoothed_k2"])]
rows = []
for bench in ("matharena", "multi_swebench", "real_webagents", "researchcodebench"):
    for r, pairs in folds.draws(bench, 1, 20, 0):
        for name, m in {"grp_c1": GroupSmoothed(c=1.0), "grp_c3": GroupSmoothed(c=3.0), "pair_only": GroupSmoothed(c=1e9)}.items():
            res = run_protocol(pairs, m.predict, seed=r)
            for key, pr in res.items():
                rows.append({"bench": bench, "run": r, "pair": key, "model": name, "alc": pair_alc(pr),
                             **{f"b{b}": pr[b]["brier"] for b in BUDGETS}})
d = pd.concat([pd.DataFrame(rows), ref[["bench", "run", "pair", "model", "alc", *[f"b{b}" for b in BUDGETS]]]])
cols = ["b1", "b3", "b7", "b15", "b31", "alc"]
t = d.groupby(["bench", "model"])[cols].mean().round(4)
print(t.to_string())
