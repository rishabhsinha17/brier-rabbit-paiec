"""Headroom for stacking the hierarchical model with simple predictors (results/paired_mix.csv, LOBO, k=1,2).
For each budget: base vs the best simple predictor (smoothed_k2, pooled_kb8_kp4, empirical_mean) chosen
(a) per benchmark with hindsight (an oracle for within-draw weights), (b) per pair with hindsight (unattainable)."""
import numpy as np, pandas as pd
d = pd.read_csv((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/results/paired_mix.csv"))
B = ["b0", "b1", "b3", "b7", "b15", "b31"]; W = dict(zip(B, [.1, .2, .2, .2, .2, .1]))
simple = ["smoothed_k2", "pooled_kb8_kp4", "empirical_mean"]
idx = ["bench", "k", "run", "pair"]
w = d.pivot_table(index=idx, columns="model", values=B)
rows = []
for b in B:
    base = w[(b, "base")]
    best_simple = pd.concat([w[(b, m)] for m in simple], axis=1).min(axis=1)
    per_bench = []
    for (bench, k), g in w.groupby(level=[0, 1]):
        cands = {m: g[(b, m)].mean() for m in ["base"] + simple}
        per_bench.append(min(cands.values()) - cands["base"])
    rows.append({"budget": b, "base": base.mean(), "best simple (mean)": min(w[(b, m)].mean() for m in simple),
                 "per-bench oracle gain": np.mean(per_bench), "per-pair oracle gain": np.minimum(base, best_simple).mean() - base.mean()})
t = pd.DataFrame(rows).set_index("budget")
print(t.round(4).to_string())
print("ALC-weighted: per-bench oracle gain", round(sum(W[b] * t.loc[b, "per-bench oracle gain"] for b in B), 4),
      "| per-pair oracle gain", round(sum(W[b] * t.loc[b, "per-pair oracle gain"] for b in B), 4))
print("per-bench oracle gain by budget excluding matharena:")
for b in B:
    gains = []
    for (bench, k), g in w.groupby(level=[0, 1]):
        if bench == "matharena":
            continue
        cands = {m: g[(b, m)].mean() for m in ["base"] + simple}
        gains.append(min(cands.values()) - cands["base"])
    print(f"  {b}: {np.mean(gains):+.4f}")
