"""Post-hoc prediction transforms on the saved LOBO predictions (results/cal_preds_bench.parquet, W20+SCP rows).

A transform may use only (prediction, budget): valid for levers that a submission could apply to its own output.
Effect = macro (bench-equal, k-averaged mean pair ALC) of transformed minus original, x1e-3, with a draw-clustered
bootstrap SE (runs resampled within benchmark; the composite's draws are shared by every k cell of a run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decompose import load  # noqa: E402

W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
LG = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))  # noqa: E731
SG = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def transforms():
    t = {}
    for s in (0.005, 0.01, 0.02):
        t[f"slip{s}"] = lambda p, b, s=s: (1 - s) * p
    for f in (0.01, 0.02, 0.03):
        t[f"floor{f}"] = lambda p, b, f=f: np.maximum(p, f)
    for a, m in ((0.05, 7), (0.10, 7), (0.10, 15), (0.15, 15)):
        t[f"ext{a}_n{m}"] = lambda p, b, a=a, m=m: np.where(b >= m, SG((1 + a * np.log2(b + 1) / 5) * LG(p)), p)
    return t


def pair_alc(df, col):
    e = (df[col] - df.label) ** 2
    g = df.assign(e2=e).groupby(["bench", "k", "run", "pair", "budget"], observed=True).e2.mean().reset_index()
    g["w"] = g.budget.map(W)
    a = (g.e2 * g.w).groupby([g.bench, g.k, g.run, g.pair]).sum().reset_index(name="alc")
    pb = g.pivot_table(index=["bench", "k", "run", "pair"], columns="budget", values="e2").reset_index()
    return a.merge(pb, on=["bench", "k", "run", "pair"])


def macro(x, col):
    return x.groupby(["bench", "k"])[col].mean().groupby("bench").mean().mean()


def boot_se(d, col, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    groups = {b: g for b, g in d.groupby("bench")}
    for _ in range(reps):
        vals = []
        for b, g in groups.items():
            runs = g.run.unique()
            pick = rng.choice(runs, len(runs))
            gg = pd.concat([g[g.run == r] for r in pick])
            vals.append(gg.groupby("k")[col].mean().mean())
        out.append(np.mean(vals))
    return float(np.std(out))


if __name__ == "__main__":
    df = load(models=("w20_SCP",))
    base = pair_alc(df, "pred")
    for name, f in transforms().items():
        df["q"] = f(df.pred.values, df.budget.values)
        x = pair_alc(df, "q")
        d = base[["bench", "k", "run", "pair"]].copy()
        d["alc"] = x.alc.values - base.alc.values
        for b in (0, 1, 3, 7, 15, 31):
            d[f"b{b}"] = x[b].values - base[b].values
        per_b = {b: round(1e3 * macro(d, f"b{b}"), 3) for b in (1, 3, 7, 15, 31)}
        per_bench = (1e3 * d.groupby(["bench", "k"]).alc.mean().groupby("bench").mean()).round(3).to_dict()
        print(f"{name:12s} ALC {1e3 * macro(d, 'alc'):+.3f} (SE {1e3 * boot_se(d, 'alc'):.3f}) per budget {per_b} "
              f"per bench {per_bench}", flush=True)
