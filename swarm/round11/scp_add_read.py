"""Read the composite additivity check (QUEUE.md, composite additivity amendment, 2026-10-07).

usage: python swarm/round11/scp_add_read.py [results/scp_add_1008.csv] [results/hr_mix_guard.csv]
Macro = bench-equal, k-averaged mean ALC (as experiments/paired.py prints it). Contrasts are paired on (bench, k, run,
pair); the SE clusters by (bench, k, run) draw and propagates through the same macro weights.
Rule: for each arm, (arm_scp - arm) macro within +-0.3e-3 of (w20_scp - w20_m20).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ARMS = [("m150_mu20_d10", "m150_mu20_d10_scp"), ("m150_mu20_d06", "m150_mu20_d06_scp")]
REF = ("w20_m20", "w20_scp")
CHAIN = {"base": 0.1921, "w20_m20": 0.1909, "m150_mu20_d10": 0.1902, "m150_mu20_d06": 0.1902, "w20_scp": 0.1900}


def macro(df: pd.DataFrame, col: str = "alc") -> float:
    return df.groupby(["bench", "k"])[col].mean().groupby("bench").mean().mean()


def contrast(df: pd.DataFrame, a: str, b: str, col: str = "alc") -> tuple[float, float, pd.Series]:
    w = df.pivot_table(index=["bench", "k", "run", "pair"], columns="model", values=col)
    d = (w[b] - w[a]).rename("d").reset_index()
    est = macro(d, "d")
    # draw-clustered SE: weight of each pair in the macro, summed within each draw
    n = d.groupby(["bench", "k"])["d"].transform("size")
    nk = d.groupby("bench")["k"].transform("nunique")
    nb = d["bench"].nunique()
    d["w"] = 1.0 / (n * nk * nb)
    g = d.assign(wd=d["w"] * (d["d"] - est)).groupby(["bench", "k", "run"])["wd"].sum()
    se = float(np.sqrt((g ** 2).sum() * len(g) / max(len(g) - 1, 1)))
    per_bench = d.groupby(["bench", "k"])["d"].mean().groupby("bench").mean()
    return float(est), se, per_bench


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/scp_add_1008.csv"
    df = pd.read_csv(path)
    print(f"{path}: {len(df)} rows, models {sorted(df['model'].unique())}")
    for m, want in CHAIN.items():
        if m in set(df["model"]):
            got = macro(df[df.model == m])
            print(f"chain {m:18s} macro {got:.5f} (expected {want:.4f}) {'ok' if round(got, 4) == want else 'MISMATCH'}")
    ref_est, ref_se, _ = contrast(df, *REF)
    print(f"\nreference (w20_scp - w20_m20): {ref_est * 1e3:+.3f}e-3 (se {ref_se * 1e3:.3f})")
    if len(sys.argv) > 2:
        h = pd.read_csv(sys.argv[2])
        h_est, _, _ = contrast(h, *REF)
        print(f"  same contrast in {sys.argv[2]}: {h_est * 1e3:+.3f}e-3 ({'identical' if abs(h_est - ref_est) < 1e-9 else 'DIFFERS'})")
    for arm, arm_scp in ARMS:
        est, se, pb = contrast(df, arm, arm_scp)
        gap = est - ref_est
        ok = abs(gap) <= 0.3e-3
        print(f"\n{arm_scp} - {arm}: {est * 1e3:+.3f}e-3 (se {se * 1e3:.3f}); minus reference {gap * 1e3:+.3f}e-3 -> "
              f"{'PASS' if ok else 'FAIL'} (|gap| <= 0.3e-3)")
        print("  per benchmark x1e-3: " + ", ".join(f"{b} {v * 1e3:+.2f}" for b, v in pb.items()))
        for col in ("b0", "b1", "b3", "b7", "b15", "b31"):
            e, s, _ = contrast(df, arm, arm_scp, col)
            print(f"  {col:>4s} {e * 1e3:+.3f} (se {s * 1e3:.3f})", end="")
        print()


if __name__ == "__main__":
    main()
