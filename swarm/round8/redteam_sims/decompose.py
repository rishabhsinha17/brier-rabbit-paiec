"""Round 8 red team: where does W20+SCP's local Brier regret sit? Uses ONLY results/cal_preds_bench.parquet
(saved LOBO evaluation predictions with labels; public benchmarks, bench-keyed subsample, 18 draws x k 1,2).

Per (model, bench, k, run, pair, budget) cell:
  brier       mean over scored rows (all recorded runs of every evaluation item; equals the platform pair Brier)
  r           the pair-draw's evaluation success rate (the pair-rate oracle predicts r for every row: Brier r(1-r))
  lvl2        (mean pred - r)^2          level error of the pair-draw
  item        var(p) - 2 cov(p, y)       item-level part (negative = the model's within-pair spread is informative)
  identity:   brier = r(1-r) + lvl2 + item (exact)
Oracles (all in-sample, i.e. optimistic bounds):
  pair-rate   r(1-r)
  IRT         sigmoid(a_pair + d_item): d_item from a per-benchmark Rasch fit to every OTHER pair's labels in the file
              (leave-pair-out), a_pair fitted to the pair-draw's own evaluation rows
  recal       per-budget logistic recalibration a_b logit(p) + c_b (pooled over benchmarks; also per benchmark)
Usage: nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round8/redteam_sims/decompose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}


def load(models=("w20_SCP", "SCP_combo", "base")) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "results" / "cal_preds_bench.parquet")
    df = df[df.model.isin(models)].copy()
    for c in ("model", "bench", "pair", "item"):
        df[c] = df[c].astype(str)
    return df


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(py=df.pred * df.label, p2=df.pred ** 2, e2=(df.pred - df.label) ** 2)
    a = g.groupby(["model", "bench", "k", "run", "pair", "budget"], observed=True).agg(
        n=("label", "size"), brier=("e2", "mean"), pbar=("pred", "mean"), r=("label", "mean"),
        p2=("p2", "mean"), py=("py", "mean")).reset_index()
    a["var_p"] = a.p2 - a.pbar ** 2
    a["cov"] = a.py - a.pbar * a.r
    a["unc"] = a.r * (1 - a.r)
    a["lvl2"] = (a.pbar - a.r) ** 2
    a["item"] = a.var_p - 2 * a["cov"]
    chk = (a.unc + a.lvl2 + a.item - a.brier).abs().max()
    assert chk < 1e-9, chk
    return a.drop(columns=["p2", "py"])


def pair_type(r: float) -> str:
    if r < 0.10:
        return "a_near0(<.10)"
    if r < 0.40:
        return "b_low(.10-.40)"
    if r < 0.70:
        return "c_mid(.40-.70)"
    return "d_easy(>=.70)"


if __name__ == "__main__":
    df = load()
    cells = cell_table(df)
    cells["ptype"] = cells.r.map(pair_type)
    cells.to_parquet(OUT / "cells.parquet", index=False)
    print(cells.groupby(["model", "budget"])[["brier", "unc", "lvl2", "item"]].mean().round(4).to_string())
    sys.stdout.flush()
