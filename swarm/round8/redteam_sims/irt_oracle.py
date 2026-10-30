"""Item-difficulty (IRT) oracle on the saved LOBO predictions (results/cal_preds_bench.parquet; public data only).

Per benchmark, a Rasch fit logit P(y=1) = a_pair + d_item on every distinct (pair, item, recorded run) label in the
file (the file's evaluation rows cover most items of every drawn pair across the 18 draws). For each pair P the item
effects are re-solved without P's labels (other pairs' a held at the full fit: leave-pair-out d). Then per pair-draw:
  IRT oracle      sigmoid(a* + d_i^{-P}) with a* fitted to the pair-draw's own evaluation rows (in-sample level)
  IRT-level-only  the pair-rate oracle r(1-r) (from decompose.py), for reference
  model + oracle item spread: sigmoid(logit(pbar_model) + (d_i^{-P} - mean d)) ... (the model's level, oracle items)
Output: irt_cells.parquet (bench, k, run, pair, n, brier_irt, brier_mlvl_irt) and a printed summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decompose import load  # noqa: E402

S_D = 3.0     # ridge SD of item effects
S_A = 5.0     # ridge SD of pair levels


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def rasch(pi, ii, y, n_p, n_i, iters=200):
    a = np.zeros(n_p)
    d = np.zeros(n_i)
    for _ in range(iters):
        p = sig(a[pi] + d[ii])
        w = p * (1 - p)
        ga = np.bincount(pi, y - p, n_p) - a / S_A ** 2
        ha = np.bincount(pi, w, n_p) + 1 / S_A ** 2
        a = a + ga / ha
        p = sig(a[pi] + d[ii])
        w = p * (1 - p)
        gd = np.bincount(ii, y - p, n_i) - d / S_D ** 2
        hd = np.bincount(ii, w, n_i) + 1 / S_D ** 2
        step = gd / hd
        d = d + step
        if np.max(np.abs(step)) < 1e-7:
            break
    return a, d


def solve_level(dv, y, iters=50):
    a = 0.0
    for _ in range(iters):
        p = sig(a + dv)
        g = np.sum(y - p) - a / 100.0
        h = np.sum(p * (1 - p)) + 1 / 100.0
        a += g / h
        if abs(g / h) < 1e-9:
            break
    return a


def main():
    df = load(models=("w20_SCP",))
    out = []
    for bench, g in df.groupby("bench"):
        b0 = g[g.budget == 0]
        lab = b0.drop_duplicates(["pair", "item", "copy"])[["pair", "item", "copy", "label"]]
        pairs = np.array(sorted(lab.pair.unique()))
        items = np.array(sorted(lab.item.unique()))
        pidx = {p: i for i, p in enumerate(pairs)}
        iidx = {it: i for i, it in enumerate(items)}
        pi = lab.pair.map(pidx).values
        ii = lab.item.map(iidx).values
        y = lab.label.values.astype(float)
        a, d = rasch(pi, ii, y, len(pairs), len(items))
        print(f"{bench}: {len(pairs)} pairs, {len(items)} items, {len(y)} labels; SD(d) {d.std():.2f}, "
              f"SD(a) {a.std():.2f}", flush=True)
        # leave-pair-out item effects: other pairs' a fixed, 1-D Newton per item without pair P's labels
        dlpo = {}
        base_eta = a[pi]
        for P, pP in pidx.items():
            keep = pi != pP
            items_P = np.unique(ii[~keep])
            sel = keep & np.isin(ii, items_P)
            dd = d.copy()
            for _ in range(30):
                p = sig(base_eta[sel] + dd[ii[sel]])
                gd = np.bincount(ii[sel], y[sel] - p, len(items)) - dd / S_D ** 2
                hd = np.bincount(ii[sel], p * (1 - p), len(items)) + 1 / S_D ** 2
                st = gd / hd
                st[np.setdiff1d(np.arange(len(items)), items_P)] = 0.0
                dd += st
                if np.max(np.abs(st)) < 1e-7:
                    break
            dlpo[P] = dd
        for (k, run, pair, budget), h in g.groupby(["k", "run", "pair", "budget"], observed=True):
            if budget not in (0, 31):
                continue
            dv = dlpo[pair][h.item.map(iidx).values]
            yy = h.label.values.astype(float)
            if budget == 0:
                astar = solve_level(dv, yy)
                b_irt = float(np.mean((sig(astar + dv) - yy) ** 2))
                out.append({"bench": bench, "k": k, "run": run, "pair": pair, "n": len(yy), "brier_irt": b_irt,
                            "sd_d_eval": float(dv.std())})
            # model level + oracle item spread: match the model's mean prediction at this budget
            pm = h.pred.values
            lv = solve_level(dv, np.full_like(pm, pm.mean()))   # level whose mean prediction equals the model's
            b_mix = float(np.mean((sig(lv + dv) - yy) ** 2))
            out[-1][f"brier_mlvl_irt_b{budget}"] = b_mix
    res = pd.DataFrame(out)
    res.to_parquet(HERE / "irt_cells.parquet", index=False)
    print(res.groupby("bench")[["brier_irt", "brier_mlvl_irt_b0", "brier_mlvl_irt_b31", "sd_d_eval"]].mean().round(4))


if __name__ == "__main__":
    main()
