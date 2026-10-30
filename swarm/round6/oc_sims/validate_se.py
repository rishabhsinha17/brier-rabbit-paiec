"""Validate the simulator's estimators against the real decision tool (swarm round 6).

Simulated appearance logs are converted to the wide frame tools/daily_report.py works on (opaque synthetic IDs), and
daily_report.estimate() (FE2 paircov + stratified pair bootstrap + CR1 + floors) is compared with oc_core.estimate()
(fe_within on cells + CR1 port + CR0 in place of the bootstrap). Reports max |diff| of the effects and CR1, the ratio
of the real bootstrap SE to CR0, and how often the approximated SE_h differs from the real one by > 5%.
usage: R swarm/round6/oc_sims/validate_se.py [n_worlds] [reps]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oc_core as oc  # noqa: E402
import oc_setup  # noqa: E402
import ld_sim as ls  # noqa: E402
import daily_report as dr  # noqa: E402
import swarm_evalsci as ev  # noqa: E402

CAL = np.array([0.012, 0.008, 0.008, 0.003, 0.001, 0.0005])


def to_wide(D: oc.Data, names):
    n = D.n
    arm_name = {-1: ev.REF, **{a: f"sim_{nm}" for a, nm in enumerate(names)}}
    rows = {"submission": [f"r{x}" for x in D.run[:n]],
            "subject_id": [f"subject_{p}" for p in D.pair[:n]],
            "benchmark_id": [f"benchmark_{D.w.stratum[p]}" for p in D.pair[:n]]}
    w = pd.DataFrame(rows)
    for j, s in enumerate(oc.SEC):
        w[s] = D.Y[:n, j]
    w["arm"] = [arm_name[int(a)] for a in D.arm[:n]]
    w["key"] = w["subject_id"] + "|" + w["benchmark_id"]
    w["cond"] = np.where(D.sh[:n], "shared", "alone")
    w["pkey"] = w["key"] + "|" + w["cond"]
    w["holdout"] = D.w.holdout[D.pair[:n]]
    w["n_unique"] = 80
    return w


DESIGN = "small"


def main():
    global DESIGN
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    DESIGN = sys.argv[3] if len(sys.argv) > 3 else "small"
    floorsup = len(sys.argv) > 4 and sys.argv[4] == "floorsup"
    t0 = time.time()
    S = oc_setup.load()
    T = ls.Tables()
    arms = [S["cands"][n] for n in S["names"]]
    rng = np.random.default_rng(11)
    res = []
    for iw in range(nw):
        c = rng.choice(len(S["w"]), p=S["w"])
        W = oc.World(T, S["cells"][c], arms, rng, extra_sd=CAL,
                     **({"target_sd": oc.FLOOR[1:], "superpop": True} if floorsup else {}))
        D = oc.Data(W)
        plan = ([(-1, 40), (0, 30), (1, 12), (2, 8), (3, 30), (4, 6)] if DESIGN == "small" else
                [(-1, 290), (0, 90), (1, 90), (2, 80), (3, 60), (4, 60), (5, 35), (6, 30)])
        for a, k in plan:
            for _ in range(k):
                D.upload(a, rng)
        for subset in ("sel", "all"):
            rows = oc.subset_rows(D, subset)
            mine = oc.estimate(D, rows)
            w = to_wide(D, S["tags"])
            w = w[~w.holdout] if subset == "sel" else w
            kt = dr.k_table(w)
            t = dr.estimate(w, "paircov", reps, 20261006, kt, subset)
            for a, tg in enumerate(S["tags"][:7]):
                for j, s in enumerate(oc.SEC):
                    r = t[(t.arm == f"sim_{tg}") & (t.section == s)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    res.append({"world": iw, "subset": subset, "arm": tg, "sec": s, "eff_tool": r.effect,
                                "eff_mine": mine["est"][a, j], "cr1_tool": r.se_cr1, "cr1_mine": mine["cr1"][a, j],
                                "boot": r.se_boot, "cr0": mine["cr0"][a, j], "seh_tool": r.se_h,
                                "seh_mine": mine["se"][a, j], "K_tool": r.K, "K_mine": mine["K"][a],
                                "binds": r.binds})
        print(f"world {iw} done {time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(res)
    print(f"cells compared: {len(df)}")
    print(f"effect max |diff| {np.nanmax(np.abs(df.eff_tool - df.eff_mine)):.2e}; CR1 max |diff| "
          f"{np.nanmax(np.abs(df.cr1_tool - df.cr1_mine)):.2e}; K mismatches {(df.K_tool != df.K_mine).sum()}")
    df["ratio"] = df.boot / df.cr0
    print("bootstrap SE / CR0 by section (median, p10, p90):")
    for s, g in df.groupby("sec"):
        print(f"  {s:4s} {g.ratio.median():.3f} {g.ratio.quantile(.1):.3f} {g.ratio.quantile(.9):.3f}   "
              f"binding part in the tool: {g.binds.value_counts().to_dict()}")
    rel = (df.seh_mine / df.seh_tool - 1).abs()
    print(f"SE_h approx vs tool: median |rel diff| {rel.median():.3f}; share > 5%: {(rel > 0.05).mean():.3f}; "
          f"max {rel.max():.3f}")
    df.to_csv(HERE / "cache" / f"validate_se_{DESIGN}.csv", index=False)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
