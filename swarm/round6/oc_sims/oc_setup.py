"""Design state for the operating-characteristics simulator (swarm round 6, statistician).

Builds, once, the objects the round-4 adaptive allocation (ld_policy.P1, the validated in-sim proxy of
tools/swarm_live_design.py) needs, from the SAME evidence file the 2026-10-07 plan used
(swarm/round4/design_sims/cache/live_design_now.npz: 37 of our logs + 54 connector logs, selection set only):
  - the 7-arm shortlist of the 10-07 plan (W20, L15, W20@-1.5, m-2.50_mu2.5_d0.6, m-2.25_mu2.5_d1.4,
    m-1.50_mu2.5_d0.6, m-2.00_mu2.0_d0.6), in the plan's order;
  - the posterior over regime cells and the Gaussian arm-effect components for those arms (SimPosterior input);
  - the per-pair SDs the plan assumes (swarm_live_design.sd_arms);
  - the regime cells and weights used to draw simulated worlds.
Output: cache/setup_oc.pkl. Nothing live is printed beyond the plan's own posterior summaries.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LIB = REPO / "swarm" / "round4" / "design_sims"
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(REPO / "tools"))
import ld_calc as lc  # noqa: E402
import ld_post as lp  # noqa: E402

CACHE = HERE / "cache"
SECS = ("alc", "b0", "b1", "b3")
SHORT = [(-2.0, 2.0, 1.0), (-1.5, 1.2, 1.0), (-1.5, 2.0, 1.0), (-2.5, 2.5, 0.6), (-2.25, 2.5, 1.4),
         (-1.5, 2.5, 0.6), (-2.0, 2.0, 0.6)]
TAGS = ["W20", "L15", "W15", "WLA", "WLB", "S6", "S7"]
# state at 2026-10-06 13:50 UTC (ledger): runs with feedback
RUNS_NOW = {"W20": 4, "L15": 5}
REF_RUNS_NOW = 10


def spec(t):
    return {"m0": t[0], "tau_mu": t[1], "tau_delta": t[2], "tau_phi": lc.TAU_PHI}


def build():
    t0 = time.time()
    P = lp.Posterior(LIB / "cache" / "live_design_now.npz").fit()
    names = [lc.arm_name(*t) for t in SHORT]
    cands = {n: spec(t) for n, t in zip(names, SHORT)}
    _, cells, w, means, covs = P.arm_post(cands, secs=SECS, max_cells=400)
    S_short, _ = P.summarize(names, w, means, covs, secs=SECS, ndraw=20000, seed=20261008)
    import swarm_live_design as sld
    sd = sld.sd_arms(cands, names, cells, w)
    out = {"names": names, "tags": TAGS, "cands": cands, "cells": cells, "w": w, "means": means, "covs": covs,
           "sd_arm": sd, "summary": S_short, "marg": P.marginals(), "runs_now": RUNS_NOW,
           "ref_runs_now": REF_RUNS_NOW}
    CACHE.mkdir(exist_ok=True)
    (CACHE / "setup_oc.pkl").write_bytes(pickle.dumps(out))
    print(f"setup built in {time.time() - t0:.1f}s: {len(cells)} cells; P(H_pub) {out['marg']['P(d<=0.85)']:.2f}")
    for n, tg in zip(names, TAGS):
        s = S_short[n]
        print(f"  {tg:4s} {n:20s} ALC {1e3 * s['mean']['alc']:+.2f} sd {1e3 * s['sd']['alc']:.2f}  P(<0) {s['p_neg']:.2f}")
    print("  per-pair SD assumed by the plan (x1e-3, alc b0 b1 b3):", np.round(1e3 * sd, 1).tolist())
    return out


def load():
    f = CACHE / "setup_oc.pkl"
    if not f.exists():
        return build()
    return pickle.loads(f.read_bytes())


if __name__ == "__main__":
    build()
