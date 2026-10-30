"""Posterior-predictive check of the generative model against the live matched SDs (swarm round 6).

Builds worlds from posterior regime cells, simulates the CURRENT live state (REF 10 runs, W20 4, L15 5) and compares
the simulated matched per-pair SDs (any condition and alone/alone, W20 and L15 vs REF) and the within-arm replicate
summaries with the same statistics on our logs (live_targets.json). Also prints the calculus-only per-pair SDs of
the seven shortlist arms. usage: R swarm/round6/oc_sims/calib_check.py [n_worlds] [extra_sd_mode]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oc_core as oc  # noqa: E402
import oc_setup  # noqa: E402
import ld_sim as ls  # noqa: E402

EXTRA = {"none": None,
         "r4": np.array([0.0, 0.0, 0.015, 0.006, 0.0, 0.0]),          # round-4 style (B1/B3 only)
         "cal": np.array([0.012, 0.008, 0.008, 0.003, 0.001, 0.0005])}  # per-(arm, pair) item/group noise


def matched_sd(D, a, alone_only=False):
    n = D.n
    arm, pair, sh, Y = D.arm[:n], D.pair[:n], D.sh[:n], D.Y[:n]
    out = {}
    for who in (a, -1):
        m = arm == who
        if alone_only:
            m &= ~sh
        p = pair[m]
        up, inv = np.unique(p, return_inverse=True)
        s = np.zeros((len(up), 7))
        np.add.at(s, inv, Y[m])
        s /= np.bincount(inv)[:, None]
        out[who] = dict(zip(up.tolist(), s))
    common = sorted(set(out[a]) & set(out[-1]))
    if len(common) < 3:
        return None, len(common)
    d = np.array([out[a][k] - out[-1][k] for k in common])
    return d.std(0, ddof=1), len(common)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    mode = sys.argv[2] if len(sys.argv) > 2 else "none"
    t0 = time.time()
    S = oc_setup.load()
    T = ls.Tables()
    arms = [S["cands"][n] for n in S["names"]]
    rng = np.random.default_rng(7)
    raw, cen, any_w, any_l, al_w, al_l = [], [], [], [], [], []
    for iw in range(nw):
        c = rng.choice(len(S["w"]), p=S["w"])
        W = oc.World(T, S["cells"][c], arms, rng, extra_sd=EXTRA[mode])
        raw.append(W.raw_budget_sd)
        cen.append(W.C.std(1, ddof=1))
        D = oc.Data(W)
        for _ in range(10):
            D.upload(-1, rng)
        for _ in range(4):
            D.upload(0, rng)
        for _ in range(5):
            D.upload(1, rng)
        for lst, a, al in ((any_w, 0, False), (any_l, 1, False), (al_w, 0, True), (al_l, 1, True)):
            sd, k = matched_sd(D, a, al)
            if sd is not None:
                lst.append(sd)
    live = json.loads((HERE / "live_targets.json").read_text())
    f = lambda v: " ".join(f"{1e3 * x:5.1f}" for x in v)  # noqa: E731
    print(f"{nw} worlds, extra_sd mode {mode}, {time.time() - t0:.1f}s")
    print("sections:", " ".join(f"{s:>5s}" for s in oc.SEC))
    cen = np.sqrt(np.mean(np.square(cen), 0))
    for a, tg in enumerate(S["tags"]):
        print(f"  centred alone contrast SD {tg:4s}: {f(cen[a])}")
    for nm, lst, key, arm in (("W20 any", any_w, "matched_sd_by_arm_any", "v5_w20_m20"),
                              ("L15 any", any_l, "matched_sd_by_arm_any", "v3b_noM_n_m15"),
                              ("W20 alone", al_w, "matched_sd_by_arm_alone", "v5_w20_m20"),
                              ("L15 alone", al_l, "matched_sd_by_arm_alone", "v3b_noM_n_m15")):
        sim = np.sqrt(np.mean(np.square(lst), 0))
        lv = live[key][arm]
        print(f"  {nm:10s} sim {f(sim)}   | live {f([lv[s] for s in oc.SEC])} (K {lv['K']})")
        lo, hi = np.percentile(np.array(lst), [10, 90], axis=0)
        print(f"  {'':10s} p10 {f(lo)}   | p90 {f(hi)}")


if __name__ == "__main__":
    main()
