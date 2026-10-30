"""Operating characteristics of the L15 calibration rule (swarm round 6, statistician).

The rule (QUEUE.md; daily_report.calib): B0 effect of L15 (m0 -1.5) vs REF on the selection set, threshold -3.2e-3.
  R0 (pre-registered z-rule, alone+P = B0 FE over our arms + the public connector arms, all cells):
     H_pub if B0 <= -3.2e-3 and z <= -1 (z vs -3.2e-3); H_cal if B0 >= -3.2e-3 and z >= +1; otherwise UNDECIDED
     -> two more L15 runs, then the side of the point estimate.
  R3 (round-3 amended reading, from 6 L15 runs): decide on the point estimates if paircov (ours only) and alone+P lie
     on the same side of -3.2e-3; if they disagree, H_cal (status quo).
Simulated truth:
  stylised: L15's per-pair B0 contrasts from the calibrated generative model (calculus + item/group noise, centred
     within benchmark strata) plus the pre-registered predictions: H_pub -13.1e-3, H_cal +6.7e-3;
  posterior: worlds from the 10-07 posterior's regime cells with d <= 0.75 (H_pub) or d >= 1.25 (H_cal), contrasts
     uncentred (the truth is whatever the calculus gives); 'right' = the side of the world's d (H_pub = d <= 0.85).
Connectors: aghasalim (one B0 arm: its five public arms make identical budget-0 predictions; 52 logs) and
doctawho42 hier-ship (2 logs), replicas from the round-2 tables, drawn from the same world before our runs.
usage: R swarm/round6/oc_sims/oc_calib.py N_WORLDS SEED MODE   (MODE stylised | posterior)
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
import ld_calc as lc  # noqa: E402
import ld_sim as ls  # noqa: E402

CAL = np.array([0.012, 0.008, 0.008, 0.003, 0.001, 0.0005])
THR = -0.0032
L15 = {"m0": -1.5, "tau_mu": 1.2, "tau_delta": 1.0, "tau_phi": lc.TAU_PHI}
COMP = ("pub_aghasalim_v3A", "pub_doctawho42_hiership")
N_COMP = (52, 2)
# (REF runs, L15 runs) at which the rule is read; the 10-07 order puts L15's 6th run 5th (REF 11), its 8th run 19th
# (REF 15), end of day REF 17 (one REF slot used by the composite platform check)
LOOKS = [(12, 6), (11, 6), (15, 8), (17, 8), (24, 12), (40, 20), (60, 30)]


def b0_estimates(D, n_ours):
    """alone+P B0 (all arms incl. connectors) and paircov B0 (ours only) on the selection set, with SE_h."""
    rows = oc.subset_rows(D, "sel")
    out = {}
    for lab, r in (("aP", rows), ("pc", rows[D.arm[rows] < n_ours])):
        arm, pair, sh, Y = D.arm[r], D.pair[r], D.sh[r], D.Y[r][:, [1]]
        n_tot = n_ours + len(COMP)
        beta = oc.paircov(arm, pair, sh, Y, n_tot, b0_col=0)
        cr1, cr0 = oc.cr_se(arm, pair, sh, Y, n_tot, cov=False)
        K = oc.k_direct(arm, pair, n_tot, D.w.P)
        se = max(oc.BOOT_INFLATE * oc.BOOT_PROXY * np.nan_to_num(cr0[1, 0]), np.nan_to_num(cr1[1, 0]),
                 oc.FLOOR[1] / np.sqrt(max(K[1], 1)))
        out[lab] = (float(beta[1, 0]), float(se), int(K[1]))
    return out


def call_r0(b, se):
    z = (b - THR) / se
    if b <= THR and z <= -1:
        return "pub"
    if b >= THR and z >= 1:
        return "cal"
    return "und"


def main():
    nw, seed, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    maxs = 70.0
    t0 = time.time()
    S = oc_setup.load()
    T = ls.Tables()
    rng0 = np.random.default_rng(seed * 7 + 3)
    recs = []
    cells = S["cells"]
    w = np.asarray(S["w"])
    hyp_list = ("pub", "cal")
    for iw in range(nw):
        if time.time() - t0 > maxs:
            break
        hyp = hyp_list[iw % 2]
        rng = np.random.default_rng(seed * 100000 + iw)
        if mode == "stylised":
            c = int(rng0.choice(len(w), p=w))
            eff = np.zeros((1, 6))
            eff[0, 0] = -0.0131 if hyp == "pub" else 0.0067
            W = oc.World(T, cells[c], [L15], rng, extra_sd=CAL, effects=eff, comp=COMP)
        else:
            dsel = np.array([lc.DS[j] <= 0.75 if hyp == "pub" else lc.DS[j] >= 1.25 for (i, j, k) in cells])
            ww = np.where(dsel, w, 0.0)
            c = int(rng0.choice(len(w), p=ww / ww.sum()))
            W = oc.World(T, cells[c], [L15], rng, extra_sd=CAL, center=False, comp=COMP)
        D = oc.Data(W)
        for ci, (nm, k) in enumerate(zip(COMP, N_COMP)):
            yb = W.comp_b0[nm]
            for _ in range(k):
                D.upload(1 + ci, rng, y_fn=lambda p, yb=yb: np.column_stack([np.zeros(len(p)), yb[p],
                                                                               np.zeros((len(p), 5))]))
        nref = nl = 0
        rec = {"hyp": hyp, "mode": mode, "cell": [int(x) for x in cells[c]], "truth_b0": float(W.truth[0, 1]),
               "d": float(lc.DS[cells[c][1]]), "looks": {}}
        for (r_, l_) in LOOKS:
            while nref < r_:
                D.upload(-1, rng)
                nref += 1
            while nl < l_:
                D.upload(0, rng)
                nl += 1
            e = b0_estimates(D, 1)
            rec["looks"][f"{r_}/{l_}"] = e
        recs.append(rec)
    out = HERE / "cache" / f"calib_{mode}.jsonl"
    with out.open("a") as fh:
        for r in recs:
            r["seed"] = seed
            fh.write(json.dumps(r) + "\n")
    print(f"{len(recs)} worlds ({mode}) in {time.time() - t0:.1f}s -> {out.name}")


def summarize():
    import pandas as pd
    for mode in ("stylised", "posterior"):
        f = HERE / "cache" / f"calib_{mode}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        print(f"\n== L15 calibration rule, {mode} truth: {len(recs)} worlds "
              f"({sum(r['hyp'] == 'pub' for r in recs)} H_pub, {sum(r['hyp'] == 'cal' for r in recs)} H_cal)")
        if mode == "posterior":
            for h in ("pub", "cal"):
                tb = np.array([r["truth_b0"] for r in recs if r["hyp"] == h]) * 1e3
                print(f"  true L15 B0 in {h} worlds (x1e-3): median {np.median(tb):+.1f}, p10 {np.percentile(tb, 10):+.1f},"
                      f" p90 {np.percentile(tb, 90):+.1f}; share below -3.2: {(tb < -3.2).mean():.2f}")
        looks = list(recs[0]["looks"])
        print(f"  {'REF/L15':8s} {'hyp':4s} {'K':>5s} {'R0 right':>13s} {'R0 wrong':>13s} {'R0 undec':>13s} "
              f"{'R0+fallback right':>18s} {'R3 right':>13s} {'point(aP) right':>16s} {'SE_h aP':>8s}")
        for lk in looks:
            for h in ("pub", "cal"):
                rr = [r for r in recs if r["hyp"] == h]
                n = len(rr)
                if not n:
                    continue
                right_side = (lambda b: b <= THR) if h == "pub" else (lambda b: b > THR)
                c0 = [call_r0(*r["looks"][lk]["aP"][:2]) for r in rr]
                ok0 = np.array([c == h for c in c0])
                bad0 = np.array([c not in (h, "und") for c in c0])
                und = np.array([c == "und" for c in c0])
                # fallback: two more L15 runs (and the REF runs that come with them in the plan) -> point estimate
                fb_key = {"12/6": "15/8", "11/6": "15/8", "15/8": "24/12", "17/8": "24/12", "24/12": "40/20",
                          "40/20": "60/30"}.get(lk)
                fb = []
                for r, c in zip(rr, c0):
                    if c != "und":
                        fb.append(c == h)
                    elif fb_key:
                        fb.append(right_side(r["looks"][fb_key]["aP"][0]))
                    else:
                        fb.append(right_side(r["looks"][lk]["aP"][0]))
                r3 = []
                for r in rr:
                    a, p = r["looks"][lk]["aP"][0], r["looks"][lk]["pc"][0]
                    side = "pub" if (a <= THR and p <= THR) else "cal"     # disagreement or both above -> H_cal
                    r3.append(side == h)
                pt = np.array([right_side(r["looks"][lk]["aP"][0]) for r in rr])
                K = np.median([r["looks"][lk]["aP"][2] for r in rr])
                se = np.median([r["looks"][lk]["aP"][1] for r in rr]) * 1e3

                def ps(x):
                    x = np.asarray(x, float)
                    return f"{x.mean():.2f} ({np.sqrt(max(x.mean() * (1 - x.mean()), 1 / len(x)) / len(x)):.2f})"
                print(f"  {lk:8s} {h:4s} {K:5.0f} {ps(ok0):>13s} {ps(bad0):>13s} {ps(und):>13s} {ps(fb):>18s} "
                      f"{ps(r3):>13s} {ps(pt):>16s} {se:8.1f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["summary"]:
        summarize()
    else:
        main()
