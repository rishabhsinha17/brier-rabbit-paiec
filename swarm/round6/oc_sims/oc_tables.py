"""Markdown tables for swarm/round6/operating_characteristics.md from the simulation records (swarm round 6).

usage: R swarm/round6/oc_sims/oc_tables.py [boot_proxy]
Reads every cache/oc_*.jsonl (direct chunks and the queue job, de-duplicated by world seed). Rates come with binomial
Monte Carlo SEs; zero counts get the exact one-sided 95% upper bound (1 - 0.05^(1/n)).
Counterfactual SE rules re-evaluate items (b) and (e) of the final checklist from the stored full-data effects and
sandwich SEs (nothing is re-simulated; screens and allocation are as run):
  pre   SE_h = max(1.2 x boot, CR1, floor/sqrt(K))   (as pre-registered; boot = BOOT_PROXY x CR0)
  K60   the floor term is dropped for arms with K_full >= 60
  boot  for K_full >= 60, SE = 1.2 x the cell-based stratified pair bootstrap only (CR1 on raw sightings and the
        floor dropped); below 60 as pre-registered
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FLOOR = np.array([0.0147, 0.055, 0.0487, 0.0132, 0.0076, 0.0043, 0.0039])
TAGS = ["W20", "L15", "W20@-1.5", "m-2.50_mu2.5_d0.6", "m-2.25_mu2.5_d1.4", "m-1.50_mu2.5_d0.6", "m-2.00_mu2.0_d0.6"]


def load(exact_only=True):
    recs = []
    for f in sorted(glob.glob(str(HERE / "cache" / "oc_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if exact_only and r.get("final_se") != "exact":
                    continue
                recs.append(r)
    df = pd.DataFrame(recs).drop_duplicates(subset=["scenario", "policy", "model", "strata", "wseed"])
    return df


def rate(x):
    x = np.asarray(x, bool)
    n = len(x)
    k = int(x.sum())
    if n == 0:
        return "-"
    if k == 0:
        return f"0/{n} (<= {1 - 0.05 ** (1 / n):.3f})"
    p = k / n
    return f"{p:.3f} ± {math.sqrt(p * (1 - p) / n):.3f}"


def ship_rule(r, rule, proxy, nmode="queue"):
    """Shipped arm under SE rule 'pre' | 'K60' | 'boot' (re-evaluating (b) and (e) only); nmode 'queue' counts every
    eligible arm as a finalist (QUEUE round-4 item 7), 'tool' only those with ALC < 0 and z <= -1 (daily_report)."""
    if not isinstance(r.get("full_cr1"), list):
        return r["ship"] if rule == "pre" else None
    elig = r["eligible"]
    zz = {}
    for a in elig:
        est = np.array(r["full_est"][a], float)
        cr1 = np.nan_to_num(np.array(r["full_cr1"][a], float))
        cr0 = np.nan_to_num(np.array(r["full_cr0"][a], float))
        K = r["K_full"][a]
        boot = (np.nan_to_num(np.array(r["full_boot"][a], float)) if isinstance(r.get("full_boot"), list)
                else proxy * cr0)
        se = 1.2 * boot if rule == "boot" else np.fmax(1.2 * boot, cr1)
        if rule == "pre" or (rule in ("K60", "boot") and K < 60):
            se = np.fmax(se, FLOOR / math.sqrt(max(K, 1)))
        zz[a] = est / se
    if nmode == "tool":
        n = max(1, sum(1 for a in elig if r["full_est"][a][0] < 0 and zz[a][0] <= -1.0))
    else:
        n = len(elig)
    thr = -1.64 if n <= 1 else -1.8
    best, be = -1, np.inf
    for a in elig:
        est = np.array(r["full_est"][a], float)
        z = zz[a]
        it = r["items"][str(a)] if str(a) in r["items"] else r["items"][a]
        ok = it["a"] and it["c"] and it["d"] and it["tie"] and z[0] <= thr and bool(np.all(z[1:] <= 2.0))
        if ok and est[0] < be:
            best, be = a, est[0]
    return best


BONF = {1: -1.645, 2: -1.960, 3: -2.128, 4: -2.241, 5: -2.326, 6: -2.394, 7: -2.450}   # z_(0.05/n), one-sided


def ship_thr(r, mode):
    """Shipped arm with the pre-registered SE but another (b) threshold: 'bonf' = Bonferroni z_(0.05/n_final);
    'step' = -1.64 (1 finalist), -1.8 (2-3), -2.0 (4 or more)."""
    elig = r["eligible"]
    n = len(elig)
    if mode == "tooln":   # daily_report's default --finalists: arms whose screen says FINALIST (ALC < 0, z <= -1)
        n = max(1, sum(1 for a in elig if r["full_alc"][a] < 0 and r["full_z"][a] <= -1.0))
        thr = -1.64 if n <= 1 else -1.8
    elif mode == "bonf":
        thr = BONF.get(n, -2.45)
    else:
        thr = -1.64 if n <= 1 else -1.8 if n <= 3 else -2.0
    best, be = -1, np.inf
    for a in elig:
        it = r["items"][str(a)] if str(a) in r["items"] else r["items"][a]
        ok = it["a"] and it["c"] and it["d"] and it["e"] and it["tie"] and r["full_z"][a] <= thr
        if ok and r["full_alc"][a] < be:
            best, be = a, r["full_alc"][a]
    return best


def item_rates(rows, arm=None):
    """Pass rate of each checklist item over eligible arm-worlds (or for one arm)."""
    acc = {k: [] for k in ("a", "b", "c", "d", "e")}
    for r in rows:
        for a in r["eligible"]:
            if arm is not None and a != arm:
                continue
            it = r["items"][str(a)] if str(a) in r["items"] else r["items"][a]
            for k in acc:
                acc[k].append(bool(it[k]))
    return {k: (np.mean(v) if v else float("nan")) for k, v in acc.items()}


def truth(r):
    return r["pop_truth"] if isinstance(r.get("pop_truth"), list) else r["truth"]


def main():
    proxy = float(sys.argv[1]) if len(sys.argv) > 1 else 0.97
    df = load(exact_only="--all" not in sys.argv)
    print(f"records: {len(df)} worlds; boot proxy {proxy} x CR0\n")
    groups = list(df.groupby(["scenario", "policy", "model", "strata"]))
    # ---- null table
    print("| world model (null) | worlds | P(switch) pre-registered | K60 fix | boot-only fix | finalists | ALC z of finalists: SD | 1% of min z | P(drop) per started arm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (scn, pol, model, strata), g in groups:
        if scn != "null":
            continue
        rows = [r for _, r in g.iterrows()]
        out = {}
        for rule in ("pre", "K60", "boot"):
            sh = [ship_rule(r, rule, proxy) for r in rows]
            sh = [s for s in sh if s is not None]
            out[rule] = rate(np.array(sh) >= 0) if sh else "-"
        zs, mins = [], []
        for r in rows:
            z = [r["full_z"][a] for a in r["eligible"] if r["full_z"][a] is not None and np.isfinite(r["full_z"][a])]
            zs += z
            if z:
                mins.append(min(z))
        runs = np.array([r["runs"] for r in rows])
        drop = np.array([r["dropped"] for r in rows])
        st = runs >= 6
        print(f"| {model}/{strata} ({pol}) | {len(rows)} | {out['pre']} | {out['K60']} | {out['boot']} | "
              f"{np.mean([r['n_final'] for r in rows]):.1f} | {np.std(zs, ddof=1):.2f} | {np.quantile(mins, .01):+.2f} | "
              f"{rate(drop[st])} |")
    # ---- combined amendment candidates
    print("\n| scenario | world model | worlds | pre-registered (QUEUE n_final) | pre, daily_report n_final | K60 (QUEUE n) | K60 + daily_report n |")
    print("|---|---|---|---|---|---|---|")
    for (scn, pol, model, strata), g in groups:
        rows = [r for _, r in g.iterrows()]
        arm = int(scn.split(":")[1]) if scn.startswith("one:") else None
        cells = []
        for rule, nm in (("pre", "queue"), ("pre", "tool"), ("K60", "queue"), ("K60", "tool")):
            sh = np.array([ship_rule(r, rule, proxy, nm) for r in rows])
            cells.append(rate(sh >= 0) if arm is None else rate(sh == arm))
        print(f"| {scn} | {model}/{strata} | {len(rows)} | " + " | ".join(cells) + " |")
    # ---- counterfactual thresholds and item pass rates
    print("\n| scenario | world model | worlds | P(switch / ship effect arm) pre | daily_report default n_final | step -1.64/-1.8/-2.0 (1/2-3/4+) | Bonferroni z_(0.05/n) | item pass rates (a b c d e) over finalists |")
    print("|---|---|---|---|---|---|---|---|")
    for (scn, pol, model, strata), g in groups:
        rows = [r for _, r in g.iterrows()]
        arm = int(scn.split(":")[1]) if scn.startswith("one:") else None
        sh_b = np.array([ship_thr(r, "bonf") for r in rows])
        sh_s = np.array([ship_thr(r, "step") for r in rows])
        sh_t = np.array([ship_thr(r, "tooln") for r in rows])
        if arm is None:
            pre, bon, stp, tln = (rate(np.array([r["ship"] for r in rows]) >= 0), rate(sh_b >= 0), rate(sh_s >= 0),
                                  rate(sh_t >= 0))
        else:
            pre, bon, stp, tln = (rate(np.array([r["ship"] for r in rows]) == arm), rate(sh_b == arm),
                                  rate(sh_s == arm), rate(sh_t == arm))
        ir = item_rates(rows, arm)
        print(f"| {scn} | {model}/{strata} | {len(rows)} | {pre} | {tln} | {stp} | {bon} | " +
              " ".join(f"{ir[k]:.2f}" for k in ("a", "b", "c", "d", "e")) + " |")
    # ---- alternatives
    print("\n| scenario | world model | worlds | P(ship the effect arm) pre | K60 | boot-only | P(ship another arm) | "
          "P(effect arm dropped) | effect-arm runs (median) | effect-arm full z (median) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (scn, pol, model, strata), g in groups:
        if scn == "null":
            continue
        rows = [r for _, r in g.iterrows()]
        parts = scn.split(":")
        if parts[0] == "one":
            a = int(parts[1])
            res = {}
            for rule in ("pre", "K60", "boot"):
                sh = [ship_rule(r, rule, proxy) for r in rows]
                sh = [s for s in sh if s is not None]
                res[rule] = rate(np.array(sh) == a) if sh else "-"
            other = rate([(r["ship"] >= 0 and r["ship"] != a) for r in rows])
            dr = rate([r["dropped"][a] for r in rows])
            runs = np.median([r["runs"][a] for r in rows])
            zz = np.nanmedian([r["full_z"][a] if r["full_z"][a] is not None else np.nan for r in rows])
            print(f"| {TAGS[a]} {float(parts[2]) * 1e3:+.0f}e-3, others 0 | {model}/{strata} | {len(rows)} | {res['pre']} | "
                  f"{res['K60']} | {res['boot']} | {other} | {dr} | {runs:.0f} | {zz:+.2f} |")
        else:
            res = {}
            for rule in ("pre", "K60", "boot"):
                sh = [ship_rule(r, rule, proxy) for r in rows]
                sh = [s for s in sh if s is not None]
                res[rule] = rate(np.array(sh) >= 0) if sh else "-"
            dr = rate(np.array([r["dropped"] for r in rows])[np.array([r["runs"] for r in rows]) >= 6])
            print(f"| all arms {float(parts[1]) * 1e3:+.0f}e-3 (any switch is harmful) | {model}/{strata} | {len(rows)} | "
                  f"{res['pre']} | {res['K60']} | {res['boot']} | - | {dr} (per started arm) | - | - |")


if __name__ == "__main__":
    main()
