"""Summaries of the operating-characteristics runs (swarm round 6): one block per (scenario, policy, model, strata).

usage: R swarm/round6/oc_sims/summarize_oc.py [glob] [--all]   (default cache/oc_*.jsonl, exact final-look SE records
only; --all adds the early records whose final SE used the CR0 proxy)
For every group: worlds; P(switch) with its binomial Monte Carlo SE; P(switch to an arm whose pool truth is < 0 /
> 0); for single-arm scenarios P(switch to the effect arm); drops per arm and why; runs; finalists; commits; and
counterfactual (b)-thresholds (the checklist re-evaluated from the stored items and z, nothing re-simulated).
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
TAGS = ["W20", "L15", "W15", "WLA", "WLB", "S6", "S7"]


def p_se(x):
    x = np.asarray(x, float)
    n = len(x)
    p = x.mean() if n else float("nan")
    return p, math.sqrt(max(p * (1 - p), 1.0 / n) / n) if n else float("nan")   # floor: 1 event-equivalent


def fmt(p, se):
    return f"{p:.3f} ({se:.3f})"


def reship(r, thr_one=-1.64, thr_more=-1.8, use_hold=True, min_runs=30):
    """Re-evaluate the final checklist with other (b) thresholds from the stored per-arm items and z."""
    elig = r["eligible"]
    thr = thr_one if len(elig) <= 1 else thr_more
    best, be = -1, np.inf
    for a in elig:
        it = r["items"][str(a)] if str(a) in r["items"] else r["items"][a]
        z = r["full_z"][a]
        ok = (it["a"] or not use_hold) and it["c"] and it["d"] and it["e"] and it["tie"] and z <= thr
        if ok and r["full_alc"][a] < be:
            best, be = a, r["full_alc"][a]
    return best


FLOOR = np.array([0.0147, 0.055, 0.0487, 0.0132, 0.0076, 0.0043, 0.0039])


def reship_se(r, rule, thr_one=-1.64, thr_more=-1.8):
    """Re-evaluate (b) and (e) with another SE rule from the stored full-data effects and sandwich SEs:
    'nofloor' SE = max(1.2 x 0.97 x CR0, CR1); 'k60' drops the floor only for arms with K_full >= 60."""
    if not isinstance(r.get("full_cr1"), list):
        return None
    elig = r["eligible"]
    thr = thr_one if len(elig) <= 1 else thr_more
    best, be = -1, np.inf
    for a in elig:
        est = np.array(r["full_est"][a], float)
        cr1 = np.nan_to_num(np.array(r["full_cr1"][a], float))
        cr0 = np.nan_to_num(np.array(r["full_cr0"][a], float))
        K = r["K_full"][a]
        se = np.fmax(1.2 * 0.97 * cr0, cr1)
        if rule == "k60" and K < 60:
            se = np.fmax(se, FLOOR / np.sqrt(max(K, 1)))
        z = est / se
        it = r["items"][str(a)] if str(a) in r["items"] else r["items"][a]
        ok = it["a"] and it["c"] and it["d"] and it["tie"] and z[0] <= thr and np.all(z[1:] <= 2.0)
        if ok and est[0] < be:
            best, be = a, est[0]
    return best


def zdist(df):
    zs, mins = [], []
    for _, r in df.iterrows():
        el = r["eligible"]
        z = [r["full_z"][a] for a in el if np.isfinite(r["full_z"][a])]
        zs += z
        if z:
            mins.append(min(z))
    zs, mins = np.array(zs), np.array(mins)
    return zs, mins


def summarize(df: pd.DataFrame, title: str):
    n = len(df)
    sw = df["ship"] >= 0
    tk = lambda r: r["pop_truth"] if isinstance(r.get("pop_truth"), list) else r["truth"]  # noqa: E731
    truth = [tk(r)[r["ship"]] if r["ship"] >= 0 else 0.0 for _, r in df.iterrows()]
    better = [(r["ship"] >= 0 and tk(r)[r["ship"]] < -1e-9) for _, r in df.iterrows()]
    worse = [(r["ship"] >= 0 and tk(r)[r["ship"]] > 1e-9) for _, r in df.iterrows()]
    print(f"\n== {title}: {n} worlds")
    print(f"  P(switch) {fmt(*p_se(sw))}; to a truly better arm {fmt(*p_se(better))}; to a truly worse arm "
          f"{fmt(*p_se(worse))}; E[true ALC of the shipped arm, REF = 0] {1e3 * np.mean(truth):+.3f} x1e-3")
    scn = df["scenario"].iloc[0]
    if scn.startswith("one:"):
        a = int(scn.split(":")[1])
        hit = df["ship"] == a
        other = sw & ~hit
        print(f"  effect arm {TAGS[a]}: P(ship it) {fmt(*p_se(hit))}; P(ship another arm) {fmt(*p_se(other))}")
        dr = np.array([r["dropped"][a] for _, r in df.iterrows()])
        el = np.array([a in r["eligible"] for _, r in df.iterrows()])
        runs = np.array([r["runs"][a] for _, r in df.iterrows()])
        zz = np.array([r["full_z"][a] for _, r in df.iterrows()])
        print(f"  effect arm: dropped {dr.mean():.3f}, finalist (>= 30 runs, not dropped) {el.mean():.3f}, runs "
              f"median {np.median(runs):.0f} (p10 {np.percentile(runs, 10):.0f}, p90 {np.percentile(runs, 90):.0f}), "
              f"full-data z median {np.nanmedian(zz):+.2f}")
    runs = np.array(df["runs"].tolist())
    drop = np.array(df["dropped"].tolist())
    print("  runs per arm (mean): " + ", ".join(f"{t} {runs[:, i].mean():.0f}" for i, t in enumerate(TAGS)) +
          f"; REF {df['runs_ref'].mean():.0f}")
    started = runs >= 6
    print("  P(dropped | started): " + ", ".join(
        f"{t} {drop[started[:, i], i].mean():.2f}" if started[:, i].any() else f"{t} -" for i, t in enumerate(TAGS)))
    why = {}
    for _, r in df.iterrows():
        for i, w in enumerate(r["drop_why"]):
            if w:
                for part in w.split("+"):
                    why[part] = why.get(part, 0) + 1
    ndrop = drop.sum()
    print(f"  drops: {ndrop} over {started.sum()} started arm-worlds ({ndrop / max(started.sum(), 1):.3f}); reasons "
          f"(an arm can have several): {why}")
    print(f"  finalists per world: mean {df['n_final'].mean():.2f}, distribution "
          f"{df['n_final'].value_counts().sort_index().to_dict()}; commits {(df['commit_day'] >= 0).mean():.3f}")
    for lab, kw in (("(b) -1.64/-1.8 as pre-registered", {}), ("(b) -1.64 for any number", {"thr_more": -1.64}),
                    ("(b) -2.0 for 2+", {"thr_more": -2.0}), ("(b) -2.24 for 2+ (Bonferroni-type, 5 arms)",
                                                                {"thr_more": -2.24}),
                    ("no holdout item (a)", {"use_hold": False})):
        s2 = np.array([reship(r, **kw) for _, r in df.iterrows()])
        b2 = np.array([(x >= 0 and tk(r)[x] < -1e-9) for x, (_, r) in zip(s2, df.iterrows())])
        print(f"  counterfactual {lab:44s} P(switch) {fmt(*p_se(s2 >= 0))}; to a better arm {fmt(*p_se(b2))}")
    for rule in ("nofloor", "k60"):
        s3 = [reship_se(r, rule) for _, r in df.iterrows()]
        if all(x is None for x in s3):
            continue
        keep = [i for i, x in enumerate(s3) if x is not None]
        s3a = np.array([s3[i] for i in keep])
        rows = [df.iloc[i] for i in keep]
        b3 = np.array([(x >= 0 and tk(r)[x] < -1e-9) for x, r in zip(s3a, rows)])
        w3 = np.array([(x >= 0 and tk(r)[x] > 1e-9) for x, r in zip(s3a, rows)])
        print(f"  counterfactual SE rule {rule:8s} ({len(keep)} worlds with SE parts)         P(switch) "
              f"{fmt(*p_se(s3a >= 0))}; better {fmt(*p_se(b3))}; worse {fmt(*p_se(w3))}")
    zs, mins = zdist(df)
    if len(zs):
        print(f"  full-data ALC z of finalists: mean {zs.mean():+.2f}, SD {zs.std(ddof=1):.2f}, P(z <= -1.64) "
              f"{(zs <= -1.64).mean():.3f}, P(z <= -1.8) {(zs <= -1.8).mean():.3f} ({len(zs)} arm-worlds); min over "
              f"finalists per world: median {np.median(mins):+.2f}, 5% {np.quantile(mins, .05):+.2f}, 1% "
              f"{np.quantile(mins, .01):+.2f}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pat = args[0] if args else str(HERE / "cache" / "oc_*.jsonl")
    recs = []
    for f in sorted(glob.glob(pat)):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "--all" in sys.argv or r.get("final_se") == "exact":   # default: exact final-look SE only
                    recs.append(r)
    if not recs:
        print("no records")
        return
    df = pd.DataFrame(recs).drop_duplicates(subset=["scenario", "policy", "model", "strata", "wseed"])
    for key, g in df.groupby(["scenario", "policy", "model", "strata"], sort=True):
        summarize(g.reset_index(drop=True), " / ".join(key))


if __name__ == "__main__":
    main()
