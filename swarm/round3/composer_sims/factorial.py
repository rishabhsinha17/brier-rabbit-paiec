"""Paired factorial of the ComboModel switches on identical draws: main effects, interactions and additivity.

Factors: S = subjcal (attr_scale 1.56, attr_year_max 2026.4), C = pool_item (CavityPoolModel, sigma_x 2.5; --sigma_x),
P = pool_sim (SimilarityPoolModel with P2 shares rho_name 0.45, rho_h 0.288), G = sigma_g 1.4 (plain HierModel param). Every arm of the 2^f design runs on the
same draws; per pair-draw we form the factorial contrasts (e.g. interaction SC = SC - S - C + base), so the SEs over
pair-draws are paired.
Draw modes (public data only; LOBO subject priors as experiments/paired.py):
  bench  benchmark-keyed k=2 draws, exactly paiec.evaluate.FoldData(subsample="bench").draws (= paired.py
         --subsample bench), runs seed0.. (pdpool_bench_k2 uses runs 0-19 of the same draws).
  hsame  Multi-SWE-bench k=2: first pair by the paired.py pick order, partner = the next pair with the SAME harness
         string (pool_sim's harness effect fires), benchmark-keyed subsample.
  xname  REAL x ResearchCodeBench pairs with the SAME canonical model name, subject dicts decoupled across benchmarks
         (hidden pool: no subject dict spans benchmarks), leave-two-out prior (as the skeptic's simpool_xname.py).
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round3/composer_sims/factorial.py \
           --mode bench --bench researchcodebench --runs 6 --factors S,C [--lb REF|W20|L15] [--seed0 0]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

ROOT = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "swarm/round2/skeptic_plugins_sims"))
sys.path.insert(0, os.path.join(ROOT, "swarm/round1/postdoc_priors_sims"))
from common import BASE, draws, load_cache  # noqa: E402

import numpy as np  # noqa: E402

from paiec.ext.combo import ComboModel  # noqa: E402
from paiec.protocol import BUDGETS, materialize, pair_alc, run_protocol, unit  # noqa: E402
from paiec.evaluate import SIZE_RANGE  # noqa: E402
from paiec.subjects import name_key  # noqa: E402

LB = {"REF": dict(BASE), "W20": dict(BASE, tau_mu=2.0), "L15": dict(BASE, m0=-1.5)}
SWITCH = {"S": {"subjcal": True, "attr_scale": 1.56, "attr_year_max": 2026.4},
          "C": {"pool_item": True, "sigma_x": 2.5},
          "P": {"pool_sim": True, "rho_name": 0.45, "rho_h": 0.288},
          "G": {"sigma_g": 1.4}}  # plain param (pdadapt_sigma_g's sg14 arm)


def make_draws(mode, bench, runs, seed0):
    raw, priors = load_cache()
    if mode == "bench":
        return [(r, pairs, priors[bench]) for r, pairs in draws(bench, 2, runs, seed0, subsample="bench")]
    if mode == "hsame":
        rps = raw["multi_swebench"]
        h = lambda rp: str(rp["subject"].get("harness", "") or "").strip()  # noqa: E731
        out = []
        r = seed0
        while len(out) < runs and r < seed0 + 200:
            keys = sorted(rps, key=lambda rp: unit("pick", "multi_swebench", 2, r, rp["key"]))
            first = keys[0]
            mate = next((rp for rp in keys[1:] if h(rp) and h(rp) == h(first)), None)
            if mate is not None:
                out.append((r, materialize([first, mate], size_range=SIZE_RANGE, seed=r, subsample="bench"),
                            priors["multi_swebench"]))
            r += 1
        return out
    if mode == "xname":
        from light import load_light
        from paiec.hier import fit_subject_prior
        held = ("real_webagents", "researchcodebench")
        df = load_light()
        prior = fit_subject_prior(df[~df.bench.isin(held)])
        names = {b: {} for b in held}
        for b in held:
            for rp in raw[b]:
                nk = name_key(rp["subject"])
                if nk:
                    names[b].setdefault(nk, []).append(rp)
        common = sorted(set(names[held[0]]) & set(names[held[1]]))
        out = []
        for nm in common[seed0: seed0 + runs]:
            pairs = []
            for b in held:
                rp = sorted(names[b][nm], key=lambda x: unit("xn", 0, x["key"]))[0]
                pairs += materialize([rp], size_range=SIZE_RANGE, seed=0)
            for pr in pairs:
                pr.subject = dict(pr.subject, access_date="decoupled-" + pr.bench)
            out.append((0, pairs, prior))
        return out
    raise SystemExit(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bench")
    ap.add_argument("--bench", default="researchcodebench")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--factors", default="S,C")
    ap.add_argument("--lb", default="REF")
    ap.add_argument("--out", default="")
    ap.add_argument("--sigma_x", type=float, default=2.5)
    args = ap.parse_args()
    SWITCH["C"]["sigma_x"] = args.sigma_x
    t0 = time.time()
    fac = args.factors.split(",")
    arms = {}
    for bits in itertools.product((0, 1), repeat=len(fac)):
        on = [f for f, b in zip(fac, bits) if b]
        sw = {}
        for f in on:
            sw.update(SWITCH[f])
        arms["".join(on) or "base"] = dict(LB[args.lb], **sw)
    ds = make_draws(args.mode, args.bench, args.runs, args.seed0)
    rows = []
    for r, pairs, prior in ds:
        for name, params in arms.items():
            res = run_protocol(pairs, ComboModel(prior, params).predict, seed=r)
            for key, pr in res.items():
                rows.append({"run": r, "pair": key, "arm": name, "alc": pair_alc(pr),
                             **{f"b{b}": pr[b]["brier"] for b in BUDGETS}})
    import pandas as pd
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    cols = ["alc"] + [f"b{b}" for b in BUDGETS]
    W = {c: df.pivot_table(index=["run", "pair"], columns="arm", values=c) for c in cols}
    n = len(W["alc"])
    print(f"mode={args.mode} bench={args.bench if args.mode == 'bench' else '-'} lb={args.lb} draws={len(ds)} "
          f"pair-draws={n} base ALC {W['alc']['base'].mean():.4f}  ({time.time() - t0:.0f}s)")
    # factorial contrasts: effect of subset T = sum over U subset of T of (-1)^{|T|-|U|} arm(U)
    contrasts = {}
    for size in range(1, len(fac) + 1):
        for T in itertools.combinations(fac, size):
            coef = {}
            for usize in range(0, size + 1):
                for U in itertools.combinations(T, usize):
                    coef["".join(U) or "base"] = (-1) ** (size - usize)
            contrasts[("main " if size == 1 else "inter ") + "".join(T)] = coef
    full = "".join(fac)
    contrasts["all - base"] = {full: 1, "base": -1}
    contrasts["all - base - sum(mains)"] = dict({full: 1, "base": -1 + len(fac)}, **{f: -1 for f in fac})
    print("x1e-3 (negative = better); mean +- SE over pair-draws")
    for name, coef in contrasts.items():
        line = []
        for c in cols:
            d = sum(w * W[c][a] for a, w in coef.items())
            line.append(f"{c.upper()} {1e3 * d.mean():+.2f}" + (f" +- {1e3 * d.std(ddof=1) / np.sqrt(n):.2f}" if c == "alc" else ""))
        print(f"  {name:26s} " + " | ".join(line))


if __name__ == "__main__":
    main()
