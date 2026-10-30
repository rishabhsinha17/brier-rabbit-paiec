"""Semi-synthetic "hardened" LOBO replay (postdoc_adaptation, swarm round 1).

Why: the public pool has few near-zero pairs (P(rate < .05) ~ 0.06), so mixture/pooling mechanisms that only matter
when some pairs are near zero cannot be tested on plain LOBO. This harness takes the usual paired LOBO draws
(paiec.evaluate.FoldData, identical to experiments/paired.py) and, per scenario, replaces ALL recorded responses of
selected pairs (acquisition and evaluation items) by Bernoulli(r) draws with r ~ Beta(0.5, 30) (mean 0.016), i.e. a
near-zero pair on the same items. Which pairs are hardened:
  none     : nothing (plain LOBO, the no-harm check)
  pair30   : each pair independently with prob 0.30            (hardness is a pair property)
  bench30  : all pairs of the draw together with prob 0.30     (hardness is a benchmark property)
  mixed    : benchmark-wide with prob 0.20, else each pair with prob 0.15
Everything is keyed by sha256 hashes of (scenario, run, pair), so all models see identical data.
Models are specified exactly like experiments/paired.py --extra (same V1_TUNED defaults, same plug-in syntax).
Output CSV rows: bench, k, run, scen, pair, hard, partner_hard, model, alc, b0..b31. Public data only.

Usage (from brier_rabbit/):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round1/postdoc_adaptation_sims/harden_replay.py \
      --runs 20 --ks 2 --scens none,pair30,bench30,mixed --ref mix --out results/pdadapt_bcm.csv --extra '{...}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WORKERS = int(os.environ.get("PAIEC_WORKERS", "2"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

SCEN = {"none": (0.0, 0.0), "pair30": (0.30, 0.0), "bench30": (0.0, 0.30), "mixed": (0.15, 0.20),
        "pair15": (0.15, 0.0), "bench15": (0.0, 0.15)}
_STATE = {}


def _init():
    from paiec.data import BINARY_BENCHMARKS, load_responses
    from paiec.evaluate import FoldData

    df = load_responses()
    _STATE["df"] = df
    _STATE["folds"] = FoldData(df, BINARY_BENCHMARKS)


def harden(pairs, scen: str, r: int):
    import copy

    from paiec.protocol import unit

    pi_pair, pi_bench = SCEN[scen]
    bench_hard = unit("hb", scen, r) < pi_bench
    out, flags = [], {}
    for p in pairs:
        hard = bench_hard or (unit("hp", scen, r, p.key) < pi_pair)
        q = copy.copy(p)
        if hard:
            rng = np.random.default_rng(int(unit("rate", scen, r, p.key) * 2 ** 32))
            rate = float(rng.beta(0.5, 30.0))
            q.items = {i: {"item": v["item"], "labels": [int(rng.random() < rate) for _ in v["labels"]]}
                       for i, v in p.items.items()}
        flags[p.key] = bool(hard)
        out.append(q)
    return out, flags


def _run(task):
    from paired import build_models
    from paiec.protocol import BUDGETS, pair_alc, run_protocol

    bench, k, runs, scens, spec = task
    df = _STATE["df"]
    models = build_models(df[df.bench != bench], spec)
    models = {n: m for n, m in models.items() if n in spec}  # only the requested arms
    rows = []
    for r, pairs in _STATE["folds"].draws(bench, k, runs, 0) or []:
        for scen in scens:
            hp, flags = harden(pairs, scen, r)
            for name, model in models.items():
                res = run_protocol(hp, model.predict, seed=r)
                for key, pr in res.items():
                    partner = any(flags[o] for o in flags if o != key)
                    rows.append({"bench": bench, "k": k, "run": r, "scen": scen, "pair": key, "hard": flags[key],
                                 "partner_hard": partner, "model": name, "alc": pair_alc(pr),
                                 **{f"b{b}": pr[b]["brier"] for b in BUDGETS}})
    return rows


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--ks", default="2")
    ap.add_argument("--benches", default="matharena,multi_swebench,real_webagents,researchcodebench")
    ap.add_argument("--scens", default="none,pair30,bench30,mixed")
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--ref", default="mix")
    ap.add_argument("--out", default=str(ROOT / "results" / "pdadapt_harden.csv"))
    args = ap.parse_args()
    spec = json.loads(args.extra)
    scens = args.scens.split(",")
    tasks = [(b, int(k), args.runs, scens, spec) for b in args.benches.split(",") for k in args.ks.split(",")]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(WORKERS, 8, len(tasks)), initializer=_init) as pool:
        rows = [row for chunk in pool.map(_run, tasks) for row in chunk]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} pair-scores in {time.time() - t0:.0f}s -> {args.out}\n")
    pd.set_option("display.width", 250)
    cols = ["alc", "b0", "b1", "b3", "b7", "b15", "b31"]
    idx = ["bench", "k", "run", "scen", "pair"]
    for scen in scens:
        d = df[df.scen == scen]
        wide = d.pivot_table(index=idx, columns="model", values="alc")
        if args.ref not in wide:
            continue
        print(f"=== scenario {scen}: bench-equal mean ALC, paired diff vs {args.ref} (SE over pair-draws)")
        lines = []
        for m in wide.columns:
            diff = (wide[m] - wide[args.ref])
            per_b = diff.groupby(level=0).mean()
            lines.append({"model": m, "alc": round(wide[m].groupby(level=0).mean().mean(), 4),
                          "diff": round(per_b.mean(), 5), "se": round(diff.std() / np.sqrt(len(diff)), 5),
                          **{f"d_{b}": round(per_b[b], 4) for b in per_b.index}})
        print(pd.DataFrame(lines).to_string(index=False))
        # paired.py-style macro: bench-equal mean of the k-averaged per-(bench, k) means, and per-(bench, k) diffs
        mk = wide.groupby(level=[0, 1]).mean()
        macro = mk.groupby(level=0).mean().mean()
        print("  macro (bench-equal, k-averaged): " + "  ".join(f"{m} {v:.4f} ({v - macro[args.ref]:+.4f})" for m, v in macro.sort_values().items()))
        for (bb, kk), gg in wide.groupby(level=[0, 1]):
            dd = gg.sub(gg[args.ref], axis=0)
            print(f"  {bb:>18} k={kk} n={len(gg):3d}: " + "  ".join(
                f"{m} {dd[m].mean():+.4f}±{dd[m].std() / np.sqrt(len(dd)):.4f}" for m in gg.columns if m != args.ref))
        bud = d.pivot_table(index=idx, columns="model", values=cols[1:])
        print("  per-budget diff vs ref (pooled): " + "; ".join(
            f"{m}: " + " ".join(f"{c} {float((bud[(c, m)] - bud[(c, args.ref)]).mean()):+.4f}" for c in cols[1:])
            for m in sorted(set(c[1] for c in bud.columns)) if m != args.ref))
        # per budget and by hardness cell (pooled over benchmarks)
        for cell, g in d.groupby(["hard", "partner_hard"]):
            w = g.pivot_table(index=idx, columns="model", values=cols)
            if w.empty:
                continue
            out = {}
            for m in sorted(set(c[1] for c in w.columns)):
                if m == args.ref:
                    continue
                out[m] = {c: round(float((w[(c, m)] - w[(c, args.ref)]).mean()), 4) for c in cols}
            print(f"  cell hard={cell[0]} partner_hard={cell[1]} (n={len(w)}): " + "; ".join(
                f"{m}: " + " ".join(f"{c} {v:+.4f}" for c, v in o.items()) for m, o in out.items()))
        print()


if __name__ == "__main__":
    main()
