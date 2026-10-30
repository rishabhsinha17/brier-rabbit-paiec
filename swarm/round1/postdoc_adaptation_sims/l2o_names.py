"""Cross-pair similarity pooling test harness (postdoc_adaptation, swarm round 1). Public data only.

Modes (each draw = 2 pairs; the subject prior is trained on the benchmarks NOT in the draw):
  xname    : leave-two-benchmarks-out; one pair from each held-out benchmark, SAME canonical name (subjects.name_key)
  xrandom  : leave-two-benchmarks-out; one pair from each held-out benchmark, different names (control: must tie)
  hsame    : multi_swebench LOBO; two pairs on the benchmark with the SAME harness string and different names
  hrandom  : multi_swebench LOBO; two random pairs (as in paired.py k=2)
Pair sizes are subsampled to the platform-like 88-226 items exactly as paiec.evaluate.FoldData does. Models use the
experiments/paired.py --extra syntax and defaults. Output CSV rows: mode, combo, run, pair, model, alc, b0..b31.

Usage (from brier_rabbit/):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round1/postdoc_adaptation_sims/l2o_names.py --runs 2 \
     --modes xname,xrandom,hsame,hrandom --ref base --out results/pdadapt_simpool.csv --extra '{...}'
"""
from __future__ import annotations

import argparse
import itertools
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
BENCHES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
_STATE = {}


def _init():
    from paiec.data import load_responses
    from paiec.protocol import index_pairs
    from paiec.subjects import name_key

    df = load_responses()
    _STATE["df"] = df
    raw = {b: index_pairs(df[df.bench == b]) for b in BENCHES}
    for b, rps in raw.items():
        for rp in rps:
            rp["name"] = name_key(rp["subject"])
            rp["harness"] = str(rp["subject"].get("harness", "") or "")
    _STATE["raw"] = raw


def _draws(mode, combo, runs, limit):
    from paiec.protocol import unit

    raw = _STATE["raw"]
    out = []
    if mode in ("xname", "xrandom"):
        b1, b2 = combo
        n1 = {rp["name"] for rp in raw[b1] if rp["name"]}
        n2 = {rp["name"] for rp in raw[b2] if rp["name"]}
        if mode == "xname":
            for nm in sorted(n1 & n2):
                for r in range(runs):
                    pick = []
                    for b in (b1, b2):
                        cands = sorted((rp for rp in raw[b] if rp["name"] == nm), key=lambda rp: unit("xn", r, rp["key"]))
                        pick.append(cands[0])
                    out.append((r, pick))
        else:
            for r in range(runs * 4):
                a = sorted(raw[b1], key=lambda rp: unit("xr1", r, rp["key"]))[0]
                cands = [rp for rp in raw[b2] if rp["name"] != a["name"]]
                c = sorted(cands, key=lambda rp: unit("xr2", r, rp["key"]))[0]
                out.append((r, [a, c]))
    else:
        b = combo[0]
        rps = raw[b]
        if mode == "hsame":
            pairs = [(x, y) for x, y in itertools.combinations(rps, 2)
                     if x["harness"] and x["harness"] == y["harness"] and x["name"] != y["name"]]
        else:
            pairs = list(itertools.combinations(rps, 2))
        pairs = sorted(pairs, key=lambda xy: unit(mode, xy[0]["key"], xy[1]["key"]))[:limit]
        out = [(i % max(runs, 1), list(xy)) for i, xy in enumerate(pairs)]
    return out


def _run(task):
    from paired import build_models
    from paiec.protocol import BUDGETS, materialize, pair_alc, run_protocol

    mode, combo, runs, limit, spec = task
    df = _STATE["df"]
    held = list(combo)
    models = build_models(df[~df.bench.isin(held)], spec)
    models = {n: m for n, m in models.items() if n in spec}
    rows = []
    for i, (r, picks) in enumerate(_draws(mode, combo, runs, limit)):
        pairs = []
        for rp in picks:
            pairs += materialize([rp], size_range=(88, 226), seed=r)
        for name, model in models.items():
            res = run_protocol(pairs, model.predict, seed=r)
            for key, pr in res.items():
                rows.append({"mode": mode, "combo": "+".join(combo), "draw": i, "run": r, "pair": key, "model": name,
                             "alc": pair_alc(pr), **{f"b{b}": pr[b]["brier"] for b in BUDGETS}})
    return rows


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--modes", default="xname,xrandom,hsame,hrandom")
    ap.add_argument("--limit", type=int, default=40, help="max draws per within-benchmark mode")
    ap.add_argument("--combos", default="", help="comma list like matharena+multi_swebench (default: all 6)")
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--out", default=str(ROOT / "results" / "pdadapt_simpool.csv"))
    args = ap.parse_args()
    spec = json.loads(args.extra)
    combos = [tuple(c.split("+")) for c in args.combos.split(",")] if args.combos else list(itertools.combinations(BENCHES, 2))
    tasks = []
    for mode in args.modes.split(","):
        if mode in ("xname", "xrandom"):
            tasks += [(mode, c, args.runs, args.limit, spec) for c in combos]
        else:
            tasks.append((mode, ("multi_swebench",), args.runs, args.limit, spec))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(WORKERS, 8, len(tasks)), initializer=_init) as pool:
        rows = [row for chunk in pool.map(_run, tasks) for row in chunk]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} pair-scores in {time.time() - t0:.0f}s -> {args.out}\n")
    cols = ["alc", "b0", "b1", "b3", "b7", "b15", "b31"]
    for mode, g in df.groupby("mode"):
        w = g.pivot_table(index=["combo", "draw", "pair"], columns="model", values=cols)
        print(f"=== {mode}: {len(w)} pair-draws; diff vs {args.ref} (mean, SE over pair-draws)")
        for m in sorted(set(c[1] for c in w.columns)):
            if m == args.ref:
                continue
            parts = []
            for c in cols:
                d = w[(c, m)] - w[(c, args.ref)]
                parts.append(f"{c} {d.mean():+.4f}" + (f"±{d.std() / np.sqrt(len(d)):.4f}" if c == "alc" else ""))
            print(f"  {m}: " + "  ".join(parts))
        print(f"  {args.ref} absolute ALC {w[('alc', args.ref)].mean():.4f}")
    print()


if __name__ == "__main__":
    main()
