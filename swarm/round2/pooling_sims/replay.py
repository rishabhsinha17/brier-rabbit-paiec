"""Paired LOBO replay with a benchmark-keyed item subsample, for cross-subject item pooling studies.

Why: paiec.protocol.materialize subsamples each pair's items with a PAIR-keyed hash, so two pairs of one benchmark
share fewer items than complete platform groups do (head A alert). Here the subsample can be keyed on the benchmark:
one common item order and one common size target per (benchmark, run), each pair keeping the items it has in that
order, then the platform's per-pair persistent 50/50 split (unchanged). Modes:
  pair    = exactly evaluate.FoldData.draws (pair-keyed subsample; sanity check against experiments/paired.py)
  bench   = same pairs as 'pair', benchmark-keyed subsample
  matched = benchmark-keyed subsample, and partners are chosen among pairs whose item sets overlap the first pair's by
            >= 80% of the smaller set (platform groups are complete benchmarks; on MathArena / Multi-SWE-bench random
            partners often attempted disjoint item subsets). Falls back to the hash order when no such partner exists.
Every model is scored on identical draws. A probe 'model' records partner-label coverage (share of evaluation rows
whose item, keyed like HierModel by benchmark_id + content, carries a label from another subject) per pair-budget.

Usage (repo root):  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/pooling_sims/replay.py \
    --benches researchcodebench --ks 2 --modes bench --runs 3 --ref base --extra '{...}' --out <csv>
Data come from cache/replay_cache.pkl (build_cache.py), i.e. the same public LOBO priors as experiments/paired.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
WORKERS = int(os.environ.get("PAIEC_WORKERS", "2"))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from paiec.evaluate import SIZE_RANGE  # noqa: E402
from paiec.hier import _item_key, _sk  # noqa: E402
from paiec.protocol import BUDGETS, MIN_ITEMS, Pair, materialize, pair_alc, run_protocol, unit  # noqa: E402

V1_TUNED = {"m0": -0.5, "tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0}
_STATE = {}


def load_cache():
    if "cache" not in _STATE:
        with open(HERE / "cache" / "replay_cache.pkl", "rb") as f:
            _STATE["cache"] = pickle.load(f)
    return _STATE["cache"]


def materialize_bench(raw_pairs, bench, seed, size_range=SIZE_RANGE, min_items=MIN_ITEMS):
    """Benchmark-keyed subsample (common order + common size target), then the per-pair 50/50 split of protocol."""
    lo, hi = size_range
    target = lo + int(unit(seed, "size", bench) * (hi - lo + 1))
    out = []
    for rp in raw_pairs:
        key, items = rp["key"], rp["items"]
        ids = sorted(items, key=lambda i: unit(seed, "subsample", bench, i))
        ids = ids[: max(min_items, min(target, len(ids)))]
        order = sorted(ids, key=lambda i: unit(seed, "split", key, i))
        n_acq = math.ceil(len(order) / 2)
        acq = sorted(order[:n_acq], key=lambda i: unit(seed, "order", key, i))
        out.append(Pair(key=key, bench=rp["bench"], benchmark_id=rp["benchmark_id"],
                        subject_id=f"subject_{100000 + int(unit('sid', key) * 900000)}",
                        subject=dict(rp["subject"]), items={i: items[i] for i in ids}, acq=acq, eval=order[n_acq:]))
    return out


def choose(raw, bench, k, r, mode):
    keys = sorted((rp["key"] for rp in raw), key=lambda key: unit("pick", bench, k, r, key))
    if mode != "matched":
        return keys[:k]
    sets = {rp["key"]: set(rp["items"]) for rp in raw}
    chosen = [keys[0]]
    for key in keys[1:]:
        if len(chosen) == k:
            break
        if all(len(sets[key] & sets[c]) / min(len(sets[key]), len(sets[c])) >= 0.8 for c in chosen):
            chosen.append(key)
    for key in keys:
        if len(chosen) == k:
            break
        if key not in chosen:
            chosen.append(key)
    return chosen


def draws(bench, k, runs, mode, seed0=0):
    raw = load_cache()["raw"][bench]
    if len(raw) < k:
        return
    n = runs if len(raw) > 1 else min(runs, 10)
    for r in range(seed0, seed0 + n):
        keys = set(choose(raw, bench, k, r, mode))
        sel = [rp for rp in raw if rp["key"] in keys]
        if mode == "pair":
            yield r, materialize(sel, size_range=SIZE_RANGE, seed=r)
        else:
            yield r, materialize_bench(sel, bench, r)


def build_models(bench, spec):
    cache = load_cache()
    from paiec.hier import HierModel
    zoo = {}
    for name, params in spec.items():
        params = dict(params)
        base = cache["subject_prior"][bench] if params.pop("prior", "names") == "subjects" else cache["abilities"][bench]
        cls_path = params.pop("class", None)
        cls = HierModel
        if cls_path:
            mod_name, cls_name = cls_path.split(":")
            cls = getattr(importlib.import_module(mod_name), cls_name)
        zoo[name] = cls(base, dict(V1_TUNED, **params))
    return zoo


class CoverageProbe:
    """Returns 1.0 when another subject has labelled the item (benchmark_id + content key), else 0.0."""

    def predict(self, input, labeled=None):
        subject, item = input
        s, b, ik = _sk(subject), str(item.get("benchmark_id", "")), _item_key(item)
        for (subj, it), _ in labeled or []:
            if str(it.get("benchmark_id", "")) == b and _sk(subj) != s and _item_key(it) == ik:
                return 1.0
        return 0.0


def _run(task):
    bench, k, mode, runs, seed0, spec = task
    rows = []
    for r, pairs in draws(bench, k, runs, mode, seed0) or []:
        rec = []
        run_protocol(pairs, CoverageProbe().predict, seed=r, record=rec)
        cov = {}
        for key, budget, prob, _ in rec:
            a = cov.setdefault((key, budget), [0.0, 0])
            a[0] += prob
            a[1] += 1
        models = build_models(bench, spec)  # fresh instances per draw (no cross-draw caches)
        for name, model in models.items():
            t0 = time.time()
            res = run_protocol(pairs, model.predict, seed=r)
            dt = time.time() - t0
            for key, pr in res.items():
                row = {"bench": bench, "k": k, "mode": mode, "run": r, "pair": key, "model": name,
                       "alc": pair_alc(pr), "sec": dt / len(res)}
                for b in BUDGETS:
                    row[f"b{b}"] = pr[b]["brier"]
                    c = cov.get((key, b), [0.0, 1])
                    row[f"cov{b}"] = c[0] / max(c[1], 1)
                rows.append(row)
    return rows


def summarize(df, ref, out=print):
    import pandas as pd
    pd.set_option("display.width", 250)
    budgets = [f"b{b}" for b in BUDGETS]
    for (mode, k), g in df.groupby(["mode", "k"]):
        out(f"\n=== mode={mode} k={k} ===")
        wide = g.pivot_table(index=["bench", "run", "pair"], columns="model", values="alc")
        lines = []
        for bench, gb in wide.groupby(level=0):
            line = {"bench": bench, "n": len(gb)}
            for m in wide.columns:
                d = gb[m] - gb[ref]
                line[m] = (f"{gb[m].mean():.4f}" if m == ref else
                           f"{d.mean():+.4f}±{d.std(ddof=1) / np.sqrt(len(d)):.4f}" if len(d) > 1 else f"{d.mean():+.4f}")
            lines.append(line)
        d_all = wide.sub(wide[ref], axis=0)
        bench_mean = d_all.groupby(level=0).mean()
        line = {"bench": "MACRO(bench-equal)", "n": len(wide)}
        for m in wide.columns:
            line[m] = f"{wide[ref].groupby(level=0).mean().mean():.4f}" if m == ref else f"{bench_mean[m].mean():+.4f}"
        lines.append(line)
        out(pd.DataFrame(lines).to_string(index=False))
        cov = g[g.model == ref].groupby("bench")[[f"cov{b}" for b in BUDGETS]].mean()
        out("partner-label coverage of eval rows (mean over pairs):\n" + cov.round(3).to_string())
        out("per-budget Brier diff vs ref (bench-equal macro; SE over pairs pooled):")
        per = []
        for m in wide.columns:
            if m == ref:
                continue
            line = {"model": m}
            for bcol in budgets:
                w2 = g.pivot_table(index=["bench", "run", "pair"], columns="model", values=bcol)
                d = w2[m] - w2[ref]
                line[bcol] = f"{d.groupby(level=0).mean().mean():+.5f}({d.std(ddof=1) / np.sqrt(len(d)):.5f})"
            per.append(line)
        out(pd.DataFrame(per).to_string(index=False))
        sec = g.groupby("model")["sec"].mean()
        out("mean seconds per pair-run: " + ", ".join(f"{m} {v:.2f}" for m, v in sec.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="real_webagents,researchcodebench,multi_swebench,matharena")
    ap.add_argument("--ks", default="2")
    ap.add_argument("--modes", default="bench")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=5, help="runs per task (parallel granularity)")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pandas as pd

    spec = json.loads(args.extra)
    tasks = []
    for bench in args.benches.split(","):
        for k in args.ks.split(","):
            for mode in args.modes.split(","):
                for s in range(args.seed0, args.seed0 + args.runs, args.chunk):
                    tasks.append((bench, int(k), mode, min(args.chunk, args.seed0 + args.runs - s), s, spec))
    t0 = time.time()
    if WORKERS <= 1:
        rows = [row for t in tasks for row in _run(t)]
    else:
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(tasks))) as pool:
            rows = [row for chunk in pool.map(_run, tasks) for row in chunk]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} pair-scores in {time.time() - t0:.0f}s -> {args.out}")
    summarize(df, args.ref)


if __name__ == "__main__":
    main()
