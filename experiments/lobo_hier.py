"""LOBO replay of the v1 hierarchical model (optionally with parameter overrides as JSON)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiec.data import BINARY_BENCHMARKS, load_responses  # noqa: E402
from paiec.evaluate import FoldData, replay  # noqa: E402
from paiec.hier import HierModel, fit_abilities  # noqa: E402


def fold_prior(train_df) -> dict:
    """Benchmark-level prior from training benchmarks: mean and sd of benchmark logit levels."""
    rates = train_df.groupby(["bench", "pair"])["label"].mean().groupby(level=0).mean().clip(0.02, 0.98)
    levels = np.log(rates / (1 - rates))
    return {"m0": float(levels.mean()), "tau_mu": float(max(0.8, levels.std(ddof=0)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--params", default="{}", help="JSON overrides for HierModel params")
    ap.add_argument("--fixed-prior", action="store_true", help="use --params m0/tau_mu instead of per-fold training prior")
    args = ap.parse_args()
    ks = tuple(int(k) for k in args.ks.split(","))
    overrides = json.loads(args.params)

    t0 = time.time()
    df = load_responses()
    folds = FoldData(df, BINARY_BENCHMARKS)
    summary = {}
    for bench in BINARY_BENCHMARKS:
        train = df[df.bench != bench]
        params = dict(overrides)
        if not args.fixed_prior:
            params.update(fold_prior(train))
        model = HierModel(fit_abilities(train), params)
        res = replay(folds, bench, model.predict, ks=ks, runs=args.runs)
        summary[bench] = res
        line = " | ".join(f"k={k} {v['alc']:.4f}±{v['se']:.4f} (B0 {v['per_budget'][0]:.3f} B31 {v['per_budget'][31]:.3f})"
                          for k, v in res.items())
        print(f"{bench:18s} m0={params.get('m0', 'def'):>6} {line}  ({time.time() - t0:.0f}s)", flush=True)
    for k in ks:
        vals = [summary[b][k]["alc"] for b in summary if k in summary[b]]
        print(f"macro k={k}: {np.mean(vals):.4f} over {len(vals)} benchmarks")


if __name__ == "__main__":
    main()
