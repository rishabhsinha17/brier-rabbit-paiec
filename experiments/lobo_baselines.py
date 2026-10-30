"""Leave-one-benchmark-out replay of the platform protocol for the reference baselines.

Compositions: k pairs from the held-out benchmark per draw. k=1 mimics the platform, where most pairs are alone on
their benchmark; larger k measures how much cross-subject pooling could add.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiec.baselines import Constant, EmpiricalMean, PooledSmoothed, SmoothedMean  # noqa: E402
from paiec.data import BINARY_BENCHMARKS, load_responses  # noqa: E402
from paiec.protocol import BUDGETS, index_pairs, materialize, pair_alc, run_protocol, unit  # noqa: E402


def train_prior(train_df) -> float:
    """Mean of pair-level success rates over training pairs (pairs carry equal weight in the metric)."""
    return float(train_df.groupby("pair")["label"].mean().mean())


def make_models(p0: float) -> dict:
    models = {"constant_0.5": Constant(0.5), "constant_p0": Constant(p0), "empirical_mean": EmpiricalMean()}
    for kappa in (1, 2, 4, 8):
        models[f"smoothed_k{kappa}"] = SmoothedMean(p0, kappa)
    for kb in (2, 8):
        for kp in (2, 4):
            models[f"pooled_kb{kb}_kp{kp}"] = PooledSmoothed(p0, kb, kp)
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40, help="draws per (fold, composition)")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "lobo_baselines.json"))
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    t0 = time.time()
    df = load_responses()
    print(f"loaded {len(df):,} binary responses in {time.time() - t0:.1f}s", flush=True)
    out = {}
    for bench in BINARY_BENCHMARKS:
        train = df[df.bench != bench]
        p0 = train_prior(train)
        models = make_models(p0)
        raw = index_pairs(df[df.bench == bench])
        for k in ks:
            scores = {name: [] for name in models}
            budgets = {name: {b: [] for b in BUDGETS} for name in models}
            n_runs = 0
            for r in range(args.runs):
                pairs = materialize(raw, size_range=(88, 226), seed=r)
                if len(pairs) < k:
                    break
                chosen = sorted(pairs, key=lambda p: unit("pick", bench, k, r, p.key))[:k]
                for name, model in models.items():
                    res = run_protocol(chosen, model.predict, seed=r)
                    for pr in res.values():
                        scores[name].append(pair_alc(pr))
                        for b in BUDGETS:
                            budgets[name][b].append(pr[b]["brier"])
                n_runs += 1
                if len(pairs) == k and k == 1 and n_runs >= 10:
                    break  # single-pair benchmark: only the split/subsample varies
            if not n_runs:
                continue
            out[f"{bench}|k={k}"] = {
                name: {"alc": float(np.mean(v)), "n_pairs": len(v),
                       "per_budget": {b: float(np.mean(budgets[name][b])) for b in BUDGETS}}
                for name, v in scores.items()
            }
            best = min(out[f"{bench}|k={k}"].items(), key=lambda kv: kv[1]["alc"])
            print(f"{bench:18s} k={k} runs={n_runs:3d} p0={p0:.3f} best={best[0]} {best[1]['alc']:.4f} "
                  f"| const0.5 {out[f'{bench}|k={k}']['constant_0.5']['alc']:.4f} "
                  f"| emp {out[f'{bench}|k={k}']['empirical_mean']['alc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
