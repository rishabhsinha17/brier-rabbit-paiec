"""Paired LOBO comparison: every model is scored on identical draws, so differences have small standard errors."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import os

# Laptop-friendly defaults: cap worker processes and keep BLAS single-threaded in every worker.
WORKERS = int(os.environ.get("PAIEC_WORKERS", "2"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V1_TUNED = {"m0": -0.5, "tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0}
_STATE = {}


def _init(subsample: str = "pair"):
    from paiec.data import BINARY_BENCHMARKS, load_responses
    from paiec.evaluate import FoldData

    df = load_responses()
    _STATE["df"] = df
    _STATE["folds"] = FoldData(df, BINARY_BENCHMARKS, subsample=subsample)


def build_models(train, spec: dict) -> dict:
    from paiec.baselines import Constant, EmpiricalMean, PooledSmoothed, SmoothedMean
    from paiec.hier import HierModel, fit_abilities, fit_subject_prior

    p0 = float(train.groupby("pair")["label"].mean().mean())
    abilities = fit_abilities(train)
    subject_prior = fit_subject_prior(train)
    zoo = {
        "const_0.5": lambda: Constant(0.5),
        "empirical_mean": lambda: EmpiricalMean(),
        "smoothed_k2": lambda: SmoothedMean(p0, 2),
        "pooled_kb8_kp4": lambda: PooledSmoothed(p0, 8, 4),
        "v1_default": lambda: HierModel(abilities, {"m0": -0.5, "tau_mu": 1.2}),
        "v1_tuned": lambda: HierModel(abilities, V1_TUNED),
        "v1_subjects": lambda: HierModel(subject_prior, V1_TUNED),
    }
    np_prior = None
    for name, params in spec.items():  # extra variants from the command line
        params = dict(params)
        kind = params.pop("prior", "names")
        if kind == "subjects_np":  # provider-free attribute ridge, refit inside the fold (paiec/ext/np_prior.py)
            if np_prior is None:
                from paiec.ext.np_prior import fit_subject_prior_np
                np_prior = fit_subject_prior_np(train)
            base = np_prior
        else:
            base = subject_prior if kind == "subjects" else abilities
        cls_path = params.pop("class", None)  # plug-in: "paiec.ext.module:ClassName" taking (prior, params)
        if cls_path:
            import importlib
            mod_name, cls_name = cls_path.split(":")
            cls = getattr(importlib.import_module(mod_name), cls_name)
            zoo[name] = (lambda pr=params, ab=base, c=cls: c(ab, dict(V1_TUNED, **pr)))
        else:
            zoo[name] = (lambda pr=params, ab=base: HierModel(ab, dict(V1_TUNED, **pr)))
    return {name: make() for name, make in zoo.items()}


def _run(task):
    from paiec.protocol import BUDGETS, pair_alc, run_protocol

    bench, k, runs, spec = task
    df = _STATE["df"]
    models = build_models(df[df.bench != bench], spec)
    rows = []
    for r, pairs in _STATE["folds"].draws(bench, k, runs, 0) or []:
        for name, model in models.items():
            res = run_protocol(pairs, model.predict, seed=r)
            for key, pr in res.items():
                rows.append({"bench": bench, "k": k, "run": r, "pair": key, "model": name, "alc": pair_alc(pr),
                             **{f"b{b}": pr[b]["brier"] for b in BUDGETS}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--benches", default="matharena,multi_swebench,real_webagents,researchcodebench")
    ap.add_argument("--extra", default="{}", help='JSON {"name": {param overrides}} of extra v1 variants')
    ap.add_argument("--ref", default="v1_tuned")
    ap.add_argument("--out", default=str(ROOT / "results" / "paired.csv"))
    ap.add_argument("--subsample", choices=["pair", "bench"], default="pair",
                    help="item subsample key: 'pair' (default, all existing results) or 'bench' (co-drawn pairs of a "
                         "benchmark keep the same items where they overlap, closer to the platform's complete groups)")
    args = ap.parse_args()
    import pandas as pd

    spec = json.loads(args.extra)
    tasks = [(b, int(k), args.runs, spec) for b in args.benches.split(",") for k in args.ks.split(",")]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(WORKERS, 8, len(tasks)), initializer=_init,
                             initargs=(args.subsample,)) as pool:
        rows = [row for chunk in pool.map(_run, tasks) for row in chunk]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if args.subsample != "pair":
        print(f"subsample={args.subsample} (benchmark-keyed item subsample)")
    print(f"{len(df)} pair-scores in {time.time() - t0:.0f}s -> {args.out}\n")
    # Per (bench, k): mean ALC per model; paired diff vs ref with SE over pairs.
    wide = df.pivot_table(index=["bench", "k", "run", "pair"], columns="model", values="alc")
    models = list(wide.columns)
    summary = []
    for (bench, k), g in wide.groupby(level=[0, 1]):
        line = {"bench": bench, "k": k, "n": len(g)}
        for m in models:
            d = g[m] - g[args.ref]
            line[m] = f"{g[m].mean():.4f} ({d.mean():+.4f}±{d.std() / np.sqrt(len(d)):.4f})" if m != args.ref else f"{g[m].mean():.4f}"
        summary.append(line)
    pd.set_option("display.width", 300)
    print(pd.DataFrame(summary).to_string(index=False))
    macro = wide.groupby(level=[0, 1]).mean().groupby(level=0).mean().mean()
    print("\nmacro (bench-equal, k-averaged):\n" + macro.sort_values().round(4).to_string())
    budgets = df.groupby("model")[[c for c in df.columns if c.startswith("b") and c[1:].isdigit()]].mean()
    print("\nper-budget Brier (pooled over pairs):\n" + budgets.round(4).to_string())


if __name__ == "__main__":
    main()
