"""Coordinate search over v1 structural parameters (incl. tau_mu, how fast the benchmark level adapts) with LOBO replay, folds evaluated in parallel processes.

The benchmark-level prior mean m0 is passed in, not tuned here: five public benchmarks cannot tell us where hidden
benchmarks sit, so tuning m0 on them would just memorise the public mix. Its width tau_mu is tuned, because it sets
how fast the level adapts to revealed labels.
"""
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

GRID = {
    "tau_mu": [0.8, 1.2, 1.6, 2.0],
    "lam": [0.6, 1.0, 1.3, 1.6],
    "tau_delta": [0.4, 0.7, 1.0, 1.4],
    "sigma_i": [0.6, 1.2, 1.8, 2.4],
    "sigma_g": [0.0001, 0.3, 0.6, 1.0],
    "tau_phi": [0.0001, 0.3, 0.6],
    "tau_theta": [0.3, 0.6, 1.0],
}
_STATE = {}
_PRIOR_KIND = __import__('os').environ.get('PAIEC_PRIOR', 'subjects')


def _init():
    from paiec.data import BINARY_BENCHMARKS, load_responses
    from paiec.evaluate import FoldData
    from paiec.hier import fit_abilities, fit_subject_prior

    df = load_responses()
    _STATE["folds"] = FoldData(df, BINARY_BENCHMARKS)
    fit = fit_subject_prior if _PRIOR_KIND == "subjects" else fit_abilities
    _STATE["abilities"] = {b: fit(df[df.bench != b]) for b in BINARY_BENCHMARKS}


def _eval_fold(task):
    from paiec.evaluate import replay
    from paiec.hier import HierModel

    bench, params, ks, runs, seed0 = task
    model = HierModel(_STATE["abilities"][bench], params)
    return bench, replay(_STATE["folds"], bench, model.predict, ks=ks, runs=runs, seed0=seed0)


def objective(pool, benches, params, ks, runs, seed0):
    out = dict(pool.map(_eval_fold, [(b, params, ks, runs, seed0) for b in benches]))
    per_bench = {b: float(np.mean([out[b][k]["alc"] for k in ks if k in out[b]])) for b in benches}
    return float(np.mean(list(per_bench.values()))), per_bench, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--benches", default="matharena,multi_swebench,real_webagents,researchcodebench")
    ap.add_argument("--start", default='{"m0": -0.5, "tau_mu": 1.2}')
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "tune_hier.json"))
    args = ap.parse_args()
    ks = tuple(int(k) for k in args.ks.split(","))
    benches = args.benches.split(",")
    params = json.loads(args.start)
    t0 = time.time()
    history = []
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(benches)), initializer=_init) as pool:
        best, per_bench, _ = objective(pool, benches, params, ks, args.runs, 0)
        print(f"start {best:.4f} {per_bench} ({time.time() - t0:.0f}s)", flush=True)
        for sweep in range(args.sweeps):
            for name, values in GRID.items():
                for v in values:
                    if params.get(name) == v:
                        continue
                    trial = dict(params, **{name: v})
                    score, pb, _ = objective(pool, benches, trial, ks, args.runs, 0)
                    history.append({"params": trial, "score": score, "per_bench": pb})
                    flag = ""
                    if score < best - 1e-4:
                        best, params, per_bench, flag = score, trial, pb, "  <- best"
                    print(f"sweep{sweep} {name}={v}: {score:.4f}{flag} ({time.time() - t0:.0f}s)", flush=True)
        # Fresh seeds for an unbiased read of the chosen setting.
        final, pb, detail = objective(pool, benches, params, ks, args.runs, 1000)
    print(f"chosen {json.dumps(params)}\nfresh-seed score {final:.4f} {pb}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"chosen": params, "fresh_score": final, "fresh_per_bench": pb,
                                          "fresh_detail": detail, "history": history}, indent=1, default=str))


if __name__ == "__main__":
    main()
