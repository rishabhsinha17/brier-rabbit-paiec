"""Leave-two-benchmarks-out replay to measure the shared-level term (tau_M) on multi-benchmark draws.

Each draw holds one pair from each of two held-out benchmarks; the model trains on the remaining three. The total
prior variance of a benchmark level is held fixed (tau_mu_inner^2 + tau_M^2 = tau_mu^2), so a single-benchmark draw
behaves identically and only cross-benchmark sharing differs. Public benchmark levels differ at random, so this
measures the downside of sharing; the upside (a common shift of hidden benchmarks) can only be measured live.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
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
BENCHES = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
BASE = {"m0": -0.5, "tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0}
TAU_MS = [0.0, 0.5, 0.8, 1.0]
_STATE = {}


def _init():
    from paiec.data import load_responses
    from paiec.protocol import index_pairs

    df = load_responses()
    _STATE["df"] = df
    _STATE["raw"] = {b: index_pairs(df[df.bench == b]) for b in BENCHES}


def _run(task):
    from paiec.hier import HierModel, fit_abilities
    from paiec.protocol import materialize, pair_alc, run_protocol, unit

    (a, b), runs = task
    df = _STATE["df"]
    abilities = fit_abilities(df[~df.bench.isin([a, b])])
    rows = []
    for r in range(runs):
        chosen = []
        for bench in (a, b):
            raw = _STATE["raw"][bench]
            key = sorted((rp["key"] for rp in raw), key=lambda k: unit("l2o", a, b, r, k))[0]
            chosen += materialize([rp for rp in raw if rp["key"] == key], size_range=(88, 226), seed=r)
        for tm in TAU_MS:
            params = dict(BASE, tau_M=tm, tau_mu=math.sqrt(max(BASE["tau_mu"] ** 2 - tm ** 2, 1e-6)) if tm else BASE["tau_mu"])
            res = run_protocol(chosen, HierModel(abilities, params).predict, seed=r)
            for key, pr in res.items():
                rows.append({"pair_bench": key.split("|")[0], "combo": f"{a}+{b}", "run": r, "tau_M": tm,
                             "alc": pair_alc(pr), "b0": pr[0]["brier"], "b1": pr[1]["brier"], "b3": pr[3]["brier"]})
    return rows


def main() -> None:
    import pandas as pd

    combos = list(itertools.combinations(BENCHES, 2))
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(combos)), initializer=_init) as pool:
        rows = [row for chunk in pool.map(_run, [(c, 20) for c in combos]) for row in chunk]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "l2o_shared_level.csv"
    df.to_csv(out, index=False)
    wide = df.pivot_table(index=["combo", "run", "pair_bench"], columns="tau_M", values="alc")
    print("mean pair ALC by tau_M (paired):")
    for tm in TAU_MS:
        d = wide[tm] - wide[0.0]
        print(f"  tau_M={tm:.1f}: {wide[tm].mean():.4f}  diff vs 0: {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
    print("\nper held-out benchmark (diff vs tau_M=0):")
    print((wide.sub(wide[0.0], axis=0)).groupby(level=2).mean().round(4).to_string())
    print("\nB1 by tau_M:", df.groupby("tau_M")["b1"].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
