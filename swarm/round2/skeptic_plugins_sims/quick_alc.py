"""Paired LOBO quick screen on the cached public pairs (same draws, priors and params as experiments/paired.py,
subsample "pair"): ALC and per-budget Brier differences vs a reference model, SE over pair-draws.
usage: python quick_alc.py '<bench:k:runs,...>' '<{"name": [plugin key or null, {param overrides on cur_final}], ...}>' [ref]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, PLUGINS, draws, load_cache, make  # noqa: E402

import numpy as np  # noqa: E402

from paiec.protocol import BUDGETS, pair_alc, run_protocol  # noqa: E402


def main():
    spec_runs = [s.split(":") for s in sys.argv[1].split(",")]
    models = json.loads(sys.argv[2])
    ref = sys.argv[3] if len(sys.argv) > 3 else next(iter(models))
    _, priors = load_cache()
    t0 = time.time()
    rows = []
    for bench, k, runs in spec_runs:
        ms = {n: make(PLUGINS.get(key) if key else None, priors[bench], dict(BASE, **over)) for n, (key, over) in models.items()}
        for r, pairs in draws(bench, int(k), int(runs)):
            for n, m in ms.items():
                res = run_protocol(pairs, m.predict, seed=r)
                for pk, pr in res.items():
                    rows.append((bench, int(k), r, pk, n, pair_alc(pr), *[pr[b]["brier"] for b in BUDGETS]))
    import pandas as pd

    df = pd.DataFrame(rows, columns=["bench", "k", "run", "pair", "model", "alc", *[f"b{b}" for b in BUDGETS]])
    w = df.pivot_table(index=["bench", "k", "run", "pair"], columns="model", values=["alc", *[f"b{b}" for b in BUDGETS]])
    print(f"{len(w)} pair-draws ({time.time() - t0:.0f}s); diffs x1e-3 vs {ref}, SE over pair-draws")
    for (bench, k), g in w.groupby(level=[0, 1]):
        for n in models:
            if n == ref:
                continue
            d = g[("alc", n)] - g[("alc", ref)]
            pb = " ".join(f"B{b} {1e3 * (g[(f'b{b}', n)] - g[(f'b{b}', ref)]).mean():+.2f}" for b in BUDGETS)
            print(f"  {bench:18s} k={k} n={len(d):3d} {n:16s} ALC {1e3 * d.mean():+.2f} +- {1e3 * d.std() / np.sqrt(len(d)):.2f} | {pb}"
                  f" | ref ALC {g[('alc', ref)].mean():.4f}")


if __name__ == "__main__":
    main()
