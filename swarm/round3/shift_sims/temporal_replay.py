"""Full-model replay of the temporal hold-out (validates shift_screen.py temporal, fixed world).

For cutoff T and each LOBO benchmark b: fit_subject_prior on the OTHER benchmarks using only subjects released before T
(strict: undated subjects excluded), plus the provider-free ridge refit on the same rows. Every NEW pair of b
(release >= T, >= 80 items) is replayed alone (k = 1) with the real protocol (platform-sized item subsample, sha256
acquisition, item_features groups, repeated trials, damped-Newton Laplace HierModel) for `seeds` seeds.
Models: REF (cur_final params, m0 = the old-cohort optimum of the exact screen), A156 (SubjCalHier 1.56 + clamp 2026.4),
A156+NP (ShiftNPHier with the fold's provider-free coefficients), A156+SY1.0 (year term unscaled).
usage: python temporal_replay.py --Ts 2025.5,2026.0 --seeds 3 --out results/shift_temporal_replay.csv [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paiec.data import load_responses  # noqa: E402
from paiec.evaluate import SIZE_RANGE  # noqa: E402
from paiec.ext.pd1_subjcal import SubjCalHier  # noqa: E402
from paiec.hier import HierModel, fit_subject_prior  # noqa: E402
from paiec.protocol import BUDGETS, index_pairs, materialize, pair_alc, run_protocol  # noqa: E402
from paiec.subjects import _year  # noqa: E402
from shift_np import ShiftNPHier  # noqa: E402
import variants as vr  # noqa: E402

LIVE = json.loads((ROOT / "submission_src" / "v3b_noM_n_m0n20" / "params.json").read_text())
M0 = {2025.5: -1.47, 2026.0: -1.09}  # old-cohort ALC optimum of REF in shift_screen.py temporal (bracket mean)
BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")


def build_models(prior, coef_np, var_np, m0):
    base = dict(LIVE, m0=m0)
    a156 = dict(base, attr_scale=1.56, attr_year_max=2026.4)
    return {"REF": HierModel(prior, base), "A156": SubjCalHier(prior, a156),
            "A156+NP": ShiftNPHier(prior, dict(a156, attr_coef=[float(c) for c in coef_np], attr_var=float(var_np))),
            "A156+SY1.0": ShiftNPHier(prior, dict(a156, attr_year_mult=1.0))}


_STATE = {}


def _init():
    df = load_responses()
    rel_pair = df.drop_duplicates("pair").set_index("pair")["release_date"].map(_year)
    _STATE.update(df=df, rel_pair=rel_pair, rel=df.pair.map(rel_pair))


def _task(task):
    T, b, seeds, limit = task
    df, rel_pair, rel = _STATE["df"], _STATE["rel_pair"], _STATE["rel"]
    keep_old = rel.notna() & (rel < T)
    train = df[(df.bench != b) & keep_old]
    prior = fit_subject_prior(train)
    X, y, w = vr.ridge_design(train)
    alpha = np.ones(X.shape[1]); alpha[0] = 0.0; alpha[vr.PROV_IDX] = np.inf
    coef_np, var_np = vr.ridge_alpha(X, y, w, alpha)
    raw = [rp for rp in index_pairs(df[df.bench == b]) if pd.notna(rel_pair.get(rp["key"])) and rel_pair[rp["key"]] >= T]
    if limit:
        raw = raw[:limit]
    rows = []
    for rp in raw:
        for seed in range(seeds):
            pairs = materialize([rp], size_range=SIZE_RANGE, seed=seed)
            for name, model in build_models(prior, coef_np, var_np, M0.get(T, -1.5)).items():
                res = run_protocol(pairs, model.predict, seed=seed)
                for key, pr in res.items():
                    rows.append({"T": T, "bench": b, "pair": key, "seed": seed, "model": name, "alc": pair_alc(pr),
                                 **{f"b{bb}": pr[bb]["brier"] for bb in BUDGETS}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", default="2025.5,2026.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--limit", type=int, default=0, help="max new pairs per benchmark (0 = all; smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "results" / "shift_temporal_replay.csv"))
    args = ap.parse_args()
    t0 = time.time()
    tasks = [(float(T), b, args.seeds, args.limit) for T in args.Ts.split(",") for b in args.benches.split(",")]
    workers = max(1, min(int(os.environ.get("PAIEC_WORKERS", "2")), len(tasks)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        rows = [row for chunk in pool.map(_task, tasks) for row in chunk]
    R = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(args.out, index=False)
    cols = ["alc"] + [f"b{bb}" for bb in BUDGETS]
    W = R.pivot_table(index=["T", "bench", "pair", "seed"], columns="model", values=cols)
    print(f"{len(R)} rows in {time.time() - t0:.0f}s -> {args.out}")
    print("variant minus REF, x1e-3 (mean over new-pair x seed runs; SE clustered by pair)")
    for T, g in W.groupby(level="T"):
        for m in ("A156", "A156+NP", "A156+SY1.0"):
            line = []
            for c in cols:
                d = (g[(c, m)] - g[(c, "REF")]) * 1e3
                per_pair = d.groupby(level="pair").mean()
                se = per_pair.std(ddof=1) / np.sqrt(len(per_pair)) if len(per_pair) > 1 else float("nan")
                line.append(f"{c} {d.mean():+.2f}" + (f" ± {se:.2f}" if c == "alc" else ""))
            byb = ((g[("alc", m)] - g[("alc", "REF")]) * 1e3).groupby(level="bench").mean().round(2).to_dict()
            print(f"T={T} {m:11s} n={len(g)} | " + " | ".join(line) + f" | per bench {byb}")
    print("pre-registered (exact screen, fixed world): T=2025.5 A156 -4.19, A156+NP -5.95, A156+SY1.0 -1.00; "
          "T=2026.0 A156 -4.55, A156+NP -7.33, A156+SY1.0 -0.42 (x1e-3)")


if __name__ == "__main__":
    main()
