"""Round 12 local guard (submissions/QUEUE.md, round-12 amendment item 5): replay w20_SCP, F0 and F2 on identical draws, with
every prior fitted without the held-out benchmark. L1 = the four original folds with the combo_v1 settings; L2 = the wide folds
listed in swarm/round12/wide_folds.csv. One csv per (benchmark, k) under swarm/round12/results/<set>/, so a stopped run resumes.

usage (from brier_rabbit/):
  PAIEC_WORKERS=2 nice -n 19 ../.venv/bin/python swarm/round12/replay.py L1
  PAIEC_WORKERS=2 nice -n 19 ../.venv/bin/python swarm/round12/replay.py L2
  ../.venv/bin/python swarm/round12/replay.py read       # the guard reading of everything finished so far
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "swarm" / "round12"))
from experiments.paired import V1_TUNED  # noqa: E402
from paiec.data import SUBJECT_FIELDS, _text, benchmark_alias, load_responses  # noqa: E402
from paiec.evaluate import FoldData  # noqa: E402
from paiec.ext.combo import ComboModel  # noqa: E402
from paiec.hier import fit_subject_prior  # noqa: E402
from paiec.protocol import BUDGETS, pair_alc, run_protocol  # noqa: E402
import full_prior  # noqa: E402

WORKERS = int(os.environ.get("PAIEC_WORKERS", "2"))
MAIN = ROOT.parent / "data" / "measurement-db-main"
OUT = ROOT / "swarm" / "round12"
CACHE = OUT / "cache"
W20_SCP = {"m0": -2.0, "damped_newton": True, "tau_mu": 2.0, "subjcal": True, "attr_scale": 1.56, "attr_year_max": 2026.4,
           "pool_item": True, "sigma_x": 3.0, "pool_sim": True, "rho_name": 0.45, "rho_h": 0.288}
ORIGINAL = ("real_webagents", "researchcodebench", "multi_swebench", "matharena")  # the order of combo_v1
SETS = {"L1": {"runs": 18, "ks": (1, 2), "arms": ("F0", "F2"), "folds": None},
        "L2": {"runs": 6, "ks": (1, 2), "arms": ("F0", "F2"), "folds": "wide_folds.csv", "max_items": 2000},
        "L1E": {"runs": 18, "ks": (1, 2), "arms": ("E", "E2"), "folds": None},
        "L2b": {"runs": 6, "ks": (1, 2), "arms": ("E", "E2"), "folds": "wide_folds_b.csv", "max_items": 6000},
        "L1F": {"runs": 18, "ks": (1, 2), "arms": ("E3",), "folds": None},
        "L2c": {"runs": 6, "ks": (1, 2), "arms": ("E3",), "folds": "wide_folds_c.csv", "max_items": 6000}}
MAX_PAIRS = 40  # wide folds: enough for 6 runs at k 1 and 2; memory, not selection (same subset for all arms)


def rank(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def load_wide(bench: str, max_items: int) -> pd.DataFrame:
    """One new benchmark in the shape of paiec.data.load_responses."""
    base = MAIN / bench / "formatted_tables"
    resp = pd.read_parquet(base / "responses.parquet",
                           columns=["subject_id", "item_id", "trial", "test_condition", "interactors", "response"])
    resp = resp[resp["response"].isin([0.0, 1.0])]
    sizes = resp.groupby("subject_id")["item_id"].nunique()
    subj_ids = sorted(sizes[sizes >= 80].index, key=lambda s: rank("wide-pair", bench, s))[:MAX_PAIRS]
    resp = resp[resp["subject_id"].isin(subj_ids)]
    item_ids = sorted(resp["item_id"].unique(), key=lambda i: rank("wide-item", bench, i))[:max_items]
    resp = resp[resp["item_id"].isin(set(item_ids))]
    items = pd.read_parquet(base / "items.parquet", columns=["item_id", "content", "item_features"])
    subjects = pd.read_parquet(base / "subjects.parquet")
    df = resp.merge(items, on="item_id", how="left").merge(subjects, on="subject_id", how="left")
    for field in SUBJECT_FIELDS:
        df[field] = df[field].map(lambda v: _text(v).strip()) if field in df else ""
    df["item_content"] = df["content"].map(_text)
    df["item_features"] = df["item_features"].map(lambda v: _text(v).strip())
    df["interactors"] = df["interactors"].map(lambda v: _text(v).strip())
    df["bench"] = bench
    df["benchmark_id"] = benchmark_alias(bench)
    df["label"] = df["response"].astype("int8")
    df["pair"] = bench + "|" + df["subject_id"].astype(str)
    return df[["bench", "benchmark_id", "pair", "subject_id", "display_name", "item_id", "trial", "test_condition", "label",
               "item_content", "item_features", "interactors", *SUBJECT_FIELDS]]


def prepare_original() -> None:
    """Per-benchmark tables and leave-one-out reference priors of the original folds (one pass, then cached)."""
    CACHE.mkdir(exist_ok=True)
    df = None
    for bench in (*ORIGINAL, "__none__"):
        prior_path = CACHE / f"ref_prior_{bench}.json"
        if prior_path.exists() and (bench == "__none__" or (CACHE / f"bench_{bench}.pkl").exists()):
            continue
        df = load_responses() if df is None else df
        prior = fit_subject_prior(df[df.bench != bench])
        prior_path.write_text(json.dumps({"names": {k: list(v) for k, v in prior["names"].items()}, "attr": prior["attr"]}))
        if bench != "__none__":
            df[df.bench == bench].to_pickle(CACHE / f"bench_{bench}.pkl")


def priors_for(bench: str, in_original: bool, models: list[str]) -> dict:
    ref = json.loads((CACHE / f"ref_prior_{bench if in_original else '__none__'}.json").read_text())
    out = {}
    if "w20_SCP" in models:
        out["w20_SCP"] = ref
    rows = None
    for name in models:
        if name in ("F0", "F2", "E", "E2", "E3"):
            rows = pd.read_pickle(OUT / "pair_rates.pkl") if rows is None else rows
        if name in ("F0", "F2"):
            out[name] = full_prior.fit(rows, name, exclude=bench)
        elif name in ("E", "E2", "E3"):
            out[name] = full_prior.coverage(ref, full_prior.fit(rows, "F2", exclude=bench), name)
    return out


def run_fold(task) -> str:
    which, bench, k, runs, models = task
    in_original = SETS[which]["folds"] is None
    df = pd.read_pickle(CACHE / f"bench_{bench}.pkl") if in_original else load_wide(bench, SETS[which]["max_items"])
    folds = FoldData(df, [bench], subsample="bench")
    priors = priors_for(bench, in_original, models)
    built = {name: ComboModel(priors[name], dict(V1_TUNED, **W20_SCP)) for name in models}
    rows = []
    t0 = time.time()
    for r, pairs in folds.draws(bench, k, runs, 0) or []:
        for name, model in built.items():
            res = run_protocol(pairs, model.predict, seed=r)
            for key, pr in res.items():
                rows.append({"bench": bench, "k": k, "run": r, "pair": key, "model": name, "alc": pair_alc(pr),
                             **{f"b{b}": pr[b]["brier"] for b in BUDGETS}})
    out = OUT / "results" / which
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{bench}__k{k}.csv", index=False)
    return f"{which} {bench} k{k}: {len(rows)} rows, {time.time() - t0:.0f}s"


def run_set(which: str) -> None:
    spec = SETS[which]
    models = ["w20_SCP", *spec["arms"]]
    prepare_original()
    benches = list(ORIGINAL) if spec["folds"] is None else pd.read_csv(OUT / spec["folds"])["bench"].tolist()
    tasks = [(which, b, k, spec["runs"], models) for b in benches for k in spec["ks"]
             if not (OUT / "results" / which / f"{b}__k{k}.csv").exists()]
    print(f"{which}: {len(tasks)} folds to run with {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed([pool.submit(run_fold, t) for t in tasks]):
            print(fut.result(), flush=True)


def read() -> None:
    """Guard reading: paired difference arm - w20_SCP per benchmark (k averaged), benchmark-equal mean, SE over benchmarks."""
    for which, spec in SETS.items():
        files = sorted((OUT / "results" / which).glob("*.csv")) if (OUT / "results" / which).exists() else []
        if not files:
            print(which, "no results yet")
            continue
        empty = [f.stem for f in files if f.stat().st_size < 5]
        if empty:
            print(which, "folds with no eligible draw:", ", ".join(empty))
        df = pd.concat([pd.read_csv(f) for f in files if f.stat().st_size >= 5], ignore_index=True)
        wide = df.pivot_table(index=["bench", "k", "run", "pair"], columns="model", values="alc")
        per = wide.groupby(level=[0, 1]).mean().groupby(level=0).mean()
        print(f"\n{which}: {per.shape[0]} benchmarks, {len(files)} folds, {len(wide)} pair-scores; ALC x1e-3, arm - w20_SCP")
        for arm in spec["arms"]:
            d = (per[arm] - per["w20_SCP"]) * 1e3
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            print(f"  {arm}: mean {d.mean():+.3f}  SE {se:.3f}  z {d.mean() / se:+.2f}  worst benchmark {d.max():+.3f}  best {d.min():+.3f}")
            print("     per benchmark:", " ".join(f"{v:+.2f}" for v in d.sort_values()))
        budgets = df.groupby("model")[[f"b{b}" for b in BUDGETS]].mean() * 1e3
        print("  mean Brier by budget x1e-3:\n" + budgets.round(2).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "read":
        read()
    else:
        run_set(sys.argv[1])
