"""Replay the LOBO draws of results/combo_v1.csv and SAVE every evaluation prediction (swarm round 6, calibration).

experiments/paired.py stores per-pair Brier scores only. This script reuses its machinery unchanged (imported, never
edited): `paired._init` (data + FoldData with the chosen item subsample), `paired.build_models` (the 7-model zoo plus
the JSON plug-in specs) and the draw loop of `paired._run` (models built once per (benchmark, k), draws r = 0..runs-1
from FoldData.draws, run_protocol(pairs, model.predict, seed=r)). The only addition is run_protocol's own `record`
hook, which lists every scored evaluation row as (pair, budget, prediction, label). With the default arguments every
model sees the same draws, seeds and evidence as in combo_v1, so its per-pair Brier scores are bit-identical (checked at
the end against the reference CSV).

Per mode (bench and/or pair item subsample) it writes, next to --out:
  <out>_<mode>.parquet      one row per scored evaluation row: model, bench, k, run, pair, item, copy (index of the
                            recorded run of that item), budget, pred, label, seq (position in run_protocol's record)
  <out>_<mode>_scores.csv   paired.py-format pair scores (bench,k,run,pair,model,alc,b0..b31) plus the platform-style
                            per-pair ECE (e0..e31; ten fixed bins, count-weighted, as on the leaderboard), rows, unique
  <out>_<mode>_meta.json    arguments, model specs, row counts, compute seconds per model, versions, check results

Usage (repo root; the queue job cal_preds runs exactly this):
  ../.venv/bin/python swarm/round6/cal_sims/replay_preds.py --modes bench:18,pair:20 --ks 1,2 --out results/cal_preds \
      --analyze 'swarm/round6/cal_sims/out_{mode}'
  bench:18 = combo_v1's draws (all five models chain bitwise); pair:20 = adj_wide's draws (base, smoothed_k2 and
  empirical_mean chain bitwise; the composites are new there). --analyze runs analyze.py on each mode's file.
Smoke test (one benchmark and k):
  PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round6/cal_sims/replay_preds.py \
      --modes bench:6 --ks 1 --benches real_webagents --out results/cal_preds_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import paired  # noqa: E402  experiments/paired.py: single-threaded BLAS defaults, WORKERS from PAIEC_WORKERS

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BUDGETS = (0, 1, 3, 7, 15, 31)
BENCHES = "real_webagents,researchcodebench,multi_swebench,matharena"  # combo_v1's order (order does not change draws)

# The three plug-in arms exactly as in compute_queue/done/combo_v1.json (verified against that file at start-up when it
# exists). smoothed_k2 and empirical_mean come from paired.build_models' zoo.
_SCP = {"class": "paiec.ext.combo:ComboModel", "subjcal": True, "attr_scale": 1.56, "attr_year_max": 2026.4,
        "pool_item": True, "sigma_x": 3.0, "pool_sim": True, "rho_name": 0.45, "rho_h": 0.288}
SPEC = {
    "base": {"prior": "subjects", "m0": -2.0, "damped_newton": True},
    "SCP_combo": {"prior": "subjects", "m0": -2.0, "damped_newton": True, **_SCP},
    "w20_SCP": {"prior": "subjects", "m0": -2.0, "damped_newton": True, "tau_mu": 2.0, **_SCP},
}
MODELS = "base,SCP_combo,w20_SCP,smoothed_k2,empirical_mean"
CHECK = {"bench": "results/combo_v1.csv", "pair": "results/adj_wide.csv"}  # chained reference pair scores per mode


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _init_worker(subsample: str) -> None:
    paired._init(subsample)


def _task(task):
    """One (benchmark, k) cell: paired._run's loop with run_protocol's record hook on the kept models."""
    from paiec.protocol import pair_alc, run_protocol

    bench, k, runs, spec, keep = task
    df = paired._STATE["df"]
    zoo = paired.build_models(df[df.bench != bench], spec)
    models = {name: zoo[name] for name in keep}
    parts, scores = [], []
    secs = {name: 0.0 for name in keep}
    nb = len(BUDGETS)
    for r, pairs in paired._STATE["folds"].draws(bench, k, runs, 0) or []:
        # run_protocol's evaluate() order: budget -> pair -> evaluation item -> recorded run of the item
        tmpl = [(p.key, str(item), j, int(y)) for p in pairs for item in p.eval
                for j, y in enumerate(p.items[item]["labels"])]
        n = len(tmpl)
        t_key = [t[0] for t in tmpl]
        t_item = [t[1] for t in tmpl]
        t_copy = np.fromiter((t[2] for t in tmpl), dtype=np.int16, count=n)
        t_y = np.fromiter((t[3] for t in tmpl), dtype=np.int8, count=n)
        for name, model in models.items():
            rec: list = []
            t0 = time.perf_counter()
            res = run_protocol(pairs, model.predict, seed=r, record=rec)
            secs[name] += time.perf_counter() - t0
            if len(rec) != nb * n or [x[0] for x in rec] != t_key * nb:
                raise RuntimeError(f"record layout changed ({bench} k={k} run={r} {name})")
            r_b = np.fromiter((x[1] for x in rec), dtype=np.int16, count=nb * n)
            r_y = np.fromiter((x[3] for x in rec), dtype=np.int8, count=nb * n)
            if not (np.array_equal(r_b, np.repeat(np.asarray(BUDGETS, dtype=np.int16), n))
                    and np.array_equal(r_y, np.tile(t_y, nb))):
                raise RuntimeError(f"record budgets/labels out of order ({bench} k={k} run={r} {name})")
            parts.append(pd.DataFrame({
                "model": name, "run": np.int16(r), "pair": t_key * nb, "item": t_item * nb,
                "copy": np.tile(t_copy, nb), "budget": r_b.astype(np.int8),
                "pred": np.fromiter((x[2] for x in rec), dtype=np.float64, count=nb * n),
                "label": r_y, "seq": np.arange(nb * n, dtype=np.int32),
            }))
            for key, pr in res.items():  # same pair order and fields as paired._run, plus the platform ECE
                scores.append({"bench": bench, "k": k, "run": r, "pair": key, "model": name, "alc": pair_alc(pr),
                               **{f"b{b}": pr[b]["brier"] for b in BUDGETS},
                               **{f"e{b}": pr[b]["ece"] for b in BUDGETS},
                               "rows": pr[0]["rows"], "unique": pr[0]["unique"]})
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(preds):
        preds.insert(1, "bench", bench)
        preds.insert(2, "k", np.int8(k))
        for c in ("model", "bench", "pair", "item"):
            preds[c] = preds[c].astype("category")
    return bench, k, preds, scores, secs


def check_against(scores: pd.DataFrame, ref_path: Path) -> list[str]:
    """Bitwise comparison of pair scores with a reference paired.py CSV (overlapping keys only)."""
    if not ref_path.exists():
        return [f"check: {ref_path} not found, skipped"]
    ref = pd.read_csv(ref_path, float_precision="round_trip")
    keys = ["bench", "k", "run", "pair", "model"]
    cols = ["alc"] + [f"b{b}" for b in BUDGETS]
    m = scores[keys + cols].merge(ref[keys + cols], on=keys, how="left", suffixes=("", "_ref"))
    out = []
    for model, g in m.groupby("model", sort=False):
        hit = g["alc_ref"].notna().to_numpy()
        if not hit.any():
            out.append(f"check {ref_path.name}: {model}: not in the reference file")
            continue
        a = g.loc[hit, cols].to_numpy(float)
        b = g.loc[hit, [c + "_ref" for c in cols]].to_numpy(float)
        d = np.abs(a - b)
        exact = int((a == b).all(axis=1).sum())
        out.append(f"check {ref_path.name}: {model}: {hit.sum()}/{len(g)} pair-scores matched, bit-identical "
                   f"{exact}/{hit.sum()}, max|diff| {d.max():.3g}" + ("  EXACT" if exact == hit.sum() else "  MISMATCH"))
    return out


def summarize(scores: pd.DataFrame, ref: str) -> str:
    """paired.py's macro (bench-equal, k-averaged) ALC and per-budget Brier pooled over pairs."""
    wide = scores.pivot_table(index=["bench", "k", "run", "pair"], columns="model", values="alc")
    macro = wide.groupby(level=[0, 1]).mean().groupby(level=0).mean().mean()
    lines = ["macro (bench-equal, k-averaged):", macro.sort_values().round(4).to_string()]
    if ref in wide.columns:
        lines.append(f"\nmacro diff vs {ref} (x1e-3):")
        diff = wide.sub(wide[ref], axis=0).groupby(level=[0, 1]).mean().groupby(level=0).mean().mean() * 1e3
        lines.append(diff.round(3).to_string())
    budgets = scores.groupby("model")[[f"b{b}" for b in BUDGETS]].mean()
    eces = scores.groupby("model")[[f"e{b}" for b in BUDGETS]].mean()
    lines += ["\nper-budget Brier (pooled over pairs):", budgets.round(4).to_string(),
              "\nplatform-style ECE per budget (per pair, ten fixed bins; mean over pairs):", eces.round(4).to_string()]
    return "\n".join(lines)


def _verify_spec() -> str:
    job = ROOT / "compute_queue" / "done" / "combo_v1.json"
    if not job.exists():
        return "spec: combo_v1.json not found; embedded spec used as written"
    args = json.loads(job.read_text())["args"]
    extra = json.loads(args[args.index("--extra") + 1])
    bad = [name for name, params in SPEC.items() if extra.get(name) != params]
    if bad:
        raise SystemExit(f"embedded spec differs from combo_v1.json for {bad}; refusing to run")
    return "spec: base / SCP_combo / w20_SCP identical to compute_queue/done/combo_v1.json"


def run_mode(mode: str, runs: int, args, spec: dict, keep: list[str]) -> None:
    tasks = [(b, int(k), runs, spec, keep) for b in args.benches.split(",") for k in args.ks.split(",")]
    workers = min(paired.WORKERS, 8, len(tasks))
    t0 = time.time()
    print(f"[{mode}] {len(tasks)} cells x {runs} runs, models {keep}, {workers} worker(s)", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mode,)) as pool:
        futures = {pool.submit(_task, t): i for i, t in enumerate(tasks)}
        done = {}
        for fut in as_completed(futures):
            res = fut.result()
            done[futures[fut]] = res
            print(f"[{mode}] cell {len(done)}/{len(tasks)} done: {res[0]} k={res[1]} ({len(res[2])} rows) "
                  f"at {time.time() - t0:.0f}s", flush=True)
        results = [done[i] for i in range(len(tasks))]  # task order, so the file layout never depends on timing
    preds = pd.concat([r[2] for r in results if len(r[2])], ignore_index=True)
    for c in ("model", "bench", "pair", "item"):
        preds[c] = preds[c].astype("category")
    scores = pd.DataFrame([row for r in results for row in r[3]])
    secs = {name: round(sum(r[4][name] for r in results), 1) for name in keep}
    base = _path(f"{args.out}_{mode}")
    base.parent.mkdir(parents=True, exist_ok=True)
    pq = base.with_name(base.name + ".parquet")
    try:
        preds.to_parquet(pq, index=False, compression="zstd")
    except Exception as exc:  # pragma: no cover - pyarrow missing or broken: fall back to csv.gz
        print(f"parquet failed ({exc!r}); writing csv.gz", flush=True)
        pq = base.with_name(base.name + ".csv.gz")
        preds.to_csv(pq, index=False, float_format="%.17g")
    scores.to_csv(base.with_name(base.name + "_scores.csv"), index=False)
    ref = CHECK.get(mode) if args.check == "auto" else (args.check or None)
    checks = check_against(scores, _path(ref)) if ref else []
    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": mode, "runs": runs,
        "ks": args.ks, "benches": args.benches, "models": keep, "spec": spec,
        "rows": int(len(preds)), "pair_scores": int(len(scores)),
        "rows_per_model": preds.groupby("model", observed=True).size().astype(int).to_dict(),
        "compute_seconds_per_model": secs, "wall_seconds": round(time.time() - t0, 1),
        "preds_file": str(pq.relative_to(ROOT)) if pq.is_relative_to(ROOT) else str(pq),
        "preds_sha256": hashlib.sha256(pq.read_bytes()).hexdigest(),
        "checks": checks, "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "command": " ".join(sys.argv),
    }
    base.with_name(base.name + "_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{mode}] {len(preds)} prediction rows, {len(scores)} pair-scores in {time.time() - t0:.0f}s -> {pq}")
    print(f"[{mode}] compute seconds per model: {secs}")
    print(summarize(scores, "base"))
    for line in checks:
        print(line)
    print(flush=True)
    if args.analyze:  # predictions are already on disk; an analysis failure never loses them
        outdir = args.analyze.format(mode=mode)
        cmd = [sys.executable, str(Path(__file__).with_name("analyze.py")), "--preds", str(pq),
               "--ref-scores", str(_path(ref)) if ref else "", "--outdir", outdir]
        print(f"[{mode}] analysis: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(proc.stdout + (f"\n[{mode}] analysis FAILED (exit {proc.returncode}):\n{proc.stderr}"
                             if proc.returncode else ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", default="bench:18",
                    help="comma list of subsample:runs, e.g. bench:18 (combo_v1's draws) or bench:18,pair:20 "
                         "(pair:20 = adj_wide's draws)")
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--benches", default=BENCHES)
    ap.add_argument("--models", default=MODELS, help="names from paired.py's zoo or this script's SPEC")
    ap.add_argument("--extra", default="", help="optional JSON of extra plug-in specs (paired.py --extra format)")
    ap.add_argument("--out", default="results/cal_preds", help="output stem; _<mode>.parquet etc. are appended")
    ap.add_argument("--check", default="auto",
                    help="reference pair-score CSV for the bitwise check ('auto' = combo_v1 for bench, adj_wide for "
                         "pair; '' = skip)")
    ap.add_argument("--analyze", default="",
                    help="after each mode, run analyze.py into this outdir ({mode} is substituted), e.g. "
                         "swarm/round6/cal_sims/out_{mode}; '' = skip")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="skip a later mode when minutes used + the previous mode's minutes exceed this (0 = off)")
    args = ap.parse_args()
    print(_verify_spec(), flush=True)
    spec = dict(SPEC, **(json.loads(args.extra) if args.extra else {}))
    keep = [m.strip() for m in args.models.split(",") if m.strip()]
    zoo = {"const_0.5", "empirical_mean", "smoothed_k2", "pooled_kb8_kp4", "v1_default", "v1_tuned", "v1_subjects"}
    unknown = [m for m in keep if m not in spec and m not in zoo]
    if unknown:
        raise SystemExit(f"unknown models {unknown}")
    spec = {name: params for name, params in spec.items() if name in keep}
    t_start, last = time.time(), 0.0
    for i, item in enumerate(args.modes.split(",")):
        mode, _, runs = item.partition(":")
        used = (time.time() - t_start) / 60
        if i and args.max_minutes and used + last > args.max_minutes:  # the next mode would not fit the budget
            print(f"[{mode}] SKIPPED: {used:.1f} min used, the previous mode took {last:.1f} min, budget "
                  f"{args.max_minutes} min (rerun it alone with --modes {item.strip()})", flush=True)
            continue
        t_mode = time.time()
        run_mode(mode.strip(), int(runs or 18), args, spec, keep)
        last = (time.time() - t_mode) / 60


if __name__ == "__main__":
    main()
