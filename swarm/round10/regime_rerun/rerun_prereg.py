"""Round-10 rerun of the round-2 regime test on the PRE-REGISTERED connector set (no adjudicator file is edited).

The round-2 fit (swarm/round2/adjudicator_sims, 2026-10-06) used 77 formative logs: 22 ours, 52 aghasalim and 3
doctawho42, including doctawho42 run 1 (arm dw_legacyP), which the pre-registered public set of submissions/QUEUE.md
excludes (pub_aghasalim_v1, v2, v3A, v3B, v3C, pub_doctawho42_hiership). This wrapper reruns the same chain with that
one log dropped and nothing else changed: the same 22 own logs (ids <= 963848, the set present at the round-2 export),
the same 52 + 2 public logs, the same bootstrap seed, the same scripts imported and monkeypatched for paths only.
Steps (all output under swarm/round10/regime_rerun/):
 0. check: rebuild the 77-log evidence and compare it with draws_all.npz (point estimates must match exactly)
 1. live_evidence.export_draws on the 76-log set -> draws_prereg.npz, plus the global B31 summaries
 2. regime_test3.main() on it (named regimes, 315-point grid, predicted vs observed tables)
 3. regime_subject.main() and regime_dtheta.main() on it (the H_cal joint fit)
 4. extras: leave-one-arm-out check of the R1 verdict; predicted vs observed of the 52 contrasts (CSV + PNG)
Compliance: anonymous IDs are opaque join keys; only arm-level contrasts and global summaries are printed.
usage: nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round10/regime_rerun/rerun_prereg.py [--smoke]
"""
from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ADJ = ROOT / "swarm" / "round2" / "adjudicator_sims"
OUT = Path(__file__).resolve().parent
SMOKE = "--smoke" in sys.argv
sys.argv = [a for a in sys.argv if a != "--smoke"]
sys.path.insert(0, str(ADJ))
import live_evidence as le  # noqa: E402
import swarm_evalsci as ev  # noqa: E402  (tools/, put on the path by live_evidence)

SEED = 20261006                      # tools/swarm_evalsci.RNG seed; re-seeded before every export (fresh-process state)
DROP = ("dw_run1",)                  # doctawho42 run 1 = dw_legacyP, outside the pre-registered public set
OURS_22 = sorted(f.stem for f in le.load.__globals__["FEEDBACK"].glob("*") if f.stem.isdigit() and int(f.stem) <= 963848)
NPZ = OUT / "draws_prereg.npz"
_orig_load_all = le.load_all
_filter = {"drop": (), "ours": OURS_22}


def load_all_filtered(include_competitors: bool = True):
    df, arm_of = _orig_load_all(include_competitors)
    keep = ~df["submission"].isin(_filter["drop"])
    if _filter["ours"] is not None:
        keep &= (df["team"] != "ours") | df["submission"].isin(_filter["ours"])
    return df[keep].reset_index(drop=True), arm_of


le.load_all = load_all_filtered     # export_draws and wide() see the filtered evidence


class Tee:
    def __init__(self, path: Path):
        self.f, self.o = open(path, "w"), sys.__stdout__

    def write(self, s):
        self.f.write(s)
        self.o.write(s)

    def flush(self):
        self.f.flush()
        self.o.flush()


def b31_summary(w: pd.DataFrame) -> dict:
    """Global B31 summaries only: distinct-pair mean (pair B31 = mean over its sightings) and the benchmark-balanced
    mean (equal weight per benchmark) with SE = sqrt(sum_b var_b / n_b) / B. No per-benchmark value is returned."""
    p = w.groupby("key").agg(b31=("b31", "mean"), bench=("benchmark_id", "first"))
    g = p.groupby("bench")["b31"]
    m, v, n = g.mean(), g.var(ddof=1), g.size()
    return {"pair_mean": float(p.b31.mean()), "pair_se": float(p.b31.std(ddof=1) / np.sqrt(len(p))),
            "bal_mean": float(m.mean()), "bal_se": float(np.sqrt((v / n).sum()) / len(m)), "pairs": int(len(p)),
            "benchmarks": int(len(m))}


def evidence(drop, ours):
    _filter.update(drop=drop, ours=ours)
    df, arm_of = le.load_all(True)
    w = le.wide(df, arm_of)
    return w, le.snapshot_check(w), b31_summary(w)


def step0_check(reps: int) -> dict:
    """Rebuild the round-2 77-log evidence (dw_run1 kept) and compare with draws_all.npz."""
    w, snap, b31 = evidence((), OURS_22)
    print("[step 0] 77-log rebuild snapshot:", json.dumps(snap, default=str))
    print("[step 0] 77-log B31 summaries (global):", json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                                                  for k, v in b31.items()}))
    ev.RNG = np.random.default_rng(SEED)
    new = le.export_draws(reps, True, None)
    old = np.load(ADJ / "draws_all.npz", allow_pickle=True)
    res = {"arms_equal": list(new["arms"]) == list(old["arms"])}
    for k in ("alone_point", "paircov_point", "k_alone", "k_any", "n_pairs_alone", "n_pairs_any", "mean_b31"):
        a, b = np.asarray(new[k], float), np.asarray(old[k], float)
        res[k] = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else f"shape {a.shape} vs {b.shape}"
    if reps == old["alone_draws"].shape[0]:
        res["alone_draws"] = float(np.nanmax(np.abs(new["alone_draws"] - old["alone_draws"])))
        res["paircov_draws"] = float(np.nanmax(np.abs(new["paircov_draws"] - old["paircov_draws"])))
    ok = res["arms_equal"] and all(isinstance(v, float) and v < 1e-9 for k, v in res.items() if k != "arms_equal")
    print("[step 0] reproduction of draws_all.npz:", json.dumps(res), "->", "EXACT" if ok else "MISMATCH")
    return {"snapshot": snap, "b31": b31, "repro": res, "ok": ok}


def step1_export(reps: int) -> dict:
    w, snap, b31 = evidence(DROP, OURS_22)
    print("[step 1] 76-log (pre-registered connector set) snapshot:", json.dumps(snap, default=str))
    print("[step 1] 76-log B31 summaries (global):", json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                                                  for k, v in b31.items()}))
    ev.RNG = np.random.default_rng(SEED)
    out = le.export_draws(reps, True, NPZ)
    print(f"[step 1] wrote {NPZ.name}: arms {list(out['arms'])}, draws {out['alone_draws'].shape}")
    return {"snapshot": snap, "b31": b31}


def regime_modules():
    """Import the adjudicator's regime scripts and redirect their outputs to OUT (paths only; no file is edited)."""
    import regime_test3 as r3
    import regime_variants as rv
    import regime_subject as rs
    import regime_dtheta as rd
    r3.HERE = rs.HERE = rd.HERE = OUT            # regime3_*_draws_prereg.csv, regime_subject_grid.csv, regime_dtheta_grid.csv
    _V = rv.V

    class V(_V):                                 # regime_subject/dtheta hard-code HERE/draws_all.npz: load NPZ instead
        def __init__(self, path, *a, **k):
            super().__init__(NPZ, *a, **k)
    rv.V = V
    return r3, rv, rs, rd


def run_main(mod, name: str, argv=None):
    sys.argv = [sys.argv[0]] + (argv or [])
    t0 = time.time()
    with contextlib.redirect_stdout(Tee(OUT / f"{name}_prereg.txt")):
        mod.main()
    print(f"[{name}] done in {time.time() - t0:.0f} s -> {name}_prereg.txt")


def step4_extras(r3, rv) -> dict:
    """(a) leave-one-arm-out check of the named-regime verdicts (d_total vs the best grid point of each subset);
    (b) predicted vs observed of the 52 hub-referenced contrasts at the best grid point (CSV + PNG)."""
    D = r3.Data(NPZ)
    grid = [(p, D.pred({k: v for k, v in p.items() if not k.startswith("_")})) for p in rv.grid_pools()]
    named = {k: D.pred(p) for k, p in rv.REPS.items()}

    def fit(arms):
        tot = [sum(D.chi2(pr, arms=arms)[:2]) for _, pr in grid]
        j = int(np.argmin(tot))
        row = {"best_total": tot[j], "best_shift": grid[j][0]["shift"], "best_scale": grid[j][0]["scale"],
               "best_pi_low": grid[j][0]["pi_low"], "best_low_rate": grid[j][0]["_low_rate"]}
        for k, pr in named.items():
            row[k] = sum(D.chi2(pr, arms=arms)[:2]) - tot[j]
        return row
    rows = [dict(left_out="(none)", **fit(D.arms))]
    for a in D.arms:
        rows.append(dict(left_out=a, **fit([x for x in D.arms if x != a])))
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "loao_prereg.csv", index=False)
    print("\n[step 4a] leave-one-arm-out: d_total of the named regimes vs the best grid point of each subset")
    print(L.round(1).to_string(index=False))
    b = rows[0]
    pool = {"shift": b["best_shift"], "scale": b["best_scale"], "pi_low": b["best_pi_low"],
            "mu_low": r3.LOGIT(b["best_low_rate"]), "sd_low": 0.5 if b["best_low_rate"] > 0.02 else 0.0}
    secs = ["b0", "b1", "b3", "b7"]
    pr = D.pred(pool)
    mu, md = D.mu_md(pr, D.arms, secs)
    y, X = D.hub_obs(D.arms, secs)
    se = np.nanstd(X, axis=0, ddof=1)
    T = pd.DataFrame({"arm": np.repeat(D.arms, len(secs)), "sec": secs * len(D.arms), "obs": 1e3 * y, "pred": 1e3 * mu,
                      "se_boot": 1e3 * se, "md": 1e3 * md})
    T["z"] = (T.obs - T.pred) / np.sqrt(T.se_boot ** 2 + T.md ** 2)
    T.to_csv(OUT / "pred_vs_obs_best_prereg.csv", index=False)
    print(f"\n[step 4b] 52 contrasts at the best point {pool}: |z| max {T.z.abs().max():.2f}, "
          f"mean z^2 {np.mean(T.z ** 2):.2f}, arms with |z| > 2: {sorted(set(T[T.z.abs() > 2].arm))}")
    return {"loao": rows, "best_pool": {k: float(v) for k, v in pool.items()}, "pred_vs_obs": T}


def plot(T: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for s, m in zip(("b0", "b1", "b3", "b7"), "osD^"):
        t = T[T.sec == s]
        ax.errorbar(t.pred, t.obs, yerr=np.sqrt(t.se_boot ** 2 + t.md ** 2), fmt=m, ms=4, lw=0.8, capsize=2, label=s.upper())
    lim = [min(T.pred.min(), T.obs.min()) - 5, max(T.pred.max(), T.obs.max()) + 5]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("predicted contrast vs hub (x1e-3 Brier)")
    ax.set_ylabel("observed alone-FE contrast vs hub (x1e-3)")
    ax.set_title("Regime test, pre-registered connector set: best grid point", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    pd.set_option("display.width", 260)
    t0 = time.time()
    reps = 60 if SMOKE else 600
    print(f"rerun_prereg: smoke={SMOKE}, reps={reps}, own logs {len(OURS_22)} (<= 963848), drop {DROP}")
    summary = {"smoke": SMOKE, "reps": reps, "own_logs": OURS_22, "drop": list(DROP)}
    summary["step0"] = step0_check(reps if not SMOKE else 5)
    summary["step1"] = step1_export(reps)
    r3, rv, rs, rd = regime_modules()
    run_main(r3, "regime_test3", [str(NPZ)])
    if not SMOKE:
        run_main(rs, "regime_subject")
        run_main(rd, "regime_dtheta")
    ex = step4_extras(r3, rv)
    try:
        plot(ex["pred_vs_obs"], OUT / "pred_vs_obs_best_prereg.png")
    except Exception as e:  # the plot is a convenience; never fail the job on it
        print("[step 4b] plot skipped:", repr(e))
    old = pd.read_csv(ADJ / "regime3_named_draws_all.csv").set_index("regime")
    new = pd.read_csv(OUT / "regime3_named_draws_prereg.csv").set_index("regime")
    g_old = pd.read_csv(ADJ / "regime3_grid_draws_all.csv").sort_values("total").iloc[0]
    g_new = pd.read_csv(OUT / "regime3_grid_draws_prereg.csv").sort_values("total").iloc[0]
    cmp = pd.DataFrame({"total_round2": old["total"], "total_prereg": new["total"],
                        "dchi2_round2": old["total"] - g_old["total"], "dchi2_prereg": new["total"] - g_new["total"],
                        "contrasts_only_round2": old["chi2"] - old["chi2"].min(),
                        "contrasts_only_prereg": new["chi2"] - new["chi2"].min()}).round(1)
    print("\n== SUMMARY: named regimes, delta chi2 from the best grid point (round-2 77 logs vs pre-registered 76 logs)")
    print(cmp.sort_values("dchi2_prereg").to_string())
    for tag, g in (("round-2", g_old), ("prereg", g_new)):
        print(f"best grid point {tag}: total {g['total']:.1f} (contrasts {g['chi2']:.1f} + B31 {g['chi2_b31']:.1f}) at shift "
              f"{g['shift']}, scale {g['scale']}, pi_low {g['pi_low']} @ {g['low_rate']}; mean r {g['mean_r']:.3f}, "
              f"r(1-r) {g['mean_rr']:.3f}, P(r<5%) {g['p_lt05']:.3f}")
    summary["named"] = cmp.reset_index().to_dict("records")
    summary["grid_best"] = {"round2": g_old.to_dict(), "prereg": g_new.to_dict()}
    summary["loao"] = ex["loao"]
    summary["best_pool"] = ex["best_pool"]
    summary["seconds"] = round(time.time() - t0)
    (OUT / "summary_prereg.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\nrerun_prereg finished in {summary['seconds']} s; summary_prereg.json written")


if __name__ == "__main__":
    main()
