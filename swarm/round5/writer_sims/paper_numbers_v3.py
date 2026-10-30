"""Local numbers for report/paper_draft_v3.md (swarm round 5, writer). Reads saved replay CSVs only; no replay is run.

Convention (the one used everywhere in the v3 draft, identical to swarm/round3/report_writer_sims/paper_tables.py):
- pairing: within one CSV every model was scored on identical draws; rows are joined on (bench, k, run, pair);
- per draw (bench, k, run) the per-pair differences are averaged; per (bench, k) cell the mean over draws and
  SE = sd(draw means) / sqrt(#draws) ("draw-clustered");
- macro = mean over benchmarks of the mean over k (benchmark-equal, k-averaged); the macro SE combines cell SEs.
Values are printed in units of 1e-3 Brier ALC (arm minus reference; negative favours the arm).

Usage (from brier_rabbit/):
  PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/writer_sims/paper_numbers_v3.py \
      > swarm/round5/writer_sims/paper_numbers_v3.out
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
R = REPO / "results"
BUD = ("b0", "b1", "b3", "b7", "b15", "b31")
SEC = ("alc",) + BUD
BENCH = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
SHORT = {"matharena": "MA", "multi_swebench": "MSWE", "real_webagents": "REAL", "researchcodebench": "RCB"}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def paired(df: pd.DataFrame, model: str, ref: str, other: pd.DataFrame | None = None, ref_model: str | None = None,
           keys=("bench", "k", "run", "pair")) -> pd.DataFrame:
    """Per-pair differences model - ref (ref may come from another CSV with identical draws)."""
    keys = list(keys)
    a = df[df.model == model].set_index(keys)[list(SEC)]
    src = df if other is None else other
    b = src[src.model == (ref_model or ref)].set_index(keys)[list(SEC)]
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).reset_index()


def summarize(d: pd.DataFrame, ks=None) -> dict:
    if ks is not None:
        d = d[d["k"].isin(ks)]
    per_draw = d.groupby(["bench", "k", "run"])[list(SEC)].mean().reset_index()
    cells = per_draw.groupby(["bench", "k"])
    mean = cells[list(SEC)].mean()
    se = cells[list(SEC)].agg(lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan"))
    benches = mean.index.get_level_values(0).unique()
    nk = {b: int((mean.index.get_level_values(0) == b).sum()) for b in benches}
    w = pd.Series({idx: 1.0 / (len(benches) * nk[idx[0]]) for idx in mean.index})
    macro = {c: float((mean[c] * w).sum()) for c in SEC}
    macro_se = {c: float(math.sqrt(((w ** 2) * (se[c] ** 2)).sum())) for c in SEC}
    per_bench = mean["alc"].groupby(level=0).mean().to_dict()
    per_bench_se = {}
    for b in benches:
        sub = se.loc[b, "alc"]
        per_bench_se[b] = float(math.sqrt((sub ** 2).sum()) / len(sub))
    # pair-level SE (for reference only): SE of the macro if pair-draws were independent
    return {"macro": macro, "se": macro_se, "bench": per_bench, "bench_se": per_bench_se,
            "draws": int(per_draw.shape[0]), "pairs": int(len(d))}


def fmt(r: dict, label: str) -> str:
    m, s = r["macro"], r["se"]
    b = " ".join(f"{SHORT.get(k, k)} {1e3 * r['bench'].get(k, float('nan')):+.2f}" for k in BENCH if k in r["bench"])
    bud = " ".join(f"{x.upper()} {1e3 * m[x]:+.2f}({1e3 * s[x]:.2f})" for x in BUD)
    return (f"{label:<46s} ALC {1e3 * m['alc']:+.3f} ± {1e3 * s['alc']:.3f} | {b} | {bud} | draws {r['draws']} "
            f"pair-draws {r['pairs']}")


def absmacro(df: pd.DataFrame, model: str, ks=None) -> float:
    d = df[df.model == model]
    if ks is not None:
        d = d[d["k"].isin(ks)]
    m = d.groupby(["bench", "k"])["alc"].mean()
    return float(m.groupby(level=0).mean().mean())


def show(df, model, ref, label=None, ks=None, other=None, ref_model=None):
    d = paired(df, model, ref, other=other, ref_model=ref_model)
    r = summarize(d, ks)
    print(fmt(r, label or f"{model} - {ref}" + (f" k={ks}" if ks else "")))
    return r


def section(t: str) -> None:
    print("\n== " + t)


def main() -> None:
    pd.set_option("display.width", 200)

    # ---------------------------------------------------------------- combo_v1: the shipped composite
    cv = load("combo_v1")
    section("combo_v1 (bench-keyed, ks 1,2, 18 runs): absolute macros")
    for m in ("const_0.5", "empirical_mean", "smoothed_k2", "v1_default", "v1_tuned", "v1_subjects", "base",
              "S_subjcal", "C_cav30", "P_simpool", "SCP_combo", "w20_base", "w20_SCP", "l15_base", "l15_SCP"):
        print(f"  {m:<16s} {absmacro(cv, m):.4f}   k1 {absmacro(cv, m, [1]):.4f}   k2 {absmacro(cv, m, [2]):.4f}")
    section("combo_v1 components vs base (REF)")
    for m in ("S_subjcal", "C_cav30", "P_simpool", "SCP_combo"):
        show(cv, m, "base")
    for m in ("S_subjcal", "C_cav30", "SCP_combo"):
        for k in (1, 2):
            show(cv, m, "base", ks=[k])
    section("combo_v1 transfer to other level blocks")
    show(cv, "w20_SCP", "w20_base")
    show(cv, "l15_SCP", "l15_base")
    section("combo_v1 level blocks (no composite / with composite)")
    for m, ref in (("w20_base", "base"), ("l15_base", "base"), ("w20_SCP", "SCP_combo"), ("l15_SCP", "SCP_combo"),
                   ("w20_SCP", "base"), ("l15_SCP", "base")):
        show(cv, m, ref)
        for k in (1, 2):
            show(cv, m, ref, ks=[k])
    section("combo_v1 additivity: SCP - S - C - P + 2 base")
    keys = ["bench", "k", "run", "pair"]
    piv = cv.pivot_table(index=keys, columns="model", values=list(SEC))
    add = pd.DataFrame({c: piv[(c, "SCP_combo")] - piv[(c, "S_subjcal")] - piv[(c, "C_cav30")] - piv[(c, "P_simpool")]
                        + 2 * piv[(c, "base")] for c in SEC}).reset_index()
    print(fmt(summarize(add), "additivity residual"))
    section("combo_v1 ladder on identical bench-keyed draws")
    for m, ref in (("smoothed_k2", "const_0.5"), ("v1_default", "smoothed_k2"), ("v1_tuned", "v1_default"),
                   ("v1_subjects", "v1_tuned"), ("base", "v1_subjects"), ("base", "smoothed_k2"),
                   ("SCP_combo", "smoothed_k2"), ("w20_SCP", "smoothed_k2"), ("l15_SCP", "smoothed_k2")):
        show(cv, m, ref)
    n_fired = paired(cv, "P_simpool", "base")
    print(f"  P changed {(n_fired['alc'].abs() > 0).sum()} of {len(n_fired)} pair-draws "
          f"(benches: {n_fired.loc[n_fired['alc'].abs() > 0, 'bench'].value_counts().to_dict()})")

    # ---------------------------------------------------------------- combo_sy: SY1.0 decision
    cs = load("combo_sy")
    section("combo_sy (combo_v1 draws): reproduction check vs combo_v1")
    for m in ("base", "S_subjcal", "SCP_combo"):
        a = cs[cs.model == m].set_index(keys)["alc"]
        b = cv[cv.model == m].set_index(keys)["alc"]
        com = a.index.intersection(b.index)
        print(f"  {m}: {len(com)} joined rows, max |diff| {float((a.loc[com] - b.loc[com]).abs().max()):.2e}")
    section("combo_sy contrasts")
    show(cs, "S_sy10", "base")
    show(cs, "SCP_sy10", "base")
    show(cs, "SCP_sy10", "SCP_combo")
    show(cs, "S_sy10", "S_subjcal")
    show(cs, "w20_SCP_sy10", "w20_SCP", other=cv, label="w20_SCP_sy10 - w20_SCP (combo_v1)")

    # ---------------------------------------------------------------- combo_np: NP decision
    cn = load("combo_np")
    section("combo_np contrasts (rule: SCP_np - SCP_combo <= +0.2 macro, no benchmark worse than +0.5)")
    r = show(cn, "SCP_np", "SCP_combo")
    print("  per-bench SE: " + ", ".join(f"{SHORT[b]} ±{1e3 * r['bench_se'][b]:.2f}" for b in BENCH))
    show(cn, "S_np", "base")
    show(cn, "base_np", "base")
    show(cn, "SCP_np", "base")

    # ---------------------------------------------------------------- pdpool_bench_k2: decisive cavity replay
    pp = load("pdpool_bench_k2")
    section("pdpool_bench_k2 (bench-keyed, k=2, 20 runs, random partners) vs base")
    for m in ("cav_x1.5", "cav_x2.0", "cav_x2.5", "cav_x3.0", "cav_x4.0", "headA_x2.0", "map_x2.0", "si10", "si15",
              "si20"):
        r = show(pp, m, "base")
        if m in ("cav_x3.0", "map_x2.0", "cav_x2.0"):
            print("    per-bench SE: " + ", ".join(f"{SHORT[b]} ±{1e3 * r['bench_se'][b]:.2f}" for b in BENCH))

    # ---------------------------------------------------------------- old vs new subsample on identical picks
    tw = load("builder_pair_twin_k2")
    section("pair-keyed (old) vs bench-keyed (new) subsample on identical picks (k=2, 20 runs)")
    for m in ("cav_x2.0", "cav_x3.0", "headA_x2.0", "si15", "si20"):
        old = paired(tw, m, "base")
        new = paired(pp, m, "base")
        print(fmt(summarize(old), f"{m} - base, pair-keyed"))
        print(fmt(summarize(new), f"{m} - base, bench-keyed"))
        j = new.set_index(keys)[list(SEC)].sub(old.set_index(keys)[list(SEC)]).dropna().reset_index()
        print(fmt(summarize(j), f"{m}: bench-keyed minus pair-keyed"))

    # ---------------------------------------------------------------- adj_wide: local guard of live level arms
    aw = load("adj_wide")
    section("adj_wide (pair-keyed, ks 1,2, 20 runs) vs base (REF)")
    for m in ("n15D", "w20_m20", "w20_m15", "w25_m25"):
        show(aw, m, "base")
        for k in (1, 2):
            show(aw, m, "base", ks=[k])
    print(f"  base absolute macro {absmacro(aw, 'base'):.4f}")

    # ---------------------------------------------------------------- sigma_g
    sg = load("pdadapt_sigma_g")
    sg = sg[sg.scen == "none"]
    section("pdadapt_sigma_g (pair-keyed, ks 1,2, 20 runs) vs base")
    for m in ("sg14", "sg20"):
        show(sg, m, "base")

    # ---------------------------------------------------------------- simpool (P)
    sp = load("pdadapt_simpool")
    section("pdadapt_simpool: sp (P2 at REF) - base, mean over pair-draws, SE clustered by draw")
    for mode in ("xname", "xrandom", "hsame", "hrandom"):
        s = sp[sp["mode"] == mode]
        a = s[s.model == "sp"].set_index(["combo", "draw", "run", "pair"])[list(SEC)]
        b = s[s.model == "base"].set_index(["combo", "draw", "run", "pair"])[list(SEC)]
        d = (a - b.loc[a.index]).reset_index()
        g = d.groupby(["combo", "draw", "run"])[list(SEC)].mean()
        n = len(g)
        mean = d[list(SEC)].mean()
        se_pair = d["alc"].std(ddof=1) / math.sqrt(len(d))
        se_draw = g["alc"].std(ddof=1) / math.sqrt(n)
        print(f"  {mode:<8s} ALC {1e3 * mean['alc']:+.3f} (pair SE {1e3 * se_pair:.3f}, draw SE {1e3 * se_draw:.3f}; "
              f"{len(d)} pair-draws, {n} draws) | " + " ".join(f"{x.upper()} {1e3 * mean[x]:+.2f}" for x in BUD))

    # ---------------------------------------------------------------- temporal hold-out (S variants)
    st = load("shift_temporal_replay")
    section("shift_temporal_replay: variant - REF, mean over (new pair, seed), SE clustered by pair")
    for T in sorted(st["T"].unique()):
        s = st[st["T"] == T]
        ref = s[s.model == "REF"].set_index(["pair", "seed"])[list(SEC)]
        for m in ("A156", "A156+NP", "A156+SY1.0"):
            a = s[s.model == m].set_index(["pair", "seed"])[list(SEC)]
            d = (a - ref.loc[a.index]).reset_index()
            per_pair = d.groupby("pair")["alc"].mean()
            se = per_pair.std(ddof=1) / math.sqrt(len(per_pair))
            print(f"  T={T} {m:<11s} ALC {1e3 * d['alc'].mean():+.2f} ± {1e3 * se:.2f} (n={len(d)}, pairs "
                  f"{len(per_pair)}) | " + " ".join(f"{x.upper()} {1e3 * d[x].mean():+.2f}" for x in BUD))
    # SY1.0 vs A156 directly
    for T in sorted(st["T"].unique()):
        s = st[st["T"] == T]
        a = s[s.model == "A156+SY1.0"].set_index(["pair", "seed"])["alc"]
        b = s[s.model == "A156"].set_index(["pair", "seed"])["alc"]
        d = (a - b.loc[a.index]).reset_index()
        per_pair = d.groupby("pair")["alc"].mean()
        print(f"  T={T} A156+SY1.0 - A156: {1e3 * d['alc'].mean():+.2f} ± "
              f"{1e3 * per_pair.std(ddof=1) / math.sqrt(len(per_pair)):.2f}")


if __name__ == "__main__":
    main()
