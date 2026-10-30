"""New local numbers for report/paper_draft_v3_1.md (swarm round 7, writer). Reads saved replay CSVs only.

Same convention as swarm/round5/writer_sims/paper_numbers_v3.py (the v3 draft):
- pairing: within one CSV every model was scored on identical draws; rows joined on (bench, k, run, pair);
- per draw (bench, k, run) the per-pair differences are averaged; per (bench, k) cell the mean over draws and
  SE = sd(draw means) / sqrt(#draws) ("draw-clustered");
- macro = mean over benchmarks of the mean over k (benchmark-equal, k-averaged); its SE combines the cell SEs.
For reference only, a "pair-level" SE treats every pair-draw as independent (sd / sqrt(n)).
Units: 1e-3 Brier ALC (arm minus reference; negative favours the arm).

Usage (from brier_rabbit/):
  PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round7/writer_sims/paper_numbers_v3_1.py \
      > swarm/round7/writer_sims/paper_numbers_v3_1.out
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
KEYS = ["bench", "k", "run", "pair"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def paired(df: pd.DataFrame, model: str, ref: str) -> pd.DataFrame:
    a = df[df.model == model].set_index(KEYS)[list(SEC)]
    b = df[df.model == ref].set_index(KEYS)[list(SEC)]
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), (model, ref, len(common), len(a), len(b))
    return (a.loc[common] - b.loc[common]).reset_index()


def cells(d: pd.DataFrame) -> pd.DataFrame:
    """Per (bench, k) cell: mean over draws, draw-clustered SE, pair-level SE, n draws, n pair-draws."""
    per_draw = d.groupby(["bench", "k", "run"])[list(SEC)].mean().reset_index()
    out = []
    for (b, k), g in per_draw.groupby(["bench", "k"]):
        raw = d[(d.bench == b) & (d.k == k)]
        row = {"bench": b, "k": k, "draws": len(g), "pairdraws": len(raw)}
        for c in SEC:
            row[c] = g[c].mean()
            row[c + "_se"] = g[c].std(ddof=1) / math.sqrt(len(g))
            row[c + "_sep"] = raw[c].std(ddof=1) / math.sqrt(len(raw))
        out.append(row)
    return pd.DataFrame(out)


def macro(d: pd.DataFrame, ks=None) -> dict:
    if ks is not None:
        d = d[d["k"].isin(ks)]
    c = cells(d)
    benches = sorted(c.bench.unique())
    nk = c.groupby("bench").size().to_dict()
    w = np.array([1.0 / (len(benches) * nk[b]) for b in c.bench])
    res = {}
    for s in SEC:
        res[s] = float((c[s].values * w).sum())
        res[s + "_se"] = float(math.sqrt(((w ** 2) * (c[s + "_se"].values ** 2)).sum()))
    pb = {}
    for b in benches:
        sub = c[c.bench == b]
        pb[b] = (float(sub["alc"].mean()), float(math.sqrt((sub["alc_se"] ** 2).sum()) / len(sub)))
    res["bench"] = pb
    res["draws"] = int(c.draws.sum())
    res["pairdraws"] = int(c.pairdraws.sum())
    return res


def line(label: str, r: dict) -> str:
    z = r["alc"] / r["alc_se"] if r["alc_se"] > 0 else float("nan")
    b = " ".join(f"{SHORT[k]} {1e3 * v[0]:+.2f}({1e3 * v[1]:.2f})" for k, v in r["bench"].items())
    bud = " ".join(f"{x.upper()} {1e3 * r[x]:+.2f}({1e3 * r[x + '_se']:.2f})" for x in BUD)
    return (f"{label:<44s} ALC {1e3 * r['alc']:+.3f} ± {1e3 * r['alc_se']:.3f} (z {z:+.2f}) | {b} | {bud} | "
            f"draws {r['draws']} pair-draws {r['pairdraws']}")


def show_cells(d: pd.DataFrame, label: str) -> pd.DataFrame:
    c = cells(d)
    for _, r in c.iterrows():
        parts = []
        for s in SEC:
            z = r[s] / r[s + "_se"] if r[s + "_se"] > 0 else float("nan")
            parts.append(f"{s.upper()} {1e3 * r[s]:+.2f}({1e3 * r[s + '_se']:.2f}; pair {1e3 * r[s + '_sep']:.2f}; z {z:+.1f})")
        print(f"  {label} {SHORT[r.bench]} k={int(r.k)} [{int(r.draws)} draws, {int(r.pairdraws)} pair-draws]: "
              + " ".join(parts))
    return c


def absmacro(df: pd.DataFrame, model: str, ks=None) -> float:
    d = df[df.model == model]
    if ks is not None:
        d = d[d["k"].isin(ks)]
    m = d.groupby(["bench", "k"])["alc"].mean()
    return float(m.groupby(level=0).mean().mean())


def section(t: str) -> None:
    print("\n== " + t)


def main() -> None:
    cv = load("combo_v1")

    # ------------------------------------------------------------ P: count of changed pair-draws with a tolerance
    section("combo_v1: P (P_simpool) - base, pair-draws that change (fact-check item 1)")
    d = paired(cv, "P_simpool", "base")
    absd = d[list(SEC)].abs()
    anycol = absd.max(axis=1)
    big = d[absd["alc"] > 1e-6]
    print(f"  |dALC| > 1e-6: {len(big)} of {len(d)} pair-draws; benches {big.bench.value_counts().to_dict()}, "
          f"k {big.k.value_counts().to_dict()}, max |dALC| {absd['alc'].max():.3e}")
    rest = absd[absd["alc"] <= 1e-6]
    print(f"  dALC != 0 (any size): {(absd['alc'] > 0).sum()}; any column != 0: {(anycol > 0).sum()}; "
          f"max |d| over all columns among the rest: {rest.max(axis=1).max():.2e}")
    k1 = d[d.k == 1]
    nz1 = (k1[list(SEC)].abs().max(axis=1) > 0)
    print(f"  k=1 pair-draws with any nonzero difference: {int(nz1.sum())} of {len(k1)}; "
          f"max |d| {k1[list(SEC)].abs().max(axis=1).max():.2e}")
    c = paired(cv, "C_cav30", "base")
    print(f"  C - base: max |d| at k=1 over all columns {c[c.k == 1][list(SEC)].abs().max().max():.2e}; "
          f"max |dB0| over all k {c['b0'].abs().max():.2e}")

    # ------------------------------------------------------------ combo_v1: S cost on RCB at B0 by k
    section("combo_v1: S and S+C+P - base on ResearchCodeBench by k (B0 and ALC)")
    for m in ("S_subjcal", "SCP_combo"):
        dd = paired(cv, m, "base")
        show_cells(dd[dd.bench == "researchcodebench"], m)

    # ------------------------------------------------------------ combo_dense
    dn = load("combo_dense")
    section("combo_dense (bench-keyed, REAL and RCB, k 4 and 8, 10 draws per cell) vs base")
    for m in ("SCP_combo", "C_cav30"):
        dd = paired(dn, m, "base")
        c = show_cells(dd, m)
        worst = []
        for _, r in c.iterrows():
            for s in BUD[1:]:
                worst.append((1e3 * r[s], r[s] / r[s + "_se"], SHORT[r.bench], int(r.k), s.upper()))
        worst.sort(reverse=True)
        print(f"  {m}: worst B1-B31 cell {worst[0][0]:+.2f} (z {worst[0][1]:+.2f}; {worst[0][2]} k={worst[0][3]} "
              f"{worst[0][4]})")
    for m in ("base", "SCP_combo", "C_cav30"):
        print(f"  absolute macro {m}: {absmacro(dn, m):.4f}")
    rule = ("job rule: SCP and C not worse than base at any budget by more than 2 SE, and B1 not worse than +0.5e-3")
    print("  " + rule)

    # ------------------------------------------------------------ contrib_ma_clean
    mc = load("contrib_ma_clean")
    section("contrib_ma_clean (MathArena cleaned: system prompts, misaligned, duplicated AIME, rubric rows) vs base")
    for m in ("SCP_combo", "C_cav30"):
        dd = paired(mc, m, "base")
        show_cells(dd, m)
        print(line(f"{m} - base, k-averaged", macro(dd)))
    for m in ("SCP_combo", "C_cav30"):
        dd = paired(cv, m, "base")
        dd = dd[dd.bench == "matharena"]
        print(line(f"{m} - base, uncleaned MathArena (combo_v1)", macro(dd)))
        show_cells(dd, m + " (uncleaned)")
    base = mc[mc.model == "base"]
    print(f"  cleaned MathArena base: pooled B0 Brier {base['b0'].mean():.4f}, ALC macro {absmacro(mc, 'base'):.4f}; "
          f"uncleaned base B0 {cv[(cv.model == 'base') & (cv.bench == 'matharena')]['b0'].mean():.4f}")

    # ------------------------------------------------------------ fw_kstress
    fk = load("fw_kstress")
    section("fw_kstress: reproduction of combo_v1 rows at k 1,2")
    for m in ("SCP_combo", "w20_SCP", "l15_SCP"):
        a = fk[(fk.model == m) & (fk.k.isin([1, 2]))].set_index(KEYS)[list(SEC)]
        b = cv[cv.model == m].set_index(KEYS)[list(SEC)]
        com = a.index.intersection(b.index)
        print(f"  {m}: {len(com)} of {len(b)} joined, max |diff| {float((a.loc[com] - b.loc[com]).abs().max().max()):.2e}")
    section("fw_kstress: W20+SCP - REF+SCP and L15+SCP - W20+SCP by pairs per benchmark (k)")
    for m, ref in (("w20_SCP", "SCP_combo"), ("l15_SCP", "w20_SCP"), ("l15_SCP", "SCP_combo")):
        dd = paired(fk, m, ref)
        for k in (1, 2, 4):
            print(line(f"{m} - {ref} k={k}", macro(dd, [k])))
        print(line(f"{m} - {ref} k-avg(1,2)", macro(dd, [1, 2])))
    section("fw_kstress: variance splits at W20 (pre-registered R1, swarm/round4/frontier_watch.md section 5)")
    for m in ("w20_s54_SCP", "w20_s85_SCP"):
        dd = paired(fk, m, "w20_SCP")
        inv_k1 = dd[dd.k == 1][list(SEC)].abs().max().max()
        inv_b0 = dd["b0"].abs().max()
        print(f"  {m}: invariance max |d| at k=1 {inv_k1:.2e}, at B0 (all k) {inv_b0:.2e}")
        for k in (2, 4):
            r = macro(dd, [k])
            print(line(f"{m} - w20_SCP k={k}", r))
        r2, r4 = macro(dd, [2]), macro(dd, [4])
        b_bad = [f"{SHORT[b]} k={k} {1e3 * v[0]:+.2f}" for k, rr in ((2, r2), (4, r4)) for b, v in rr["bench"].items()
                 if 1e3 * v[0] > 0.3]
        ok_b = (1e3 * r2["alc"] <= -0.5) and (r2["alc"] / r2["alc_se"] <= -2)
        ok_c = 1e3 * r4["alc"] <= 0
        print(f"  R1 {m}: (b) k=2 {1e3 * r2['alc']:+.2f} z {r2['alc'] / r2['alc_se']:+.2f} -> {'pass' if ok_b else 'FAIL'}; "
              f"(c) k=4 {1e3 * r4['alc']:+.2f} -> {'pass' if ok_c else 'FAIL'}; (d) benchmarks above +0.3: "
              f"{b_bad if b_bad else 'none'} -> {'FAIL' if b_bad else 'pass'}")
    dd = paired(fk, "w20_SCP", "SCP_combo")
    r4 = macro(dd, [4])
    print(f"  R2: W20+SCP - REF+SCP at k=4 {1e3 * r4['alc']:+.2f} (z {r4['alc'] / r4['alc_se']:+.2f}); fires if > +0.5 "
          f"with z >= 1.5 -> {'FIRES' if (1e3 * r4['alc'] > 0.5 and r4['alc'] / r4['alc_se'] >= 1.5) else 'does not fire'}")
    for m in ("SCP_combo", "w20_SCP", "l15_SCP"):
        print(f"  absolute macro {m}: k1 {absmacro(fk, m, [1]):.4f} k2 {absmacro(fk, m, [2]):.4f} "
              f"k4 {absmacro(fk, m, [4]):.4f}")

    # ------------------------------------------------------------ ld_guard_1007
    lg = load("ld_guard_1007")
    aw = load("adj_wide")
    section("ld_guard_1007 (pair-keyed, ks 1,2, 20 runs; adj_wide draws) vs base")
    a = lg[lg.model == "base"].set_index(KEYS)[list(SEC)]
    b = aw[aw.model == "base"].set_index(KEYS)[list(SEC)]
    com = a.index.intersection(b.index)
    print(f"  base reproduces adj_wide: {len(com)} of {len(b)} joined, max |diff| "
          f"{float((a.loc[com] - b.loc[com]).abs().max().max()):.2e}")
    for m in ("m225_mu25_d14", "m250_mu25_d06"):
        dd = paired(lg, m, "base")
        print(line(f"{m} - base", macro(dd)))
        print(f"  absolute macro {m} {absmacro(lg, m):.4f} (base {absmacro(lg, 'base'):.4f})")
    for m in ("w20_m15",):
        dd = paired(aw, m, "base")
        print(line(f"{m} - base (adj_wide; W20 at m0 -1.5)", macro(dd)))


if __name__ == "__main__":
    main()
