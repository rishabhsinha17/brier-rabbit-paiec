"""Markdown rows of Tables 2, 3, C2 (sigma_g rows), C4 and C5 of report/paper_draft_v3.md, printed from saved replay CSVs (no replay is run).

Convention: swarm/round5/writer_sims/paper_numbers_v3.py (paired on identical draws, benchmark-equal, k-averaged,
draw-clustered SE; values x1e-3, arm minus reference, negative favours the arm).

Usage (from brier_rabbit/):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/writer_sims/paper_tables_v3.py \
      > swarm/round5/writer_sims/paper_tables_v3.out
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_numbers_v3 import BENCH, BUD, absmacro, load, paired, summarize  # noqa: E402


def f1(x: float) -> str:
    v = round(1e3 * x, 1)
    if v == 0:
        return "0.0"
    return f"{v:+.1f}"


def f2(x: float) -> str:
    v = round(1e3 * x, 2)
    if v == 0:
        return "0.00"
    return f"{v:+.2f}"


def row(label: str, r: dict, macro_abs: str = "", vs: str = "", bench: bool = False, two: bool = True) -> str:
    m, s = r["macro"], r["se"]
    d = f"{f2(m['alc']) if two else f1(m['alc'])} ({1e3 * s['alc']:.2f})" if two else f"{f1(m['alc'])} ({1e3 * s['alc']:.1f})"
    cells = [label]
    if macro_abs != "" or vs != "":
        cells += [macro_abs, d, vs]
    else:
        cells += [d]
    if bench:
        cells += [f2(r["bench"][b]) for b in BENCH]
    cells += [f1(m[b]) for b in BUD]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    cv = load("combo_v1")
    cs = load("combo_sy")
    cn = load("combo_np")

    print("== Table 3: local ladder (combo_v1 draws)")
    ladder = [
        ("Constant 0.5", "const_0.5", None, ""),
        ("Organizer empirical mean", "empirical_mean", None, ""),
        ("Smoothed mean ($\\kappa = 2$)", "smoothed_k2", "const_0.5", "constant"),
        ("Hierarchical, untuned ($m_0 = -0.5$)", "v1_default", "smoothed_k2", "smoothed"),
        ("Hierarchical, tuned", "v1_tuned", "v1_default", "untuned"),
        ("+ subject prior ($m_0 = -0.5$)", "v1_subjects", "v1_tuned", "tuned"),
        ("REF: damped Newton, $m_0 = -2.0$", "base", "v1_subjects", "+ subject prior"),
        ("REF + S + C + P (submitted composite)", "SCP_combo", "base", "REF"),
        ("W20 + S + C + P", "w20_SCP", "SCP_combo", "REF + S+C+P"),
        ("L15 + S + C + P", "l15_SCP", "SCP_combo", "REF + S+C+P"),
    ]
    for label, model, ref, vs in ladder:
        a = f"{absmacro(cv, model):.4f}"
        if ref is None:
            print("| " + " | ".join([label, a] + [""] * 8) + " |")
            continue
        r = summarize(paired(cv, model, ref))
        print(row(label, r, a, vs, two=False))
    for label, model in (("REF vs smoothed mean", "base"), ("REF + S+C+P vs smoothed mean", "SCP_combo"),
                         ("W20 + S+C+P vs smoothed mean", "w20_SCP"), ("L15 + S+C+P vs smoothed mean", "l15_SCP")):
        r = summarize(paired(cv, model, "smoothed_k2"))
        print(row(label, r, "", "smoothed", two=False))

    print("\n== Table 2: composite ablation (combo_v1); Table C5: subject-prior variants (combo_sy, combo_np); k splits for Appendix C")
    t3 = [
        ("S: attribute rescale x1.56", cv, "S_subjcal", "base", None),
        ("C: cavity pooling, $\\sigma_x = 3.0$", cv, "C_cav30", "base", None),
        ("P: name and harness pooling", cv, "P_simpool", "base", None),
        ("S + C + P (REF level)", cv, "SCP_combo", "base", None),
        ("S + C + P on W20, vs W20", cv, "w20_SCP", "w20_base", None),
        ("S + C + P on L15, vs L15", cv, "l15_SCP", "l15_base", None),
        ("SY1.0 instead of S, inside S+C+P", cs, "SCP_sy10", "SCP_combo", None),
        ("NP prior added, inside S+C+P", cn, "SCP_np", "SCP_combo", None),
    ]
    for label, df, m, ref, _ in t3:
        r = summarize(paired(df, m, ref))
        print(row(label, r, bench=True))
    for k in (1, 2):
        for label, m, ref in (("W20 vs REF", "w20_base", "base"), ("W20 + S+C+P vs REF + S+C+P", "w20_SCP", "SCP_combo"),
                              ("L15 vs REF", "l15_base", "base"), ("S+C+P vs REF", "SCP_combo", "base")):
            r = summarize(paired(cv, m, ref), ks=[k])
            print(row(f"k={k}: {label}", r, bench=True))

    print("\n== Table C1 (old pair-keyed ladder, sets A/B): see swarm/round3/report_writer_sims/paper_tables.out (unchanged)")

    print("\n== Table C4 (decisive cavity replay, pdpool_bench_k2: bench-keyed, k=2, 20 draws per benchmark)")
    pp = load("pdpool_bench_k2")
    for label, m in (("Cavity, $\\sigma_x = 1.5$", "cav_x1.5"), ("Cavity, $\\sigma_x = 2.0$", "cav_x2.0"),
                     ("Cavity, $\\sigma_x = 2.5$", "cav_x2.5"), ("Cavity, $\\sigma_x = 3.0$ (selected)", "cav_x3.0"),
                     ("Cavity, $\\sigma_x = 4.0$", "cav_x4.0"), ("Two-scale item pooling, $\\sigma_x = 2.0$", "headA_x2.0"),
                     ("Plug-in MAP item offsets, SD 2.0", "map_x2.0"), ("$\\sigma_i = 1.0$", "si10"),
                     ("$\\sigma_i = 1.5$", "si15"), ("$\\sigma_i = 2.0$", "si20")):
        r = summarize(paired(pp, m, "base"))
        print(row(label, r, bench=True))
    tw = load("builder_pair_twin_k2")
    keys = ["bench", "k", "run", "pair"]
    for m in ("cav_x2.0", "cav_x3.0"):
        old = paired(tw, m, "base")
        new = paired(pp, m, "base")
        j = new.set_index(keys).sub(old.set_index(keys)).dropna().reset_index()
        print(row(f"{m}: pair-keyed", summarize(old), bench=True))
        print(row(f"{m}: bench-keyed minus pair-keyed", summarize(j), bench=True))

    print("\n== Table C2 sigma_g rows (pdadapt_sigma_g, pair-keyed, ks 1,2, 20 draws)")
    sg = load("pdadapt_sigma_g")
    sg = sg[sg.scen == "none"]
    for m in ("sg14", "sg20"):
        print(row(m, summarize(paired(sg, m, "base")), bench=True))

    print("\n== Appendix C local guard (adj_wide, pair-keyed, ks 1,2, 20 draws)")
    aw = load("adj_wide")
    for label, m in (("L15 (m0 -1.5)", "n15D"), ("W20 (tau_mu 2.0)", "w20_m20"), ("W20 at m0 -1.5", "w20_m15"),
                     ("tau_mu 2.5 at m0 -2.5", "w25_m25")):
        print(row(label, summarize(paired(aw, m, "base")), bench=True))


if __name__ == "__main__":
    main()
