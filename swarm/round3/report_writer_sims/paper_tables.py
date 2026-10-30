"""Paired local tables for report/paper_draft_v2.md (no recomputation of any replay; reads saved CSVs only).

For every comparison (csv, model, reference) it prints the benchmark-equal macro Brier ALC of both models, the
paired macro difference and its standard error, the per-budget macro differences, and the per-benchmark differences.

Pairing: within one CSV every model was scored on identical draws (same bench, k, run, pairs, revealed labels).
SE: per (bench, k) cell, the difference is averaged per draw (run), its SE is sd(draw means)/sqrt(#draws); the macro
is the mean over benchmarks of the mean over k, so its SE combines the cell SEs with weights 1/(#bench * #k).

Usage (from brier_rabbit/): ../.venv/bin/python swarm/round3/report_writer_sims/paper_tables.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BUDGETS = ("b0", "b1", "b3", "b7", "b15", "b31")

# (label, csv, model, reference)
COMPARISONS = [
    # Ladder (Table 2 of the paper)
    ("Smoothed mean (kappa=2) vs constant 0.5", "results/paired.csv", "smoothed_k2", "const_0.5"),
    ("v1 hierarchical (untuned) vs smoothed mean", "results/paired.csv", "v1_default", "smoothed_k2"),
    ("v1 tuned vs v1 untuned", "results/paired.csv", "v1_tuned", "v1_default"),
    ("+ subject prior (names + attributes) vs v1 tuned", "results/paired_subjects.csv", "v1_subjects", "v1_tuned"),
    ("m0 -1.5 vs m0 -0.5 (undamped, subject prior)", "results/paired_damped.csv", "n15", "v1_subjects"),
    ("damped Newton vs undamped, m0 -1.5", "results/paired_damped.csv", "n15D", "n15"),
    ("REF (damped, m0 -2.0) vs subject prior at m0 -0.5", "results/paired_damped.csv", "n20D", "v1_subjects"),
    ("m0 -2.0 vs m0 -1.5 (damped)", "results/paired_damped.csv", "n20D", "n15D"),
    ("damped m0 -1.5 vs smoothed mean (set B)", "results/paired_damped.csv", "n15D", "smoothed_k2"),
    ("REF (damped m0 -2.0) vs smoothed mean (set B)", "results/paired_damped.csv", "n20D", "smoothed_k2"),
    ("REF (base) vs subject prior at m0 -0.5 (set C)", "results/paired_mix.csv", "base", "v1_subjects"),
    ("m0 -1.0 vs m0 -1.5 (damped)", "results/paired_damped.csv", "n10D", "n15D"),
    # Negative / neutral results
    ("t(3) level prior vs normal, m0 -1.5 (damped)", "results/paired_damped.csv", "t15D", "n15D"),
    ("t(3) level prior vs normal, m0 -2.0 (damped)", "results/paired_damped.csv", "t20D", "n20D"),
    ("mixture pi .25 @ m0 -1.0 vs base (m0 -2.0)", "results/paired_mix.csv", "mix25_m10", "base"),
    ("mixture pi .35 @ m0 -1.0 vs base (m0 -2.0)", "results/paired_mix.csv", "mix35_m10", "base"),
    ("mixture pi .25 @ m0 -1.5 vs base (m0 -2.0)", "results/paired_mix.csv", "mix25_m15", "base"),
    ("narrow pair prior d06 (q0 .25) vs base", "results/pd1_priors.csv", "d06_q25", "base"),
    ("all taus x0.8 (x08, q0 .25) vs base", "results/pd1_priors.csv", "x08_q25", "base"),
    ("attribute-ability rescale a156 vs base", "results/pd1_priors.csv", "a156", "base"),
    ("ability-tilted mixture vs untilted mix25", "results/pd1_priors.csv", "mix25_m10_tilt", "mix25_m10"),
    ("lam 1.3 vs lam 1.0 (base)", "results/paired_alltok_lam.csv", "lam13", "base"),
    ("lam 1.6 vs lam 1.0 (base)", "results/paired_alltok_lam.csv", "lam16", "base"),
    ("all item_features tokens vs first token", "results/paired_alltok_lam.csv", "alltok", "base"),
    ("ability as fitted effect (lam 1.0) vs fixed offset, m0 -1.5", "results/paired_abilityparam.csv", "ap_l10", "s_m15"),
    ("two-scale item pooling x2.0 vs base (k=2, pair-keyed)", "results/headA_itempool.csv", "pool_x2.0", "base"),
]


def compare(df: pd.DataFrame, model: str, ref: str) -> dict:
    a = df[df.model == model].set_index(["bench", "k", "run", "pair"])
    b = df[df.model == ref].set_index(["bench", "k", "run", "pair"])
    common = a.index.intersection(b.index)
    d = (a.loc[common, ["alc", *BUDGETS]] - b.loc[common, ["alc", *BUDGETS]]).reset_index()
    per_draw = d.groupby(["bench", "k", "run"])[["alc", *BUDGETS]].mean().reset_index()
    cells = per_draw.groupby(["bench", "k"])
    mean = cells[["alc", *BUDGETS]].mean()
    se = cells["alc"].agg(lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan"))
    n_b = mean.index.get_level_values(0).nunique()
    n_k = {bench: (mean.index.get_level_values(0) == bench).sum() for bench in mean.index.get_level_values(0).unique()}
    w = pd.Series({idx: 1.0 / (n_b * n_k[idx[0]]) for idx in mean.index})
    macro = {c: float((mean[c] * w).sum()) for c in ["alc", *BUDGETS]}
    macro_se = float(math.sqrt(((w ** 2) * (se ** 2)).sum()))
    per_bench = mean["alc"].groupby(level=0).mean().to_dict()

    def abs_macro(frame):
        m = frame.groupby(["bench", "k"])["alc"].mean()
        return float(m.groupby(level=0).mean().mean())

    return {"model_macro": abs_macro(df[df.model == model]), "ref_macro": abs_macro(df[df.model == ref]),
            "diff": macro, "se": macro_se, "per_bench": per_bench, "n_pairs": len(common),
            "draws": int(per_draw.shape[0])}


def main() -> None:
    cache = {}
    print("| comparison | model macro | ref macro | diff ALC (SE) x1e-3 | B0 | B1 | B3 | B7 | B15 | B31 | "
          "MathArena | Multi-SWE | REAL | RCB | pairs | draws |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, csv, model, ref in COMPARISONS:
        if csv not in cache:
            cache[csv] = pd.read_csv(REPO / csv)
        r = compare(cache[csv], model, ref)
        pb = r["per_bench"]
        d = r["diff"]
        print(f"| {label} | {r['model_macro']:.4f} | {r['ref_macro']:.4f} | {1e3 * d['alc']:+.2f} ({1e3 * r['se']:.2f}) | "
              + " | ".join(f"{1e3 * d[b]:+.1f}" for b in BUDGETS) + " | "
              + " | ".join(f"{1e3 * pb.get(x, float('nan')):+.1f}" for x in
                           ("matharena", "multi_swebench", "real_webagents", "researchcodebench"))
              + f" | {r['n_pairs']} | {r['draws']} |")


if __name__ == "__main__":
    main()
