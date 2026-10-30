"""Table C7 and the calibration sentences of report/paper_draft_v3_1.md (swarm round 7b, writer B).

Reads only the saved outputs of the round-6 calibration job (swarm/round6/cal_sims/out_{bench,pair}/calib_long.csv,
written by analyze.py); rounds them consistently (the summary.md tables print 4 decimals, the figure 3).
Units: ECE as a probability; REL, RES, bias (mean prediction minus success rate) and differences in 1e-3 Brier.

Usage (from brier_rabbit/):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round7/writer_sims/writer_B_cal_table.py \
      > swarm/round7/writer_sims/writer_B_cal_table.out
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "swarm" / "round6" / "cal_sims"
BUD = ["0", "1", "3", "7", "15", "31", "alc"]
NAME = {"base": "REF", "SCP_combo": "REF + S + C + P", "w20_SCP": "W20 + S + C + P"}


def load(mode: str) -> pd.DataFrame:
    d = pd.read_csv(OUT / f"out_{mode}" / "calib_long.csv", dtype={"budget": str})
    return d[d.scope == "all"].set_index(["model", "budget"])


def main() -> None:
    d = load("bench")
    print("budgets:", sorted(set(d.index.get_level_values(1))))
    print("\n== ECE (10 equal-mass bins, pooled, pair-equal weights) +- draw-clustered SE; floor")
    for m in ("base", "SCP_combo", "w20_SCP"):
        cells = [f"{d.loc[(m, b), 'ece']:.3f} ± {d.loc[(m, b), 'ece_se']:.3f}" for b in BUD]
        floors = [f"{d.loc[(m, b), 'ece_floor']:.4f}" for b in BUD]
        print(f"{NAME[m]:<18s} | " + " | ".join(cells))
        print(f"{'  floor':<18s} | " + " | ".join(floors))
    print("\n== REF: REL / RES / bias (1e-3); UNC")
    m = "base"
    print(" | ".join(f"{1e3 * d.loc[(m, b), 'rel']:.1f} / {1e3 * d.loc[(m, b), 'res']:.1f} / "
                     f"{1e3 * (d.loc[(m, b), 'mean_pred'] - d.loc[(m, b), 'base_rate']):+.0f}" for b in BUD))
    print("UNC", {b: round(1e3 * d.loc[(m, b), "unc"], 2) for b in BUD})
    for m in ("SCP_combo", "w20_SCP"):
        print(f"\n== {NAME[m]} minus REF (1e-3, paired bootstrap SE): dREL / dRES ; dBrier")
        print(" | ".join(f"{1e3 * d.loc[(m, b), 'd_rel']:+.1f} ({1e3 * d.loc[(m, b), 'd_rel_se']:.1f}) / "
                         f"{1e3 * d.loc[(m, b), 'd_res']:+.1f} ({1e3 * d.loc[(m, b), 'd_res_se']:.1f})" for b in BUD))
        print(" | ".join(f"{1e3 * d.loc[(m, b), 'd_brier']:+.2f} ± {1e3 * d.loc[(m, b), 'd_brier_se']:.2f}" for b in BUD))
        print(" bias:", " | ".join(f"{1e3 * (d.loc[(m, b), 'mean_pred'] - d.loc[(m, b), 'base_rate']):+.0f}" for b in BUD))
    print("\n== platform-style ECE (ALC column) and its floor")
    for m in ("base", "SCP_combo", "w20_SCP"):
        print(f"{NAME[m]:<18s} {d.loc[(m, 'alc'), 'ece_platform']:.4f} ± {d.loc[(m, 'alc'), 'ece_platform_se']:.4f}; "
              f"floor {d.loc[(m, 'alc'), 'ece_platform_floor']:.4f}")
    print("\n== ECE floor range over the three models and six budgets plus ALC:",
          f"{d.loc[['base', 'SCP_combo', 'w20_SCP'], 'ece_floor'].min():.4f} to "
          f"{d.loc[['base', 'SCP_combo', 'w20_SCP'], 'ece_floor'].max():.4f}")
    print("\n== per benchmark, REF bias (1e-3) by budget")
    full = pd.read_csv(OUT / "out_bench" / "calib_long.csv", dtype={"budget": str})
    for sc in ("matharena", "multi_swebench", "real_webagents", "researchcodebench"):
        s = full[(full.scope == sc) & (full.model == "base")].set_index("budget")
        print(f"{sc:<18s} " + " ".join(f"B{b} {1e3 * (s.loc[b, 'mean_pred'] - s.loc[b, 'base_rate']):+.0f}"
                                         for b in BUD))
    p = load("pair")
    print("\n== pair-keyed draws (out_pair): ALC-weighted ECE and W20 + S + C + P minus REF")
    for m in ("base", "SCP_combo", "w20_SCP"):
        print(f"{NAME[m]:<18s} ECE {p.loc[(m, 'alc'), 'ece']:.4f} ± {p.loc[(m, 'alc'), 'ece_se']:.4f}")
    m = "w20_SCP"
    print(f"dBrier {1e3 * p.loc[(m, 'alc'), 'd_brier']:+.2f} ± {1e3 * p.loc[(m, 'alc'), 'd_brier_se']:.2f}; "
          f"dREL {1e3 * p.loc[(m, 'alc'), 'd_rel']:+.2f} ± {1e3 * p.loc[(m, 'alc'), 'd_rel_se']:.2f}; "
          f"dRES {1e3 * p.loc[(m, 'alc'), 'd_res']:+.2f} ± {1e3 * p.loc[(m, 'alc'), 'd_res_se']:.2f}")


if __name__ == "__main__":
    main()
