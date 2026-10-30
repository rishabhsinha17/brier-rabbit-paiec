"""Read results/combo_v1.csv against the pre-registered composite rule (swarm/round3/composer.md section 6).

Prints, x1e-3 vs base (negative = better): bench-equal k-averaged macro with SE, per benchmark, per budget; the
additivity residual SCP - (S + C + P - 2 base); the transfer of the composite gain to the W20 and L15 level blocks;
the consistency check against results/pdpool_bench_k2.csv (k=2 runs 0-17: base and C_cav30 must equal its base and
cav_x3.0 rows exactly); and the PASS/FAIL of each rule.
usage: ../.venv/bin/python swarm/round3/composer_sims/read_combo_v1.py [results/combo_v1.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
BUD = ["b0", "b1", "b3", "b7", "b15", "b31"]


def macro(d: pd.Series):
    """Bench-equal, k-averaged mean of a per-pair difference and its SE (independent pair-draws)."""
    g = d.groupby(level=["bench", "k"])
    m = g.mean().groupby(level="bench").mean()
    v = (g.var(ddof=1) / g.size()).groupby(level="bench").sum() / g.size().groupby(level="bench").count() ** 2
    return float(m.mean()), float(np.sqrt(v.sum()) / len(m)), m


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/combo_v1.csv"
    df = pd.read_csv(path)
    W = {c: df.pivot_table(index=["bench", "k", "run", "pair"], columns="model", values=c) for c in ["alc", *BUD]}
    a = W["alc"]
    C = next(c for c in a.columns if c.startswith("C_cav"))
    ok = {}
    print(f"{path.name}: {len(a)} pair-draws; arms {sorted(c for c in a.columns if c[0] in 'SCPwl' or c == 'base')}")
    print("\nALC vs base (x1e-3): macro +- SE | per benchmark | per budget (pooled, x1e-3)")
    res = {}
    for arm in ["S_subjcal", C, "P_simpool", "SCP_combo"]:
        m, se, per = macro(a[arm] - a["base"])
        res[arm] = (m, se, per)
        pb = " ".join(f"{b.upper()} {1e3 * (W[b][arm] - W[b]['base']).mean():+.2f}" for b in BUD)
        print(f"  {arm:10s} {1e3 * m:+.3f} +- {1e3 * se:.3f} | " + " ".join(f"{k[:5]} {1e3 * v:+.2f}" for k, v in per.items())
              + f" | {pb}")
    add = a["SCP_combo"] - a["S_subjcal"] - a[C] - a["P_simpool"] + 2 * a["base"]
    m_add, se_add, _ = macro(add)
    print(f"\nadditivity SCP - (S + C + P - 2 base): {1e3 * m_add:+.3f} +- {1e3 * se_add:.3f}")
    trans = {}
    for lb in ("w20", "l15"):
        m, se, per = macro(a[f"{lb}_SCP"] - a[f"{lb}_base"])
        mb, seb, _ = macro(a[f"{lb}_base"] - a["base"])
        trans[lb] = m
        print(f"{lb}: level block vs REF {1e3 * mb:+.3f} +- {1e3 * seb:.3f}; composite gain on it {1e3 * m:+.3f} +- {1e3 * se:.3f} "
              f"(REF: {1e3 * res['SCP_combo'][0]:+.3f}) | " + " ".join(f"{k[:5]} {1e3 * v:+.2f}" for k, v in per.items()))
    # pre-registered rules
    scp_m, scp_se, scp_per = res["SCP_combo"]
    ok["1 SCP macro <= -0.5e-3 and <= best single"] = scp_m <= -0.5e-3 and scp_m <= min(res[x][0] for x in res if x != "SCP_combo")
    ok["2 no benchmark worse than +0.3e-3"] = bool((scp_per <= 0.3e-3).all())
    worst_z = max(((W[b]["SCP_combo"] - W[b]["base"]).mean() / ((W[b]["SCP_combo"] - W[b]["base"]).std(ddof=1)
                                                                  / np.sqrt(len(a)))) for b in BUD if
                  (W[b]["SCP_combo"] - W[b]["base"]).std() > 0)
    ok["3 no budget worse than base by > 2 SE"] = worst_z <= 2.0
    ok["4 additivity |residual| <= 0.2e-3"] = abs(m_add) <= 0.2e-3
    ok["5 transfer: W20/L15 gain <= -0.3e-3 and within 0.3e-3 of REF's"] = all(
        t <= -0.3e-3 and abs(t - scp_m) <= 0.3e-3 for t in trans.values())
    pp = ROOT / "results/pdpool_bench_k2.csv"
    if pp.exists():
        q = pd.read_csv(pp)
        q = q[(q.k == 2) & (q.run <= 17)].pivot_table(index=["bench", "k", "run", "pair"], columns="model", values="alc")
        j = a.join(q[["base", "cav_x3.0"]].rename(columns={"base": "pp_base", "cav_x3.0": "pp_cav"}), how="inner")
        dmax = float(max((j["base"] - j["pp_base"]).abs().max(), (j[C] - j["pp_cav"]).abs().max()))
        ok["6 consistency with pdpool_bench_k2 (k=2 runs 0-17)"] = len(j) > 0 and dmax < 1e-12
        print(f"\nconsistency: {len(j)} joined k=2 pair-draws, max |diff| base/{C} vs pdpool base/cav_x3.0 = {dmax:.1e}")
    print("\npre-registered rules:")
    for k, v in ok.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("SHIP the composite on the winning level block" if all(ok.values()) else
          "DO NOT ship the composite as is: drop the failing component(s) (section 6 of composer.md)")


if __name__ == "__main__":
    main()
