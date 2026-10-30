"""Section 1f: does benchmark difficulty adapt to the subject vintage? MathArena's competitions are sub-benchmarks with
their own subject sets. Additive fit on smoothed logits l[s, c] = a_s + b_c (subjects with >= 3 competitions,
competitions with >= 8 such subjects; alternating least squares). b_c = difficulty for a FIXED subject (higher =
easier). Compare b_c with the competition date and with the mean release year of the subjects evaluated on it, and the
competition's realised mean logit over its own subjects. Human contests (AIME/HMMT/...) are not designed for models;
the late-2025 sets (arxivmath, apex) are curated research-level/model-hard problems."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shiftlib as sl  # noqa: E402

# approximate public dates of each competition (month the problems were released)
DATES = {"aime_2025": 2025.12, "aime_2025_I": 2025.12, "aime_2025_II": 2025.12, "hmmt_feb_2025": 2025.12,
         "brumo_2025": 2025.29, "smt_2025": 2025.29, "cmimc_2025": 2025.29, "usamo_2025": 2025.21,
         "imo_2025": 2025.54, "imc_2025": 2025.62, "hmmt_nov_2025": 2025.87, "putnam_2025": 2025.95,
         "kangaroo_2025_1_2": 2025.21, "kangaroo_2025_3_4": 2025.21, "kangaroo_2025_5_6": 2025.21,
         "kangaroo_2025_7_8": 2025.21, "kangaroo_2025_9_10": 2025.21, "kangaroo_2025_11_12": 2025.21,
         "arxivmath_1225": 2025.96, "arxivmath_0126": 2026.04, "arxivmath_0226": 2026.12,
         "aime_2026": 2026.12, "aime_2026_I": 2026.12, "aime_2026_II": 2026.12, "hmmt_feb_2026": 2026.12,
         "apex_2025": 2025.5, "apex_shortlist": 2025.6}


def main():
    pd.set_option("display.width", 200)
    df = sl.load()
    pt = sl.pair_table(df)
    d = df[df.bench == "matharena"].copy()
    d["comp"] = d.item_features.str.split(";").str[0].str.replace("competition=", "", regex=False)
    g = d.groupby(["pair", "comp"]).label.agg(["sum", "size"]).reset_index()
    g = g[g["size"] >= 10]
    g["l"] = np.log((g["sum"] + 0.5) / (g["size"] - g["sum"] + 0.5))
    for _ in range(3):  # prune to a connected, well-covered block
        nc = g.groupby("pair").comp.nunique()
        g = g[g.pair.isin(nc[nc >= 3].index)]
        ns = g.groupby("comp").pair.nunique()
        g = g[g.comp.isin(ns[ns >= 8].index)]
    a = g.groupby("pair").l.mean()
    b = pd.Series(0.0, index=g.comp.unique())
    for _ in range(200):
        b = (g.l - g.pair.map(a)).groupby(g.comp).mean()
        b = b - b.mean()
        a = (g.l - g.comp.map(b)).groupby(g.pair).mean()
    rel = pt.set_index("pair").rel
    out = []
    for c, h in g.groupby("comp"):
        out.append({"comp": c, "date": DATES.get(c, np.nan), "n_subj": h.pair.nunique(),
                    "b_fixed_subject": b[c], "mean_logit_own_subjects": h.l.mean(),
                    "mean_subject_ability_a": h.pair.map(a).mean(),
                    "mean_release_of_subjects": h.pair.map(rel).mean(),
                    "share_dated": h.pair.map(rel).notna().mean()})
    O = pd.DataFrame(out).sort_values("date")
    print(f"MathArena two-way fit: {g.pair.nunique()} subjects x {g.comp.nunique()} competitions, {len(g)} cells")
    print(O.round(2).to_string(index=False))
    x = O.dropna(subset=["date"])
    for col in ("b_fixed_subject", "mean_logit_own_subjects", "mean_subject_ability_a"):
        sl_, ic = np.polyfit(x.date, x[col], 1)
        print(f"   slope of {col} on competition date: {sl_:+.2f} per year (n={len(x)})")
    hum = x[~x.comp.str.startswith(("arxivmath", "apex"))]
    mdl = x[x.comp.str.startswith(("arxivmath", "apex"))]
    print(f"   human contests: mean b {hum.b_fixed_subject.mean():+.2f}, mean own-subject logit "
          f"{hum.mean_logit_own_subjects.mean():+.2f}, subject ability {hum.mean_subject_ability_a.mean():+.2f}")
    print(f"   model-era sets (arxivmath/apex): mean b {mdl.b_fixed_subject.mean():+.2f}, mean own-subject logit "
          f"{mdl.mean_logit_own_subjects.mean():+.2f}, subject ability {mdl.mean_subject_ability_a.mean():+.2f}")


if __name__ == "__main__":
    main()
