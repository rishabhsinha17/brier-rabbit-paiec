"""Section 1b: temporal hold-out of the subject prior (public data only).

For cutoff T and held-out benchmark b: fit_subject_prior on the OTHER benchmarks using only subjects released before T
("strict": undated subjects excluded from training; "lenient": undated kept, as the live prior would have them).
Evaluate b's eligible pairs: NEW = release >= T, OLD = release < T. The level a live-tuned m0 would learn from the
older pool is m_old = mean_OLD(lg - theta); the B0 bias of the new cohort at that level is
off_new = mean_NEW(lg - theta) - m_old (+ = under-predicted). Also the new cohort's name coverage, mean theta and
within-benchmark slope. Variants: base ridge, attr x1.56 (+ year clamp at T, i.e. no extrapolation past training).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shiftlib as sl  # noqa: E402

BENCHES = sl.LOBO


def theta_variant(pr, subj, scale=1.0, ymax=0.0):
    t, v, s = sl.theta(pr, subj)
    if s == "attr" and (scale != 1.0 or ymax):
        x = sl.attributes(subj)
        if ymax:
            x[1] = min(x[1], ymax - 2025.0)
        t = scale * float(np.dot(pr["attr"]["coef"], x))
    return t, v, s


def run(T, mode, df, pt):
    rel_by_pair = pt.set_index("pair").rel
    rel = df.pair.map(rel_by_pair)
    dated_old = rel.notna() & (rel < T)
    keep = dated_old | (rel.isna() if mode == "lenient" else False)
    rows = []
    for b in BENCHES:
        tr = df[(df.bench != b) & keep]
        pr = sl.prior(tr)
        ev = pt[pt.eligible & (pt.bench == b) & pt.rel.notna()]
        for _, r in ev.iterrows():
            for lab, sc, ym in (("base", 1.0, 0.0), ("a156", 1.56, 0.0), ("a156_clampT", 1.56, T)):
                t, v, s = theta_variant(pr, r.subject, sc, ym)
                rows.append({"T": T, "mode": mode, "bench": b, "pair": r.pair, "variant": lab, "new": r.rel >= T,
                             "rel": r.rel, "lg": r.lg, "th": t, "src": s, "n_names": len(pr["names"]),
                             "year_coef": pr["attr"]["coef"][1]})
    return pd.DataFrame(rows)


def summarise(R):
    out = []
    for (T, mode, var), g in R.groupby(["T", "mode", "variant"]):
        g = g.copy()
        g["res"] = g.lg - g.th
        m_old = g[~g.new].groupby("bench").res.mean()
        g["off"] = g.res - g.bench.map(m_old)
        new = g[g.new & g.bench.isin(m_old.index)]
        old = g[~g.new]
        # within-bench slope among NEW pairs (benches with >= 3 new pairs)
        nb = new.groupby("bench").filter(lambda h: len(h) >= 3)
        xc = nb.th - nb.groupby("bench").th.transform("mean")
        yc = nb.lg - nb.groupby("bench").lg.transform("mean")
        slope = float((xc * yc).sum() / (xc * xc).sum()) if len(nb) >= 4 and (xc * xc).sum() > 1e-9 else np.nan
        # bench-equal mean offset of the new cohort
        per_b = new.groupby("bench").off.agg(["mean", "size"])
        out.append({"T": T, "mode": mode, "variant": var, "n_new": len(new), "n_old": len(old),
                    "new_name_share": float((new.src == "name").mean()) if len(new) else np.nan,
                    "old_name_share": float((old.src == "name").mean()),
                    "new_mean_th": new.th.mean(), "old_mean_th": old.th.mean(),
                    "off_new_pairw": new.off.mean(), "se_off_new": new.off.std(ddof=1) / np.sqrt(max(len(new), 1)),
                    "off_new_benchw": per_b["mean"].mean(),
                    "off_new_attr": new[new.src == "attr"].off.mean(), "off_new_name": new[new.src == "name"].off.mean(),
                    "slope_new": slope, "year_coef": g.year_coef.mean(), "per_bench": per_b.round(2).to_dict("index")})
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 200)
    df = sl.load()
    pt = sl.pair_table(df)
    R = pd.concat([run(T, mode, df, pt) for T in (2025.5, 2026.0) for mode in ("strict", "lenient")])
    R.to_csv(Path(__file__).with_name("calib_temporal_rows.csv"), index=False)
    S = summarise(R)
    print("Temporal hold-out: prior fit on subjects released before T (LOBO); offsets in logits (+ = new cohort "
          "under-predicted vs the level learned on the old cohort)")
    print(S.drop(columns=["per_bench"]).round(3).to_string(index=False))
    print("\nper-benchmark new-cohort offsets (mean, n):")
    for _, r in S.iterrows():
        print(f"  T={r['T']} {r['mode']:8s} {r.variant:12s} {r.per_bench}")


if __name__ == "__main__":
    main()
