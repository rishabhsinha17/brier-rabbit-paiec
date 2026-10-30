"""Section 1a: B0-relevant calibration of the LOBO subject prior on held-out public pairs, split by vintage, source,
effort, provider and size words.

For every eligible pair on the 4 LOBO benchmarks: theta from fit_subject_prior on the OTHER benchmarks (exactly what
experiments/paired.py uses), realised smoothed pair logit lg. Within a benchmark the B0-optimal level is
m_b = mean(lg - theta); a subgroup's B0 bias at that level is off_g = mean_g(lg - theta) - m_b (positive = the prior
under-predicts the subgroup, negative = over-predicts). Calibration slope = within-benchmark OLS slope of lg on theta.
Also reported with SubjCal (attr x1.56, year clamp 2026.4).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shiftlib as sl  # noqa: E402


def vint(rel):
    if rel is None or (isinstance(rel, float) and np.isnan(rel)):
        return "undated"
    if rel < 2025.0:
        return "a<2025"
    if rel < 2025.5:
        return "b2025H1"
    if rel < 2026.0:
        return "c2025H2"
    return "d2026"


def build(scale=1.0, ymax=0.0):
    df = sl.load()
    P = sl.lobo_priors(df)
    g = sl.pair_table(df)
    g = g[g.eligible & g.bench.isin(sl.LOBO)].copy()
    th, tv, src = [], [], []
    for _, r in g.iterrows():
        t, v, s = sl.theta(P[r.bench], r.subject)
        if s == "attr" and (scale != 1.0 or ymax):
            x = sl.attributes(r.subject)
            if ymax:
                x[1] = min(x[1], ymax - 2025.0)
            t = scale * float(np.dot(P[r.bench]["attr"]["coef"], x))
        th.append(t); tv.append(v); src.append(s)
    g["th"], g["tv"], g["src"] = th, tv, src
    g["vint"] = [vint(x) for x in g.rel]
    g["res"] = g.lg - g.th
    g["m_b"] = g.groupby("bench").res.transform("mean")
    g["off"] = g.res - g.m_b
    return g


def slope_within(h, x="th", y="lg"):
    xc = h[x] - h.groupby("bench")[x].transform("mean")
    yc = h[y] - h.groupby("bench")[y].transform("mean")
    den = float((xc * xc).sum())
    if den < 1e-9 or len(h) < 4:
        return np.nan, np.nan
    b = float((xc * yc).sum()) / den
    e = yc - b * xc
    se = float(np.sqrt((e * e).sum() / max(len(h) - 2 - h.bench.nunique(), 1) / den))
    return b, se


def table(g, by):
    rows = []
    for key, h in g.groupby(by):
        b, se = slope_within(h)
        rows.append({by: key, "n": len(h), "mean_th": h.th.mean(), "sd_th": h.th.std(), "mean_off": h.off.mean(),
                     "se_off": h.off.std(ddof=1) / np.sqrt(len(h)) if len(h) > 1 else np.nan,
                     "slope": b, "se_slope": se})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    for scale, ymax, lab in ((1.0, 0.0, "BASE prior"), (1.56, 2026.4, "SubjCal a1.56 + clamp 2026.4")):
        g = build(scale, ymax)
        print(f"\n######## {lab}: {len(g)} eligible LOBO pairs; off = B0 bias at the benchmark-optimal level "
              f"(+ = prior under-predicts); slope = within-bench OLS of lg on theta")
        print("per benchmark: m_b (best level, logit) and mean theta:")
        print(g.groupby("bench").agg(n=("lg", "size"), mean_lg=("lg", "mean"), mean_th=("th", "mean"),
                                     m_b=("m_b", "first"), share_attr=("src", lambda s: np.mean(s == "attr"))).round(3).to_string())
        for by in ("src", "vint"):
            print(f"\nby {by}:")
            print(table(g, by).round(3).to_string(index=False))
        g["src_vint"] = g.src + ":" + g.vint
        print("\nby source x vintage:")
        print(table(g, "src_vint").round(3).to_string(index=False))
        print("\nper benchmark x vintage (mean off, n):")
        print(g.pivot_table(index="bench", columns="vint", values="off", aggfunc=["mean", "size"]).round(2).to_string())
        # attribute-level splits among attr-source pairs and all pairs
        for col in ("think", "small", "large"):
            print(f"\n{col} (all pairs):")
            print(table(g, col).round(3).to_string(index=False))
        g["effort_any"] = (g.effort > 0).astype(int)
        print("\neffort present (all pairs):")
        print(table(g, "effort_any").round(3).to_string(index=False))
        g["prov"] = [s.get("provider", "") or "(none)" for s in g.subject]
        t = table(g, "prov")
        print("\nprovider field (all pairs, n>=4):")
        print(t[t.n >= 4].round(3).to_string(index=False))
        # regression of off on release year (dated pairs), by source
        for s in ("name", "attr"):
            h = g[(g.src == s) & g.rel.notna()]
            X = np.c_[np.ones(len(h)), h.rel.values - 2025.0]
            beta, *_ = np.linalg.lstsq(X, h.off.values, rcond=None)
            e = h.off.values - X @ beta
            cov = np.linalg.inv(X.T @ X) * (e @ e) / max(len(h) - 2, 1)
            print(f"\noff ~ release year, {s}-source dated pairs n={len(h)}: intercept {beta[0]:+.3f} "
                  f"slope/yr {beta[1]:+.3f} ± {np.sqrt(cov[1, 1]):.3f}")
        g.drop(columns=["subject"]).to_csv(Path(__file__).with_name(f"calib_lobo_{'base' if scale == 1 else 'a156'}.csv"), index=False)


if __name__ == "__main__":
    main()
