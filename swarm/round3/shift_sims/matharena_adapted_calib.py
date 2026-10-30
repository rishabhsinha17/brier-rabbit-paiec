"""Section 1f (cont.): our LOBO subject prior (MathArena held out, as in the public screen) on MathArena's
sub-benchmarks. Per competition c: subjects' theta, realised smoothed logit l_sc (>= 10 labels), B0-optimal level
m_c = mean(l - theta), within-competition slope of l on theta. Classes: human contests vs model-era sets
(arxivmath_*, apex_*: curated to be hard for current models, i.e. difficulty adapted to frontier subjects).
Base prior and SubjCal (a156 + clamp)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shiftlib as sl  # noqa: E402
import variants as vr  # noqa: E402


def main():
    pd.set_option("display.width", 200)
    df = sl.load()
    P = sl.lobo_priors(df)
    pt = sl.pair_table(df).set_index("pair")
    fold = vr.Fold(P["matharena"], df[df.bench != "matharena"])
    d = df[df.bench == "matharena"].copy()
    d["comp"] = d.item_features.str.split(";").str[0].str.replace("competition=", "", regex=False)
    g = d.groupby(["pair", "comp"]).label.agg(["sum", "size"]).reset_index()
    g = g[g["size"] >= 10]
    g["l"] = np.log((g["sum"] + 0.5) / (g["size"] - g["sum"] + 0.5))
    g["cls"] = np.where(g.comp.str.startswith(("arxivmath", "apex")), "model-era", "human")
    for spec in ("REF", "A156"):
        th = {p: vr.theta(fold, pt.at[p, "subject"], spec)[0] for p in g.pair.unique()}
        src = {p: vr.theta(fold, pt.at[p, "subject"], spec)[2] for p in g.pair.unique()}
        g["th"] = g.pair.map(th)
        g["src"] = g.pair.map(src)
        g["res"] = g.l - g.th
        rows = []
        for (cls, comp), h in g.groupby(["cls", "comp"]):
            if len(h) < 8:
                continue
            xc = h.th - h.th.mean()
            yc = h.l - h.l.mean()
            rows.append({"cls": cls, "comp": comp, "n": len(h), "mean_th": h.th.mean(), "mean_l": h.l.mean(),
                         "m_c": h.res.mean(), "slope": float((xc * yc).sum() / (xc * xc).sum())})
        C = pd.DataFrame(rows)
        print(f"\n== {spec}: per competition (n >= 8 subjects)")
        print(C.round(2).to_string(index=False))
        for cls, h in C.groupby("cls"):
            w = h.n / h.n.sum()
            print(f"   {cls:9s}: {len(h)} competitions, subject-weighted mean theta {np.sum(w * h.mean_th):+.2f}, "
                  f"mean realised logit {np.sum(w * h.mean_l):+.2f}, B0-optimal level m_c {np.sum(w * h.m_c):+.2f}, "
                  f"mean within-competition slope {np.sum(w * h.slope):+.2f}")
        # pooled within-competition slope per class
        for cls, h in g.groupby("cls"):
            xc = h.th - h.groupby("comp").th.transform("mean")
            yc = h.l - h.groupby("comp").l.transform("mean")
            print(f"   pooled within-competition slope, {cls}: {float((xc * yc).sum() / (xc * xc).sum()):+.2f} "
                  f"(n cells {len(h)})")


if __name__ == "__main__":
    main()
