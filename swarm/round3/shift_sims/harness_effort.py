"""Section 2: are harness and reasoning-effort effects stable across (sub-)benchmarks? Public data only.

Public coverage: harness varies only inside Multi-SWE-bench (12 models x 6 main harnesses), reasoning effort only
inside MathArena (6 canonical names with 2-3 effort levels). Cross-benchmark stability is therefore untestable directly;
we use the natural sub-benchmarks (MSWE languages, MathArena competitions) as a LOWER bound on how benchmark-specific
the effects are.

2a MSWE: smoothed logit per (subject, language); additive fit logit = model + harness + language (+ harness x language)
   by least squares on the complete 12 x 6 block; report SD of harness main effects, SD of harness x language
   interaction, SD of model x harness interaction, and how often the best harness changes across languages.
2b MathArena: within canonical name, logit difference between effort levels per competition; mean effect and its
   competition-to-competition SD vs the binomial noise SD.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shiftlib as sl  # noqa: E402


def smooth_logit(k, m):
    return np.log((k + 0.5) / (m - k + 0.5)), 1.0 / (k + 0.5) + 1.0 / (m - k + 0.5)


def mswe(df, pt):
    d = df[df.bench == "multi_swebench"].copy()
    d["lang"] = d.item_features.str.split(";").str[0].str.replace("lang=", "", regex=False)
    subj = pt[pt.bench == "multi_swebench"].set_index("pair")
    d["model"] = d.pair.map(subj.key)
    d["harness"] = d.pair.map(subj.harness).str.lower().replace({"mswe-agent": "mswe-agent"})
    main_h = d.groupby("harness").pair.nunique()
    main_h = main_h[main_h >= 10].index
    d = d[d.harness.isin(main_h)]
    g = d.groupby(["model", "harness", "lang"]).label.agg(["sum", "size"]).reset_index()
    g["lg"], g["v"] = smooth_logit(g["sum"], g["size"])
    models = g.groupby("model").harness.nunique()
    full = models[models == len(main_h)].index
    g = g[g.model.isin(full)]
    print(f"2a MSWE: {len(full)} models x {len(main_h)} harnesses ({', '.join(main_h)}) x {g.lang.nunique()} languages; "
          f"{len(g)} cells, median labels/cell {int(g['size'].median())}")
    # overall (all languages pooled) harness effects
    tot = d[d.model.isin(full)].groupby(["model", "harness"]).label.agg(["sum", "size"]).reset_index()
    tot["lg"], _ = smooth_logit(tot["sum"], tot["size"])
    M = tot.pivot(index="model", columns="harness", values="lg")
    h_eff = (M - M.mean(axis=1).values[:, None]).mean()
    mh = M - M.mean(axis=1).values[:, None] - h_eff.values[None, :]
    print("   harness main effects (pooled languages, logit, centred):", h_eff.round(2).to_dict())
    print(f"   SD harness main = {h_eff.std(ddof=1):.3f}; SD model x harness interaction = "
          f"{np.sqrt((mh.values ** 2).sum() / ((M.shape[0] - 1) * (M.shape[1] - 1))):.3f}")
    best = M.idxmax(axis=1)
    print(f"   best harness per model: {best.value_counts().to_dict()} (models whose best harness = the overall best "
          f"'{h_eff.idxmax()}': {(best == h_eff.idxmax()).mean():.2f})")
    # harness x language
    rows = []
    for lang, h in g.groupby("lang"):
        Mh = h.pivot(index="model", columns="harness", values="lg")
        Vh = h.pivot(index="model", columns="harness", values="v")
        eff = (Mh - Mh.mean(axis=1).values[:, None]).mean()
        noise = np.sqrt(Vh.mean().mean() / len(Mh))  # approx SE of a harness effect in this language
        rows.append({"lang": lang, **{f"h_{c}": eff[c] for c in Mh.columns}, "best": eff.idxmax(), "se_eff": noise})
    L = pd.DataFrame(rows).set_index("lang")
    H = L[[c for c in L.columns if c.startswith("h_")]]
    inter = H - H.mean()
    sd_inter = float(np.sqrt((inter.values ** 2).sum() / ((H.shape[0] - 1) * (H.shape[1] - 1))))
    noise = float(np.sqrt((L.se_eff ** 2).mean()))
    print("   harness effect per language (centred within language):")
    print(H.round(2).to_string())
    print(f"   SD harness x language interaction = {sd_inter:.3f} (binomial-noise SD of a cell effect about {noise:.3f}; "
          f"excess SD = {np.sqrt(max(sd_inter ** 2 - noise ** 2, 0)):.3f}); SD of harness main across languages' "
          f"mean = {H.mean().std(ddof=1):.3f}")
    print(f"   best harness per language: {L.best.value_counts().to_dict()}")
    rk = H.T.rank()
    cors = [np.corrcoef(rk[a], rk[b])[0, 1] for a, b in itertools.combinations(rk.columns, 2)]
    print(f"   mean Spearman correlation of harness rankings between languages: {np.mean(cors):.2f} "
          f"(min {np.min(cors):.2f})")
    return h_eff


def matharena(df, pt):
    d = df[df.bench == "matharena"].copy()
    d["comp"] = d.item_features.str.split(";").str[0].str.replace("competition=", "", regex=False)
    subj = pt[pt.bench == "matharena"].set_index("pair")
    d["key"] = d.pair.map(subj.key)
    d["eff"] = d.pair.map(subj.eff_raw).replace({"": "default"})
    lvl = {"low": 1, "medium": 2, "default": 2.5, "high": 3, "xhigh": 4, "max": 4}
    multi = d.groupby("key").eff.nunique()
    multi = multi[multi >= 2].index
    g = d[d.key.isin(multi)].groupby(["key", "eff", "comp"]).label.agg(["sum", "size"]).reset_index()
    g["lg"], g["v"] = smooth_logit(g["sum"], g["size"])
    rows = []
    for key, h in g.groupby("key"):
        effs = sorted(h.eff.unique(), key=lambda e: lvl.get(e, 2.5))
        lo, hi = effs[0], effs[-1]
        a = h[h.eff == lo].set_index("comp")
        b = h[h.eff == hi].set_index("comp")
        common = a.index.intersection(b.index)
        for c in common:
            rows.append({"key": key, "lo": lo, "hi": hi, "comp": c, "diff": b.at[c, "lg"] - a.at[c, "lg"],
                         "var": b.at[c, "v"] + a.at[c, "v"], "n": min(a.at[c, "size"], b.at[c, "size"])})
    R = pd.DataFrame(rows)
    print(f"\n2b MathArena effort contrasts (highest minus lowest effort of the same canonical name, per competition): "
          f"{R.key.nunique()} names, {len(R)} name x competition cells")
    S = R.groupby(["key", "lo", "hi"]).agg(n_comp=("diff", "size"), mean_diff=("diff", "mean"), sd_diff=("diff", "std"),
                                           noise_sd=("var", lambda v: float(np.sqrt(np.mean(v))))).reset_index()
    print(S.round(2).to_string(index=False))
    w = 1 / R["var"]
    mean = float((w * R["diff"]).sum() / w.sum())
    # random-effects (DerSimonian-Laird) heterogeneity across competitions after removing each name's mean
    R["dev"] = R["diff"] - R.groupby("key")["diff"].transform("mean")
    Q = float((R.dev ** 2 / R["var"]).sum())
    dfree = len(R) - R.key.nunique()
    tau2 = max(0.0, (Q - dfree) / (w.sum() - (w ** 2).sum() / w.sum()))
    print(f"   precision-weighted mean effort effect {mean:+.2f} logit; competition x effort heterogeneity: Q={Q:.1f} on "
          f"{dfree} df, tau={np.sqrt(tau2):.2f} logit")
    C = R.groupby("comp")["diff"].agg(["mean", "size"]).sort_values("mean")
    print("   mean effort effect by competition (names pooled):")
    print(C.round(2).T.to_string())
    return R


def per_bench_ridge(pt):
    """Attribute ridge fit separately on each benchmark (pair-level centred logits, weight 1, alpha 1, like
    subjects.ridge_fit) to see whether coefficients agree across benchmarks."""
    from paiec.subjects import ridge_fit
    el = pt[(pt.n >= 30) & (pt.bench != "swe_rebench")].copy()
    el["c"] = el.lg - el.groupby("bench").lg.transform("mean")
    out = {}
    for b, h in el.groupby("bench"):
        X = h[sl.ATTR_NAMES].values.tolist()
        coef, rv = ridge_fit(X, h.c.tolist(), [1.0] * len(h), alpha=1.0)
        out[b] = coef
        out[b + "_n"] = len(h)
    C = pd.DataFrame({b: v for b, v in out.items() if not b.endswith("_n")}, index=sl.ATTR_NAMES)
    print("\n2c per-benchmark attribute ridge (pair-level centred logit, alpha 1; columns = benchmark):")
    present = pd.DataFrame({b: el[el.bench == b][sl.ATTR_NAMES].abs().sum() for b in C.columns})
    C2 = C.where(present > 0)
    C2["pooled_sd"] = C2.std(axis=1, ddof=1)
    print(C2.round(2).to_string())
    print("   (blank = attribute absent on that benchmark; n pairs:",
          {b: out[b + "_n"] for b in C.columns}, ")")


def main():
    pd.set_option("display.width", 250)
    df = sl.load()
    pt = sl.pair_table(df)
    mswe(df, pt)
    matharena(df, pt)
    per_bench_ridge(pt)


if __name__ == "__main__":
    main()
