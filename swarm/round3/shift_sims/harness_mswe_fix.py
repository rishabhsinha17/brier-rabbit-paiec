"""2a (corrected): in Multi-SWE-bench, harnesses are nested in language families: Python is run with agentless /
openhands / swe-agent, the 7 other languages with the multi-language ports magentless / mopenhands / mswe-agent.
So harness x language is estimable only inside the M-family over the 7 non-Python languages. Model effects are
removed within each language (12 models in common)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shiftlib as sl  # noqa: E402
from harness_effort import smooth_logit  # noqa: E402


def main():
    pd.set_option("display.width", 220)
    df = sl.load()
    pt = sl.pair_table(df)
    d = df[df.bench == "multi_swebench"].copy()
    d["lang"] = d.item_features.str.split(";").str[0].str.replace("lang=", "", regex=False)
    subj = pt[pt.bench == "multi_swebench"].set_index("pair")
    d["model"] = d.pair.map(subj.key)
    d["harness"] = d.pair.map(subj.harness).str.lower()
    fam = ["magentless", "mopenhands", "mswe-agent"]
    x = d[d.harness.isin(fam) & (d.lang != "python")]
    g = x.groupby(["model", "harness", "lang"]).label.agg(["sum", "size"]).reset_index()
    g["lg"], g["v"] = smooth_logit(g["sum"], g["size"])
    ok = g.groupby("model").apply(lambda h: h.harness.nunique() == 3 and h.lang.nunique() == 7)
    g = g[g.model.isin(ok[ok].index)]
    print(f"M-family: {g.model.nunique()} models x 3 harnesses x 7 languages, {len(g)} cells, median labels/cell "
          f"{int(g['size'].median())}")
    # two-way within language: effect[h, lang] = mean over models of (lg - model mean in that language)
    g["dev"] = g.lg - g.groupby(["model", "lang"]).lg.transform("mean")
    E = g.groupby(["lang", "harness"]).dev.mean().unstack()
    Vn = g.groupby(["lang", "harness"]).v.mean().unstack() / g.model.nunique()
    main = E.mean()
    inter = E - main
    sd_inter = float(np.sqrt((inter.values ** 2).sum() / ((E.shape[0] - 1) * (E.shape[1] - 1))))
    noise = float(np.sqrt(Vn.values.mean() * (2 / 3)))  # a centred cell effect carries ~2/3 of the cell noise
    print("harness effect per language (model-adjusted, logit):")
    print(E.round(2).to_string())
    print(f"main effects (mean over languages): {main.round(2).to_dict()}; SD of main effects {main.std(ddof=1):.2f}")
    print(f"SD of harness x language interaction {sd_inter:.2f}; binomial noise SD of a cell effect about {noise:.2f}; "
          f"excess (true) interaction SD about {np.sqrt(max(sd_inter ** 2 - noise ** 2, 0)):.2f}")
    print(f"best harness per language: {E.idxmax(axis=1).value_counts().to_dict()}")
    rk = E.T.rank()
    cors = [np.corrcoef(rk[a], rk[b])[0, 1] for a, b in itertools.combinations(rk.columns, 2)]
    print(f"mean Spearman of harness rankings between language pairs {np.mean(cors):.2f} (share negative "
          f"{np.mean(np.array(cors) < 0):.2f})")
    # model x harness interaction (all languages pooled within the family)
    t = x[x.model.isin(g.model.unique())].groupby(["model", "harness"]).label.agg(["sum", "size"]).reset_index()
    t["lg"], t["v"] = smooth_logit(t["sum"], t["size"])
    M = t.pivot(index="model", columns="harness", values="lg")
    res = M - M.mean(axis=1).values[:, None] - (M - M.mean(axis=1).values[:, None]).mean().values[None, :]
    sd_mh = float(np.sqrt((res.values ** 2).sum() / ((M.shape[0] - 1) * (M.shape[1] - 1))))
    nz = float(np.sqrt(t.v.mean() * (2 / 3)))
    print(f"model x harness interaction SD {sd_mh:.2f} (noise about {nz:.2f}); best harness per model "
          f"{M.idxmax(axis=1).value_counts().to_dict()}")
    # Python family for completeness
    py = d[(d.lang == "python")].groupby(["model", "harness"]).label.agg(["sum", "size"]).reset_index()
    py["lg"], _ = smooth_logit(py["sum"], py["size"])
    P = py.pivot(index="model", columns="harness", values="lg").dropna()
    print(f"Python family ({len(P)} models): harness effects {(P - P.mean(axis=1).values[:, None]).mean().round(2).to_dict()}, "
          f"best per model {P.idxmax(axis=1).value_counts().to_dict()}")
    # how big is harness spread relative to the model spread? (same subjects)
    print(f"spread of model effects (SD over models, M-family pooled) {M.mean(axis=1).std(ddof=1):.2f} vs harness main SD "
          f"{(M - M.mean(axis=1).values[:, None]).mean().std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
