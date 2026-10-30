"""Within-benchmark association of Rasch item difficulty (all pairs in the saved predictions) with simple text features
of the public items (length, line count, digit share). Public data only; prints correlations per benchmark."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decompose import load
from irt_oracle import rasch
DATA = HERE.parents[3] / "data" / "measurement-db"
df = load(models=("w20_SCP",))
for bench, g in df.groupby("bench"):
    b0 = g[g.budget == 0].drop_duplicates(["pair", "item", "copy"])
    pairs = sorted(b0.pair.unique()); items = sorted(b0.item.unique())
    pi = b0.pair.map({p: i for i, p in enumerate(pairs)}).values
    ii = b0.item.map({t: i for i, t in enumerate(items)}).values
    a, d = rasch(pi, ii, b0.label.values.astype(float), len(pairs), len(items))
    cnt = np.bincount(ii, minlength=len(items))
    it = pd.read_parquet(DATA / bench / "items.parquet", columns=["item_id", "content", "item_features"])
    it = it.set_index("item_id").reindex(items)
    c = it.content.fillna("").astype(str)
    f = pd.DataFrame({"d": d, "n": cnt, "loglen": np.log1p(c.str.len().values),
                      "lines": np.log1p(c.str.count("\n").values),
                      "digits": c.map(lambda t: sum(ch.isdigit() for ch in t) / max(len(t), 1)).values})
    f = f[f.n >= 3]
    # within-group (item_features first token) correlation too
    grp = it.item_features.fillna("").astype(str).map(lambda s: s.split(";")[0]).reindex(items).values
    f["grp"] = pd.Series(grp, index=range(len(items))).reindex(f.index).values
    fw = f.copy()
    for col in ("d", "loglen", "lines", "digits"):
        fw[col] = f[col] - f.groupby("grp")[col].transform("mean")
    corr = {col: round(float(np.corrcoef(f.d, f[col])[0, 1]), 3) for col in ("loglen", "lines", "digits")}
    corrw = {col: round(float(np.corrcoef(fw.d, fw[col])[0, 1]), 3) for col in ("loglen", "lines", "digits")}
    print(f"{bench:18s} items {len(f):4d} SD(d) {f.d.std():.2f}  corr(d, x) {corr}  within-group {corrw}"
          f"  SD(d) within group {fw.d.std():.2f}  missing content {int(it.content.isna().sum())}")
