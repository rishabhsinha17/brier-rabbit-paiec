"""Within-benchmark item-level signal available at prediction time (public data only).
Item easiness e_j = mean over responses of (y - pair rate). Reports Spearman of e_j with numeric item_features tokens
(within first-token group) and with log content length, plus the share of item easiness variance explained by groups."""
import sys, re
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from paiec.data import load_responses

df = load_responses()
df["pr"] = df.groupby("pair")["label"].transform("mean")
df["res"] = df["label"] - df["pr"]
for b, g in df.groupby("bench"):
    if g.pair.nunique() < 5:
        continue
    it = g.groupby("item_id").agg(e=("res", "mean"), n=("res", "size"), feats=("item_features", "first"),
                                  content=("item_content", "first"), nsub=("pair", "nunique"))
    it = it[it.nsub >= 3]
    it["grp"] = it.feats.map(lambda f: f.split(";")[0] if f else "")
    it["loglen"] = np.log(it.content.str.len() + 1)
    tot = it.e.var()
    within = it.groupby("grp").e.transform(lambda x: x - x.mean())
    print(f"\n{b}: items {len(it)} (>=3 subjects), var(e)={tot:.4f}, share explained by first-token group: {1 - within.var() / tot:.3f}")
    r, pv = spearmanr(it.loglen, it.e)
    rw, _ = spearmanr(it.loglen, within)
    print(f"  log length vs easiness: Spearman {r:+.3f} (p={pv:.1e}); within group {rw:+.3f}")
    # numeric tokens
    keys = {}
    for f in it.feats:
        for t in f.split(";")[1:]:
            if "=" in t:
                k, v = t.split("=", 1)
                if re.fullmatch(r"-?\d+(\.\d+)?", v.strip()):
                    keys.setdefault(k, 0)
                    keys[k] += 1
    for k in keys:
        x = it.feats.map(lambda f: next((float(t.split("=", 1)[1]) for t in f.split(";") if t.startswith(k + "=") and re.fullmatch(r"-?\d+(\.\d+)?", t.split("=", 1)[1].strip())), np.nan))
        ok = x.notna()
        r, pv = spearmanr(x[ok], it.e[ok])
        rw, pvw = spearmanr(x[ok], within[ok])
        print(f"  numeric key {k}: n={ok.sum()} Spearman {r:+.3f} (p={pv:.1e}); within group {rw:+.3f} (p={pvw:.1e})")
        # within-group Spearman averaged over groups with >=5 items
        rs = []
        for gname, gg in it[ok].groupby("grp"):
            if len(gg) >= 5 and gg.e.std() > 0:
                rs.append(spearmanr(x[gg.index], gg.e)[0])
        print(f"    per-group Spearman over {len(rs)} groups: mean {np.nanmean(rs):+.3f}, median {np.nanmedian(rs):+.3f}, share negative {np.mean(np.array(rs) < 0):.2f}")
