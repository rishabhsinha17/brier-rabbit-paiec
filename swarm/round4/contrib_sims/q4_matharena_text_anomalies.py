"""Q4: MathArena content anomalies: short/metadata-like contents, content_hash collisions, same-text groups within one competition."""
import hashlib, re
from collections import Counter
from pathlib import Path
import pandas as pd
DATE_RE = r"^\s*Date: "
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)

def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d

it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
F = it.item_features.map(feats)
it["competition"] = F.map(lambda d: d.get("competition", ""))
it["problem_idx"] = F.map(lambda d: d.get("problem_idx", ""))
it["len"] = it.content.str.len()
r = pd.read_parquet(ROOT / "matharena" / "response.parquet")
nresp = r.groupby("item_id").size(); nsubj = r.groupby("item_id").subject_id.nunique(); mean = r.groupby("item_id").response.mean()
it["n_resp"] = it.item_id.map(nresp); it["n_subj"] = it.item_id.map(nsubj); it["mean_resp"] = it.item_id.map(mean)
# 1) content_hash collisions: same hash, different content
g = it.groupby("content_hash").content.nunique()
coll = g[g > 1]
print("content_hash values mapping to >1 distinct content:", len(coll))
for h in coll.index:
    sub = it[it.content_hash == h].drop_duplicates("content")
    for _, row in sub.iterrows():
        print(f"  hash {h} item {row.item_id} comp {row.competition} idx {row.problem_idx} len {row.len} "
              f"sha256[:16] {hashlib.sha256(row.content.encode()).hexdigest()[:16]} repr {row.content[:70]!r}...{row.content[-40:]!r}")
# also: is content_hash == sha256(content)[:16]?
calc = it.content.map(lambda c: hashlib.sha256(c.encode()).hexdigest()[:16])
print("content_hash == sha256(content)[:16]:", int((calc == it.content_hash).sum()), "of", len(it))
# 2) short or metadata-like contents
print("\ncontent length quantiles:", it.len.quantile([0, .01, .05, .5, .95, 1]).to_dict())
meta = it[it.content.str.match(r"^\s*Date: \d{4}-\d{2}-\d{2}")]
print("contents starting with 'Date: YYYY-MM-DD':", len(meta), "items;", meta.content.nunique(), "distinct texts")
print("  by competition:", meta.competition.value_counts().to_dict())
print("  lengths:", sorted(meta.len.unique())[:10], "...")
print("  responses on these items:", int(meta.n_resp.sum()), "| mean response", round(float((meta.mean_resp * meta.n_resp).sum() / meta.n_resp.sum()), 3))
print(meta[["item_id", "competition", "problem_idx", "len", "n_resp", "n_subj", "mean_resp"]].head(12).to_string())
for c in meta.content.unique()[:3]:
    print("  full content:", repr(c))
short = it[it.len < 80]
print("\nitems with content < 80 chars:", len(short), "| by competition:", short.competition.value_counts().to_dict())
# arxivmath items overall: how many look normal?
for comp in sorted(c for c in it.competition.unique() if c.startswith("arxivmath")):
    sub = it[it.competition == comp]
    nd = int(sub.content.str.match(DATE_RE).sum())
    print(f"  {comp}: items {len(sub)}, 'Date:' contents {nd}, median len {int(sub.len.median())}")
# 3) same text within one competition but different problem_idx
grp = it.groupby(["competition", "content"]).problem_idx.nunique()
w = grp[grp > 1]
print("\n(competition, text) groups with >1 problem_idx:", len(w), "| items involved:",
      int(it.set_index(["competition", "content"]).index.isin(w.index).sum()))
print("  by competition:", Counter(c for c, _ in w.index))
