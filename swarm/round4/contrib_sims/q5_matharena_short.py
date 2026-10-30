"""Q5: what are the short MathArena contents and the within-competition same-text groups; who answered the 'Date:' items."""
from collections import Counter
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 100)
def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
F = it.item_features.map(feats)
it["competition"] = F.map(lambda d: d.get("competition", "")); it["problem_idx"] = F.map(lambda d: d.get("problem_idx", ""))
it["len"] = it.content.str.len()
s = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")
r = pd.read_parquet(ROOT / "matharena" / "response.parquet").merge(s[["subject_id", "display_name"]], on="subject_id")
short = it[it.len < 80]
print("distinct short texts:", short.content.nunique())
for t, n in Counter(short.content).most_common(12):
    print(f"  x{n:3d} {t!r}")
date = it[it.content.str.match(r"^\s*Date: \d{4}-")]
rd = r[r.item_id.isin(date.item_id)]
print("\n'Date:' items answered by:", rd.display_name.value_counts().to_dict())
# does the same subject also answer the canonical version of those problems?
for _, row in date.head(5).iterrows():
    same = it[(it.competition == row.competition) & (it.problem_idx == row.problem_idx)]
    print(f"  {row.competition} idx {row.problem_idx}: item IDs for this (competition, idx): {len(same)}; "
          f"texts: {[t[:50] for t in same.content.unique()]}")
# Which subjects answered short 'canonical-looking' contents
rs = r[r.item_id.isin(short.item_id)]
print("\nshort-content responses by subject (top):", rs.display_name.value_counts().head(10).to_dict())
# apex_shortlist: same text under several problem_idx
ap = it[it.competition == "apex_shortlist"]
grp = ap.groupby("content").problem_idx.nunique()
print("\napex_shortlist: items", len(ap), "distinct texts", ap.content.nunique(), "| texts with >1 problem_idx:", int((grp > 1).sum()))
for t in grp[grp > 1].index[:3]:
    sub = ap[ap.content == t]
    print(f"  idx {sorted(sub.problem_idx.unique(), key=int)} gc distinct {sub.grading_criterion.nunique()} text {t[:120]!r}")
# items sharing (competition, problem_idx): how many IDs and why
cp = it.groupby(["competition", "problem_idx"]).item_id.nunique()
print("\n(competition, problem_idx) with >1 item ID:", int((cp > 1).sum()), "of", len(cp))
print("  by competition:", Counter(c for (c, i), n in cp.items() if n > 1))
