"""Q20: for a subject that answered the same problem text under two competition labels (e.g. aime_2025 and aime_2025_I),
are the per-trial outcomes identical (same runs released twice) or different (independent runs)?"""
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
it["comp"] = it.item_features.map(lambda v: feats(v).get("competition"))
it = it[~it.content.str.startswith(("You are ChatGPT", "Date: 20", "You are given a Math Kangaroo")) & it.verifier.str.contains("exact_matcher")]
r = pd.read_parquet(ROOT / "matharena" / "response.parquet").merge(it[["item_id", "content", "comp"]], on="item_id")
vec = r.sort_values("trial").groupby(["subject_id", "content", "comp"]).response.agg(lambda v: tuple(int(x) for x in v)).reset_index()
multi = vec.groupby(["subject_id", "content"]).comp.nunique()
keys = multi[multi > 1].index
v = vec.set_index(["subject_id", "content"]).loc[keys].reset_index()
same = v.groupby(["subject_id", "content"]).response.agg(lambda s: len(set(s)) == 1)
combo = v.groupby(["subject_id", "content"]).comp.agg(lambda s: "+".join(sorted(s)))
out = pd.DataFrame({"same": same, "combo": combo})
print("cells answered under >=2 competition labels:", len(out), "| identical per-trial outcome vectors:", int(out.same.sum()),
      f"({out.same.mean():.1%})")
print(out.groupby("combo").same.agg(["size", "mean"]).sort_values("size", ascending=False).round(3).head(8).to_string())
tot = r.merge(out.reset_index()[["subject_id", "content"]], on=["subject_id", "content"])
print("responses in these cells:", len(tot), "of", len(r), "final-answer rows")

# stronger test: only cells whose per-trial vector is mixed (not all 0 / all 1) in at least one copy
vv = v.copy()
vv["mixed"] = vv.response.map(lambda t: 0 < sum(t) < len(t))
mixed_keys = vv.groupby(["subject_id", "content"]).mixed.any()
o2 = out.join(mixed_keys)
for combo in ["aime_2026+aime_2026_I", "aime_2025+aime_2025_I", "aime_2025+aime_2025_II", "apex_2025+smt_2025"]:
    sub = o2[(o2.combo == combo) & o2.mixed]
    print(f"{combo}: cells with a mixed outcome vector {len(sub)}, identical vectors {int(sub.same.sum())}")
# trial counts per copy for AIME duplicates, and number of second-copy rows
aime = o2[o2.combo.str.startswith("aime") & o2.same]
rows = r.merge(aime.reset_index()[["subject_id", "content"]], on=["subject_id", "content"])
print("AIME cells with identical vectors:", len(aime), "| rows in both copies:", len(rows), "| second-copy rows:", len(rows) // 2,
      f"= {len(rows) // 2 / 89482:.1%} of the 89,482 MathArena final-answer rows")
print("subjects involved:", aime.reset_index().subject_id.nunique())
