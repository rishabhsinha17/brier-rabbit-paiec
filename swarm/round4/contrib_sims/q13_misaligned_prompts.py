"""Q13: across all MathArena final-answer competitions, find items whose prompt text is the statement of a DIFFERENT
problem index than the one it is graded as (text suffix matches another index's majority text). Report subject,
competition, index offset, responses and success vs the same statement's success under its own index."""
from collections import Counter
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
F = it.item_features.map(feats)
it["comp"] = F.map(lambda d: d.get("competition", "")); it["idx"] = F.map(lambda d: int(d.get("problem_idx", -1)))
skip = it.content.str.startswith(("You are ChatGPT", "Date: 20", "You are given a Math Kangaroo"))
it = it[~skip & (it.verifier.str.contains('"exact_matcher"'))].copy()
it["suf"] = it.content.str.strip().str[-120:]
r = pd.read_parquet(ROOT / "matharena" / "response.parquet")
s = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")[["subject_id", "display_name"]]
rr = r.merge(it[["item_id", "comp", "idx", "suf"]], on="item_id").merge(s, on="subject_id")
# majority suffix per (comp, idx) by number of subjects
maj = (rr.groupby(["comp", "idx", "suf"]).subject_id.nunique().reset_index()
         .sort_values("subject_id", ascending=False).drop_duplicates(["comp", "idx"]))
own = {(c, i): sf for c, i, sf in zip(maj.comp, maj.idx, maj.suf)}
where = {}
for (c, i), sf in own.items():
    where.setdefault((c, sf), []).append(i)
def status(row):
    if own.get((row.comp, row.idx)) == row.suf: return 0
    alt = where.get((row.comp, row.suf))
    return (alt[0] - row.idx) if alt else None
cells = rr.groupby(["display_name", "comp", "idx", "suf"]).agg(n=("response", "size"), succ=("response", "mean")).reset_index()
cells["offset"] = cells.apply(status, axis=1)
mis = cells[cells.offset.notna() & (cells.offset != 0)]
nomatch = cells[cells.offset.isna()]
print("cells (subject, comp, idx, text):", len(cells), "| aligned:", int((cells.offset == 0).sum()),
      "| text = another index's majority text (misaligned):", len(mis), "| text unmatched (template/rare variant):", len(nomatch))
if len(mis):
    print(mis.groupby(["display_name", "comp"]).agg(items=("idx", "size"), idx_min=("idx", "min"), idx_max=("idx", "max"),
          offsets=("offset", lambda v: dict(Counter(v))), responses=("n", "sum"), success=("succ", "mean")).to_string())
    # same statements under their own index: success of the other subjects
    ref = cells[cells.offset == 0].groupby(["comp", "suf"]).succ.mean()
    mis2 = mis.assign(ref=[ref.get((c, sf), float("nan")) for c, sf in zip(mis.comp, mis.suf)])
    print(f"\nmisaligned cells: mean success {mis2.succ.mean():.3f} vs other subjects on the same statements under the "
          f"correct index {mis2.ref.mean():.3f}")
# GLM 4.6 elsewhere
g = rr[rr.display_name == "GLM 4.6"]
print("\nGLM 4.6 success by competition:", g.groupby("comp").response.agg(["size", "mean"]).round(3).to_dict("index"))

# --- follow-ups: the single smt_2025 case and the unmatched cells
x = cells[(cells.display_name == "Grok 3 Mini (low)") & (cells.comp == "smt_2025") & (cells.idx.isin([43, 44, 45]))]
print("\nGrok 3 Mini (low) smt_2025 idx 43-45:\n", x[["idx", "offset", "n", "succ", "suf"]].to_string(index=False))
for i in (43, 44):
    print(f"  majority suffix idx {i}: {own[('smt_2025', i)][-90:]!r}")
gold = pd.read_parquet(ROOT / "matharena" / "items.parquet", columns=["item_id", "item_features", "grading_criterion"])
gold["comp"] = gold.item_features.map(lambda v: feats(v).get("competition")); gold["idx"] = gold.item_features.map(lambda v: int(feats(v).get("problem_idx", -1)))
print("  gold idx 43/44:", gold[(gold.comp == "smt_2025") & gold.idx.isin([43, 44])].drop_duplicates("idx")[["idx", "grading_criterion"]].values.tolist())
um = nomatch.groupby(["display_name", "comp"]).agg(items=("idx", "size"), responses=("n", "sum"), success=("succ", "mean"))
print("\nunmatched cells by subject/competition (top 15):\n", um.sort_values("items", ascending=False).head(15).to_string())
