"""Q11: apex_shortlist same-text/different-index groups (do gold answers differ?); cross-competition duplicate problems
answered by the same subject (agreement); split Gemini subjects' competitions."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
F = it.item_features.map(feats)
it["competition"] = F.map(lambda d: d.get("competition", "")); it["problem_idx"] = F.map(lambda d: d.get("problem_idx", ""))
it["ref"] = it.verifier.map(lambda v: json.loads(v).get("reference_answer_sha256", "") if str(v).startswith("{") else "")
ap = it[it.competition == "apex_shortlist"]
grp = ap.groupby("content").agg(idx=("problem_idx", lambda s: tuple(sorted(set(s), key=int))), refs=("ref", "nunique"),
                                gcs=("grading_criterion", "nunique"), n=("item_id", "size"))
g2 = grp[grp.idx.map(len) > 1]
print("apex_shortlist texts under >1 problem_idx:", len(g2), "| with >1 distinct reference answer hash:", int((g2.refs > 1).sum()))
print(g2[["idx", "refs", "gcs", "n"]].head(20).to_string(index=False))
t = g2.index[0]
sub = ap[ap.content == t][["problem_idx", "grading_criterion", "ref"]]
print("\nexample grading_criterion values:\n", sub.drop_duplicates("problem_idx").to_string(index=False)[:1500])
print("\ntext of example (end):", repr(t[-300:]))
# same problem text in two competitions, same subject answered both
r = pd.read_parquet(ROOT / "matharena" / "response.parquet")
s = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")
SYS = it.content.str.startswith("You are ChatGPT") | it.content.str.startswith("Date: 20") | it.content.str.startswith("You are given a Math Kangaroo")
ok = it[~SYS & (it.ref != "")]
rr = r.merge(ok[["item_id", "content", "competition", "ref"]], on="item_id")
cell = rr.groupby(["subject_id", "content", "competition"]).response.mean().reset_index()
multi = cell.groupby(["subject_id", "content"]).competition.nunique()
m = multi[multi > 1]
print("\n(subject, problem text) cells answered under >=2 competitions:", len(m), "| subjects:", m.reset_index().subject_id.nunique())
pairs = cell.set_index(["subject_id", "content"]).loc[m.index].reset_index()
w = pairs.pivot_table(index=["subject_id", "content"], columns="competition", values="response")
print("   competition combos:", pairs.groupby(["subject_id", "content"]).competition.apply(lambda s: "+".join(sorted(s))).value_counts().head(6).to_dict())
# same text, same gold answer?
refchk = rr.groupby("content").ref.nunique()
print("   cross-competition duplicate texts with >1 reference answer hash:", int((refchk[refchk.index.isin(m.reset_index().content)] > 1).sum()))
sd = pairs.groupby(["subject_id", "content"]).response.agg(lambda v: v.max() - v.min())
print(f"   mean |difference| in per-competition success rate for the same subject and problem: {sd.mean():.3f}")
# split Gemini subjects
for name in ["Gemini 2.5 Flash (Thinking)", "gemini-2.5-flash (think)", "Gemini 2.5 Pro", "gemini-2.5-pro*"]:
    sid = s[s.display_name == name].subject_id.iloc[0]
    comps = sorted(it[it.item_id.isin(r[r.subject_id == sid].item_id)].competition.unique())
    print(f"   {name!r}: competitions {comps}")
