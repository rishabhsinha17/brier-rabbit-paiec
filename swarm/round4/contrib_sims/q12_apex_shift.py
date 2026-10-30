"""Q12: apex_shortlist: are problem texts and reference answers misaligned for some runs? Per (problem_idx, text variant):
reference answer, subjects, responses, success rate."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 70)
def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
F = it.item_features.map(feats)
it["competition"] = F.map(lambda d: d.get("competition", "")); it["idx"] = F.map(lambda d: int(d.get("problem_idx", -1)))
ap = it[it.competition == "apex_shortlist"].copy()
ap["gold"] = ap.grading_criterion.map(lambda g: json.loads(g).get("reference_answer", "?"))
def body(t):
    t = t.split("\\boxed{}.", 1)[-1].strip()
    return t[:55].replace("\n", " ")
ap["body"] = ap.content.map(body)
# assign each distinct text an id
tid = {t: k for k, t in enumerate(sorted(ap.content.unique(), key=lambda t: ap[ap.content == t].idx.min()))}
ap["text_id"] = ap.content.map(tid)
r = pd.read_parquet(ROOT / "matharena" / "response.parquet")
s = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")
rr = r.merge(ap[["item_id"]], on="item_id")
st = rr.groupby("item_id").agg(n=("response", "size"), subj=("subject_id", "nunique"), succ=("response", "mean"))
ap = ap.merge(st, left_on="item_id", right_index=True)
# text ids that occur under 2 indices
multi_t = ap.groupby("text_id").idx.nunique()
ap["text_multi"] = ap.text_id.map(multi_t) > 1
cols = ["idx", "text_id", "text_multi", "gold", "subj", "n", "succ", "body"]
print(ap.sort_values(["idx", "text_id"])[cols][(ap.idx >= 26) & (ap.idx <= 50)].to_string(index=False))
# which subjects answered the 'second' variant? (text whose min idx < this idx: shifted)
ap["first_idx_of_text"] = ap.text_id.map(ap.groupby("text_id").idx.min())
shifted = ap[ap.text_multi & (ap.idx != ap.first_idx_of_text)]
nonsh = ap[ap.text_multi & (ap.idx == ap.first_idx_of_text)]
print("\nitems where the text also appears at the previous index (candidate shifted):", len(shifted),
      f"responses {int(shifted.n.sum())}, success {float((shifted.succ*shifted.n).sum()/shifted.n.sum()):.3f}")
print("items where the text first appears:", len(nonsh), f"responses {int(nonsh.n.sum())}, success {float((nonsh.succ*nonsh.n).sum()/nonsh.n.sum()):.3f}")
sh_subj = r[r.item_id.isin(shifted.item_id)].merge(s[["subject_id", "display_name"]]).display_name.value_counts()
ns_subj = r[r.item_id.isin(nonsh.item_id)].merge(s[["subject_id", "display_name"]]).display_name.value_counts()
print("subjects on 'shifted' items:", sh_subj.to_dict())
print("subjects on 'first' items:", ns_subj.to_dict())
