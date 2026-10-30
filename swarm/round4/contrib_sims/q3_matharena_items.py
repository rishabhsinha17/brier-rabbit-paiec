"""Q3: MathArena item IDs vs distinct texts: decompose the gap by cause (bc8204d8)."""
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
print("items columns:", list(it.columns), "rows", len(it))
for c in it.columns:
    print(f"  {c:18s} null {it[c].isna().sum():5d} distinct {it[c].nunique()}")
r = pd.read_parquet(ROOT / "matharena" / "response.parquet")
binr = r[r.response.isin([0.0, 1.0])]
frac = r[r.response.notna() & ~r.response.isin([0.0, 1.0])]
print("responses", len(r), "binary", len(binr), "fractional", len(frac), "null", r.response.isna().sum())
F = it.item_features.map(feats)
it["competition"] = F.map(lambda d: d.get("competition", ""))
it["problem_idx"] = F.map(lambda d: d.get("problem_idx", ""))
it["prompt_source"] = F.map(lambda d: d.get("prompt_source", ""))
it["image_detail"] = F.map(lambda d: d.get("image_detail", ""))
print("feature keys:", Counter(k for d in F for k in d))
print("prompt_source:", it.prompt_source.value_counts().to_dict(), "| image_detail:", it.image_detail.value_counts().to_dict())
print("verifier values (top):", it.verifier.value_counts().head(8).to_dict())
print("grading_criterion null:", it.grading_criterion.isna().sum())
# proof vs final-answer items: an item is 'proof' if any response is fractional OR verifier mentions judge
resp_items = set(r.item_id)
bin_items = set(binr.item_id)
frac_items = set(frac.item_id)
print("items with any response", len(resp_items), "| with binary responses", len(bin_items),
      "| with fractional responses", len(frac_items), "| items in table w/o responses", len(set(it.item_id) - resp_items),
      "| response item_ids missing from items table", len(resp_items - set(it.item_id)))
PROOF = {"imc_2025", "imo_2025", "miklos_2025", "putnam_2025", "usamo_2025"}
it["is_proof_comp"] = it.competition.isin(PROOF)
print("items by proof competition:", it.is_proof_comp.value_counts().to_dict())
bi = it[it.item_id.isin(bin_items)]
print("items with binary rows:", len(bi), "| of which proof-competition items:", int(bi.is_proof_comp.sum()))
rb = binr.merge(it[["item_id", "competition", "is_proof_comp"]], on="item_id")
print("binary rows from proof competitions:", int(rb.is_proof_comp.sum()), "of", len(rb))
print("  by proof competition:", rb[rb.is_proof_comp].competition.value_counts().to_dict())
print("  proof rows exactly 0 vs 1:", rb[rb.is_proof_comp].response.value_counts().to_dict())
# Distinct texts among binary items
print("\nbinary item IDs", bi.item_id.nunique(), "distinct content", bi.content.nunique(),
      "distinct content_hash", bi.content_hash.nunique())
fa = bi[~bi.is_proof_comp]
pr = bi[bi.is_proof_comp]
print("final-answer comps: item IDs", fa.item_id.nunique(), "distinct texts", fa.content.nunique())
print("proof comps (binary rows only): item IDs", pr.item_id.nunique(), "distinct texts", pr.content.nunique())
# within final-answer: text groups spanning several item IDs, classify by what differs
g = fa.groupby("content")
multi = [(t, grp) for t, grp in g if grp.item_id.nunique() > 1]
cause = Counter()
examples = {}
for t, grp in multi:
    comps = grp.competition.nunique(); ps = grp.prompt_source.nunique(); im = grp.image_detail.nunique()
    pi = grp.problem_idx.nunique(); ver = grp.verifier.nunique(); gc = grp.grading_criterion.nunique(dropna=False)
    am = grp.asset_manifest.nunique(dropna=False)
    key = []
    if comps > 1: key.append("competition")
    if pi > 1 and comps == 1: key.append("problem_idx")
    if ps > 1: key.append("prompt_source")
    if im > 1: key.append("image_detail")
    if ver > 1: key.append("verifier")
    if gc > 1: key.append("grading_criterion")
    if am > 1: key.append("asset_manifest")
    k = "+".join(key) or "other"
    cause[k] += 1
    examples.setdefault(k, []).append(grp)
print("\nfinal-answer text groups with >1 item ID:", len(multi), "| extra IDs:", sum(grp.item_id.nunique() - 1 for _, grp in multi))
for k, v in cause.most_common():
    extra_ids = sum(grp.item_id.nunique() - 1 for grp in examples[k])
    print(f"  differs by {k:40s} groups {v:4d}  extra IDs {extra_ids}")
# competition pairs that share texts
pairs = Counter()
for t, grp in multi:
    cs = tuple(sorted(grp.competition.unique()))
    if len(cs) > 1: pairs[cs] += 1
print("\ncompetition sets sharing identical texts (top 25):")
for cs, n in pairs.most_common(25):
    print(f"  {n:4d}  {cs}")
for k in examples:
    print(f"\nEXAMPLE for {k}:")
    print(examples[k][0][["item_id", "competition", "problem_idx", "prompt_source", "image_detail", "asset_manifest", "verifier"]].to_string())
    print("   text:", repr(examples[k][0].content.iloc[0][:160]))
