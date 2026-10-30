"""Q2: MathArena subjects with empty registry fields: impact, duplicates, cross-benchmark links lost."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
from paiec.subjects import canonical_name  # heuristic used in our pipeline
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)

def extra(s, key):
    for part in str(s or "").split(";"):
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return ""

s = pd.read_parquet(ROOT / "matharena" / "subjects.parquet")
r = pd.read_parquet(ROOT / "matharena" / "response.parquet")
it = pd.read_parquet(ROOT / "matharena" / "items.parquet", columns=["item_id", "item_features"])
s["noname"] = s.normalized_name.isna()
s["model_config"] = s.subject_features_extra.map(lambda v: extra(v, "model_config"))
s["source_model_name"] = s.subject_features_extra.map(lambda v: extra(v, "source_model_name"))
print("display_name == source_model_name for all rows:", (s.display_name == s.source_model_name).all())
# responses
r = r.merge(s[["subject_id", "noname"]], on="subject_id", how="left")
print("responses total", len(r), "| rows from empty-name subjects", int(r.noname.sum()),
      f"({r.noname.mean():.1%})")
binr = r[r.response.isin([0.0, 1.0])]
print("binary rows total", len(binr), "| from empty-name subjects", int(binr.noname.sum()), f"({binr.noname.mean():.1%})")
# eligible pairs (>=80 distinct items with a binary response) by group
cnt = binr.groupby("subject_id").item_id.nunique()
s["n_items_bin"] = s.subject_id.map(cnt).fillna(0).astype(int)
s["eligible80"] = s.n_items_bin >= 80
print("eligible (>=80 binary items) subjects: named", int(s[~s.noname].eligible80.sum()), "of", int((~s.noname).sum()),
      "| empty", int(s[s.noname].eligible80.sum()), "of", int(s.noname.sum()))
# competitions per subject group
it["competition"] = it.item_features.map(lambda v: extra(v, "competition"))
rc = r.merge(it[["item_id", "competition"]], on="item_id", how="left")
comp = rc.groupby("subject_id").competition.nunique()
s["n_comp"] = s.subject_id.map(comp)
print("competitions per subject (median): named", s[~s.noname].n_comp.median(), "| empty", s[s.noname].n_comp.median())
# duplicate model_config
dup = s[s.duplicated("model_config", keep=False)].sort_values("model_config")
print("\nsubjects sharing a model_config:", len(dup))
print(dup[["display_name", "normalized_name", "model_config", "reasoning_effort", "n_items_bin", "n_comp"]].to_string())
# do the duplicate-config subjects ever answer the same competitions?
for cfg, g in dup.groupby("model_config"):
    comps = [set(rc[rc.subject_id == sid].competition.unique()) for sid in g.subject_id]
    inter = set.intersection(*comps)
    print(f"  {cfg}: competitions overlap {sorted(inter)}; union size {len(set.union(*comps))}")
# cross-benchmark links: canonical keys of empty-name subjects vs names present in other benchmarks
others = []
for b in ["mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]:
    o = pd.read_parquet(ROOT / b / "subjects.parquet")
    others += [(b, n) for n in o.normalized_name.dropna().unique()]
okeys = {}
for b, n in others:
    okeys.setdefault(canonical_name(n), set()).add(b)
named_keys = {canonical_name(n) for n in s.normalized_name.dropna()}
s["key"] = s.source_model_name.map(canonical_name)
e = s[s.noname].copy()
e["in_other_bench"] = e.key.map(lambda k: ",".join(sorted(okeys.get(k, []))))
e["key_in_named_matharena"] = e.key.isin(named_keys)
print("\nempty-name subjects whose heuristic key matches a normalized_name in another benchmark:",
      int((e.in_other_bench != "").sum()), "| matches a named MathArena subject:", int(e.key_in_named_matharena.sum()))
print(e[["display_name", "key", "in_other_bench", "key_in_named_matharena", "n_items_bin"]].to_string())
# provider prefixes of model_config for empty rows
print("\nmodel_config provider prefix counts among empty:", e.model_config.str.split("/").str[0].value_counts().to_dict())
