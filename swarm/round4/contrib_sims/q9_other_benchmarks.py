"""Q9: other public benchmarks: Multi-SWE duplicate texts, REAL trials, harness spelling, cross-benchmark metadata
consistency, response-key duplicates, trial numbering, characterization counts (bc8204d8)."""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
import re
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
BENCH = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
T = {}
for b in BENCH:
    T[b] = {t: pd.read_parquet(ROOT / b / f"{t}.parquet") for t in ("subjects", "items", "response", "benchmarks")}
# characterization counts
print("== characterization.yaml row counts vs parquet")
for b in BENCH:
    txt = open(ROOT / b / "characterization.yaml").read().split("source_claims:")[0]
    rows = {m.group(1): int(m.group(2)) for m in re.finditer(r"^  (\w+):\n    rows: (\d+)", txt, re.M)}
    got = {"benchmarks": len(T[b]["benchmarks"]), "items": len(T[b]["items"]), "responses": len(T[b]["response"]),
           "subjects": len(T[b]["subjects"])}
    mism = {k: (rows.get(k), got[k]) for k in got if rows.get(k) != got[k]}
    print(f"  {b:18s} declared {rows} | mismatches {mism or 'none'}")
# benchmarks.parquet self-reported counts
print("== benchmarks.parquet n_subjects/n_items/n_responses vs tables")
for b in BENCH:
    bm = T[b]["benchmarks"].iloc[0]
    print(f"  {b:18s} n_subjects {bm.n_subjects} ({len(T[b]['subjects'])}) n_items {bm.n_items} ({len(T[b]['items'])}) "
          f"n_responses {bm.n_responses} ({len(T[b]['response'])}) max_trial {bm.max_trial} ({T[b]['response'].trial.max()}) "
          f"coverage {bm.coverage:.4f} response_type {bm.response_type}")
# response key duplicates and trial numbering
print("== response keys")
for b in BENCH:
    r = T[b]["response"]
    key = ["subject_id", "item_id", "trial", "test_condition", "interactors"]
    dups = r.duplicated(key, keep=False).sum()
    cells = r.groupby(["subject_id", "item_id", "test_condition"], dropna=False).trial.agg(["min", "max", "count"])
    gaps = int(((cells["max"] - cells["min"] + 1) != cells["count"]).sum())
    start1 = int((cells["min"] != 1).sum())
    orphan_s = int((~r.subject_id.isin(T[b]["subjects"].subject_id)).sum()); orphan_i = int((~r.item_id.isin(T[b]["items"].item_id)).sum())
    vals = r.response.dropna()
    print(f"  {b:18s} rows {len(r):7d} dup-key rows {dups} | cells {len(cells)}, trial gaps {gaps}, not starting at 1 {start1} | "
          f"orphans subj {orphan_s} item {orphan_i} | null responses {int(r.response.isna().sum())} | "
          f"out of [0,1] {int(((vals < 0) | (vals > 1)).sum())} | binary-valued {int(vals.isin([0, 1]).sum())}")
# Multi-SWE duplicates: what differs
it = T["multi_swebench"]["items"]
g = it.groupby("content")
dup = [grp for _, grp in g if len(grp) > 1]
diff = Counter()
for grp in dup:
    cols = tuple(c for c in ["raw_item_id", "item_features", "grading_criterion", "verifier", "asset_manifest"] if grp[c].nunique(dropna=False) > 1)
    diff[cols] += 1
print("== Multi-SWE-bench texts with >1 item ID:", len(dup), "| differing columns:", dict(diff))
ex = dup[0]
print("   example raw_item_ids:", ex.raw_item_id.tolist(), "features", ex.item_features.tolist())
# do the duplicates' responses agree? (same subject answering both IDs)
r = T["multi_swebench"]["response"]
agree = tot = 0
for grp in dup:
    sub = r[r.item_id.isin(grp.item_id)]
    pv = sub.pivot_table(index="subject_id", columns="item_id", values="response", aggfunc="mean")
    pv = pv.dropna()
    if pv.shape[1] >= 2:
        tot += len(pv); agree += int((pv.nunique(axis=1) == 1).sum())
print(f"   subjects answering 2+ IDs of a duplicate text: {tot}; identical outcome on all IDs: {agree}")
# REAL: multiple trials per cell
r = T["real_webagents"]["response"]; s = T["real_webagents"]["subjects"]
c = r.groupby(["subject_id", "item_id"]).size()
print("== REAL cells with >1 trial:", int((c > 1).sum()), "of", len(c), "| subjects involved:",
      r[r.set_index(['subject_id', 'item_id']).index.isin(c[c > 1].index)].subject_id.nunique())
multi = c[c > 1].reset_index().subject_id.value_counts()
print("   per subject:", {s.set_index('subject_id').display_name[k]: int(v) for k, v in multi.items()})
# harness spelling variants
h = T["multi_swebench"]["subjects"].harness.dropna()
low = Counter(x.lower() for x in h.unique())
print("== Multi-SWE harness values:", sorted(h.unique()), "| case-insensitive collisions:", [k for k, v in low.items() if v > 1])
ms = T["multi_swebench"]["subjects"]
print("   null harness rows:", ms[ms.harness.isna()][["display_name", "normalized_name", "access_date"]].to_dict("records"))
print("   null release_date rows:", ms[ms.release_date.isna()][["display_name", "normalized_name", "harness"]].to_dict("records"))
rs = T["real_webagents"]["subjects"]
print("== REAL null release_date:", rs[rs.release_date.isna()][["display_name", "normalized_name", "provider"]].to_dict("records"))
# cross-benchmark consistency of registry fields
rows = []
for b in BENCH:
    for _, x in T[b]["subjects"].iterrows():
        if pd.notna(x.normalized_name):
            rows.append((x.normalized_name, b, x.provider, x.release_date))
R = pd.DataFrame(rows, columns=["name", "bench", "provider", "release_date"])
inc = R.groupby("name").agg(prov=("provider", lambda v: sorted(set(map(str, v)))), rel=("release_date", lambda v: sorted(set(map(str, v)))), benches=("bench", lambda v: sorted(set(v))))
bad = inc[(inc.prov.map(len) > 1) | (inc.rel.map(len) > 1)]
print("== normalized_name with inconsistent provider/release_date across rows:", len(bad))
print(bad.to_string()[:3000])
